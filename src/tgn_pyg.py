"""Reference TGN (Rossi et al., 2020) built from PyTorch Geometric's official components:
``TGNMemory`` + ``IdentityMessage`` + ``LastAggregator`` + ``LastNeighborLoader`` and the
``TransformerConv`` graph-attention embedding of PyG's examples/tgn.py.

Adaptation to transaction streams: every transaction emits three interaction events
customer->device, customer->card, customer->IP whose raw message is the transaction's features
plus a relation one-hot. Transactions are processed in time-ordered batches; each batch is SCORED
with memory/neighbourhood state from earlier batches only, THEN its events update memory and the
neighbour loader (exactly the order used in PyG's example) -- so no future information is used.
The transaction is classified from the TGN embeddings of its customer, device, card and IP plus
its own features. Labels never enter memory.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import average_precision_score

from src.temporal_gnn import RELATIONS, TemporalIndex, _Standardiser, _target_matrix

torch.set_num_threads(1)


def available() -> bool:
    try:
        import torch_geometric  # noqa: F401
        return True
    except Exception:
        return False


class _GraphAttentionEmbedding(nn.Module):
    """As in PyG examples/tgn.py."""

    def __init__(self, in_channels, out_channels, msg_dim, time_enc):
        super().__init__()
        from torch_geometric.nn import TransformerConv
        self.time_enc = time_enc
        edge_dim = msg_dim + time_enc.out_channels
        self.conv = TransformerConv(in_channels, out_channels // 2, heads=2, dropout=0.1, edge_dim=edge_dim)

    def forward(self, x, last_update, edge_index, t, msg):
        rel_t = last_update[edge_index[0]] - t
        rel_t_enc = self.time_enc(rel_t.to(x.dtype))
        edge_attr = torch.cat([rel_t_enc, msg], dim=-1)
        return self.conv(x, edge_index, edge_attr)


class TGNReference:
    def __init__(self, index: TemporalIndex, mem_dim: int = 64, time_dim: int = 32, emb_dim: int = 64,
                 lr: float = 1e-3, epochs: int = 3, batch: int = 200, neighbours: int = 10, seed: int = 42,
                 val_df: Optional[pd.DataFrame] = None):
        self.index, self.mem_dim, self.time_dim, self.emb_dim = index, mem_dim, time_dim, emb_dim
        self.lr, self.epochs, self.batch, self.k, self.seed, self.val_df = lr, epochs, batch, neighbours, seed, val_df
        self.name = "TGN (PyG reference)"
        offsets = np.concatenate([[0], np.cumsum(index.codes.max(0) + 1)[:-1]])
        self.nodes = (index.codes + offsets).astype(np.int64)  # [N, 4] global ids: customer, device, card, ip
        self.n_nodes = int(self.nodes.max()) + 1
        self.t = torch.from_numpy(((index.ts_h - index.ts_h.min()) * 3600).astype(np.int64))  # seconds

    def _build(self, d_feat: int):
        from torch_geometric.nn import TGNMemory
        from torch_geometric.nn.models.tgn import IdentityMessage, LastAggregator
        torch.manual_seed(self.seed)
        self.msg_dim = d_feat + 3
        self.memory = TGNMemory(self.n_nodes, self.msg_dim, self.mem_dim, self.time_dim,
                                message_module=IdentityMessage(self.msg_dim, self.mem_dim, self.time_dim),
                                aggregator_module=LastAggregator())
        self.gnn = _GraphAttentionEmbedding(self.mem_dim, self.emb_dim, self.msg_dim, self.memory.time_enc)
        self.head = nn.Sequential(nn.Linear(4 * self.emb_dim + d_feat, 64), nn.ReLU(), nn.Linear(64, 1))
        seen, self.params = set(), []  # time encoder is shared by memory and attention: deduplicate
        for p_ in list(self.memory.parameters()) + list(self.gnn.parameters()) + list(self.head.parameters()):
            if id(p_) not in seen:
                seen.add(id(p_))
                self.params.append(p_)

    def _stream(self, upto: int, train_mask=None, y=None, opt=None) -> np.ndarray:
        from torch_geometric.nn.models.tgn import LastNeighborLoader
        training = opt is not None
        # Switch mode BEFORE resetting: TGNMemory.train(False) applies a GRU update to every node,
        # so resetting afterwards guarantees every stream starts from the same all-zero memory.
        self.memory.train(training)
        self.gnn.train(training)
        self.memory.reset_state()
        loader = LastNeighborLoader(self.n_nodes, size=self.k)
        assoc = torch.empty(self.n_nodes, dtype=torch.long)
        X = torch.from_numpy(self.X)
        rel_eye = torch.eye(3)
        preds = np.zeros(upto, dtype=np.float32)
        pos_w = torch.tensor(5.0)

        for s in range(0, upto, self.batch):
            e = min(upto, s + self.batch)
            train_step = bool(training and train_mask[s:e].any())
            nodes = torch.from_numpy(self.nodes[s:e])  # [b,4]
            with torch.set_grad_enabled(train_step):
                n_id = nodes.reshape(-1).unique()
                n_id, edge_index, e_id = loader(n_id)
                assoc[n_id] = torch.arange(n_id.size(0))
                z, last_update = self.memory(n_id)
                if edge_index.numel():
                    z = self.gnn(z, last_update, edge_index, self.ev_t[e_id], self.ev_msg[e_id])
                zz = z[assoc[nodes]].reshape(e - s, -1)
                logit = self.head(torch.cat([zz, X[s:e]], 1)).squeeze(-1)
                if train_step:
                    m = torch.from_numpy(train_mask[s:e])
                    loss = F.binary_cross_entropy_with_logits(logit[m], torch.from_numpy(y[s:e][train_mask[s:e]]),
                                                              pos_weight=pos_w)
                    opt.zero_grad()
                    loss.backward()
                    opt.step()
            preds[s:e] = torch.sigmoid(logit.detach()).numpy()
            # update memory + neighbourhood with this batch's events (after scoring it)
            b = e - s
            src = nodes[:, 0].repeat_interleave(3)
            dst = nodes[:, 1:].reshape(-1)
            ids = torch.arange(s, e).repeat_interleave(3)
            rel = torch.arange(3).repeat(b)
            msg = torch.cat([X[ids], rel_eye[rel]], 1)
            t = self.t[ids]
            self.memory.update_state(src, dst, t, msg)
            loader.insert(src, dst)
            # store events for later attention lookups (event ids are global, append-only)
            start = self.n_ev
            self.ev_t[start:start + len(t)] = t
            self.ev_msg[start:start + len(t)] = msg
            self.n_ev += len(t)
            self.memory.detach()
        return preds

    def _reset_events(self, upto: int):
        self.ev_t = torch.zeros(3 * upto, dtype=torch.long)
        self.ev_msg = torch.zeros(3 * upto, self.msg_dim)
        self.n_ev = 0

    def fit(self, train_df: pd.DataFrame, y: np.ndarray, full_df: Optional[pd.DataFrame] = None) -> "TGNReference":
        assert full_df is not None
        raw = _target_matrix(full_df)
        self.X = _Standardiser().fit(raw[train_df["_row"].to_numpy()])(raw)
        self._build(self.X.shape[1])
        opt = torch.optim.Adam(self.params, lr=self.lr)
        upto = int(train_df["_row"].max()) + 1
        mask = np.zeros(upto, dtype=bool)
        mask[train_df["_row"].to_numpy()] = True
        yy = np.zeros(upto, dtype=np.float32)
        yy[train_df["_row"].to_numpy()] = np.asarray(y, dtype=np.float32)
        best, best_state = -1.0, None
        for _ in range(self.epochs):
            self._reset_events(upto)
            self._stream(upto, mask, yy, opt)
            if self.val_df is not None:
                vu = int(self.val_df["_row"].max()) + 1
                self._reset_events(vu)
                with torch.no_grad():
                    p = self._stream(vu)
                ap = average_precision_score(self.val_df["fraud_label"], p[self.val_df["_row"].to_numpy()])
                if ap > best:
                    best = ap
                    best_state = [x.detach().clone() for x in self.params]
        if best_state is not None:
            with torch.no_grad():
                for p_, b in zip(self.params, best_state):
                    p_.copy_(b)
        self._cache = None
        return self

    def predict_proba(self, df: pd.DataFrame, history=None) -> np.ndarray:
        upto = int(df["_row"].max()) + 1
        if self._cache is None or len(self._cache) < upto:
            self._reset_events(upto)
            with torch.no_grad():
                self._cache = self._stream(upto)
        return self._cache[df["_row"].to_numpy()]
