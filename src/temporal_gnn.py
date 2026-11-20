"""Stronger relational baselines over a strictly-past temporal neighbourhood.

All models below see, for each transaction i, the K most recent EARLIER transactions that
share its customer / device / card / IP (a transaction-entity-transaction 2-hop temporal
neighbourhood, as in TGAT-style samplers). Because neighbours are strictly earlier, the
same index is leakage-free for every split; no labels are used unless a label delay is
given (C2), in which case a neighbour's label is visible only once it has matured.

Models
------
* ``TemporalGNN(kind="sage")``  mean aggregation, weights shared across relations (GraphSAGE)
* ``TemporalGNN(kind="gat")``   target-conditioned attention over neighbours (GAT / TGAT-style)
* ``TemporalGNN(kind="rgcn")``  relation-specific weights, mean aggregation (R-GCN-style)
* ``TemporalGNN(kind="care")``  CARE-GNN-style label-aware neighbour filtering (top-rho most
                               label-similar neighbours per relation; fixed rho, no RL)
* ``TGNMemoryModel``            TGN-style per-entity GRU memory trained through time
* ``NeighborAggXGB``            GADBench-style gradient-boosted trees on aggregated
                               neighbour features
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import average_precision_score

from src.classifier_utils import get_classifier
from src.preprocessing import PAYMENT_METHODS, TRANSACTION_FEATURE_COLS, prepare_feature_matrix

torch.set_num_threads(1)  # see graph_model.py: avoids an OpenMP deadlock with XGBoost on macOS

RELATIONS = ["customer_id", "device_id", "card_id", "ip_id"]
N_NEIGH_FEATS = 13  # see TemporalIndex.neighbour_features


class TemporalIndex:
    """Strictly-past K-recent neighbour index over a chronologically sorted dataframe.

    The dataframe must be the FULL chronological dataset (all splits); row r of it is
    referenced by the ``_row`` column that models use to locate transactions.
    """

    def __init__(self, tx: pd.DataFrame, k: int = 20, label_delay_days: Optional[float] = None):
        assert (np.diff(tx["timestamp"].to_numpy().astype("datetime64[ns]").astype(np.int64)) >= 0).all(), \
            "tx must be sorted chronologically"
        self.k = k
        n = len(tx)
        self.n = n
        self.codes = np.stack([pd.factorize(tx[c].astype(str))[0] for c in RELATIONS], axis=1).astype(np.int64)
        self.nbr = np.full((n, len(RELATIONS), k), -1, dtype=np.int32)
        for r in range(len(RELATIONS)):
            code = self.codes[:, r]
            order = np.argsort(code, kind="stable")  # within a code: chronological row order
            sc = code[order]
            for j in range(1, k + 1):
                prev = np.arange(n) - j
                ok = prev >= 0
                ok[ok] = sc[prev[ok]] == sc[ok]
                vals = np.where(ok, order[np.clip(prev, 0, None)], -1)
                self.nbr[order, r, j - 1] = vals
        self.ts_h = tx["timestamp"].to_numpy().astype("datetime64[ns]").astype(np.int64) / 3.6e12
        amount = tx["amount"].to_numpy(dtype=np.float64)
        hour = tx["timestamp"].dt.hour.to_numpy() + tx["timestamp"].dt.minute.to_numpy() / 60
        pm = tx["payment_method"].astype(str).to_numpy()
        # per-row intrinsic features (known at the time the row is a neighbour)
        self.base = np.column_stack(
            [np.log1p(amount), tx["success"].to_numpy(dtype=np.float64),
             np.sin(2 * np.pi * hour / 24), np.cos(2 * np.pi * hour / 24)]
            + [(pm == m).astype(np.float64) for m in PAYMENT_METHODS]
        ).astype(np.float32)
        self.y = tx["fraud_label"].to_numpy(dtype=np.float32)
        self.label_delay_h = None if label_delay_days is None or not np.isfinite(label_delay_days) else label_delay_days * 24

    @property
    def n_neigh_feats(self) -> int:
        return N_NEIGH_FEATS + (2 if self.label_delay_h is not None else 0)

    def neighbour_features(self, rows: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Features [B, R, K, F] and validity mask [B, R, K] for target rows."""
        nb = self.nbr[rows]  # [B,R,K]
        valid = nb >= 0
        j = np.where(valid, nb, 0)
        base = self.base[j]  # [B,R,K,8]
        same = (self.codes[j] == self.codes[rows][:, None, None, :]).astype(np.float32)  # [B,R,K,4]
        dt = np.log1p(np.clip(self.ts_h[rows][:, None, None] - self.ts_h[j], 0, None)).astype(np.float32)[..., None]
        parts = [base, same, dt]
        if self.label_delay_h is not None:
            known = (self.ts_h[j] + self.label_delay_h <= self.ts_h[rows][:, None, None]).astype(np.float32)
            parts += [known[..., None], (known * self.y[j])[..., None]]
        feats = np.concatenate(parts, axis=-1) * valid[..., None]
        return feats.astype(np.float32), valid


class _Standardiser:
    def fit(self, X: np.ndarray) -> "_Standardiser":
        self.mu, self.sd = X.mean(0), X.std(0) + 1e-6
        return self

    def __call__(self, X: np.ndarray) -> np.ndarray:
        return ((X - self.mu) / self.sd).astype(np.float32)


def _target_matrix(df: pd.DataFrame) -> np.ndarray:
    X = prepare_feature_matrix(df, TRANSACTION_FEATURE_COLS)
    X[:, : len(TRANSACTION_FEATURE_COLS) - 1] = np.sign(X[:, : len(TRANSACTION_FEATURE_COLS) - 1]) * np.log1p(
        np.abs(X[:, : len(TRANSACTION_FEATURE_COLS) - 1]))
    return X


class _Net(nn.Module):
    def __init__(self, kind: str, d_target: int, d_neigh: int, n_rel: int, hidden: int, rho: float):
        super().__init__()
        self.kind, self.n_rel, self.rho = kind, n_rel, rho
        self.target = nn.Sequential(nn.Linear(d_target, hidden), nn.ReLU(), nn.Linear(hidden, hidden))
        n_w = n_rel if kind in ("rgcn", "care") else 1
        self.neigh = nn.ModuleList([nn.Sequential(nn.Linear(d_neigh, hidden), nn.ReLU(), nn.Linear(hidden, hidden))
                                    for _ in range(n_w)])
        if kind == "gat":
            self.q = nn.Linear(hidden, hidden)
            self.kproj = nn.Linear(hidden, hidden)
        if kind == "care":
            self.aux_t = nn.Linear(d_target, 1)  # label predictor on the target
            self.aux_n = nn.Linear(d_neigh, 1)   # label predictor on neighbours (same label space)
        self.head = nn.Sequential(nn.Linear(hidden * (1 + n_rel) + n_rel, hidden), nn.ReLU(), nn.Linear(hidden, 1))

    def forward(self, xt, xn, valid):
        B, R, K, _ = xn.shape
        ht = self.target(xt)  # [B,H]
        aggs, aux = [], None
        if self.kind == "care":
            pt = torch.sigmoid(self.aux_t(xt))  # [B,1]
            pn = torch.sigmoid(self.aux_n(xn)).squeeze(-1)  # [B,R,K]
            sim = -(pn - pt[:, :, None]).abs()
            sim = sim.masked_fill(~valid, -1e9)
            n_keep = torch.clamp((valid.sum(-1).float() * self.rho).ceil(), min=1).long()  # [B,R]
            rank = sim.argsort(-1, descending=True).argsort(-1)
            valid = valid & (rank < n_keep[..., None])
            aux = self.aux_t(xt).squeeze(-1)
        for r in range(R):
            net = self.neigh[r if len(self.neigh) > 1 else 0]
            hn = net(xn[:, r])  # [B,K,H]
            m = valid[:, r].float()  # [B,K]
            if self.kind == "gat":
                att = (self.q(ht)[:, None, :] * self.kproj(hn)).sum(-1) / hn.shape[-1] ** 0.5
                att = att.masked_fill(m == 0, -1e9)
                w = torch.softmax(att, -1) * (m.sum(-1, keepdim=True) > 0)
            else:
                w = m / m.sum(-1, keepdim=True).clamp(min=1)
            aggs.append((w[..., None] * hn).sum(1))
        counts = torch.log1p(valid.sum(-1).float())  # [B,R]
        logit = self.head(torch.cat([ht] + aggs + [counts], -1)).squeeze(-1)
        return logit, aux


class TemporalGNN:
    """GNN over the strictly-past temporal neighbourhood (see module docstring)."""

    def __init__(self, index: TemporalIndex, kind: str = "sage", hidden: int = 64, lr: float = 1e-3,
                 epochs: int = 12, batch: int = 512, rho: float = 0.5, seed: int = 42, val_df: Optional[pd.DataFrame] = None,
                 target_fn=None, focal_gamma: float = 0.0):
        """target_fn(df) -> feature matrix for the target transaction (default: our transaction
        features). Used on IEEE-CIS to give GNNs the same rich features as the tabular baseline."""
        self.target_fn = target_fn or _target_matrix
        self.index, self.kind, self.hidden, self.lr = index, kind, hidden, lr
        self.epochs, self.batch, self.rho, self.seed, self.val_df = epochs, batch, rho, seed, val_df
        self.focal_gamma = focal_gamma  # > 0: focal loss baseline (Lin et al., 2017)
        self.name = {"sage": "Temporal GraphSAGE", "gat": "Temporal GAT", "rgcn": "Temporal R-GCN",
                     "care": "CARE-style GNN"}[kind]

    def _tensors(self, df: pd.DataFrame):
        rows = df["_row"].to_numpy()
        xn, valid = self.index.neighbour_features(rows)
        xn = self.std_n(xn) * valid[..., None]
        return (torch.from_numpy(self.std_t(self.target_fn(df))), torch.from_numpy(xn), torch.from_numpy(valid))

    def fit(self, train_df: pd.DataFrame, y: np.ndarray, sample_weight: Optional[np.ndarray] = None,
            ring_info: Optional[dict] = None) -> "TemporalGNN":
        """ring_info: {"index": RingIndex over train_df positions, "lam": weight, "batch_rows": int} adds the
        ring-survival loss (src/ring_survival.py) on whole (pseudo-)rings after each epoch's BCE pass."""
        torch.manual_seed(self.seed)
        rng = np.random.default_rng(self.seed)
        Xt = self.target_fn(train_df)
        self.std_t = _Standardiser().fit(Xt)
        # Fit the neighbour standardiser on a RANDOM sample spread over the whole training period.
        # (Bug fix: the first 20,000 rows covered only the first minutes of AMLworld, so hour/time-gap
        # features got near-zero s.d. and test inputs reached |z| ~ 1e4, destabilising training.)
        tr_rows = train_df["_row"].to_numpy()
        pick = np.sort(rng.choice(len(tr_rows), size=min(20000, len(tr_rows)), replace=False))
        xn_sample, v_sample = self.index.neighbour_features(tr_rows[pick])
        flat = xn_sample[v_sample]
        self.std_n = _Standardiser().fit(flat if len(flat) else xn_sample.reshape(-1, xn_sample.shape[-1]))
        xt, xn, valid = self._tensors(train_df)
        yt = torch.from_numpy(np.asarray(y, dtype=np.float32))
        wt = None if sample_weight is None else torch.from_numpy(np.asarray(sample_weight, dtype=np.float32))
        self.net = _Net(self.kind, xt.shape[1], xn.shape[-1], xn.shape[1], self.hidden, self.rho)
        opt = torch.optim.Adam(self.net.parameters(), lr=self.lr, weight_decay=1e-5)
        pos_w = torch.tensor(min(20.0, float((1 - yt.mean()) / max(yt.mean(), 1e-6)) ** 0.5))
        best, best_state = -1.0, None
        if self.val_df is not None:
            vt = self._tensors(self.val_df)
            yv = self.val_df["fraud_label"].to_numpy()
        for _ in range(self.epochs):
            self.net.train()
            perm = rng.permutation(len(yt))
            for s in range(0, len(perm), self.batch):
                b = torch.from_numpy(perm[s:s + self.batch])
                logit, aux = self.net(xt[b], xn[b], valid[b])
                wb = None if wt is None else wt[b]  # TVW per-example weights (src/clock_weights.py)
                if self.focal_gamma > 0:
                    ce = F.binary_cross_entropy_with_logits(logit, yt[b], pos_weight=pos_w, reduction="none")
                    pr = torch.sigmoid(logit)
                    pt = pr * yt[b] + (1 - pr) * (1 - yt[b])
                    fl = (1 - pt).pow(self.focal_gamma) * ce
                    loss = (fl * wb).mean() if wb is not None else fl.mean()
                else:
                    loss = F.binary_cross_entropy_with_logits(logit, yt[b], weight=wb, pos_weight=pos_w)
                if aux is not None:
                    loss = loss + 0.5 * F.binary_cross_entropy_with_logits(aux, yt[b], weight=wb, pos_weight=pos_w)
                opt.zero_grad()
                loss.backward()
                opt.step()
            if ring_info is not None:  # ring-survival term: optimise the exact takedown value per ring
                from src.ring_survival import ring_survival_loss
                rix, lam = ring_info["index"], ring_info["lam"]
                for segs in rix.batches(rng, ring_info.get("batch_rows", self.batch)):
                    pos, seg = rix.select(segs, rng if ring_info.get("prefix") else None)
                    rows = torch.from_numpy(rix.rows[pos])
                    logit, _ = self.net(xt[rows], xn[rows], valid[rows])
                    loss = lam * ring_survival_loss(logit, torch.from_numpy(seg),
                                                    torch.from_numpy(rix.remaining_share[pos]).float(),
                                                    torch.from_numpy(rix.ring_weight[segs]).float())
                    opt.zero_grad()
                    loss.backward()
                    opt.step()
            if self.val_df is not None:  # early stopping on validation PR-AUC
                ap = average_precision_score(yv, self._predict(*vt))
                if ap > best:
                    best, best_state = ap, {k: v.clone() for k, v in self.net.state_dict().items()}
        if best_state is not None:
            self.net.load_state_dict(best_state)
        return self

    def _predict(self, xt, xn, valid) -> np.ndarray:
        self.net.eval()
        out = []
        with torch.no_grad():
            for s in range(0, len(xt), 4096):
                out.append(torch.sigmoid(self.net(xt[s:s + 4096], xn[s:s + 4096], valid[s:s + 4096])[0]).numpy())
        return np.concatenate(out)

    def predict_proba(self, df: pd.DataFrame, history: Optional[pd.DataFrame] = None) -> np.ndarray:
        return self._predict(*self._tensors(df))


class NeighborAggXGB:
    """GADBench-style baseline: transaction features + aggregated temporal-neighbour features."""

    def __init__(self, index: TemporalIndex, seed: int = 42, classifier=None, target_fn=None):
        self.index, self.name = index, "Neighbour-Aggregation XGBoost"
        self.target_fn = target_fn
        self.model = classifier if classifier is not None else get_classifier(seed)

    def _X(self, df: pd.DataFrame) -> np.ndarray:
        xn, valid = self.index.neighbour_features(df["_row"].to_numpy())
        v = valid[..., None].astype(np.float32)
        cnt = valid.sum(-1, keepdims=True).astype(np.float32)  # [B,R,1]
        mean = (xn * v).sum(2) / np.clip(cnt, 1, None)
        mx = np.where(valid[..., None], xn, -1e3).max(2)
        mx[cnt[..., 0] == 0] = 0
        agg = np.concatenate([mean, mx, cnt], -1).reshape(len(df), -1)
        base = self.target_fn(df) if self.target_fn else prepare_feature_matrix(df, TRANSACTION_FEATURE_COLS)
        return np.hstack([base, agg])

    def fit(self, train_df: pd.DataFrame, y: np.ndarray, sample_weight: Optional[np.ndarray] = None) -> "NeighborAggXGB":
        self.model.fit(self._X(train_df), y, sample_weight=sample_weight)
        return self

    def predict_proba(self, df: pd.DataFrame, history: Optional[pd.DataFrame] = None) -> np.ndarray:
        return self.model.predict_proba(self._X(df))[:, 1]


class TGNMemoryModel:
    """TGN-style memory model (Rossi et al., 2020), simplified for transaction streams.

    Every customer / device / card / IP has a memory vector updated by a GRU from the
    messages of the transactions it takes part in. Transactions are processed in time-ordered
    batches; a batch is scored with the memory as it stood BEFORE the batch (strictly-past
    information), then its messages are applied at the start of the next batch -- so gradients
    flow through the memory updater, as in TGN. Memory is label-free.
    """

    def __init__(self, index: TemporalIndex, mem_dim: int = 32, hidden: int = 64, lr: float = 1e-3,
                 epochs: int = 4, batch: int = 200, seed: int = 42, val_df: Optional[pd.DataFrame] = None):
        self.index, self.mem_dim, self.hidden, self.lr = index, mem_dim, hidden, lr
        self.epochs, self.batch, self.seed, self.val_df = epochs, batch, seed, val_df
        self.name = "TGN-style Memory"
        R = len(RELATIONS)
        offsets = np.concatenate([[0], np.cumsum(index.codes.max(0) + 1)[:-1]])
        self.nodes = (index.codes + offsets).astype(np.int64)  # [N,R] global node ids
        self.n_nodes = int(self.nodes.max()) + 1
        self.msg_dim = index.base.shape[1] + R

    def _build(self, d_target: int):
        torch.manual_seed(self.seed)
        R = len(RELATIONS)
        self.gru = nn.GRUCell(self.msg_dim + 1, self.mem_dim)
        self.head = nn.Sequential(nn.Linear(d_target + R * (self.mem_dim + 2), self.hidden), nn.ReLU(),
                                  nn.Linear(self.hidden, 1))
        self.params = list(self.gru.parameters()) + list(self.head.parameters())

    def _stream(self, Xt_all: np.ndarray, upto: int, train_mask: Optional[np.ndarray] = None,
                y: Optional[np.ndarray] = None, opt=None) -> np.ndarray:
        """Run the memory over rows [0, upto); optionally train on rows in train_mask."""
        R = len(RELATIONS)
        mem = torch.zeros(self.n_nodes, self.mem_dim)
        last = torch.zeros(self.n_nodes)
        seen = torch.zeros(self.n_nodes, dtype=torch.bool)
        pend_nodes = pend_msg = pend_t = None
        preds = np.zeros(upto, dtype=np.float32)
        rel_onehot = np.eye(R, dtype=np.float32)
        pos_w = torch.tensor(5.0)
        for s in range(0, upto, self.batch):
            e = min(upto, s + self.batch)
            train_step = bool(opt is not None and train_mask[s:e].any())
            with torch.set_grad_enabled(train_step):
                cur = mem
                if pend_nodes is not None:
                    dt = torch.log1p((pend_t - last[pend_nodes]).clamp(min=0))[:, None]
                    upd = self.gru(torch.cat([pend_msg, dt], 1), mem[pend_nodes])
                    cur = mem.index_put((pend_nodes,), upd)
                nodes = torch.from_numpy(self.nodes[s:e])  # [b,R]
                t_now = torch.from_numpy(self.index.ts_h[s:e].astype(np.float32))
                m = cur[nodes]  # [b,R,D]
                last_after = last.index_put((pend_nodes,), pend_t) if pend_nodes is not None else last
                seen_after = seen.index_put((pend_nodes,), torch.ones_like(pend_nodes, dtype=torch.bool)) \
                    if pend_nodes is not None else seen
                dts = torch.log1p((t_now[:, None] - last_after[nodes]).clamp(min=0)) * seen_after[nodes]
                feat = torch.cat([torch.from_numpy(Xt_all[s:e]), m.reshape(e - s, -1), dts,
                                  seen_after[nodes].float()], 1)
                logit = self.head(feat).squeeze(-1)
                if train_step:
                    msk = torch.from_numpy(train_mask[s:e])
                    loss = F.binary_cross_entropy_with_logits(logit[msk], torch.from_numpy(y[s:e][train_mask[s:e]]),
                                                              pos_weight=pos_w)
                    opt.zero_grad()
                    loss.backward()
                    opt.step()
            preds[s:e] = torch.sigmoid(logit.detach()).numpy()
            # commit the pending update, then queue this batch's messages (last event per node wins)
            mem, last, seen = cur.detach(), last_after.detach(), seen_after
            flat_nodes = self.nodes[s:e].reshape(-1)
            rows = np.repeat(np.arange(s, e), R)
            rel = np.tile(np.arange(R), e - s)
            _, last_idx = np.unique(flat_nodes[::-1], return_index=True)
            keep = len(flat_nodes) - 1 - last_idx
            pend_nodes = torch.from_numpy(flat_nodes[keep])
            pend_msg = torch.from_numpy(np.concatenate([self.index.base[rows[keep]], rel_onehot[rel[keep]]], 1))
            pend_t = torch.from_numpy(self.index.ts_h[rows[keep]].astype(np.float32))
        return preds

    def fit(self, train_df: pd.DataFrame, y: np.ndarray, full_df: Optional[pd.DataFrame] = None) -> "TGNMemoryModel":
        assert full_df is not None, "TGNMemoryModel needs the full chronological dataframe"
        Xt_raw = _target_matrix(full_df)
        self.std_t = _Standardiser().fit(Xt_raw[train_df["_row"].to_numpy()])
        self.Xt_all = self.std_t(Xt_raw)
        self._build(self.Xt_all.shape[1])
        opt = torch.optim.Adam(self.params, lr=self.lr)
        upto = int(train_df["_row"].max()) + 1
        mask = np.zeros(upto, dtype=bool)
        mask[train_df["_row"].to_numpy()] = True
        yy = np.zeros(upto, dtype=np.float32)
        yy[train_df["_row"].to_numpy()] = np.asarray(y, dtype=np.float32)
        best, best_state = -1.0, None
        for _ in range(self.epochs):
            self._stream(self.Xt_all, upto, mask, yy, opt)
            if self.val_df is not None:
                p = self._stream(self.Xt_all, int(self.val_df["_row"].max()) + 1)
                ap = average_precision_score(self.val_df["fraud_label"], p[self.val_df["_row"].to_numpy()])
                if ap > best:
                    best = ap
                    best_state = ([x.detach().clone() for x in self.gru.parameters()],
                                  [x.detach().clone() for x in self.head.parameters()])
        if best_state is not None:
            with torch.no_grad():
                for p_, b in zip(self.gru.parameters(), best_state[0]):
                    p_.copy_(b)
                for p_, b in zip(self.head.parameters(), best_state[1]):
                    p_.copy_(b)
        self._cache = None
        return self

    def predict_proba(self, df: pd.DataFrame, history: Optional[pd.DataFrame] = None) -> np.ndarray:
        upto = int(df["_row"].max()) + 1
        if self._cache is None or len(self._cache) < upto:
            self._cache = self._stream(self.Xt_all, upto)
        return self._cache[df["_row"].to_numpy()]
