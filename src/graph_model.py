"""Relationship / combined tabular classifiers and a lightweight GraphSAGE model."""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

# Single-threaded torch: the graph is small, and on macOS torch's OpenMP pool deadlocks once
# XGBoost's (separate) libomp pool has been used in the same process. It also keeps the
# encoder bit-for-bit deterministic.
torch.set_num_threads(1)

from src.classifier_utils import get_classifier
from src.graph_features import GRAPH_FEATURE_COLS
from src.preprocessing import TRANSACTION_FEATURE_COLS, prepare_feature_matrix, sort_chronologically


class TabularRiskModel:
    """Gradient-boosted classifier over an explicit list of feature columns."""

    def __init__(self, feature_cols: List[str], name: str = "model", seed: int = 42):
        self.feature_cols = list(feature_cols)
        self.name = name
        self.model = get_classifier(seed)

    def fit(self, df: pd.DataFrame, y: np.ndarray, sample_weight: Optional[np.ndarray] = None) -> "TabularRiskModel":
        self.model.fit(prepare_feature_matrix(df, self.feature_cols), y, sample_weight=sample_weight)
        return self

    def predict_proba(self, df: pd.DataFrame, history: Optional[pd.DataFrame] = None) -> np.ndarray:
        return self.model.predict_proba(prepare_feature_matrix(df, self.feature_cols))[:, 1]


class RelationshipModel(TabularRiskModel):
    def __init__(self, seed: int = 42):
        super().__init__(GRAPH_FEATURE_COLS, name="Relationship Intelligence Model", seed=seed)


class CombinedModel(TabularRiskModel):
    def __init__(self, seed: int = 42):
        super().__init__(
            TRANSACTION_FEATURE_COLS + GRAPH_FEATURE_COLS, name="Combined Transaction + Relationship Model", seed=seed
        )


# ---------------------------------------------------------------------------- GraphSAGE

ENTITY_COLS = [("customer_id", 0), ("device_id", 1), ("card_id", 2), ("ip_id", 3), ("merchant_id", 4)]
N_TYPES = len(ENTITY_COLS)


class SimpleGraphSAGE(nn.Module):
    """Two-layer mean-aggregation GraphSAGE encoder (no PyG dependency)."""

    def __init__(self, in_dim: int, hidden: int = 32, out_dim: int = 16):
        super().__init__()
        self.lin1 = nn.Linear(in_dim * 2, hidden)
        self.lin2 = nn.Linear(hidden * 2, out_dim)

    @staticmethod
    def _mean_neigh(h: torch.Tensor, src: torch.Tensor, dst: torch.Tensor, deg: torch.Tensor) -> torch.Tensor:
        agg = torch.zeros_like(h)
        agg.index_add_(0, dst, h[src])
        return agg / deg

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        src, dst = edge_index[0], edge_index[1]
        deg = torch.zeros(x.size(0), device=x.device).index_add_(0, dst, torch.ones_like(dst, dtype=torch.float32))
        deg = deg.clamp(min=1).unsqueeze(1)
        h = F.relu(self.lin1(torch.cat([x, self._mean_neigh(x, src, dst, deg)], dim=1)))
        return self.lin2(torch.cat([h, self._mean_neigh(h, src, dst, deg)], dim=1))


class _EntityGraph:
    """Undirected entity graph built from a set of (past) transactions."""

    def __init__(self, history: pd.DataFrame):
        keys = []
        for col, t in ENTITY_COLS:
            keys.append(np.char.add(f"{t}:", history[col].astype(str).to_numpy()))
        all_keys = np.concatenate(keys)
        uniq, inv = np.unique(all_keys, return_inverse=True)
        self.node_index: Dict[str, int] = {k: i for i, k in enumerate(uniq)}
        n_tx = len(history)
        node_of = inv.reshape(N_TYPES, n_tx)  # [type, tx]

        # node features: one-hot type, log txn count, failure rate, log degree
        n = len(uniq)
        succ = history["success"].to_numpy(dtype=np.float32)
        count = np.zeros(n, dtype=np.float32)
        fails = np.zeros(n, dtype=np.float32)
        ntype = np.zeros(n, dtype=np.int64)
        for t in range(N_TYPES):
            np.add.at(count, node_of[t], 1.0)
            np.add.at(fails, node_of[t], 1.0 - succ)
            ntype[node_of[t]] = t

        pairs = []
        for a in range(N_TYPES):
            for b in range(a + 1, N_TYPES):
                pairs.append(np.stack([node_of[a], node_of[b]]))
        e = np.unique(np.concatenate(pairs, axis=1), axis=1)
        self.edge_index = np.concatenate([e, e[::-1]], axis=1).astype(np.int64)
        deg = np.bincount(self.edge_index[1], minlength=n).astype(np.float32)

        feat = np.zeros((n, N_TYPES + 3), dtype=np.float32)
        feat[np.arange(n), ntype] = 1.0
        feat[:, N_TYPES] = np.log1p(count)
        feat[:, N_TYPES + 1] = fails / np.maximum(count, 1)
        feat[:, N_TYPES + 2] = np.log1p(deg)
        self.x = feat


class GraphSAGEModel:
    """GraphSAGE embeddings (self-supervised link prediction) + gradient-boosted classifier.

    Leakage-safe protocol:
      * fit(): the training period is split in time. The graph is built from the first
        ``graph_frac`` of it; the classifier is trained on embeddings of the *later*
        training transactions, which are not part of that graph (so a row never
        contributes to its own embedding, matching what happens at test time).
      * predict_proba(df, history): embeddings come from a graph built only from
        ``history`` (all transactions before ``df``). The encoder is inductive, so it can
        be applied to the larger history graph. Entities first seen inside ``df`` get a
        zero embedding.
    """

    EMBED_DIM = 16

    def __init__(self, hidden: int = 32, epochs: int = 60, graph_frac: float = 0.6, seed: int = 42):
        self.name = "GraphSAGE Model"
        self.hidden, self.epochs, self.graph_frac, self.seed = hidden, epochs, graph_frac, seed
        self.encoder: Optional[SimpleGraphSAGE] = None
        self.classifier = get_classifier(seed)

    def _embed(self, graph: _EntityGraph) -> np.ndarray:
        with torch.no_grad():
            return self.encoder(torch.from_numpy(graph.x), torch.from_numpy(graph.edge_index)).numpy()

    def _tx_matrix(self, df: pd.DataFrame, graph: _EntityGraph, emb: np.ndarray) -> np.ndarray:
        out = np.zeros((len(df), N_TYPES * (self.EMBED_DIM + 1)), dtype=np.float32)
        for col, t in ENTITY_COLS:
            keys = [f"{t}:{v}" for v in df[col].astype(str).to_numpy()]
            idx = np.array([graph.node_index.get(k, -1) for k in keys])
            seen = idx >= 0
            base = t * (self.EMBED_DIM + 1)
            out[seen, base : base + self.EMBED_DIM] = emb[idx[seen]]
            out[:, base + self.EMBED_DIM] = seen  # "entity seen before" indicator
        return out

    def _train_encoder(self, graph: _EntityGraph) -> None:
        torch.manual_seed(self.seed)
        gen = torch.Generator().manual_seed(self.seed)
        x, ei = torch.from_numpy(graph.x), torch.from_numpy(graph.edge_index)
        self.encoder = SimpleGraphSAGE(x.size(1), self.hidden, self.EMBED_DIM)
        opt = torch.optim.Adam(self.encoder.parameters(), lr=0.01)
        n_e = min(20_000, ei.size(1))
        for _ in range(self.epochs):
            opt.zero_grad()
            out = self.encoder(x, ei)
            idx = torch.randint(0, ei.size(1), (n_e,), generator=gen)
            src, dst = ei[0, idx], ei[1, idx]
            neg = torch.randint(0, x.size(0), (n_e,), generator=gen)
            pos_s = (out[src] * out[dst]).sum(1)
            neg_s = (out[src] * out[neg]).sum(1)
            loss = F.binary_cross_entropy_with_logits(
                torch.cat([pos_s, neg_s]), torch.cat([torch.ones(n_e), torch.zeros(n_e)])
            )
            loss.backward()
            opt.step()
        self.encoder.eval()

    def fit(self, train_df: pd.DataFrame, y: np.ndarray) -> "GraphSAGEModel":
        order = np.argsort(train_df["timestamp"].to_numpy(), kind="stable")
        df = train_df.iloc[order].reset_index(drop=True)
        y = np.asarray(y)[order]
        cut = int(len(df) * self.graph_frac)
        graph = _EntityGraph(df.iloc[:cut])
        self._train_encoder(graph)
        X = self._tx_matrix(df.iloc[cut:], graph, self._embed(graph))
        self.classifier.fit(X, y[cut:])
        return self

    def predict_proba(self, df: pd.DataFrame, history: Optional[pd.DataFrame] = None) -> np.ndarray:
        if history is None or len(history) == 0:
            raise ValueError("GraphSAGEModel.predict_proba needs the transactions preceding df as history")
        if history["timestamp"].max() > df["timestamp"].min():
            raise ValueError("history must end before the scored transactions start (leakage guard)")
        graph = _EntityGraph(sort_chronologically(history))
        return self.classifier.predict_proba(self._tx_matrix(df, graph, self._embed(graph)))[:, 1]


class GraphSAGETransductive(GraphSAGEModel):
    """'Published-style' transductive GraphSAGE (C3 contrast): the graph is built from ALL
    transactions -- including validation/test -- and each training row contributes to its own
    embedding. This is the leaky protocol common in the literature; never use it for claims."""

    def __init__(self, all_df: pd.DataFrame, **kw):
        super().__init__(**kw)
        self.name = "GraphSAGE (transductive)"
        self.all_df = all_df

    def fit(self, train_df: pd.DataFrame, y: np.ndarray) -> "GraphSAGETransductive":
        self.graph = _EntityGraph(sort_chronologically(self.all_df))
        self._train_encoder(self.graph)
        self.emb = self._embed(self.graph)
        self.classifier.fit(self._tx_matrix(train_df, self.graph, self.emb), y)
        return self

    def predict_proba(self, df: pd.DataFrame, history: Optional[pd.DataFrame] = None) -> np.ndarray:
        return self.classifier.predict_proba(self._tx_matrix(df, self.graph, self.emb))[:, 1]
