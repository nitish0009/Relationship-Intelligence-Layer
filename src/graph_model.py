"""Relationship intelligence and combined classifiers + lightweight GraphSAGE."""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.classifier_utils import get_classifier
from src.graph_features import GRAPH_FEATURE_COLS
from src.preprocessing import TRANSACTION_FEATURE_COLS, prepare_feature_matrix


class TabularRiskModel:
    """Generic tabular classifier for graph-only or combined features."""

    def __init__(self, feature_cols: List[str], name: str = "model"):
        self.feature_cols = feature_cols
        self.name = name
        self.model = get_classifier()
        self.tx_encoders: dict = {}
        self.use_transaction_features = any(
            c in TRANSACTION_FEATURE_COLS for c in feature_cols
        ) or any(c in TRANSACTION_FEATURE_COLS for c in [])

    def _build_X(self, df: pd.DataFrame) -> np.ndarray:
        parts = []
        has_tx = any(c in self.feature_cols for c in TRANSACTION_FEATURE_COLS)
        if has_tx:
            X_tx, self.tx_encoders = prepare_feature_matrix(
                df, TRANSACTION_FEATURE_COLS, self.tx_encoders
            )
            parts.append(X_tx)
        graph_cols = [c for c in self.feature_cols if c in GRAPH_FEATURE_COLS]
        if graph_cols:
            parts.append(df[graph_cols].fillna(0).values.astype(np.float32))
        if not parts:
            raise ValueError("No features available")
        return np.hstack(parts) if len(parts) > 1 else parts[0]

    def fit(self, df: pd.DataFrame, y: np.ndarray) -> "TabularRiskModel":
        X = self._build_X(df)
        self.model.fit(X, y)
        return self

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        X = self._build_X(df)
        return self.model.predict_proba(X)[:, 1]

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump(
            {
                "model": self.model,
                "feature_cols": self.feature_cols,
                "tx_encoders": self.tx_encoders,
                "name": self.name,
            },
            path,
        )

    @classmethod
    def load(cls, path: str) -> "TabularRiskModel":
        data = joblib.load(path)
        obj = cls(feature_cols=data["feature_cols"], name=data["name"])
        obj.model = data["model"]
        obj.tx_encoders = data["tx_encoders"]
        return obj


class RelationshipModel(TabularRiskModel):
    def __init__(self):
        super().__init__(feature_cols=GRAPH_FEATURE_COLS, name="Relationship Intelligence Model")


class CombinedModel(TabularRiskModel):
    def __init__(self):
        super().__init__(
            feature_cols=TRANSACTION_FEATURE_COLS + GRAPH_FEATURE_COLS,
            name="Combined Transaction + Relationship Model",
        )


class SimpleGraphSAGE(nn.Module):
    """Lightweight sparse message-passing encoder (no PyG dependency)."""

    def __init__(self, in_dim: int, hidden: int = 32, out_dim: int = 16):
        super().__init__()
        self.lin1 = nn.Linear(in_dim * 2, hidden)
        self.lin2 = nn.Linear(hidden, out_dim)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        n = x.size(0)
        src, dst = edge_index[0], edge_index[1]
        neigh = torch.zeros_like(x)
        neigh.index_add_(0, dst, x[src])
        deg = torch.zeros(n, device=x.device)
        deg.index_add_(0, dst, torch.ones(dst.size(0), device=x.device))
        deg = deg.clamp(min=1).unsqueeze(1)
        neigh = neigh / deg
        h = torch.cat([x, neigh], dim=1)
        h = F.relu(self.lin1(h))
        return self.lin2(h)


class GraphSAGEModel:
    """Train GraphSAGE-style embeddings on a sampled entity graph."""

    EMBED_DIM = 16
    MAX_TRAIN_TX = 12_000  # keep graph small for one-day MVP

    def __init__(self, hidden: int = 32):
        self.hidden = hidden
        self.encoder: Optional[SimpleGraphSAGE] = None
        self.classifier = get_classifier()
        self.node_id_map: Dict[str, int] = {}
        self.node_features: Optional[np.ndarray] = None
        self.edge_index: Optional[np.ndarray] = None
        self._cached_emb: Optional[np.ndarray] = None
        self.fitted = False

    def _build_graph_from_train(self, train_df: pd.DataFrame) -> None:
        sample = train_df
        if len(sample) > self.MAX_TRAIN_TX:
            sample = train_df.sample(self.MAX_TRAIN_TX, random_state=42)

        nodes: Dict[str, Tuple[int, list]] = {}
        edges: List[Tuple[int, int]] = []

        def get_node(key: str, ntype: int) -> int:
            if key not in nodes:
                idx = len(nodes)
                nodes[key] = (idx, [ntype, 0, 0, 0])
            return nodes[key][0]

        for _, row in sample.iterrows():
            ids = [
                (f"c:{row['customer_id']}", 0),
                (f"d:{row['device_id']}", 1),
                (f"k:{row['card_id']}", 2),
                (f"i:{row['ip_id']}", 3),
                (f"m:{row['merchant_id']}", 4),
            ]
            nids = [get_node(k_, t) for k_, t in ids]
            for i in range(len(nids)):
                for j in range(i + 1, len(nids)):
                    edges.append((nids[i], nids[j]))
                    edges.append((nids[j], nids[i]))
            succ = int(row["success"])
            for key, _ in ids:
                _, f = nodes[key]
                f[1] += 1
                f[2] += succ
                f[3] += 1 - succ

        n = len(nodes)
        self.node_id_map = {k: v[0] for k, v in nodes.items()}
        feat = np.zeros((n, 4), dtype=np.float32)
        for _, (idx, f) in nodes.items():
            feat[idx] = [
                f[0] / 4.0,
                np.log1p(f[1]),
                f[2] / max(f[1], 1),
                f[3] / max(f[1], 1),
            ]
        if not edges:
            edges = [(0, 0)]
        ei = np.array(edges, dtype=np.int64).T  # [2, E]
        self.node_features = feat
        self.edge_index = ei

    def _encode_all(self) -> np.ndarray:
        x = torch.tensor(self.node_features, dtype=torch.float32)
        ei = torch.tensor(self.edge_index, dtype=torch.long)
        with torch.no_grad():
            return self.encoder(x, ei).numpy()

    def _transaction_embeddings(self, df: pd.DataFrame) -> np.ndarray:
        if self._cached_emb is None:
            self._cached_emb = self._encode_all()
        all_emb = self._cached_emb
        out = np.zeros((len(df), 5 * self.EMBED_DIM), dtype=np.float32)
        keys_cols = [
            ("customer_id", "c:"),
            ("device_id", "d:"),
            ("card_id", "k:"),
            ("ip_id", "i:"),
            ("merchant_id", "m:"),
        ]
        for i, (_, row) in enumerate(df.iterrows()):
            parts = []
            for col, prefix in keys_cols:
                key = f"{prefix}{row[col]}"
                if key in self.node_id_map:
                    parts.append(all_emb[self.node_id_map[key]])
                else:
                    parts.append(np.zeros(self.EMBED_DIM, dtype=np.float32))
            out[i] = np.concatenate(parts)
        return out

    def fit(self, train_df: pd.DataFrame, y: np.ndarray) -> "GraphSAGEModel":
        self._build_graph_from_train(train_df)
        in_dim = self.node_features.shape[1]
        self.encoder = SimpleGraphSAGE(in_dim, self.hidden, self.EMBED_DIM)
        x = torch.tensor(self.node_features, dtype=torch.float32)
        ei = torch.tensor(self.edge_index, dtype=torch.long)
        opt = torch.optim.Adam(self.encoder.parameters(), lr=0.02)

        for _ in range(40):
            opt.zero_grad()
            out = self.encoder(x, ei)
            # sample edges for link-prediction loss
            e = self.edge_index
            n_e = min(2000, e.shape[1])
            idx = np.random.choice(e.shape[1], size=n_e, replace=False)
            src, dst = e[0, idx], e[1, idx]
            pos_score = (out[src] * out[dst]).sum(dim=1)
            neg_dst = torch.randint(0, out.size(0), (n_e,))
            neg_score = (out[src] * out[neg_dst]).sum(dim=1)
            loss = F.binary_cross_entropy_with_logits(
                torch.cat([pos_score, neg_score]),
                torch.cat([torch.ones(n_e), torch.zeros(n_e)]),
            )
            loss.backward()
            opt.step()

        self._cached_emb = self._encode_all()
        # Align labels if we subsampled for graph; train classifier on full train embeddings
        X = self._transaction_embeddings(train_df)
        self.classifier.fit(X, y)
        self.fitted = True
        return self

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        self._cached_emb = self._encode_all()
        X = self._transaction_embeddings(df)
        return self.classifier.predict_proba(X)[:, 1]

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump(
            {
                "encoder_state": self.encoder.state_dict() if self.encoder else None,
                "classifier": self.classifier,
                "node_id_map": self.node_id_map,
                "node_features": self.node_features,
                "edge_index": self.edge_index,
                "hidden": self.hidden,
            },
            path,
        )

    @classmethod
    def load(cls, path: str) -> "GraphSAGEModel":
        data = joblib.load(path)
        obj = cls(hidden=data["hidden"])
        obj.node_id_map = data["node_id_map"]
        obj.node_features = data["node_features"]
        obj.edge_index = data["edge_index"]
        obj.classifier = data["classifier"]
        if data["encoder_state"]:
            in_dim = obj.node_features.shape[1]
            obj.encoder = SimpleGraphSAGE(in_dim, obj.hidden, obj.EMBED_DIM)
            obj.encoder.load_state_dict(data["encoder_state"])
        obj.fitted = True
        return obj
