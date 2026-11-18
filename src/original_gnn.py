"""Wrapper that runs the ORIGINAL CARE-GNN / PC-GNN implementations (scripts/run_original_gnn.py)
on our transaction graph, in a subprocess.

Requires the third-party code: ``scripts/fetch_third_party.sh`` (pinned commits, not vendored).

Graph: transaction nodes; three relations built from the strictly-past temporal neighbour index
(K most recent EARLIER transactions): (1) same device, (2) same IP, (3) same customer or card
(CARE-GNN/PC-GNN are hard-wired to exactly three relations). Self-loops are added as in the
authors' preprocessing. Node features: our transaction features, min-max scaled with training
statistics. No labels enter the graph; edges only point to earlier transactions, so the
authors' transductive training is leakage-free here.
"""

from __future__ import annotations

import os
import pickle
import subprocess
import sys
import tempfile
from typing import Optional

import numpy as np
import pandas as pd

from src.preprocessing import TRANSACTION_FEATURE_COLS, prepare_feature_matrix
from src.temporal_gnn import RELATIONS, TemporalIndex

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUNNER = os.path.join(ROOT, "scripts", "run_original_gnn.py")


def third_party_available() -> bool:
    return all(os.path.isdir(os.path.join(ROOT, "third_party", d)) for d in ("CARE-GNN", "PC-GNN"))


class OriginalFraudGNN:
    def __init__(self, index: TemporalIndex, kind: str, full_df: pd.DataFrame, val_df: pd.DataFrame,
                 seed: int = 42, epochs: Optional[int] = None, row_norm: bool = True):
        self.index, self.kind, self.full_df, self.val_df, self.seed, self.epochs = index, kind, full_df, val_df, seed, epochs
        self.row_norm = row_norm
        self.name = {"care": "CARE-GNN (original code)", "pcgnn": "PC-GNN (original code)"}[kind]

    def _graph(self, train_df: pd.DataFrame) -> dict:
        nbr = self.index.nbr
        r = {c: RELATIONS.index(c) for c in RELATIONS}
        groups = [[r["device_id"]], [r["ip_id"]], [r["customer_id"], r["card_id"]]]
        adj = []
        for rels in groups:
            d = {}
            for i in range(nbr.shape[0]):
                s = {i}
                for rr in rels:
                    s.update(int(x) for x in nbr[i, rr] if x >= 0)
                d[i] = s
            adj.append(d)
        homo = {i: adj[0][i] | adj[1][i] | adj[2][i] for i in range(nbr.shape[0])}
        X = prepare_feature_matrix(self.full_df, TRANSACTION_FEATURE_COLS).astype(np.float64)
        tr = train_df["_row"].to_numpy()
        lo, hi = X[tr].min(0), X[tr].max(0)
        X = np.clip((X - lo) / np.where(hi > lo, hi - lo, 1.0), 0, 1)
        rows = self.full_df["_row"].to_numpy()
        rest = np.setdiff1d(rows, np.concatenate([tr, self.val_df["_row"].to_numpy()]))
        return {"features": X.astype(np.float32), "labels": self.full_df["fraud_label"].to_numpy(),
                "adj": adj, "homo": homo, "idx_train": tr, "idx_val": self.val_df["_row"].to_numpy(), "idx_test": rest}

    def fit(self, train_df: pd.DataFrame, y: np.ndarray) -> "OriginalFraudGNN":
        g = self._graph(train_df)
        with tempfile.TemporaryDirectory() as tmp:
            gp, op = os.path.join(tmp, "g.pkl"), os.path.join(tmp, "out.npz")
            with open(gp, "wb") as f:
                pickle.dump(g, f, protocol=4)
            cmd = [sys.executable, RUNNER, self.kind, gp, op, "--seed", str(self.seed)]
            if self.epochs:
                cmd += ["--epochs", str(self.epochs)]
            if not self.row_norm:
                cmd += ["--no-row-norm"]
            proc = subprocess.run(cmd, capture_output=True, text=True)
            if proc.returncode != 0:
                raise RuntimeError(f"{self.name} failed:\n{proc.stdout[-2000:]}\n{proc.stderr[-3000:]}")
            o = np.load(op)
            self.log = proc.stdout
        self.pred = pd.Series(np.concatenate([o["p_val"], o["p_test"]]),
                              index=np.concatenate([g["idx_val"], g["idx_test"]]))
        return self

    def predict_proba(self, df: pd.DataFrame, history=None) -> np.ndarray:
        return self.pred.loc[df["_row"].to_numpy()].to_numpy()
