"""Elliptic++ (Elmougy & Liu, KDD 2023) on the ring's clock.

Data: data/Elliptic++/txs_features.csv, txs_classes.csv, txs_edgelist.csv from the authors' Google Drive
(linked from https://github.com/git-disl/EllipticPlusPlus); git-ignored, never redistributed.

* Nodes are Bitcoin transactions in 49 time steps (~2 weeks each); class 1 = illicit, 2 = licit,
  3 = unknown. Unknown transactions are dropped from training and scoring (standard practice).
* Rings: connected components of illicit transactions in the money-flow graph. All edges join
  transactions of the same time step, so the within-ring clock is the MONEY-FLOW ORDER: if B spends an
  output of A (edge A -> B), A happened first. Transactions are ordered by topological rank.
* Amount: total_BTC. Model: XGBoost on the provided 182 local + aggregate features (the standard strong
  baseline on Elliptic). Split by time step: train 1-29, validation 30-34, test 35-49.
* Training rings for TVW are components of TRAINING illicit transactions only (no test labels).
"""

from __future__ import annotations

import os

import networkx as nx
import numpy as np
import pandas as pd

ELL_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "Elliptic++")
TRAIN_END, VAL_END = 29, 34


def _rings(ids: np.ndarray, edges: pd.DataFrame) -> tuple[dict, dict]:
    """Component id and topological rank for each id in ``ids`` (illicit transactions)."""
    s = set(ids.tolist())
    g = nx.DiGraph()
    g.add_nodes_from(s)
    e = edges[edges.txId1.isin(s) & edges.txId2.isin(s)]
    g.add_edges_from(zip(e.txId1, e.txId2))
    comp, rank = {}, {}
    for c, nodes in enumerate(nx.weakly_connected_components(g)):
        sub = g.subgraph(nodes)
        try:
            order = list(nx.topological_sort(sub))
        except nx.NetworkXUnfeasible:  # a cycle: fall back to id order
            order = sorted(nodes)
        for r, n in enumerate(order):
            comp[n], rank[n] = c, r
    return comp, rank


def load() -> pd.DataFrame:
    f = pd.read_csv(os.path.join(ELL_DIR, "txs_features.csv"))
    c = pd.read_csv(os.path.join(ELL_DIR, "txs_classes.csv"))
    e = pd.read_csv(os.path.join(ELL_DIR, "txs_edgelist.csv"))
    df = f.merge(c, on="txId")
    df = df[df["class"] != 3].reset_index(drop=True).copy()
    df["fraud_label"] = (df["class"] == 1).astype(int)
    ill = df.loc[df.fraud_label == 1, "txId"].to_numpy()
    comp, rank = _rings(ill, e)
    df["ring_id"] = df.txId.map(comp).fillna(-1).astype(int)
    df["ring_rank"] = df.txId.map(rank).fillna(0).astype(int)
    base = pd.Timestamp("2020-01-01")
    df["timestamp"] = base + pd.to_timedelta((df["Time step"] - 1) * 14, unit="D") + \
        pd.to_timedelta(df["ring_rank"], unit="s")
    df["amount"] = df["total_BTC"].astype(float).clip(lower=0)
    df["fraud_pattern"] = np.where(df.fraud_label == 1, "illicit_component", "")
    df = df.sort_values(["timestamp", "txId"], kind="stable").reset_index(drop=True)
    seq = df[df.fraud_label == 1].groupby("ring_id").cumcount()
    df["ring_seq"] = -1
    df.loc[seq.index, "ring_seq"] = seq.to_numpy()
    df["_row"] = np.arange(len(df))
    df.attrs["edges"] = e
    return df


def feature_cols(df: pd.DataFrame) -> list:
    drop = {"txId", "Time step", "class", "fraud_label", "ring_id", "ring_rank", "timestamp", "amount",
            "fraud_pattern", "ring_seq", "_row"}
    return [c for c in df.columns if c not in drop]


def split(df: pd.DataFrame):
    t = df["Time step"]
    return df[t <= TRAIN_END].copy(), df[(t > TRAIN_END) & (t <= VAL_END)].copy(), df[t > VAL_END].copy()


def train_rings(train: pd.DataFrame, edges: pd.DataFrame) -> np.ndarray:
    """Ring ids from TRAINING labels only (components of training illicit transactions)."""
    comp, _ = _rings(train.loc[train.fraud_label == 1, "txId"].to_numpy(), edges)
    return train.txId.map(comp).fillna(-1).astype(int).to_numpy()
