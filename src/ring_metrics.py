"""Ring-level early-detection metrics (C1): evaluating detection on the ring's clock.

Row-level metrics (PR-AUC, recall) ignore *when* a ring is caught. Operationally what
matters is how much damage a ring does before its first alert. For every fraud ring
that STARTS inside the scored window we measure, at a fixed (validation-tuned) threshold:

* txns_before_alert    -- ring transactions completed before the first alert
* amount_before_alert  -- money lost before the first alert
* hours_to_alert       -- time from the ring's first transaction to its first alert
* prevented_takedown   -- share of the ring's money at/after the first alert, assuming the
                          ring is taken down once alerted (upper bound of operational value)
* prevented_blocking   -- share of the ring's money in individually flagged transactions
                          (lower bound: no ring-level action)

Rings that are never alerted count all their transactions/money as lost.
"""

from __future__ import annotations

from typing import Dict, List

import numpy as np
import pandas as pd


def add_ring_sequence(tx: pd.DataFrame) -> pd.DataFrame:
    """Add ring_seq (0-based position of a fraud txn within its ring, over the FULL timeline)."""
    tx = tx.copy()
    tx["ring_seq"] = -1
    fraud = tx["fraud_label"] == 1
    order = tx.loc[fraud].sort_values(["timestamp", "transaction_id"], kind="stable")
    tx.loc[order.index, "ring_seq"] = order.groupby("ring_id").cumcount().to_numpy()
    return tx


def ring_detection_table(scored: pd.DataFrame, scores: np.ndarray, threshold: float) -> pd.DataFrame:
    """Per-ring timing table for rings whose first transaction lies inside ``scored``.

    ``scored`` must contain ring_id, ring_seq, timestamp, amount, fraud_label.
    """
    df = scored[["ring_id", "ring_seq", "timestamp", "amount", "fraud_label", "fraud_pattern"]].copy()
    df["alert"] = np.asarray(scores) >= threshold
    df = df[df["fraud_label"] == 1].sort_values(["ring_id", "ring_seq"], kind="stable")
    rows: List[Dict] = []
    for rid, g in df.groupby("ring_id", sort=True):
        if g["ring_seq"].min() != 0:  # ring started before the scored window -> no clean clock
            continue
        alerts = np.flatnonzero(g["alert"].to_numpy())
        amt = g["amount"].to_numpy()
        total = float(amt.sum())
        if len(alerts):
            k = int(alerts[0])
            hours = (g["timestamp"].iloc[k] - g["timestamp"].iloc[0]).total_seconds() / 3600
            lost, prevented_td = float(amt[:k].sum()), float(amt[k:].sum()) / total
        else:
            k, hours, lost, prevented_td = len(g), np.nan, total, 0.0
        rows.append(
            {
                "ring_id": rid,
                "fraud_pattern": g["fraud_pattern"].iloc[0],
                "ring_txns": len(g),
                "ring_amount": total,
                "detected": bool(len(alerts)),
                "txns_before_alert": k,
                "amount_before_alert": lost,
                "hours_to_alert": hours,
                "prevented_takedown": prevented_td,
                "prevented_blocking": float(amt[g["alert"].to_numpy()].sum()) / total,
            }
        )
    return pd.DataFrame(rows)


def summarise_rings(table: pd.DataFrame) -> Dict[str, float]:
    if table.empty:
        return {k: float("nan") for k in ["n_rings", "ring_recall", "median_txns_before_alert",
                                          "mean_txns_before_alert", "amount_lost_share",
                                          "median_hours_to_alert", "prevented_takedown",
                                          "prevented_blocking"]}
    return {
        "n_rings": float(len(table)),
        "ring_recall": float(table["detected"].mean()),
        "median_txns_before_alert": float(table["txns_before_alert"].median()),
        "mean_txns_before_alert": float(table["txns_before_alert"].mean()),
        "amount_lost_share": float(table["amount_before_alert"].sum() / table["ring_amount"].sum()),
        "median_hours_to_alert": float(table["hours_to_alert"].median()),
        "prevented_takedown": float((table["prevented_takedown"] * table["ring_amount"]).sum() / table["ring_amount"].sum()),
        "prevented_blocking": float((table["prevented_blocking"] * table["ring_amount"]).sum() / table["ring_amount"].sum()),
    }


def warmup_curve(scored: pd.DataFrame, scores: np.ndarray, threshold: float, max_k: int = 20) -> pd.DataFrame:
    """Recall on the k-th transaction of a ring (k = 1..max_k), rings starting in-window only.

    Shows how many transactions a ring must make before a model starts recognising it.
    """
    df = scored[["ring_id", "ring_seq", "fraud_label"]].copy()
    df["alert"] = np.asarray(scores) >= threshold
    df = df[df["fraud_label"] == 1]
    starts = df.groupby("ring_id")["ring_seq"].min()
    df = df[df["ring_id"].isin(starts[starts == 0].index)]
    df = df[df["ring_seq"] < max_k]
    curve = df.groupby("ring_seq")["alert"].agg(["mean", "size"]).reset_index()
    curve.columns = ["k", "recall", "n"]
    curve["k"] = curve["k"] + 1
    return curve
