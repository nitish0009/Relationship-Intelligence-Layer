"""Chronological splitting and transaction-level feature engineering."""

from __future__ import annotations

from typing import Tuple

import numpy as np
import pandas as pd

RANDOM_SEED = 42


def chronological_split(
    df: pd.DataFrame, train_frac: float = 0.70, val_frac: float = 0.15
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split by time: first 70% train, next 15% val, final 15% test."""
    df = df.sort_values("timestamp").reset_index(drop=True)
    n = len(df)
    train_end = int(n * train_frac)
    val_end = int(n * (train_frac + val_frac))

    train = df.iloc[:train_end].copy()
    val = df.iloc[train_end:val_end].copy()
    test = df.iloc[val_end:].copy()
    return train, val, test


def _expanding_stats(
    series: pd.Series, timestamps: pd.Series, window_hours: int = 48
) -> pd.DataFrame:
    """Compute expanding counts/rates using only past data (shift by 1)."""
    df = pd.DataFrame({"ts": timestamps, "val": series})
    df = df.sort_values("ts").reset_index(drop=True)
    return df


def build_transaction_features(transactions: pd.DataFrame) -> pd.DataFrame:
    """
    Build transaction-level features using only information available BEFORE each transaction.
    Uses cumulative/shifted statistics to avoid leakage.
    """
    tx = transactions.sort_values("timestamp").reset_index(drop=True).copy()
    tx["hour"] = tx["timestamp"].dt.hour
    tx["day_of_week"] = tx["timestamp"].dt.dayofweek

    # Per-entity cumulative stats (shifted so current tx not included)
    for entity in ["customer_id", "device_id", "card_id", "merchant_id"]:
        grp = tx.groupby(entity)
        tx[f"{entity}_txn_count"] = grp.cumcount()
        tx[f"{entity}_success_sum"] = grp["success"].cumsum().shift(1, fill_value=0)
        tx[f"{entity}_amount_sum"] = grp["amount"].cumsum().shift(1, fill_value=0)
        fail = (1 - tx["success"]).astype(float)
        tx[f"{entity}_fail_sum"] = (
            fail.groupby(tx[entity]).cumsum().shift(1, fill_value=0)
        )

    tx["customer_success_rate"] = np.where(
        tx["customer_id_txn_count"] > 0,
        tx["customer_id_success_sum"] / tx["customer_id_txn_count"],
        1.0,
    )
    tx["customer_avg_amount"] = np.where(
        tx["customer_id_txn_count"] > 0,
        tx["customer_id_amount_sum"] / tx["customer_id_txn_count"],
        tx["amount"].median(),
    )

    # Recent velocity: transactions in last 24h (approximate via rolling on time index)
    tx["ts_numeric"] = tx["timestamp"].astype(np.int64) // 10**9
    for entity in ["customer_id", "device_id", "card_id"]:
        counts = np.zeros(len(tx), dtype=np.float32)
        last_seen: dict = {}
        window = 24 * 3600
        ts_arr = tx["ts_numeric"].to_numpy()
        keys = tx[entity].to_numpy()
        for i in range(len(tx)):
            key = keys[i]
            ts = int(ts_arr[i])
            hist = last_seen.get(key)
            if hist is None:
                hist = []
                last_seen[key] = hist
            hist[:] = [t for t in hist if ts - t <= window]
            counts[i] = len(hist)
            hist.append(ts)
        tx[f"{entity}_recent_velocity"] = counts

    tx["recent_transaction_velocity"] = (
        tx["customer_id_recent_velocity"]
        + tx["device_id_recent_velocity"]
        + tx["card_id_recent_velocity"]
    ) / 3.0

    tx["recent_failure_rate"] = np.where(
        tx["customer_id_txn_count"] > 0,
        tx["customer_id_fail_sum"] / tx["customer_id_txn_count"],
        0.0,
    )

    tx["device_transaction_count"] = tx["device_id_txn_count"]
    tx["card_transaction_count"] = tx["card_id_txn_count"]
    tx["merchant_transaction_count"] = tx["merchant_id_txn_count"]
    tx["customer_transaction_count"] = tx["customer_id_txn_count"]

    return tx


TRANSACTION_FEATURE_COLS = [
    "amount",
    "hour",
    "day_of_week",
    "payment_method",
    "customer_transaction_count",
    "customer_success_rate",
    "customer_avg_amount",
    "device_transaction_count",
    "card_transaction_count",
    "merchant_transaction_count",
    "recent_transaction_velocity",
    "recent_failure_rate",
]


def prepare_feature_matrix(
    df: pd.DataFrame, feature_cols: list[str], fit_encoders: dict | None = None
) -> Tuple[np.ndarray, dict]:
    """One-hot encode categoricals; return numeric matrix."""
    work = df.copy()
    encoders = fit_encoders or {}

    cat_cols = [c for c in feature_cols if work[c].dtype == object or c == "payment_method"]
    num_cols = [c for c in feature_cols if c not in cat_cols]

    for col in cat_cols:
        if col not in encoders:
            encoders[col] = sorted(work[col].astype(str).unique().tolist())
        dummies = pd.get_dummies(work[col].astype(str), prefix=col)
        for cat in encoders[col]:
            col_name = f"{col}_{cat}"
            dummies[col_name] = (work[col].astype(str) == cat).astype(int)
        # keep only encoder columns
        dummy_cols = [f"{col}_{cat}" for cat in encoders[col]]
        for dc in dummy_cols:
            if dc not in dummies.columns:
                dummies[dc] = 0
        work = pd.concat([work, dummies[dummy_cols]], axis=1)

    all_num = num_cols + [f"{c}_{cat}" for c in cat_cols for cat in encoders.get(c, [])]
    X = work[all_num].fillna(0).values.astype(np.float32)
    return X, encoders


def get_feature_names(feature_cols: list[str], encoders: dict) -> list[str]:
    names = []
    cat_cols = [c for c in feature_cols if c == "payment_method"]
    num_cols = [c for c in feature_cols if c not in cat_cols]
    names.extend(num_cols)
    for col in cat_cols:
        for cat in encoders.get(col, []):
            names.append(f"{col}_{cat}")
    return names
