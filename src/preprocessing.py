"""Chronological splitting and transaction-level feature engineering.

Every feature for transaction *t* is computed from transactions strictly before *t*
(plus *t*'s own intrinsic attributes: amount, time, payment method).
"""

from __future__ import annotations

from collections import deque
from typing import Tuple

import numpy as np
import pandas as pd

PAYMENT_METHODS = ["card", "netbanking", "upi", "wallet"]

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

# Attributes of the transaction itself (no history). Used for the graph-hidden sanity check.
INTRINSIC_FEATURE_COLS = ["amount", "hour", "day_of_week", "payment_method"]


def sort_chronologically(df: pd.DataFrame) -> pd.DataFrame:
    """Deterministic chronological order (ties broken by transaction_id)."""
    return df.sort_values(["timestamp", "transaction_id"], kind="stable").reset_index(drop=True)


def chronological_split(
    df: pd.DataFrame, train_frac: float = 0.70, val_frac: float = 0.15
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split by time: first 70% train, next 15% validation, final 15% test."""
    df = sort_chronologically(df)
    n = len(df)
    train_end = int(n * train_frac)
    val_end = int(n * (train_frac + val_frac))
    return df.iloc[:train_end].copy(), df.iloc[train_end:val_end].copy(), df.iloc[val_end:].copy()


def _exclusive_group_cumsum(values: pd.Series, keys: pd.Series) -> pd.Series:
    """Per-group running sum of *previous* rows only (excludes the current row)."""
    return values.groupby(keys).cumsum() - values


def _trailing_window_count(keys: np.ndarray, ts_s: np.ndarray, window_s: int) -> np.ndarray:
    """For each row, number of earlier rows with the same key within the trailing window."""
    out = np.zeros(len(keys), dtype=np.float32)
    hist: dict = {}
    for i in range(len(keys)):
        q = hist.get(keys[i])
        if q is None:
            q = hist[keys[i]] = deque()
        cutoff = ts_s[i] - window_s
        while q and q[0] < cutoff:
            q.popleft()
        out[i] = len(q)
        q.append(ts_s[i])
    return out


def build_transaction_features(transactions: pd.DataFrame) -> pd.DataFrame:
    """Transaction-level features using only information available before each transaction."""
    tx = sort_chronologically(transactions).copy()
    tx["hour"] = tx["timestamp"].dt.hour
    tx["day_of_week"] = tx["timestamp"].dt.dayofweek

    success = tx["success"].astype(float)
    fail = 1.0 - success
    amount = tx["amount"].astype(float)
    for entity in ["customer_id", "device_id", "card_id", "merchant_id"]:
        tx[f"{entity}_txn_count"] = tx.groupby(entity).cumcount()
    cust = tx["customer_id"]
    cust_n = tx["customer_id_txn_count"]
    cust_success = _exclusive_group_cumsum(success, cust)
    cust_fail = _exclusive_group_cumsum(fail, cust)
    cust_amount = _exclusive_group_cumsum(amount, cust)

    has_hist = cust_n > 0
    # No history -> neutral prior (success 1.0, failure 0.0) and 0 for the average amount;
    # the model can tell these apart via customer_transaction_count == 0.
    tx["customer_success_rate"] = np.where(has_hist, cust_success / cust_n.clip(lower=1), 1.0)
    tx["recent_failure_rate"] = np.where(has_hist, cust_fail / cust_n.clip(lower=1), 0.0)
    tx["customer_avg_amount"] = np.where(has_hist, cust_amount / cust_n.clip(lower=1), 0.0)

    ts_s = (tx["timestamp"].astype("int64") // 10**9).to_numpy()
    velocities = [
        _trailing_window_count(tx[e].to_numpy(), ts_s, 24 * 3600) for e in ["customer_id", "device_id", "card_id"]
    ]
    tx["recent_transaction_velocity"] = np.mean(velocities, axis=0)

    tx["customer_transaction_count"] = tx["customer_id_txn_count"]
    tx["device_transaction_count"] = tx["device_id_txn_count"]
    tx["card_transaction_count"] = tx["card_id_txn_count"]
    tx["merchant_transaction_count"] = tx["merchant_id_txn_count"]
    return tx.drop(columns=[f"{e}_txn_count" for e in ["customer_id", "device_id", "card_id", "merchant_id"]])


def prepare_feature_matrix(df: pd.DataFrame, feature_cols: list[str]) -> np.ndarray:
    """Numeric matrix; payment_method is one-hot encoded over a fixed category list."""
    parts = []
    num_cols = [c for c in feature_cols if c != "payment_method"]
    if num_cols:
        parts.append(df[num_cols].astype(float).fillna(0.0).to_numpy(dtype=np.float32))
    if "payment_method" in feature_cols:
        pm = df["payment_method"].astype(str).to_numpy()
        parts.append(np.stack([(pm == m) for m in PAYMENT_METHODS], axis=1).astype(np.float32))
    return np.hstack(parts)
