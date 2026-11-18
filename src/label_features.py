"""Neighbour fraud-label features under a realistic label delay (C2).

In production a transaction's fraud label is only confirmed after a delay (chargeback /
investigation, typically 30-90 days). A transaction at time t_j therefore "matures" at
t_j + D: from then on its label may be used by features of later transactions. Before
maturity it contributes nothing.

Features for transaction i use only transactions j < i with t_j + D <= t_i.
D = +inf disables all label information (features are identically zero).
"""

from __future__ import annotations

from collections import defaultdict, deque
from typing import Dict, List

import numpy as np
import pandas as pd

from src.preprocessing import sort_chronologically

LABEL_FEATURE_COLS = [
    "device_known_fraud_txns",
    "ip_known_fraud_txns",
    "card_known_fraud_txns",
    "customer_known_fraud_txns",
    "device_known_fraud_rate",
    "ip_known_fraud_rate",
    "card_known_fraud_rate",
    "device_flagged_customers",
    "ip_flagged_customers",
]

_ENTITY_COLS = {"device": "device_id", "ip": "ip_id", "card": "card_id", "customer": "customer_id"}


def build_label_features(transactions: pd.DataFrame, delay_days: float) -> pd.DataFrame:
    """Return the input (chronologically sorted) with LABEL_FEATURE_COLS appended."""
    tx = sort_chronologically(transactions)
    n = len(tx)
    out = np.zeros((n, len(LABEL_FEATURE_COLS)), dtype=np.float32)
    if not np.isfinite(delay_days):
        return pd.concat([tx.drop(columns=LABEL_FEATURE_COLS, errors="ignore"),
                          pd.DataFrame(out, columns=LABEL_FEATURE_COLS)], axis=1)

    ts = tx["timestamp"].to_numpy().astype("datetime64[ns]").astype(np.int64)
    delay_ns = int(delay_days * 86_400e9)
    ents = {k: tx[c].to_numpy() for k, c in _ENTITY_COLS.items()}
    y = tx["fraud_label"].to_numpy().astype(np.int8)

    matured_n: Dict[str, Dict[str, int]] = {k: defaultdict(int) for k in _ENTITY_COLS}
    matured_fraud: Dict[str, Dict[str, int]] = {k: defaultdict(int) for k in _ENTITY_COLS}
    flagged: set = set()  # customers with >= 1 matured fraud transaction
    cust_devices: Dict[str, set] = defaultdict(set)
    cust_ips: Dict[str, set] = defaultdict(set)
    dev_customers: Dict[str, set] = defaultdict(set)
    ip_customers: Dict[str, set] = defaultdict(set)
    dev_flagged: Dict[str, int] = defaultdict(int)
    ip_flagged: Dict[str, int] = defaultdict(int)
    pending: deque = deque()  # transaction indices in time order (release order is the same)

    for i in range(n):
        # 1) release every earlier transaction whose label has matured by t_i
        while pending and ts[pending[0]] + delay_ns <= ts[i]:
            j = pending.popleft()
            for k in _ENTITY_COLS:
                matured_n[k][ents[k][j]] += 1
                if y[j]:
                    matured_fraud[k][ents[k][j]] += 1
            c = ents["customer"][j]
            if y[j] and c not in flagged:
                flagged.add(c)
                for d in cust_devices[c]:
                    dev_flagged[d] += 1
                for ip in cust_ips[c]:
                    ip_flagged[ip] += 1

        # 2) features for transaction i (state strictly before i, labels matured only)
        d, ip, k_, c = ents["device"][i], ents["ip"][i], ents["card"][i], ents["customer"][i]
        dn, ipn, kn = matured_n["device"].get(d, 0), matured_n["ip"].get(ip, 0), matured_n["card"].get(k_, 0)
        df_, ipf, kf = matured_fraud["device"].get(d, 0), matured_fraud["ip"].get(ip, 0), matured_fraud["card"].get(k_, 0)
        out[i] = [
            df_, ipf, kf, matured_fraud["customer"].get(c, 0),
            df_ / dn if dn else 0.0, ipf / ipn if ipn else 0.0, kf / kn if kn else 0.0,
            dev_flagged.get(d, 0) - (1 if (c in flagged and c in dev_customers[d]) else 0),
            ip_flagged.get(ip, 0) - (1 if (c in flagged and c in ip_customers[ip]) else 0),
        ]

        # 3) add transaction i to the (label-free) link structure; queue its label
        if d not in cust_devices[c]:
            cust_devices[c].add(d)
            dev_customers[d].add(c)
            if c in flagged:
                dev_flagged[d] += 1
        if ip not in cust_ips[c]:
            cust_ips[c].add(ip)
            ip_customers[ip].add(c)
            if c in flagged:
                ip_flagged[ip] += 1
        pending.append(i)

    feat = pd.DataFrame(out, columns=LABEL_FEATURE_COLS)
    return pd.concat([tx.drop(columns=LABEL_FEATURE_COLS, errors="ignore"), feat], axis=1)
