"""Human-readable relationship risk signals (not fraud determinations)."""

from __future__ import annotations

from typing import Dict, List, Tuple

import pandas as pd

# feature, message template, ranking weight, direction ("high" or "low" is suspicious)
SIGNALS: List[Tuple[str, str, float, str]] = [
    ("device_unique_customers", "Device previously used by {v:.0f} different customers.", 1.0, "high"),
    ("recent_device_accounts", "{v:.0f} distinct accounts used this device in the last 7 days.", 0.95, "high"),
    ("network_failure_rate", "Connected device/card/IP history shows a {v:.1%} failure rate.", 0.92, "high"),
    ("ip_unique_customers", "IP previously associated with {v:.0f} accounts.", 0.9, "high"),
    ("recent_ip_accounts", "{v:.0f} distinct accounts used this IP in the last 7 days.", 0.88, "high"),
    ("new_relationship_count_7d", "{v:.0f} of this transaction's 4 relationships are less than 7 days old.", 0.87, "high"),
    ("card_unique_customers", "Card previously linked to {v:.0f} customers.", 0.85, "high"),
    ("relationship_velocity", "Customer is forming {v:.2f} new relationships per day.", 0.82, "high"),
    ("device_failure_rate", "Device failure rate is {v:.1%}.", 0.8, "high"),
    ("relationship_recency", "Average age of the customer's device/card/IP links is only {v:.1f} days.", 0.78, "low"),
    ("shared_entity_count", "{v:.0f} other customers share both this device and this IP.", 0.75, "high"),
    ("device_account_entropy", "Device usage is spread across many accounts (entropy {v:.2f}).", 0.7, "high"),
    ("network_size", "Local relationship network spans {v:.0f} entities.", 0.6, "high"),
]


def fit_signal_thresholds(train_df: pd.DataFrame) -> Dict[str, float]:
    """90th percentile (or 10th for 'low' signals) of each feature on the TRAINING split."""
    out = {}
    for key, _, _, direction in SIGNALS:
        if key in train_df.columns:
            out[key] = float(train_df[key].quantile(0.90 if direction == "high" else 0.10))
    return out


def explain_transaction(row: pd.Series, thresholds: Dict[str, float], top_k: int = 5) -> List[str]:
    """Return up to top_k signals where this transaction is extreme relative to training data."""
    signals: List[Tuple[float, str]] = []
    for key, template, weight, direction in SIGNALS:
        if key not in thresholds:
            continue
        val, thr = float(row[key]), thresholds[key]
        if direction == "high" and val > max(thr, 0) and val > 0:
            signals.append((weight * min(val / max(thr, 1e-9), 3.0), template.format(v=val)))
        elif direction == "low" and val < thr:
            signals.append((weight * (1.0 + (thr - val) / max(thr, 1e-9)), template.format(v=val)))
    signals.sort(key=lambda s: s[0], reverse=True)
    return [s for _, s in signals[:top_k]] or ["Relationship context is within normal historical bounds."]
