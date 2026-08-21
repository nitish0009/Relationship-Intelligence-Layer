"""Human-readable relationship risk explanations."""

from __future__ import annotations

from typing import Dict, List, Tuple

import pandas as pd

from src.graph_features import GRAPH_FEATURE_COLS


def _signal_templates() -> List[Tuple[str, str, float]]:
    """feature_key, message template, weight for ranking."""
    return [
        ("device_unique_customers", "Device connected to {v:.0f} unique customers.", 1.0),
        ("recent_device_accounts", "{v:.0f} device-customer relationships created within 7 days.", 0.95),
        ("ip_unique_customers", "IP associated with {v:.0f} accounts.", 0.9),
        ("recent_ip_accounts", "{v:.0f} IP-account relationships created within 7 days.", 0.88),
        ("card_unique_customers", "Card linked to {v:.0f} customers.", 0.85),
        ("network_failure_rate", "Connected network shows elevated failure rate ({v:.1%}).", 0.92),
        ("device_failure_rate", "Device failure rate is {v:.1%}.", 0.8),
        ("new_relationship_count_7d", "{v:.0f} new entity relationships in the last 7 days.", 0.87),
        ("relationship_velocity", "Relationship velocity is {v:.2f} new links/day.", 0.82),
        ("shared_entity_count", "{v:.0f} customers share both device and IP with this user.", 0.75),
        ("network_size", "Local relationship network spans {v:.0f} entities.", 0.6),
        ("device_account_entropy", "Device usage is spread across many accounts (entropy {v:.2f}).", 0.7),
        ("relationship_recency", "Average relationship age is only {v:.1f} days.", 0.78),
        ("merchant_concentration", "Merchant concentration index is {v:.2f}.", 0.5),
    ]


def _percentile_thresholds(train_df: pd.DataFrame) -> Dict[str, float]:
    thresholds = {}
    for col in GRAPH_FEATURE_COLS:
        if col in train_df.columns:
            thresholds[col] = float(train_df[col].quantile(0.90))
    return thresholds


def explain_transaction(
    row: pd.Series,
    relationship_score: float,
    thresholds: Dict[str, float],
    top_k: int = 5,
) -> Dict:
    """Produce relationship risk signals (not accusations)."""
    signals: List[Tuple[float, str]] = []

    for key, template, weight in _signal_templates():
        val = row.get(key, 0)
        thr = thresholds.get(key, 0)
        if key == "relationship_recency":
            # lower is more suspicious
            if val < max(thr * 0.3, 14):
                msg = template.format(v=val)
                severity = weight * (1 - min(val / 30, 1))
                signals.append((severity, msg))
        elif val >= max(thr, 1):
            msg = template.format(v=val)
            severity = weight * min(val / max(thr, 1), 3)
            signals.append((severity, msg))

    signals.sort(key=lambda x: x[0], reverse=True)
    explanations = [s[1] for s in signals[:top_k]]

    if not explanations:
        explanations = ["Relationship context appears within normal historical bounds."]

    combined = relationship_score
    if combined >= 0.75:
        decision = "HIGH_RISK"
    elif combined >= 0.4:
        decision = "REVIEW"
    else:
        decision = "ALLOW"

    return {
        "relationship_risk_score": round(float(relationship_score), 4),
        "decision_hint": decision,
        "explanations": explanations,
        "disclaimer": "Relationship context is anomalous relative to historical patterns; not a fraud determination.",
    }


def decision_from_scores(base: float, rel: float, combined: float) -> str:
    score = combined
    if score >= 0.75:
        return "HIGH_RISK"
    if score >= 0.4 or (rel >= 0.65 and base < 0.3):
        return "REVIEW"
    return "ALLOW"
