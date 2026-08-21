"""Temporal graph-derived relationship features (no future leakage)."""

from __future__ import annotations

from collections import defaultdict
from datetime import timedelta
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

GRAPH_FEATURE_COLS = [
    "device_unique_customers",
    "device_unique_cards",
    "device_unique_ips",
    "customer_unique_devices",
    "customer_unique_cards",
    "ip_unique_customers",
    "card_unique_customers",
    "device_failure_rate",
    "card_failure_rate",
    "ip_failure_rate",
    "recent_device_accounts",
    "recent_card_accounts",
    "recent_ip_accounts",
    "new_relationship_count_7d",
    "relationship_velocity",
    "shared_entity_count",
    "network_size",
    "network_failure_rate",
    "device_account_entropy",
    "merchant_concentration",
    "relationship_recency",
]

SIMPLE_COUNT_COLS = [
    "device_unique_customers",
    "customer_unique_devices",
    "ip_unique_customers",
    "card_unique_customers",
    "shared_entity_count",
    "network_size",
]

TEMPORAL_EXTRA_COLS = [
    "new_relationship_count_7d",
    "relationship_velocity",
    "relationship_recency",
    "recent_device_accounts",
    "device_failure_rate",
    "network_failure_rate",
    "device_account_entropy",
    "merchant_concentration",
]


class TemporalGraphTracker:
    """Incrementally track entity relationships using only past transactions."""

    def __init__(self):
        # edge stores: (entity_a, entity_b) -> list of (timestamp, success)
        self.device_customers: Dict[str, dict] = defaultdict(lambda: defaultdict(list))
        self.device_cards: Dict[str, set] = defaultdict(set)
        self.device_ips: Dict[str, set] = defaultdict(set)
        self.customer_devices: Dict[str, set] = defaultdict(set)
        self.customer_cards: Dict[str, set] = defaultdict(set)
        self.ip_customers: Dict[str, dict] = defaultdict(lambda: defaultdict(list))
        self.card_customers: Dict[str, dict] = defaultdict(lambda: defaultdict(list))
        self.card_events: Dict[str, list] = defaultdict(list)
        self.device_events: Dict[str, list] = defaultdict(list)
        self.ip_events: Dict[str, list] = defaultdict(list)
        self.customer_merchants: Dict[str, dict] = defaultdict(lambda: defaultdict(int))
        self.relationship_first_seen: Dict[tuple, pd.Timestamp] = {}

    def _record_relationship(self, key: tuple, ts: pd.Timestamp) -> None:
        if key not in self.relationship_first_seen:
            self.relationship_first_seen[key] = ts

    def observe(self, row: pd.Series) -> None:
        """Update graph state AFTER features for this row are computed."""
        ts = row["timestamp"]
        c, d, k, ip, m = (
            row["customer_id"],
            row["device_id"],
            row["card_id"],
            row["ip_id"],
            row["merchant_id"],
        )
        success = int(row["success"])

        self._record_relationship(("customer", c, "device", d), ts)
        self._record_relationship(("customer", c, "card", k), ts)
        self._record_relationship(("customer", c, "ip", ip), ts)
        self._record_relationship(("customer", c, "merchant", m), ts)

        self.device_customers[d][c].append((ts, success))
        self.device_cards[d].add(k)
        self.device_ips[d].add(ip)
        self.customer_devices[c].add(d)
        self.customer_cards[c].add(k)
        self.ip_customers[ip][c].append((ts, success))
        self.card_customers[k][c].append((ts, success))
        self.card_events[k].append((ts, success))
        self.device_events[d].append((ts, success))
        self.ip_events[ip].append((ts, success))
        self.customer_merchants[c][m] += 1

    def _failure_rate(self, events: list) -> float:
        if not events:
            return 0.0
        return 1.0 - sum(s for _, s in events) / len(events)

    def _recent_unique(
        self, entity_map: dict, entity_id: str, ts: pd.Timestamp, days: int = 7
    ) -> int:
        cutoff = ts - timedelta(days=days)
        if entity_id not in entity_map:
            return 0
        recent = set()
        for cust, evts in entity_map[entity_id].items():
            if any(t >= cutoff for t, _ in evts):
                recent.add(cust)
        return len(recent)

    def _entropy(self, counts: dict) -> float:
        total = sum(counts.values())
        if total == 0:
            return 0.0
        probs = [v / total for v in counts.values()]
        return -sum(p * np.log2(p + 1e-12) for p in probs)

    def _new_relationships_7d(
        self, c: str, d: str, k: str, ip: str, m: str, ts: pd.Timestamp
    ) -> int:
        cutoff = ts - timedelta(days=7)
        keys = [
            ("customer", c, "device", d),
            ("customer", c, "card", k),
            ("customer", c, "ip", ip),
            ("customer", c, "merchant", m),
        ]
        count = 0
        for key in keys:
            first = self.relationship_first_seen.get(key)
            if first is not None and first >= cutoff:
                count += 1
        return count

    def _relationship_recency(self, c: str, d: str, k: str, ip: str, ts: pd.Timestamp) -> float:
        ages = []
        for key in [
            ("customer", c, "device", d),
            ("customer", c, "card", k),
            ("customer", c, "ip", ip),
        ]:
            first = self.relationship_first_seen.get(key)
            if first is not None:
                ages.append((ts - first).total_seconds() / 86400.0)
        if not ages:
            return 365.0
        return float(np.mean(ages))

    def _relationship_velocity(self, c: str, ts: pd.Timestamp, days: int = 7) -> float:
        cutoff = ts - timedelta(days=days)
        new_count = sum(
            1
            for key, first in self.relationship_first_seen.items()
            if key[0] == "customer" and key[1] == c and first >= cutoff
        )
        return new_count / max(days, 1)

    def _merchant_concentration(self, c: str) -> float:
        counts = self.customer_merchants.get(c, {})
        if not counts:
            return 0.0
        total = sum(counts.values())
        max_share = max(counts.values()) / total
        return max_share

    def _network_size(self, c: str, d: str, k: str, ip: str) -> int:
        nodes = {c, d, k, ip}
        nodes.update(self.customer_devices.get(c, set()))
        nodes.update(self.customer_cards.get(c, set()))
        for cust in self.device_customers.get(d, {}):
            nodes.add(cust)
        for cust in self.ip_customers.get(ip, {}):
            nodes.add(cust)
        return len(nodes)

    def _shared_entity_count(self, c: str, d: str, ip: str) -> int:
        dev_custs = set(self.device_customers.get(d, {}).keys())
        ip_custs = set(self.ip_customers.get(ip, {}).keys())
        shared = (dev_custs & ip_custs) - {c}
        return len(shared)

    def compute_features(self, row: pd.Series) -> dict:
        """Compute graph features using state BEFORE this transaction."""
        ts = row["timestamp"]
        c, d, k, ip = row["customer_id"], row["device_id"], row["card_id"], row["ip_id"]

        dev_custs = set(self.device_customers.get(d, {}).keys())
        dev_cards = self.device_cards.get(d, set())
        dev_ips = self.device_ips.get(d, set())
        cust_devs = self.customer_devices.get(c, set())
        cust_cards = self.customer_cards.get(c, set())
        ip_custs = set(self.ip_customers.get(ip, {}).keys())
        card_custs = set(self.card_customers.get(k, {}).keys())

        device_cust_counts = {cust: len(evts) for cust, evts in self.device_customers.get(d, {}).items()}

        network_events = (
            list(self.device_events.get(d, []))
            + list(self.card_events.get(k, []))
            + list(self.ip_events.get(ip, []))
        )

        return {
            "device_unique_customers": len(dev_custs),
            "device_unique_cards": len(dev_cards),
            "device_unique_ips": len(dev_ips),
            "customer_unique_devices": len(cust_devs),
            "customer_unique_cards": len(cust_cards),
            "ip_unique_customers": len(ip_custs),
            "card_unique_customers": len(card_custs),
            "device_failure_rate": self._failure_rate(self.device_events.get(d, [])),
            "card_failure_rate": self._failure_rate(self.card_events.get(k, [])),
            "ip_failure_rate": self._failure_rate(self.ip_events.get(ip, [])),
            "recent_device_accounts": self._recent_unique(self.device_customers, d, ts, 7),
            "recent_card_accounts": self._recent_unique(self.card_customers, k, ts, 7),
            "recent_ip_accounts": self._recent_unique(self.ip_customers, ip, ts, 7),
            "new_relationship_count_7d": self._new_relationships_7d(c, d, k, ip, row["merchant_id"], ts),
            "relationship_velocity": self._relationship_velocity(c, ts),
            "shared_entity_count": self._shared_entity_count(c, d, ip),
            "network_size": self._network_size(c, d, k, ip),
            "network_failure_rate": self._failure_rate(network_events),
            "device_account_entropy": self._entropy(device_cust_counts),
            "merchant_concentration": self._merchant_concentration(c),
            "relationship_recency": self._relationship_recency(c, d, k, ip, ts),
        }


def build_graph_features(transactions: pd.DataFrame) -> pd.DataFrame:
    """Process transactions chronologically; features use only past data."""
    tx = transactions.sort_values("timestamp").reset_index(drop=True)
    tracker = TemporalGraphTracker()
    feature_rows: List[dict] = []

    for row in tx.itertuples(index=False):
        row_s = pd.Series(row._asdict())
        feats = tracker.compute_features(row_s)
        feature_rows.append(feats)
        tracker.observe(row_s)

    feat_df = pd.DataFrame(feature_rows)
    return pd.concat([tx.reset_index(drop=True), feat_df], axis=1)


def get_ablation_feature_sets() -> Dict[str, List[str]]:
    return {
        "transaction_only": [],
        "transaction_plus_counts": SIMPLE_COUNT_COLS,
        "transaction_plus_temporal": SIMPLE_COUNT_COLS + TEMPORAL_EXTRA_COLS,
        "transaction_plus_full_graph": GRAPH_FEATURE_COLS,
    }
