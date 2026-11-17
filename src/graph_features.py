"""Temporal relationship (graph) features with no future leakage.

Transactions are processed in chronological order. For each transaction the features are
computed from the graph state built from *earlier* transactions only, and the transaction
is added to the graph afterwards.

All state updates are O(1) amortised (running counters + sliding windows), except the
set intersections in ``network_size`` / ``shared_entity_count`` which are O(min degree).
"""

from __future__ import annotations

from collections import defaultdict, deque
from typing import Dict, List

import numpy as np
import pandas as pd

from src.preprocessing import sort_chronologically

WINDOW_DAYS = 7

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


class SlidingDistinct:
    """Number of distinct keys seen within a trailing time window (timestamps non-decreasing)."""

    __slots__ = ("events", "counts")

    def __init__(self):
        self.events: deque = deque()
        self.counts: Dict[str, int] = {}

    def expire(self, cutoff) -> None:
        while self.events and self.events[0][0] < cutoff:
            _, key = self.events.popleft()
            self.counts[key] -= 1
            if self.counts[key] == 0:
                del self.counts[key]

    def add(self, ts, key: str) -> None:
        self.events.append((ts, key))
        self.counts[key] = self.counts.get(key, 0) + 1

    def __len__(self) -> int:
        return len(self.counts)


class _EntityStats:
    """Per-entity running counters."""

    __slots__ = ("n", "fail", "customers", "recent", "xlogx")

    def __init__(self):
        self.n = 0
        self.fail = 0
        self.customers: Dict[str, int] = {}  # customer -> txn count (insertion ordered)
        self.recent = SlidingDistinct()
        self.xlogx = 0.0  # sum over customers of n_c * log2(n_c), for O(1) entropy

    def failure_rate(self) -> float:
        return self.fail / self.n if self.n else 0.0

    def entropy(self) -> float:
        if self.n == 0:
            return 0.0
        return float(np.log2(self.n) - self.xlogx / self.n)

    def observe(self, ts, customer: str, success: int) -> None:
        self.n += 1
        self.fail += 1 - success
        prev = self.customers.get(customer, 0)
        self.customers[customer] = prev + 1
        self.xlogx += (prev + 1) * np.log2(prev + 1) - (prev * np.log2(prev) if prev else 0.0)
        self.recent.add(ts, customer)


class TemporalGraphTracker:
    """Incrementally maintained customer/device/card/IP/merchant graph."""

    def __init__(self, window_days: int = WINDOW_DAYS):
        self.window = pd.Timedelta(days=window_days)
        self.window_days = window_days
        self.device: Dict[str, _EntityStats] = defaultdict(_EntityStats)
        self.card: Dict[str, _EntityStats] = defaultdict(_EntityStats)
        self.ip: Dict[str, _EntityStats] = defaultdict(_EntityStats)
        self.device_cards: Dict[str, set] = defaultdict(set)
        self.device_ips: Dict[str, set] = defaultdict(set)
        self.customer_devices: Dict[str, set] = defaultdict(set)
        self.customer_cards: Dict[str, set] = defaultdict(set)
        self.customer_merchants: Dict[str, Dict[str, int]] = defaultdict(dict)
        self.customer_merchant_max: Dict[str, int] = defaultdict(int)
        self.customer_n: Dict[str, int] = defaultdict(int)
        self.first_seen: Dict[tuple, pd.Timestamp] = {}
        self.customer_new_links: Dict[str, deque] = defaultdict(deque)  # first-seen times of c's links

    @staticmethod
    def _links(c, d, k, ip, m):
        return [("device", c, d), ("card", c, k), ("ip", c, ip), ("merchant", c, m)]

    def compute_features(self, ts, c, d, k, ip, m) -> list:
        """Features from the state BEFORE this transaction (order = GRAPH_FEATURE_COLS)."""
        cutoff = ts - self.window
        dev, crd, ipa = self.device.get(d), self.card.get(k), self.ip.get(ip)
        for st in (dev, crd, ipa):
            if st is not None:
                st.recent.expire(cutoff)
        dev_custs = dev.customers if dev else {}
        ip_custs = ipa.customers if ipa else {}
        cust_devs = self.customer_devices.get(c, set())
        cust_cards = self.customer_cards.get(c, set())

        # shared_entity_count: other customers seen on both this device and this IP
        small, large = (dev_custs, ip_custs) if len(dev_custs) <= len(ip_custs) else (ip_custs, dev_custs)
        shared_both = sum(1 for x in small if x in large)
        shared_entity = shared_both - (1 if (c in dev_custs and c in ip_custs) else 0)

        # network_size: |{c,d,k,ip} ∪ devices(c) ∪ cards(c) ∪ customers(d) ∪ customers(ip)|
        n_dev_nodes = len(cust_devs) + (d not in cust_devs)
        n_card_nodes = len(cust_cards) + (k not in cust_cards)
        cust_union = len(dev_custs) + len(ip_custs) - shared_both
        c_in_union = (c in dev_custs) or (c in ip_custs)
        network_size = n_dev_nodes + n_card_nodes + 1 + cust_union + (not c_in_union)

        net_n = (dev.n if dev else 0) + (crd.n if crd else 0) + (ipa.n if ipa else 0)
        net_fail = (dev.fail if dev else 0) + (crd.fail if crd else 0) + (ipa.fail if ipa else 0)

        # relationships: a link not seen before counts as new (age 0)
        links = self._links(c, d, k, ip, m)
        new_links = 0
        ages = []
        for key in links:
            first = self.first_seen.get(key)
            if first is None or first >= cutoff:
                new_links += 1
            if key[0] != "merchant":
                ages.append(0.0 if first is None else (ts - first).total_seconds() / 86400.0)

        recent_links = self.customer_new_links.get(c)
        if recent_links is not None:
            while recent_links and recent_links[0] < cutoff:
                recent_links.popleft()
        n_unseen = sum(1 for key in links if key not in self.first_seen)
        velocity = ((len(recent_links) if recent_links else 0) + n_unseen) / self.window_days

        cust_n = self.customer_n.get(c, 0)
        return [
            len(dev_custs),
            len(self.device_cards.get(d, ())),
            len(self.device_ips.get(d, ())),
            len(cust_devs),
            len(cust_cards),
            len(ip_custs),
            len(crd.customers) if crd else 0,
            dev.failure_rate() if dev else 0.0,
            crd.failure_rate() if crd else 0.0,
            ipa.failure_rate() if ipa else 0.0,
            len(dev.recent) if dev else 0,
            len(crd.recent) if crd else 0,
            len(ipa.recent) if ipa else 0,
            new_links,
            velocity,
            shared_entity,
            network_size,
            net_fail / net_n if net_n else 0.0,
            dev.entropy() if dev else 0.0,
            self.customer_merchant_max.get(c, 0) / cust_n if cust_n else 0.0,
            float(np.mean(ages)),
        ]

    def observe(self, ts, c, d, k, ip, m, success: int) -> None:
        """Add this transaction to the graph (call AFTER compute_features)."""
        for key in self._links(c, d, k, ip, m):
            if key not in self.first_seen:
                self.first_seen[key] = ts
                self.customer_new_links[c].append(ts)
        self.device[d].observe(ts, c, success)
        self.card[k].observe(ts, c, success)
        self.ip[ip].observe(ts, c, success)
        self.device_cards[d].add(k)
        self.device_ips[d].add(ip)
        self.customer_devices[c].add(d)
        self.customer_cards[c].add(k)
        cm = self.customer_merchants[c]
        cm[m] = cm.get(m, 0) + 1
        if cm[m] > self.customer_merchant_max[c]:
            self.customer_merchant_max[c] = cm[m]
        self.customer_n[c] += 1


def build_graph_features(transactions: pd.DataFrame) -> pd.DataFrame:
    """Append GRAPH_FEATURE_COLS; each row uses only strictly earlier transactions."""
    tx = sort_chronologically(transactions)
    tracker = TemporalGraphTracker()
    cols = ["timestamp", "customer_id", "device_id", "card_id", "ip_id", "merchant_id", "success"]
    feats: List[list] = []
    for ts, c, d, k, ip, m, s in tx[cols].itertuples(index=False, name=None):
        feats.append(tracker.compute_features(ts, c, d, k, ip, m))
        tracker.observe(ts, c, d, k, ip, m, int(s))
    feat_df = pd.DataFrame(feats, columns=GRAPH_FEATURE_COLS)
    return pd.concat([tx.drop(columns=GRAPH_FEATURE_COLS, errors="ignore"), feat_df], axis=1)


def get_ablation_feature_sets() -> Dict[str, List[str]]:
    return {
        "transaction_only": [],
        "transaction_plus_counts": SIMPLE_COUNT_COLS,
        "transaction_plus_temporal": SIMPLE_COUNT_COLS + TEMPORAL_EXTRA_COLS,
        "transaction_plus_full_graph": GRAPH_FEATURE_COLS,
    }
