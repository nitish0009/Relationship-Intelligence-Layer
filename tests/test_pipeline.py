"""Tests for leakage safety, reproducibility and the synthetic-data methodology."""

import hashlib
import os
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from src.data_generator import GeneratorConfig, generate_all  # noqa: E402
from src.graph_features import GRAPH_FEATURE_COLS, SlidingDistinct, build_graph_features  # noqa: E402
from src.preprocessing import (  # noqa: E402
    TRANSACTION_FEATURE_COLS,
    build_transaction_features,
    chronological_split,
)

SMALL = GeneratorConfig(n_customers=2_000, n_transactions=8_000, n_merchants=200)


@pytest.fixture(scope="module")
def small_tx():
    return generate_all(None, seed=7, config=SMALL)["transactions"]


@pytest.fixture(scope="module")
def small_features(small_tx):
    return build_graph_features(build_transaction_features(small_tx))


# ------------------------------------------------------------------ splitting


def test_chronological_split_no_overlap():
    df = pd.DataFrame(
        {
            "transaction_id": [f"T{i:03d}" for i in range(100)],
            "timestamp": pd.date_range("2024-01-01", periods=100, freq="h"),
            "fraud_label": 0,
        }
    )
    train, val, test = chronological_split(df)
    assert (len(train), len(val), len(test)) == (70, 15, 15)
    assert train["timestamp"].max() < val["timestamp"].min()
    assert val["timestamp"].max() < test["timestamp"].min()


# ------------------------------------------------------------------ leakage


def test_transaction_history_is_per_entity_and_exclusive():
    """Regression test: cumulative stats must not bleed across entities or include the row."""
    df = pd.DataFrame(
        {
            "transaction_id": [f"T{i}" for i in range(6)],
            "timestamp": pd.date_range("2024-01-01", periods=6, freq="h"),
            "customer_id": ["A", "B", "A", "B", "A", "B"],
            "device_id": ["d"] * 6,
            "card_id": ["k"] * 6,
            "merchant_id": ["m"] * 6,
            "amount": [100.0, 999.0, 300.0, 999.0, 500.0, 999.0],
            "success": [0, 1, 1, 1, 1, 1],
            "payment_method": ["upi"] * 6,
        }
    )
    out = build_transaction_features(df)
    a = out[out.customer_id == "A"]
    assert a["customer_transaction_count"].tolist() == [0, 1, 2]
    assert a["customer_success_rate"].tolist() == [1.0, 0.0, 0.5]  # prior, then A's own history
    assert a["recent_failure_rate"].tolist() == [0.0, 1.0, 0.5]
    assert a["customer_avg_amount"].tolist() == [0.0, 100.0, 200.0]
    b = out[out.customer_id == "B"]
    assert b["customer_success_rate"].tolist() == [1.0, 1.0, 1.0]  # A's failure must not leak into B


def test_features_do_not_depend_on_future_rows(small_tx, small_features):
    """Truncating the future must leave every earlier feature value unchanged."""
    k = 5_000
    head = small_tx.sort_values(["timestamp", "transaction_id"]).iloc[:k]
    trunc = build_graph_features(build_transaction_features(head))
    cols = [c for c in TRANSACTION_FEATURE_COLS + GRAPH_FEATURE_COLS if c != "payment_method"]
    pd.testing.assert_frame_equal(
        small_features.iloc[:k][cols].reset_index(drop=True), trunc[cols].reset_index(drop=True)
    )


def _bruteforce_graph_features(tx: pd.DataFrame, i: int) -> dict:
    """Reference implementation straight from the feature definitions (O(N) per row)."""
    row = tx.iloc[i]
    past = tx.iloc[:i]
    ts, c, d, k, ip, m = row[["timestamp", "customer_id", "device_id", "card_id", "ip_id", "merchant_id"]]
    cutoff = ts - pd.Timedelta(days=7)
    on_d, on_k, on_ip, of_c = past[past.device_id == d], past[past.card_id == k], past[past.ip_id == ip], past[past.customer_id == c]

    def fail(x):
        return float(1 - x.success.mean()) if len(x) else 0.0

    dev_c, ip_c = set(on_d.customer_id), set(on_ip.customer_id)
    nodes = {("c", c), ("d", d), ("k", k), ("i", ip)}
    nodes |= {("d", x) for x in of_c.device_id} | {("k", x) for x in of_c.card_id}
    nodes |= {("c", x) for x in dev_c | ip_c}
    counts = on_d.customer_id.value_counts().to_numpy(dtype=float)
    entropy = float(-(counts / counts.sum() * np.log2(counts / counts.sum())).sum()) if len(counts) else 0.0

    first = {}
    for col, typ in [("device_id", "device"), ("card_id", "card"), ("ip_id", "ip"), ("merchant_id", "merchant")]:
        firsts = of_c.groupby(col).timestamp.min()
        first[typ] = firsts
    links = {"device": d, "card": k, "ip": ip, "merchant": m}
    new7 = sum(1 for t, v in links.items() if v not in first[t].index or first[t][v] >= cutoff)
    all_first = pd.concat(list(first.values()))
    n_unseen = sum(1 for t, v in links.items() if v not in first[t].index)
    velocity = ((all_first >= cutoff).sum() + n_unseen) / 7
    ages = [0.0 if v not in first[t].index else (ts - first[t][v]).total_seconds() / 86400
            for t, v in links.items() if t != "merchant"]
    net = pd.concat([on_d, on_k, on_ip])
    return {
        "device_unique_customers": len(dev_c),
        "device_unique_cards": on_d.card_id.nunique(),
        "device_unique_ips": on_d.ip_id.nunique(),
        "customer_unique_devices": of_c.device_id.nunique(),
        "customer_unique_cards": of_c.card_id.nunique(),
        "ip_unique_customers": len(ip_c),
        "card_unique_customers": on_k.customer_id.nunique(),
        "device_failure_rate": fail(on_d),
        "card_failure_rate": fail(on_k),
        "ip_failure_rate": fail(on_ip),
        "recent_device_accounts": on_d[on_d.timestamp >= cutoff].customer_id.nunique(),
        "recent_card_accounts": on_k[on_k.timestamp >= cutoff].customer_id.nunique(),
        "recent_ip_accounts": on_ip[on_ip.timestamp >= cutoff].customer_id.nunique(),
        "new_relationship_count_7d": new7,
        "relationship_velocity": velocity,
        "shared_entity_count": len((dev_c & ip_c) - {c}),
        "network_size": len(nodes),
        "network_failure_rate": fail(net),
        "device_account_entropy": entropy,
        "merchant_concentration": float(of_c.merchant_id.value_counts().max() / len(of_c)) if len(of_c) else 0.0,
        "relationship_recency": float(np.mean(ages)),
    }


def test_graph_features_match_bruteforce_reference(small_features):
    tx = small_features
    rng = np.random.default_rng(0)
    fraud_rows = np.flatnonzero(tx.fraud_label.to_numpy() == 1)
    rows = np.concatenate([rng.choice(len(tx), 60, replace=False), rng.choice(fraud_rows, 40, replace=False)])
    for i in rows:
        expected = _bruteforce_graph_features(tx, int(i))
        for col, val in expected.items():
            assert tx.iloc[i][col] == pytest.approx(val, abs=1e-9), (i, col)


def test_sliding_distinct_window():
    s = SlidingDistinct()
    t0 = pd.Timestamp("2024-01-01")
    for day, key in [(0, "a"), (1, "b"), (2, "a"), (9, "c")]:
        s.expire(t0 + pd.Timedelta(days=day) - pd.Timedelta(days=7))
        s.add(t0 + pd.Timedelta(days=day), key)
    s.expire(t0 + pd.Timedelta(days=9) - pd.Timedelta(days=7))
    assert sorted(s.counts) == ["a", "c"]  # 'b' (day 1) expired, 'a' kept via day 2


def test_graphsage_refuses_future_history(small_features):
    from src.graph_model import GraphSAGEModel

    train, val, _ = chronological_split(small_features)
    model = GraphSAGEModel(epochs=2).fit(train, train.fraud_label.to_numpy())
    model.predict_proba(val, history=train)  # fine
    with pytest.raises(ValueError):
        model.predict_proba(train, history=val)


# ------------------------------------------------------------------ reproducibility


def _generate_hash(hash_seed: str) -> str:
    code = (
        "import sys, hashlib; sys.path.insert(0, %r);"
        "from src.data_generator import generate_all, GeneratorConfig;"
        "tx = generate_all(None, seed=42, config=GeneratorConfig(n_customers=1500, n_transactions=5000, n_merchants=150))['transactions'];"
        "print(hashlib.sha256(tx.to_csv(index=False).encode()).hexdigest())" % PROJECT_ROOT
    )
    env = {**os.environ, "PYTHONHASHSEED": hash_seed}
    out = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True, check=True)
    return out.stdout.strip().splitlines()[-1]


def test_generator_reproducible_across_processes_and_hash_seeds():
    assert _generate_hash("1") == _generate_hash("2")


def test_different_seeds_give_different_data():
    a = generate_all(None, seed=1, config=SMALL)["transactions"]
    b = generate_all(None, seed=2, config=SMALL)["transactions"]
    assert hashlib.sha256(a.to_csv().encode()).digest() != hashlib.sha256(b.to_csv().encode()).digest()


# ------------------------------------------------------------------ synthetic-data methodology


def test_schema(small_tx):
    for col in ["transaction_id", "timestamp", "customer_id", "device_id", "card_id", "ip_id", "merchant_id",
                "amount", "payment_method", "success", "fraud_label", "fraud_pattern", "ring_id",
                "graph_hidden_fraud", "legitimate_shared_env"]:
        assert col in small_tx.columns
    assert small_tx.transaction_id.is_unique
    assert small_tx.fraud_label.mean() == pytest.approx(0.035, abs=0.002)


def test_fraud_rings_rotate_so_test_rings_are_unseen():
    tx = generate_all(None, seed=42, config=GeneratorConfig(n_customers=5_000, n_transactions=25_000, n_merchants=300))["transactions"]
    train, _, test = chronological_split(tx)
    tr_f, te_f = train[train.fraud_label == 1], test[test.fraud_label == 1]
    assert set(te_f.ring_id).isdisjoint(set(tr_f.ring_id))
    # devices that a test-period ring uses repeatedly were never seen during training
    ring_devices = te_f.device_id.value_counts()
    assert set(ring_devices[ring_devices >= 3].index).isdisjoint(set(train.device_id))


def test_graph_hidden_fraud_has_legit_intrinsics():
    tx = generate_all(None, seed=3, config=GeneratorConfig(n_customers=10_000, n_transactions=60_000, n_merchants=500))["transactions"]
    hidden = tx[tx.graph_hidden_fraud == 1]
    legit = tx[tx.fraud_label == 0]
    for q in [0.1, 0.25, 0.5, 0.75, 0.9]:
        assert hidden.amount.quantile(q) == pytest.approx(legit.amount.quantile(q), rel=0.12)
    assert hidden.success.mean() == pytest.approx(legit.success.mean(), abs=0.03)
    pm_h = hidden.payment_method.value_counts(normalize=True)
    pm_l = legit.payment_method.value_counts(normalize=True)
    assert (pm_h - pm_l).abs().max() < 0.05
    hours_h = hidden.timestamp.dt.hour.value_counts(normalize=True).sort_index()
    hours_l = legit.timestamp.dt.hour.value_counts(normalize=True).sort_index()
    assert (hours_h - hours_l).abs().max() < 0.03


def test_new_users_also_transact_legitimately():
    data = generate_all(None, seed=5, config=SMALL)
    cust, tx = data["customers"], data["transactions"]
    new_users = set(cust.loc[cust.segment == "new_user", "customer_id"])
    legit_new = tx[(tx.fraud_label == 0) & tx.customer_id.isin(new_users)]
    assert len(legit_new) > 0


def test_threshold_selected_on_given_data_only():
    from src.evaluation import select_threshold

    y = np.array([0, 0, 0, 1, 1])
    p = np.array([0.1, 0.2, 0.3, 0.8, 0.9])
    assert select_threshold(y, p) == pytest.approx(0.8)
