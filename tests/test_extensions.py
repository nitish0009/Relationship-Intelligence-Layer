"""Tests for C1-C4 components: delayed labels, ring metrics, temporal GNNs, generator knobs."""

import os
import sys

import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

import src  # noqa: F401,E402
from src.data_generator import AdversaryConfig, GeneratorConfig, generate_all  # noqa: E402
from src.label_features import LABEL_FEATURE_COLS, build_label_features  # noqa: E402
from src.preprocessing import build_transaction_features, sort_chronologically  # noqa: E402
from src.ring_metrics import add_ring_sequence, ring_detection_table, summarise_rings, warmup_curve  # noqa: E402

SMALL = GeneratorConfig(n_customers=2_000, n_transactions=8_000, n_merchants=200)


@pytest.fixture(scope="module")
def small_tx():
    return sort_chronologically(generate_all(None, seed=11, config=SMALL)["transactions"])


# ------------------------------------------------------------------ C2: delayed labels


def _bruteforce_label(tx: pd.DataFrame, i: int, delay_days: float) -> dict:
    row, past = tx.iloc[i], tx.iloc[:i]
    matured = past[past.timestamp + pd.Timedelta(days=delay_days) <= row.timestamp]
    out = {}
    for ent, col in [("device", "device_id"), ("ip", "ip_id"), ("card", "card_id")]:
        m = matured[matured[col] == row[col]]
        out[f"{ent}_known_fraud_txns"] = int(m.fraud_label.sum())
        out[f"{ent}_known_fraud_rate"] = float(m.fraud_label.mean()) if len(m) else 0.0
    out["customer_known_fraud_txns"] = int(matured[matured.customer_id == row.customer_id].fraud_label.sum())
    flagged = set(matured.loc[matured.fraud_label == 1, "customer_id"])
    for ent, col in [("device", "device_id"), ("ip", "ip_id")]:
        linked = set(past.loc[past[col] == row[col], "customer_id"]) - {row.customer_id}
        out[f"{ent}_flagged_customers"] = len(linked & flagged)
    return out


@pytest.mark.parametrize("delay", [0.0, 2.0, 15.0])
def test_label_features_match_bruteforce(small_tx, delay):
    feats = build_label_features(small_tx, delay)
    rng = np.random.default_rng(int(delay))
    fraud_rows = np.flatnonzero(feats.fraud_label.to_numpy() == 1)
    rows = np.concatenate([rng.choice(len(feats), 40, replace=False), rng.choice(fraud_rows, 40, replace=False)])
    for i in rows:
        for col, val in _bruteforce_label(feats, int(i), delay).items():
            assert feats.iloc[i][col] == pytest.approx(val, abs=1e-6), (i, col, delay)


def test_infinite_delay_means_no_label_information(small_tx):
    feats = build_label_features(small_tx, float("inf"))
    assert (feats[LABEL_FEATURE_COLS].to_numpy() == 0).all()


def test_label_features_never_see_own_or_future_labels(small_tx):
    """Flipping a transaction's label must not change features of itself or earlier rows,
    nor of rows before its maturity time."""
    d = 3.0
    base = build_label_features(small_tx, d)
    i = int(np.flatnonzero(base.fraud_label.to_numpy() == 1)[50])
    flipped = small_tx.copy()
    flipped.loc[flipped.transaction_id == base.transaction_id.iloc[i], "fraud_label"] = 0
    alt = build_label_features(flipped, d)
    mature_at = base.timestamp.iloc[i] + pd.Timedelta(days=d)
    unaffected = (base.timestamp < mature_at).to_numpy()
    pd.testing.assert_frame_equal(base.loc[unaffected, LABEL_FEATURE_COLS], alt.loc[unaffected, LABEL_FEATURE_COLS])


# ------------------------------------------------------------------ C1: ring metrics


def _toy_window():
    ts = pd.date_range("2024-01-01", periods=6, freq="h")
    return pd.DataFrame({
        "transaction_id": [f"T{i}" for i in range(6)], "timestamp": ts,
        "ring_id": [1, 1, 1, 1, 2, 2], "ring_seq": [0, 1, 2, 3, 3, 4],  # ring 2 started before the window
        "amount": [100.0, 200.0, 300.0, 400.0, 50.0, 50.0], "fraud_label": 1, "fraud_pattern": "x",
    })


def test_ring_detection_table_on_toy_example():
    df = _toy_window()
    scores = np.array([0.1, 0.2, 0.9, 0.1, 0.9, 0.9])
    tab = ring_detection_table(df, scores, 0.5)
    assert list(tab.ring_id) == [1]  # ring 2 excluded: no clean clock
    r = tab.iloc[0]
    assert r.txns_before_alert == 2 and r.amount_before_alert == 300.0 and r.hours_to_alert == 2.0
    assert r.prevented_takedown == pytest.approx(0.7) and r.prevented_blocking == pytest.approx(0.3)
    s = summarise_rings(tab)
    assert s["ring_recall"] == 1.0 and s["amount_lost_share"] == pytest.approx(0.3)
    never = ring_detection_table(df, np.zeros(6), 0.5).iloc[0]
    assert not never.detected and never.amount_before_alert == 1000.0 and never.prevented_takedown == 0.0


def test_warmup_curve_and_ring_sequence():
    df = _toy_window()
    w = warmup_curve(df, np.array([0.1, 0.9, 0.9, 0.9, 0.0, 0.0]), 0.5)
    assert w.set_index("k")["recall"].to_dict() == {1: 0.0, 2: 1.0, 3: 1.0, 4: 1.0}
    tx = pd.DataFrame({"transaction_id": ["a", "b", "c", "d"], "timestamp": pd.date_range("2024", periods=4, freq="D"),
                       "ring_id": [5, -1, 5, 6], "fraud_label": [1, 0, 1, 1]})
    assert add_ring_sequence(tx)["ring_seq"].tolist() == [0, -1, 1, 0]


# ------------------------------------------------------------------ stronger baselines


def test_temporal_index_is_strictly_past_and_most_recent(small_tx):
    from src.temporal_gnn import RELATIONS, TemporalIndex

    idx = TemporalIndex(small_tx, k=5)
    rng = np.random.default_rng(0)
    for i in rng.choice(len(small_tx), 200, replace=False):
        for r, col in enumerate(RELATIONS):
            nb = idx.nbr[i, r][idx.nbr[i, r] >= 0]
            same = np.flatnonzero(small_tx[col].to_numpy()[:i] == small_tx[col].iloc[i])
            assert sorted(nb.tolist()) == same[-5:].tolist()


def test_tgn_memory_has_no_future_leakage(small_tx):
    from src.temporal_gnn import TemporalIndex, TGNMemoryModel

    tx = build_transaction_features(small_tx)
    tx["_row"] = np.arange(len(tx))
    train = tx.iloc[:5000]
    model = TGNMemoryModel(TemporalIndex(tx), epochs=1).fit(train, train.fraud_label.to_numpy(), full_df=tx)
    p_short = model._stream(model.Xt_all, 6000)
    p_full = model._stream(model.Xt_all, len(tx))
    np.testing.assert_allclose(p_short, p_full[:6000], rtol=1e-5, atol=1e-6)


def test_threshold_at_fpr_matches_budget():
    from src.pipeline import threshold_at_fpr

    rng = np.random.default_rng(0)
    y = (rng.random(20_000) < 0.05).astype(int)
    p = rng.random(20_000)
    thr = threshold_at_fpr(y, p, 0.01)
    assert (p[y == 0] > thr).mean() == pytest.approx(0.01, abs=0.001)


# ------------------------------------------------------------------ generator knobs (C2-C4)


def _fraud(cfg, seed=3):
    d = generate_all(None, seed=seed, config=cfg)
    return d["transactions"], d["rings"], d["ips"]


def test_infra_reuse_dial():
    base = dict(n_customers=3_000, n_transactions=15_000, n_merchants=200)
    tx0, _, _ = _fraud(GeneratorConfig(**base, infra_reuse=0.0))
    tx1, _, _ = _fraud(GeneratorConfig(**base, infra_reuse=1.0))

    def shared_device_share(tx):
        f = tx[tx.fraud_label == 1]
        rings_per_dev = f.groupby("device_id").ring_id.nunique()
        return float(f.device_id.map(rings_per_dev).gt(1).mean())

    assert shared_device_share(tx0) < 0.1  # fresh rings: almost no device crosses rings
    assert shared_device_share(tx1) > 0.5  # fixed pool: most fraud txns use cross-ring devices


def test_ring_lifetime_knob():
    _, rings, _ = _fraud(GeneratorConfig(n_customers=2_000, n_transactions=10_000, ring_duration_days=(30, 60)))
    full = rings[(rings.planned_start > pd.Timestamp("2024-03-01")) & (rings.planned_end < pd.Timestamp("2024-09-30"))]
    assert full.lifetime_days.between(30, 60).all()


def test_adversary_rotation_and_hiding():
    base = dict(n_customers=3_000, n_transactions=15_000, n_merchants=200)
    adv = AdversaryConfig(device_max_uses=2, carrier_ip_share=1.0, adapt_from=0.0)
    tx, rings, ips = _fraud(GeneratorConfig(**base, adversary=adv))
    f = tx[tx.fraud_label == 1]
    legit_devices = set(tx.loc[tx.fraud_label == 0, "device_id"])
    ring_dev_uses = f[~f.device_id.isin(legit_devices)].groupby(["ring_id", "device_id"]).size()
    assert ring_dev_uses.max() <= 2
    carrier = set(ips.loc[ips.kind == "carrier", "ip_id"])
    assert f.ip_id.isin(carrier).mean() > 0.99
    assert rings.adapted.all()


def test_slow_factor_stretches_rings():
    base = dict(n_customers=2_000, n_transactions=10_000)
    _, r1, _ = _fraud(GeneratorConfig(**base))
    _, r4, _ = _fraud(GeneratorConfig(**base, adversary=AdversaryConfig(slow_factor=4.0)))
    assert r4.lifetime_days.mean() > 2.5 * r1.lifetime_days.mean()


# ------------------------------------------------------------------ IEEE-CIS mapping


def test_ieee_mapping_and_proxy_rings():
    from experiments.ieee_cis import map_ieee

    trans = pd.DataFrame({
        "TransactionID": [1, 2, 3, 4, 5], "isFraud": [1, 1, 0, 1, 1],
        "TransactionDT": [86400, 90000, 100000, 200000, 300000], "TransactionAmt": [10.0, 20.0, 30.0, 40.0, 50.0],
        "ProductCD": ["W", "W", "C", "C", "W"], "card1": [111, 111, 222, 333, 444], "addr1": [1, 1, 2, 3, 4],
        "D1": [0, 0, 5, 0, 0],
    })
    ident = pd.DataFrame({"TransactionID": [4, 5], "DeviceInfo": ["phoneA", "phoneA"], "id_30": ["os", "os"],
                          "id_31": ["br", "br"], "id_33": ["1x1", "1x1"]})
    tx = map_ieee(trans, ident)
    assert tx.customer_id.iloc[0] == tx.customer_id.iloc[1]  # same uid (card1, addr1, account start)
    assert tx.device_id.iloc[0] != tx.device_id.iloc[1]  # missing device -> unique token, no fake sharing
    assert tx.device_id.iloc[3] == tx.device_id.iloc[4]
    rings = tx.ring_id.tolist()
    assert rings[2] == -1  # legit
    assert rings[0] == rings[1] and rings[3] == rings[4] and rings[0] != rings[3]
    assert (tx.timestamp.iloc[0] == pd.Timestamp("2017-12-02"))
