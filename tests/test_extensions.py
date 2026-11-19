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


# ------------------------------------------------------------------ audit fixes (IEEE first-fraud, AMLworld)


def test_first_fraud_mask_marks_only_each_clients_first_fraud():
    from experiments.ieee_cis import first_fraud_mask

    tx = pd.DataFrame({"transaction_id": list("abcdef"), "timestamp": pd.date_range("2024", periods=6, freq="h"),
                       "customer_id": ["u1", "u1", "u2", "u1", "u2", "u3"], "fraud_label": [0, 1, 1, 1, 1, 0]})
    assert first_fraud_mask(tx).tolist() == [False, True, True, False, False, False]


def test_parse_aml_patterns(tmp_path):
    from experiments.amlworld import parse_patterns

    p = tmp_path / "pat.txt"
    p.write_text("BEGIN LAUNDERING ATTEMPT - FAN-OUT:  Max 2-degree Fan-Out\n"
                 "2022/09/01 00:06,021,800A,012,800B,10.00,Euro,10.00,Euro,ACH,1\n"
                 "2022/09/01 04:33,021,800A,020,800C,20.00,Euro,20.00,Euro,ACH,1\n"
                 "END LAUNDERING ATTEMPT - FAN-OUT\n\n"
                 "BEGIN LAUNDERING ATTEMPT - CYCLE:  Max 2 hops\n"
                 "2022/09/02 01:00,001,800X,002,800Y,5.00,US Dollar,5.00,US Dollar,Wire,1\n"
                 "END LAUNDERING ATTEMPT - CYCLE\n")
    df = parse_patterns(str(p))
    assert df["attempt"].tolist() == [0, 0, 1] and df["typology"].tolist() == ["fan-out", "fan-out", "cycle"]
    assert df["From Bank"].iloc[0] == "021"  # leading zeros preserved for matching


def test_pyg_reference_tgn_has_no_future_leakage(small_tx):
    pytest.importorskip("torch_geometric")
    import torch

    from src.temporal_gnn import TemporalIndex
    from src.tgn_pyg import TGNReference

    tx = build_transaction_features(small_tx)
    tx["_row"] = np.arange(len(tx))
    train = tx.iloc[:5000]
    m = TGNReference(TemporalIndex(tx), epochs=1).fit(train, train.fraud_label.to_numpy(), full_df=tx)
    m._reset_events(6000)
    with torch.no_grad():
        a = m._stream(6000)
    m._reset_events(len(tx))
    with torch.no_grad():
        b = m._stream(len(tx))
    np.testing.assert_allclose(a, b[:6000], atol=1e-6)


# ---------------------------------------------------------------- C5: takedown-value weighting

def _toy_tvw():
    import pandas as pd
    t = pd.to_datetime("2024-01-01") + pd.to_timedelta([0, 1, 2, 3, 4, 5], unit="h")
    return pd.DataFrame({
        "timestamp": t, "amount": [10.0, 20.0, 5.0, 30.0, 1.0, 7.0],
        "fraud_label": [1, 1, 0, 1, 1, 0],
        "customer_id": ["a", "b", "c", "d", "e", "f"],
        "device_id": ["D1", "D1", "D9", "D2", "D2", "D9"],   # ring {0,1} via D1, ring {3,4} via D2
        "card_id": ["K1", "K2", "K3", "K4", "K5", "K6"], "ip_id": ["NAT"] * 6,  # NAT shared by 6 customers
        "ring_id": [7, 7, -1, 8, 8, -1],
    })


def test_pseudo_rings_link_specific_entities_only():
    from src.clock_weights import pseudo_rings
    df = _toy_tvw()
    r = pseudo_rings(df, max_customers=3)
    assert r[2] == -1 and r[5] == -1
    assert r[0] == r[1] and r[3] == r[4] and r[0] != r[3]  # the shared NAT IP must not merge rings
    r_loose = pseudo_rings(df, max_customers=10)
    assert r_loose[0] == r_loose[3]  # a coarse key would merge them -- hence the specificity filter


def test_remaining_value_and_weights():
    import numpy as np
    from src.clock_weights import pseudo_rings, remaining_value, tvw_weights
    df = _toy_tvw()
    R = remaining_value(df, pseudo_rings(df, max_customers=3))
    assert np.allclose(R, [30, 20, 0, 31, 1, 0])
    assert np.allclose(tvw_weights(df, 0.0), 1.0)
    w = tvw_weights(df, 1.0, max_customers=3)
    f = df.fraud_label.to_numpy() == 1
    assert np.isclose(w[f].mean(), 1.0) and np.allclose(w[~f], 1.0)  # class balance unchanged
    assert w[0] > w[1] and w[3] > w[4]  # earlier ring transactions weigh more


def test_tvw_uses_training_rows_only():
    """Weights depend only on the rows passed in: appending future rows cannot change them."""
    import numpy as np
    import pandas as pd
    from src.clock_weights import tvw_weights
    df = _toy_tvw()
    w_a = tvw_weights(df.iloc[:4], 1.0, max_customers=3)
    w_b = tvw_weights(pd.concat([df.iloc[:4], df.iloc[4:]]).iloc[:4], 1.0, max_customers=3)
    assert np.allclose(w_a, w_b)


def test_tvw_ess_tempering():
    import numpy as np
    from src.clock_weights import _ess_share, temper
    rel = np.r_[np.full(99, 0.01), 1000.0]  # one huge transaction
    assert _ess_share(rel) < 0.05
    t, g = temper(rel, 0.5)
    assert _ess_share(t) >= 0.5 - 1e-6 and 0 <= g < 1
    flat = np.linspace(0.5, 1.5, 100)  # already well spread -> untouched
    t2, g2 = temper(flat, 0.5)
    assert g2 == 1.0 and np.allclose(t2, flat)


def test_gnn_neighbour_standardiser_uses_whole_training_period():
    """Regression: the neighbour standardiser must not be fitted on only the earliest rows
    (on AMLworld those covered 6 minutes, giving |z| ~ 1e4 at test time)."""
    import numpy as np
    from src.pipeline import build_dataset
    from src.data_generator import GeneratorConfig
    from src.temporal_gnn import TemporalGNN, TemporalIndex
    ds = build_dataset(7, GeneratorConfig(n_customers=3000, n_transactions=25000), label_delay_days=30.0)
    idx = TemporalIndex(ds.tx, k=5)
    m = TemporalGNN(idx, "sage", epochs=1, seed=0)
    m.fit(ds.train, ds.train.fraud_label.to_numpy())
    xn, v = idx.neighbour_features(ds.test["_row"].to_numpy())
    z = np.abs(m.std_n(xn)[v])
    assert np.isfinite(z).all() and z.max() < 1e3


# ---------------------------------------------------------------- ring-survival loss (exact takedown value)

def test_takedown_share_matches_brute_force():
    import itertools
    import numpy as np
    import src  # noqa: F401  (loads XGBoost's OpenMP before torch; see src/__init__.py)
    import torch
    from src.ring_survival import RingIndex, takedown_share
    rng = np.random.default_rng(0)
    ring = np.array([0, 1, 0, -1, 1, 0, 2])
    time = np.array([1, 2, 3, 4, 5, 6, 7])
    amount = rng.uniform(1, 10, len(ring))
    ix = RingIndex(ring, time, amount)
    z = torch.tensor(rng.normal(size=len(ix.rows)), dtype=torch.float64)
    v = takedown_share(z, torch.from_numpy(ix.seg), torch.from_numpy(ix.remaining_share), ix.n_rings).numpy()
    p = torch.sigmoid(z).numpy()
    for r in range(ix.n_rings):
        k = np.flatnonzero(ix.seg == r)
        a = amount[ix.rows[k]]
        exp_saved = 0.0  # enumerate every alert pattern; ring stopped at the first alert
        for alerts in itertools.product([0, 1], repeat=len(k)):
            pr = np.prod([p[k[i]] if alerts[i] else 1 - p[k[i]] for i in range(len(k))])
            if any(alerts):
                exp_saved += pr * a[alerts.index(1):].sum()
        assert np.isclose(v[r], exp_saved / a.sum())


def test_ring_survival_discounts_late_transactions():
    """Unlike TVW, once the first transaction is likely alerted, the gradient on later ones vanishes."""
    import numpy as np
    import src  # noqa: F401  (loads XGBoost's OpenMP before torch; see src/__init__.py)
    import torch
    from src.ring_survival import RingIndex, ring_survival_loss
    ix = RingIndex(np.array([0, 0, 0]), np.array([1, 2, 3]), np.array([1.0, 1.0, 1.0]))
    seg, rem, w = torch.from_numpy(ix.seg), torch.from_numpy(ix.remaining_share), torch.from_numpy(ix.ring_weight)
    for first, expect_small in [(-4.0, False), (6.0, True)]:
        z = torch.tensor([first, 0.0, 0.0], dtype=torch.float64, requires_grad=True)
        ring_survival_loss(z, seg, rem, w).backward()
        late = abs(float(z.grad[2]))
        assert (late < 1e-2) == expect_small


def test_tvw_v2_equals_v1_without_tempering_and_keeps_clock_with_it():
    import numpy as np
    import pandas as pd
    from src.clock_weights import remaining_value, tvw_weights
    t = pd.to_datetime("2024-01-01") + pd.to_timedelta(np.arange(8), unit="h")
    base = dict(timestamp=t, fraud_label=[1, 1, 1, 0, 1, 1, 1, 0],
                customer_id=list("aaabcccd"), device_id=list("xyzuvwqr"), card_id=list("12345678"), ip_id=list("ABCDEFGH"))
    # comparable amounts: ESS high, gamma = 1 -> v2 == v1
    df = pd.DataFrame({**base, "amount": [3.0, 2.0, 1.0, 5.0, 3.0, 2.0, 1.0, 5.0]})
    assert np.allclose(tvw_weights(df, 1.0, tempering="ring"), tvw_weights(df, 1.0, tempering="global"))
    # one ring is 10^6 x larger: tempering needed; v2 must keep each ring's within-ring ratio R_1/R_3 = 6
    df2 = pd.DataFrame({**base, "amount": [3e6, 2e6, 1e6, 5.0, 3.0, 2.0, 1.0, 5.0]})
    w = tvw_weights(df2, 1.0, tempering="ring")
    assert np.isclose(w[0] / w[2], 6.0) and np.isclose(w[4] / w[6], 6.0)
    wg = tvw_weights(df2, 1.0, tempering="global")
    assert wg[4] / wg[6] < 6.0 - 1e-6  # v1 flattens the clock inside rings


def test_identity_columns_always_link():
    import numpy as np
    import pandas as pd
    from src.clock_weights import pseudo_rings
    # receiving account "R" is shared by 12 distinct senders (a fan-in pattern)
    n = 12
    df = pd.DataFrame({"timestamp": pd.date_range("2024-01-01", periods=n, freq="h"), "amount": 1.0,
                       "fraud_label": 1, "customer_id": [f"S{i}" for i in range(n)], "device_id": "R",
                       "card_id": [f"c{i}" for i in range(n)], "ip_id": [f"i{i}" for i in range(n)]})
    assert len(set(pseudo_rings(df))) == n                                   # infrastructure rule splits it
    assert len(set(pseudo_rings(df, identity_cols=("customer_id", "device_id")))) == 1  # identity rule keeps it whole


def test_identity_key_links_accounts_across_roles():
    import pandas as pd
    from src.clock_weights import pseudo_rings
    # chain A -> B -> C: B is receiver of transfer 1 and sender of transfer 2
    df = pd.DataFrame({"timestamp": pd.date_range("2024-01-01", periods=2, freq="h"), "amount": 1.0, "fraud_label": 1,
                       "customer_id": ["SA", "SB"], "device_id": ["RB", "RC"], "card_id": ["k1", "k2"], "ip_id": ["i1", "i2"]})
    ids = ("customer_id", "device_id")
    assert len(set(pseudo_rings(df, identity_cols=ids))) == 2
    assert len(set(pseudo_rings(df, identity_cols=ids, identity_key=lambda v: v[1:]))) == 1
