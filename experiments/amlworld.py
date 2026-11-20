"""Fix 7: second real-data replication on IBM AMLworld (Altman et al., NeurIPS 2023 D&B).

Data: HI-Small_Trans.csv + HI-Small_Patterns.txt from
https://www.kaggle.com/datasets/ealtman2019/ibm-transactions-for-anti-money-laundering-aml
into data/AMLworld/ (git-ignored).

Why AMLworld: it has GROUND-TRUTH laundering patterns (370 attempts, 8 typologies: fan-in,
fan-out, cycle, bipartite, stack, scatter-gather, gather-scatter, random). These act as real
"rings", so C1 (ring timing) and C2 (label delay vs ring lifetime) can be tested without proxies.

Scope: first 10 days only (2022-09-01..10). Days 11-18 contain almost only laundering
(~60% prevalence vs 0.1%) -- a known artefact of pattern completion in the generator.

Schema mapping:
  customer_id    sender account (bank|account)
  device_id      receiver account (bank|account)   -> fan-in = many senders per receiver
  card_id        sender bank | payment format       (sending channel)
  ip_id          receiver bank
  merchant_id    receiving currency
  amount         Amount Paid converted to USD with fixed Sep-2022 reference rates
  payment_method Payment Format; success = 1 (not available)
  ring_id        laundering attempt from the patterns file; laundering transactions not listed in
                 any pattern get singleton ring ids

Outputs: results/amlworld/
"""

from __future__ import annotations

import gc
import os
import time

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

from experiments.common import BUDGETS, PRIMARY_BUDGET, Timer, make_model, out_dir, ring_rows, save_json, score_all
from src.evaluation import bootstrap_ci, cluster_bootstrap_indices
from src.graph_features import build_graph_features
from src.label_features import LABEL_FEATURE_COLS, build_label_features
from src.pipeline import FEATURE_FAMILIES, Dataset, _split, fit_score, tabular, tune_xgb
from src.preprocessing import build_transaction_features, set_payment_methods
from src.ring_metrics import add_ring_sequence
from src.temporal_gnn import TemporalGNN, TemporalIndex, NeighborAggXGB

AML_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "AMLworld")
END = pd.Timestamp("2022-09-11")
USD = {"US Dollar": 1.0, "Euro": 1.0, "Swiss Franc": 1.03, "Yuan": 0.145, "Shekel": 0.29, "Rupee": 0.0125,
       "UK Pound": 1.15, "Yen": 0.007, "Ruble": 0.0165, "Bitcoin": 20000.0, "Canadian Dollar": 0.76,
       "Australian Dollar": 0.68, "Mexican Peso": 0.05, "Saudi Riyal": 0.266, "Brazil Real": 0.19}
COLS = ["Timestamp", "From Bank", "From Account", "To Bank", "To Account", "Amount Received", "Receiving Currency",
        "Amount Paid", "Payment Currency", "Payment Format", "Is Laundering"]
KEY = ["Timestamp", "From Bank", "From Account", "To Bank", "To Account", "Amount Paid", "Payment Currency"]
DELAYS = [0, 0.25, 0.5, 1, 2, 4, float("inf")]


# AMLworld schema for TVW pseudo-rings: sender ('S'+bank|acct) and receiver ('R'+bank|acct) are both ACCOUNT
# identities; strip the role prefix so an account links across roles (receiver of one transfer, sender of next).
AML_IDENTITY_COLS = ("customer_id", "device_id")


def account_key(v: str) -> str:
    return v[1:]


def parse_patterns(path: str) -> pd.DataFrame:
    """Rows of every laundering attempt with attempt id and typology."""
    rows, attempt, typ = [], -1, ""
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line.startswith("BEGIN LAUNDERING ATTEMPT"):
                attempt += 1
                typ = line.split(" - ", 1)[1].split(":")[0].strip().lower().replace(" ", "_")
            elif line and not line.startswith("END"):
                parts = line.split(",")
                if len(parts) >= 11:
                    rows.append(parts[:11] + [attempt, typ])
    return pd.DataFrame(rows, columns=COLS + ["attempt", "typology"])


def load_aml(prefix: str = "HI-Small") -> pd.DataFrame:
    tpath = os.path.join(AML_DIR, f"{prefix}_Trans.csv")
    if not os.path.exists(tpath):
        raise FileNotFoundError(f"{tpath} not found -- see module docstring")
    raw = pd.read_csv(tpath, dtype=str, header=0, names=COLS)
    pat = parse_patterns(os.path.join(AML_DIR, f"{prefix}_Patterns.txt"))
    raw = raw.merge(pat[KEY + ["attempt", "typology"]].drop_duplicates(KEY), on=KEY, how="left")
    ts = pd.to_datetime(raw["Timestamp"], format="%Y/%m/%d %H:%M")
    keep = (ts < END).to_numpy()
    raw, ts = raw[keep].reset_index(drop=True), ts[keep].reset_index(drop=True)
    y = raw["Is Laundering"].astype(int).to_numpy()
    ring = raw["attempt"].fillna(-1).astype(int).to_numpy().copy()
    lone = (y == 1) & (ring < 0)
    ring[lone] = ring.max() + 1 + np.arange(lone.sum())  # singleton "rings" for unlisted laundering
    ring[y == 0] = -1
    tx = pd.DataFrame({
        "transaction_id": [f"A{i:08d}" for i in range(len(raw))],
        "timestamp": ts,
        "customer_id": "S" + raw["From Bank"] + "|" + raw["From Account"],
        "device_id": "R" + raw["To Bank"] + "|" + raw["To Account"],
        "card_id": "C" + raw["From Bank"] + "|" + raw["Payment Format"],
        "ip_id": "B" + raw["To Bank"],
        "merchant_id": "M" + raw["Receiving Currency"],
        "amount": (raw["Amount Paid"].astype(float) * raw["Payment Currency"].map(USD)).astype(np.float32),
        "payment_method": raw["Payment Format"],
        "success": 1,
        "fraud_label": y,
        "ring_id": ring,
        "fraud_pattern": np.where(y == 1, raw["typology"].fillna("unlisted"), ""),
        "graph_hidden_fraud": 0, "legitimate_shared_env": 0, "segment": "na",
    })
    return tx


def _downcast(df: pd.DataFrame) -> pd.DataFrame:
    for c in df.columns:
        if df[c].dtype == np.float64:
            df[c] = df[c].astype(np.float32)
    return df


def _set_labels(ds: Dataset, delay: float) -> None:
    """Recompute ONLY the delayed-label columns for all splits, in place (memory-friendly)."""
    lab = build_label_features(ds.tx[["transaction_id", "timestamp", "customer_id", "device_id", "card_id", "ip_id",
                                      "fraud_label"]], delay)[LABEL_FEATURE_COLS].to_numpy(np.float32)
    for part in (ds.tx, ds.train, ds.val, ds.test):
        part[LABEL_FEATURE_COLS] = lab[part["_row"].to_numpy()]


GNNS = ["Temporal GraphSAGE", "Temporal GAT"]
ENTITY_SAGE = ["GraphSAGE (entity, inductive)", "GraphSAGE (transductive)"]
GNN_GRID = [{"hidden": 32, "lr": 1e-3}, {"hidden": 64, "lr": 1e-3}, {"hidden": 32, "lr": 3e-3}, {"hidden": 64, "lr": 3e-3}]
LEGIT_SAMPLE = 0.3


def _fit_gnn(name, ds, idx, seed, params):
    """Temporal GNNs on 3.5M training rows: train on all laundering + a LEGIT_SAMPLE share of
    legitimate rows (neighbour features still come from the full graph; PR-AUC is rank-based and
    the alert-budget thresholds are set on the full validation split). Hidden size / learning
    rate are tuned once on a 300k validation subsample (seed 42) and reused for every seed."""
    from src.evaluation import select_threshold
    kind = {"Temporal GraphSAGE": "sage", "Temporal GAT": "gat"}[name]
    vsub = ds.val.sample(min(300_000, len(ds.val)), random_state=0)
    vsub = vsub if vsub.fraud_label.sum() else ds.val

    def fit(p, s):
        rng = np.random.default_rng(s)
        sub = ds.train[(ds.train.fraud_label == 1) | (rng.random(len(ds.train)) < LEGIT_SAMPLE)]
        return TemporalGNN(idx, kind, seed=s, epochs=8, val_df=vsub, **p).fit(sub, sub.fraud_label.to_numpy())

    if name not in params:
        best = (-1.0, GNN_GRID[0])
        for p in GNN_GRID:
            ap = average_precision_score(vsub.fraud_label, fit(p, 42).predict_proba(vsub))
            print(f"     tune {name} {p}: val PR-AUC {ap:.4f}", flush=True)
            best = max(best, (ap, p), key=lambda t: t[0])
        params[name] = best[1]
    m = fit(params[name], seed)
    p_val, p_test = m.predict_proba(ds.val), m.predict_proba(ds.test)
    return {"p_val": p_val, "p_test": p_test, "threshold": select_threshold(ds.val.fraud_label.to_numpy(), p_val)}


def prepare(out: str, prefix: str = "HI-Small"):
    """Load AMLworld, build all features and the chronological split (shared by run / run_gnn_only)."""
    cache = os.path.join(AML_DIR, "cache", f"{prefix}_features.parquet")  # git-ignored (data/)
    if os.path.exists(cache):
        with Timer("load cached features"):
            tx = pd.read_parquet(cache)
        set_payment_methods(sorted(tx["payment_method"].unique()))
    else:
        with Timer("load"):
            tx = load_aml(prefix)
        set_payment_methods(sorted(tx["payment_method"].unique()))
        with Timer("transaction features"):
            tx = _downcast(build_transaction_features(tx))
        with Timer("graph features"):
            tx = _downcast(build_graph_features(tx))
        with Timer("label features"):
            tx = _downcast(build_label_features(tx, 1.0))
        tx = add_ring_sequence(tx)
        os.makedirs(os.path.dirname(cache), exist_ok=True)
        tx.to_parquet(cache)
    tx["_row"] = np.arange(len(tx))
    train, val, test = _split(tx)
    f = tx[tx.fraud_label == 1]
    rings = f.groupby("ring_id").agg(lifetime_days=("timestamp", lambda s: (s.max() - s.min()).total_seconds() / 86400),
                                     n_events=("timestamp", "size"), typology=("fraud_pattern", "first")).reset_index()
    rings["hidden_share"], rings["adapted"] = np.nan, 0
    ds = Dataset(tx, train, val, test, rings, 42, 1.0, {})
    gc.collect()
    listed = rings[rings.typology != "unlisted"]
    print(f"AMLworld (10 days): {len(tx):,} txns, laundering {tx.fraud_label.mean():.3%}, "
          f"{len(listed)} pattern rings (median {listed.n_events.median():.0f} txns, median lifetime "
          f"{listed.lifetime_days.median():.2f} d), test laundering {test.fraud_label.sum()}", flush=True)
    listed.describe().to_csv(os.path.join(out, "pattern_ring_stats.csv"))
    save_json({"n": len(tx), "train": len(train), "val": len(val), "test": len(test)}, os.path.join(out, "split.json"))

    return ds, f, rings


def run(seeds=(42, 43, 44)) -> None:
    out = out_dir("amlworld")
    t0 = time.time()
    ds, f, rings = prepare(out)
    tx, train, val, test = ds.tx, ds.train, ds.val, ds.test
    idx = TemporalIndex(tx, k=10)
    rows, ring_tabs, scores = [], [], {}
    names = list(FEATURE_FAMILIES) + ["Neighbour-Aggregation XGBoost"] + GNNS + ENTITY_SAGE
    params = {}
    for seed in seeds:
        for name in names:
            with Timer(f"{name} seed {seed}"):
                if name in FEATURE_FAMILIES:
                    if name not in params:
                        params[name] = tune_xgb(FEATURE_FAMILIES[name], ds, n_iter=4, seed=42)
                    model = tabular(FEATURE_FAMILIES[name], seed, params[name], name)
                    res = fit_score(model, ds)
                elif name == "Neighbour-Aggregation XGBoost":
                    from src.classifier_utils import get_classifier
                    from src.pipeline import XGB_SPACE, tune
                    nb = lambda p, s: NeighborAggXGB(idx, s, classifier=get_classifier(s, **p))
                    if name not in params:
                        params[name] = tune(lambda p: nb(p, 42), ds, XGB_SPACE, 4, 42)
                    res = fit_score(nb(params[name], seed), ds)
                elif name in ENTITY_SAGE:
                    # C3 on real data: entity GraphSAGE, leakage-safe inductive vs published-style
                    # transductive (graph over ALL transactions incl. test)
                    from experiments.common import make_model
                    res = fit_score(make_model(name, ds, seed, {}, idx), ds)
                else:
                    res = _fit_gnn(name, ds, idx, seed, params)
            sc = score_all(ds, res)
            y = ds.test.fraud_label.to_numpy()
            row = {"seed": seed, "model": name, **sc["row"]}
            if seed == seeds[0]:
                boot = cluster_bootstrap_indices(ds.test, 300, 42)
                rng = np.random.default_rng(0)
                iid = [rng.integers(0, len(y), len(y)) for _ in range(300)]
                row["pr_auc_ci_ring"] = "[%.3f, %.3f]" % bootstrap_ci(y, res["p_test"], boot)
                row["pr_auc_ci_iid"] = "[%.3f, %.3f]" % bootstrap_ci(y, res["p_test"], iid)
                scores[name] = res
                for b in BUDGETS:
                    rt = ring_rows(ds, res["p_test"], sc["thr"][b], model=name, budget=b)
                    ring_tabs.append(rt[(rt.ring_txns >= 3) & (rt.fraud_pattern != "unlisted")] if len(rt) else rt)
            rows.append(row)
            print(f"   {name} seed {seed}: PR-AUC {sc['row']['pr_auc']:.4f}", flush=True)
            pd.DataFrame(rows).to_csv(os.path.join(out, "metrics_by_seed.csv"), index=False)
    m = pd.DataFrame(rows)
    summ = m.groupby("model", sort=False)[["pr_auc", "roc_auc", "recall_at_1pct_fpr"]].agg(["mean", "std"])
    summ.columns = [f"{a}_{b}" for a, b in summ.columns]
    summ.to_csv(os.path.join(out, "metrics_summary.csv"))
    nonempty = [t for t in ring_tabs if len(t)]
    if nonempty:  # no qualifying rings in the test window -> skip C1 tables
        rt = pd.concat(nonempty, ignore_index=True)
        rt.to_csv(os.path.join(out, "ring_tables.csv"), index=False)
        from experiments.main import _pool_rings
        _pool_rings(rt, ["model", "budget"]).to_csv(os.path.join(out, "ring_summary.csv"), index=False)

    # C2 on real rings: label-delay sweep + model-free within-pattern reachability
    sweep = []
    start = f.groupby("ring_id")["timestamp"].min()
    tf = ds.test[ds.test.fraud_label == 1]
    for d in DELAYS:
        _set_labels(ds, d)
        for seed in seeds:
            res = fit_score(tabular(FEATURE_FAMILIES["Combined + delayed labels"], seed,
                                    params.get("Combined + delayed labels")), ds)
            sweep.append({"delay_days": d, "seed": seed,
                          "pr_auc": average_precision_score(ds.test.fraud_label, res["p_test"]),
                          "reachability": 0.0 if not np.isfinite(d) else float(
                              ((tf.timestamp >= tf.ring_id.map(start) + pd.Timedelta(days=d)) & (tf.ring_seq > 0)).mean())})
        print(f"   delay {d}: done", flush=True)
        pd.DataFrame(sweep).to_csv(os.path.join(out, "label_delay_sweep.csv"), index=False)

    # C3 analogue: test laundering whose sender or receiver account was used in training laundering
    trf = ds.train[ds.train.fraud_label == 1]
    seen = ds.test.customer_id.isin(set(trf.customer_id) | set(trf.device_id.str.replace("R", "S", n=1))) | \
        ds.test.device_id.isin(set(trf.device_id) | set(trf.customer_id.str.replace("S", "R", n=1)))
    strata = []
    for name, res in scores.items():
        thr = score_all(ds, res)["thr"][PRIMARY_BUDGET]
        for label, mk in [("seen-bad account", seen), ("unseen accounts", ~seen)]:
            fm = (mk & (ds.test.fraud_label == 1)).to_numpy()
            strata.append({"model": name, "stratum": label, "n_fraud": int(fm.sum()),
                           "recall": float((res["p_test"][fm] >= thr).mean()) if fm.any() else np.nan})
    pd.DataFrame(strata).to_csv(os.path.join(out, "seen_vs_unseen.csv"), index=False)
    tef = ds.test[ds.test.fraud_label == 1]
    sc_ = tef.customer_id.isin(set(trf.customer_id))
    sd_ = tef.device_id.isin(set(trf.device_id))
    save_json({"seen_bad_customer": float(sc_.mean()), "seen_bad_device": float(sd_.mean()),
               "seen_bad_any": float((sc_ | sd_).mean()), "fraud_rate": float(ds.tx.fraud_label.mean())},
              os.path.join(out, "calibration_stats.json"))
    save_json({k: v for k, v in params.items()}, os.path.join(out, "tuned_params.json"))
    save_json({"seeds": list(seeds), "legit_sample_gnn": LEGIT_SAMPLE, "minutes": (time.time() - t0) / 60},
              os.path.join(out, "run_info.json"))
    print(summ.round(4).to_string())
    print(f"amlworld done in {(time.time() - t0) / 60:.1f} min")


def run_gnn_only(seeds=(42, 43, 44)) -> None:
    """Refit ONLY the temporal GNNs (after the standardiser fix in src/temporal_gnn.py) and replace
    their rows in results/amlworld/ -- the XGBoost and entity-GraphSAGE results are unaffected."""
    out = out_dir("amlworld")
    t0 = time.time()
    ds, f, rings = prepare(out)
    idx = TemporalIndex(ds.tx, k=10)
    params, rows, tabs, scores = {}, [], [], {}
    y = ds.test.fraud_label.to_numpy()
    for seed in seeds:
        for name in GNNS:
            with Timer(f"{name} seed {seed}"):
                res = _fit_gnn(name, ds, idx, seed, params)
            sc = score_all(ds, res)
            row = {"seed": seed, "model": name, **sc["row"]}
            if seed == seeds[0]:
                boot = cluster_bootstrap_indices(ds.test, 300, 42)
                rng = np.random.default_rng(0)
                iid = [rng.integers(0, len(y), len(y)) for _ in range(300)]
                row["pr_auc_ci_ring"] = "[%.3f, %.3f]" % bootstrap_ci(y, res["p_test"], boot)
                row["pr_auc_ci_iid"] = "[%.3f, %.3f]" % bootstrap_ci(y, res["p_test"], iid)
                scores[name] = res
                for b in BUDGETS:
                    rt = ring_rows(ds, res["p_test"], sc["thr"][b], model=name, budget=b)
                    tabs.append(rt[(rt.ring_txns >= 3) & (rt.fraud_pattern != "unlisted")] if len(rt) else rt)
            rows.append(row)
            print(f"   {name} seed {seed}: PR-AUC {sc['row']['pr_auc']:.4f}", flush=True)
    m = pd.read_csv(os.path.join(out, "metrics_by_seed.csv"))
    m = pd.concat([m[~m.model.isin(GNNS)], pd.DataFrame(rows)], ignore_index=True)
    m.to_csv(os.path.join(out, "metrics_by_seed.csv"), index=False)
    summ = m.groupby("model", sort=False)[["pr_auc", "roc_auc", "recall_at_1pct_fpr"]].agg(["mean", "std"])
    summ.columns = [f"{a}_{b}" for a, b in summ.columns]
    summ.to_csv(os.path.join(out, "metrics_summary.csv"))
    rp = os.path.join(out, "ring_tables.csv")
    if os.path.exists(rp):
        rt = pd.read_csv(rp)
        rt = pd.concat([rt[~rt.model.isin(GNNS)]] + [t for t in tabs if len(t)], ignore_index=True)
        rt.to_csv(rp, index=False)
        from experiments.main import _pool_rings
        _pool_rings(rt, ["model", "budget"]).to_csv(os.path.join(out, "ring_summary.csv"), index=False)
    tp = os.path.join(out, "tuned_params.json")
    import json
    old = json.load(open(tp)) if os.path.exists(tp) else {}
    old.update(params)
    save_json(old, tp)
    print(summ.round(4).to_string())
    print(f"amlworld gnn-only done in {(time.time() - t0) / 60:.1f} min")
