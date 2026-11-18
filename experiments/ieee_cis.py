"""IEEE-CIS real-data replication (Vesta / IEEE Computational Intelligence Society, Kaggle 2019).

Data: download train_transaction.csv and train_identity.csv from
https://www.kaggle.com/competitions/ieee-fraud-detection/data into data/IEEE_CIS/
(competition rules forbid redistribution -- the folder is git-ignored).

Schema mapping onto the synthetic schema (documented limitations in README):
  timestamp      2017-12-01 + TransactionDT seconds (the community-standard reference date)
  customer_id    "uid" = card1 + addr1 + account-start day (TransactionDT/86400 - D1), the
                 widely used client identifier for this dataset
  card_id        card1..card6
  device_id      DeviceInfo + id_30 + id_31 + id_33 (identity rows only); otherwise a unique
                 per-transaction token so missing data never creates fake sharing
  ip_id          network-fingerprint PROXY: id_17 + id_19 + id_20 (IEEE-CIS has no IP field)
  merchant_id    ProductCD
  payment_method ProductCD
  success        not available -> constant 1 (failure-rate features carry no information)
  ring_id        PROXY rings: connected components of fraud transactions linked by a shared
                 uid, card or device (no ground-truth rings exist in IEEE-CIS)

Experiments: model comparison (incl. a Vesta-features baseline), C1 on proxy rings,
C2 label-delay sweep, C3 real-data analogue (seen vs unseen fraud entities; transductive vs
inductive GraphSAGE), G5 interval widths, G6 first-seen entities.
Outputs: results/ieee/
"""

from __future__ import annotations

import os
import time

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

from experiments.common import (BUDGETS, PRIMARY_BUDGET, Timer, make_model, out_dir, ring_rows, save_json, score_all,
                                tune_model)
from src.classifier_utils import get_classifier
from src.evaluation import bootstrap_ci, cluster_bootstrap_indices
from src.graph_features import build_graph_features
from src.label_features import LABEL_FEATURE_COLS, build_label_features
from src.pipeline import Dataset, FEATURE_FAMILIES, _split, fit_score, with_label_delay
from src.preprocessing import build_transaction_features, set_payment_methods
from src.ring_metrics import add_ring_sequence
from src.temporal_gnn import NeighborAggXGB, TemporalGNN, TemporalIndex

IEEE_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "IEEE_CIS")
START = pd.Timestamp("2017-12-01")
DELAYS = [0, 1, 3, 7, 14, 30, 60, 90, float("inf")]


def _key(df: pd.DataFrame, cols, prefix: str) -> pd.Series:
    parts = [df[c].astype("string").fillna("?") for c in cols]
    out = parts[0]
    for p in parts[1:]:
        out = out + "|" + p
    return prefix + out


def map_ieee(trans: pd.DataFrame, ident: pd.DataFrame | None = None) -> pd.DataFrame:
    """Map raw IEEE-CIS tables onto the project's transaction schema (pure function, testable)."""
    df = trans.merge(ident, on="TransactionID", how="left") if ident is not None else trans.copy()
    missing = [c for c in ["DeviceInfo", "id_30", "id_31", "id_33", "id_17", "id_19", "id_20", "D1", "addr1",
                           "card2", "card3", "card4", "card5", "card6"] if c not in df.columns]
    if missing:
        df = pd.concat([df, pd.DataFrame(np.nan, index=df.index, columns=missing)], axis=1)
    tid = df["TransactionID"].astype(str)
    day = (df["TransactionDT"] // 86400).astype("Int64")
    acct_start = (day - df["D1"].round().astype("Int64")).astype("string").fillna("?")
    out = pd.DataFrame({
        "transaction_id": "TX" + tid,
        "timestamp": START + pd.to_timedelta(df["TransactionDT"], unit="s"),
        "customer_id": _key(df.assign(_s=acct_start), ["card1", "addr1", "_s"], "U"),
        "card_id": _key(df, ["card1", "card2", "card3", "card4", "card5", "card6"], "K"),
        "merchant_id": "M" + df["ProductCD"].astype(str),
        "amount": df["TransactionAmt"].astype(float),
        "payment_method": df["ProductCD"].astype(str),
        "success": 1,
        "fraud_label": df["isFraud"].astype(int),
    })
    has_dev = df["DeviceInfo"].notna()
    out["device_id"] = np.where(has_dev, _key(df, ["DeviceInfo", "id_30", "id_31", "id_33"], "D"), "Dnone_" + tid)
    has_net = df[["id_17", "id_19", "id_20"]].notna().all(axis=1)
    out["ip_id"] = np.where(has_net, _key(df, ["id_17", "id_19", "id_20"], "I"), "Inone_" + tid)
    out["ring_id"] = proxy_rings(out)
    out["fraud_pattern"] = np.where(out.fraud_label == 1, "ieee_proxy_ring", "")
    out["graph_hidden_fraud"] = 0
    out["legitimate_shared_env"] = 0
    out["segment"] = "na"
    return out


MAX_LINK_CUSTOMERS = 10


def specific_keys(tx: pd.DataFrame, col: str, max_customers: int = MAX_LINK_CUSTOMERS) -> set:
    """Entity keys shared by at most ``max_customers`` distinct customers. Coarse keys (e.g. the
    device fingerprint 'Windows|chrome 63.0', shared by ~2,900 customers) would merge unrelated
    fraud into one giant component. Computed over the full table: this is an evaluation
    construct for grouping fraud, never a model feature."""
    c = tx.groupby(col)["customer_id"].nunique()
    keys = set(c[c <= max_customers].index)
    return {k for k in keys if not str(k).startswith(("Dnone_", "Inone_"))}


def proxy_rings(tx: pd.DataFrame) -> np.ndarray:
    """Connected components of fraud transactions sharing a uid, or a SPECIFIC card / device
    (shared by <= MAX_LINK_CUSTOMERS customers), via union-find."""
    fraud = np.flatnonzero(tx["fraud_label"].to_numpy() == 1)
    allowed = {"card_id": specific_keys(tx, "card_id"), "device_id": specific_keys(tx, "device_id")}
    parent = {int(i): int(i) for i in fraud}

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for col in ["customer_id", "card_id", "device_id"]:
        first: dict = {}
        for i, v in zip(fraud, tx[col].to_numpy()[fraud]):
            if col in allowed and v not in allowed[col]:
                continue
            if v in first:
                ra, rb = find(int(i)), find(first[v])
                if ra != rb:
                    parent[ra] = rb
            else:
                first[v] = int(i)
    ring = np.full(len(tx), -1, dtype=np.int64)
    roots = {r: k for k, r in enumerate(sorted({find(int(i)) for i in fraud}))}
    for i in fraud:
        ring[i] = roots[find(int(i))]
    return ring


def load_ieee(nrows: int | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    tpath = os.path.join(IEEE_DIR, "train_transaction.csv")
    ipath = os.path.join(IEEE_DIR, "train_identity.csv")
    if not os.path.exists(tpath):
        raise FileNotFoundError(
            f"{tpath} not found. Download train_transaction.csv and train_identity.csv from "
            "https://www.kaggle.com/competitions/ieee-fraud-detection/data into data/IEEE_CIS/")
    raw = pd.read_csv(tpath, nrows=nrows)
    ident = pd.read_csv(ipath) if os.path.exists(ipath) else None
    return raw, ident


class VestaXGB:
    """Strong realistic tabular baseline: XGBoost on Vesta's ~430 engineered features (NaN kept)."""

    def __init__(self, raw: pd.DataFrame, extra_cols=None, seed: int = 42, params=None):
        self.name = "Vesta features (XGBoost)" + (" + our graph features" if extra_cols else "")
        drop = {"TransactionID", "isFraud", "TransactionDT"}
        feats = raw.drop(columns=[c for c in raw.columns if c in drop])
        for c in feats.columns:
            if feats[c].dtype == object or str(feats[c].dtype) in ("string", "str"):
                feats[c] = feats[c].astype("category").cat.codes.replace(-1, np.nan)
        self.X = feats.astype(np.float32)
        self.extra_cols = extra_cols or []
        self.model = get_classifier(seed, **(params or {}))

    def _mat(self, df: pd.DataFrame) -> np.ndarray:
        X = self.X.iloc[df["_raw"].to_numpy()].to_numpy()
        if self.extra_cols:
            X = np.hstack([X, df[self.extra_cols].to_numpy(dtype=np.float32)])
        return X

    def fit(self, df, y):
        self.model.fit(self._mat(df), y)
        return self

    def predict_proba(self, df, history=None):
        return self.model.predict_proba(self._mat(df))[:, 1]


def build_ieee_dataset(raw: pd.DataFrame, ident, label_delay_days: float = 30.0) -> Dataset:
    set_payment_methods(sorted(raw["ProductCD"].astype(str).unique()))
    tx = map_ieee(raw, ident)
    tx["_raw"] = np.arange(len(tx))  # row in the raw table (for Vesta features)
    with Timer("ieee transaction features"):
        tx = build_transaction_features(tx)
    with Timer("ieee graph features"):
        tx = build_graph_features(tx)
    with Timer("ieee label features"):
        tx = build_label_features(tx, label_delay_days)
    tx = add_ring_sequence(tx)
    tx["_row"] = np.arange(len(tx))
    train, val, test = _split(tx)
    rings = (tx[tx.fraud_label == 1].groupby("ring_id")
             .agg(lifetime_days=("timestamp", lambda s: (s.max() - s.min()).total_seconds() / 86400),
                  n_events=("timestamp", "size")).reset_index())
    rings["hidden_share"], rings["adapted"] = np.nan, 0
    return Dataset(tx, train, val, test, rings, 42, label_delay_days, {})


def first_fraud_mask(tx: pd.DataFrame) -> np.ndarray:
    """True for each client's (uid's) FIRST fraud transaction. IEEE-CIS labels every transaction of
    a client after a reported fraud as fraud, so later fraud rows are partly predictable from the
    labelling convention itself; first-fraud rows are not."""
    f = tx[tx.fraud_label == 1].sort_values(["timestamp", "transaction_id"])
    first_ids = set(f.drop_duplicates("customer_id")["transaction_id"])
    return tx["transaction_id"].isin(first_ids).to_numpy()


def _first_fraud_ap(test: pd.DataFrame, p: np.ndarray) -> float:
    m = (test["fraud_label"].to_numpy() == 0) | test["first_fraud"].to_numpy()
    return float(average_precision_score(test["fraud_label"].to_numpy()[m], p[m]))


def calibration_stats(ds) -> dict:
    """Statistics comparable with the synthetic generator (fix 4: sim-to-real calibration)."""
    trf, tef = ds.train[ds.train.fraud_label == 1], ds.test[ds.test.fraud_label == 1]
    spec_card, spec_dev = specific_keys(ds.tx, "card_id"), specific_keys(ds.tx, "device_id")
    seen_c = tef.customer_id.isin(set(trf.customer_id))
    seen_d = tef.device_id.isin(set(trf.device_id) & spec_dev)
    seen_k = tef.card_id.isin(set(trf.card_id) & spec_card)
    return {"seen_bad_customer": float(seen_c.mean()), "seen_bad_device": float(seen_d.mean()),
            "seen_bad_card": float(seen_k.mean()), "seen_bad_any": float((seen_c | seen_d | seen_k).mean()),
            "fraud_rate": float(ds.tx.fraud_label.mean())}


def run(seeds=(42, 43, 44), nrows: int | None = None) -> None:
    out = out_dir("ieee")
    t0 = time.time()
    raw, ident = load_ieee(nrows)
    ds = build_ieee_dataset(raw, ident, 30.0)
    ff = first_fraud_mask(ds.tx)
    for part in (ds.tx, ds.train, ds.val, ds.test):
        part["first_fraud"] = ff[part["_row"].to_numpy()]
    print(f"IEEE-CIS: {len(ds.tx):,} txns, fraud {ds.tx.fraud_label.mean():.2%}, proxy rings "
          f"{ds.rings.shape[0]:,}; test fraud {int(ds.test.fraud_label.sum())} of which first-of-client "
          f"{int(ds.test.first_fraud.sum())}", flush=True)
    ds.rings.describe().to_csv(os.path.join(out, "proxy_ring_stats.csv"))
    save_json(calibration_stats(ds), os.path.join(out, "calibration_stats.json"))
    idx = TemporalIndex(ds.tx, k=20)
    graph_cols = FEATURE_FAMILIES["Combined (full graph)"][len(FEATURE_FAMILIES["Transaction Baseline"]):]
    vesta = VestaXGB(raw)
    vesta_fn = lambda df: vesta._mat(df)  # NaN kept (trees)

    def vesta_dense_fn(df):  # NaN -> 0 plus missingness indicators (neural nets)
        X = vesta._mat(df)
        return np.hstack([np.nan_to_num(np.sign(X) * np.log1p(np.abs(X))), np.isnan(X).astype(np.float32)])

    from src.classifier_utils import get_classifier
    from src.pipeline import XGB_SPACE, tune
    vesta_factories = {
        "Vesta features (XGBoost)": lambda p, s: VestaXGB(raw, None, s, p),
        "Vesta + our graph features": lambda p, s: VestaXGB(raw, graph_cols, s, p),
        "Vesta + graph + delayed labels": lambda p, s: VestaXGB(raw, graph_cols + LABEL_FEATURE_COLS, s, p),
        "Vesta + Neighbour-Aggregation XGBoost":
            lambda p, s: NeighborAggXGB(idx, s, classifier=get_classifier(s, **p), target_fn=vesta_fn),
    }
    ours = ["Transaction Baseline", "Combined (full graph)", "Combined + delayed labels",
            "Neighbour-Aggregation XGBoost", "Temporal GraphSAGE", "Temporal GAT", "TGN-style Memory",
            "GraphSAGE (entity, inductive)", "GraphSAGE (transductive)"]
    from experiments.common import ORIGINALS, originals_enabled
    if originals_enabled():
        ours += ORIGINALS
    order = ours[:3] + list(vesta_factories) + ours[3:] + ["Vesta + Temporal GAT"]
    params: dict = {}
    rows, ring_tabs, scores = [], [], {}
    for seed in seeds:
        for name in order:
            with Timer(f"{name} seed {seed}"):
                if name not in params:  # every model tuned once on validation (seed 42), reused for other seeds
                    if name in vesta_factories:
                        params[name] = tune(lambda p: vesta_factories[name](p, 42), ds, XGB_SPACE, 8, 42)
                    elif name in ("GraphSAGE (entity, inductive)", "GraphSAGE (transductive)", "Vesta + Temporal GAT") \
                            or name in ORIGINALS:
                        params[name] = {}
                    else:
                        params[name] = tune_model(name, ds, 42, idx, n_iter=8)
                if name in vesta_factories:
                    model = vesta_factories[name](params[name], seed)
                elif name == "Vesta + Temporal GAT":
                    model = TemporalGNN(idx, "gat", seed=seed, val_df=ds.val, target_fn=vesta_dense_fn)
                else:
                    model = make_model(name, ds, seed, params[name], idx)
                res = fit_score(model, ds)
            sc = score_all(ds, res)
            y = ds.test.fraud_label.to_numpy()
            row = {"seed": seed, "model": name, **sc["row"], "pr_auc_first_fraud": _first_fraud_ap(ds.test, res["p_test"])}
            if seed == seeds[0]:
                boot = cluster_bootstrap_indices(ds.test, 300, 42)
                rng = np.random.default_rng(0)
                iid = [rng.integers(0, len(y), len(y)) for _ in range(300)]
                row["pr_auc_ci_ring"] = "[%.3f, %.3f]" % bootstrap_ci(y, res["p_test"], boot)
                row["pr_auc_ci_iid"] = "[%.3f, %.3f]" % bootstrap_ci(y, res["p_test"], iid)
                scores[name] = res
                for b in BUDGETS:  # C1 on proxy rings with >= 3 transactions in the test window
                    rt_b = ring_rows(ds, res["p_test"], sc["thr"][b], model=name, budget=b)
                    ring_tabs.append(rt_b[rt_b.ring_txns >= 3] if len(rt_b) else rt_b)
            rows.append(row)
            print(f"   PR-AUC {sc['row']['pr_auc']:.4f}  first-fraud PR-AUC {row['pr_auc_first_fraud']:.4f}", flush=True)
            pd.DataFrame(rows).to_csv(os.path.join(out, "metrics_by_seed.csv"), index=False)
    m = pd.DataFrame(rows)
    summ = m.groupby("model", sort=False)[["pr_auc", "pr_auc_first_fraud", "roc_auc", "recall_at_1pct_fpr"]].agg(["mean", "std"])
    summ.columns = [f"{a}_{b}" for a, b in summ.columns]
    summ.to_csv(os.path.join(out, "metrics_summary.csv"))
    save_json(params, os.path.join(out, "tuned_params.json"))
    nonempty = [t for t in ring_tabs if len(t)]
    if nonempty:  # no qualifying rings in the test window -> skip C1 tables
        rt = pd.concat(nonempty, ignore_index=True)
        rt.to_csv(os.path.join(out, "ring_tables.csv"), index=False)
        from experiments.main import _pool_rings
        _pool_rings(rt, ["model", "budget"]).to_csv(os.path.join(out, "ring_summary.csv"), index=False)

    # C3 analogue: recall on test fraud whose entity was / was not used by fraud in training
    trf = ds.train[ds.train.fraud_label == 1]
    te = ds.test
    spec_card, spec_dev = specific_keys(ds.tx, "card_id"), specific_keys(ds.tx, "device_id")
    seen = (te["customer_id"].isin(set(trf.customer_id))
            | te["card_id"].isin(set(trf.card_id) & spec_card)
            | te["device_id"].isin(set(trf.device_id) & spec_dev))
    strata = []
    for name, res in scores.items():
        thr = score_all(ds, res)["thr"][PRIMARY_BUDGET]
        for label, mk in [("seen-bad entity", seen), ("unseen entities", ~seen)]:
            f = (mk & (te.fraud_label == 1)).to_numpy()
            strata.append({"model": name, "stratum": label, "n_fraud": int(f.sum()),
                           "recall": float((res["p_test"][f] >= thr).mean()) if f.any() else np.nan})
    pd.DataFrame(strata).to_csv(os.path.join(out, "seen_vs_unseen.csv"), index=False)

    # C2: label-delay sweep, all fraud AND first-fraud-of-client only
    sweep = []
    base = with_label_delay(ds, float("inf"))
    for d in DELAYS:
        dsd = base if not np.isfinite(d) else with_label_delay(base, d)
        for part in (dsd.train, dsd.val, dsd.test):
            part["first_fraud"] = ff[part["_row"].to_numpy()]
        for seed in seeds:
            for name, mk in [("Combined + delayed labels", lambda s: make_model("Combined + delayed labels", dsd, s,
                                                                               params["Combined + delayed labels"], idx)),
                             ("Vesta + graph + delayed labels",
                              lambda s: vesta_factories["Vesta + graph + delayed labels"](params["Vesta + graph + delayed labels"], s))]:
                res = fit_score(mk(seed), dsd)
                sweep.append({"delay": d, "seed": seed, "model": name,
                              "pr_auc": average_precision_score(dsd.test.fraud_label, res["p_test"]),
                              "pr_auc_first_fraud": _first_fraud_ap(dsd.test, res["p_test"])})
        print(f"   delay {d}: done", flush=True)
        pd.DataFrame(sweep).to_csv(os.path.join(out, "label_delay_sweep.csv"), index=False)
    save_json({"n": len(ds.tx), "fraud_rate": float(ds.tx.fraud_label.mean()), "seeds": list(seeds),
               "minutes": (time.time() - t0) / 60}, os.path.join(out, "run_info.json"))
    print(summ.round(4).to_string())
    print(f"ieee done in {(time.time() - t0) / 60:.1f} min")
