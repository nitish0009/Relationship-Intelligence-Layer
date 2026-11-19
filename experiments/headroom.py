"""Pre-registered headroom check (ordinary models only, NO TVW): how much ring money does the control
detector lose before the first alert on each real dataset?

Budgets are PREVALENCE-MATCHED: the validation threshold admits c x (#validation positives) alerts, for
c in BUDGET_MULTIPLES. (A fixed FPR budget is ~0.14 alerts per fraud on the simulator but ~5.5 per
laundering transfer on AMLworld, which hides any early-detection headroom there.)

Rule fixed in advance: TVW is evaluated only where the control loses >= 5% of multi-transaction ring
money at some multiple; every dataset is reported either way.

Outputs (results/headroom/): headroom.csv
"""

from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd

from experiments.common import Timer, out_dir
from src.ring_metrics import ring_detection_table

BUDGET_MULTIPLES = [0.05, 0.15, 0.3, 0.5, 1.0]
MIN_RING = 2  # the ring clock needs at least two transactions


def threshold_at_multiple(y_val: np.ndarray, p_val: np.ndarray, c: float) -> float:
    k = max(1, int(round(c * (y_val == 1).sum())))
    return float(np.sort(p_val)[::-1][min(k, len(p_val)) - 1])


def ring_rows(test: pd.DataFrame, p_test: np.ndarray, y_val, p_val, tag: dict) -> list:
    out = []
    for c in BUDGET_MULTIPLES:
        thr = threshold_at_multiple(y_val, p_val, c)
        t = ring_detection_table(test, p_test, thr)
        for label, tt in [("all rings", t), (f">={MIN_RING} txns", t[t.ring_txns >= MIN_RING] if len(t) else t)]:
            if len(tt):
                out.append({**tag, "budget_multiple": c, "rings": label, "n_rings": len(tt),
                            "lost_share": float(tt.amount_before_alert.sum() / tt.ring_amount.sum()),
                            "first_txn_recall_proxy": float((tt.txns_before_alert == 0).mean())})
    return out


def _elliptic(seeds):
    from experiments import elliptic as E
    from src.pipeline import XGB_SPACE
    from src.classifier_utils import get_classifier
    from sklearn.metrics import average_precision_score
    df = E.load()
    tr, va, te = E.split(df)
    cols = E.feature_cols(df)
    Xtr, Xva, Xte = (d[cols].to_numpy(np.float32) for d in (tr, va, te))
    ytr, yva = tr.fraud_label.to_numpy(), va.fraud_label.to_numpy()
    rng = np.random.default_rng(0)
    best, params = -1, {}
    for i in range(10):  # validation-tuned XGBoost (first candidate = defaults)
        p = {} if i == 0 else {k: v[rng.integers(len(v))] for k, v in XGB_SPACE.items()}
        m = get_classifier(42, **p).fit(Xtr, ytr)
        ap = average_precision_score(yva, m.predict_proba(Xva)[:, 1])
        if ap > best:
            best, params = ap, p
    rows = []
    for seed in seeds:
        m = get_classifier(seed, **params).fit(Xtr, ytr)
        rows += ring_rows(te, m.predict_proba(Xte)[:, 1], yva, m.predict_proba(Xva)[:, 1],
                          {"dataset": "Elliptic++", "model": "XGBoost (Elliptic features)", "seed": seed})
    return rows, params


def _aml(prefix, seeds):
    from experiments import amlworld as A
    from src.pipeline import FEATURE_FAMILIES, XGB_SPACE, tabular, tune, fit_score
    from src.classifier_utils import get_classifier
    from src.temporal_gnn import NeighborAggXGB, TemporalIndex
    ds, f, rings = A.prepare(out_dir("headroom"), prefix)
    idx = TemporalIndex(ds.tx, k=10)
    listed = set(ds.tx.loc[(ds.tx.fraud_label == 1) & (ds.tx.fraud_pattern != "unlisted"), "ring_id"])
    test = ds.test.copy()
    test.loc[~test.ring_id.isin(listed) & (test.fraud_label == 1), "ring_seq"] = -1  # pattern rings only
    tp = os.path.join(out_dir("amlworld"), "tuned_params.json")
    known = json.load(open(tp)) if (prefix == "HI-Small" and os.path.exists(tp)) else {}
    fac = {"Combined (full graph)": lambda p, s: tabular(FEATURE_FAMILIES["Combined (full graph)"], s, p),
           "Neighbour-Aggregation XGBoost": lambda p, s: NeighborAggXGB(idx, s, classifier=get_classifier(s, **p))}
    rows = []
    for name, mk in fac.items():
        params = known.get(name)
        if params is None:
            params = tune(lambda p: mk(p, 42), ds, XGB_SPACE, 4, 42)
        for seed in seeds:
            with Timer(f"{prefix} {name} seed {seed}"):
                r = fit_score(mk(params, seed), ds)
            rows += ring_rows(test, r["p_test"], ds.val.fraud_label.to_numpy(), r["p_val"],
                              {"dataset": f"AMLworld {prefix}", "model": name, "seed": seed})
    return rows


def run(seeds=(42, 43, 44), datasets=("elliptic", "HI-Small", "LI-Small")) -> None:
    out = out_dir("headroom")
    rows = []
    for d in datasets:
        with Timer(f"headroom {d}"):
            rows += _elliptic(seeds)[0] if d == "elliptic" else _aml(d, seeds)
        pd.DataFrame(rows).to_csv(os.path.join(out, "headroom.csv"), index=False)
    h = pd.DataFrame(rows)
    s = h.groupby(["dataset", "model", "rings", "budget_multiple"])[["lost_share", "n_rings"]].mean()
    print(s.round(4).to_string())
