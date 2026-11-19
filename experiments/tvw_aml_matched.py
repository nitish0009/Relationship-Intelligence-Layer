"""Pre-registered TVW test on AMLworld at prevalence-matched budgets (eligible per results/headroom/:
the control loses 36-69% of pattern-ring money at 0.15x / 0.3x positives).

Same validation-tuned model (hand-crafted graph features; neighbour-aggregation XGBoost), same seed, with vs
without TVW (beta = 1, ESS tempering, pseudo-rings from training data only). 5 seeds; paired ring-clustered
bootstrap CIs on money lost before the first alert, pattern rings with >= 2 transactions. Every outcome is
reported. Audit diagnostics: pseudo-ring vs true-pattern agreement (ARI) on training laundering and the
tempering exponent gamma actually applied.

Outputs (results/tvw_aml_matched/): diagnostics.json, runs.csv, ring_tables.csv, paired.csv
"""

from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score, average_precision_score

from experiments.common import Timer, out_dir, save_json
from experiments.headroom import BUDGET_MULTIPLES, MIN_RING, threshold_at_multiple
from src.clock_weights import pseudo_rings, remaining_value, temper
from src.ring_metrics import ring_detection_table


def _gamma(train, ring):
    from src.clock_weights import ring_totals, temper_rings
    R, T = remaining_value(train, ring), ring_totals(train, ring)
    f = train.fraud_label.to_numpy() == 1
    rel = R[f] / max(R[f].mean(), 1e-12)
    _, g = temper_rings(R[f] / np.maximum(T[f], 1e-12), T[f], 0.5)  # v2: ring-decomposed tempering
    ess_raw = rel.sum() ** 2 / (rel ** 2).sum() / len(rel)
    return float(g), float(ess_raw)


def run(seeds=(42, 43, 44, 45, 46), prefix: str = "HI-Small", out_name: str = "tvw_aml_matched") -> None:
    from experiments import amlworld as A
    from src.classifier_utils import get_classifier
    from src.clock_weights import TVWModel
    from src.pipeline import FEATURE_FAMILIES, XGB_SPACE, fit_score, tabular, tune
    from src.temporal_gnn import NeighborAggXGB, TemporalIndex
    out = out_dir(out_name)
    ds, f, rings = A.prepare(out, prefix)
    idx = TemporalIndex(ds.tx, k=10)
    # ---- audit diagnostics
    tr = ds.train
    pr = pseudo_rings(tr, identity_cols=A.AML_IDENTITY_COLS, identity_key=A.account_key)
    m = (tr.fraud_label.to_numpy() == 1) & (tr.fraud_pattern.to_numpy() != "unlisted")
    g, ess_raw = _gamma(tr, pr)
    diag = {"pseudo_rings": int(len(np.unique(pr[pr >= 0]))), "true_pattern_rings_train": int(tr.ring_id[m].nunique()),
            "ari_pseudo_vs_true_patterns": float(adjusted_rand_score(tr.ring_id.to_numpy()[m], pr[m])),
            "tempering_gamma": g, "raw_ess_share": ess_raw}
    save_json(diag, os.path.join(out, "diagnostics.json"))
    print("diagnostics", diag, flush=True)
    # ---- pattern rings only, as in the headroom check
    listed = set(ds.tx.loc[(ds.tx.fraud_label == 1) & (ds.tx.fraud_pattern != "unlisted"), "ring_id"])
    test = ds.test.copy()
    test.loc[~test.ring_id.isin(listed) & (test.fraud_label == 1), "ring_seq"] = -1
    tp = os.path.join(out_dir("amlworld"), "tuned_params.json")
    known = json.load(open(tp)) if (prefix == "HI-Small" and os.path.exists(tp)) else {}
    fac = {"Combined (full graph)": lambda p, s: tabular(FEATURE_FAMILIES["Combined (full graph)"], s, p),
           "Neighbour-Aggregation XGBoost": lambda p, s: NeighborAggXGB(idx, s, classifier=get_classifier(s, **p))}
    yv, yt = ds.val.fraud_label.to_numpy(), ds.test.fraud_label.to_numpy()
    runs, tabs = [], []
    for name, mk in fac.items():
        params = known.get(name) or tune(lambda p: mk(p, 42), ds, XGB_SPACE, 4, 42)
        for seed in seeds:
            for variant in ("control", "TVW"):
                with Timer(f"{name} {variant} seed {seed}"):
                    model = mk(params, seed) if variant == "control" else TVWModel(
                        mk(params, seed), 1.0, identity_cols=A.AML_IDENTITY_COLS, identity_key=A.account_key)
                    r = fit_score(model, ds)
                for c in BUDGET_MULTIPLES:
                    thr = threshold_at_multiple(yv, r["p_val"], c)
                    t = ring_detection_table(test, r["p_test"], thr)
                    t = t[t.ring_txns >= MIN_RING]
                    runs.append({"model": name, "seed": seed, "variant": variant, "budget_multiple": c,
                                 "pr_auc": average_precision_score(yt, r["p_test"]),
                                 "lost_share": float(t.amount_before_alert.sum() / t.ring_amount.sum())})
                    tabs.append(t.assign(model=name, seed=seed, variant=variant, budget_multiple=c))
            pd.DataFrame(runs).to_csv(os.path.join(out, "runs.csv"), index=False)
    rt = pd.concat(tabs, ignore_index=True)
    rt.to_csv(os.path.join(out, "ring_tables.csv"), index=False)
    from src.ring_bootstrap import paired_ring_bootstrap  # rings recur across seeds: resample RINGS
    rows = []
    for (name, c), g_ in rt.groupby(["model", "budget_multiple"]):
        rows.append({"model": name, "budget_multiple": c,
                     **paired_ring_bootstrap(g_[g_.variant == "control"], g_[g_.variant == "TVW"])})
    p = pd.DataFrame(rows)
    p.to_csv(os.path.join(out, "paired.csv"), index=False)
    print(p.round(4).to_string(index=False))
