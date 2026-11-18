"""Fix 3: are the findings artefacts of the generator's assumptions? (sensitivity + hard preset)

Part A -- one-factor-at-a-time sweep over the generator's key assumptions. For every setting we
re-measure the paper's core quantities and check that their SIGN is stable:
  * relational gain        PR-AUC(Combined) - PR-AUC(Transaction Baseline)            (> 0)
  * neighbour-agg gain     PR-AUC(Neighbour-Agg XGB) - PR-AUC(Combined)               (>= 0)
  * C1 money-lost ratio    lost(Baseline) / lost(Combined) at the 0.5% alert budget   (> 1)
  * transductive inflation PR-AUC(GraphSAGE transductive) - PR-AUC(inductive)         (> 0)
  * no-delay inflation     PR-AUC(labels, D=0) - PR-AUC(labels, D=30)                 (>= 0)

Part B -- full model comparison under HARD_PRESET (more camouflage, noisier infrastructure,
more legitimate sharing) to show conclusions do not depend on near-ceiling scores.

Outputs (results/sensitivity/): sweep.csv, sweep_summary.csv, hard_metrics.csv,
hard_summary.csv, hard_rings.csv
"""

from __future__ import annotations

import os
import time

import numpy as np
import pandas as pd

from experiments.common import PRIMARY_BUDGET, make_model, out_dir, ring_rows, score_all, tune_model
from src.data_generator import HARD_PRESET, GeneratorConfig
from src.pipeline import FEATURE_FAMILIES, build_dataset, fit_score, tabular, with_label_delay
from src.temporal_gnn import TemporalIndex

FACTORS = {
    "ring_size_scale": [0.5, 2.0],
    "ring_hidden_concentration": [0.5, 5.0, None],
    "graph_hidden_share": [0.25, 0.65, 0.85],
    "noisy_infra_share": [0.0, 0.5, 0.75],
    "household_share": [0.05, 0.25],
    "n_carrier_ips": [100, 1000],
    "fraud_rate": [0.01, 0.07],
}
DEFAULTS = {f: getattr(GeneratorConfig(), f) for f in FACTORS}
HARD_MODELS = list(FEATURE_FAMILIES) + ["Neighbour-Aggregation XGBoost", "Temporal GraphSAGE", "Temporal GAT",
                                        "GraphSAGE (entity, inductive)", "GraphSAGE (transductive)"]


def _lost(ds, res) -> float:
    sc = score_all(ds, res)
    rt = ring_rows(ds, res["p_test"], sc["thr"][PRIMARY_BUDGET])
    return float(rt.amount_before_alert.sum() / rt.ring_amount.sum()) if len(rt) else np.nan


def _measure(cfg: GeneratorConfig, seed: int) -> dict:
    ds = build_dataset(seed, cfg, label_delay_days=30.0)
    idx = TemporalIndex(ds.tx, k=20)
    r = {}
    res = {n: fit_score(make_model(n, ds, seed, None, idx), ds) for n in
           ["Transaction Baseline", "Combined (full graph)", "Neighbour-Aggregation XGBoost",
            "Combined + delayed labels", "GraphSAGE (entity, inductive)", "GraphSAGE (transductive)"]}
    ap = {n: score_all(ds, v)["row"]["pr_auc"] for n, v in res.items()}
    ds0 = with_label_delay(ds, 0.0)
    ap0 = score_all(ds0, fit_score(tabular(FEATURE_FAMILIES["Combined + delayed labels"], seed), ds0))["row"]["pr_auc"]
    lost_b, lost_c = _lost(ds, res["Transaction Baseline"]), _lost(ds, res["Combined (full graph)"])
    r.update({
        "pr_auc_baseline": ap["Transaction Baseline"], "pr_auc_combined": ap["Combined (full graph)"],
        "relational_gain": ap["Combined (full graph)"] - ap["Transaction Baseline"],
        "nbragg_gain": ap["Neighbour-Aggregation XGBoost"] - ap["Combined (full graph)"],
        "c1_lost_baseline": lost_b, "c1_lost_combined": lost_c,
        "c1_lost_ratio": lost_b / lost_c if lost_c and lost_c > 0 else np.nan,
        "transductive_inflation": ap["GraphSAGE (transductive)"] - ap["GraphSAGE (entity, inductive)"],
        "no_delay_inflation": ap0 - ap["Combined + delayed labels"],
        "fraud_rate": float(ds.tx.fraud_label.mean()), "n_rings": int(ds.rings.shape[0]),
    })
    return r


def run(seeds=(42, 43, 44), hard_seeds=(42, 43, 44, 45, 46)) -> None:
    out = out_dir("sensitivity")
    t0 = time.time()
    rows = []
    for seed in seeds:  # default setting once per seed
        print(f"sensitivity: default seed={seed}", flush=True)
        rows.append({"factor": "default", "value": "default", "seed": seed, **_measure(GeneratorConfig(), seed)})
    for factor, values in FACTORS.items():
        for v in values:
            for seed in seeds:
                print(f"sensitivity: {factor}={v} seed={seed}", flush=True)
                rows.append({"factor": factor, "value": str(v), "seed": seed,
                             **_measure(GeneratorConfig(**{factor: v}), seed)})
        pd.DataFrame(rows).to_csv(os.path.join(out, "sweep.csv"), index=False)  # incremental save
    sweep = pd.DataFrame(rows)
    sweep.to_csv(os.path.join(out, "sweep.csv"), index=False)
    keys = ["relational_gain", "nbragg_gain", "c1_lost_ratio", "transductive_inflation", "no_delay_inflation"]
    summ = sweep.groupby(["factor", "value"], sort=False)[keys + ["pr_auc_baseline", "pr_auc_combined"]].mean()
    summ.to_csv(os.path.join(out, "sweep_summary.csv"))
    print(summ.round(3).to_string())

    # Part B: hard preset, full comparison, every model tuned per seed
    hrows, hrings = [], []
    for seed in hard_seeds:
        print(f"hard preset: seed={seed}", flush=True)
        ds = build_dataset(seed, GeneratorConfig(**HARD_PRESET), label_delay_days=30.0)
        idx = TemporalIndex(ds.tx, k=20)
        for name in HARD_MODELS:
            params = tune_model(name, ds, seed, idx, n_iter=10)
            res = fit_score(make_model(name, ds, seed, params, idx), ds)
            sc = score_all(ds, res)
            hrows.append({"seed": seed, "model": name, **sc["row"]})
            rt = ring_rows(ds, res["p_test"], sc["thr"][PRIMARY_BUDGET], seed=seed, model=name)
            hrings.append(rt)
        pd.DataFrame(hrows).to_csv(os.path.join(out, "hard_metrics.csv"), index=False)
    hm = pd.DataFrame(hrows)
    hs = hm.groupby("model", sort=False)[["pr_auc", "roc_auc", "recall_at_1pct_fpr"]].agg(["mean", "std"])
    hs.columns = [f"{a}_{b}" for a, b in hs.columns]
    hs.to_csv(os.path.join(out, "hard_summary.csv"))
    hr = pd.concat([r for r in hrings if len(r)], ignore_index=True)
    hr.to_csv(os.path.join(out, "hard_rings.csv"), index=False)
    print(hs.round(4).to_string())
    print(f"sensitivity done in {(time.time() - t0) / 60:.1f} min")
