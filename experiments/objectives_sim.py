"""Pre-registered H4 (PREREGISTRATION_IJCAI.md): is TVW more than reweighting?

Hard preset, 10 seeds. Each model is tuned on validation for the control; every variant reuses those
hyper-parameters and the seed. XGBoost (hand-crafted graph features, neighbour aggregation): control, TVW,
class-balanced, amount-weighted. Temporal GNNs (GraphSAGE, GAT): control, TVW, amount-weighted, focal (gamma 2).

Outputs (results/objectives_sim/): runs.csv, ring_tables.csv (resumable per seed)
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

from experiments.common import BUDGETS, Timer, make_model, out_dir, tune_model
from src.data_generator import HARD_PRESET, GeneratorConfig
from src.pipeline import build_dataset, fit_score, threshold_at_fpr
from src.ring_metrics import ring_detection_table

XGB = {"Combined (full graph)": ["control", "TVW", "class-balanced", "amount-weighted"],
       "Neighbour-Aggregation XGBoost": ["control", "TVW", "class-balanced", "amount-weighted"]}
GNN = {"Temporal GraphSAGE": ["control", "TVW", "amount-weighted", "focal"],
       "Temporal GAT": ["control", "TVW", "amount-weighted", "focal"]}


def _model(name, variant, ds, seed, params, idx):
    from src.clock_weights import TVWModel
    from src.objectives import WeightedModel
    if variant == "focal":
        return make_model(name, ds, seed, {**params, "focal_gamma": 2.0}, idx)
    base = make_model(name, ds, seed, params, idx)
    if variant == "control":
        return base
    if variant == "TVW":
        return TVWModel(base, 1.0)
    return WeightedModel(base, variant)


def run(seeds=tuple(range(42, 52)), config=None, out_name: str = "objectives_sim") -> None:
    from src.temporal_gnn import TemporalIndex
    out = out_dir(out_name)
    runs_p, rt_p = os.path.join(out, "runs.csv"), os.path.join(out, "ring_tables.csv")
    done = set(pd.read_csv(runs_p).seed) if os.path.exists(runs_p) else set()
    cfg = config or GeneratorConfig(**HARD_PRESET)
    for seed in seeds:
        if seed in done:
            continue
        ds = build_dataset(seed, cfg, label_delay_days=30.0)
        idx = TemporalIndex(ds.tx, k=20)
        yv, yt = ds.val.fraud_label.to_numpy(), ds.test.fraud_label.to_numpy()
        rows, tabs = [], []
        for name, vlist in {**XGB, **GNN}.items():
            params = tune_model(name, ds, seed, idx)
            for variant in vlist:
                with Timer(f"seed {seed} {name} {variant}"):
                    r = fit_score(_model(name, variant, ds, seed, params, idx), ds)
                for b in BUDGETS:
                    thr = threshold_at_fpr(yv, r["p_val"], b)
                    t = ring_detection_table(ds.test, r["p_test"], thr)
                    rows.append({"seed": seed, "model": name, "variant": variant, "budget": b,
                                 "pr_auc": average_precision_score(yt, r["p_test"]),
                                 "lost_share": float(t.amount_before_alert.sum() / t.ring_amount.sum())})
                    tabs.append(t.assign(seed=seed, model=name, variant=variant, budget=b))
        pd.DataFrame(rows).to_csv(runs_p, mode="a", header=not os.path.exists(runs_p), index=False)
        pd.concat(tabs).to_csv(rt_p, mode="a", header=not os.path.exists(rt_p), index=False)
        print(f"   seed {seed} done", flush=True)
