"""Original CARE-GNN / PC-GNN code on the HARD preset with every Table-1 metric (same 5 seeds as
`main --preset hard`): PR-AUC, recall at 1% FPR, ring money lost at 0.1% / 0.5% budgets and recall on a
ring's first transaction.

Hyper-parameters: the validation-tuned configuration from results/originals_tuning/runs.csv where that
seed was already tuned (42-44); otherwise the same validation sweep (experiments.common.ORIGINAL_SPACE).
Requires scripts/fetch_third_party.sh and RIL_RUN_THIRD_PARTY=1 (runs external code).

Outputs (results/originals_hard/): metrics_by_seed.csv, ring_tables.csv, ring_summary.csv, warmup.csv,
summary.csv
"""

from __future__ import annotations

import ast
import os

import numpy as np
import pandas as pd

from experiments.common import (BUDGETS, ORIGINALS, Timer, make_model, originals_enabled, out_dir, ring_rows,
                                score_all, tune_original)
from experiments.main import _pool_rings
from src.data_generator import HARD_PRESET, GeneratorConfig
from src.pipeline import build_dataset, fit_score
from src.ring_metrics import warmup_curve
from src.temporal_gnn import TemporalIndex


def _known_params() -> dict:
    p = os.path.join(out_dir("originals_tuning"), "runs.csv")
    if not os.path.exists(p):
        return {}
    r = pd.read_csv(p)
    r = r[(r.preset == "hard") & (r.config == "validation-tuned")]
    return {(int(x.seed), x.model): ast.literal_eval(x.params) for _, x in r.iterrows()}


def run(seeds=(42, 43, 44, 45, 46)) -> None:
    if not originals_enabled():
        raise SystemExit("needs third_party/ (scripts/fetch_third_party.sh) and RIL_RUN_THIRD_PARTY=1")
    out = out_dir("originals_hard")
    known = _known_params()
    rows, tabs, warm = [], [], []
    for seed in seeds:
        ds = build_dataset(seed, GeneratorConfig(**HARD_PRESET), label_delay_days=30.0)
        idx = TemporalIndex(ds.tx, k=20)
        for name in ORIGINALS:
            with Timer(f"seed {seed} {name}"):
                params = known.get((seed, name))
                if params is None:
                    params = tune_original(name, ds, seed, idx)
                res = fit_score(make_model(name, ds, seed, params, idx), ds)
            sc = score_all(ds, res)
            rows.append({"seed": seed, "model": name, "params": str(params), **sc["row"]})
            for b in BUDGETS:
                tabs.append(ring_rows(ds, res["p_test"], sc["thr"][b], seed=seed, model=name, budget=b))
            w = warmup_curve(ds.test, res["p_test"], sc["thr"][0.005])
            w["seed"], w["model"] = seed, name
            warm.append(w)
            print(f"   {name} seed {seed}: PR-AUC {sc['row']['pr_auc']:.4f}", flush=True)
        pd.DataFrame(rows).to_csv(os.path.join(out, "metrics_by_seed.csv"), index=False)
    m = pd.DataFrame(rows)
    summ = m.groupby("model", sort=False)[["pr_auc", "recall_at_1pct_fpr"]].agg(["mean", "std"])
    summ.columns = [f"{a}_{b}" for a, b in summ.columns]
    summ.to_csv(os.path.join(out, "summary.csv"))
    rt = pd.concat([t for t in tabs if not t.empty], ignore_index=True)
    rt.to_csv(os.path.join(out, "ring_tables.csv"), index=False)
    _pool_rings(rt, ["model", "budget"]).to_csv(os.path.join(out, "ring_summary.csv"), index=False)
    wu = pd.concat(warm, ignore_index=True)
    wu = wu.groupby(["model", "k"]).apply(lambda g: pd.Series({"recall": np.average(g.recall, weights=g.n),
                                                               "n": g.n.sum()}), include_groups=False).reset_index()
    wu.to_csv(os.path.join(out, "warmup.csv"), index=False)
    print(summ.round(4).to_string())
