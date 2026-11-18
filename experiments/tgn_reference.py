"""Fix 5 (part): reference TGN (PyTorch Geometric components) on the main synthetic setting.

Same seeds, splits, thresholds and C1 metrics as ``main``; results are merged into the main
tables by the report. Outputs: results/main_tgn_reference/
"""

from __future__ import annotations

import os

import pandas as pd

from experiments.common import BUDGETS, PRIMARY_BUDGET, Timer, make_model, out_dir, ring_rows, score_all
from src.evaluation import subset_recall
from src.pipeline import build_dataset, fit_score
from src.temporal_gnn import TemporalIndex

NAME = "TGN (PyG reference)"


def run(seeds=(42, 43, 44, 45, 46)) -> None:
    out = out_dir("main_tgn_reference")
    rows, rings = [], []
    for seed in seeds:
        ds = build_dataset(seed, label_delay_days=30.0)
        idx = TemporalIndex(ds.tx, k=20)
        with Timer(f"{NAME} seed {seed}"):
            res = fit_score(make_model(NAME, ds, seed, {}, idx), ds)
        sc = score_all(ds, res)
        rows.append({"seed": seed, "model": NAME, **sc["row"],
                     "graph_hidden_recall": subset_recall(ds.test, res["p_test"], "graph_hidden_fraud", res["threshold"])})
        for b in BUDGETS:
            rings.append(ring_rows(ds, res["p_test"], sc["thr"][b], seed=seed, model=NAME, budget=b))
        print(f"   seed {seed}: PR-AUC {sc['row']['pr_auc']:.4f}", flush=True)
        pd.DataFrame(rows).to_csv(os.path.join(out, "metrics_by_seed.csv"), index=False)
    pd.concat([r for r in rings if len(r)], ignore_index=True).to_csv(os.path.join(out, "ring_tables.csv"), index=False)
