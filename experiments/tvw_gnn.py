"""C5 on graph neural networks: does takedown-value weighting (TVW) also move the ring clock for
the temporal GNNs (GraphSAGE, GAT, CARE-style fraud GNN)?

Same protocol as experiments/tvw.py on the hard preset: each GNN is tuned on validation for the
control (beta=0), and the identical architecture / hyper-parameters / seed are retrained with
TVW weights in the (weighted) BCE loss. Early stopping stays on validation PR-AUC for both, so
any ring-level change comes from the weights alone.

Outputs (results/tvw_gnn/): runs.csv, ring_tables.csv, summary.csv, paired.csv, beta_curve.csv
"""

from __future__ import annotations

import os
import time

import pandas as pd

from experiments.common import Timer, make_model, out_dir, save_json, tune_model
from experiments.tvw import _evaluate, report
from src.data_generator import HARD_PRESET, GeneratorConfig
from src.pipeline import build_dataset
from src.temporal_gnn import TemporalIndex

GNN_MODELS = ["Temporal GraphSAGE", "Temporal GAT", "CARE-style GNN"]
GNN_BETAS = [0.0, 0.5, 1.0]


def run(seeds=tuple(range(42, 52))) -> None:
    out = out_dir("tvw_gnn")
    t0 = time.time()
    runs, tabs = [], []
    for seed in seeds:
        with Timer(f"tvw_gnn seed {seed}"):
            ds = build_dataset(seed, GeneratorConfig(**HARD_PRESET), label_delay_days=30.0)
            idx = TemporalIndex(ds.tx, k=20)
            cache = {m: tune_model(m, ds, seed, idx) for m in GNN_MODELS}
            r, t = _evaluate(ds, lambda name, p, s: make_model(name, ds, s, p, idx), seed, "synthetic (hard)",
                             GNN_MODELS, oracle=True, cache=cache, betas=GNN_BETAS)
        runs += r; tabs += t
        pd.DataFrame(runs).to_csv(os.path.join(out, "runs.csv"), index=False)
        pd.concat(tabs, ignore_index=True).to_csv(os.path.join(out, "ring_tables.csv"), index=False)
    report(out)
    save_json({"seeds": list(seeds), "betas": GNN_BETAS, "minutes": (time.time() - t0) / 60},
              os.path.join(out, "run_info.json"))
    print(f"tvw_gnn done in {(time.time() - t0) / 60:.1f} min")
