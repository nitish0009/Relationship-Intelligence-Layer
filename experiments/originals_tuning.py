"""Fix: are the original CARE-GNN / PC-GNN results an artefact of the authors' review-graph defaults?

For each seed, compare the authors' default configuration with the configuration picked by a
small validation sweep (experiments.common.ORIGINAL_SPACE) on the default and the hard preset.
Requires scripts/fetch_third_party.sh and RIL_RUN_THIRD_PARTY=1 (runs external code).

Outputs (results/originals_tuning/): runs.csv, summary.csv
"""

from __future__ import annotations

import os

import pandas as pd
from sklearn.metrics import average_precision_score

from experiments.common import ORIGINALS, Timer, make_model, originals_enabled, out_dir, tune_original
from src.data_generator import HARD_PRESET, GeneratorConfig
from src.pipeline import build_dataset, fit_score
from src.temporal_gnn import TemporalIndex


def run(seeds=(42, 43, 44)) -> None:
    if not originals_enabled():
        raise SystemExit("needs third_party/ (scripts/fetch_third_party.sh) and RIL_RUN_THIRD_PARTY=1")
    out = out_dir("originals_tuning")
    rows = []
    for preset, cfg in [("default", None), ("hard", GeneratorConfig(**HARD_PRESET))]:
        for seed in seeds:
            ds = build_dataset(seed, cfg, label_delay_days=30.0)
            idx = TemporalIndex(ds.tx, k=20)
            for name in ORIGINALS:
                with Timer(f"{preset} seed {seed} {name}"):
                    tuned = tune_original(name, ds, seed, idx)
                    for label, p in [("authors' defaults", {}), ("validation-tuned", tuned)]:
                        res = fit_score(make_model(name, ds, seed, p, idx), ds)
                        rows.append({"preset": preset, "seed": seed, "model": name, "config": label, "params": str(p),
                                     "pr_auc": average_precision_score(ds.test.fraud_label, res["p_test"])})
                        print(f"   {label} {p}: test PR-AUC {rows[-1]['pr_auc']:.4f}", flush=True)
            pd.DataFrame(rows).to_csv(os.path.join(out, "runs.csv"), index=False)
    r = pd.DataFrame(rows)
    s = r.groupby(["preset", "model", "config"])["pr_auc"].agg(["mean", "std"]).reset_index()
    s.to_csv(os.path.join(out, "summary.csv"), index=False)
    print(s.round(4).to_string(index=False))
