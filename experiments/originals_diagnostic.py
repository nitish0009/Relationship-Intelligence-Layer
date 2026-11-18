"""Diagnostic for fix 5: are the original CARE-GNN / PC-GNN handicapped by the authors' row-sum
feature normalisation (designed for bag-of-words features)? Seed 42, both settings.
Outputs: results/main_originals_diagnostic/diagnostic.csv"""

from __future__ import annotations

import os

import pandas as pd

from experiments.common import out_dir, score_all
from src.original_gnn import OriginalFraudGNN
from src.pipeline import build_dataset, fit_score
from src.temporal_gnn import TemporalIndex


def run(seeds=(42,)) -> None:
    out = out_dir("main_originals_diagnostic")
    rows = []
    for seed in seeds:
        ds = build_dataset(seed, label_delay_days=30.0)
        idx = TemporalIndex(ds.tx, k=20)
        for kind in ["care", "pcgnn"]:
            for row_norm in [True, False]:
                m = OriginalFraudGNN(idx, kind, ds.tx, ds.val, seed=seed, row_norm=row_norm)
                sc = score_all(ds, fit_score(m, ds))
                rows.append({"seed": seed, "model": m.name, "row_norm": row_norm, **sc["row"]})
                print(rows[-1]["model"], "row_norm", row_norm, "PR-AUC", round(rows[-1]["pr_auc"], 4), flush=True)
                pd.DataFrame(rows).to_csv(os.path.join(out, "diagnostic.csv"), index=False)
