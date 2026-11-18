"""Illustrative examples for the paper: explanations + real past neighbourhood of graph-hidden
fraud where relationship context helps most (largest combined - baseline score gap)."""

from __future__ import annotations

import os

import numpy as np
import pandas as pd

from experiments.common import out_dir
from src.evaluation import plot_relationship_network
from src.explainability import explain_transaction, fit_signal_thresholds
from src.pipeline import FEATURE_FAMILIES, build_dataset, fit_score, tabular


def run(seed: int = 42, k: int = 5) -> None:
    out = out_dir("main")
    ds = build_dataset(seed, label_delay_days=30.0)
    base = fit_score(tabular(FEATURE_FAMILIES["Transaction Baseline"], seed), ds)["p_test"]
    comb = fit_score(tabular(FEATURE_FAMILIES["Combined (full graph)"], seed), ds)["p_test"]
    te = ds.test.reset_index(drop=True)
    hid = np.flatnonzero(te["graph_hidden_fraud"].to_numpy() == 1)
    top = hid[np.argsort(-(comb[hid] - base[hid]), kind="stable")[:k]]
    thr = fit_signal_thresholds(ds.train)
    rows = [{"transaction_id": te.transaction_id[i], "fraud_pattern": te.fraud_pattern[i],
             "baseline_score": round(float(base[i]), 4), "combined_score": round(float(comb[i]), 4),
             "signals": " | ".join(explain_transaction(te.iloc[i], thr))} for i in top]
    pd.DataFrame(rows).to_csv(os.path.join(out, "example_explanations.csv"), index=False)
    if len(top):
        plot_relationship_network(te.iloc[top[0]], ds.tx, os.path.join(out, "relationship_network.png"))
    print(pd.DataFrame(rows).to_string(index=False))
