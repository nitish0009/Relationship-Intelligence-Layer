"""G5: do confidence intervals mean anything when fraud comes in rings? (coverage study)

A model is trained ONCE (seed 42). It is then evaluated on the test splits of many
independent synthetic datasets drawn from the same process. Their mean PR-AUC approximates
the model's true expected test PR-AUC. For each test split we compute a 95% CI by
(a) i.i.d. row bootstrap and (b) ring-clustered bootstrap, and record whether it covers the
truth. Nominal coverage is 95%.

Outputs (results/g5/): coverage.csv, per_dataset.csv
"""

from __future__ import annotations

import os
import time

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

from experiments.common import out_dir
from src.evaluation import cluster_bootstrap_indices
from src.pipeline import FEATURE_FAMILIES, build_dataset, tabular

MODELS = ["Transaction Baseline", "Combined (full graph)"]


def _iid_indices(n: int, n_boot: int, seed: int):
    rng = np.random.default_rng(seed)
    return [rng.integers(0, n, n) for _ in range(n_boot)]


def run(n_datasets: int = 30, n_boot: int = 500) -> None:
    out = out_dir("g5")
    t0 = time.time()
    train_ds = build_dataset(42, label_delay_days=float("inf"))
    models = {m: tabular(FEATURE_FAMILIES[m], 42).fit(train_ds.train, train_ds.train.fraud_label.to_numpy())
              for m in MODELS}
    rows = []
    for k in range(n_datasets):
        seed = 1000 + k
        print(f"g5: dataset {k + 1}/{n_datasets}", flush=True)
        ds = build_dataset(seed, label_delay_days=float("inf"))
        y = ds.test.fraud_label.to_numpy()
        iid = _iid_indices(len(y), n_boot, seed)
        clu = cluster_bootstrap_indices(ds.test, n_boot, seed)
        for m, model in models.items():
            p = model.predict_proba(ds.test)
            ap = average_precision_score(y, p)
            r = {"dataset_seed": seed, "model": m, "pr_auc": ap}
            for name, idx in [("iid", iid), ("ring_clustered", clu)]:
                vals = np.array([average_precision_score(y[i], p[i]) for i in idx])
                r[f"{name}_lo"], r[f"{name}_hi"] = np.percentile(vals, [2.5, 97.5])
            rows.append(r)
    df = pd.DataFrame(rows)
    truth = df.groupby("model")["pr_auc"].mean()
    df["truth"] = df["model"].map(truth)
    df.to_csv(os.path.join(out, "per_dataset.csv"), index=False)
    cov = []
    for m, g in df.groupby("model"):
        for name in ["iid", "ring_clustered"]:
            cov.append({"model": m, "method": name,
                        "coverage": float(((g[f"{name}_lo"] <= g.truth) & (g.truth <= g[f"{name}_hi"])).mean()),
                        "mean_width": float((g[f"{name}_hi"] - g[f"{name}_lo"]).mean()),
                        "true_sd_of_test_pr_auc": float(g.pr_auc.std()), "n_datasets": len(g)})
    cov = pd.DataFrame(cov)
    cov.to_csv(os.path.join(out, "coverage.csv"), index=False)
    print(cov.round(4).to_string(index=False))
    print(f"g5 done in {(time.time() - t0) / 60:.1f} min")
