"""C3: infrastructure-reuse dose-response -- how much do leaky protocols inflate results?

The generator's infra_reuse p is the probability that a ring's devices / IPs / accounts come
from a persistent attacker pool (p=0: every ring is fresh; p=1: the fixed-pool setup common in
synthetic benchmarks). Realistic (leakage-safe) models are compared with 'published-style'
protocols along the dial:
  * transductive GraphSAGE (graph includes test transactions)
  * label features with NO delay (D = 0)
  * random (non-chronological) split

Outputs (results/c3/): dose.csv, overlap.csv, dose_response.png, inflation.png
"""

from __future__ import annotations

import os
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

from experiments.common import PRIMARY_BUDGET, make_model, out_dir, ring_rows, score_all
from src.data_generator import GeneratorConfig
from src.pipeline import FEATURE_FAMILIES, build_dataset, fit_score, random_split, tabular, with_label_delay
from src.temporal_gnn import TemporalIndex

REUSE = [0.0, 0.25, 0.5, 0.75, 1.0]
REALISTIC = ["Transaction Baseline", "Combined (full graph)", "Combined + delayed labels",
             "Neighbour-Aggregation XGBoost", "Temporal GAT", "GraphSAGE (entity, inductive)"]


def _overlap(ds) -> dict:
    """Share of test fraud transactions whose device / IP / account was used by fraud in training."""
    trf, tef = ds.train[ds.train.fraud_label == 1], ds.test[ds.test.fraud_label == 1]
    return {f"seen_bad_{c.split('_')[0]}": float(tef[c].isin(set(trf[c])).mean())
            for c in ["device_id", "ip_id", "customer_id"]}


def run(seeds=(42, 43, 44)) -> None:
    out = out_dir("c3")
    t0 = time.time()
    rows, ov = [], []
    for p in REUSE:
        for seed in seeds:
            print(f"c3: reuse={p} seed={seed}", flush=True)
            ds = build_dataset(seed, GeneratorConfig(infra_reuse=p), label_delay_days=30.0)
            ov.append({"reuse": p, "seed": seed, **_overlap(ds)})
            idx = TemporalIndex(ds.tx, k=20)

            window_start = ds.test["timestamp"].min()

            def record(name, protocol, dset, res):
                sc = score_all(dset, res)
                rt = ring_rows(dset, res["p_test"], sc["thr"][PRIMARY_BUDGET])
                # like-for-like: PR-AUC on test rows inside the chronological test window only
                # (random splits otherwise also test early, cold-start rows -- a composition artefact)
                w = (dset.test["timestamp"] >= window_start).to_numpy()
                yw = dset.test["fraud_label"].to_numpy()[w]
                rows.append({"reuse": p, "seed": seed, "model": name, "protocol": protocol,
                             "pr_auc": sc["row"]["pr_auc"], "recall_at_1pct_fpr": sc["row"]["recall_at_1pct_fpr"],
                             "pr_auc_test_window": float(average_precision_score(yw, res["p_test"][w])),
                             "ring_amount_lost_share": float(rt.amount_before_alert.sum() / rt.ring_amount.sum())
                             if len(rt) else np.nan})

            for name in REALISTIC:
                record(name, "realistic", ds, fit_score(make_model(name, ds, seed, None, idx), ds))
            # published-style variants
            record("GraphSAGE (transductive)", "transductive graph", ds,
                   fit_score(make_model("GraphSAGE (transductive)", ds, seed), ds))
            ds0 = with_label_delay(ds, 0.0)
            record("Combined + labels (no delay)", "no label delay", ds0,
                   fit_score(tabular(FEATURE_FAMILIES["Combined + delayed labels"], seed), ds0))
            rs = random_split(ds0, seed)
            record("Combined + labels (no delay, random split)", "random split + no delay", rs,
                   fit_score(tabular(FEATURE_FAMILIES["Combined + delayed labels"], seed), rs))
            rs30 = random_split(ds, seed)
            record("Combined (full graph), random split", "random split", rs30,
                   fit_score(tabular(FEATURE_FAMILIES["Combined (full graph)"], seed), rs30))
    dose = pd.DataFrame(rows)
    dose.to_csv(os.path.join(out, "dose.csv"), index=False)
    pd.DataFrame(ov).groupby("reuse").mean(numeric_only=True).drop(columns="seed").to_csv(os.path.join(out, "overlap.csv"))
    _plots(dose, out)
    print(dose.groupby(["model", "reuse"])["pr_auc"].mean().unstack().round(4).to_string())
    print(f"c3 done in {(time.time() - t0) / 60:.1f} min")


def _plots(dose: pd.DataFrame, out: str) -> None:
    g = dose.groupby(["model", "protocol", "reuse"])["pr_auc_test_window"].agg(["mean", "std"]).reset_index()
    plt.figure(figsize=(9, 6))
    for (m, prot), x in g.groupby(["model", "protocol"]):
        ls = "-" if prot == "realistic" else "--"
        plt.errorbar(x.reuse, x["mean"], yerr=x["std"], ls=ls, marker="o", ms=4, capsize=2, label=m)
    plt.xlabel("Infrastructure reuse probability p (0 = fresh rings, 1 = fixed attacker pool)")
    plt.ylabel("PR-AUC on the chronological test window")
    plt.title("Dose-response: solid = leakage-safe protocol, dashed = published-style protocol")
    plt.legend(fontsize=7)
    plt.tight_layout()
    plt.savefig(os.path.join(out, "dose_response.png"), dpi=130)
    plt.close()

    m = dose.groupby(["model", "reuse"])["pr_auc_test_window"].mean().unstack(0)
    pairs = [("GraphSAGE (transductive)", "GraphSAGE (entity, inductive)"),
             ("Combined + labels (no delay)", "Combined + delayed labels"),
             ("Combined + labels (no delay, random split)", "Combined + delayed labels"),
             ("Combined (full graph), random split", "Combined (full graph)")]
    plt.figure(figsize=(8, 5))
    for leaky, safe in pairs:
        plt.plot(m.index, (m[leaky] - m[safe]) * 100, marker="o", label=f"{leaky}  vs  {safe}")
    plt.axhline(0, color="k", lw=0.8)
    plt.xlabel("Infrastructure reuse probability p")
    plt.ylabel("Inflation of PR-AUC (pp)")
    plt.legend(fontsize=7)
    plt.tight_layout()
    plt.savefig(os.path.join(out, "inflation.png"), dpi=130)
    plt.close()
