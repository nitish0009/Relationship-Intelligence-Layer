"""Main experiment: all models x seeds, C1 ring-level early detection, G4 segments, G6 entity age.

Outputs (results/main/):
  metrics_by_seed.csv, metrics_summary.csv    row-level metrics (val-tuned F1 threshold)
  ring_tables.csv, ring_summary.csv            C1 per ring / pooled over seeds, per alert budget
  ring_by_sophistication.csv                   C1 split by ring hidden share
  warmup.csv, warmup.png                       recall on the k-th transaction of a ring
  segment_fpr.csv, entity_age.csv              G4 / G6 at the primary alert budget
  graph_hidden_recall.csv, tuned_params.json, run_info.json, *.png
"""

from __future__ import annotations

import os
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from experiments.common import (BUDGETS, GNNS, PRIMARY_BUDGET, TABULAR, Timer, make_model, out_dir, ring_rows,
                                save_json, score_all, tune_model)
from src.classifier_utils import classifier_name
from src.data_generator import DATA_DIR
from src.evaluation import subset_recall
from src.graph_model import TabularRiskModel
from src.pipeline import build_dataset, fit_score
from src.preprocessing import INTRINSIC_FEATURE_COLS
from src.ring_metrics import warmup_curve
from src.temporal_gnn import TemporalIndex

MODELS = TABULAR + ["Neighbour-Aggregation XGBoost"] + GNNS + ["GraphSAGE (entity, inductive)"]


def _models():
    from experiments.common import ORIGINALS, originals_enabled
    return MODELS + (ORIGINALS if originals_enabled() else [])


def _entity_age_buckets(ds) -> pd.Series:
    created = ds.extras["customers"].set_index("customer_id")["created_at"]
    age_days = (ds.test["timestamp"] - pd.to_datetime(ds.test["customer_id"].map(created))).dt.total_seconds() / 86400
    return pd.cut(age_days, [-np.inf, 1, 7, 30, np.inf], labels=["account <1d", "account 1-7d", "account 7-30d",
                                                                    "account >30d"]).astype(str)


def run(seeds=(42, 43, 44, 45, 46)) -> None:
    out = out_dir("main")
    t0 = time.time()
    rows, ring_tabs, warm, seg, age, gh = [], [], [], [], [], []
    gnn_params: dict = {}
    tuned_log: dict = {}
    for si, seed in enumerate(seeds):
        print(f"\n=== main: seed {seed} ===", flush=True)
        ds = build_dataset(seed, label_delay_days=30.0, save_dir=DATA_DIR if si == 0 else None)
        idx = TemporalIndex(ds.tx, k=20)
        yt = ds.test["fraud_label"].to_numpy()
        age_bucket = _entity_age_buckets(ds).to_numpy()
        first_dev = (ds.test["device_transaction_count"] == 0).to_numpy()
        tuned_log[seed] = {}
        for name in _models():
            with Timer(name):
                # every model is tuned on EVERY seed's validation split (original-code models use
                # the authors' published defaults)
                params = tune_model(name, ds, seed, idx)
                tuned_log[seed][name] = params
                res = fit_score(make_model(name, ds, seed, params, idx), ds)
            sc = score_all(ds, res)
            rows.append({"seed": seed, "model": name, **sc["row"]})
            gh.append({"seed": seed, "model": name,
                       "graph_hidden_recall": subset_recall(ds.test, res["p_test"], "graph_hidden_fraud", res["threshold"])})
            for b in BUDGETS:
                tab = ring_rows(ds, res["p_test"], sc["thr"][b], seed=seed, model=name, budget=b)
                ring_tabs.append(tab)
            thr = sc["thr"][PRIMARY_BUDGET]
            alert = res["p_test"] >= thr
            w = warmup_curve(ds.test, res["p_test"], thr)
            w["seed"], w["model"] = seed, name
            warm.append(w)
            for s in ["residential", "carrier_nat", "household", "public_wifi"]:
                m = (ds.test["segment"] == s).to_numpy()
                seg.append({"seed": seed, "model": name, "segment": s, "fpr": float(alert[m].mean()), "n": int(m.sum())})
            for label, mask in [*[(b, age_bucket == b) for b in np.unique(age_bucket)], ("first-seen device", first_dev)]:
                f, l = mask & (yt == 1), mask & (yt == 0)
                age.append({"seed": seed, "model": name, "bucket": label, "recall": float(alert[f].mean()) if f.any() else np.nan,
                            "fpr": float(alert[l].mean()) if l.any() else np.nan, "n_fraud": int(f.sum()), "n_legit": int(l.sum())})
        # graph-hidden sanity check (intrinsic attributes only)
        intr = TabularRiskModel(INTRINSIC_FEATURE_COLS, seed=seed).fit(ds.train, ds.train["fraud_label"].to_numpy())
        hm = ((ds.test["graph_hidden_fraud"] == 1) | (ds.test["fraud_label"] == 0)).to_numpy()
        gh.append({"seed": seed, "model": "_intrinsic_sanity_auc",
                   "graph_hidden_recall": float(roc_auc_score(yt[hm], intr.predict_proba(ds.test[hm])))})

    metrics = pd.DataFrame(rows)
    metrics.to_csv(os.path.join(out, "metrics_by_seed.csv"), index=False)
    summary = metrics.groupby("model", sort=False)[["roc_auc", "pr_auc", "recall_at_1pct_fpr", "f1"]].agg(["mean", "std"])
    summary.columns = [f"{a}_{b}" for a, b in summary.columns]
    summary.to_csv(os.path.join(out, "metrics_summary.csv"))

    rt = pd.concat([t for t in ring_tabs if not t.empty], ignore_index=True)
    rt.to_csv(os.path.join(out, "ring_tables.csv"), index=False)
    ring_summary = _pool_rings(rt, ["model", "budget"])
    ring_summary.to_csv(os.path.join(out, "ring_summary.csv"), index=False)
    rt["sophistication"] = pd.cut(rt["hidden_share"], [-0.01, 0.33, 0.67, 1.0],
                                  labels=["crude (<33% hidden)", "mixed", "camouflaged (>67% hidden)"]).astype(str)
    _pool_rings(rt[rt.budget == PRIMARY_BUDGET], ["model", "sophistication"]).to_csv(
        os.path.join(out, "ring_by_sophistication.csv"), index=False)

    wu = pd.concat(warm, ignore_index=True)
    wu = wu.groupby(["model", "k"]).apply(lambda g: pd.Series({"recall": np.average(g.recall, weights=g.n),
                                                               "n": g.n.sum()}), include_groups=False).reset_index()
    wu.to_csv(os.path.join(out, "warmup.csv"), index=False)
    pd.DataFrame(seg).groupby(["model", "segment"], sort=False).agg(fpr=("fpr", "mean"), n=("n", "sum")).reset_index() \
        .to_csv(os.path.join(out, "segment_fpr.csv"), index=False)
    pd.DataFrame(age).groupby(["model", "bucket"], sort=False).agg(
        recall=("recall", "mean"), fpr=("fpr", "mean"), n_fraud=("n_fraud", "sum"), n_legit=("n_legit", "sum")
    ).reset_index().to_csv(os.path.join(out, "entity_age.csv"), index=False)
    pd.DataFrame(gh).groupby("model", sort=False)["graph_hidden_recall"].agg(["mean", "std"]).to_csv(
        os.path.join(out, "graph_hidden_recall.csv"))
    save_json({str(k): v for k, v in tuned_log.items()}, os.path.join(out, "tuned_params.json"))
    save_json({"seeds": list(seeds), "classifier": classifier_name(), "budgets": BUDGETS,
               "primary_budget": PRIMARY_BUDGET, "minutes": (time.time() - t0) / 60}, os.path.join(out, "run_info.json"))
    _plots(out, summary, ring_summary, wu)
    print(summary.round(4).to_string())
    print(ring_summary[ring_summary.budget == PRIMARY_BUDGET].round(3).to_string(index=False))
    print(f"main done in {(time.time() - t0) / 60:.1f} min")


def _pool_rings(rt: pd.DataFrame, by) -> pd.DataFrame:
    def agg(g):
        return pd.Series({
            "n_rings": len(g),
            "ring_recall": g["detected"].mean(),
            "median_txns_before_alert": g["txns_before_alert"].median(),
            "mean_txns_before_alert": g["txns_before_alert"].mean(),
            "amount_lost_share": g["amount_before_alert"].sum() / g["ring_amount"].sum(),
            "median_hours_to_alert": g["hours_to_alert"].median(),
            "prevented_takedown": (g["prevented_takedown"] * g["ring_amount"]).sum() / g["ring_amount"].sum(),
            "prevented_blocking": (g["prevented_blocking"] * g["ring_amount"]).sum() / g["ring_amount"].sum(),
        })
    return rt.groupby(by, sort=False).apply(agg, include_groups=False).reset_index()


def _plots(out, summary, ring_summary, wu) -> None:
    order = summary["pr_auc_mean"].sort_values().index
    plt.figure(figsize=(8, 6))
    plt.barh(order, summary.loc[order, "pr_auc_mean"], xerr=summary.loc[order, "pr_auc_std"], color="#4C72B0")
    plt.xlabel("Test PR-AUC (mean ± std over seeds)")
    plt.xlim(0.5, 1)
    plt.tight_layout()
    plt.savefig(os.path.join(out, "model_comparison.png"), dpi=130)
    plt.close()

    plt.figure(figsize=(9, 5))
    for m in wu["model"].unique():
        g = wu[wu.model == m]
        plt.plot(g.k, g.recall, marker="o", ms=3, label=m)
    plt.xlabel("k-th transaction of the ring")
    plt.ylabel(f"Recall at {PRIMARY_BUDGET:.1%} alert budget")
    plt.title("Ring warm-up: how quickly each model recognises a new ring")
    plt.legend(fontsize=7, ncol=2)
    plt.tight_layout()
    plt.savefig(os.path.join(out, "warmup.png"), dpi=130)
    plt.close()

    piv = ring_summary.pivot(index="model", columns="budget", values="amount_lost_share")
    piv = piv.loc[piv[PRIMARY_BUDGET].sort_values(ascending=False).index]
    piv.plot.barh(figsize=(8, 6), color=["#C44E52", "#DD8452", "#55A868"])
    plt.xlabel("Share of ring money lost before the first alert")
    plt.legend(title="alert budget (val FPR)", labels=[f"{b:.1%}" for b in piv.columns])
    plt.tight_layout()
    plt.savefig(os.path.join(out, "ring_money_lost.png"), dpi=130)
    plt.close()
