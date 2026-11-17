#!/usr/bin/env python3
"""One-command reproducible Relationship Intelligence experiment.

    python run_experiment.py                 # single run, seed 42, writes results/
    python run_experiment.py --seeds 42 43 44 45 46   # + multi-seed stability table
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

from src.baseline_model import BaselineModel
from src.classifier_utils import classifier_name
from src.data_generator import DATA_DIR, generate_all
from src.evaluation import (
    RESULTS_DIR,
    bootstrap_ci,
    bootstrap_delta_ci,
    build_results_table,
    cluster_bootstrap_indices,
    compute_metrics,
    false_positive_rate,
    generate_conclusion,
    plot_confusion_matrix,
    plot_graph_hidden_fraud,
    plot_model_comparison,
    plot_pr_curves,
    plot_relationship_network,
    plot_roc_curves,
    select_threshold,
    subset_recall,
)
from src.explainability import explain_transaction, fit_signal_thresholds
from src.graph_features import build_graph_features, get_ablation_feature_sets
from src.graph_model import CombinedModel, GraphSAGEModel, RelationshipModel, TabularRiskModel
from src.preprocessing import (
    INTRINSIC_FEATURE_COLS,
    TRANSACTION_FEATURE_COLS,
    build_transaction_features,
    chronological_split,
)

BASELINE = "Transaction Baseline"
COMBINED = "Combined Transaction + Relationship Model"


def fit_and_score(model, train_df, val_df, test_df):
    """Fit on train; score val (history = train) and test (history = train + val)."""
    y_train = train_df["fraud_label"].to_numpy()
    model.fit(train_df, y_train)
    p_val = model.predict_proba(val_df, history=train_df)
    p_test = model.predict_proba(test_df, history=pd.concat([train_df, val_df]))
    threshold = select_threshold(val_df["fraud_label"].to_numpy(), p_val)
    return p_test, threshold


def run(seed: int, save: bool, n_boot: int = 1000) -> dict:
    t0 = time.time()
    log = print if save else (lambda *a, **k: None)

    log("Step 1/6 — Generating synthetic dataset")
    entities = generate_all(DATA_DIR if save else None, seed=seed)

    log("Step 2/6 — Feature engineering (strictly past data only)")
    tx = build_graph_features(build_transaction_features(entities["transactions"]))
    train_df, val_df, test_df = chronological_split(tx)
    y_test = test_df["fraud_label"].to_numpy()
    test_rings = set(test_df.loc[test_df.fraud_label == 1, "ring_id"])
    train_rings = set(train_df.loc[train_df.fraud_label == 1, "ring_id"])
    log(f"  Split sizes — train {len(train_df):,} | val {len(val_df):,} | test {len(test_df):,}")
    log(f"  Fraud rate — train {train_df.fraud_label.mean():.2%} | val {val_df.fraud_label.mean():.2%} | "
        f"test {test_df.fraud_label.mean():.2%}")
    log(f"  Test fraud rings: {len(test_rings)} ({len(test_rings & train_rings)} also active in train)")

    log(f"Step 3/6 — Training models ({classifier_name()}, thresholds tuned on validation)")
    models = {
        BASELINE: BaselineModel(seed),
        "Relationship Intelligence Model": RelationshipModel(seed),
        COMBINED: CombinedModel(seed),
        "GraphSAGE Model": GraphSAGEModel(seed=seed),
    }
    preds, thresholds = {}, {}
    for name, model in models.items():
        preds[name], thresholds[name] = fit_and_score(model, train_df, val_df, test_df)
        log(f"  {name}: done")

    log("Step 4/6 — Evaluation (ring-clustered bootstrap CIs)")
    boot = cluster_bootstrap_indices(test_df, n_boot=n_boot, seed=seed)
    all_metrics = {}
    for name, p in preds.items():
        m = compute_metrics(y_test, p, thresholds[name])
        m["pr_auc_lo"], m["pr_auc_hi"] = bootstrap_ci(y_test, p, boot)
        all_metrics[name] = m
    table = build_results_table(all_metrics)
    delta_ci = bootstrap_delta_ci(y_test, preds[BASELINE], preds[COMBINED], boot)

    gh_recall = {n: subset_recall(test_df, p, "graph_hidden_fraud", thresholds[n]) for n, p in preds.items()}
    shared = test_df["legitimate_shared_env"].to_numpy() == 1
    legit_fpr = {
        n: false_positive_rate(y_test[shared], p[shared], thresholds[n]) for n, p in preds.items()
    }

    # Sanity check: graph-hidden fraud must be indistinguishable on intrinsic attributes alone.
    intrinsic = TabularRiskModel(INTRINSIC_FEATURE_COLS, name="intrinsic", seed=seed)
    intrinsic.fit(train_df, train_df["fraud_label"].to_numpy())
    hid_mask = (test_df["graph_hidden_fraud"] == 1) | (test_df["fraud_label"] == 0)
    hidden_auc = float(roc_auc_score(y_test[hid_mask.to_numpy()], intrinsic.predict_proba(test_df[hid_mask])))

    log("Step 5/6 — Ablation study")
    abl_rows = []
    for name, graph_cols in get_ablation_feature_sets().items():
        model = TabularRiskModel(TRANSACTION_FEATURE_COLS + graph_cols, name=name, seed=seed)
        p, thr = fit_and_score(model, train_df, val_df, test_df)
        abl_rows.append({"Ablation": name, **compute_metrics(y_test, p, thr)})
    ablation = pd.DataFrame(abl_rows)

    extras = {
        "delta_pr_auc_ci": delta_ci,
        "graph_hidden_recall": gh_recall,
        "legit_shared_fpr": legit_fpr,
        "hidden_intrinsic_auc": hidden_auc,
    }
    conclusion = generate_conclusion(table, ablation, extras)

    if save:
        log("Step 6/6 — Writing results and plots")
        os.makedirs(RESULTS_DIR, exist_ok=True)
        table.to_csv(os.path.join(RESULTS_DIR, "metrics_table.csv"), index=False)
        ablation.to_csv(os.path.join(RESULTS_DIR, "ablation_metrics.csv"), index=False)
        plot_roc_curves(y_test, preds, os.path.join(RESULTS_DIR, "roc_curve.png"))
        plot_pr_curves(y_test, preds, os.path.join(RESULTS_DIR, "precision_recall_curve.png"))
        plot_model_comparison(table, os.path.join(RESULTS_DIR, "model_comparison.png"))
        plot_confusion_matrix(y_test, preds[COMBINED], thresholds[COMBINED], os.path.join(RESULTS_DIR, "confusion_matrix.png"))
        plot_graph_hidden_fraud(gh_recall, os.path.join(RESULTS_DIR, "graph_hidden_fraud.png"))

        # Examples: graph-hidden test fraud where relationship context helps most
        # (largest combined − baseline score gap) -> real neighbourhood + explanations
        signal_thr = fit_signal_thresholds(train_df)
        test_reset = test_df.reset_index(drop=True)
        hidden_idx = np.flatnonzero(test_reset["graph_hidden_fraud"].to_numpy() == 1)
        gap = preds[COMBINED][hidden_idx] - preds[BASELINE][hidden_idx]
        top = hidden_idx[np.argsort(-gap, kind="stable")[:5]]
        rows = []
        for i in top:
            r = test_reset.iloc[i]
            rows.append({
                "transaction_id": r["transaction_id"], "fraud_pattern": r["fraud_pattern"],
                "baseline_score": round(float(preds[BASELINE][i]), 4),
                "combined_score": round(float(preds[COMBINED][i]), 4),
                "signals": " | ".join(explain_transaction(r, signal_thr)),
            })
        pd.DataFrame(rows).to_csv(os.path.join(RESULTS_DIR, "example_explanations.csv"), index=False)
        if len(top):
            plot_relationship_network(test_reset.iloc[top[0]], tx, os.path.join(RESULTS_DIR, "relationship_network.png"))

        summary = {
            "seed": seed,
            "classifier": classifier_name(),
            "n_transactions": int(len(tx)),
            "fraud_rate": float(tx.fraud_label.mean()),
            "split_sizes": [len(train_df), len(val_df), len(test_df)],
            "test_fraud_rings": len(test_rings),
            "test_rings_also_in_train": len(test_rings & train_rings),
            **{k: v for k, v in extras.items()},
        }
        with open(os.path.join(RESULTS_DIR, "run_summary.json"), "w") as f:
            json.dump(summary, f, indent=2)
        with open(os.path.join(RESULTS_DIR, "conclusion.txt"), "w") as f:
            f.write(conclusion + "\n")

        log("\n" + table.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
        log("\n" + ablation[["Ablation", "pr_auc", "recall_at_1pct_fpr", "f1"]].to_string(index=False))
        log("\n" + conclusion)
        log(f"\nCompleted in {(time.time() - t0) / 60:.1f} min. Results in {RESULTS_DIR}/")

    out = {"seed": seed}
    for name, m in all_metrics.items():
        out[f"{name} | PR-AUC"] = m["pr_auc"]
        out[f"{name} | Recall@1%FPR"] = m["recall_at_1pct_fpr"]
        out[f"{name} | Graph-hidden recall"] = gh_recall[name]
    out["Combined − Baseline PR-AUC"] = all_metrics[COMBINED]["pr_auc"] - all_metrics[BASELINE]["pr_auc"]
    out["Hidden-fraud intrinsic AUC"] = hidden_auc
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seeds", type=int, nargs="+", default=[42],
                    help="first seed is the primary run (saved to results/); extra seeds add a stability table")
    ap.add_argument("--n-boot", type=int, default=1000)
    args = ap.parse_args()

    rows = [run(args.seeds[0], save=True, n_boot=args.n_boot)]
    if len(args.seeds) > 1:
        for s in args.seeds[1:]:
            print(f"\nMulti-seed stability: running seed {s} ...")
            rows.append(run(s, save=False, n_boot=200))
        df = pd.DataFrame(rows).set_index("seed")
        df.to_csv(os.path.join(RESULTS_DIR, "multi_seed_metrics.csv"))
        agg = df.agg(["mean", "std", "min", "max"]).T
        agg.to_csv(os.path.join(RESULTS_DIR, "multi_seed_summary.csv"))
        print("\nMulti-seed summary (each seed = new dataset + new models):")
        print(agg.to_string(float_format=lambda x: f"{x:.4f}"))


if __name__ == "__main__":
    main()
