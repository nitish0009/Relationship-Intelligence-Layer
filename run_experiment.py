#!/usr/bin/env python3
"""One-command reproducible Relationship Intelligence experiment."""

from __future__ import annotations

import os
import sys
import time

import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

from src.baseline_model import BaselineModel
from src.classifier_utils import classifier_name
from src.data_generator import DATA_DIR, generate_all
from src.evaluation import (
    RESULTS_DIR,
    build_results_table,
    compute_metrics,
    evaluate_subset_recall,
    false_positive_rate,
    generate_conclusion,
    plot_confusion_matrix,
    plot_graph_hidden_fraud,
    plot_model_comparison,
    plot_pr_curves,
    plot_relationship_network,
    plot_roc_curves,
)
from src.graph_features import (
    GRAPH_FEATURE_COLS,
    SIMPLE_COUNT_COLS,
    TEMPORAL_EXTRA_COLS,
    build_graph_features,
    get_ablation_feature_sets,
)
from src.graph_model import CombinedModel, GraphSAGEModel, RelationshipModel, TabularRiskModel
from src.preprocessing import (
    TRANSACTION_FEATURE_COLS,
    build_transaction_features,
    chronological_split,
)

MODEL_DIR = os.path.join(RESULTS_DIR, "models")


def run_ablation(train_df, val_df, test_df, y_train, y_test, test_probs_dict) -> pd.DataFrame:
    """Ablation: transaction + counts + temporal + full graph."""
    ablation_sets = get_ablation_feature_sets()
    rows = []
    for name, graph_cols in ablation_sets.items():
        if name == "transaction_only":
            model = BaselineModel()
        else:
            feats = TRANSACTION_FEATURE_COLS + graph_cols
            model = TabularRiskModel(feats, name=name)
        model.fit(train_df, y_train)
        y_prob = model.predict_proba(test_df)
        m = compute_metrics(y_test, y_prob)
        rows.append({"Ablation": name, **m})
        test_probs_dict[name] = y_prob
    return pd.DataFrame(rows)


def main() -> None:
    t0 = time.time()
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(MODEL_DIR, exist_ok=True)

    print("=" * 52)
    print("Step 1/6 — Generating synthetic dataset")
    print("=" * 52)
    entities = generate_all(DATA_DIR)

    print("\nStep 2/6 — Feature engineering (no temporal leakage)")
    tx = entities["transactions"]
    tx = build_transaction_features(tx)
    print("  Building temporal graph features (this may take a few minutes)...")
    tx = build_graph_features(tx)

    train_df, val_df, test_df = chronological_split(tx)
    print(f"  Split sizes — train: {len(train_df):,}, val: {len(val_df):,}, test: {len(test_df):,}")
    print(f"  Test fraud rate: {test_df['fraud_label'].mean():.2%}")

    y_train = train_df["fraud_label"].values
    y_val = val_df["fraud_label"].values
    y_test = test_df["fraud_label"].values

    print("\nStep 3/6 — Training models")
    print(f"  Using classifier: {classifier_name()}")
    baseline = BaselineModel()
    baseline.fit(train_df, y_train)
    baseline.save(os.path.join(MODEL_DIR, "baseline.joblib"))

    relationship = RelationshipModel()
    relationship.fit(train_df, y_train)
    relationship.save(os.path.join(MODEL_DIR, "relationship.joblib"))

    combined = CombinedModel()
    combined.fit(train_df, y_train)
    combined.save(os.path.join(MODEL_DIR, "combined.joblib"))

    predictions = {
        "Transaction Baseline": baseline.predict_proba(test_df),
        "Relationship Intelligence Model": relationship.predict_proba(test_df),
        "Combined Transaction + Relationship Model": combined.predict_proba(test_df),
    }

    # GraphSAGE (lightweight, no PyG required)
    print("  Training GraphSAGE model...")
    try:
        graphsage = GraphSAGEModel()
        graphsage.fit(train_df, y_train)
        predictions["GraphSAGE Model"] = graphsage.predict_proba(test_df)
        graphsage.save(os.path.join(MODEL_DIR, "graphsage.joblib"))
    except Exception as e:
        print(f"  GraphSAGE skipped: {e}")

    print("\nStep 4/6 — Evaluation")
    all_metrics = {}
    for name, y_prob in predictions.items():
        all_metrics[name] = compute_metrics(y_test, y_prob)

    results_table = build_results_table(all_metrics)
    results_table.to_csv(os.path.join(RESULTS_DIR, "metrics_table.csv"), index=False)
    print("\n" + results_table.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    # Graph-hidden fraud recall
    gh_recalls = {}
    for key, label in [
        ("graph_hidden_baseline", "Transaction Baseline"),
        ("graph_hidden_combined", "Combined Transaction + Relationship Model"),
    ]:
        if label in predictions:
            gh_recalls[label] = evaluate_subset_recall(
                test_df,
                predictions[label],
                "graph_hidden_fraud",
            )
    if "GraphSAGE Model" in predictions:
        gh_recalls["GraphSAGE Model"] = evaluate_subset_recall(
            test_df, predictions["GraphSAGE Model"], "graph_hidden_fraud"
        )

    # Legitimate shared environment FPR (combined model)
    test_reset = test_df.reset_index(drop=True)
    legit_fpr = false_positive_rate(
        test_reset.loc[test_reset["legitimate_shared_env"] == 1, "fraud_label"].values,
        predictions["Combined Transaction + Relationship Model"][
            test_reset["legitimate_shared_env"] == 1
        ],
    )

    # Ablation
    print("\nStep 5/6 — Ablation study")
    ablation_probs = {}
    ablation_df = run_ablation(train_df, val_df, test_df, y_train, y_test, ablation_probs)
    ablation_df.to_csv(os.path.join(RESULTS_DIR, "ablation_metrics.csv"), index=False)
    print(ablation_df[["Ablation", "pr_auc", "recall_at_1pct_fpr"]].to_string(index=False))

    # Visualizations
    print("\nStep 6/6 — Generating plots")
    plot_roc_curves(y_test, predictions, os.path.join(RESULTS_DIR, "roc_curve.png"))
    plot_pr_curves(y_test, predictions, os.path.join(RESULTS_DIR, "precision_recall_curve.png"))
    plot_model_comparison(results_table, os.path.join(RESULTS_DIR, "model_comparison.png"))
    plot_confusion_matrix(
        y_test,
        predictions["Combined Transaction + Relationship Model"],
        os.path.join(RESULTS_DIR, "confusion_matrix.png"),
    )
    plot_graph_hidden_fraud(gh_recalls, os.path.join(RESULTS_DIR, "graph_hidden_fraud.png"))

    # Example suspicious tx for network viz
    suspicious = test_df[
        (test_df["fraud_label"] == 1) & (test_df["device_unique_customers"] > test_df["device_unique_customers"].quantile(0.9))
    ]
    if len(suspicious) > 0:
        example = suspicious.iloc[0]
        plot_relationship_network(example, os.path.join(RESULTS_DIR, "relationship_network.png"))

    extras = {
        "graph_hidden_baseline": gh_recalls.get("Transaction Baseline", 0),
        "graph_hidden_combined": gh_recalls.get("Combined Transaction + Relationship Model", 0),
        "legitimate_shared_fpr": legit_fpr,
    }
    conclusion = generate_conclusion(results_table, extras)
    print("\n" + conclusion)

    elapsed = time.time() - t0
    print(f"\nExperiment completed in {elapsed / 60:.1f} minutes.")
    print(f"Results saved to {RESULTS_DIR}/")


if __name__ == "__main__":
    main()
