"""Model evaluation metrics and visualizations."""

from __future__ import annotations

import os
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    auc,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "results")


def compute_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5) -> Dict[str, float]:
    y_pred = (y_prob >= threshold).astype(int)
    metrics = {
        "roc_auc": float(roc_auc_score(y_true, y_prob)) if len(np.unique(y_true)) > 1 else 0.0,
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
    }
    prec, rec, _ = precision_recall_curve(y_true, y_prob)
    metrics["pr_auc"] = float(auc(rec, prec))
    metrics["recall_at_1pct_fpr"] = recall_at_fpr(y_true, y_prob, target_fpr=0.01)
    metrics["threshold"] = threshold
    return metrics


def recall_at_fpr(y_true: np.ndarray, y_prob: np.ndarray, target_fpr: float = 0.01) -> float:
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    idx = np.searchsorted(fpr, target_fpr, side="right") - 1
    idx = max(0, min(idx, len(tpr) - 1))
    return float(tpr[idx])


def false_positive_rate(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5) -> float:
    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0


def evaluate_subset_recall(
    df: pd.DataFrame, y_prob: np.ndarray, mask_col: str, label_col: str = "fraud_label"
) -> float:
    work = df.reset_index(drop=True)
    subset = work[work[mask_col] == 1]
    if len(subset) == 0:
        return 0.0
    pos = subset.index.to_numpy()
    y_true = subset[label_col].values
    y_p = y_prob[pos]
    y_pred = (y_p >= 0.5).astype(int)
    return float(recall_score(y_true, y_pred, zero_division=0))


def build_results_table(all_metrics: Dict[str, Dict[str, float]]) -> pd.DataFrame:
    rows = []
    for name, m in all_metrics.items():
        rows.append(
            {
                "Model": name,
                "ROC-AUC": m["roc_auc"],
                "PR-AUC": m["pr_auc"],
                "Precision": m["precision"],
                "Recall": m["recall"],
                "F1": m["f1"],
                "Recall @ 1% FPR": m["recall_at_1pct_fpr"],
            }
        )
    return pd.DataFrame(rows)


def plot_roc_curves(
    y_true: np.ndarray, predictions: Dict[str, np.ndarray], output_path: str
) -> None:
    plt.figure(figsize=(8, 6))
    for name, y_prob in predictions.items():
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f"{name} (AUC={roc_auc:.3f})")
    plt.plot([0, 1], [0, 1], "k--", alpha=0.4)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curves")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=120)
    plt.close()


def plot_pr_curves(
    y_true: np.ndarray, predictions: Dict[str, np.ndarray], output_path: str
) -> None:
    plt.figure(figsize=(8, 6))
    for name, y_prob in predictions.items():
        prec, rec, _ = precision_recall_curve(y_true, y_prob)
        pr_auc = auc(rec, prec)
        plt.plot(rec, prec, label=f"{name} (PR-AUC={pr_auc:.3f})")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curves")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=120)
    plt.close()


def plot_model_comparison(table: pd.DataFrame, output_path: str) -> None:
    metrics = ["ROC-AUC", "PR-AUC", "F1", "Recall @ 1% FPR"]
    plot_df = table.melt(id_vars=["Model"], value_vars=metrics, var_name="Metric", value_name="Score")
    plt.figure(figsize=(10, 6))
    sns.barplot(data=plot_df, x="Metric", y="Score", hue="Model")
    plt.title("Model Comparison")
    plt.xticks(rotation=15)
    plt.tight_layout()
    plt.savefig(output_path, dpi=120)
    plt.close()


def plot_confusion_matrix(
    y_true: np.ndarray, y_prob: np.ndarray, output_path: str, title: str = "Combined Model"
) -> None:
    y_pred = (y_prob >= 0.5).astype(int)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Legit", "Fraud"], yticklabels=["Legit", "Fraud"])
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title(f"Confusion Matrix — {title}")
    plt.tight_layout()
    plt.savefig(output_path, dpi=120)
    plt.close()


def plot_graph_hidden_fraud(
    recalls: Dict[str, float], output_path: str
) -> None:
    names = list(recalls.keys())
    values = [recalls[n] for n in names]
    plt.figure(figsize=(7, 5))
    bars = plt.bar(names, values, color=["#4C72B0", "#55A868", "#C44E52", "#8172B2"][: len(names)])
    plt.ylim(0, 1)
    plt.ylabel("Recall on Graph-Hidden Fraud")
    plt.title("Graph-Hidden Fraud Detection")
    for bar, v in zip(bars, values):
        plt.text(bar.get_x() + bar.get_width() / 2, v + 0.02, f"{v:.1%}", ha="center")
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    plt.savefig(output_path, dpi=120)
    plt.close()


def plot_relationship_network(tx_row: pd.Series, output_path: str) -> None:
    """Visualize local neighborhood for an example suspicious transaction."""
    import networkx as nx

    G = nx.Graph()
    center = tx_row["transaction_id"]
    entities = [
        ("Customer", tx_row["customer_id"]),
        ("Device", tx_row["device_id"]),
        ("Card", tx_row["card_id"]),
        ("IP", tx_row["ip_id"]),
        ("Merchant", tx_row["merchant_id"]),
    ]
    G.add_node(center, ntype="transaction")
    for label, eid in entities:
        G.add_node(eid, ntype=label.lower())
        G.add_edge(center, eid)

    # star from device to suggest farm pattern
    if tx_row.get("device_unique_customers", 0) > 3:
        for i in range(min(5, int(tx_row.get("device_unique_customers", 5)) - 1)):
            extra = f"C_extra_{i}"
            G.add_node(extra, ntype="customer")
            G.add_edge(tx_row["device_id"], extra)

    pos = nx.spring_layout(G, seed=42)
    colors = []
    for n in G.nodes():
        ntype = G.nodes[n].get("ntype", "")
        color_map = {
            "transaction": "#E45756",
            "customer": "#4C78A8",
            "device": "#F58518",
            "card": "#54A24B",
            "ip": "#B279A2",
            "merchant": "#72B7B2",
        }
        colors.append(color_map.get(ntype, "#999999"))

    plt.figure(figsize=(9, 7))
    nx.draw_networkx_nodes(G, pos, node_color=colors, node_size=600, alpha=0.9)
    nx.draw_networkx_edges(G, pos, alpha=0.5)
    labels = {n: (n if len(n) <= 10 else n[:8] + "..") for n in G.nodes()}
    nx.draw_networkx_labels(G, pos, labels, font_size=8)
    plt.title(
        f"Relationship Network — {center}\n"
        f"(device customers: {tx_row.get('device_unique_customers', 'N/A')}, "
        f"network failure: {tx_row.get('network_failure_rate', 0):.2f})"
    )
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(output_path, dpi=120)
    plt.close()


def generate_conclusion(table: pd.DataFrame, extras: Dict[str, float]) -> str:
    baseline = table[table["Model"].str.contains("Baseline", case=False)].iloc[0]
    rel = table[table["Model"].str.contains("Relationship", case=False)].iloc[0]
    combined = table[table["Model"].str.contains("Combined", case=False)].iloc[0]

    inc_pr = (combined["PR-AUC"] - baseline["PR-AUC"]) * 100
    lines = [
        "=" * 52,
        "RELATIONSHIP INTELLIGENCE EXPERIMENT",
        "=" * 52,
        "",
        f"Baseline PR-AUC:       {baseline['PR-AUC']:.4f}",
        f"Relationship PR-AUC:  {rel['PR-AUC']:.4f}",
        f"Combined PR-AUC:      {combined['PR-AUC']:.4f}",
        "",
        f"Baseline Recall@1% FPR:       {baseline['Recall @ 1% FPR']:.1%}",
        f"Combined Recall@1% FPR:       {combined['Recall @ 1% FPR']:.1%}",
        "",
        f"Graph-hidden fraud recall:",
        f"  Baseline: {extras.get('graph_hidden_baseline', 0):.1%}",
        f"  Combined: {extras.get('graph_hidden_combined', 0):.1%}",
        "",
        f"Legitimate shared-device FPR: {extras.get('legitimate_shared_fpr', 0):.1%}",
        "",
        f"Incremental PR-AUC improvement: {inc_pr:+.2f} percentage points",
        "",
        "Conclusion:",
    ]

    if inc_pr > 1 and extras.get("graph_hidden_combined", 0) > extras.get("graph_hidden_baseline", 0):
        lines.append(
            "Relationship context adds measurable predictive value beyond transaction-level "
            "features, especially for fraud that appears normal in isolation (graph-hidden fraud)."
        )
    elif inc_pr > 0:
        lines.append(
            "Relationship features provide modest incremental value; combined modeling improves "
            "detection but transaction features remain dominant."
        )
    else:
        lines.append(
            "In this synthetic setup, relationship features alone did not outperform the baseline; "
            "review feature engineering or fraud pattern strength."
        )

    lines.append("=" * 52)
    return "\n".join(lines)
