"""Metrics, threshold selection, uncertainty estimates and plots."""

from __future__ import annotations

import os
from typing import Dict, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")


# ----------------------------------------------------------------------------- metrics

def select_threshold(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """F1-maximising threshold, chosen on the validation set (never on test)."""
    prec, rec, thr = precision_recall_curve(y_true, y_prob)
    f1 = 2 * prec[:-1] * rec[:-1] / np.clip(prec[:-1] + rec[:-1], 1e-12, None)
    return float(thr[int(np.argmax(f1))]) if len(thr) else 0.5


def recall_at_fpr(y_true: np.ndarray, y_prob: np.ndarray, target_fpr: float = 0.01) -> float:
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    idx = max(0, int(np.searchsorted(fpr, target_fpr, side="right")) - 1)
    return float(tpr[idx])


def compute_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float) -> Dict[str, float]:
    y_pred = (y_prob >= threshold).astype(int)
    both = len(np.unique(y_true)) > 1
    return {
        "roc_auc": float(roc_auc_score(y_true, y_prob)) if both else float("nan"),
        "pr_auc": float(average_precision_score(y_true, y_prob)) if both else float("nan"),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "recall_at_1pct_fpr": recall_at_fpr(y_true, y_prob) if both else float("nan"),
        "threshold": float(threshold),
    }


def false_positive_rate(y_true: np.ndarray, y_prob: np.ndarray, threshold: float) -> float:
    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, _, _ = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return float(fp / (fp + tn)) if (fp + tn) else 0.0


def subset_recall(df: pd.DataFrame, y_prob: np.ndarray, mask_col: str, threshold: float) -> float:
    mask = df[mask_col].to_numpy() == 1
    y = df["fraud_label"].to_numpy()[mask]
    if y.sum() == 0:
        return float("nan")
    return float(recall_score(y, (y_prob[mask] >= threshold).astype(int), zero_division=0))


def build_results_table(all_metrics: Dict[str, Dict[str, float]]) -> pd.DataFrame:
    rows = []
    for name, m in all_metrics.items():
        rows.append(
            {
                "Model": name,
                "ROC-AUC": m["roc_auc"],
                "PR-AUC": m["pr_auc"],
                "PR-AUC 95% CI": f"[{m['pr_auc_lo']:.3f}, {m['pr_auc_hi']:.3f}]" if "pr_auc_lo" in m else "",
                "Precision": m["precision"],
                "Recall": m["recall"],
                "F1": m["f1"],
                "Recall @ 1% FPR": m["recall_at_1pct_fpr"],
                "Threshold (val)": m["threshold"],
            }
        )
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------- uncertainty

def cluster_bootstrap_indices(df: pd.DataFrame, n_boot: int = 1000, seed: int = 42) -> List[np.ndarray]:
    """Bootstrap resamples that respect fraud-ring clustering.

    Fraud transactions from the same ring are strongly correlated, so they are resampled
    as whole rings; legitimate transactions are resampled individually.
    """
    rng = np.random.default_rng(seed)
    labels = df["fraud_label"].to_numpy()
    rings = df["ring_id"].to_numpy() if "ring_id" in df.columns else np.where(labels == 1, 0, -1)
    legit_idx = np.flatnonzero(labels == 0)
    ring_ids = np.unique(rings[labels == 1])
    ring_members = [np.flatnonzero((rings == r) & (labels == 1)) for r in ring_ids]
    out = []
    for _ in range(n_boot):
        li = rng.choice(legit_idx, size=len(legit_idx), replace=True)
        picked = rng.integers(0, len(ring_members), size=len(ring_members))
        out.append(np.concatenate([li] + [ring_members[p] for p in picked]))
    return out


def bootstrap_ci(y_true: np.ndarray, y_prob: np.ndarray, boot_idx: List[np.ndarray], fn=average_precision_score):
    vals = np.array([fn(y_true[i], y_prob[i]) for i in boot_idx])
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def bootstrap_delta_ci(y_true, prob_a, prob_b, boot_idx, fn=average_precision_score):
    """Paired CI for fn(b) - fn(a) using identical resamples."""
    d = np.array([fn(y_true[i], prob_b[i]) - fn(y_true[i], prob_a[i]) for i in boot_idx])
    return float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))


# ----------------------------------------------------------------------------- plots

def plot_roc_curves(y_true: np.ndarray, predictions: Dict[str, np.ndarray], output_path: str) -> None:
    plt.figure(figsize=(8, 6))
    for name, y_prob in predictions.items():
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        plt.plot(fpr, tpr, label=f"{name} (AUC={roc_auc_score(y_true, y_prob):.3f})")
    plt.plot([0, 1], [0, 1], "k--", alpha=0.4)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curves (test)")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(output_path, dpi=120)
    plt.close()


def plot_pr_curves(y_true: np.ndarray, predictions: Dict[str, np.ndarray], output_path: str) -> None:
    plt.figure(figsize=(8, 6))
    for name, y_prob in predictions.items():
        prec, rec, _ = precision_recall_curve(y_true, y_prob)
        plt.plot(rec, prec, label=f"{name} (AP={average_precision_score(y_true, y_prob):.3f})")
    plt.axhline(y_true.mean(), color="k", ls="--", alpha=0.4, label=f"Prevalence ({y_true.mean():.1%})")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curves (test)")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(output_path, dpi=120)
    plt.close()


def plot_model_comparison(table: pd.DataFrame, output_path: str) -> None:
    metrics = ["ROC-AUC", "PR-AUC", "F1", "Recall @ 1% FPR"]
    plot_df = table.melt(id_vars=["Model"], value_vars=metrics, var_name="Metric", value_name="Score")
    plt.figure(figsize=(10, 6))
    sns.barplot(data=plot_df, x="Metric", y="Score", hue="Model")
    plt.ylim(0, 1)
    plt.title("Model Comparison (test, thresholds tuned on validation)")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(output_path, dpi=120)
    plt.close()


def plot_confusion_matrix(
    y_true: np.ndarray, y_prob: np.ndarray, threshold: float, output_path: str, title: str = "Combined Model"
) -> None:
    cm = confusion_matrix(y_true, (y_prob >= threshold).astype(int), labels=[0, 1])
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Legit", "Fraud"], yticklabels=["Legit", "Fraud"])
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title(f"Confusion Matrix — {title}\n(threshold {threshold:.3f} from validation)")
    plt.tight_layout()
    plt.savefig(output_path, dpi=120)
    plt.close()


def plot_graph_hidden_fraud(recalls: Dict[str, float], output_path: str) -> None:
    names = list(recalls.keys())
    values = [recalls[n] for n in names]
    plt.figure(figsize=(8, 5))
    bars = plt.bar(names, values, color=["#4C72B0", "#55A868", "#C44E52", "#8172B2"][: len(names)])
    plt.ylim(0, 1.05)
    plt.ylabel("Recall on graph-hidden fraud")
    plt.title("Graph-Hidden Fraud Detection (test, validation-tuned thresholds)")
    for bar, v in zip(bars, values):
        plt.text(bar.get_x() + bar.get_width() / 2, v + 0.02, f"{v:.1%}", ha="center")
    plt.xticks(rotation=15, ha="right", fontsize=8)
    plt.tight_layout()
    plt.savefig(output_path, dpi=120)
    plt.close()


def plot_relationship_network(tx_row: pd.Series, history: pd.DataFrame, output_path: str, max_neighbours: int = 12) -> None:
    """Draw the *actual* past neighbourhood of a transaction (no synthetic nodes)."""
    import networkx as nx

    past = history[history["timestamp"] < tx_row["timestamp"]]
    G = nx.Graph()
    center = tx_row["transaction_id"]
    G.add_node(center, ntype="transaction")
    for col, ntype in [("customer_id", "customer"), ("device_id", "device"), ("card_id", "card"),
                       ("ip_id", "ip"), ("merchant_id", "merchant")]:
        G.add_node(tx_row[col], ntype=ntype)
        G.add_edge(center, tx_row[col])
    # other customers previously seen on the same device / IP / card
    for col in ["device_id", "ip_id", "card_id"]:
        others = past.loc[past[col] == tx_row[col], "customer_id"]
        for cust in others.drop_duplicates().head(max_neighbours):
            if cust != tx_row["customer_id"]:
                G.add_node(cust, ntype="customer")
                G.add_edge(tx_row[col], cust)

    color_map = {"transaction": "#E45756", "customer": "#4C78A8", "device": "#F58518",
                 "card": "#54A24B", "ip": "#B279A2", "merchant": "#72B7B2"}
    pos = nx.spring_layout(G, seed=42)
    plt.figure(figsize=(9, 7))
    nx.draw_networkx_nodes(G, pos, node_color=[color_map[G.nodes[n]["ntype"]] for n in G.nodes()], node_size=500, alpha=0.9)
    nx.draw_networkx_edges(G, pos, alpha=0.5)
    nx.draw_networkx_labels(G, pos, font_size=7)
    for ntype, color in color_map.items():
        plt.scatter([], [], c=color, label=ntype)
    plt.legend(loc="lower left", fontsize=8)
    plt.title(
        f"Past relationship network — {center} ({tx_row.get('fraud_pattern') or 'legit'})\n"
        f"device customers: {tx_row['device_unique_customers']}, IP customers: {tx_row['ip_unique_customers']}, "
        f"network failure rate: {tx_row['network_failure_rate']:.2f}"
    )
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(output_path, dpi=120)
    plt.close()


# ----------------------------------------------------------------------------- conclusion

def generate_conclusion(table: pd.DataFrame, ablation: pd.DataFrame, extras: Dict[str, object]) -> str:
    row = {r["Model"]: r for _, r in table.iterrows()}
    base = row["Transaction Baseline"]
    comb = row["Combined Transaction + Relationship Model"]
    rel = row["Relationship Intelligence Model"]
    lo, hi = extras["delta_pr_auc_ci"]
    delta = comb["PR-AUC"] - base["PR-AUC"]

    lines = [
        "=" * 60,
        "RELATIONSHIP INTELLIGENCE EXPERIMENT",
        "=" * 60,
        f"PR-AUC  baseline {base['PR-AUC']:.4f} {base['PR-AUC 95% CI']} | relationship {rel['PR-AUC']:.4f} "
        f"{rel['PR-AUC 95% CI']} | combined {comb['PR-AUC']:.4f} {comb['PR-AUC 95% CI']}",
        f"Recall@1%FPR  baseline {base['Recall @ 1% FPR']:.1%} | combined {comb['Recall @ 1% FPR']:.1%}",
        f"Combined − baseline PR-AUC: {delta * 100:+.2f} pp (95% CI {lo * 100:+.2f} to {hi * 100:+.2f} pp, ring-clustered bootstrap)",
        "",
        "Graph-hidden fraud recall (validation-tuned thresholds):",
    ]
    for name, v in extras["graph_hidden_recall"].items():
        lines.append(f"  {name}: {v:.1%}")
    lines.append(
        f"Graph-hidden sanity check — intrinsic-only classifier AUC (hidden fraud vs legit): "
        f"{extras['hidden_intrinsic_auc']:.3f} (≈0.5 means truly indistinguishable)"
    )
    lines.append("False-positive rate on legitimate shared environments:")
    for name, v in extras["legit_shared_fpr"].items():
        lines.append(f"  {name}: {v:.2%}")

    abl = ablation.set_index("Ablation")["pr_auc"]
    best = abl.idxmax()
    lines += ["", "Ablation PR-AUC: " + ", ".join(f"{k} {v:.4f}" for k, v in abl.items())]
    lines.append(f"Best ablation variant: {best}.")
    counts_gain = abl["transaction_plus_counts"] - abl["transaction_only"]
    full_gain = abl["transaction_plus_full_graph"] - abl["transaction_only"]
    if full_gain > 0:
        lines.append(
            f"Simple relationship counts deliver {counts_gain / full_gain:.0%} of the full-graph PR-AUC gain."
        )

    lines += ["", "Conclusion:"]
    if lo > 0:
        lines.append(
            "Relationship context adds statistically reliable predictive value beyond transaction-level features "
            "(the 95% CI of the PR-AUC gain excludes zero), including on fraud whose intrinsic attributes look normal."
        )
    elif delta > 0:
        lines.append("Relationship features improve PR-AUC, but the 95% CI includes zero — the gain is not conclusive.")
    else:
        lines.append("Relationship features did not improve over the transaction baseline in this setup.")
    lines.append("=" * 60)
    return "\n".join(lines)
