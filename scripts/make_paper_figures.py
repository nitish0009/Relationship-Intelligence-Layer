#!/usr/bin/env python3
"""Paper figures (paper/figures/*.pdf), generated from results/ only -- rerun after any experiment.

  fig_ringloss.pdf  money lost before a ring's first alert, hard preset, ring-clustered 95% CIs
  fig_warmup.pdf    recall on the k-th transaction of a new ring (hard preset)
  fig_leakage.pdf   transductive-protocol inflation vs infrastructure reuse, with real datasets
  fig_tvw.pdf       effect of takedown-value weighting (TVW) on ring losses, all datasets
"""

from __future__ import annotations

import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "results")
OUT = os.path.join(ROOT, "paper", "figures")

# validated categorical slots (dataviz reference palette, light mode) + neutral
BLUE, ORANGE, AQUA, YELLOW, GREY = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#8c8b86"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"

plt.rcParams.update({
    "font.family": "serif", "font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7, "axes.edgecolor": INK2,
    "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2, "axes.spines.top": False,
    "axes.spines.right": False, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.axisbelow": True, "pdf.fonttype": 42, "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
})

SHORT = {"Transaction Baseline": "Transaction features", "+ Relationship counts": "+ relationship counts",
         "+ Temporal relationships": "+ temporal relationships", "Combined (full graph)": "Hand-crafted graph",
         "Combined + delayed labels": "Hand-crafted + labels", "Neighbour-Aggregation XGBoost": "Neighbour-agg. XGB",
         "Temporal GraphSAGE": "Temporal GraphSAGE", "Temporal GAT": "Temporal GAT", "Temporal R-GCN": "Temporal R-GCN",
         "CARE-style GNN": "CARE-style GNN", "TGN-style Memory": "TGN-style memory",
         "GraphSAGE (entity, inductive)": "Entity GraphSAGE"}
GNNS = {"Temporal GraphSAGE", "Temporal GAT", "Temporal R-GCN", "CARE-style GNN", "TGN-style Memory",
        "GraphSAGE (entity, inductive)"}


def _save(fig, name: str) -> None:
    """PDF for LaTeX + 300-dpi PNG (the EAI template asks for figures as separate .jpg/.png files)."""
    fig.savefig(os.path.join(OUT, name + ".pdf"))
    fig.savefig(os.path.join(OUT, name + ".png"), dpi=300)
    plt.close(fig)


def _ring_ci(rt: pd.DataFrame, n_boot: int = 2000, seed: int = 0):
    rng = np.random.default_rng(seed)
    amt, lost = rt.ring_amount.to_numpy(), rt.amount_before_alert.to_numpy()
    b = [lost[i].sum() / amt[i].sum() for i in (rng.integers(0, len(rt), len(rt)) for _ in range(n_boot))]
    return lost.sum() / amt.sum(), np.percentile(b, 2.5), np.percentile(b, 97.5)


def fig_ringloss(budget: float = 0.001) -> None:
    rt = pd.read_csv(os.path.join(RES, "main_hard", "ring_tables.csv"))
    rt = rt[rt.budget == budget]
    rows = [(m, *_ring_ci(g)) for m, g in rt.groupby("model")]
    df = pd.DataFrame(rows, columns=["model", "lost", "lo", "hi"]).sort_values("lost", ascending=False)
    fig, ax = plt.subplots(figsize=(2.6, 2.5))
    y = np.arange(len(df))
    col = [BLUE if m in GNNS else GREY for m in df.model]
    ax.barh(y, 100 * df.lost, height=0.62, color=col, zorder=2)
    ax.errorbar(100 * df.lost, y, xerr=[100 * (df.lost - df.lo), 100 * (df.hi - df.lost)], fmt="none",
                ecolor=INK, elinewidth=0.7, capsize=1.5, zorder=3)
    ax.set_yticks(y, [SHORT.get(m, m) for m in df.model])
    ax.set_xlabel("Money lost before first alert (%)")
    ax.grid(axis="y", visible=False)
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color=BLUE, label="graph neural network"), Patch(color=GREY, label="feature-based (XGBoost)")],
              loc="upper right", frameon=False, fontsize=6)
    _save(fig, "fig_ringloss")


def fig_warmup() -> None:
    w = pd.read_csv(os.path.join(RES, "main_hard", "warmup.csv"))
    fig, ax = plt.subplots(figsize=(2.6, 2.5))
    for m, c in [("Transaction Baseline", GREY), ("Combined (full graph)", ORANGE),
                 ("Neighbour-Aggregation XGBoost", AQUA), ("Temporal GAT", BLUE)]:
        g = w[(w.model == m) & (w.k <= 12)]
        ax.plot(g.k, 100 * g.recall, color=c, lw=1.6, marker="o", ms=3, label=SHORT[m])
    ax.set_xlabel("k-th transaction of a new fraud ring")
    ax.set_ylabel("Recall (%) at 0.5% FPR budget")
    ax.set_xticks(range(1, 13))
    ax.set_ylim(0, 100)
    ax.legend(frameon=False, loc="lower center", bbox_to_anchor=(0.45, 1.0), ncol=2, fontsize=6)
    _save(fig, "fig_warmup")


def fig_leakage() -> None:
    d = pd.read_csv(os.path.join(RES, "c3", "dose.csv"))
    col = "pr_auc_test_window" if "pr_auc_test_window" in d else "pr_auc"
    g = d.pivot_table(index=["reuse", "seed"], columns="model", values=col)
    infl = (100 * (g["GraphSAGE (transductive)"] - g["GraphSAGE (entity, inductive)"])).groupby("reuse")
    m, s, n = infl.mean(), infl.std(), infl.count()
    ci = 1.96 * s / np.sqrt(n)
    fig, ax = plt.subplots(figsize=(2.6, 2.3))
    ax.fill_between(m.index, m - ci, m + ci, color=BLUE, alpha=0.18, lw=0)
    ax.plot(m.index, m, color=BLUE, lw=1.6, marker="o", ms=3, label="synthetic (mean, 95% CI)")
    cal_p = os.path.join(RES, "calibration", "calibration.json")
    if os.path.exists(cal_p):
        cal = json.load(open(cal_p))
        for name, c, lab in [("ieee", ORANGE, "IEEE-CIS (observed)"), ("amlworld", AQUA, "AMLworld (observed)")]:
            t = cal.get(name, {}).get("transductive_inflation")
            if t:
                lo, hi = t["p_hat_range"]
                ax.errorbar((lo + hi) / 2, t["observed_pp"], xerr=(hi - lo) / 2, fmt="s", color=c, ms=4,
                            capsize=2, elinewidth=0.9, label=lab)
    ax.axhline(0, color=INK2, lw=0.6)
    ax.set_xlabel("Reuse $p$ (real data: estimated $\\hat p$)")
    ax.set_ylabel("PR-AUC inflation (pp)")
    ax.legend(frameon=False, loc="upper right", fontsize=6)
    _save(fig, "fig_leakage")


def fig_tvw(budget: float = 0.005) -> None:
    parts = []
    for sub in ("tvw", "tvw_gnn"):
        p = os.path.join(RES, sub, "paired.csv")
        if os.path.exists(p):
            parts.append(pd.read_csv(p))
    if not parts:
        return
    p = pd.concat(parts, ignore_index=True)
    p = p[(p.budget == budget) & (p.variant == "TVW (beta=1)") & (p.model != "Transaction Baseline")
          & (p.dataset != "synthetic (default)")]
    p = p.drop_duplicates(["dataset", "model"]).sort_values(["dataset", "model"]).reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(2.6, 0.2 * len(p) + 0.9))
    y = np.arange(len(p))[::-1]
    col = [BLUE if m in GNNS else ORANGE for m in p.model]
    ax.errorbar(p.delta_pp, y, xerr=[p.delta_pp - p.ci_lo_pp, p.ci_hi_pp - p.delta_pp], fmt="none", ecolor=INK2,
                elinewidth=0.8, capsize=1.5)
    ax.scatter(p.delta_pp, y, c=col, s=14, zorder=3)
    ax.axvline(0, color=INK, lw=0.7)
    ds_short = {"synthetic (hard)": "Sim.", "AMLworld": "AML", "IEEE-CIS": "IEEE"}
    ax.set_yticks(y, [f"{ds_short.get(d, d)} | {SHORT.get(m, m)}" for d, m in zip(p.dataset, p.model)], fontsize=6.5)
    ax.set_xlabel("$\\Delta$ money lost (pp)")
    ax.grid(axis="y", visible=False)
    _save(fig, "fig_tvw")


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    for f in (fig_ringloss, fig_warmup, fig_leakage, fig_tvw):
        f()
        print("ok", f.__name__)
