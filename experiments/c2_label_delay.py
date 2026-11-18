"""C2: label delay x ring lifetime (x infrastructure reuse).

Question: when do features built from neighbours' *confirmed fraud labels* help, given that
labels arrive only after a delay D, and fraud rings live only L days?

Outputs (results/c2/): grid.csv, reachability.csv, phase_reuse*.png
"""

from __future__ import annotations

import os
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from experiments.common import PRIMARY_BUDGET, out_dir, ring_rows, score_all
from src.data_generator import GeneratorConfig
from src.graph_features import GRAPH_FEATURE_COLS
from src.label_features import LABEL_FEATURE_COLS
from src.pipeline import build_dataset, fit_score, tabular, with_label_delay
from src.preprocessing import TRANSACTION_FEATURE_COLS as T

LIFETIMES = [(1, 3), (3, 7), (7, 14), (14, 30), (30, 60), (60, 120)]
DELAYS = [0, 1, 3, 7, 14, 30, 60, 90, float("inf")]
REUSE = [0.0, 0.5]
MODELS = {
    "Combined (no labels)": T + GRAPH_FEATURE_COLS,
    "Combined + delayed labels": T + GRAPH_FEATURE_COLS + LABEL_FEATURE_COLS,
    "Transaction + delayed labels": T + LABEL_FEATURE_COLS,
}


def label_reachability(test: pd.DataFrame, tx: pd.DataFrame, delay: float) -> float:
    """Model-free ceiling: share of test fraud txns for which some EARLIER transaction of the
    same ring already has a matured label (t_ring_start + D <= t). Within-ring only."""
    f = tx[tx.fraud_label == 1]
    start = f.groupby("ring_id")["timestamp"].min()
    tf = test[test.fraud_label == 1]
    if not np.isfinite(delay):
        return 0.0
    first = tf["ring_id"].map(start)
    return float(((tf["timestamp"] >= first + pd.Timedelta(days=delay)) & (tf["ring_seq"] > 0)).mean())


def run(seeds=(42, 43, 44)) -> None:
    out = out_dir("c2")
    t0 = time.time()
    rows, reach = [], []
    for reuse in REUSE:
        for lt in LIFETIMES:
            for seed in seeds:
                print(f"c2: reuse={reuse} lifetime={lt} seed={seed}", flush=True)
                cfg = GeneratorConfig(ring_duration_days=lt, infra_reuse=reuse)
                base = build_dataset(seed, cfg, label_delay_days=float("inf"))
                for d in DELAYS:
                    ds = base if not np.isfinite(d) else with_label_delay(base, d)
                    reach.append({"reuse": reuse, "lifetime": f"{lt[0]}-{lt[1]}d", "seed": seed, "delay": d,
                                  "within_ring_reachability": label_reachability(ds.test, ds.tx, d)})
                    for name, cols in MODELS.items():
                        if name == "Combined (no labels)" and d != DELAYS[0]:
                            continue  # label-free: identical for every delay, fit once
                        res = fit_score(tabular(cols, seed), ds)
                        sc = score_all(ds, res)
                        rt = ring_rows(ds, res["p_test"], sc["thr"][PRIMARY_BUDGET])
                        rows.append({
                            "reuse": reuse, "lifetime": f"{lt[0]}-{lt[1]}d", "lifetime_mid": np.mean(lt), "seed": seed,
                            "delay": d, "model": name, "pr_auc": sc["row"]["pr_auc"],
                            "recall_at_1pct_fpr": sc["row"]["recall_at_1pct_fpr"],
                            "ring_amount_lost_share": float(rt.amount_before_alert.sum() / rt.ring_amount.sum()) if len(rt) else np.nan,
                        })
    grid = pd.DataFrame(rows)
    grid.to_csv(os.path.join(out, "grid.csv"), index=False)
    pd.DataFrame(reach).groupby(["reuse", "lifetime", "delay"], sort=False)["within_ring_reachability"].mean() \
        .reset_index().to_csv(os.path.join(out, "reachability.csv"), index=False)
    _phase_plots(grid, out)
    print(f"c2 done in {(time.time() - t0) / 60:.1f} min")


def _phase_plots(grid: pd.DataFrame, out: str) -> None:
    g = grid.groupby(["reuse", "lifetime", "lifetime_mid", "delay", "model"])["pr_auc"].mean().reset_index()
    for reuse in g.reuse.unique():
        x = g[g.reuse == reuse]
        base = x[x.model == "Combined (no labels)"].set_index("lifetime")["pr_auc"]
        lab = x[x.model == "Combined + delayed labels"].copy()
        lab["gain_pp"] = (lab["pr_auc"] - lab["lifetime"].map(base)) * 100
        piv = lab.pivot_table(index="lifetime_mid", columns="delay", values="gain_pp")
        piv.index = [l for l in lab.drop_duplicates("lifetime_mid").sort_values("lifetime_mid")["lifetime"]]
        fig, ax = plt.subplots(figsize=(9, 5))
        im = ax.imshow(piv.values, cmap="RdBu", vmin=-np.nanmax(np.abs(piv.values)), vmax=np.nanmax(np.abs(piv.values)),
                       aspect="auto")
        ax.set_xticks(range(piv.shape[1]), [("∞" if not np.isfinite(c) else f"{c:g}d") for c in piv.columns])
        ax.set_yticks(range(piv.shape[0]), piv.index)
        ax.set_xlabel("Label delay D")
        ax.set_ylabel("Ring lifetime L")
        for i in range(piv.shape[0]):
            for j in range(piv.shape[1]):
                ax.text(j, i, f"{piv.values[i, j]:+.1f}", ha="center", va="center", fontsize=8)
        ax.set_title(f"PR-AUC gain from delayed-label features (pp), infra reuse = {reuse}")
        fig.colorbar(im)
        fig.tight_layout()
        fig.savefig(os.path.join(out, f"phase_reuse{reuse}.png"), dpi=130)
        plt.close(fig)
