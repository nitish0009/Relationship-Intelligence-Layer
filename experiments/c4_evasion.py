"""C4: cost of evasion -- how much must attackers spend to defeat each feature family?

Attackers adapt by rotating devices / accounts / IPs, hiding behind carrier-grade NAT, or
slowing down. Two defender regimes:
  * static     -- attackers adapt only in the test period (adapt_from=0.85); models were
                  trained on non-adapted rings
  * retrained  -- all rings adapt (adapt_from=0); models are trained on adapted behaviour
Attacker cost is measured from ground truth as resources per 100 ring transactions.

Outputs (results/c4/): evasion.csv, cost.csv, evasion_<strategy>.png
"""

from __future__ import annotations

import os
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from experiments.common import PRIMARY_BUDGET, make_model, out_dir, score_all
from src.data_generator import AdversaryConfig, GeneratorConfig
from src.pipeline import build_dataset, fit_score
from src.ring_metrics import ring_detection_table
from src.temporal_gnn import TemporalIndex

STRATEGIES = (
    [("none", "-", {})]
    + [("device rotation", k, {"device_max_uses": k}) for k in [20, 10, 5, 2, 1]]
    + [("account rotation", k, {"account_max_uses": k}) for k in [10, 5, 2, 1]]
    + [("IP rotation", k, {"ip_max_uses": k}) for k in [10, 5, 2, 1]]
    + [("carrier-NAT hiding", q, {"carrier_ip_share": q}) for q in [0.25, 0.5, 0.75, 1.0]]
    + [("slow-and-low", s, {"slow_factor": s}) for s in [2, 4, 8]]
    + [("all combined", "max", {"device_max_uses": 2, "account_max_uses": 2, "ip_max_uses": 2,
                                "carrier_ip_share": 0.5, "slow_factor": 4})]
)
MODELS = ["Transaction Baseline", "+ Relationship counts", "+ Temporal relationships", "Combined (full graph)",
          "Combined + delayed labels", "Neighbour-Aggregation XGBoost"]
DEFENDERS = {"static": 0.85, "retrained": 0.0}


def run(seeds=(42, 43)) -> None:
    out = out_dir("c4")
    t0 = time.time()
    rows, cost = [], []
    for strat, level, kw in STRATEGIES:
        for defender, adapt_from in DEFENDERS.items():
            if strat == "none" and defender == "retrained":
                continue
            for seed in seeds:
                print(f"c4: {strat}={level} defender={defender} seed={seed}", flush=True)
                cfg = GeneratorConfig(adversary=AdversaryConfig(adapt_from=adapt_from, **kw))
                ds = build_dataset(seed, cfg, label_delay_days=30.0)
                r = ds.rings
                adapted_ids = set(r.loc[r.adapted == 1, "ring_id"]) if strat != "none" else set(r.ring_id)
                ra = r[r.ring_id.isin(adapted_ids)]
                per100 = lambda c: float(ra[c].sum() / ra["n_events"].sum() * 100) if len(ra) else np.nan
                cost.append({"strategy": strat, "level": str(level), "defender": defender, "seed": seed,
                             "devices_per_100": per100("n_devices"), "accounts_per_100": per100("n_accounts"),
                             "ips_per_100": per100("n_ips"), "mean_lifetime_days": float(ra.lifetime_days.mean())})
                idx = TemporalIndex(ds.tx, k=20)
                for name in MODELS:
                    res = fit_score(make_model(name, ds, seed, None, idx), ds)
                    sc = score_all(ds, res)
                    thr = sc["thr"][PRIMARY_BUDGET]
                    rt = ring_detection_table(ds.test, res["p_test"], thr)
                    rt = rt[rt.ring_id.isin(adapted_ids)] if len(rt) else rt
                    te = ds.test
                    fm = (te.fraud_label == 1) & te.ring_id.isin(adapted_ids)
                    rows.append({
                        "strategy": strat, "level": str(level), "defender": defender, "seed": seed, "model": name,
                        "pr_auc": sc["row"]["pr_auc"],
                        "adapted_txn_recall": float((res["p_test"][fm.to_numpy()] >= thr).mean()) if fm.any() else np.nan,
                        "adapted_rings": len(rt),
                        "amount_lost_share": float(rt.amount_before_alert.sum() / rt.ring_amount.sum()) if len(rt) else np.nan,
                    })
    ev = pd.DataFrame(rows)
    ev.to_csv(os.path.join(out, "evasion.csv"), index=False)
    pd.DataFrame(cost).to_csv(os.path.join(out, "cost.csv"), index=False)
    _plots(ev, pd.DataFrame(cost), out)
    print(f"c4 done in {(time.time() - t0) / 60:.1f} min")


def _plots(ev: pd.DataFrame, cost: pd.DataFrame, out: str) -> None:
    base = ev[ev.strategy == "none"].groupby("model")["adapted_txn_recall"].mean()
    for strat in [s for s in ev.strategy.unique() if s not in ("none", "all combined")]:
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)
        for ax, defender in zip(axes, DEFENDERS):
            x = ev[(ev.strategy == strat) & (ev.defender == defender)]
            c = cost[(cost.strategy == strat) & (cost.defender == defender)].groupby("level", sort=False).mean(numeric_only=True)
            res_col = {"device rotation": "devices_per_100", "account rotation": "accounts_per_100",
                       "IP rotation": "ips_per_100"}.get(strat)
            for m in x.model.unique():
                g = x[x.model == m].groupby("level", sort=False)["adapted_txn_recall"].mean()
                xs = c.loc[g.index, res_col].to_numpy() if res_col else [float(v) for v in g.index]
                ax.plot([c0 for c0 in xs], g.to_numpy(), marker="o", label=m)
                ax.scatter([], [])
            ax.set_title(f"{strat} — {defender} defender")
            ax.set_xlabel(f"attacker {res_col.replace('_', ' ') if res_col else 'strategy level'}")
            ax.set_ylabel(f"recall on adapted ring txns ({PRIMARY_BUDGET:.1%} budget)")
        axes[0].legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(os.path.join(out, f"evasion_{strat.replace(' ', '_')}.png"), dpi=130)
        plt.close(fig)
