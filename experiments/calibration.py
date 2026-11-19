"""Fix 4: sim-to-real calibration -- where do the real datasets sit on the synthetic reuse dial?

For the synthetic generator at infrastructure-reuse levels p, we compute the same statistic that
we measure on real data: the share of TEST fraud whose customer / device / card was already used
by fraud in TRAINING ("seen-bad" share). Interpolating the real value on the synthetic curve gives
an estimated reuse level p_hat for IEEE-CIS (and AMLworld for customer/device). We then check
whether the synthetic dose-response (C3) at p_hat PREDICTS the transductive-graph inflation
actually observed on IEEE-CIS -- an out-of-sample consistency test of the generator.

Inputs: results/ieee/calibration_stats.json, results/amlworld/calibration_stats.json (if present),
        results/c3/dose.csv, results/ieee/metrics_summary.csv
Outputs (results/calibration/): synthetic_curve.csv, calibration.json, calibration.png
"""

from __future__ import annotations

import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from experiments.common import out_dir
from src.data_generator import GeneratorConfig, generate_all
from src.evaluation import RESULTS_DIR
from src.preprocessing import chronological_split

REUSE = [0.0, 0.1, 0.25, 0.4, 0.5, 0.75, 1.0]


def _seen(train: pd.DataFrame, test: pd.DataFrame) -> dict:
    trf, tef = train[train.fraud_label == 1], test[test.fraud_label == 1]
    c = tef.customer_id.isin(set(trf.customer_id))
    d = tef.device_id.isin(set(trf.device_id))
    k = tef.card_id.isin(set(trf.card_id))
    return {"seen_bad_customer": c.mean(), "seen_bad_device": d.mean(), "seen_bad_card": k.mean(),
            "seen_bad_any": (c | d | k).mean(), "seen_bad_customer_or_device": (c | d).mean()}


def _interp(curve: pd.DataFrame, stat: str, value: float) -> float:
    x, y = curve.index.to_numpy(dtype=float), curve[stat].to_numpy()
    order = np.argsort(y)
    return float(np.interp(value, y[order], x[order]))


def run(seeds=(42, 43, 44)) -> None:
    out = out_dir("calibration")
    rows = []
    for p in REUSE:
        for seed in seeds:
            tx = generate_all(None, seed=seed, config=GeneratorConfig(infra_reuse=p))["transactions"]
            tr, _, te = chronological_split(tx)
            rows.append({"reuse": p, "seed": seed, **_seen(tr, te)})
    curve = pd.DataFrame(rows).groupby("reuse").mean(numeric_only=True).drop(columns="seed")
    curve.to_csv(os.path.join(out, "synthetic_curve.csv"))
    res = {}
    for name, stat_map in [("ieee", {"seen_bad_any": "seen_bad_any", "seen_bad_customer": "seen_bad_customer"}),
                           ("amlworld", {"seen_bad_customer_or_device": "seen_bad_any",
                                         "seen_bad_customer": "seen_bad_customer"})]:
        path = os.path.join(RESULTS_DIR, name if name != "ieee" else "ieee", "calibration_stats.json")
        if not os.path.exists(path):
            continue
        real = json.load(open(path))
        res[name] = {"real": real, "p_hat": {syn: _interp(curve, syn, real[r]) for syn, r in stat_map.items()}}
    # consistency test: predicted vs observed transductive inflation on every real dataset that has
    # both entity-GraphSAGE runs (IEEE-CIS; AMLworld)
    dose_p = os.path.join(RESULTS_DIR, "c3", "dose.csv")
    if os.path.exists(dose_p):
        d = pd.read_csv(dose_p)
        col = "pr_auc_test_window" if "pr_auc_test_window" in d else "pr_auc"
        g = d.pivot_table(index="reuse", columns="model", values=col)
        infl = (g["GraphSAGE (transductive)"] - g["GraphSAGE (entity, inductive)"]) * 100
        for name in list(res):
            mp = os.path.join(RESULTS_DIR, name, "metrics_summary.csv")
            if not os.path.exists(mp):
                continue
            s = pd.read_csv(mp, index_col=0)
            if not {"GraphSAGE (entity, inductive)", "GraphSAGE (transductive)"} <= set(s.index):
                continue
            obs_ind, obs_tr = s.loc["GraphSAGE (entity, inductive)", "pr_auc_mean"], s.loc["GraphSAGE (transductive)", "pr_auc_mean"]
            rng_ = sorted(res[name]["p_hat"].values())
            pred = [float(np.interp(p, infl.index, infl.values)) for p in rng_]
            obs = float((obs_tr - obs_ind) * 100)
            res[name]["transductive_inflation"] = {
                "p_hat_range": rng_, "predicted_pp_at_p_hat_range": pred, "observed_pp": obs,
                "observed_within_predicted_band": bool(min(pred) - 2 <= obs <= max(pred) + 2),
                "observed_inductive_pr_auc": float(obs_ind), "observed_transductive_pr_auc": float(obs_tr),
                "note": "absolute pp is the comparable quantity (band = predicted range +/- 2 pp); relative "
                        "inflation is not comparable across base rates"}
    with open(os.path.join(out, "calibration.json"), "w") as f:
        json.dump(res, f, indent=2, default=float)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for stat in ["seen_bad_any", "seen_bad_customer", "seen_bad_device"]:
        ax.plot(curve.index, curve[stat], marker="o", label=f"synthetic: {stat}")
    for name, r in res.items():
        ax.axhline(r["real"]["seen_bad_customer"], ls="--", lw=1, label=f"{name}: seen_bad_customer")
    ax.set_xlabel("infrastructure reuse p")
    ax.set_ylabel("share of test fraud on entities seen in training fraud")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "calibration.png"), dpi=130)
    print(curve.round(3).to_string())
    print(json.dumps(res, indent=2, default=float))
