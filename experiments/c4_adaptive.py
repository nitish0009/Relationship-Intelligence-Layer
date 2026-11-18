"""Fix 6: adaptive (model-aware) attacker for C4.

The fixed-strategy C4 attackers do not optimise against the defender. Here the attacker has
black-box query access to a FROZEN deployed model (trained on non-adapted data) and greedily
searches the evasion space:

    device rotation, account rotation, IP rotation, carrier-NAT share, slow-and-low factor

At each step every one-notch-stronger move is evaluated on the attacker's own shadow simulation
(independent seeds), and the move with the best detection-reduction per unit of extra cost is
taken. Cost index = (devices + accounts + IPs per 100 ring txns) / no-evasion value
                    + (mean ring lifetime / no-evasion lifetime - 1)
The final strategy against each defender is then scored on held-out seeds, and a RETRAINED
defender (trained on data with that strategy) is evaluated on the same held-out seeds.

Outputs (results/c4_adaptive/): paths.csv, final.csv
"""

from __future__ import annotations

import os
import time
from typing import Dict, Tuple

import numpy as np
import pandas as pd

from experiments.common import PRIMARY_BUDGET, make_model, out_dir, score_all
from src.data_generator import AdversaryConfig, GeneratorConfig
from src.pipeline import build_dataset, fit_score, threshold_at_fpr
from src.temporal_gnn import TemporalIndex

LADDERS = {
    "device_max_uses": [None, 20, 10, 5, 2, 1],
    "account_max_uses": [None, 10, 5, 2, 1],
    "ip_max_uses": [None, 10, 5, 2, 1],
    "carrier_ip_share": [0.0, 0.25, 0.5, 0.75, 1.0],
    "slow_factor": [1.0, 2.0, 4.0, 8.0],
}
DEFENDERS = ["Transaction Baseline", "+ Temporal relationships", "Combined (full graph)",
             "Combined + delayed labels", "Neighbour-Aggregation XGBoost"]
SHADOW_SEEDS = (501, 502)
HOLDOUT_SEEDS = (901, 902)
MAX_STEPS = 10


def _cfg(pos: Dict[str, int]) -> GeneratorConfig:
    kw = {k: LADDERS[k][i] for k, i in pos.items()}
    return GeneratorConfig(adversary=AdversaryConfig(adapt_from=0.0, **kw))


class _Deployed:
    """Defenders trained once on non-adapted data, with frozen alert thresholds."""

    def __init__(self, seed: int = 42):
        ds = build_dataset(seed, label_delay_days=30.0)
        idx = TemporalIndex(ds.tx, k=20)
        self.models, self.thr = {}, {}
        for name in DEFENDERS:
            m = make_model(name, ds, seed, None, idx)
            res = fit_score(m, ds)
            self.models[name] = m
            self.thr[name] = threshold_at_fpr(ds.val.fraud_label.to_numpy(), res["p_val"], PRIMARY_BUDGET)


def _evaluate(deployed: _Deployed, cfg: GeneratorConfig, seeds) -> Tuple[Dict[str, float], float, dict]:
    """Mean recall on (all-adapted) ring transactions for each frozen defender, plus cost stats."""
    rec = {d: [] for d in DEFENDERS}
    stats = []
    for seed in seeds:
        ds = build_dataset(seed, cfg, label_delay_days=30.0)
        idx = TemporalIndex(ds.tx, k=20)
        f = ds.tx[ds.tx.fraud_label == 1]
        for d in DEFENDERS:
            m = deployed.models[d]
            if hasattr(m, "index"):
                m.index = idx  # neighbour features of the NEW dataset; the classifier stays frozen
            rec[d].append(float((m.predict_proba(f) >= deployed.thr[d]).mean()))
        r = ds.rings
        stats.append({"res_per_100": float((r.n_devices + r.n_accounts + r.n_ips).sum() / r.n_events.sum() * 100),
                      "lifetime": float(r.lifetime_days.mean())})
    st = pd.DataFrame(stats).mean().to_dict()
    return {d: float(np.mean(v)) for d, v in rec.items()}, st


def run(seeds=None) -> None:
    out = out_dir("c4_adaptive")
    t0 = time.time()
    deployed = _Deployed()
    cache: Dict[tuple, tuple] = {}

    def ev(pos):
        key = tuple(sorted(pos.items()))
        if key not in cache:
            cache[key] = _evaluate(deployed, _cfg(pos), SHADOW_SEEDS)
        return cache[key]

    start = {k: 0 for k in LADDERS}
    rec0, st0 = ev(start)

    def cost(st):
        return st["res_per_100"] / st0["res_per_100"] + (st["lifetime"] / st0["lifetime"] - 1)

    paths, finals = [], []
    for d in DEFENDERS:
        pos, (rec, st) = dict(start), ev(start)
        paths.append({"defender": d, "step": 0, "move": "none", **pos, "cost": cost(st), "shadow_recall": rec[d]})
        for step in range(1, MAX_STEPS + 1):
            best = None
            for k in LADDERS:
                if pos[k] + 1 >= len(LADDERS[k]):
                    continue
                cand = {**pos, k: pos[k] + 1}
                r2, s2 = ev(cand)
                gain = rec[d] - r2[d]
                dc = max(cost(s2) - cost(st), 1e-3)
                eff = gain / dc
                if gain > 0.005 and (best is None or eff > best[0]):
                    best = (eff, k, cand, r2, s2)
            if best is None:
                break
            _, k, pos, rec, st = best
            paths.append({"defender": d, "step": step, "move": k, **pos, "cost": cost(st), "shadow_recall": rec[d]})
            print(f"adaptive vs {d}: step {step} {k} -> recall {rec[d]:.3f} cost {cost(st):.2f}", flush=True)
        # held-out evaluation: frozen defender vs. its adversary's final strategy, then retrained defender
        frozen_rec, hst = _evaluate(deployed, _cfg(pos), HOLDOUT_SEEDS)
        retr = []
        for hs in HOLDOUT_SEEDS:
            dsr = build_dataset(hs, _cfg(pos), label_delay_days=30.0)  # attacker strategy present in training too
            idx = TemporalIndex(dsr.tx, k=20)
            res = fit_score(make_model(d, dsr, hs, None, idx), dsr)
            thr = score_all(dsr, res)["thr"][PRIMARY_BUDGET]
            fm = (dsr.test.fraud_label == 1).to_numpy()
            retr.append(float((res["p_test"][fm] >= thr).mean()))
        finals.append({"defender": d, **{k: LADDERS[k][i] for k, i in pos.items()}, "cost": cost(hst),
                       "no_evasion_recall": ev(start)[0][d], "frozen_recall_vs_adaptive": frozen_rec[d],
                       "retrained_recall_vs_adaptive": float(np.mean(retr))})
        pd.DataFrame(paths).to_csv(os.path.join(out, "paths.csv"), index=False)
        pd.DataFrame(finals).to_csv(os.path.join(out, "final.csv"), index=False)
    print(pd.DataFrame(finals).round(3).to_string(index=False))
    print(f"adaptive done in {(time.time() - t0) / 60:.1f} min ({len(cache)} strategies simulated)")
