"""Pre-registered rolling-origin evaluation on AMLworld (see PREREGISTRATION_IJCAI.md).

Test window = one day d, validation = day d-1, training = all earlier days (>= 2 days). Rings = ground-truth
laundering patterns with >= 2 transactions starting in the test day. Windows whose test day ends before the
development validation period began are PRIMARY (never used during development); the rest are SECONDARY.

For every window x seed: hand-crafted graph features and neighbour-aggregation XGBoost (control, TVW v2;
neighbour aggregation also class-balanced and amount-weighted), temporal GraphSAGE and GAT (control, TVW v2;
trained on all laundering + 30% of legitimate rows). Hyper-parameters are shared by all variants of a model.

Resumable: finished (window, seed, model, variant) units are skipped on restart.
Outputs (results/<out_name>/): windows.csv, runs.csv, ring_tables.csv, params.json
"""

from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

from experiments.common import Timer, out_dir, save_json
from experiments.headroom import BUDGET_MULTIPLES, MIN_RING, threshold_at_multiple
from src.evaluation import select_threshold
from src.pipeline import Dataset
from src.ring_metrics import ring_detection_table

LEGIT_SAMPLE = 0.3
XGB_VARIANTS = {"Combined (full graph)": ["control", "TVW"],
                "Neighbour-Aggregation XGBoost": ["control", "TVW", "class-balanced", "amount-weighted"]}
GNN_VARIANTS = {"Temporal GraphSAGE": ["control", "TVW"], "Temporal GAT": ["control", "TVW"]}
GNN_GRID = [{"hidden": 64, "lr": 3e-3}, {"hidden": 32, "lr": 1e-3}]


def _windows(tx: pd.DataFrame, dev_val_start: pd.Timestamp, first_test_day: int = 4):
    day = (tx.timestamp - tx.timestamp.min().normalize()).dt.days + 1
    out = []
    for d in range(first_test_day, int(day.max()) + 1):
        test_end = tx.timestamp.min().normalize() + pd.Timedelta(days=d)
        out.append({"test_day": d, "primary": bool(test_end <= dev_val_start)})
    return day, out


def _done(path: str) -> set:
    if not os.path.exists(path):
        return set()
    r = pd.read_csv(path)
    return set(zip(r.test_day, r.seed, r.model, r.variant))


def run(prefix: str = "HI-Small", seeds=(42, 43, 44), out_name: str | None = None, test_days=None,
        models=None, gnn_epochs: int = 6, val_sample: int = 300_000, secondary_seeds=(42,)) -> None:
    from experiments import amlworld as A
    from src.classifier_utils import get_classifier
    from src.clock_weights import TVWModel, tvw_weights
    from src.objectives import WeightedModel
    from src.pipeline import FEATURE_FAMILIES, XGB_SPACE, tabular, tune
    from src.temporal_gnn import NeighborAggXGB, TemporalGNN, TemporalIndex

    out = out_dir(out_name or f"rolling_{prefix.replace('-', '_').lower()}")
    ds0, f, rings = A.prepare(out, prefix)
    tx = ds0.tx
    day, wins = _windows(tx, ds0.val.timestamp.min())
    if test_days:
        wins = [w for w in wins if w["test_day"] in set(test_days)]
    pd.DataFrame(wins).to_csv(os.path.join(out, "windows.csv"), index=False)
    print("windows:", wins, flush=True)
    idx = TemporalIndex(tx, k=10)
    listed = set(tx.loc[(tx.fraud_label == 1) & (tx.fraud_pattern != "unlisted"), "ring_id"])
    tvw_kw = dict(identity_cols=A.AML_IDENTITY_COLS, identity_key=A.account_key)
    runs_p, rt_p, par_p = (os.path.join(out, n) for n in ("runs.csv", "ring_tables.csv", "params.json"))
    done = _done(runs_p)
    tp = os.path.join(out_dir("amlworld"), "tuned_params.json")
    params = json.load(open(par_p)) if os.path.exists(par_p) else \
        ({k: v for k, v in json.load(open(tp)).items()} if (prefix == "HI-Small" and os.path.exists(tp)) else {})
    variants = {**XGB_VARIANTS, **GNN_VARIANTS}
    if models:
        variants = {m: v for m, v in variants.items() if m in models}

    def xgb_factory(name):
        if name in FEATURE_FAMILIES:
            return lambda p, s: tabular(FEATURE_FAMILIES[name], s, p, name)
        return lambda p, s: NeighborAggXGB(idx, s, classifier=get_classifier(s, **p))

    for w in wins:
        d = w["test_day"]
        tr = tx[day <= d - 2]
        va = tx[day == d - 1]
        te = tx[day == d].copy()
        te.loc[~te.ring_id.isin(listed) & (te.fraud_label == 1), "ring_seq"] = -1  # pattern rings only
        ds = Dataset(tx, tr, va, te, rings, 42, 1.0, {})
        yv, yt = va.fraud_label.to_numpy(), te.fraud_label.to_numpy()
        vsub = va.sample(min(val_sample, len(va)), random_state=0)
        print(f"== {prefix} test day {d} ({'primary' if w['primary'] else 'secondary'}): train {len(tr):,}, "
              f"test laundering {int(yt.sum())}", flush=True)
        for name, vlist in variants.items():
            is_gnn = name in GNN_VARIANTS
            if name not in params:  # tune once (first window), shared by all variants and windows
                with Timer(f"tune {name}"):
                    if is_gnn:
                        best = (-1.0, GNN_GRID[0])
                        for p in GNN_GRID:
                            m = _gnn(idx, name, tr, vsub, 42, p, gnn_epochs)
                            best = max(best, (average_precision_score(vsub.fraud_label, m.predict_proba(vsub)), p),
                                       key=lambda t: t[0])
                        params[name] = best[1]
                    else:
                        params[name] = tune(lambda p: xgb_factory(name)(p, 42), ds, XGB_SPACE, 4, 42)
                save_json(params, par_p)
            for seed in (seeds if w["primary"] else secondary_seeds):
                for variant in vlist:
                    if (d, seed, name, variant) in done:
                        continue
                    with Timer(f"day {d} {name} {variant} seed {seed}"):
                        if is_gnn:
                            wts = tvw_weights(tr, 1.0, **tvw_kw) if variant == "TVW" else None
                            m = _gnn(idx, name, tr, vsub, seed, params[name], gnn_epochs, weights=wts)
                            p_val, p_test = m.predict_proba(va), m.predict_proba(te)
                        else:
                            base = xgb_factory(name)(params[name], seed)
                            model = {"control": lambda: base, "TVW": lambda: TVWModel(base, 1.0, **tvw_kw),
                                     "class-balanced": lambda: WeightedModel(base, "class-balanced"),
                                     "amount-weighted": lambda: WeightedModel(base, "amount-weighted")}[variant]()
                            model.fit(tr, tr.fraud_label.to_numpy())
                            p_val, p_test = model.predict_proba(va), model.predict_proba(te)
                    rows, tabs = [], []
                    for c in BUDGET_MULTIPLES:
                        thr = threshold_at_multiple(yv, p_val, c)
                        t = ring_detection_table(te, p_test, thr)
                        t = t[t.ring_txns >= MIN_RING] if len(t) else t
                        rows.append({"test_day": d, "primary": w["primary"], "seed": seed, "model": name, "variant": variant,
                                     "budget_multiple": c, "pr_auc": average_precision_score(yt, p_test),
                                     "n_rings": len(t),
                                     "lost_share": float(t.amount_before_alert.sum() / t.ring_amount.sum()) if len(t) else np.nan})
                        if len(t):
                            tabs.append(t.assign(test_day=d, primary=w["primary"], seed=seed, model=name, variant=variant,
                                                 budget_multiple=c))
                    pd.DataFrame(rows).to_csv(runs_p, mode="a", header=not os.path.exists(runs_p), index=False)
                    if tabs:
                        pd.concat(tabs).to_csv(rt_p, mode="a", header=not os.path.exists(rt_p), index=False)
    print(f"rolling {prefix} done", flush=True)


def _gnn(idx, name, tr, vsub, seed, p, epochs, weights=None):
    from src.temporal_gnn import TemporalGNN
    kind = {"Temporal GraphSAGE": "sage", "Temporal GAT": "gat"}[name]
    rng = np.random.default_rng(seed)
    keep = (tr.fraud_label.to_numpy() == 1) | (rng.random(len(tr)) < LEGIT_SAMPLE)
    sub = tr[keep]
    m = TemporalGNN(idx, kind, seed=seed, epochs=epochs, val_df=vsub, **p)
    return m.fit(sub, sub.fraud_label.to_numpy(), sample_weight=None if weights is None else weights[keep])
