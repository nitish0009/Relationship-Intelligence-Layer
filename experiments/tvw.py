"""C5 (method): takedown-value weighting (TVW) -- training on the ring's clock (src/clock_weights.py).

For each dataset x seed x base model we fit the SAME classifier (same tuned hyper-parameters)
with beta in BETAS and compare, at fixed validation alert budgets, the ring-level outcome C1
measures: share of ring money lost before the first alert, ring recall, recall on the first
transaction of a ring, plus row-level PR-AUC (to show what TVW costs, if anything).

Variants reported:
  beta = 0          the ordinary model (control)
  beta = 1          the pure takedown-value surrogate (no tuning; the derivation's choice)
  beta = val-sel.   beta chosen on the validation split by money lost before first alert
  oracle (beta=1)   true ring ids instead of pseudo-rings (synthetic only; measures the gap)

Datasets: synthetic hard preset and default preset (10 seeds each), AMLworld (ground-truth
laundering patterns, 3 seeds), IEEE-CIS (proxy rings, Vesta features, 3 seeds). Real datasets are
skipped if their files are missing.

Statistics: per-seed paired differences and a ring-clustered paired bootstrap over the pooled
ring tables (rings resampled within seed; TVW and control share every resample).

Outputs (results/tvw/): runs.csv, ring_tables.csv, summary.csv, paired.csv, beta_curve.csv, tvw.png
"""

from __future__ import annotations

import os
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

from experiments.common import BUDGETS, PRIMARY_BUDGET, Timer, out_dir, save_json
from src.clock_weights import TVWModel
from src.data_generator import HARD_PRESET, GeneratorConfig
from src.pipeline import FEATURE_FAMILIES, XGB_SPACE, build_dataset, fit_score, tabular, threshold_at_fpr, tune
from src.ring_metrics import ring_detection_table
from src.temporal_gnn import NeighborAggXGB, TemporalIndex

BETAS = [0.0, 0.25, 0.5, 0.75, 1.0]
SYN_MODELS = ["Transaction Baseline", "Combined (full graph)", "Combined + delayed labels",
              "Neighbour-Aggregation XGBoost"]


def _ring_table(df, p, thr, min_txns=1):
    t = ring_detection_table(df, p, thr)
    return t[t.ring_txns >= min_txns] if len(t) else t


def _lost(t):
    return float(t.amount_before_alert.sum() / t.ring_amount.sum()) if len(t) else np.nan


def _evaluate(ds, factory, seed, tag, models, oracle=True, min_txns=1, ring_filter=None, first_mask=None,
              cache=None, n_iter=8, betas=None):
    """Fit every (model, beta) once; returns run rows and ring tables. Hyper-parameters are tuned
    on validation for the control model (beta=0) and shared by every beta (``cache``: tune once,
    reuse across seeds -- used for the large real datasets)."""
    runs, tabs = [], []
    yv, yt = ds.val.fraud_label.to_numpy(), ds.test.fraud_label.to_numpy()
    first = (ds.test.ring_seq.to_numpy() == 0) & (yt == 1) if first_mask is None else first_mask
    for name in models:
        if cache is not None and name in cache:
            params = cache[name]
        else:
            params = tune(lambda p: factory(name, p, seed), ds, XGB_SPACE, n_iter, seed)
            if cache is not None:
                cache[name] = params
        variants = [(b, False) for b in (betas or BETAS)] + ([(1.0, True)] if oracle else [])
        for beta, orc in variants:
            res = fit_score(TVWModel(factory(name, params, seed), beta, oracle=orc), ds)
            for b in BUDGETS:
                tv = threshold_at_fpr(yv, res["p_val"], b)
                vt, tt = _ring_table(ds.val, res["p_val"], tv, min_txns), _ring_table(ds.test, res["p_test"], tv, min_txns)
                if ring_filter is not None:
                    vt, tt = ring_filter(vt), ring_filter(tt)
                runs.append({"dataset": tag, "seed": seed, "model": name, "beta": beta, "oracle": orc, "budget": b,
                             "val_lost": _lost(vt), "test_lost": _lost(tt),
                             "ring_recall": float(tt.detected.mean()) if len(tt) else np.nan,
                             "first_txn_recall": float((res["p_test"][first] >= tv).mean()) if first.any() else np.nan,
                             "pr_auc": average_precision_score(yt, res["p_test"]), "n_rings": len(tt)})
                if len(tt):
                    tt = tt.assign(dataset=tag, seed=seed, model=name, beta=beta, oracle=orc, budget=b)
                    tabs.append(tt[["dataset", "seed", "model", "beta", "oracle", "budget", "ring_id",
                                    "ring_amount", "amount_before_alert", "detected", "txns_before_alert"]])
        print(f"   [{tag} seed {seed}] {name}: done", flush=True)
    return runs, tabs


def _select(runs: pd.DataFrame) -> pd.DataFrame:
    """Add the validation-selected beta as its own variant (per dataset/seed/model/budget)."""
    base = runs[~runs.oracle]
    keys = ["dataset", "seed", "model", "budget"]
    pick = base.loc[base.sort_values("beta").groupby(keys)["val_lost"].idxmin()]  # ties -> smallest beta
    return pick.assign(variant="TVW (beta val-selected)")


def _variants(runs: pd.DataFrame) -> pd.DataFrame:
    v = runs.copy()
    v["variant"] = np.select([v.oracle, v.beta == 0, v.beta == 1], ["TVW oracle rings (beta=1)", "control (beta=0)",
                                                                     "TVW (beta=1)"], "")
    return pd.concat([v[v.variant != ""], _select(runs)], ignore_index=True)


def _paired(tabs: pd.DataFrame, var: pd.DataFrame, n_boot=2000) -> pd.DataFrame:
    """Ring-clustered paired bootstrap of lost-share(TVW) - lost-share(control), pooled over seeds."""
    rows = []
    rng = np.random.default_rng(0)
    for (ds_, model, budget), g in var.groupby(["dataset", "model", "budget"]):
        ctrl = g[g.variant == "control (beta=0)"]
        for vname in ["TVW (beta=1)", "TVW (beta val-selected)", "TVW oracle rings (beta=1)"]:
            gv = g[g.variant == vname]
            if gv.empty:
                continue
            A, B = [], []
            for seed in sorted(set(ctrl.seed)):
                cb = ctrl[ctrl.seed == seed].beta.iloc[0]
                vr = gv[gv.seed == seed].iloc[0]
                sel = (tabs.dataset == ds_) & (tabs.model == model) & (tabs.budget == budget) & (tabs.seed == seed)
                a = tabs[sel & (tabs.beta == cb) & ~tabs.oracle].set_index("ring_id")
                b = tabs[sel & (tabs.beta == vr.beta) & (tabs.oracle == vr.oracle)].set_index("ring_id")
                common = a.index.intersection(b.index)
                A.append(a.loc[common]); B.append(b.loc[common])
            A, B = pd.concat(A, ignore_index=True), pd.concat(B, ignore_index=True)
            if A.empty:
                continue
            amt, la, lb = A.ring_amount.to_numpy(), A.amount_before_alert.to_numpy(), B.amount_before_alert.to_numpy()
            d0 = (lb.sum() - la.sum()) / amt.sum()
            boots = []
            for _ in range(n_boot):
                i = rng.integers(0, len(A), len(A))
                boots.append((lb[i].sum() - la[i].sum()) / amt[i].sum())
            boots = np.array(boots)
            per_seed = g.pivot_table(index="seed", columns="variant", values="test_lost")
            wins = (per_seed[vname] < per_seed["control (beta=0)"]).mean() if vname in per_seed else np.nan
            rows.append({"dataset": ds_, "model": model, "budget": budget, "variant": vname, "n_rings": len(A),
                         "lost_control": la.sum() / amt.sum(), "lost_tvw": lb.sum() / amt.sum(),
                         "delta_pp": 100 * d0, "ci_lo_pp": 100 * np.percentile(boots, 2.5),
                         "ci_hi_pp": 100 * np.percentile(boots, 97.5), "p_not_better": float((boots >= 0).mean()),
                         "seed_win_rate": wins,
                         "relative_reduction": 1 - (lb.sum() / la.sum()) if la.sum() > 0 else np.nan})
    return pd.DataFrame(rows)


def _synthetic(tag, cfg, seeds):
    runs, tabs = [], []
    for seed in seeds:
        with Timer(f"{tag} seed {seed}"):
            ds = build_dataset(seed, cfg, label_delay_days=30.0)
            idx = TemporalIndex(ds.tx, k=20)

            def factory(name, p, s):
                if name in FEATURE_FAMILIES:
                    return tabular(FEATURE_FAMILIES[name], s, p, name)
                from src.classifier_utils import get_classifier
                return NeighborAggXGB(idx, s, classifier=get_classifier(s, **p))
            r, t = _evaluate(ds, factory, seed, tag, SYN_MODELS)
        runs += r; tabs += t
    return runs, tabs


def _aml(seeds):
    from experiments import amlworld as A
    from src.graph_features import build_graph_features
    from src.label_features import build_label_features
    from src.pipeline import Dataset, _split
    from src.preprocessing import build_transaction_features, set_payment_methods
    from src.ring_metrics import add_ring_sequence
    tx = A.load_aml()
    set_payment_methods(sorted(tx["payment_method"].unique()))
    tx = A._downcast(build_label_features(A._downcast(build_graph_features(A._downcast(build_transaction_features(tx)))), 1.0))
    tx = add_ring_sequence(tx)
    tx["_row"] = np.arange(len(tx))
    tr, va, te = _split(tx)
    ds = Dataset(tx, tr, va, te, pd.DataFrame(), 42, 1.0, {})
    idx = TemporalIndex(tx, k=10)
    listed = set(tx.loc[(tx.fraud_label == 1) & (tx.fraud_pattern != "unlisted"), "ring_id"])

    def factory(name, p, s):
        if name in FEATURE_FAMILIES:
            return tabular(FEATURE_FAMILIES[name], s, p, name)
        from src.classifier_utils import get_classifier
        return NeighborAggXGB(idx, s, classifier=get_classifier(s, **p))
    runs, tabs, cache = [], [], {}
    for seed in seeds:
        r, t = _evaluate(ds, factory, seed, "AMLworld", ["Combined (full graph)", "Combined + delayed labels",
                                                         "Neighbour-Aggregation XGBoost"],
                         oracle=False, min_txns=3, ring_filter=lambda d: d[d.ring_id.isin(listed)] if len(d) else d,
                         cache=cache, n_iter=4)
        runs += r; tabs += t
    return runs, tabs


def _ieee(seeds):
    from experiments import ieee_cis as I
    from src.label_features import LABEL_FEATURE_COLS
    raw, ident = I.load_ieee()
    ds = I.build_ieee_dataset(raw, ident, 30.0)
    ff = I.first_fraud_mask(ds.tx)
    graph_cols = FEATURE_FAMILIES["Combined (full graph)"][len(FEATURE_FAMILIES["Transaction Baseline"]):]
    fac = {"Vesta features (XGBoost)": None, "Vesta + our graph features": graph_cols,
           "Vesta + graph + delayed labels": graph_cols + LABEL_FEATURE_COLS}

    def factory(name, p, s):
        return I.VestaXGB(raw, fac[name], s, p)
    import json
    pth = os.path.join(out_dir("ieee"), "tuned_params.json")  # validation-tuned in the ieee experiment
    cache = {k: v for k, v in json.load(open(pth)).items() if k in fac} if os.path.exists(pth) else {}
    runs, tabs = [], []
    for seed in seeds:
        r, t = _evaluate(ds, factory, seed, "IEEE-CIS", list(fac), oracle=False, min_txns=3, cache=cache, n_iter=8,
                         first_mask=ff[ds.test["_row"].to_numpy()] & (ds.test.fraud_label.to_numpy() == 1))
        runs += r; tabs += t
    return runs, tabs


def run(seeds=tuple(range(42, 52)), real_seeds=(42, 43, 44), datasets=("hard", "default", "aml", "ieee"),
        out_name: str = "tvw") -> None:
    out = out_dir(out_name)
    t0 = time.time()
    runs, tabs = [], []

    def save():
        pd.DataFrame(runs).to_csv(os.path.join(out, "runs.csv"), index=False)
        pd.concat(tabs, ignore_index=True).to_csv(os.path.join(out, "ring_tables.csv"), index=False)

    jobs = {"hard": lambda: _synthetic("synthetic (hard)", GeneratorConfig(**HARD_PRESET), seeds),
            "default": lambda: _synthetic("synthetic (default)", GeneratorConfig(), seeds),
            "aml": lambda: _aml(real_seeds), "ieee": lambda: _ieee(real_seeds)}
    for d in datasets:
        try:
            with Timer(f"tvw {d}"):
                r, t = jobs[d]()
        except FileNotFoundError as e:
            print(f"skip {d}: {e}", flush=True)
            continue
        runs += r; tabs += t
        save()
    report(out)
    save_json({"seeds": list(seeds), "real_seeds": list(real_seeds), "betas": BETAS,
               "minutes": (time.time() - t0) / 60}, os.path.join(out, "run_info.json"))
    print(f"tvw done in {(time.time() - t0) / 60:.1f} min")


def merge(parts=("tvw_syn", "tvw_aml", "tvw_ieee"), out_name: str = "tvw") -> None:
    """Combine runs written by parallel lanes into results/<out_name> and rebuild the report."""
    out = out_dir(out_name)
    for f in ("runs.csv", "ring_tables.csv"):
        pd.concat([pd.read_csv(os.path.join(out_dir(p), f)) for p in parts], ignore_index=True).to_csv(
            os.path.join(out, f), index=False)
    report(out)


def report(out: str) -> None:
    runs = pd.read_csv(os.path.join(out, "runs.csv"))
    tabs = pd.read_csv(os.path.join(out, "ring_tables.csv"))
    var = _variants(runs)
    summ = var.groupby(["dataset", "model", "budget", "variant"], sort=False)[
        ["test_lost", "ring_recall", "first_txn_recall", "pr_auc", "beta"]].mean().reset_index()
    summ.to_csv(os.path.join(out, "summary.csv"), index=False)
    paired = _paired(tabs, var)
    paired.to_csv(os.path.join(out, "paired.csv"), index=False)
    curve = runs[~runs.oracle].groupby(["dataset", "model", "budget", "beta"])[
        ["test_lost", "first_txn_recall", "pr_auc"]].mean().reset_index()
    curve.to_csv(os.path.join(out, "beta_curve.csv"), index=False)

    pr = paired[(paired.budget == PRIMARY_BUDGET) & (paired.variant == "TVW (beta=1)")]
    if len(pr):
        fig, ax = plt.subplots(figsize=(8, 0.45 * len(pr) + 1.5))
        lab = pr.dataset + " | " + pr.model
        y = np.arange(len(pr))
        ax.errorbar(pr.delta_pp, y, xerr=[pr.delta_pp - pr.ci_lo_pp, pr.ci_hi_pp - pr.delta_pp], fmt="o", color="#4C72B0")
        ax.axvline(0, color="k", lw=0.8)
        ax.set_yticks(y, lab, fontsize=8)
        ax.set_xlabel("Change in share of ring money lost before first alert (pp, TVW beta=1 minus control)\n"
                      f"alert budget {PRIMARY_BUDGET:.1%} val FPR; 95% ring-clustered paired bootstrap CI")
        plt.tight_layout()
        plt.savefig(os.path.join(out, "tvw.png"), dpi=130)
        plt.close()
    with pd.option_context("display.width", 200):
        print(paired[paired.budget == PRIMARY_BUDGET].round(3).to_string(index=False))
