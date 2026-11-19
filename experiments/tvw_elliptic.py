"""Pre-registered TVW test on Elliptic++ (eligible: control loses 85-100% of multi-transaction ring money,
results/headroom/). Same validation-tuned XGBoost, same seed, with vs without TVW (beta = 1, ESS tempering).
Training rings = components of TRAINING illicit transactions only. Paired ring-clustered bootstrap CIs on the
money-lost share at prevalence-matched budgets; 5 seeds. Every outcome is reported.

Outputs (results/tvw_elliptic/): runs.csv, ring_tables.csv, paired.csv
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

from experiments.common import out_dir
from experiments.headroom import BUDGET_MULTIPLES, MIN_RING, threshold_at_multiple
from src.classifier_utils import get_classifier
from src.clock_weights import tvw_weights
from src.pipeline import XGB_SPACE
from src.ring_metrics import ring_detection_table


def run(seeds=(42, 43, 44, 45, 46)) -> None:
    from experiments import elliptic as E
    out = out_dir("tvw_elliptic")
    df = E.load()
    tr, va, te = E.split(df)
    cols = E.feature_cols(df)
    Xtr, Xva, Xte = (d[cols].to_numpy(np.float32) for d in (tr, va, te))
    ytr, yva, yte = tr.fraud_label.to_numpy(), va.fraud_label.to_numpy(), te.fraud_label.to_numpy()
    rng = np.random.default_rng(0)
    best, params = -1, {}
    for i in range(10):  # tuned once on validation for the CONTROL (shared by TVW)
        p = {} if i == 0 else {k: v[rng.integers(len(v))] for k, v in XGB_SPACE.items()}
        ap = average_precision_score(yva, get_classifier(42, **p).fit(Xtr, ytr).predict_proba(Xva)[:, 1])
        if ap > best:
            best, params = ap, p
    ring = E.train_rings(tr, df.attrs["edges"])
    w = tvw_weights(tr, 1.0, ring=ring)
    f = ytr == 1
    print(f"train rings: {len(np.unique(ring[ring >= 0]))}, ESS share {w[f].sum()**2 / (w[f]**2).sum() / f.sum():.2f}",
          flush=True)
    runs, tabs = [], []
    first = (te.ring_seq.to_numpy() == 0) & (yte == 1)
    for seed in seeds:
        for variant, sw in [("control", None), ("TVW", w)]:
            m = get_classifier(seed, **params).fit(Xtr, ytr, sample_weight=sw)
            pv, pt = m.predict_proba(Xva)[:, 1], m.predict_proba(Xte)[:, 1]
            for c in BUDGET_MULTIPLES:
                thr = threshold_at_multiple(yva, pv, c)
                t = ring_detection_table(te, pt, thr)
                t = t[t.ring_txns >= MIN_RING]
                runs.append({"seed": seed, "variant": variant, "budget_multiple": c, "pr_auc": average_precision_score(yte, pt),
                             "lost_share": float(t.amount_before_alert.sum() / t.ring_amount.sum()),
                             "first_txn_recall": float((pt[first] >= thr).mean()), "n_rings": len(t)})
                tabs.append(t.assign(seed=seed, variant=variant, budget_multiple=c))
        print(f"   seed {seed} done", flush=True)
    r, rt = pd.DataFrame(runs), pd.concat(tabs, ignore_index=True)
    r.to_csv(os.path.join(out, "runs.csv"), index=False)
    rt.to_csv(os.path.join(out, "ring_tables.csv"), index=False)
    from src.ring_bootstrap import paired_ring_bootstrap  # rings recur across seeds: resample RINGS
    rows = []
    for c in BUDGET_MULTIPLES:
        g_ = rt[rt.budget_multiple == c]
        rows.append({"budget_multiple": c, **paired_ring_bootstrap(g_[g_.variant == "control"], g_[g_.variant == "TVW"])})
    p = pd.DataFrame(rows)
    p.to_csv(os.path.join(out, "paired.csv"), index=False)
    print(p.round(3).to_string(index=False))
    print(r.groupby(["variant", "budget_multiple"])[["pr_auc", "first_txn_recall"]].mean().round(4).to_string())
