"""Paired ring-clustered bootstrap for ring-clock metrics when the SAME rings recur across seeds.

On real datasets every seed scores the same test rings, so (seed, ring) rows are not independent: the
independent unit is the ring. Resample ring ids with replacement and keep all of a ring's seeds.
(On the simulator each seed is a new dataset, so (seed, ring) is already the right unit.)
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def paired_ring_bootstrap(a: pd.DataFrame, b: pd.DataFrame, n_boot: int = 4000, seed: int = 0) -> dict:
    """a, b: ring tables (control, treatment) with ring_id, ring_amount, amount_before_alert,
    txns_before_alert; rows may repeat per seed. Returns money-weighted delta with CI plus count metrics."""
    A = a.groupby("ring_id")[["ring_amount", "amount_before_alert"]].sum()
    B = b.groupby("ring_id")[["amount_before_alert"]].sum().reindex(A.index)
    amt, la, lb = A.ring_amount.to_numpy(), A.amount_before_alert.to_numpy(), B.amount_before_alert.to_numpy()
    rng = np.random.default_rng(seed)
    n = len(A)
    boots = np.array([(lb[i].sum() - la[i].sum()) / amt[i].sum() for i in (rng.integers(0, n, n) for _ in range(n_boot))])
    ta = a.groupby("ring_id").txns_before_alert.mean().reindex(A.index)
    tb = b.groupby("ring_id").txns_before_alert.mean().reindex(A.index)
    top = A.ring_amount / A.ring_amount.sum()
    return {"n_rings": n, "lost_control": la.sum() / amt.sum(), "lost_tvw": lb.sum() / amt.sum(),
            "delta_pp": 100 * (lb.sum() - la.sum()) / amt.sum(),
            "ci_lo_pp": 100 * np.percentile(boots, 2.5), "ci_hi_pp": 100 * np.percentile(boots, 97.5),
            "rings_earlier": int((tb < ta - 1e-9).sum()), "rings_later": int((tb > ta + 1e-9).sum()),
            "median_txns_before_alert_control": float(ta.median()), "median_txns_before_alert_tvw": float(tb.median()),
            "largest_ring_money_share": float(top.max())}
