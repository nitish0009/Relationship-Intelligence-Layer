"""Takedown-value weighting (TVW): training a fraud model on the ring's clock.

Row-level training treats every fraud transaction as equally valuable. Operationally they are
not: if a ring is taken down at its first alert, alerting its k-th transaction saves all the money
the ring would still move from k onwards, R_k = sum_{j >= k} amount_j. Early transactions are worth
the most to catch, and they are also the hardest: the ring's relational footprint is still small.

Derivation. Let p_k be the probability that transaction k of a ring is alerted. The money a
takedown policy saves on a ring is  V = sum_k P(first alert at k) * R_k,  and since
P(first alert at k) <= p_k,  V <= sum_k p_k R_k.  Maximising this linear upper bound is, up to the
link function, a weighted likelihood in which fraud example k carries weight proportional to R_k.
TVW therefore trains the usual classifier with per-example weights

    w_k = (1 - beta) + beta * R_k / mean(R)     for fraud rows,   w = 1 for legitimate rows,

tempered as  rel_k = (R_k / mean R)^gamma  with the largest gamma in [0, 1] whose effective
sample size  ESS = (sum w)^2 / sum w^2  is at least ``min_ess`` (default 0.5) of the fraud count
(self-normalised importance weighting; Kong 1992, Owen 2013). Heavy-tailed amounts (AMLworld:
amount CV 35, raw ESS = 4 of 2,856) would otherwise put almost all weight on a few transactions;
on the synthetic data raw ESS is already ~55%, so gamma = 1 there. The weights are then
renormalised so the mean fraud weight stays 1 (class balance is unchanged, only WHICH fraud
matters changes). beta is chosen on the validation split by the ring-level objective (money lost
before the first alert at the primary alert budget) -- beta = 0 is the ordinary model, so
selection can always fall back to it.

Ring membership is NOT taken from ground truth. Training fraud is grouped into pseudo-rings by
union-find over shared SPECIFIC entities (a device / IP / card / account used by at most
``max_customers`` distinct customers in the training window, so carrier NAT, public Wi-Fi and
coarse fingerprints cannot merge unrelated fraud). Only training labels and training-window
entity statistics are used, so the method applies unchanged to datasets without ring labels
(IEEE-CIS, AMLworld). An oracle variant using true ring ids measures the pseudo-ring gap.
"""

from __future__ import annotations

from typing import Iterable, Optional

import numpy as np
import pandas as pd

LINK_COLS = ("customer_id", "device_id", "card_id", "ip_id")


def pseudo_rings(train: pd.DataFrame, cols: Iterable[str] = LINK_COLS, max_customers: int = 10,
                 identity_cols: Iterable[str] = ("customer_id",), identity_key=None) -> np.ndarray:
    """Pseudo-ring id per row of ``train`` (-1 for legitimate rows).

    ``identity_key`` (optional) maps an identity value to a role-free key, so that the same entity links
    across roles (e.g. on AMLworld an account is the receiver 'R<bank|acct>' of one transfer and the sender
    'S<bank|acct>' of the next; identity_key strips the role prefix). Identity columns then share one
    namespace.

    IDENTITY columns (accounts / customers) always link fraud rows; INFRASTRUCTURE columns (devices, IPs,
    cards) link only through specific values used by <= max_customers customers, so shared NATs, Wi-Fi and
    coarse fingerprints cannot merge unrelated fraud. Which columns are identities is a property of the
    dataset schema (e.g. on AMLworld the 'device' slot holds the receiving ACCOUNT)."""
    identity_cols = set(identity_cols)
    shared_first: dict = {}  # one namespace for all identity columns when identity_key is given
    y = train["fraud_label"].to_numpy()
    fraud = np.flatnonzero(y == 1)
    parent = np.arange(len(fraud))

    def find(a: int) -> int:
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for col in cols:
        if col not in train.columns:
            continue
        shared = identity_key is not None and col in identity_cols
        if col in identity_cols:
            ok = None
        else:
            n_cust = train.groupby(col, sort=False)["customer_id"].nunique()
            ok = set(n_cust.index[n_cust <= max_customers])
        first: dict = shared_first if shared else {}
        for j, v in enumerate(train[col].to_numpy()[fraud]):
            if shared:
                v = identity_key(v)
            if ok is not None and v not in ok:
                continue
            if v in first:
                ra, rb = find(j), find(first[v])
                if ra != rb:
                    parent[ra] = rb
            else:
                first[v] = j
    roots = np.array([find(j) for j in range(len(fraud))], dtype=np.int64)
    ring = np.full(len(train), -1, dtype=np.int64)
    ring[fraud] = np.unique(roots, return_inverse=True)[1]
    return ring


def remaining_value(train: pd.DataFrame, ring: np.ndarray) -> np.ndarray:
    """R_k: money the ring moves from this transaction onwards (0 for legitimate rows)."""
    out = np.zeros(len(train), dtype=np.float64)
    m = ring >= 0
    if not m.any():
        return out
    d = pd.DataFrame({"r": ring[m], "t": train["timestamp"].to_numpy()[m],
                      "a": train["amount"].to_numpy(np.float64)[m], "i": np.flatnonzero(m)})
    d = d.sort_values(["r", "t", "i"], kind="stable")
    # reverse cumulative sum within ring = amount at and after k
    d["R"] = d.iloc[::-1].groupby("r", sort=False)["a"].cumsum().iloc[::-1]
    out[d["i"].to_numpy()] = d["R"].to_numpy()
    return out


def _ess_share(w: np.ndarray) -> float:
    return float(w.sum() ** 2 / (w ** 2).sum() / len(w))


def temper(rel: np.ndarray, min_ess: float) -> tuple[np.ndarray, float]:
    """Largest gamma in [0, 1] with ESS(rel^gamma) >= min_ess * n (bisection; ESS falls with gamma)."""
    if min_ess <= 0 or _ess_share(rel) >= min_ess:
        return rel, 1.0
    lo, hi = 0.0, 1.0
    for _ in range(30):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if _ess_share(rel ** mid) >= min_ess else (lo, mid)
    return rel ** lo, lo


def ring_totals(train: pd.DataFrame, ring: np.ndarray) -> np.ndarray:
    """T_r of each row's ring (0 for legitimate rows)."""
    out = np.zeros(len(train))
    m = ring >= 0
    tot = pd.Series(train["amount"].to_numpy(np.float64)[m]).groupby(ring[m]).transform("sum").to_numpy()
    out[m] = tot
    return out


def temper_rings(within: np.ndarray, ring_money: np.ndarray, min_ess: float) -> tuple[np.ndarray, float]:
    """u = within * (T / mean T)^gamma with the largest gamma in [0, 1] such that ESS(u) >= min_ess * n.
    Only the ACROSS-ring money factor is tempered; the within-ring clock R_k / T_r in (0, 1] is kept intact."""
    t = ring_money / max(ring_money.mean(), 1e-12)
    if min_ess <= 0 or _ess_share(within * t) >= min_ess:
        return within * t, 1.0
    lo, hi = 0.0, 1.0
    for _ in range(30):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if _ess_share(within * t ** mid) >= min_ess else (lo, mid)
    return within * t ** lo, lo


def tvw_weights(train: pd.DataFrame, beta: float, ring: Optional[np.ndarray] = None,
                max_customers: int = 10, min_ess: float = 0.5, tempering: str = "ring",
                identity_cols: Iterable[str] = ("customer_id",), identity_key=None) -> np.ndarray:
    """tempering='ring' (default, v2): w ~ (R_k / T_r) * T_r^gamma, tempering only the across-ring money
    factor; identical to 'global' when gamma = 1. tempering='global' (v1): w ~ (R_k / mean R)^gamma, which
    also flattens the within-ring clock (kept for the ablation)."""
    y = train["fraud_label"].to_numpy()
    w = np.ones(len(train), dtype=np.float64)
    if beta <= 0 or not (y == 1).any():
        return w
    if ring is None:
        ring = pseudo_rings(train, max_customers=max_customers, identity_cols=identity_cols,
                            identity_key=identity_key)
    R = remaining_value(train, ring)
    f = y == 1
    if tempering == "ring":
        T = ring_totals(train, ring)
        rel, _ = temper_rings(R[f] / np.maximum(T[f], 1e-12), T[f], min_ess)
        rel = rel / max(rel.mean(), 1e-12)
    else:
        rel, _ = temper(R[f] / max(R[f].mean(), 1e-12), min_ess)
    wf = (1 - beta) + beta * rel
    w[f] = wf / wf.mean()
    return w


class TVWModel:
    """Wraps any model whose ``fit`` accepts ``sample_weight`` (XGBoost-based models here)."""

    def __init__(self, base, beta: float, oracle: bool = False, max_customers: int = 10, min_ess: float = 0.5,
                 tempering: str = "ring", identity_cols: Iterable[str] = ("customer_id",), identity_key=None):
        self.base, self.beta, self.oracle, self.max_customers = base, beta, oracle, max_customers
        self.min_ess, self.tempering, self.identity_cols = min_ess, tempering, tuple(identity_cols)
        self.identity_key = identity_key
        self.name = getattr(base, "name", "model") + f" + TVW(beta={beta})"

    def fit(self, train_df: pd.DataFrame, y: np.ndarray) -> "TVWModel":
        ring = None
        if self.oracle:
            ring = np.where(train_df["fraud_label"].to_numpy() == 1, train_df["ring_id"].to_numpy(), -1).astype(np.int64)
        w = tvw_weights(train_df, self.beta, ring=ring, max_customers=self.max_customers, min_ess=self.min_ess,
                        tempering=self.tempering, identity_cols=self.identity_cols, identity_key=self.identity_key)
        self.base.fit(train_df, y, sample_weight=w)
        return self

    def predict_proba(self, df: pd.DataFrame, history=None) -> np.ndarray:
        return self.base.predict_proba(df, history=history)
