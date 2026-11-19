"""Ring-survival loss: optimising the expected takedown value directly.

For a ring r with transactions k = 1..n (time order) and amounts a_k, let R_k = sum_{j>=k} a_j be the
money still to be moved from k onwards and T = R_1 the ring total. If the ring is taken down at its first
alert and transaction k is alerted with probability p_k (independently given the model scores), then

    P(first alert at k) = p_k * prod_{j<k} (1 - p_j)                          (survival form)
    V_r / T_r          = sum_k P(first alert at k) * R_k / T_r   in [0, 1]       (share of ring money saved)

TVW (src/clock_weights.py) maximises the first-order upper bound sum_k p_k R_k, which keeps pushing late
transactions up even when an early one is already likely to be caught. The ring-survival loss maximises
V_r / T_r exactly; its gradient on a late transaction is discounted by the probability that the ring
survived until then. Normalising by T_r makes the term scale-free within a ring (no heavy-tail weights,
hence no ESS tempering), and rings are weighted by T_r^gamma_ring (gamma_ring = 0: every ring counts
equally; 1: money-weighted).

The term only involves fraud rows; legitimate rows (false positives) are controlled by the ordinary BCE
term it is added to:  L = BCE + lambda * mean_r w_r (1 - V_r / T_r).
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F

# Single-threaded torch, as in graph_model.py / temporal_gnn.py: once XGBoost's libomp is loaded, a
# multi-threaded torch kernel (e.g. logsigmoid) segfaults on macOS.
torch.set_num_threads(1)


class RingIndex:
    """Flat, time-sorted layout of training fraud rows grouped by (pseudo-)ring.

    ``rows`` are positions into the training arrays; within each ring they are in time order.
    """

    def __init__(self, ring: np.ndarray, time: np.ndarray, amount: np.ndarray, gamma_ring: float = 0.0):
        f = np.flatnonzero(ring >= 0)
        order = np.lexsort((f, time[f], ring[f]))          # by ring, then time, then position
        self.rows = f[order]
        r = ring[self.rows]
        self.n_rings = int(len(np.unique(r)))
        starts = np.r_[0, np.flatnonzero(np.diff(r)) + 1]
        self.seg = np.zeros(len(self.rows), dtype=np.int64)  # ring segment id (0..n_rings-1) per row
        self.seg[starts[1:]] = 1
        self.seg = np.cumsum(self.seg)
        self.starts = starts
        a = amount[self.rows].astype(np.float64)
        # R_k / T_r : remaining share of the ring's money at and after k
        rev = np.zeros_like(a)
        for s, e in zip(starts, np.r_[starts[1:], len(a)]):
            c = np.cumsum(a[s:e][::-1])[::-1]
            rev[s:e] = c / max(c[0], 1e-12)
        self.remaining_share = rev
        tot = np.array([a[s:e].sum() for s, e in zip(starts, np.r_[starts[1:], len(a)])])
        w = tot ** gamma_ring
        self.ring_weight = w / w.mean()

    def batches(self, rng: np.random.Generator, max_rows: int):
        """Yield arrays of segment ids whose rows total about ``max_rows`` (whole rings only)."""
        sizes = np.diff(np.r_[self.starts, len(self.rows)])
        perm = rng.permutation(self.n_rings)
        cur, n = [], 0
        for s in perm:
            cur.append(s)
            n += sizes[s]
            if n >= max_rows:
                yield np.array(cur)
                cur, n = [], 0
        if cur:
            yield np.array(cur)

    def select(self, segs: np.ndarray, rng: Optional[np.random.Generator] = None):
        """Flat positions (into self.rows) of the chosen rings, kept ring-contiguous and in time order.
        With ``rng``, each ring is truncated to a random prefix of length m ~ Uniform{1..n} (prefix training:
        at deployment a ring is only observed up to 'now', so later, easier transactions cannot be relied on)."""
        sizes = np.diff(np.r_[self.starts, len(self.rows)])
        use = sizes[segs] if rng is None else rng.integers(1, sizes[segs] + 1)
        idx = np.concatenate([np.arange(self.starts[s], self.starts[s] + m) for s, m in zip(segs, use)])
        new_seg = np.repeat(np.arange(len(segs)), use)
        return idx, new_seg


def takedown_share(logits: torch.Tensor, seg: torch.Tensor, remaining: torch.Tensor, n_seg: int) -> torch.Tensor:
    """V_r / T_r for each ring segment. ``logits``, ``seg``, ``remaining`` are flat, ring-contiguous and
    time-ordered within each ring."""
    remaining = remaining.to(logits.dtype)
    log_surv_step = F.logsigmoid(-logits)                       # log(1 - p_k)
    csum = torch.cumsum(log_surv_step, 0)
    # exclusive cumulative sum within each ring: subtract the cumsum at the ring's start
    seg_start_csum = torch.zeros(n_seg, dtype=csum.dtype, device=csum.device)
    first = torch.ones_like(seg, dtype=torch.bool)
    first[1:] = seg[1:] != seg[:-1]
    seg_start_csum[seg[first]] = (csum - log_surv_step)[first]
    log_surv_before = csum - log_surv_step - seg_start_csum[seg]  # sum_{j<k} log(1 - p_j)
    q = torch.sigmoid(logits) * torch.exp(log_surv_before)     # P(first alert at k)
    v = torch.zeros(n_seg, dtype=logits.dtype, device=logits.device)
    return v.index_add(0, seg, q * remaining)


def ring_survival_loss(logits: torch.Tensor, seg: torch.Tensor, remaining: torch.Tensor,
                       ring_weight: torch.Tensor) -> torch.Tensor:
    """mean_r w_r * (1 - V_r / T_r)  -- the share of ring money NOT saved."""
    v = takedown_share(logits, seg, remaining, len(ring_weight))
    return (ring_weight * (1 - v)).mean()


class RingSurvivalModel:
    """Wraps a TemporalGNN: ordinary BCE + lam * ring-survival loss on pseudo-rings (no ring labels).
    ``oracle=True`` uses the true ring ids instead (simulator only)."""

    def __init__(self, base, lam: float = 1.0, gamma_ring: float = 0.0, oracle: bool = False,
                 max_customers: int = 10, prefix: bool = False):
        self.base, self.lam, self.gamma_ring, self.oracle, self.max_customers = base, lam, gamma_ring, oracle, max_customers
        self.prefix = prefix
        self.name = getattr(base, "name", "model") + f" + ring-survival(lam={lam})"

    def fit(self, train_df, y, sample_weight=None):
        from src.clock_weights import pseudo_rings
        if self.oracle:
            ring = np.where(train_df["fraud_label"].to_numpy() == 1, train_df["ring_id"].to_numpy(), -1).astype(np.int64)
        else:
            ring = pseudo_rings(train_df, max_customers=self.max_customers)
        t = train_df["timestamp"].to_numpy().astype("datetime64[ns]").astype(np.int64)
        ix = RingIndex(ring, t, train_df["amount"].to_numpy(np.float64), self.gamma_ring)
        self.base.fit(train_df, y, sample_weight=sample_weight, ring_info={"index": ix, "lam": self.lam,
                                                                             "prefix": self.prefix})
        return self

    def predict_proba(self, df, history=None):
        return self.base.predict_proba(df, history=history)
