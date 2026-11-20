"""Training-objective baselines that TVW must beat (it is not "just reweighting").

* class-balanced  -- inverse class frequency: total fraud weight = total legitimate weight
* amount-weighted -- example-dependent cost-sensitive learning (Bahnsen et al., 2015): a fraud example
                     costs its own amount; tempered with the same ESS rule as TVW for a fair comparison
* focal (GNNs)    -- focal loss (Lin et al., 2017), gamma = 2: down-weights easy examples

All return per-row weights for the TRAINING frame (legitimate rows weight 1 unless stated) and are wrapped
by WeightedModel, mirroring TVWModel's interface so every model accepts them unchanged.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.clock_weights import temper


def class_balanced_weights(train: pd.DataFrame) -> np.ndarray:
    y = train["fraud_label"].to_numpy()
    n, p = len(y), max(int(y.sum()), 1)
    w = np.where(y == 1, n / (2 * p), n / (2 * max(n - p, 1)))
    return (w / w.mean()).astype(np.float64)


def amount_weights(train: pd.DataFrame, min_ess: float = 0.5) -> np.ndarray:
    y = train["fraud_label"].to_numpy()
    w = np.ones(len(train), dtype=np.float64)
    f = y == 1
    if f.any():
        a = train["amount"].to_numpy(np.float64)[f]
        rel, _ = temper(a / max(a.mean(), 1e-12), min_ess)
        w[f] = rel / rel.mean()
    return w


SCHEMES = {"class-balanced": class_balanced_weights, "amount-weighted": amount_weights}


class WeightedModel:
    """Wraps any model whose fit accepts sample_weight; ``scheme`` in SCHEMES."""

    def __init__(self, base, scheme: str):
        self.base, self.scheme = base, scheme
        self.name = getattr(base, "name", "model") + f" + {scheme}"

    def fit(self, train_df, y, sample_weight=None):
        self.base.fit(train_df, y, sample_weight=SCHEMES[self.scheme](train_df))
        return self

    def predict_proba(self, df, history=None):
        return self.base.predict_proba(df, history=history)
