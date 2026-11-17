"""Transaction-level baseline classifier."""

from __future__ import annotations

from src.graph_model import TabularRiskModel
from src.preprocessing import TRANSACTION_FEATURE_COLS


class BaselineModel(TabularRiskModel):
    def __init__(self, seed: int = 42):
        super().__init__(TRANSACTION_FEATURE_COLS, name="Transaction Baseline", seed=seed)
