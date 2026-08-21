"""Baseline transaction-level fraud classifier."""

from __future__ import annotations

import os
from typing import Any

import joblib
import numpy as np
import pandas as pd

from src.classifier_utils import get_classifier
from src.preprocessing import (
    TRANSACTION_FEATURE_COLS,
    prepare_feature_matrix,
)


class BaselineModel:
    def __init__(self):
        self.model = get_classifier()
        self.encoders: dict = {}
        self.feature_cols = TRANSACTION_FEATURE_COLS

    def fit(self, df: pd.DataFrame, y: np.ndarray) -> "BaselineModel":
        X, self.encoders = prepare_feature_matrix(df, self.feature_cols)
        self.model.fit(X, y)
        return self

    def predict_proba(self, df) -> np.ndarray:
        X, _ = prepare_feature_matrix(df, self.feature_cols, self.encoders)
        return self.model.predict_proba(X)[:, 1]

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump({"model": self.model, "encoders": self.encoders, "feature_cols": self.feature_cols}, path)

    @classmethod
    def load(cls, path: str) -> "BaselineModel":
        data = joblib.load(path)
        obj = cls()
        obj.model = data["model"]
        obj.encoders = data["encoders"]
        obj.feature_cols = data["feature_cols"]
        return obj
