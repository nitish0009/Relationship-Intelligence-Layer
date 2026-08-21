"""Shared classifier factory with XGBoost fallback."""

from __future__ import annotations

from typing import Any

from sklearn.ensemble import HistGradientBoostingClassifier

HAS_XGB = False
_xgb = None

try:
    import xgboost as xgb

    _ = xgb.XGBClassifier()
    HAS_XGB = True
    _xgb = xgb
except Exception:
    HAS_XGB = False


def get_classifier(**kwargs) -> Any:
    if HAS_XGB and _xgb is not None:
        return _xgb.XGBClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.08,
            subsample=0.8,
            colsample_bytree=0.8,
            eval_metric="logloss",
            random_state=42,
            n_jobs=-1,
            **kwargs,
        )
    return HistGradientBoostingClassifier(
        max_iter=200,
        max_depth=6,
        learning_rate=0.08,
        random_state=42,
        **kwargs,
    )


def classifier_name() -> str:
    return "XGBoost" if HAS_XGB else "HistGradientBoostingClassifier"
