"""Shared gradient-boosting classifier factory (XGBoost, sklearn fallback)."""

from __future__ import annotations

from typing import Any

from sklearn.ensemble import HistGradientBoostingClassifier

RANDOM_SEED = 42

try:
    import xgboost as _xgb

    HAS_XGB = True
except Exception:  # pragma: no cover - fallback path
    _xgb = None
    HAS_XGB = False


DEFAULT_PARAMS = dict(n_estimators=200, max_depth=6, learning_rate=0.08, subsample=0.8, colsample_bytree=0.8)


def get_classifier(seed: int = RANDOM_SEED, **overrides) -> Any:
    """Deterministic for a given seed (XGBoost 'hist' on CPU is deterministic)."""
    if HAS_XGB:
        params = {**DEFAULT_PARAMS, **overrides}
        return _xgb.XGBClassifier(**params, tree_method="hist", eval_metric="logloss", random_state=seed, n_jobs=4)
    return HistGradientBoostingClassifier(max_iter=200, max_depth=6, learning_rate=0.08, random_state=seed)


def classifier_name() -> str:
    return "XGBoost" if HAS_XGB else "HistGradientBoostingClassifier"
