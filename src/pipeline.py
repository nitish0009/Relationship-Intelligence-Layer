"""Shared experiment pipeline: dataset construction, model fitting/scoring, evaluation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

from src.classifier_utils import get_classifier
from src.data_generator import GeneratorConfig, generate_all
from src.evaluation import compute_metrics, select_threshold
from src.graph_features import GRAPH_FEATURE_COLS, SIMPLE_COUNT_COLS, TEMPORAL_EXTRA_COLS, build_graph_features
from src.graph_model import GraphSAGEModel, TabularRiskModel
from src.label_features import LABEL_FEATURE_COLS, build_label_features
from src.preprocessing import TRANSACTION_FEATURE_COLS, build_transaction_features, chronological_split
from src.ring_metrics import add_ring_sequence, ring_detection_table, summarise_rings, warmup_curve

T = TRANSACTION_FEATURE_COLS
FEATURE_FAMILIES: Dict[str, List[str]] = {
    "Transaction Baseline": T,
    "+ Relationship counts": T + SIMPLE_COUNT_COLS,
    "+ Temporal relationships": T + SIMPLE_COUNT_COLS + TEMPORAL_EXTRA_COLS,
    "Combined (full graph)": T + GRAPH_FEATURE_COLS,
    "Combined + delayed labels": T + GRAPH_FEATURE_COLS + LABEL_FEATURE_COLS,
}


@dataclass
class Dataset:
    tx: pd.DataFrame
    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame
    rings: pd.DataFrame
    seed: int
    label_delay_days: Optional[float] = None
    extras: dict = field(default_factory=dict)


def _split(tx: pd.DataFrame):
    return chronological_split(tx)


def build_dataset(seed: int = 42, config: Optional[GeneratorConfig] = None,
                  label_delay_days: Optional[float] = 30.0, save_dir: Optional[str] = None) -> Dataset:
    data = generate_all(save_dir, seed=seed, config=config)
    tx = build_graph_features(build_transaction_features(data["transactions"]))
    tx = build_label_features(tx, label_delay_days if label_delay_days is not None else float("inf"))
    tx = add_ring_sequence(tx)
    tx["_row"] = np.arange(len(tx))
    train, val, test = _split(tx)
    return Dataset(tx, train, val, test, data["rings"], seed, label_delay_days,
                   {"customers": data["customers"], "ips": data["ips"]})


def random_split(ds: Dataset, seed: int = 0) -> Dataset:
    """'Published-style' random 70/15/15 split of the same featurised rows (C3 contrast).

    Features stay strictly past-only; the leak is that train and test interleave in time,
    so test rings are mostly also present in training."""
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(ds.tx))
    a, b = int(0.70 * len(perm)), int(0.85 * len(perm))
    pick = lambda idx: ds.tx.iloc[np.sort(idx)].copy()
    return Dataset(ds.tx, pick(perm[:a]), pick(perm[a:b]), pick(perm[b:]), ds.rings, ds.seed,
                   ds.label_delay_days, ds.extras)


def with_label_delay(ds: Dataset, delay_days: float) -> Dataset:
    """Recompute only the delayed-label features (cheap) for a new label delay."""
    tx = build_label_features(ds.tx, delay_days)
    train, val, test = _split(tx)
    return Dataset(tx, train, val, test, ds.rings, ds.seed, delay_days, ds.extras)


# ----------------------------------------------------------------------------- tuning

XGB_SPACE = {
    "n_estimators": [200, 400, 800],
    "max_depth": [3, 4, 6, 8],
    "learning_rate": [0.03, 0.05, 0.08, 0.15],
    "subsample": [0.7, 0.8, 1.0],
    "colsample_bytree": [0.6, 0.8, 1.0],
    "min_child_weight": [1, 5, 10],
    "reg_lambda": [0.0, 1.0, 5.0],
}


def tune(factory: Callable[[dict], object], ds: Dataset, space: Dict[str, list], n_iter: int = 20,
         seed: int = 42) -> dict:
    """Random search on VALIDATION PR-AUC (test is never touched). The first candidate is the
    default configuration ({}), so tuning can never do worse than the default on validation."""
    rng = np.random.default_rng(seed)
    y_tr, y_va = ds.train["fraud_label"].to_numpy(), ds.val["fraud_label"].to_numpy()
    best, best_params = -1.0, {}
    tried = set()
    for i in range(n_iter):
        params = {} if i == 0 else {k: v[int(rng.integers(len(v)))] for k, v in space.items()}
        key = tuple(sorted(params.items()))
        if key in tried:
            continue
        tried.add(key)
        m = factory(params)
        if hasattr(m, "_stream"):
            m.fit(ds.train, y_tr, full_df=ds.tx)
        else:
            m.fit(ds.train, y_tr)
        ap = average_precision_score(y_va, m.predict_proba(ds.val, history=ds.train))
        if ap > best:
            best, best_params = ap, params
    return best_params


def tune_xgb(cols: List[str], ds: Dataset, n_iter: int = 20, seed: int = 42) -> dict:
    return tune(lambda p: tabular(cols, seed, p), ds, XGB_SPACE, n_iter, seed)


def threshold_at_fpr(y_val: np.ndarray, p_val: np.ndarray, fpr: float) -> float:
    """Alert-budget threshold: the score above which a fraction ``fpr`` of VALIDATION
    legitimate transactions would be alerted (capacity-constrained operation)."""
    neg = np.sort(p_val[y_val == 0])[::-1]
    return float(neg[min(len(neg) - 1, int(fpr * len(neg)))])


# ----------------------------------------------------------------------------- fit / score

def fit_score(model, ds: Dataset) -> Dict[str, object]:
    """Fit on train, tune the threshold on validation, score test. Uniform model interface."""
    y_tr = ds.train["fraud_label"].to_numpy()
    if hasattr(model, "_stream"):  # TGN-style models need the full chronological stream
        model.fit(ds.train, y_tr, full_df=ds.tx)
    else:
        model.fit(ds.train, y_tr)
    p_val = model.predict_proba(ds.val, history=ds.train)
    p_test = model.predict_proba(ds.test, history=pd.concat([ds.train, ds.val]))
    thr = select_threshold(ds.val["fraud_label"].to_numpy(), p_val)
    return {"p_val": p_val, "p_test": p_test, "threshold": thr}


def evaluate(ds: Dataset, out: Dict[str, object]) -> Dict[str, float]:
    y = ds.test["fraud_label"].to_numpy()
    m = compute_metrics(y, out["p_test"], out["threshold"])
    rings = summarise_rings(ring_detection_table(ds.test, out["p_test"], out["threshold"]))
    return {**m, **{f"ring_{k}": v for k, v in rings.items()}}


def tabular(cols: List[str], seed: int = 42, params: Optional[dict] = None, name: str = "model") -> TabularRiskModel:
    m = TabularRiskModel(cols, name=name, seed=seed)
    if params:
        m.model = get_classifier(seed, **params)
    return m
