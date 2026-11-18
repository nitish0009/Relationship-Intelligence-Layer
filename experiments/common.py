"""Model registry and shared evaluation helpers for all experiments."""

from __future__ import annotations

import json
import os
import time
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from src.evaluation import RESULTS_DIR, compute_metrics
from src.graph_model import GraphSAGEModel, GraphSAGETransductive
from src.pipeline import FEATURE_FAMILIES, XGB_SPACE, Dataset, fit_score, tabular, threshold_at_fpr, tune
from src.ring_metrics import ring_detection_table, summarise_rings, warmup_curve
from src.original_gnn import OriginalFraudGNN, third_party_available
from src.temporal_gnn import NeighborAggXGB, TemporalGNN, TemporalIndex, TGNMemoryModel

BUDGETS = [0.001, 0.005, 0.01]  # alert budgets = validation FPR (0.1%, 0.5%, 1%)
PRIMARY_BUDGET = 0.005

TABULAR = list(FEATURE_FAMILIES)  # feature-family ablation (all gradient-boosted trees)
GNNS = ["Temporal GraphSAGE", "Temporal GAT", "Temporal R-GCN", "CARE-style GNN", "TGN-style Memory"]
GNN_KIND = {"Temporal GraphSAGE": "sage", "Temporal GAT": "gat", "Temporal R-GCN": "rgcn", "CARE-style GNN": "care"}
GNN_SPACE = {"hidden": [32, 64], "lr": [1e-3, 3e-3]}
TGN_SPACE = {"mem_dim": [32, 64], "lr": [1e-3, 3e-3]}

# Original third-party implementations run only when fetched (scripts/fetch_third_party.sh)
# AND explicitly enabled with RIL_RUN_THIRD_PARTY=1 (they execute external code).
ORIGINALS = ["CARE-GNN (original code)", "PC-GNN (original code)"]


def originals_enabled() -> bool:
    return os.environ.get("RIL_RUN_THIRD_PARTY") == "1" and third_party_available()


def out_dir(name: str) -> str:
    d = os.path.join(RESULTS_DIR, name)
    os.makedirs(d, exist_ok=True)
    return d


def make_model(name: str, ds: Dataset, seed: int, params: Optional[dict] = None,
               idx: Optional[TemporalIndex] = None):
    params = params or {}
    if name in FEATURE_FAMILIES:
        return tabular(FEATURE_FAMILIES[name], seed, params, name=name)
    if name == "Neighbour-Aggregation XGBoost":
        from src.classifier_utils import get_classifier
        return NeighborAggXGB(idx, seed=seed, classifier=get_classifier(seed, **params))
    if name in GNN_KIND:
        return TemporalGNN(idx, GNN_KIND[name], seed=seed, val_df=ds.val, **params)
    if name == "TGN-style Memory":
        return TGNMemoryModel(idx, seed=seed, val_df=ds.val, **params)
    if name == "TGN (PyG reference)":
        from src.tgn_pyg import TGNReference
        return TGNReference(idx, seed=seed, val_df=ds.val, **params)
    if name in ORIGINALS:
        kind = "care" if name.startswith("CARE") else "pcgnn"
        return OriginalFraudGNN(idx, kind, ds.tx, ds.val, seed=seed)
    if name == "GraphSAGE (entity, inductive)":
        return GraphSAGEModel(seed=seed)
    if name == "GraphSAGE (transductive)":
        return GraphSAGETransductive(ds.tx, seed=seed)
    raise KeyError(name)


def tune_model(name: str, ds: Dataset, seed: int, idx: TemporalIndex, n_iter: int = 15) -> dict:
    if name in FEATURE_FAMILIES or name == "Neighbour-Aggregation XGBoost":
        return tune(lambda p: make_model(name, ds, seed, p, idx), ds, XGB_SPACE, n_iter, seed)
    if name in GNN_KIND:
        return tune(lambda p: make_model(name, ds, seed, p, idx), ds, GNN_SPACE, 8, seed)
    if name == "TGN-style Memory":
        return tune(lambda p: make_model(name, ds, seed, p, idx), ds, TGN_SPACE, 8, seed)
    return {}


def score_all(ds: Dataset, out: Dict[str, object]) -> Dict[str, object]:
    """Thresholds (val F1 + alert budgets) and test metrics for one fitted model."""
    yv = ds.val["fraud_label"].to_numpy()
    yt = ds.test["fraud_label"].to_numpy()
    res = {"row": compute_metrics(yt, out["p_test"], out["threshold"]), "thr": {"f1": out["threshold"]}}
    for b in BUDGETS:
        res["thr"][b] = threshold_at_fpr(yv, out["p_val"], b)
    return res


def ring_rows(ds: Dataset, p_test: np.ndarray, thr: float, **tags) -> pd.DataFrame:
    tab = ring_detection_table(ds.test, p_test, thr)
    if tab.empty:
        return tab
    tab = tab.merge(ds.rings[["ring_id", "lifetime_days", "hidden_share", "adapted"]], on="ring_id", how="left")
    for k, v in tags.items():
        tab[k] = v
    return tab


def save_json(obj, path: str) -> None:
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=float)


class Timer:
    def __init__(self, label: str):
        self.label = label

    def __enter__(self):
        self.t = time.time()
        return self

    def __exit__(self, *a):
        print(f"  [{self.label}] {time.time() - self.t:.1f}s", flush=True)
