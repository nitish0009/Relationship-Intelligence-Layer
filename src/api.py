"""FastAPI service for relationship risk analysis."""

from __future__ import annotations

import os
from typing import Optional

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from src.baseline_model import BaselineModel
from src.explainability import _percentile_thresholds, decision_from_scores, explain_transaction
from src.graph_model import CombinedModel, RelationshipModel

PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))
DATA_PATH = os.path.join(PROJECT_ROOT, "data", "transactions.csv")
MODEL_DIR = os.path.join(PROJECT_ROOT, "results", "models")

app = FastAPI(title="Relationship Intelligence Layer", version="0.1.0")

_transactions: Optional[pd.DataFrame] = None
_baseline: Optional[BaselineModel] = None
_relationship: Optional[RelationshipModel] = None
_combined: Optional[CombinedModel] = None
_thresholds: dict = {}


class AnalyzeRequest(BaseModel):
    transaction_id: str


class AnalyzeResponse(BaseModel):
    transaction_id: str
    base_risk_score: float
    relationship_risk_score: float
    combined_risk_score: float
    decision: str
    explanations: list[str]


def _load_artifacts() -> None:
    global _transactions, _baseline, _relationship, _combined, _thresholds
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError("Run python run_experiment.py first to generate data and models.")
    _transactions = pd.read_csv(DATA_PATH, parse_dates=["timestamp"])
    _baseline = BaselineModel.load(os.path.join(MODEL_DIR, "baseline.joblib"))
    _relationship = RelationshipModel.load(os.path.join(MODEL_DIR, "relationship.joblib"))
    _combined = CombinedModel.load(os.path.join(MODEL_DIR, "combined.joblib"))
    train_end = int(len(_transactions) * 0.70)
    _thresholds = _percentile_thresholds(_transactions.iloc[:train_end])


@app.on_event("startup")
def startup() -> None:
    if os.path.exists(os.path.join(MODEL_DIR, "baseline.joblib")):
        _load_artifacts()


@app.post("/risk/analyze", response_model=AnalyzeResponse)
def analyze_risk(req: AnalyzeRequest) -> AnalyzeResponse:
    if _transactions is None:
        try:
            _load_artifacts()
        except FileNotFoundError as e:
            raise HTTPException(status_code=503, detail=str(e))

    row_df = _transactions[_transactions["transaction_id"] == req.transaction_id]
    if row_df.empty:
        raise HTTPException(status_code=404, detail=f"Transaction {req.transaction_id} not found")

    row = row_df.iloc[0:1]
    base_score = float(_baseline.predict_proba(row)[0])
    rel_score = float(_relationship.predict_proba(row)[0])
    comb_score = float(_combined.predict_proba(row)[0])

    exp = explain_transaction(row.iloc[0], rel_score, _thresholds)
    decision = decision_from_scores(base_score, rel_score, comb_score)

    return AnalyzeResponse(
        transaction_id=req.transaction_id,
        base_risk_score=round(base_score, 4),
        relationship_risk_score=round(rel_score, 4),
        combined_risk_score=round(comb_score, 4),
        decision=decision,
        explanations=exp["explanations"],
    )


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}
