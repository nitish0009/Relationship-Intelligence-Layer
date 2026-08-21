"""Simple Streamlit demo for Relationship Intelligence Layer."""

from __future__ import annotations

import os
import sys

import pandas as pd
import streamlit as st

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from src.baseline_model import BaselineModel
from src.explainability import _percentile_thresholds, decision_from_scores, explain_transaction
from src.graph_model import CombinedModel, RelationshipModel

st.set_page_config(page_title="RIL Demo", layout="wide")

DATA_PATH = os.path.join(PROJECT_ROOT, "data", "transactions.csv")
MODEL_DIR = os.path.join(PROJECT_ROOT, "results", "models")


@st.cache_resource
def load_models():
    tx = pd.read_csv(DATA_PATH, parse_dates=["timestamp"])
    baseline = BaselineModel.load(os.path.join(MODEL_DIR, "baseline.joblib"))
    relationship = RelationshipModel.load(os.path.join(MODEL_DIR, "relationship.joblib"))
    combined = CombinedModel.load(os.path.join(MODEL_DIR, "combined.joblib"))
    train_end = int(len(tx) * 0.70)
    thresholds = _percentile_thresholds(tx.iloc[:train_end])
    return tx, baseline, relationship, combined, thresholds


def main():
    st.title("Relationship Intelligence Layer — Demo")
    st.caption("Research prototype using synthetic data. Not a production fraud system.")

    if not os.path.exists(os.path.join(MODEL_DIR, "baseline.joblib")):
        st.warning("Run `python run_experiment.py` first.")
        return

    tx, baseline, relationship, combined, thresholds = load_models()
    tx_ids = tx["transaction_id"].tolist()
    selected = st.selectbox("Transaction ID", tx_ids, index=min(1000, len(tx_ids) - 1))

    row = tx[tx["transaction_id"] == selected].iloc[0:1]
    r = row.iloc[0]

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Transaction")
        st.write(f"**ID:** {r['transaction_id']}")
        st.write(f"**Amount:** ₹{r['amount']:,.2f}")
        st.write(f"**Customer:** {r['customer_id']}")
        st.write(f"**Device:** {r['device_id']}")
        st.write(f"**Merchant:** {r['merchant_id']}")

    base_score = float(baseline.predict_proba(row)[0])
    rel_score = float(relationship.predict_proba(row)[0])
    comb_score = float(combined.predict_proba(row)[0])
    decision = decision_from_scores(base_score, rel_score, comb_score)
    exp = explain_transaction(r, rel_score, thresholds)

    with col2:
        st.subheader("Risk Scores")
        st.metric("Base Risk Score", f"{base_score:.2f}")
        st.metric("Relationship Risk", f"{rel_score:.2f}")
        st.metric("Combined Risk", f"{comb_score:.2f}")
        st.info(f"Decision: **{decision}**")

    st.subheader("Relationship Evidence")
    for i, line in enumerate(exp["explanations"], 1):
        st.write(f"{i}. {line}")

    net_img = os.path.join(PROJECT_ROOT, "results", "relationship_network.png")
    if os.path.exists(net_img):
        st.subheader("Example Relationship Network")
        st.image(net_img, caption="Sample suspicious network from test set")


if __name__ == "__main__":
    main()
