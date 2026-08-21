"""Smoke tests for RIL pipeline."""

import os
import sys

import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)


def test_chronological_split_no_overlap():
    from src.preprocessing import chronological_split

    df = pd.DataFrame(
        {"timestamp": pd.date_range("2024-01-01", periods=100, freq="h"), "fraud_label": 0}
    )
    train, val, test = chronological_split(df)
    assert train["timestamp"].max() < val["timestamp"].min()
    assert val["timestamp"].max() < test["timestamp"].min()


def test_graph_features_no_future_leakage():
    from src.graph_features import build_graph_features

    df = pd.DataFrame(
        {
            "transaction_id": [f"T{i}" for i in range(5)],
            "timestamp": pd.date_range("2024-01-01", periods=5, freq="D"),
            "customer_id": ["C1", "C1", "C2", "C2", "C1"],
            "device_id": ["D1", "D1", "D1", "D2", "D1"],
            "card_id": ["K1", "K1", "K2", "K2", "K1"],
            "ip_id": ["I1", "I1", "I1", "I2", "I1"],
            "merchant_id": ["M1", "M2", "M1", "M1", "M2"],
            "amount": [100, 200, 150, 300, 120],
            "payment_method": ["upi"] * 5,
            "success": [1, 1, 0, 1, 1],
            "fraud_label": [0, 0, 1, 0, 0],
        }
    )
    out = build_graph_features(df)
    # first tx should have zero prior device customers
    assert out.iloc[0]["device_unique_customers"] == 0
    # third tx: device D1 should know C1 only (not C2 yet at feature time... actually C2 uses D1 at idx 2)
    assert out.iloc[2]["device_unique_customers"] >= 1


def test_data_generator_schema():
    from src.data_generator import generate_all
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        data = generate_all(tmp, seed=42)
        tx = data["transactions"]
        required = [
            "transaction_id",
            "timestamp",
            "customer_id",
            "device_id",
            "card_id",
            "ip_id",
            "merchant_id",
            "amount",
            "payment_method",
            "success",
            "fraud_label",
        ]
        for col in required:
            assert col in tx.columns
