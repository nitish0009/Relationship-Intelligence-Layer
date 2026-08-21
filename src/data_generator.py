"""Synthetic payment dataset generator with realistic behavioral structure."""

from __future__ import annotations

import os
from datetime import timedelta
from typing import Tuple

import numpy as np
import pandas as pd

RANDOM_SEED = 42
N_CUSTOMERS = 20_000
N_DEVICES = 5_000
N_CARDS = 10_000
N_IPS = 8_000
N_MERCHANTS = 1_000
N_TRANSACTIONS = 100_000

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")


def _rng(seed: int = RANDOM_SEED) -> np.random.Generator:
    return np.random.default_rng(seed)


def generate_entity_tables(rng: np.random.Generator) -> dict[str, pd.DataFrame]:
    """Create static entity tables."""
    customers = pd.DataFrame(
        {
            "customer_id": [f"C{i:05d}" for i in range(N_CUSTOMERS)],
            "created_at": pd.date_range("2024-01-01", periods=N_CUSTOMERS, freq="30min"),
            "segment": rng.choice(
                ["normal", "household", "new_user"], size=N_CUSTOMERS, p=[0.82, 0.12, 0.06]
            ),
        }
    )

    devices = pd.DataFrame(
        {
            "device_id": [f"D{i:05d}" for i in range(N_DEVICES)],
            "device_type": rng.choice(
                ["mobile", "desktop", "tablet"], size=N_DEVICES, p=[0.65, 0.30, 0.05]
            ),
        }
    )

    cards = pd.DataFrame(
        {
            "card_id": [f"K{i:05d}" for i in range(N_CARDS)],
            "issuer": rng.choice(
                ["visa", "mastercard", "rupay", "amex"], size=N_CARDS, p=[0.45, 0.35, 0.15, 0.05]
            ),
        }
    )

    ips = pd.DataFrame(
        {
            "ip_id": [f"I{i:05d}" for i in range(N_IPS)],
            "region": rng.choice(
                ["north", "south", "east", "west", "central"],
                size=N_IPS,
                p=[0.22, 0.25, 0.18, 0.20, 0.15],
            ),
        }
    )

    merchants = pd.DataFrame(
        {
            "merchant_id": [f"M{i:04d}" for i in range(N_MERCHANTS)],
            "category": rng.choice(
                ["retail", "food", "travel", "digital", "utilities", "education"],
                size=N_MERCHANTS,
                p=[0.25, 0.20, 0.10, 0.20, 0.15, 0.10],
            ),
            "risk_tier": rng.choice(["low", "medium", "high"], size=N_MERCHANTS, p=[0.7, 0.25, 0.05]),
        }
    )

    return {
        "customers": customers,
        "devices": devices,
        "cards": cards,
        "ips": ips,
        "merchants": merchants,
    }


def _assign_households(
    customers: pd.DataFrame, devices: pd.DataFrame, cards: pd.DataFrame, ips: pd.DataFrame, rng: np.random.Generator
) -> Tuple[dict, dict, dict]:
    """Map household customers to shared devices/IPs/cards."""
    household_customers = customers[customers["segment"] == "household"]["customer_id"].tolist()
    n_households = max(1, len(household_customers) // 3)
    household_groups: list[list[str]] = []
    idx = 0
    for _ in range(n_households):
        size = rng.integers(2, 5)
        group = household_customers[idx : idx + size]
        if len(group) >= 2:
            household_groups.append(list(group))
        idx += size
        if idx >= len(household_customers):
            break

    shared_device_map: dict[str, str] = {}
    shared_ip_map: dict[str, str] = {}
    shared_card_map: dict[str, list[str]] = {}

    device_pool = devices["device_id"].tolist()
    ip_pool = ips["ip_id"].tolist()
    card_pool = cards["card_id"].tolist()

    for group in household_groups:
        dev = rng.choice(device_pool)
        ip = rng.choice(ip_pool)
        household_cards = rng.choice(card_pool, size=min(3, len(group)), replace=False).tolist()
        for i, cid in enumerate(group):
            shared_device_map[cid] = dev
            shared_ip_map[cid] = ip
            shared_card_map[cid] = household_cards[i % len(household_cards)]

    return shared_device_map, shared_ip_map, shared_card_map


def _pick_normal_entities(
    customer_id: str,
    rng: np.random.Generator,
    customer_devices: dict,
    customer_cards: dict,
    customer_ips: dict,
    customer_merchants: dict,
    devices: pd.DataFrame,
    cards: pd.DataFrame,
    ips: pd.DataFrame,
    merchants: pd.DataFrame,
    shared_device_map: dict,
    shared_ip_map: dict,
    shared_card_map: dict,
) -> Tuple[str, str, str, str]:
    """Pick entities for a normal transaction with stable preferences."""
    if customer_id in shared_device_map:
        device_id = shared_device_map[customer_id]
        ip_id = shared_ip_map[customer_id]
        card_id = shared_card_map[customer_id]
    else:
        if customer_id not in customer_devices:
            customer_devices[customer_id] = rng.choice(devices["device_id"].values, size=3).tolist()
            customer_cards[customer_id] = rng.choice(cards["card_id"].values, size=2).tolist()
            customer_ips[customer_id] = rng.choice(ips["ip_id"].values, size=2).tolist()

        if rng.random() < 0.92:
            device_id = rng.choice(customer_devices[customer_id][:2])
            ip_id = rng.choice(customer_ips[customer_id][:1])
        else:
            device_id = rng.choice(devices["device_id"].values)
            ip_id = rng.choice(ips["ip_id"].values)

        card_id = rng.choice(customer_cards[customer_id])

    if customer_id not in customer_merchants:
        customer_merchants[customer_id] = rng.choice(
            merchants["merchant_id"].values, size=rng.integers(3, 8)
        ).tolist()
    merchant_id = rng.choice(customer_merchants[customer_id])

    return device_id, card_id, ip_id, merchant_id


def generate_transactions(
    entities: dict[str, pd.DataFrame], rng: np.random.Generator
) -> pd.DataFrame:
    """Generate transactions with normal, shared-legitimate, and fraud patterns."""
    customers = entities["customers"]
    devices = entities["devices"]
    cards = entities["cards"]
    ips = entities["ips"]
    merchants = entities["merchants"]

    shared_device_map, shared_ip_map, shared_card_map = _assign_households(
        customers, devices, cards, ips, rng
    )

    start = pd.Timestamp("2024-03-01")
    end = pd.Timestamp("2024-09-30")
    timestamps = pd.date_range(start, end, periods=N_TRANSACTIONS)

    customer_devices: dict[str, list] = {}
    customer_cards: dict[str, list] = {}
    customer_ips: dict[str, list] = {}
    customer_merchants: dict[str, list] = {}

    normal_customers = customers[customers["segment"] == "normal"]["customer_id"].tolist()
    new_users = customers[customers["segment"] == "new_user"]["customer_id"].tolist()
    household_ids = set(shared_device_map.keys())

    # Fraud infrastructure pools (also bleed into normal IDs so graph signals aren't perfect)
    fraud_devices = rng.choice(devices["device_id"].values, size=40, replace=False).tolist()
    fraud_ips = rng.choice(ips["ip_id"].values, size=35, replace=False).tolist()
    fraud_merchants = merchants[merchants["risk_tier"] == "high"]["merchant_id"].tolist()
    if len(fraud_merchants) < 5:
        fraud_merchants = merchants["merchant_id"].sample(10, random_state=RANDOM_SEED).tolist()
    # Contaminate some normal households onto fraud devices (noise / hard negatives)
    for cid in rng.choice(list(shared_device_map.keys()) or normal_customers[:50], size=30, replace=False):
        if rng.random() < 0.4:
            shared_device_map[cid] = rng.choice(fraud_devices)

    fraud_customer_pool = list(new_users[:800]) + rng.choice(
        normal_customers, size=400, replace=False
    ).tolist()

    target_fraud_rate = 0.035
    n_fraud_target = int(N_TRANSACTIONS * target_fraud_rate)
    # Pre-assign fraud slots evenly across timeline to avoid train/test skew
    fraud_slot_indices = set(
        rng.choice(N_TRANSACTIONS, size=n_fraud_target, replace=False).tolist()
    )

    rows = []
    fraud_count = 0
    graph_hidden_count = 0
    legitimate_shared_count = 0

    payment_methods = ["upi", "card", "netbanking", "wallet"]
    pm_probs = [0.45, 0.35, 0.12, 0.08]

    for i, ts in enumerate(timestamps):
        is_fraud = False
        is_graph_hidden = False
        is_legitimate_shared = False
        fraud_pattern = ""

        # Fraud at pre-assigned evenly distributed indices
        if i in fraud_slot_indices:
            is_fraud = True
            fraud_count += 1
            pattern_type = rng.integers(0, 6)

            # With some probability, reuse normal-looking infrastructure (harder cases)
            use_noisy_infra = rng.random() < 0.25

            if pattern_type == 0:  # device farm
                fraud_pattern = "device_farm"
                device_id = rng.choice(fraud_devices)
                customer_id = rng.choice(fraud_customer_pool)
                ip_id = rng.choice(ips["ip_id"].values) if use_noisy_infra else rng.choice(fraud_ips)
                card_id = rng.choice(cards["card_id"].values)
            elif pattern_type == 1:  # card testing
                fraud_pattern = "card_testing"
                customer_id = rng.choice(fraud_customer_pool)
                device_id = rng.choice(devices["device_id"].values) if use_noisy_infra else rng.choice(fraud_devices)
                ip_id = rng.choice(fraud_ips)
                card_id = rng.choice(cards["card_id"].values)
            elif pattern_type == 2:  # account farm
                fraud_pattern = "account_farm"
                customer_id = rng.choice(new_users)
                device_id = rng.choice(fraud_devices)
                ip_id = rng.choice(fraud_ips)
                card_id = rng.choice(cards["card_id"].values)
            elif pattern_type == 3:  # IP concentration
                fraud_pattern = "ip_concentration"
                ip_id = rng.choice(fraud_ips)
                customer_id = rng.choice(fraud_customer_pool)
                device_id = rng.choice(devices["device_id"].values)
                card_id = rng.choice(cards["card_id"].values)
            elif pattern_type == 4:  # coordinated merchant
                fraud_pattern = "merchant_targeting"
                merchant_id = rng.choice(fraud_merchants)
                customer_id = rng.choice(fraud_customer_pool)
                device_id = rng.choice(fraud_devices) if not use_noisy_infra else rng.choice(devices["device_id"].values)
                ip_id = rng.choice(fraud_ips) if not use_noisy_infra else rng.choice(ips["ip_id"].values)
                card_id = rng.choice(cards["card_id"].values)
            else:  # rapid relationships
                fraud_pattern = "rapid_relationships"
                customer_id = rng.choice(new_users)
                device_id = rng.choice(fraud_devices)
                ip_id = rng.choice(fraud_ips)
                card_id = rng.choice(cards["card_id"].values)

            # Graph-hidden: normal-looking transaction-level features
            if rng.random() < 0.45:
                is_graph_hidden = True
                amount = float(rng.choice([499, 999, 1499, 1999, 2499, 2999]))
                success = True
                payment_method = rng.choice(["upi", "card"], p=[0.5, 0.5])
            else:
                amount = float(rng.lognormal(mean=7.0, sigma=0.8))
                amount = min(max(amount, 50), 50000)
                success = rng.random() > 0.55
                payment_method = rng.choice(payment_methods, p=pm_probs)

            if pattern_type != 4:
                merchant_id = rng.choice(fraud_merchants) if rng.random() < 0.6 else rng.choice(
                    merchants["merchant_id"].values
                )
        else:
            # Normal or legitimate shared
            if rng.random() < 0.15 and household_ids:
                customer_id = rng.choice(list(household_ids))
                is_legitimate_shared = True
                legitimate_shared_count += 1
            else:
                customer_id = rng.choice(normal_customers)

            device_id, card_id, ip_id, merchant_id = _pick_normal_entities(
                customer_id,
                rng,
                customer_devices,
                customer_cards,
                customer_ips,
                customer_merchants,
                devices,
                cards,
                ips,
                merchants,
                shared_device_map,
                shared_ip_map,
                shared_card_map,
            )
            amount = float(rng.lognormal(mean=7.3, sigma=0.65))
            amount = min(max(amount, 10), 25000)
            success = rng.random() > 0.04
            payment_method = rng.choice(payment_methods, p=pm_probs)
            fraud_pattern = ""

        if is_graph_hidden:
            graph_hidden_count += 1

        rows.append(
            {
                "transaction_id": f"TX{i:06d}",
                "timestamp": ts,
                "customer_id": customer_id,
                "device_id": device_id,
                "card_id": card_id,
                "ip_id": ip_id,
                "merchant_id": merchant_id,
                "amount": round(amount, 2),
                "payment_method": payment_method,
                "success": int(success),
                "fraud_label": int(is_fraud),
                "fraud_pattern": fraud_pattern,
                "graph_hidden_fraud": int(is_fraud and is_graph_hidden),
                "legitimate_shared_env": int(is_legitimate_shared and not is_fraud),
            }
        )

    tx = pd.DataFrame(rows)
    return tx


def generate_all(output_dir: str = DATA_DIR, seed: int = RANDOM_SEED) -> dict[str, pd.DataFrame]:
    """Generate and save all datasets."""
    os.makedirs(output_dir, exist_ok=True)
    rng = _rng(seed)
    entities = generate_entity_tables(rng)
    transactions = generate_transactions(entities, rng)

    for name, df in entities.items():
        path = os.path.join(output_dir, f"{name}.csv")
        df.to_csv(path, index=False)

    tx_path = os.path.join(output_dir, "transactions.csv")
    transactions.to_csv(tx_path, index=False)

    print(f"Generated {len(transactions):,} transactions "
          f"({transactions['fraud_label'].sum():,} fraud, "
          f"{transactions['graph_hidden_fraud'].sum():,} graph-hidden, "
          f"{transactions['legitimate_shared_env'].sum():,} legitimate shared)")

    return {**entities, "transactions": transactions}


if __name__ == "__main__":
    generate_all()
