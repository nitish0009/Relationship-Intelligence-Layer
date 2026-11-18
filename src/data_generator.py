"""Synthetic payment dataset generator.

Design goals (see README "Synthetic data design"):

* **Fraud rings rotate.** Every ring is created with *fresh* accounts, devices, IPs and
  stolen cards, is active only for a short window (3-21 days) and is then retired.  Rings
  active in the test period therefore use entities never seen during training, so a model
  cannot score well by memorising "bad" entity IDs -- it has to learn relational
  *behaviour* (e.g. many new accounts on one device within a week).
* **Graph-hidden fraud is genuinely hidden.** Its intrinsic transaction attributes
  (amount, payment method, success, time of day) are drawn from exactly the same
  distributions as legitimate traffic.
* **Legitimate sharing exists.** Households share devices/IPs/cards, many customers
  transact over public Wi-Fi and carrier-grade NAT IPs, and some customers borrow
  devices -- so "shared" does not mean "fraud".
* **New users are not a fraud proxy.** Legitimate customers also sign up during the
  observation window.
* **Fully reproducible.** All randomness comes from one seeded ``numpy`` Generator and
  nothing iterates over a Python ``set`` (whose order depends on ``PYTHONHASHSEED``).
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np
import pandas as pd

RANDOM_SEED = 42
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")

TIMELINE_START = pd.Timestamp("2024-03-01")
TIMELINE_END = pd.Timestamp("2024-09-30")

PAYMENT_METHODS = np.array(["upi", "card", "netbanking", "wallet"])
PM_PROBS = np.array([0.45, 0.35, 0.12, 0.08])

# Legitimate traffic peaks in the day/evening; overt fraud is flatter / more nocturnal.
_LEGIT_HOURS = np.array(
    [1, 0.6, 0.4, 0.3, 0.3, 0.5, 1.2, 2.5, 4, 5, 5.5, 6, 6.5, 6, 5.5, 5.5, 6, 6.5, 7, 7.5, 7, 5.5, 3.5, 2]
)
LEGIT_HOUR_P = _LEGIT_HOURS / _LEGIT_HOURS.sum()
_FRAUD_HOURS = np.array(
    [4, 4.5, 5, 5, 4.5, 3.5, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3.5, 4, 4]
)
FRAUD_HOUR_P = _FRAUD_HOURS / _FRAUD_HOURS.sum()

FRAUD_PATTERNS = [
    "device_farm",
    "card_testing",
    "account_farm",
    "ip_concentration",
    "merchant_targeting",
    "account_takeover",
]


@dataclass(frozen=True)
class AdversaryConfig:
    """Evasion strategies (C4). None / 0 disables a strategy.

    device_max_uses   -- each attacker device is retired after k transactions (device rotation)
    account_max_uses  -- each ring account is retired after k transactions (account rotation)
    ip_max_uses       -- each attacker IP is retired after k transactions (IP rotation / proxies)
    carrier_ip_share  -- share of ring transactions routed through carrier-grade NAT IPs
                         shared with legitimate users (hiding in the crowd)
    slow_factor       -- ring lifetime is multiplied by this factor at constant volume
                         (slow-and-low: fewer transactions per detection window)
    adapt_from        -- only rings starting at or after this fraction of the timeline adapt
                         (e.g. 0.85 = attackers change behaviour in the test period only)
    """

    device_max_uses: int | None = None
    account_max_uses: int | None = None
    ip_max_uses: int | None = None
    carrier_ip_share: float = 0.0
    slow_factor: float = 1.0
    adapt_from: float = 0.0

    def active(self) -> bool:
        return any([self.device_max_uses, self.account_max_uses, self.ip_max_uses,
                    self.carrier_ip_share > 0, self.slow_factor != 1.0])


@dataclass(frozen=True)
class GeneratorConfig:  # noqa: D101 -- see HARD_PRESET below for the robustness setting
    n_customers: int = 20_000
    n_transactions: int = 100_000
    n_merchants: int = 1_000
    fraud_rate: float = 0.035
    graph_hidden_share: float = 0.45  # mean share of fraud txns with normal-looking intrinsic attrs
    # Ring sophistication heterogeneity: each ring's hidden share ~ Beta(k*mu, k*(1-mu)) with
    # mu = graph_hidden_share. Small k -> many rings are almost fully camouflaged and others crude.
    # None -> every transaction is hidden independently with probability mu (homogeneous rings).
    ring_hidden_concentration: float | None = 1.5
    noisy_infra_share: float = 0.25  # share of fraud txns that use clean/legit-looking infra
    household_share: float = 0.12
    new_user_share: float = 0.08
    n_carrier_ips: int = 300  # carrier-grade NAT: many legit users behind one IP
    n_public_wifi_ips: int = 60
    # ring lifetime in days, sampled uniformly from [lo, hi] (C2)
    ring_duration_days: tuple[int, int] = (3, 21)
    ring_size_scale: float = 1.0  # multiplies the number of transactions per ring
    # probability that each ring device / IP / account is drawn from a persistent attacker
    # pool instead of being fresh (C3 "reuse dial"; 0 = fully rotating rings, 1 = fixed pool)
    infra_reuse: float = 0.0
    n_pool_devices: int = 40
    n_pool_ips: int = 35
    n_pool_accounts: int = 400
    adversary: AdversaryConfig = AdversaryConfig()


class _IdFactory:
    """Sequential ID allocation (deterministic, no collisions across legit/fraud)."""

    def __init__(self, prefix: str, width: int):
        self.prefix, self.width, self.n = prefix, width, 0

    def new(self) -> str:
        out = f"{self.prefix}{self.n:0{self.width}d}"
        self.n += 1
        return out

    def many(self, k: int) -> list[str]:
        return [self.new() for _ in range(int(k))]


def _timestamps(
    rng: np.random.Generator, n: int, lo: pd.Timestamp, hi: pd.Timestamp, hour_p: np.ndarray
) -> np.ndarray:
    """Sample n timestamps in [lo, hi): uniform day, hour from a diurnal profile."""
    lo_day = lo.normalize()
    n_days = max(1, (hi.normalize() - lo_day).days)
    day = rng.integers(0, n_days, size=n)
    hour = rng.choice(24, size=n, p=hour_p)
    sec = rng.uniform(0, 3600, size=n)
    ts = lo_day + pd.to_timedelta(day, unit="D") + pd.to_timedelta(hour, unit="h") + pd.to_timedelta(sec, unit="s")
    ts = ts.where(ts >= lo, lo)
    ts = ts.where(ts < TIMELINE_END, TIMELINE_END - pd.Timedelta(seconds=1))
    return np.asarray(ts)


def _legit_intrinsics(rng: np.random.Generator, n: int) -> dict[str, np.ndarray]:
    amount = np.clip(rng.lognormal(mean=7.3, sigma=0.65, size=n), 10, 25_000)
    return {
        "amount": np.round(amount, 2),
        "success": (rng.random(n) > 0.04).astype(int),
        "payment_method": rng.choice(PAYMENT_METHODS, size=n, p=PM_PROBS),
    }


def generate_all(
    output_dir: str | None = DATA_DIR, seed: int = RANDOM_SEED, config: GeneratorConfig | None = None
) -> dict[str, pd.DataFrame]:
    """Generate all entity tables + transactions; save CSVs if output_dir is given."""
    cfg = config or GeneratorConfig()
    rng = np.random.default_rng(seed)

    cust_ids = _IdFactory("C", 6)
    dev_ids = _IdFactory("D", 6)
    card_ids = _IdFactory("K", 6)
    ip_ids = _IdFactory("I", 6)

    # ------------------------------------------------------------------ merchants
    merchants = pd.DataFrame(
        {
            "merchant_id": [f"M{i:04d}" for i in range(cfg.n_merchants)],
            "category": rng.choice(
                ["retail", "food", "travel", "digital", "utilities", "education"],
                size=cfg.n_merchants,
                p=[0.25, 0.20, 0.10, 0.20, 0.15, 0.10],
            ),
            "risk_tier": rng.choice(["low", "medium", "high"], size=cfg.n_merchants, p=[0.7, 0.25, 0.05]),
        }
    )
    merchant_arr = merchants["merchant_id"].to_numpy()
    pop = 1.0 / np.arange(1, cfg.n_merchants + 1) ** 0.8  # Zipf-like popularity
    pop = pop[rng.permutation(cfg.n_merchants)]
    merchant_p = pop / pop.sum()

    # ------------------------------------------------------------------ shared legit IP pools
    residential_ips = ip_ids.many(max(1, int(cfg.n_customers / 2.5)))
    carrier_ips = ip_ids.many(cfg.n_carrier_ips)
    public_wifi_ips = ip_ids.many(cfg.n_public_wifi_ips)
    ip_kind = {**{i: "residential" for i in residential_ips}, **{i: "carrier" for i in carrier_ips},
               **{i: "public_wifi" for i in public_wifi_ips}}

    # ------------------------------------------------------------------ legit customers
    n = cfg.n_customers
    segment = rng.choice(
        ["normal", "household", "new_user"],
        size=n,
        p=[1 - cfg.household_share - cfg.new_user_share, cfg.household_share, cfg.new_user_share],
    )
    timeline_s = (TIMELINE_END - TIMELINE_START).total_seconds()
    created_at = np.where(
        segment == "new_user",
        TIMELINE_START.value + (rng.uniform(0, 0.9, size=n) * timeline_s * 1e9).astype(np.int64),
        (TIMELINE_START - pd.Timedelta(days=60)).value - (rng.uniform(0, 700, size=n) * 86400e9).astype(np.int64),
    ).astype("datetime64[ns]")

    legit_customers = cust_ids.many(n)
    aligned: list[dict | None] = [None] * n
    for j in range(n):
        if aligned[j] is not None:
            continue
        if segment[j] == "household":
            size = int(rng.integers(2, 5))
            members = [j]
            k = j + 1
            while len(members) < size and k < n:
                if segment[k] == "household" and aligned[k] is None:
                    members.append(k)
                k += 1
            if len(members) >= 2:
                shared_dev, home_ip = dev_ids.new(), ip_ids.new()
                ip_kind[home_ip] = "residential"
                shared_cards = card_ids.many(min(2, len(members)))
                for m_i, m in enumerate(members):
                    own_dev = [dev_ids.new()] if rng.random() < 0.5 else []
                    aligned[m] = {
                        "devices": [shared_dev] + own_dev,
                        "device_p": [0.7, 0.3] if own_dev else [1.0],
                        "cards": [shared_cards[m_i % len(shared_cards)]]
                        + ([card_ids.new()] if rng.random() < 0.3 else []),
                        "home_ip": home_ip,
                        "household": True,
                    }
                continue
            segment[j] = "normal"  # lone household member at the tail
        n_dev = 1 + int(rng.random() < 0.35)
        aligned[j] = {
            "devices": dev_ids.many(n_dev),
            "device_p": [1.0] if n_dev == 1 else [0.8, 0.2],
            "cards": card_ids.many(1 + int(rng.random() < 0.4)),
            "home_ip": residential_ips[int(rng.integers(len(residential_ips)))],
            "household": False,
        }
    for p in aligned:
        k = int(rng.integers(3, 9))
        p["merchants"] = rng.choice(merchant_arr, size=k, replace=False, p=merchant_p)

    all_personal_devices = np.array([d for p in aligned for d in p["devices"]])

    # ------------------------------------------------------------------ fraud rings
    n_fraud_target = int(round(cfg.n_transactions * cfg.fraud_rate))
    n_days = (TIMELINE_END - TIMELINE_START).days
    fraud_rows: list[dict] = []
    fraud_customers: list[tuple[str, np.datetime64]] = []
    established = np.flatnonzero(segment != "new_user")
    ring_id = 0
    ring_meta: list[dict] = []
    if cfg.infra_reuse > 0:
        pool_devices = dev_ids.many(cfg.n_pool_devices)
        pool_ips = ip_ids.many(cfg.n_pool_ips)
        pool_accounts = cust_ids.many(cfg.n_pool_accounts)
        for cid in pool_accounts:
            fraud_customers.append((cid, np.datetime64(TIMELINE_START - pd.Timedelta(days=30), "ns")))
        for ip in pool_ips:
            ip_kind[ip] = "fraud"
    adv = cfg.adversary
    while len(fraud_rows) < n_fraud_target:
        pattern = FRAUD_PATTERNS[int(rng.integers(len(FRAUD_PATTERNS)))]
        duration = int(rng.integers(cfg.ring_duration_days[0], cfg.ring_duration_days[1] + 1))
        # Rings may straddle the timeline edges (clipped, volume scaled by the visible share),
        # which keeps ring density uniform over time -> no prevalence drift between splits.
        start_day = int(rng.integers(-duration + 1, n_days))
        lo_day, hi_day = max(start_day, 0), min(start_day + duration, n_days)
        start = TIMELINE_START + pd.Timedelta(days=lo_day)
        end = TIMELINE_START + pd.Timedelta(days=hi_day)
        n_events = int(rng.integers(20, 51) if pattern == "card_testing" else rng.integers(10, 41))
        n_events = max(1, int(round(n_events * cfg.ring_size_scale * (hi_day - lo_day) / duration)))
        n_events = min(n_events, n_fraud_target - len(fraud_rows))
        adapted = adv.active() and (adv.adapt_from <= 0 or start_day >= adv.adapt_from * n_days)
        if adapted and adv.slow_factor != 1.0:  # slow-and-low: stretch the active window
            end = min(TIMELINE_END, start + (end - start) * adv.slow_factor)

        def new_accounts(k: int) -> list[str]:
            ids = cust_ids.many(k)
            for cid in ids:
                lead = pd.Timedelta(days=float(rng.uniform(0, 10)))
                fraud_customers.append((cid, np.datetime64(start - lead, "ns")))
            return ids

        if pattern == "device_farm":
            accounts, devices, ips = new_accounts(rng.integers(15, 41)), dev_ids.many(rng.integers(2, 5)), ip_ids.many(rng.integers(3, 9))
        elif pattern == "card_testing":
            accounts, devices, ips = new_accounts(rng.integers(1, 4)), dev_ids.many(rng.integers(1, 3)), ip_ids.many(rng.integers(1, 3))
        elif pattern == "account_farm":
            accounts, devices, ips = new_accounts(rng.integers(20, 51)), dev_ids.many(rng.integers(3, 7)), ip_ids.many(rng.integers(1, 3))
        elif pattern == "ip_concentration":
            accounts, devices, ips = new_accounts(rng.integers(10, 31)), dev_ids.many(rng.integers(8, 21)), ip_ids.many(rng.integers(1, 4))
        elif pattern == "merchant_targeting":
            accounts, devices, ips = new_accounts(rng.integers(10, 31)), dev_ids.many(rng.integers(3, 7)), ip_ids.many(rng.integers(2, 5))
        else:  # account_takeover: compromised *existing* legit accounts on attacker infra
            victims = rng.choice(established, size=int(rng.integers(5, 16)), replace=False)
            accounts = [legit_customers[v] for v in victims]
            devices, ips = dev_ids.many(rng.integers(1, 4)), ip_ids.many(rng.integers(1, 4))
        if cfg.infra_reuse > 0:  # C3: draw infrastructure from the persistent attacker pool
            devices = [pool_devices[int(rng.integers(len(pool_devices)))] if rng.random() < cfg.infra_reuse else d
                       for d in devices]
            ips = [pool_ips[int(rng.integers(len(pool_ips)))] if rng.random() < cfg.infra_reuse else i for i in ips]
            if pattern != "account_takeover":
                accounts = [pool_accounts[int(rng.integers(len(pool_accounts)))] if rng.random() < cfg.infra_reuse
                            else a for a in accounts]
        for ip in ips:
            ip_kind[ip] = "fraud"
        devices, ips, accounts = list(devices), list(ips), list(accounts)
        uses = {"device": {}, "ip": {}, "account": {}}

        def rotate(kind: str, pool: list, max_uses: int | None, factory) -> str:
            """Pick from the ring's pool; with rotation, retire an entity after max_uses."""
            if not (adapted and max_uses):
                return rng.choice(pool)
            live = [x for x in pool if uses[kind].get(x, 0) < max_uses]
            x = live[int(rng.integers(len(live)))] if live else factory()
            if not live:
                pool.append(x)
            uses[kind][x] = uses[kind].get(x, 0) + 1
            return x

        def new_rotated_account() -> str:
            if pattern == "account_takeover":
                return legit_customers[int(rng.choice(established))]
            return new_accounts(1)[0]

        # stolen cards: fresh, occasionally a real legit customer's card
        n_cards = max(2, int(len(accounts) * 1.5))
        ring_cards = card_ids.many(n_cards)
        if pattern == "account_takeover":
            ring_cards = [rng.choice(aligned[v]["cards"]) for v in victims] + ring_cards[:3]
        else:
            for c_i in range(len(ring_cards)):
                if rng.random() < 0.1:
                    ring_cards[c_i] = rng.choice(aligned[int(rng.choice(established))]["cards"])

        if pattern == "merchant_targeting":
            ring_merchants = rng.choice(merchant_arr, size=int(rng.integers(1, 3)), replace=False)
        else:
            ring_merchants = rng.choice(merchant_arr, size=int(rng.integers(2, 5)), replace=False)

        if cfg.ring_hidden_concentration is None:
            ring_hidden_p = cfg.graph_hidden_share
        else:
            kappa, mu = cfg.ring_hidden_concentration, cfg.graph_hidden_share
            ring_hidden_p = float(rng.beta(kappa * mu, kappa * (1 - mu)))
        hidden = rng.random(n_events) < ring_hidden_p
        ts_hidden = _timestamps(rng, n_events, start, end, LEGIT_HOUR_P)
        ts_overt = _timestamps(rng, n_events, start, end, FRAUD_HOUR_P)
        legit_like = _legit_intrinsics(rng, n_events)
        for e in range(n_events):
            noisy = rng.random() < cfg.noisy_infra_share
            device = rotate("device", devices, adv.device_max_uses, dev_ids.new) if not noisy or rng.random() < 0.5 else (
                dev_ids.new() if rng.random() < 0.5 else rng.choice(all_personal_devices)
            )
            ip = rotate("ip", ips, adv.ip_max_uses, ip_ids.new) if not noisy or rng.random() < 0.5 else carrier_ips[int(rng.integers(len(carrier_ips)))]
            if adapted and adv.carrier_ip_share > 0 and rng.random() < adv.carrier_ip_share:
                ip = carrier_ips[int(rng.integers(len(carrier_ips)))]
            if pattern == "card_testing" and rng.random() < 0.7:
                card = card_ids.new()
            else:
                card = rng.choice(ring_cards)
            if pattern == "merchant_targeting" or rng.random() < 0.5:
                merchant = rng.choice(ring_merchants)
            else:
                merchant = rng.choice(merchant_arr, p=merchant_p)

            if hidden[e]:
                ts = ts_hidden[e]
                amount, success, pm = legit_like["amount"][e], legit_like["success"][e], legit_like["payment_method"][e]
            else:
                ts = ts_overt[e]
                if pattern == "card_testing":
                    amount = float(np.clip(rng.lognormal(4.0, 0.8), 1, 2_000))
                    success = int(rng.random() > 0.65)
                    pm = "card"
                else:
                    amount = float(np.clip(rng.lognormal(7.9, 0.9), 50, 50_000))
                    success = int(rng.random() > 0.45)
                    pm = rng.choice(PAYMENT_METHODS, p=[0.35, 0.45, 0.05, 0.15])
                amount = round(amount, 2)
            fraud_rows.append(
                {
                    "timestamp": ts,
                    "customer_id": rotate("account", accounts, adv.account_max_uses, new_rotated_account),
                    "device_id": device,
                    "card_id": card,
                    "ip_id": ip,
                    "merchant_id": merchant,
                    "amount": amount,
                    "payment_method": pm,
                    "success": int(success),
                    "fraud_label": 1,
                    "fraud_pattern": pattern,
                    "ring_id": ring_id,
                    "graph_hidden_fraud": int(hidden[e]),
                    "legitimate_shared_env": 0,
                    "segment": "fraud",
                }
            )
        ring_rows = fraud_rows[len(fraud_rows) - n_events:]
        ring_meta.append(
            {
                "ring_id": ring_id,
                "fraud_pattern": pattern,
                "planned_start": start,
                "planned_end": end,
                "lifetime_days": (end - start).total_seconds() / 86400,
                "n_events": n_events,
                "n_devices": len({r["device_id"] for r in ring_rows}),
                "n_ips": len({r["ip_id"] for r in ring_rows}),
                "n_accounts": len({r["customer_id"] for r in ring_rows}),
                "n_cards": len({r["card_id"] for r in ring_rows}),
                "total_amount": float(sum(r["amount"] for r in ring_rows)),
                "adapted": int(adapted),
                "hidden_share": float(np.mean(hidden)),
            }
        )
        ring_id += 1

    # ------------------------------------------------------------------ legit transactions
    n_legit = cfg.n_transactions - len(fraud_rows)
    active_lo = np.maximum(created_at, np.datetime64(TIMELINE_START, "ns"))
    active_frac = (np.datetime64(TIMELINE_END, "ns") - active_lo) / np.timedelta64(int(timeline_s), "s")
    activity = rng.gamma(1.5, 1.0, size=n) * np.clip(active_frac, 0.02, 1.0)
    who = rng.choice(n, size=n_legit, p=activity / activity.sum())
    intr = _legit_intrinsics(rng, n_legit)
    # vectorised timestamps: uniform day within each customer's active window, diurnal hour
    lo_ns = active_lo[who].astype(np.int64)
    lo_day_ns = (lo_ns // 86_400_000_000_000) * 86_400_000_000_000
    n_days_active = np.maximum(1, (TIMELINE_END.value - lo_day_ns) // 86_400_000_000_000)
    day = (rng.random(n_legit) * n_days_active).astype(np.int64)
    hour = rng.choice(24, size=n_legit, p=LEGIT_HOUR_P)
    sec = rng.uniform(0, 3600, size=n_legit)
    ts_ns = lo_day_ns + day * 86_400_000_000_000 + hour * 3_600_000_000_000 + (sec * 1e9).astype(np.int64)
    ts_ns = np.clip(np.maximum(ts_ns, lo_ns), None, TIMELINE_END.value - 1_000_000_000)
    legit_ts = ts_ns.astype("datetime64[ns]")
    legit_rows: list[dict] = []
    for e, j in enumerate(who):
        p = aligned[j]
        ts = legit_ts[e]
        r = rng.random()
        if r < 0.02:  # borrowed device
            device = rng.choice(all_personal_devices)
        else:
            device = rng.choice(p["devices"], p=p["device_p"])
        r = rng.random()
        if r < 0.75:
            ip = p["home_ip"]
        elif r < 0.93:
            ip = carrier_ips[int(rng.integers(len(carrier_ips)))]
        else:
            ip = public_wifi_ips[int(rng.integers(len(public_wifi_ips)))]
        merchant = rng.choice(p["merchants"]) if rng.random() < 0.9 else rng.choice(merchant_arr, p=merchant_p)
        legit_rows.append(
            {
                "timestamp": ts,
                "customer_id": legit_customers[j],
                "device_id": device,
                "card_id": rng.choice(p["cards"]),
                "ip_id": ip,
                "merchant_id": merchant,
                "amount": intr["amount"][e],
                "payment_method": intr["payment_method"][e],
                "success": int(intr["success"][e]),
                "fraud_label": 0,
                "fraud_pattern": "",
                "ring_id": -1,
                "graph_hidden_fraud": 0,
                "legitimate_shared_env": int(p["household"] or ip_kind.get(ip) == "public_wifi"),
                "segment": "household" if p["household"] else {"public_wifi": "public_wifi", "carrier": "carrier_nat"}.get(
                    ip_kind.get(ip), "residential"),
            }
        )

    tx = pd.DataFrame(legit_rows + fraud_rows)
    tx["timestamp"] = pd.to_datetime(tx["timestamp"])
    tx = tx.sort_values(["timestamp", "customer_id"], kind="stable").reset_index(drop=True)
    tx.insert(0, "transaction_id", [f"TX{i:06d}" for i in range(len(tx))])

    # ------------------------------------------------------------------ entity tables
    customers = pd.DataFrame(
        {
            "customer_id": legit_customers + [c for c, _ in fraud_customers],
            "created_at": np.concatenate([created_at, np.array([t for _, t in fraud_customers], dtype="datetime64[ns]")]),
            "segment": list(segment) + ["new_user"] * len(fraud_customers),
        }
    )
    dev_list = [f"D{i:06d}" for i in range(dev_ids.n)]
    devices = pd.DataFrame(
        {"device_id": dev_list, "device_type": rng.choice(["mobile", "desktop", "tablet"], size=len(dev_list), p=[0.65, 0.30, 0.05])}
    )
    card_list = [f"K{i:06d}" for i in range(card_ids.n)]
    cards = pd.DataFrame(
        {"card_id": card_list, "issuer": rng.choice(["visa", "mastercard", "rupay", "amex"], size=len(card_list), p=[0.45, 0.35, 0.15, 0.05])}
    )
    ip_list = [f"I{i:06d}" for i in range(ip_ids.n)]
    ips = pd.DataFrame(
        {
            "ip_id": ip_list,
            "region": rng.choice(["north", "south", "east", "west", "central"], size=len(ip_list), p=[0.22, 0.25, 0.18, 0.20, 0.15]),
        }
    )
    ips["kind"] = ips["ip_id"].map(ip_kind).fillna("single_use")

    rings = pd.DataFrame(ring_meta)
    out = {"customers": customers, "devices": devices, "cards": cards, "ips": ips, "merchants": merchants,
           "rings": rings, "transactions": tx}
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        for name, df in out.items():
            df.to_csv(os.path.join(output_dir, f"{name}.csv"), index=False)

    print(
        f"Generated {len(tx):,} transactions ({tx['fraud_label'].sum():,} fraud in {ring_id} rings, "
        f"{tx['graph_hidden_fraud'].sum():,} graph-hidden, {tx['legitimate_shared_env'].sum():,} legitimate shared)"
    )
    return out


if __name__ == "__main__":
    generate_all()


# Harder robustness setting (fix for ceiling effects): more camouflage, noisier attacker
# infrastructure, more heterogeneous rings, smaller rings, more legitimate sharing.
HARD_PRESET = dict(graph_hidden_share=0.65, noisy_infra_share=0.5, ring_hidden_concentration=1.0,
                   ring_size_scale=0.6, household_share=0.2, n_carrier_ips=150, n_public_wifi_ips=30)
