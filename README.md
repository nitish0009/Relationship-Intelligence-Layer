# Relationship Intelligence Layer (RIL)

**Fraud rings are short-lived: evaluating relational fraud detection on the ring's clock.**

Research code testing when **relationship context** (shared customers, devices, cards, IPs, merchants) helps detect payment fraud. It runs on a controllable synthetic generator and is replicated on the public IEEE-CIS dataset.

> **Disclaimer:** This is a research prototype using synthetic data and the public IEEE-CIS dataset.

## Contributions

| | Question | Experiment |
|---|---|---|
| **C1** | How much money does a fraud ring take **before its first alert**, and how quickly does each model recognise a new ring? | `main` |
| **C2** | When do features built from neighbours' **confirmed fraud labels** help, given **label delay** vs **ring lifetime**? | `c2` |
| **C3** | How much do leaky "published-style" protocols inflate results as **attacker infrastructure reuse** varies from 0 to 1? | `c3` |
| **C4** | What does it **cost an attacker to evade** each feature family (device, account or IP rotation, carrier-NAT hiding, slow-and-low)? | `c4` |
| G5 | Do confidence intervals **cover** the truth when fraud comes in rings? | `g5` |
| G4 / G6 | False positives on **legitimately shared infrastructure**; performance on **brand-new entities** | `main` |
| Real data | Replication on **IEEE-CIS** (590K transactions) | `ieee` |

## Quick start

```bash
pip install -r requirements.txt
python -m pytest tests/ -q                 # 28 tests, ~20 s

python run_experiment.py main              # 12 models x 5 seeds, C1/G4/G6       (~30 min)
python run_experiment.py c2                # label delay x lifetime x reuse grid  (~18 min)
python run_experiment.py c3                # reuse dose-response                  (~20 min)
python run_experiment.py c4                # cost of evasion                      (~25 min)
python run_experiment.py g5                # CI coverage over 30 datasets         (~8 min)
python run_experiment.py examples          # explanations + neighbourhood plot
python run_experiment.py ieee              # IEEE-CIS replication (~1.5 h)
```

For the IEEE-CIS run, download `train_transaction.csv` and `train_identity.csv` from the [competition page](https://www.kaggle.com/competitions/ieee-fraud-detection/data) into `data/IEEE_CIS/`. Accept the competition rules first; the folder is git-ignored because redistribution is not permitted.

Every run is deterministic: one seeded RNG, no `set` iteration, single-threaded torch, and pinned package versions.

## Synthetic data (`src/data_generator.py`)

About 100K transactions over Mar–Sep 2024, with 3.5% fraud in about 140 **rotating fraud rings**. Each ring is active for a few days with fresh accounts, devices, IPs and stolen cards, then retires. There are six patterns: device farm, card testing, account farm, IP concentration, merchant targeting and account takeover.

- **Graph-hidden fraud:** amount, payment method, success and time of day follow the *legitimate* distributions. The intrinsic-feature sanity AUC is **0.498**.
- **Ring sophistication:** each ring's hidden share is drawn from a Beta distribution with mean 0.45, so some rings are almost fully camouflaged.
- **Legitimate sharing:** households, carrier-grade NAT IPs, public Wi-Fi and borrowed devices. Legitimate users also sign up during the window.
- **Settings for C2–C4:**
  - `ring_duration_days`: ring lifetime
  - `infra_reuse`: probability of drawing from a persistent attacker pool
  - `AdversaryConfig`: device, account and IP rotation, `carrier_ip_share`, `slow_factor`, `adapt_from`

## Methodology

- **No temporal leakage.** All features use strictly earlier transactions; label features use only labels that have **matured** (t + D ≤ now). Every feature family is checked against brute-force reference implementations and truncation-invariance tests.
- **Splits:** chronological 70 / 15 / 15. The test set is never used for any choice.
- **Thresholds:** F1-optimal on validation, plus **alert budgets** at 0.1%, 0.5% and 1% validation FPR (capacity-constrained operation). C1 uses the budgets.
- **Tuning:** XGBoost uses a 15-trial random search on validation PR-AUC for every seed. GNNs use a small grid tuned on the first seed.
- **Uncertainty:** 5 seeds, with each seed a new dataset and new models. CIs come from a **ring-clustered bootstrap** (see G5 for why).
- **Models:**
  - Tabular feature families (transaction → counts → temporal → full graph → delayed labels), all on XGBoost.
  - **Neighbour-aggregation XGBoost** (GADBench-style).
  - Temporal **GraphSAGE**, **GAT** and **R-GCN** over a strictly-past K=20 neighbourhood.
  - A **CARE-style** GNN with label-aware neighbour filtering.
  - A **TGN-style** memory model trained through time.
  - Entity-graph GraphSAGE, in inductive and transductive variants.

## Results (synthetic; 5 seeds unless stated)

### Model comparison (`results/main/`)

| Model | PR-AUC | Recall@1%FPR | Graph-hidden recall |
|---|---|---|---|
| Transaction Baseline | 0.821 ± 0.027 | 0.745 | 54% |
| + Relationship counts | 0.918 ± 0.014 | 0.861 | 73% |
| + Temporal relationships | 0.963 ± 0.010 | 0.929 | 79% |
| Combined (full hand-crafted graph) | 0.979 ± 0.006 | 0.968 | 83% |
| Combined + delayed labels (D = 30 d) | 0.980 ± 0.006 | 0.967 | 83% |
| CARE-style GNN | 0.980 ± 0.005 | 0.969 | 86% |
| TGN-style Memory | 0.963 ± 0.009 | 0.935 | 82% |
| Temporal R-GCN | 0.985 ± 0.004 | 0.977 | 88% |
| Temporal GraphSAGE | 0.986 ± 0.003 | 0.979 | 89% |
| Temporal GAT | 0.987 ± 0.003 | 0.981 | 89% |
| **Neighbour-Aggregation XGBoost** | **0.987 ± 0.004** | **0.985** | 87% |
| GraphSAGE (entity embeddings, inductive) | 0.805 ± 0.042 | 0.813 | 81% |

As in GADBench, gradient-boosted trees on neighbour aggregates match the best GNNs.

### C1: ring-level early detection

Share of ring money lost before the first alert (110 rings that start in the test window, pooled over seeds):

| Model | 0.1% budget | 0.5% budget | 1% budget | Recall on ring's 1st txn (0.5%) |
|---|---|---|---|---|
| Transaction Baseline | 11.1% | 6.2% | 4.7% | 26% |
| + Relationship counts | 9.4% | 4.7% | 3.3% | 39% |
| Combined | 4.6% | 2.4% | 1.6% | 55% |
| Neighbour-Aggregation XGBoost | **3.7%** | 2.0% | 0.7% | 59% |
| Temporal GAT | 3.8% | **1.6%** | **0.5%** | **64%** |
| TGN-style Memory | 7.3% | 3.3% | 1.9% | 44% |

At a tight alert budget the transaction baseline loses about **3×** more ring money than relational models. Against camouflaged rings (more than 67% hidden) it loses 7.4%, against 2.6% for Temporal GAT. See `ring_by_sophistication.csv` and `warmup.png`.

### C2: label delay × ring lifetime (`results/c2/`)

PR-AUC gain (pp) from adding delayed neighbour-label features, with fresh rings (reuse 0), 3 seeds:

| Ring lifetime | D = 0 | 1 d | 7 d | 30 d | 90 d |
|---|---|---|---|---|---|
| 1–3 d | +0.39 | +0.03 | +0.03 | +0.01 | −0.01 |
| 7–14 d | +0.84 | +0.37 | +0.07 | −0.05 | −0.07 |
| 30–60 d | +2.90 | +2.55 | +2.13 | +0.38 | +0.04 |
| 60–120 d | +3.99 | +3.71 | +3.26 | +1.00 | +0.06 |

Label features help **only when the label delay is shorter than the ring's lifetime**. A model-free reachability check explains why: once the delay exceeds the lifetime, the share of a ring's transactions after its first matured label is zero. At realistic chargeback delays (30–90 d), short-lived rings are invisible to label propagation. With 50% infrastructure reuse, labels keep a small benefit even at long delays, because labels from *previous* rings carry over through reused devices and IPs.

### C3: infrastructure-reuse dose–response (`results/c3/`)

PR-AUC inflation of leaky protocols over their leakage-safe counterparts, all scored on the same chronological test window (pp):

| Leaky protocol | p = 0 | 0.25 | 0.5 | 0.75 | 1.0 |
|---|---|---|---|---|---|
| Transductive vs inductive GraphSAGE | **+19.6** | +14.7 | +9.4 | +6.3 | +1.4 |
| Label features with no delay vs 30-day delay | +1.0 | +0.5 | +0.3 | 0.0 | 0.0 |
| Random vs chronological split | +0.4 | +0.4 | −1.1 | −0.1 | 0.0 |

Leakage inflates results most in the **realistic, fresh-ring regime**. Fixed-pool benchmarks (p = 1) hide it: every model, even the baseline, scores 0.98–1.00, so such benchmarks are saturated. Random splits must be compared on the same rows. Otherwise early, cold-start test rows make random splits *look* worse, which is a composition artefact.

### C4: cost of evasion (`results/c4/`)

Recall on adapted ring transactions at a 0.5% budget. A static defender is trained before attackers adapt; a retrained defender is trained on adapted behaviour.

| Attacker strategy (resource cost) | Baseline | Counts | +Temporal | Combined | +Labels | NbrAgg XGB |
|---|---|---|---|---|---|---|
| None (30 dev / 46 acct / 23 IP per 100 txns) | 0.63 | 0.80 | 0.89 | 0.92 | 0.91 | 0.94 |
| Fresh device per transaction (3.3× devices) | 0.49 | 0.62 | 0.76 | 0.84 | 0.85 | 0.88 |
| All traffic via carrier NAT (4× IPs) | 0.63 | **0.55** | 0.87 | 0.89 | 0.90 | 0.93 |
| Slow-and-low ×8 | 0.49 | 0.73 | 0.84 | 0.87 | 0.87 | 0.87 |
| All combined, static defender | **0.36** | 0.40 | 0.64 | 0.68 | 0.69 | **0.78** |
| All combined, retrained defender | 0.56 | 0.58 | 0.77 | 0.78 | **0.83** | **0.87** |

- No single cheap strategy defeats temporal relational features, and rotating accounts doesn't help attackers at all.
- Simple counts collapse under carrier-NAT hiding, falling below the baseline.
- Combining every strategy costs attackers about 2× devices, 3.8× IPs and 4× ring lifetime.
- Retraining recovers most of the loss. Delayed labels then add +5 pp, because slowed rings live long enough for their labels to mature, which ties C4 to C2.

### G5: confidence-interval coverage (`results/g5/`)

A model trained once is evaluated on the test splits of 30 independent datasets; nominal coverage is 95%.

| Model | i.i.d. row bootstrap | Ring-clustered bootstrap |
|---|---|---|
| Transaction Baseline | **57%** | 93% |
| Combined | **70%** | 87% |

Row-level CIs, the norm in fraud papers, are badly over-confident when fraud comes in rings.

### G4 / G6: shared infrastructure and new entities (`results/main/`)

- **G4.** Learned neighbourhood models over-flag legitimate users on shared networks:
  - Temporal GNNs: FPR on public Wi-Fi is **1.4–1.6%**, against 0.23–0.28% on residential networks.
  - Neighbour-aggregation XGBoost: **1.15%** on carrier NAT, against 0.19% on residential.

  Hand-crafted features stay much flatter (Combined: 0.21% public Wi-Fi, 0.42% carrier NAT, 0.28% residential). This extends the IP-sharing problem known from ad-click fraud (Google, KDD 2011) to learned relational models.
- **G6.** On transactions from first-seen devices, recall is 42% for the baseline against 77–85% for relational models.

### IEEE-CIS replication (`results/ieee/`)

590,540 transactions (3.5% fraud), chronological 70/15/15 split, one run. CIs are ring-clustered over proxy rings.

| Model | PR-AUC [95% CI] | Recall@1%FPR |
|---|---|---|
| Transaction Baseline (our features) | 0.137 [0.117, 0.159] | 0.093 |
| Combined (our hand-crafted graph features) | 0.190 [0.158, 0.226] | 0.136 |
| Neighbour-Aggregation XGBoost | 0.219 [0.186, 0.254] | 0.182 |
| Temporal GAT | 0.228 [0.198, 0.259] | 0.195 |
| Temporal GraphSAGE | 0.243 [0.207, 0.284] | 0.197 |
| TGN-style Memory | 0.135 [0.110, 0.163] | 0.084 |
| Vesta's ~430 engineered features (XGBoost) | 0.523 [0.482, 0.563] | 0.438 |
| Vesta + our graph features | 0.511 [0.468, 0.551] | 0.429 |
| Vesta + Neighbour-Aggregation XGBoost | 0.514 [0.472, 0.551] | 0.426 |
| Vesta + Temporal GAT | 0.453 [0.410, 0.494] | 0.378 |
| Vesta + graph + delayed labels (D = 30 d) | **0.622** [0.584, 0.659] | **0.541** |

- **Relational context helps on top of basic transaction features** (0.137 → 0.243 with temporal GNNs), in the same order as on synthetic data: hand-crafted < neighbour aggregation < temporal GNN.
- **It adds nothing on top of Vesta's engineered features** (0.511–0.514 vs 0.523, overlapping CIs). Vesta's C/D/V features already encode relational counts, so the incremental value of relationship features depends on what the feature store already contains.
- **Label delay (C2/C3 on real data).** PR-AUC of Vesta + graph + labels by delay D: 0.857 (0 d), 0.760 (1 d), 0.689 (7 d), 0.622 (30 d), 0.556 (90 d), 0.518 (no labels). **Ignoring label delay inflates PR-AUC by 23.5 pp** (0 d vs 30 d). Labels still help at 90 d (+3.8 pp), because IEEE-CIS fraud entities persist for months: it behaves like the high-reuse regime. Recall at the 0.5% budget is 0.93 on test fraud whose uid/card/device was already seen in training fraud (16% of test fraud), against 0.42 on unseen entities. The dataset's per-client labelling convention amplifies this.
- **Transductive vs inductive GraphSAGE: 0.244 vs 0.097 (+14.8 pp)**, the C3 inflation reproduced on real data.
- **C1 on 225 proxy rings** (≥ 3 test transactions, 0.5% budget): money lost before the first alert is 81.5% for the baseline, 65% for Combined and neighbour aggregation, and 61% for the Vesta models. Most IEEE fraud is single-client bursts (median proxy ring is 1 transaction), so ring-level gains are smaller than on synthetic data.
- **Ring-clustered CIs are about 2× wider than i.i.d. CIs** (Vesta: [0.482, 0.563] vs [0.505, 0.542]), consistent with G5.

The schema mapping and its limitations are documented in `experiments/ieee_cis.py`:
- **customer:** the standard `uid` (card1 + addr1 + account-start day)
- **card:** card1–6
- **device:** DeviceInfo + id_30/31/33
- **network:** a fingerprint proxy built from id_17/19/20, since IEEE-CIS has no IP field
- **rings:** proxy rings, i.e. connected fraud components linked only through keys shared by ≤ 10 customers. Coarse fingerprints such as `Windows|chrome 63.0`, shared by 2,893 customers, would otherwise merge 83% of all fraud into one component.

## Project structure

```
├── run_experiment.py        # CLI: main | c2 | c3 | c4 | g5 | examples | ieee | all
├── experiments/             # one module per experiment (+ common.py registry)
├── src/
│   ├── data_generator.py    # synthetic rings, legit sharing, reuse + adversary settings
│   ├── preprocessing.py     # transaction features, chronological split
│   ├── graph_features.py    # temporal relationship features (O(1) incremental tracker)
│   ├── label_features.py    # neighbour fraud-label features under label delay (C2)
│   ├── ring_metrics.py      # ring-level early-detection metrics (C1)
│   ├── temporal_gnn.py      # temporal index, GraphSAGE/GAT/R-GCN/CARE-style, TGN-style, NbrAgg-XGB
│   ├── graph_model.py       # tabular models, entity GraphSAGE (inductive/transductive)
│   ├── pipeline.py          # dataset building, tuning, fit/score, alert budgets
│   ├── evaluation.py        # metrics, bootstrap CIs, plots
│   └── explainability.py    # human-readable relationship risk signals
├── tests/                   # 28 leakage / reproducibility / methodology tests
├── data/                    # generated synthetic CSVs (seed 42); IEEE_CIS/ git-ignored
└── results/                 # main/ c2/ c3/ c4/ g5/ ieee/
```

## Limitations

- The synthetic generator encodes assumptions (ring sizes, lifetimes, sharing rates), so absolute numbers reflect those assumptions. Relative and qualitative findings are the claims. IEEE-CIS has no ground-truth rings, IPs or success flags, so its ring and network results use documented proxies.
- The GNNs are compact re-implementations, not the original codebases. CARE-style filtering uses a fixed keep-ratio instead of reinforcement learning, and the TGN-style model is a simplified memory variant.
- Attackers in C4 follow fixed strategies; they don't optimise against the deployed model.
- GNN hyperparameters are tuned on one seed only, for cost reasons.
