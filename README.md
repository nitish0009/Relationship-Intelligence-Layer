# Relationship Intelligence Layer (RIL)

**Fraud rings are short-lived: evaluating relational fraud detection on the ring's clock.**

Research code that asks *when* relationship context (shared customers, devices, cards, IPs and counterparties) helps detect payment fraud and money laundering. It uses a controllable synthetic generator and is replicated on two public datasets: **IEEE-CIS** (590K card transactions) and **IBM AMLworld** (5.1M transfers with ground-truth laundering patterns).

> **Disclaimer:** This is a research prototype using synthetic data and public datasets.

## Contributions

| | Question | Experiment |
|---|---|---|
| **C1** | How much money does a ring take **before its first alert**, and how fast does each model recognise a new ring? | `main`, `aml`, `ieee` |
| **C2** | When do features from neighbours' **confirmed fraud labels** help, given **label delay** vs **ring lifetime**? | `c2`, `aml`, `ieee` |
| **C3** | How much do leaky "published-style" protocols inflate results as **attacker infrastructure reuse** goes from 0 to 1? | `c3`, `calibration`, `ieee` |
| **C4** | What does it **cost an attacker to evade** each feature family, against fixed and **adaptive** attackers? | `c4`, `adaptive` |
| G5 | Do confidence intervals **cover** the truth when fraud comes in rings? | `g5` |
| G4 / G6 | False positives on **legitimately shared infrastructure**; recall on **brand-new entities** | `main` |
| **C5** | **Takedown-value weighting (TVW):** can training on the ring's clock reduce ring losses? | `tvw`, `tvw_gnn` |
| Robustness | Are findings artefacts of the generator? (sensitivity sweep + hard preset) | `sensitivity` |

## Quick start

```bash
pip install -r requirements.txt
python -m pytest tests/ -q                       # 36 tests

python run_experiment.py main --preset hard       # headline setting: 12 models x 5 seeds, tuned per seed
python run_experiment.py main                     # default preset (near ceiling)
python run_experiment.py tvw                      # C5 on synthetic + AMLworld + IEEE-CIS
python run_experiment.py tvw_gnn                  # C5 on the temporal GNNs
python scripts/make_paper_tables.py && python scripts/make_paper_figures.py   # paper/tables, paper/figures
python run_experiment.py c2 --seeds 42 43 44 45 46
python run_experiment.py c3 --seeds 42 43 44 45 46
python run_experiment.py c4 --seeds 42 43 44 45 46
python run_experiment.py g5                       # set n_datasets=100 in code for the paper run
python run_experiment.py sensitivity              # generator sweep + hard preset
python run_experiment.py adaptive                 # model-aware greedy attacker
python run_experiment.py tgn_reference            # reference TGN (PyTorch Geometric)
python run_experiment.py aml                      # IBM AMLworld (data/AMLworld/)
python run_experiment.py ieee                     # IEEE-CIS (data/IEEE_CIS/)
python run_experiment.py calibration              # sim-to-real reuse calibration

# Original CARE-GNN / PC-GNN code (fetched at pinned commits, not redistributed):
scripts/fetch_third_party.sh && RIL_RUN_THIRD_PARTY=1 python run_experiment.py main
RIL_RUN_THIRD_PARTY=1 python run_experiment.py originals_tuning   # validation sweep around the authors' defaults
```

**Data:**
- IEEE-CIS: [competition page](https://www.kaggle.com/competitions/ieee-fraud-detection/data); place `train_transaction.csv` and `train_identity.csv` in `data/IEEE_CIS/`.
- AMLworld: [dataset page](https://www.kaggle.com/datasets/ealtman2019/ibm-transactions-for-anti-money-laundering-aml); place `HI-Small_Trans.csv` and `HI-Small_Patterns.txt` in `data/AMLworld/`.

Both folders and `third_party/` are git-ignored. Runs are deterministic: one seeded RNG, no `set` iteration, single-threaded torch, pinned versions.

## Synthetic data (`src/data_generator.py`)

About 100K transactions with 3.5% fraud in about 140 **rotating fraud rings**. Each ring lives a few days with fresh accounts, devices, IPs and cards, then retires. There are six patterns: device farm, card testing, account farm, IP concentration, merchant targeting and account takeover.

- **Graph-hidden fraud:** amount, method, outcome and hour follow the legitimate distributions (intrinsic-feature sanity AUC **0.498**).
- **Ring sophistication:** each ring's hidden share is drawn from a Beta distribution.
- **Legitimate sharing:** households, carrier NAT, public Wi-Fi, borrowed devices, and legitimate new users.
- **Settings:** `ring_duration_days`, `ring_size_scale`, `infra_reuse`, `AdversaryConfig` (device, account and IP rotation; carrier-NAT share; slow factor), plus `HARD_PRESET`.

## Methodology

- **No temporal leakage.** Features use strictly earlier transactions, and label features use only labels that have **matured** (t + D ≤ now). Every feature family, the temporal neighbour index and both TGN models are tested: brute-force references for the features, truncation invariance, and no-future-leakage checks for the TGN streams.
- **Evaluation:**
  - Chronological 70/15/15 split; the test set is never used for any choice.
  - Thresholds: F1-optimal on validation, plus **alert budgets** at 0.1%, 0.5% and 1% validation FPR (C1 uses these).
  - **Every model is tuned on every seed's validation split** (XGBoost: 15-trial random search; GNNs: grid search).
  - 5 seeds, with each seed a new dataset and new models. CIs are **ring-clustered** (see G5), with paired t-tests across seeds.
- **Models:**
  - XGBoost feature families: transaction → counts → temporal → full graph → delayed labels.
  - **Neighbour-aggregation XGBoost** (GADBench-style).
  - Temporal **GraphSAGE / GAT / R-GCN** and a **CARE-style** GNN over a strictly-past K-recent neighbourhood.
  - A **TGN-style** memory model, plus a **reference TGN** built from PyTorch Geometric's `TGNMemory` / `TransformerConv`.
  - The **original CARE-GNN and PC-GNN code** at pinned commits with the authors' default hyperparameters, run in an isolated subprocess on a strictly-past graph.
  - Entity-graph GraphSAGE, inductive and transductive.

## Results: synthetic, hard preset (headline; `results/main_hard/`, 5 seeds)

More camouflage (65% graph-hidden fraud), noisier infrastructure, more legitimate sharing.

| Model | PR-AUC | Ring money lost @0.1% / @0.5% budget | 1st-txn recall |
|---|---|---|---|
| Transaction features | 0.724 ± 0.058 | 23.8% / 13.7% | 30% |
| Hand-crafted graph features | 0.947 ± 0.028 | 12.5% / 7.6% | 46% |
| Neighbour-aggregation XGBoost | 0.967 ± 0.015 | 9.7% / 4.9% | 51% |
| Temporal GraphSAGE | 0.964 ± 0.017 | 9.3% / 4.2% | 54% |
| Temporal GAT | 0.964 ± 0.014 | 8.5% / 4.5% | 52% |
| Temporal R-GCN | 0.965 ± 0.021 | 9.7% / 4.3% | 52% |
| Temporal CARE-style GNN | 0.958 ± 0.018 | 10.6% / 5.3% | 48% |
| CARE-GNN / PC-GNN, original code (validation-tuned) | 0.554 / 0.678 | | |

- GNNs vs hand-crafted features: +1.1 to +1.8 PR-AUC points on 5/5 seeds (significant for R-GCN p = 0.007 and GraphSAGE p = 0.028; GAT p = 0.06, CARE-style p = 0.09).
- **On the ring clock every temporal GNN is significantly better**: GraphSAGE / GAT / R-GCN cut money lost before the first alert by **41–45%** vs hand-crafted features (GraphSAGE −3.4 pp, ring-clustered CI [−5.2, −1.9]) and 67–69% vs transaction features; CARE-style 30%.
- Price of context: Temporal GAT alerts 1.17% of public Wi-Fi traffic vs 0.17% residential (hand-crafted: 0.08% / 0.24%).

## C5: takedown-value weighting (`src/clock_weights.py`, `results/tvw/`, `results/tvw_gnn/`)

Catching a ring's k-th transaction saves the money it would still move, R_k. The money a takedown saves is bounded by Σ p_k R_k, so TVW trains with fraud weights ∝ R_k (mixed by β, mean-normalised). Rings are **pseudo-rings** from union-find over specific shared entities (no ring labels; ARI 0.93 vs true rings). Weights are **ESS-tempered** (largest γ with effective sample size ≥ 50%), which matters for heavy-tailed amounts (AMLworld raw ESS = 4 of 2,856).

| Hard preset, 10 seeds, 0.5% budget | Money lost: ordinary → TVW | Δ pp [95% ring-clustered CI] |
|---|---|---|
| Hand-crafted graph features | 7.9% → 6.2% | −1.75 [−2.87, −0.82] |
| Neighbour-aggregation XGBoost | 5.2% → 3.9% | −1.20 [−2.12, −0.47] |
| Temporal GraphSAGE | 4.4% → 3.8% | −0.58 [−1.14, −0.04] |
| Temporal GAT | 4.4% → 4.0% | −0.40 [−1.07, +0.28] (1% budget: −0.42 [−0.82, −0.02]) |
| Temporal CARE-style GNN | 5.2% → 4.2% | −1.02 [−1.81, −0.25] |

First-transaction recall rises 3–8 points for every model; PR-AUC changes ≤ 0.5 points. **Real data is neutral:** on AMLworld relational models already lose only 0.01–0.14% of pattern money (no room), on IEEE-CIS proxy rings are mostly single-client bursts; tempered TVW changes neither, at ≤ 0.4 PR-AUC cost on IEEE. Untempered weights (`results/tvw_raw/`) cut IEEE ring losses by 4.6 pp but cost 2.4 PR-AUC points, and collapse AMLworld (0.38 → 0.13).

> **Bug fix (2026-11-19).** The temporal GNNs' neighbour-feature standardiser was fitted on the *first* 20,000 training rows instead of a random sample. On AMLworld these covered six minutes, so test inputs reached |z| ≈ 18,500 and training was unstable. Fixed in `src/temporal_gnn.py` (regression test added); all GNN results were rerun. AMLworld GNNs went from 0.12–0.16 to 0.47; synthetic and IEEE results changed only slightly. Default-preset GNN rows in `results/main/` predate the fix.

## Results: synthetic, default preset (5 seeds)

### Model comparison (`results/main/`)

| Model | PR-AUC | Recall@1%FPR | Graph-hidden recall |
|---|---|---|---|
| **CARE-GNN (original code)** | 0.654 ± 0.032 | 0.546 | 40% |
| **PC-GNN (original code)** | 0.789 ± 0.025 | 0.718 | 56% |
| GraphSAGE (entity, inductive) | 0.805 ± 0.042 | 0.813 | 81% |
| Transaction Baseline | 0.822 ± 0.027 | 0.745 | 54% |
| + Relationship counts | 0.918 ± 0.014 | 0.861 | 73% |
| TGN (PyG reference) | 0.919 ± 0.035 | 0.878 | 76% |
| + Temporal relationships | 0.963 ± 0.010 | 0.929 | 79% |
| TGN-style Memory | 0.964 ± 0.009 | 0.941 | 82% |
| Combined (full hand-crafted graph) | 0.979 ± 0.006 | 0.968 | 83% |
| CARE-style GNN (temporal) | 0.980 ± 0.005 | 0.970 | 87% |
| Temporal R-GCN | 0.985 ± 0.005 | 0.977 | 88% |
| Temporal GraphSAGE | 0.986 ± 0.003 | 0.980 | 88% |
| Temporal GAT | 0.986 ± 0.003 | 0.980 | 89% |
| **Neighbour-Aggregation XGBoost** | **0.987 ± 0.004** | **0.985** | 87% |

- Relational vs transaction features: **+15.8 pp**, 5/5 seeds, p = 1e-4.
- Temporal GAT vs hand-crafted: +0.7 pp, p = 0.006.
- Neighbour-aggregation XGBoost ≈ Temporal GAT (p = 0.05). As in GADBench, trees on neighbour aggregates match the best GNNs.
- **The original CARE-GNN and PC-GNN score below the transaction baseline.** A diagnostic shows the authors' row normalisation isn't the cause (CARE 0.660 vs 0.650; PC-GNN 0.793 vs 0.795, with vs without). Static, single-layer, undersampled training is a poor fit for transaction streams, whereas the same filtering idea inside a temporal model reaches 0.980.

### C1: ring-level early detection

Share of ring money lost before the first alert (110 rings starting in the test window):

| Model | 0.1% budget [95% CI] | 0.5% budget | Recall on ring's 1st txn |
|---|---|---|---|
| CARE-GNN (original) | 28.1% | 17.6% | 14% |
| Transaction Baseline | 11.1% [8.8, 13.8] | 6.2% | 26% |
| + Relationship counts | 9.4% | 4.7% | 39% |
| Combined | 4.6% [3.4, 5.9] | 2.4% | 55% |
| Neighbour-Aggregation XGBoost | 3.7% [2.8, 4.9] | 2.0% | 59% |
| Temporal GAT | **3.4%** | **1.4%** | **65%** |

The baseline minus Temporal GAT difference has a paired ring-bootstrap 95% CI of [5.1, 9.8] pp at the 0.1% budget.

### C2: label delay × ring lifetime (`results/c2/`)

PR-AUC gain (pp, mean ± 95% CI, 5 seeds) from delayed neighbour-label features:

| Ring lifetime | Reuse 0, D = 0 | 1 d | 7 d | 30 d | 90 d | **Reuse 0.5**, 90 d |
|---|---|---|---|---|---|---|
| 1–3 d | +0.42 ± 0.21 | 0.00 ± 0.12 | 0.00 ± 0.07 | −0.02 ± 0.07 | −0.03 ± 0.06 | +0.09 ± 0.09 |
| 7–14 d | +0.98 ± 0.65 | +0.44 ± 0.45 | +0.12 ± 0.13 | −0.03 ± 0.12 | −0.05 ± 0.09 | +0.07 ± 0.09 |
| 30–60 d | +2.65 ± 0.47 | +2.33 ± 0.50 | +1.89 ± 0.55 | +0.28 ± 0.23 | +0.02 ± 0.10 | +0.23 ± 0.17 |
| 60–120 d | +4.09 ± 0.97 | +3.86 ± 0.94 | +3.37 ± 0.92 | +0.98 ± 0.53 | +0.06 ± 0.08 | **+0.27 ± 0.12** |

Labels help **only when the delay is shorter than the ring lifetime**; a model-free reachability bound shows why. With infrastructure reuse, labels from *earlier* rings carry over even at 90 days.

### C3: reuse dose–response (`results/c3/`)

Inflation of leaky protocols over leakage-safe ones (pp, same test window, 5 seeds):

| Leaky protocol | p = 0 | 0.25 | 0.5 | 0.75 | 1.0 |
|---|---|---|---|---|---|
| Transductive vs inductive GraphSAGE | **+18.1 ± 5.6** | +14.8 ± 2.6 | +10.9 ± 4.8 | +5.5 ± 3.8 | +1.2 ± 0.7 |
| Labels with no delay vs 30-day delay | +1.1 ± 0.5 | +0.8 ± 0.6 | +0.3 ± 0.2 | 0.0 | 0.0 |
| Random vs chronological split | +0.3 ± 1.1 | +1.0 ± 1.1 | −0.7 ± 1.4 | 0.0 | 0.0 |

- Leakage inflates results most in the realistic **fresh-ring** regime. Fixed-pool benchmarks (p = 1) hide it and saturate every model at 0.98–1.00.
- Random splits show **no detectable** effect once compared on the same rows.

### C4: cost of evasion (`results/c4/`, `results/c4_adaptive/`)

**Fixed strategies** (recall on adapted ring transactions at a 0.5% budget, 5 seeds):

| Strategy | Baseline | Counts | +Temporal | Combined | +Labels | NbrAgg |
|---|---|---|---|---|---|---|
| None | 0.63 ± 0.06 | 0.81 | 0.90 | 0.93 | 0.93 | 0.95 |
| Fresh device per transaction | 0.41 ± 0.11 | 0.57 | 0.76 | 0.85 | 0.85 | 0.90 |
| All traffic via carrier NAT | 0.61 | **0.52** | 0.84 | 0.85 | 0.86 | 0.89 |
| Slow-and-low ×8 | 0.53 | 0.76 | 0.87 | 0.90 | 0.90 | 0.91 |
| All combined, static defender | **0.34** | 0.36 | 0.63 | 0.68 | 0.68 | **0.77** |
| All combined, retrained defender | 0.56 | 0.59 | 0.77 | 0.78 | **0.82** | **0.85** |

Under retraining, delayed labels add **+4.5 pp** (p = 0.001): slowed rings live long enough for their labels to mature.

**Adaptive attacker.** The attacker has black-box access to a frozen model and greedily optimises the five strategies by detection reduction per unit of resource cost; results are on held-out data.

| Defender | No evasion | Attacker cost | Frozen recall | Retrained recall |
|---|---|---|---|---|
| Transaction Baseline | 0.71 | 2.5× | **0.42** | 0.49 |
| + Temporal relationships | 0.92 | 6.0× | 0.74 | 0.86 |
| Combined | 0.95 | 6.3× | **0.83** | 0.92 |
| Combined + delayed labels | 0.95 | 6.8× | 0.84 | 0.93 |
| Neighbour-Aggregation XGBoost | 0.96 | 7.1× | 0.73 | **0.93** |

- Evading relational models costs 6–7× the attacker's normal resources.
- The attacker's best weapon is **slow-and-low**, which outwaits fixed 7-day windows.
- Neighbour aggregation is more evadable when frozen than hand-crafted features (0.73 vs 0.83), but recovers best when retrained.

### G5: confidence-interval coverage (100 independent test sets; nominal 95%)

| Model | i.i.d. row bootstrap | Ring-clustered bootstrap |
|---|---|---|
| Transaction Baseline | **53%** [43, 63] | 93% [86, 97] |
| Combined | **77%** [68, 85] | 89% [81, 94] |

### G4 / G6: shared infrastructure and new entities

- **G4 (FPR at a 0.5% budget).** Learned neighbourhood models over-flag legitimate shared networks:
  - Temporal GNNs: **1.3–1.7%** on public Wi-Fi vs 0.27–0.30% residential.
  - Neighbour-aggregation XGBoost: **1.15%** on carrier NAT vs 0.19% residential.

  Hand-crafted features stay flat (Combined: 0.21% / 0.42% / 0.28%). This extends the IP-sharing problem known from ad-click fraud (Google, KDD 2011) to learned relational models.
- **G6.** Recall on first-seen devices is 42% for the baseline vs 77–86% for relational models.

### Robustness to generator assumptions (`results/sensitivity/`)

Seven factors were swept one at a time over 17 settings (3 seeds each): ring size, sophistication, hidden share, noisy infrastructure, household sharing, carrier-NAT density and fraud rate.

| Finding | Sign holds | Range |
|---|---|---|
| Relational features beat the baseline | **100%** of runs | +11 to +28 pp |
| Neighbour aggregation ≥ hand-crafted | 98% | +0.4 to +1.4 pp |
| C1: baseline loses more ring money | **100%** | 1.5× to 9.5× |
| Transductive inflation | **100%** | +9 to +45 pp (largest at 1% fraud rate) |
| Missing label delay inflates | **100%** | +0.5 to +3.2 pp |

**Hard preset** (more camouflage, noisier infrastructure, more legitimate sharing; 5 seeds):

| Model | PR-AUC | Ring money lost (0.5% budget) |
|---|---|---|
| Baseline | 0.719 ± 0.060 | 13.5% |
| Combined | 0.947 ± 0.028 | 7.7% |
| Neighbour-Aggregation XGBoost | 0.967 ± 0.015 | 5.1% |
| Temporal GAT | 0.962 ± 0.021 | 4.3% |

Transductive vs inductive GraphSAGE: 0.966 vs 0.741 (+22.6 pp).

## Results: real data

### IBM AMLworld (`results/amlworld/`)

HI-Small, first 10 days: 5.08M transfers, 0.089% laundering, **363 ground-truth pattern rings** (median lifetime 2.0 days). Days 11–18 are excluded as a known generator artefact (about 60% laundering prevalence).

| Model (3 seeds) | PR-AUC | 95% CI (ring-clustered) |
|---|---|---|
| Transaction Baseline | 0.130 ± 0.009 | [0.093, 0.146] |
| + Relationship counts | 0.280 ± 0.030 | [0.267, 0.360] |
| + Temporal relationships | 0.342 ± 0.026 | [0.314, 0.415] |
| Combined | 0.377 ± 0.015 | [0.336, 0.436] |
| Combined + delayed labels (D = 1 d) | **0.480 ± 0.010** | [0.417, 0.523] |
| Neighbour-Aggregation XGBoost (tuned) | **0.539 ± 0.005** | |
| Temporal GraphSAGE (tuned, 30% legit sample) | 0.470 ± 0.007 | |
| Temporal GAT (tuned, 30% legit sample) | 0.465 ± 0.004 | |
| Entity GraphSAGE inductive / transductive (leaky) | 0.004 / 0.338 | |

- **Relational context triples PR-AUC on real laundering.**
- **C2 on real rings:**

  | Label delay | 0 | 0.5 d | 1 d | 2 d | 4 d | none |
  |---|---|---|---|---|---|---|
  | PR-AUC | 0.510 | 0.490 | 0.480 | 0.474 | 0.414 | 0.360 |
  | Model-free reachability | 0.54 | 0.47 | 0.39 | 0.27 | 0.05 | 0 |

  The gain fades as the delay passes typical pattern lifetimes. What remains at 4 days comes from account reuse across patterns (35% of test laundering is on accounts already seen in training laundering).
- **C1 on 28 pattern rings:** the baseline lets 44% of pattern money through before its first alert at a 0.1% budget, against 0.2–0.5% for relational models.
- Temporal GNNs (after the standardiser fix) beat hand-crafted graph features by 9 points and lose 0.4% of pattern money before the first alert.
- Transductive vs inductive entity GraphSAGE: +33 points (partly a weak, embedding-only inductive baseline; IEEE-CIS +16.7 is the cleaner estimate).

### IEEE-CIS (`results/ieee/`)

590,540 card transactions (3.5% fraud), 3 seeds, every model tuned on validation (Vesta baselines included). IEEE-CIS labels **every** transaction of a client after a reported fraud as fraud, so all metrics are also reported on each client's **first fraud transaction only**, where the labelling convention cannot help.

| Model | PR-AUC (all fraud) | PR-AUC (first fraud of client) |
|---|---|---|
| Transaction Baseline | 0.134 ± 0.003 | 0.058 |
| Combined (our hand-crafted graph) | 0.186 ± 0.006 | 0.097 |
| Neighbour-Aggregation XGBoost | 0.218 ± 0.004 | 0.117 |
| Temporal GraphSAGE | 0.224 ± 0.006 | **0.132** |
| Temporal GAT | 0.204 ± 0.010 | 0.107 |
| TGN-style Memory | 0.131 ± 0.006 | 0.058 |
| GraphSAGE inductive / **transductive** | 0.082 / **0.249** | 0.046 / **0.103** |
| Vesta's ~430 engineered features (tuned XGBoost) | 0.536 ± 0.004 | **0.472** |
| Vesta + our graph features | 0.535 ± 0.003 | 0.471 |
| Vesta + Neighbour-Aggregation XGBoost | 0.525 ± 0.006 | 0.469 |
| Vesta + Temporal GAT | 0.437 ± 0.016 | 0.336 |
| Vesta + graph + delayed labels (30 d) | **0.621 ± 0.004** | 0.454 |

- **Relational context helps basic transaction features**, including on first-fraud transactions (0.058 → 0.132, 2.3×).
- **It adds nothing on top of Vesta's engineered features** (0.535 vs 0.536; 0.471 vs 0.472), which already encode relational counts. The incremental value depends on what the feature store already contains.
- **Delayed-label gains are a labelling-convention effect.** With a 30-day delay, labels add +8.5 pp on all fraud but −1.8 pp on first-fraud transactions. Delay sweep for Vesta + graph + labels (all / first fraud): D = 0: 0.852 / 0.540; 1 d: 0.756 / 0.484; 7 d: 0.691 / 0.467; 30 d: 0.621 / 0.454; 90 d: 0.559 / 0.466; none: 0.535 / 0.471.
- **Ignoring label delay inflates PR-AUC by 8.6 pp even on first-fraud transactions** (0.540 vs 0.454), and by 23 pp overall.
- **Transductive inflation:** +16.7 pp overall and +5.7 pp on first-fraud transactions.
- **Ring-clustered CIs are about 2× wider than i.i.d. CIs** (e.g. Vesta [0.492, 0.575] vs [0.517, 0.554]).
- **C1 on 225 proxy rings** (≥ 3 test transactions; 0.5% budget): money lost before the first alert is 82% for the baseline, 65% for Combined and neighbour aggregation, and 54–61% for the Vesta models. Most IEEE fraud is single-client bursts (the median proxy ring is one transaction), so ring-level gains are smaller than on synthetic data or AMLworld.

### Sim-to-real calibration (`results/calibration/`)

The share of test fraud on entities already used by fraud in training is matched against the synthetic reuse dial:

| Dataset | Seen-bad share | Estimated reuse p̂ |
|---|---|---|
| IEEE-CIS | 16% (any entity), 15% (client) | **0.15–0.33** |
| AMLworld | 27% (sender or receiver), 24% (sender) | **0.21–0.47** |

Both real datasets sit in the **low-to-moderate reuse regime**, where the synthetic results show leakage inflation is large and model differences are visible, unlike fixed-pool benchmarks.

**Out-of-sample check:** at IEEE's p̂ the synthetic dose–response (C3) predicts **+13.6 to +16.2 pp** of transductive inflation; IEEE-CIS shows **+16.7 pp**. The comparison is in absolute pp; relative inflation is not comparable across base rates. On AMLworld (p̂ 0.21–0.47) the simulator predicts +11 to +15 pp but **+33 pp** is observed: right direction, magnitude underestimated at AMLworld's 40× lower fraud rate, and partly a weak embedding-only inductive baseline.

## Project structure

```
├── run_experiment.py        # CLI for every experiment
├── experiments/             # main, c2, c3, c4, c4_adaptive, g5, sensitivity, tgn_reference,
│                            # ieee_cis, amlworld, calibration, examples, originals_diagnostic
├── scripts/                 # fetch_third_party.sh, run_original_gnn.py (isolated runner)
├── src/
│   ├── data_generator.py    # rings, legit sharing, reuse + adversary settings, HARD_PRESET
│   ├── preprocessing.py     # transaction features, chronological split
│   ├── graph_features.py    # temporal relationship features (O(1) incremental tracker)
│   ├── label_features.py    # neighbour fraud-label features under label delay
│   ├── ring_metrics.py      # ring-level early-detection metrics
│   ├── clock_weights.py     # C5: takedown-value weighting, pseudo-rings, ESS tempering
│   ├── temporal_gnn.py      # temporal index, GraphSAGE/GAT/R-GCN/CARE-style, TGN-style, NbrAgg-XGB
│   ├── tgn_pyg.py           # reference TGN from PyTorch Geometric components
│   ├── original_gnn.py      # wrapper for original CARE-GNN / PC-GNN
│   ├── graph_model.py       # tabular models, entity GraphSAGE (inductive/transductive)
│   ├── pipeline.py          # datasets, tuning, fit/score, alert budgets, random split
│   ├── evaluation.py        # metrics, bootstrap CIs, plots
│   └── explainability.py    # human-readable relationship risk signals
├── paper/                   # IICTDS 2026 paper (EAI template); tables/ and figures/ generated from results/
├── tests/                   # 36 leakage / reproducibility / methodology tests
└── results/                 # main, c2, c3, c4, c4_adaptive, g5, sensitivity, amlworld, ieee, ...
```

## Limitations

- **Synthetic magnitudes depend on generator assumptions.** The sensitivity sweep shows the *signs* of all findings are stable, and the real datasets confirm the main effects, but absolute numbers should not be over-read.
- **Real-data proxies.** IEEE-CIS has no ground-truth rings or IP addresses (proxy rings and network fingerprints are used). AMLworld is itself simulated, though widely used.
- **Attackers.** The adaptive attacker is greedy over five strategies with black-box access; stronger optimisation could find cheaper evasions.
- **GNN training on AMLworld** uses all laundering plus a 30% sample of legitimate transfers (5M rows).
