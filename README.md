# Relationship Intelligence Layer (RIL)

Research prototype for testing whether **relationship context** around a payment transaction adds predictive value beyond conventional transaction-level features.

> **Disclaimer:** This is a research prototype using synthetic data.

## Research question

> When transaction-level behaviour looks normal, does explicitly modelling the relationships surrounding that transaction (customers, devices, cards, IPs, merchants) provide additional predictive information for detecting coordinated suspicious activity?

## Architecture

```
Synthetic data (rotating fraud rings, legit sharing)
        │
        ├── Transaction features   (src/preprocessing.py)   – strictly past history
        └── Temporal graph features (src/graph_features.py) – strictly past graph state
                        │
        ┌───────────────┼───────────────┬──────────────────┐
   Baseline        Relationship      Combined          GraphSAGE
   (txn only)      (graph only)      (txn + graph)     (embeddings of past graph)
                        │
   Thresholds tuned on validation → test metrics + ring-clustered bootstrap CIs
   + graph-hidden recall + shared-environment FPR + ablation + explanations
```

## Synthetic data design

`src/data_generator.py` produces ~100K transactions over Mar–Sep 2024 (3.5% fraud), from a single seeded RNG.

| Property | How it is generated | Why it matters |
|---|---|---|
| **Rotating fraud rings** | ~135 rings, each active 3–21 days, each with *fresh* accounts, devices, IPs and stolen cards, then retired | Test-period rings use entities never seen in training, so models can't win by memorising bad IDs |
| **Six ring patterns** | device farm, card testing, account farm, IP concentration, merchant targeting, account takeover (hijacked existing legit accounts) | Different relational signatures |
| **Noisy infrastructure** | 25% of fraud txns use clean-looking devices/IPs (carrier IPs, borrowed devices) | Graph signals are not perfect |
| **Graph-hidden fraud** (~45% of fraud) | Amount, payment method, success and hour drawn from the **same distributions as legit traffic** | Verified each run: an intrinsic-feature classifier scores ~0.51 AUC on hidden fraud vs legit |
| **Legitimate sharing** | Households share devices/IPs/cards; carrier-grade NAT IPs and public Wi-Fi shared by many legit users; occasional borrowed devices | "Shared" ≠ "fraud" |
| **Legit new users** | 8% of legit customers sign up during the window and transact normally | "New account" is not a fraud label proxy |
| **Uniform ring density** | Rings may straddle the timeline edges (clipped) | Fraud prevalence is similar in train / val / test |

## Evaluation methodology

- **No temporal leakage.** Every feature for a transaction at time *T* uses only transactions strictly before *T*. This is enforced by tests: features computed on a truncated dataset must equal those computed on the full dataset, and every graph feature is checked against a brute-force reference implementation.
- **Chronological split:** train 70% / validation 15% / test 15% (by time).
- **Thresholds are tuned on validation** (F1-maximising) and then applied unchanged to test. The test set is never used for any choice.
- **GraphSAGE is evaluated inductively.** The classifier is trained on embeddings of later training transactions computed from a graph of *earlier* training transactions. At scoring time the graph contains only transactions before the scored period (validation: train; test: train + val), and a guard raises an error if history overlaps the scored rows.
- **Metrics:** ROC-AUC, PR-AUC (average precision), Precision / Recall / F1 at the validation threshold, Recall @ 1% FPR.
- **Uncertainty:** 95% CIs come from a ring-clustered bootstrap (rings are resampled as whole units, because fraud transactions within a ring are correlated). The paired CI on the combined − baseline PR-AUC gain uses identical resamples.
- **Special evaluations:** graph-hidden fraud recall, FPR on legitimate shared environments, ablation (counts → temporal → full graph), multi-seed stability (new dataset + new models per seed).

## Quick start

```bash
pip install -r requirements.txt

python run_experiment.py                         # ~40 s, writes results/
python run_experiment.py --seeds 42 43 44 45 46  # + multi-seed stability table (~3 min)

python -m pytest tests/ -q                       # ~12 s
```

Runs are bit-for-bit reproducible: identical outputs across processes and `PYTHONHASHSEED` values with the pinned requirements.

## Project structure

```
├── data/                   # Generated CSVs (raw transactions + entity tables)
├── src/
│   ├── data_generator.py   # Synthetic data: rotating rings, legit sharing, hidden fraud
│   ├── preprocessing.py    # Transaction features + chronological split
│   ├── graph_features.py   # Temporal relationship features (O(1) incremental tracker)
│   ├── baseline_model.py   # Transaction-only classifier
│   ├── graph_model.py      # Relationship / Combined classifiers + GraphSAGE
│   ├── classifier_utils.py # XGBoost factory (sklearn fallback)
│   ├── evaluation.py       # Metrics, thresholds, bootstrap CIs, plots, conclusion
│   └── explainability.py   # Human-readable relationship risk signals
├── results/                # Metrics, plots, explanations, run summary
├── tests/test_pipeline.py  # Leakage, reproducibility and methodology tests
└── run_experiment.py       # One-command pipeline
```

## Results

Seed 42, XGBoost, test set = 15,000 transactions, 4.2% fraud, 32 rings (none active during training):

| Model | ROC-AUC | PR-AUC [95% CI] | Precision | Recall | F1 | Recall @ 1% FPR |
| ----- | ------: | --------------: | --------: | -----: | -: | --------------: |
| Transaction Baseline | 0.979 | 0.829 [0.761, 0.879] | 0.875 | 0.642 | 0.741 | 0.719 |
| Relationship Intelligence | 0.999 | 0.977 [0.964, 0.986] | 0.991 | 0.856 | 0.919 | 0.968 |
| Combined | 0.999 | **0.984** [0.973, 0.990] | 0.983 | 0.886 | 0.932 | 0.965 |
| GraphSAGE | 0.991 | 0.856 [0.794, 0.899] | 0.855 | 0.782 | 0.817 | 0.849 |

| | Baseline | Relationship | Combined | GraphSAGE |
|---|---:|---:|---:|---:|
| Graph-hidden fraud recall | 56.7% | 87.0% | 86.6% | 78.3% |
| FPR on legit shared environments | 1.69% | 0.08% | 0.12% | 0.12% |

**Ablation (PR-AUC):** transaction only 0.829 → + counts 0.925 → + temporal 0.972 → full graph **0.984**. Simple counts deliver ~62% of the gain; temporal features (recency, velocity, 7-day windows, failure rates) deliver most of the rest.

**Multi-seed stability (5 seeds, mean ± std):**

| | PR-AUC | Graph-hidden recall |
|---|---|---|
| Baseline | 0.832 ± 0.026 | 62.5% ± 4.7 |
| Relationship | 0.975 ± 0.003 | 87.2% ± 1.6 |
| Combined | 0.983 ± 0.001 | 86.7% ± 2.0 |
| GraphSAGE | 0.785 ± 0.053 | 78.0% ± 2.7 |
| **Combined − baseline** | **+15.1 pp ± 2.6** (range +11.3 to +17.6) | |

**Key findings:**

- Relationship context improves PR-AUC by **+15.5 pp** (95% CI +11.0 to +21.6 pp) and the gain is positive on every seed.
- On graph-hidden fraud, which is indistinguishable on intrinsic attributes (sanity AUC 0.51), recall rises from 57% to 87%.
- Relationship features also cut false positives on legitimately shared devices and IPs (1.7% → 0.1%).
- The lightweight GraphSAGE underperforms the hand-crafted temporal features. Ring entities are new at test time, so they have no embedding in the past graph; explicit time-windowed features capture "many new accounts on this device this week" directly.

Artifacts: `results/metrics_table.csv`, `ablation_metrics.csv`, `multi_seed_*.csv`, `run_summary.json`, `conclusion.txt`, `example_explanations.csv`, `*.png`.

## Limitations

- **Synthetic data only.** The fraud generator encodes relational structure by design, so the size of the gain reflects these assumptions and will not transfer directly to production.
- Fraud and legit activity follow simple stationary processes (no seasonality, no adaptive adversaries, no label delay).
- GraphSAGE is a minimal 2-layer encoder without PyTorch Geometric. It is not tuned and is not a strong GNN baseline.
- Hyperparameters are fixed (not tuned on validation). Only the decision threshold is tuned.
- Explanations are risk signals relative to training percentiles, not fraud determinations.
