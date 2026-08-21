# Relationship Intelligence Layer (RIL)

Research prototype for testing whether **relationship context** around a payment transaction adds predictive value beyond conventional transaction-level features.

> **Disclaimer:** This is a research prototype using synthetic data and is **not** representative of Razorpay's production fraud detection system.

## Problem

Individual transactions can look normal in isolation while the surrounding network of customers, devices, cards, IPs, and merchants reveals coordinated suspicious activity.

## Hypothesis

When transaction-level behavior looks normal, explicitly modeling relationships surrounding that transaction provides additional predictive information for detecting coordinated suspicious activity.

## Architecture

```
Synthetic Data → Transaction Features + Temporal Graph Features
                        ↓
        ┌───────────────┼───────────────┐
        ↓               ↓               ↓
   Baseline Model   Relationship Model   Combined Model
   (txn only)       (graph only)         (txn + graph)
                        ↓
              Optional GraphSAGE embeddings
                        ↓
              Evaluation + Explainability + API
```

## Synthetic Data

Generates ~100K transactions across 20K customers with:

- **Normal customers** — stable devices, merchants, mostly successful payments
- **Legitimate shared environments** — family devices, shared Wi-Fi (not fraud)
- **Coordinated fraud networks** (~3.5%) — device farms, card testing, IP concentration, etc.
- **Graph-hidden fraud** — fraudulent txs with normal-looking transaction-level features

Fixed random seed (`42`) for reproducibility.

## Evaluation Methodology

**No temporal leakage:** Graph and transaction features for time T use only data available at or before T.

**Chronological split:**

| Split | Fraction |
|-------|----------|
| Train | First 70% of time |
| Validation | Next 15% |
| Test | Final 15% |

**Metrics:** ROC-AUC, PR-AUC, Precision, Recall, F1, Recall @ 1% FPR

**Special evaluations:**

- Graph-hidden fraud recall
- False positive rate on legitimate shared-device transactions
- Ablation study (counts vs temporal vs full graph features)

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run full experiment (data → models → metrics → plots)
python run_experiment.py

# Run tests
python -m pytest tests/ -q

# Start API (after experiment)
uvicorn src.api:app --reload --port 8000

# Demo dashboard (optional)
streamlit run dashboard.py
```

### API Example

```bash
curl -X POST http://localhost:8000/risk/analyze \
  -H "Content-Type: application/json" \
  -d '{"transaction_id": "TX050000"}'
```

## Project Structure

```
├── data/                  # Generated CSVs
├── src/
│   ├── data_generator.py  # Synthetic data with fraud patterns
│   ├── preprocessing.py   # Transaction features + chronological split
│   ├── graph_features.py  # Temporal relationship features
│   ├── baseline_model.py  # Transaction-only classifier
│   ├── graph_model.py     # Relationship + Combined + GraphSAGE
│   ├── evaluation.py      # Metrics and plots
│   ├── explainability.py  # Human-readable risk signals
│   └── api.py             # FastAPI /risk/analyze
├── results/               # Metrics, plots, saved models
├── tests/
├── run_experiment.py      # One-command pipeline
└── dashboard.py           # Streamlit demo
```

## Results

From `python run_experiment.py` (synthetic data, seed=42, HistGradientBoostingClassifier):

| Model | ROC-AUC | PR-AUC | Precision | Recall | F1 | Recall @ 1% FPR |
| ----- | ------: | -----: | --------: | -----: | -: | --------------: |
| Transaction Baseline | 0.9439 | 0.6627 | 0.3962 | 0.7675 | 0.5226 | 0.5535 |
| Relationship Intelligence | 0.9902 | 0.9387 | 0.4741 | 0.9631 | 0.6354 | 0.9299 |
| Combined Model | 0.9937 | 0.9553 | 0.4991 | 0.9723 | 0.6596 | 0.9465 |
| GraphSAGE Model | 0.9961 | 0.9687 | 0.9755 | 0.8801 | 0.9253 | 0.9502 |

**Key findings:**

- Combined model PR-AUC improves **+29.3 pp** over the transaction baseline
- Graph-hidden fraud recall: Baseline **85.6%** → Combined **97.6%**
- Legitimate shared-device FPR: **1.5%** (sharing alone does not equal fraud)
- Ablation: simple relationship counts already drive most of the lift; full temporal graph features refine further

Artifacts: `results/metrics_table.csv`, `results/*.png`

## Limitations

- Synthetic data only; patterns may not transfer to production
- Lightweight GraphSAGE (no PyTorch Geometric) for install reliability
- No real-time streaming or production-scale graph storage
- Explainability provides risk signals, not fraud determinations

## Research Question

> When transaction-level behavior looks normal, does explicitly modeling the relationships surrounding that transaction provide additional predictive information for detecting coordinated suspicious activity?

The experiment prints a data-driven conclusion to stdout.
