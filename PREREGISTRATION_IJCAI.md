# Pre-registration: IJCAI 2027 confirmatory experiments

Written 2026-11-19, **before** any of the experiments below were run (`bash scripts/run_ijcai_overnight.sh`).
Method under test: TVW v2 (`src/clock_weights.py`: pseudo-rings with identity linking, ring-decomposed ESS
tempering, beta = 1). Nothing in this document is changed after results exist; deviations are reported as such.

## Data and design (fixed)
* **AMLworld HI-Small and LI-Small**, first 10 clean days (as in `experiments/amlworld.py`).
* **Rolling origin**: test window = one day d (d = 4..10), validation = day d-1, training = all earlier days
  (>= 2 days). Rings = ground-truth laundering patterns with >= 2 transactions that START in the test day.
  **Primary analysis uses only test days that end before the development validation period began** (those
  rows were never used for any model, threshold or design decision during development). Test days overlapping
  the development validation/test periods are reported separately as secondary.
* Seeds 42, 43, 44 on primary windows; seed 42 only on secondary windows (compute budget). Models: hand-crafted graph features (XGBoost), neighbour-aggregation XGBoost, temporal
  GraphSAGE and GAT (trained on all laundering + 30% of legitimate rows, 6 epochs). Hyper-parameters reused from the
  development runs (HI) or tuned once on the first window's validation day (LI); identical for control and TVW.
* Budgets: prevalence-matched multiples c in {0.05, 0.15, 0.3, 0.5, 1.0} of validation positives.
  **Primary budget: c = 0.15** (tight, high-headroom regime). Others are secondary.

## Hypotheses and tests
**H1 (primary).** With neighbour-aggregation XGBoost at c = 0.15, TVW catches more pattern rings EARLIER than
LATER than ordinary training. Unit = ring (pooled over windows and both datasets); a ring is "earlier" if its
mean (over seeds) number of transactions before the first alert is lower with TVW. One-sided sign test on
rings with a difference, alpha = 0.05.

**H2.** Across all rolling cells (dataset x window x model x budget), the money-weighted TVW gain is positively
associated with RECOVERABLE headroom (money lost in rings the control detects late, not never).
One-sided Spearman test, alpha = 0.05.

**H3 (secondary).** H1's test for temporal GraphSAGE and GAT separately (Holm correction over the two).

**H4 (simulator, objectives).** On the hard preset (10 seeds, 0.5% FPR budget), TVW reduces money lost before
the first alert more than each reweighting baseline (class-balanced, amount-weighted; plus focal for GNNs).
Paired bootstrap of loss(TVW) - loss(baseline) over rings; one-sided, Holm across comparisons.

## Reporting rules
* Every hypothesis is reported with its test statistic whether it holds or not.
* Money-weighted ring metrics are always reported WITH count-based ring metrics and the largest-ring money share
  (one ring held 43-85% of test-ring money in development).
* Real-data CIs resample RINGS (all seeds of a ring together), never (seed, ring) pairs.
* PR-AUC cost of TVW is reported for every cell.

## Amendment log (all made before any confirmatory result existed)
* 2026-11-19: development validation began 2022-09-07 14:52, so with >= 3 training days only test days 5-6
  would be primary (~26 rings per dataset). Changed to >= 2 training days (adds day 4; training = days 1-2,
  1.86M transfers). Secondary windows reduced to one seed and GNN epochs 8 -> 6 to fit one night of compute.
  Decided from row counts and dates only.
* Disclosure: a code smoke test on HI-Small test day 6, seed 42 ran neighbour-aggregation XGBoost with its real
  settings, so those outputs are identical to ONE primary cell (HI-Small, day 6, seed 42) and were seen by the
  analyst before the confirmatory run: at c = 0.15 money-weighted loss was ~0.99 for both, and a smoke run of
  the report generator showed 1 ring earlier vs 3 later with TVW on that cell (13 rings). No design decision
  used them; the cell stays in the primary analysis exactly as specified. The GNN smoke outputs (1 epoch) differ
  from the confirmatory settings and are discarded.
