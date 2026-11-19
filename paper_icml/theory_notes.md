# Theory notes for the ICML version (draft — to be checked by the authors)

Setting. Ring r has transactions k = 1..n (time order), amounts a_k > 0, remaining money R_k = Σ_{j≥k} a_j,
total T = R_1. A detector alerts transaction k with probability p_k; alerts are independent given the
scores. Takedown at the first alert saves R_{k*}, where k* is the first alerted index.

Exact objective (ring-survival):   V = Σ_k q_k R_k,   q_k = p_k Π_{j<k}(1 − p_j)   (= P(k* = k)).
TVW surrogate (first-order):      U = Σ_k p_k R_k.

## Proposition 1 (surrogate gap)
0 ≤ U − V ≤ Σ_k p_k R_k Σ_{j<k} p_j ≤ (T/2)(Σ_k p_k)^2.

Proof. U − V = Σ_k p_k R_k (1 − Π_{j<k}(1 − p_j)) ≥ 0 since each factor is in [0,1].
By the union bound 1 − Π_{j<k}(1 − p_j) ≤ Σ_{j<k} p_j, and R_k ≤ T, so
U − V ≤ T Σ_k Σ_{j<k} p_k p_j ≤ (T/2)(Σ_k p_k)^2. ∎

Reading. TVW optimises V faithfully only while the expected number of alerts per ring, Σ_k p_k, is small.
Once a detector already alerts several transactions of a ring, U keeps rewarding later alerts that V does
not value (the ring is already stopped). This is the regime of strong relational detectors, and the reason
to optimise V directly.

## Lemma 2 (gradient discounting)
With S_m = Π_{j<m}(1 − p_j),
∂V/∂p_m = S_m [ R_m − Σ_{k>m} p_k Π_{m<j<k}(1 − p_j) R_k ],
and 0 ≤ ∂V/∂p_m ≤ S_m (R_m − 0) ≤ S_m T, which → 0 as any earlier p_j → 1.
(Under U the gradient is R_m, independent of the other transactions.)

Proof. V = Σ_k p_k S_k R_k. The term k = m contributes S_m R_m; for k > m, ∂S_k/∂p_m = −S_k/(1 − p_m)
= −S_m Π_{m<j<k}(1 − p_j). Summing gives the formula. The subtracted sum is an expectation of R_k (k > m)
under a sub-probability distribution, so it lies in [0, R_{m+1}] ⊆ [0, R_m]; hence the bracket is in
[R_m − R_{m+1}, R_m] = [a_m, R_m] ⊆ [0, T]. ∎

Reading. The ring-survival loss automatically stops pushing on late transactions of rings that are
already caught early, and concentrates capacity on rings that are not — the formal version of
"train on the ring's clock". Verified numerically in tests/test_extensions.py.

## Proposition 3 (headroom bound)
For any two detectors A (control) and B, the improvement in the money-saved share satisfies
(V_B − V_A)/T ≤ 1 − V_A/T = L_A/T, the control's lost share. Pooled over rings,
Δ(saved share) ≤ Σ_r L_{A,r} / Σ_r T_r.

Reading. No ring-clock method can gain more than the control's money lost before first alert. On AMLworld
the relational controls lose 0.01–0.14% of pattern money, so any gain is bounded by ≤ 0.14 points —
the observed null result is predicted, not a failure.

## Proposition 4 (single-transaction rings)
If every ring has n = 1, then V = U = p_1 T for every ring, so the ring-survival loss and TVW both reduce
to amount-weighted positive-class likelihood; there is no within-ring (temporal) signal to exploit.
More generally the within-ring benefit vanishes as the share of rings with n ≥ 2 goes to 0.

Reading. IEEE-CIS proxy rings are mostly single-client bursts (median one transaction): ring-clock
training can only re-weight between rings by amount, which is why it is neutral there.

## Proposition 5 (weight variance / ESS) — for TVW only
For self-normalised weighted empirical risk with weights w, the variance of the weighted risk estimate
scales as σ²/ESS(w), ESS = (Σw)²/Σw² (Kong 1992). Tempering w^γ with the largest γ s.t. ESS ≥ ρ n bounds the
variance inflation by 1/ρ while keeping the weights' ordering (monotone in R). The ring-survival loss
avoids the issue: each ring's term is bounded in [0,1] (it uses R_k/T), and rings are weighted by T^γ_ring.

## Predicted experiment ("headroom curve")
Gain of ring-clock training should (i) grow with the control's lost share (Prop. 3), (ii) vanish for
single-transaction rings (Prop. 4), and (iii) for strong detectors (large Σp) favour the exact loss over TVW
(Prop. 1). Test on the simulator by varying ring size and camouflage, and place IEEE-CIS / AMLworld /
new datasets on the same axes.

## Proposition 6 (in-sample saturation of the exact objective) — motivates prefix training
From Lemma 2, ∂V/∂p_1 = R_1 − Σ_{k>1} p_k Π_{1<j<k}(1 − p_j) R_k. If the later transactions of a ring are
detected with high probability (p_2 → 1), then ∂V/∂p_1 → R_1 − R_2 = a_1: catching the FIRST transaction
is worth only its own amount. On training data, rings' later transactions are easy (their relational
footprint is large and their labels are fitted), so the exact objective saturates and stops pushing on early
transactions — exactly the ones that are hard at deployment, where a new ring is observed only up to "now".
Empirical status: NOT SUPPORTED as the explanation of the pilot. Train-ring V/T (0.66–0.72) ≈ test-ring
V/T (0.57–0.75) and early-transaction scores are not higher in training, so in-sample saturation does not
explain why the exact loss lowered first-transaction recall (0.49 → 0.38–0.44). The lemma's math stands;
the causal story is retracted. Next hypothesis: ring weighting mismatch (equal-weight rings vs the
money-weighted metric) — test gamma_ring = 1.

## Definition (prefix ring-survival objective)
At deployment a ring is observed only up to the current time. Let m ~ Uniform{1..n} be a random observation
horizon and V^{(m)} = Σ_{k≤m} q_k R_k (no credit if the ring is not alerted within the prefix). The prefix
objective is E_m[V^{(m)}] / T. With m = 1 it is p_1 (pushes the first transaction directly); averaging over m
retains the survival discounting within each prefix while removing the dependence on later, easy
transactions that will not have happened yet when the decision matters.
Closed form: E_m[V^{(m)}] = (1/n) Σ_{m=1}^{n} Σ_{k≤m} q_k R_k = Σ_k q_k R_k (n − k + 1)/n.
So the prefix objective is the exact survival objective with a positional discount (n − k + 1)/n on late
transactions. Its gradient on p_1 is R_1 − Σ_{k>1} p_k Π_{1<j<k}(1 − p_j) R_k (n − k + 1)/n, so the saturating
term is shrunk by the positional factor; for p_k → 1 on all k > 1 it tends to R_1 − R_2 (n−1)/n = a_1 + R_2/n,
i.e. the first transaction keeps a share R_2/n of the downstream value instead of nothing.
(Check: n = 1 gives p_1 R_1; matches the m = 1 case.)
