# Does layer sensitivity predict pruning tolerance?

Sensitivity-Aware Constrained PPO, step 2. Computed on 2026-09-27 **before
any constrained PPO run**, from validation data only; no policy was evaluated
on the test set for this analysis.

Sources: `evaluation/policy_landscape.py` → `policy_landscape_validation.csv`
(all 1,296 policies: sparsity, probe accuracy, full-validation accuracy and
loss, rewards, Pareto status); `evaluation/constrained_masks.py` →
`constrained_sensitivity_curves.csv`, `sensitivity_action_masks.csv`;
`evaluation/predictive_validity.py` → `sensitivity_predictive_validity.csv`.

## Verdict

**Yes: single-layer sensitivity predicts pruning tolerance in context, with
one caveat.** The layer order by sensitivity is also the reverse order of
layer size (Spearman −1.0), so analyses driven by total sparsity cannot tell
the two apart. The size-independent test (B) supports sensitivity. It is safe
to proceed to constrained PPO, and to interpret it as a test of a signal that
has real predictive content.

## Single-layer sensitivity (full validation, 5,000 images)

| Layer | Params | Drop at 10 / 20 / 30 / 40 / 60% (pp) | Accuracy-mask max | Loss-mask max |
|---|---|---|---|---|
| features.0 | 864 | 0.04 / 0.74 / 2.88 / 8.38 / 34.60 | 20% | 10% |
| features.3 | 18,432 | −0.04 / −0.08 / 0.28 / 0.32 / 1.56 | 40% | 40% |
| features.6 | 73,728 | 0.06 / 0.00 / 0.02 / 0.12 / 0.64 | 40% | 40% |
| classifier.1 | 524,288 | −0.02 / −0.08 / 0.00 / −0.12 / −0.28 | 60% | 60% |

## B. Size-independent test: damage in context

For each layer and ratio, the mean validation-accuracy damage over all 216
settings of the other three layers (the layer's average marginal effect when
combined with others):

| Layer | 10% | 20% | 30% | 40% | 60% | Tolerated (mean ≤ 1.0 pp) |
|---|---|---|---|---|---|---|
| features.0 | 0.28 | 1.17 | 3.11 | 8.41 | 34.02 | 10% |
| features.3 | 0.04 | 0.10 | 0.28 | 0.72 | 2.07 | 40% |
| features.6 | 0.03 | 0.02 | 0.03 | −0.05 | 0.24 | 60% |
| classifier.1 | −0.00 | 0.01 | 0.02 | 0.01 | 0.01 | 60% |

The ordering of in-context damage matches the sensitivity ordering exactly.
Spearman(sensitivity, in-context tolerated ratio) = −0.95 for both
definitions; exact permutation p = 0.17. With four layers there are only 24
orderings, so no p below 0.083 is attainable; the evidence is the consistent
ordering, not the p-value.

**How well the single-layer masks match in-context tolerance:**

- Accuracy mask [20, 40, 40, 60] vs in-context [10, 40, 60, 60]: agrees on 2
  of 4 layers. It is **too permissive for features.0** (in-context damage at
  20% is 1.17 pp) and **too strict for features.6**.
- Loss mask [10, 40, 40, 60]: agrees on 3 of 4; too strict for features.6 only.

## A, C, D. High-performing policies

| Set | n | features.0 | features.3 | features.6 | classifier.1 |
|---|---|---|---|---|---|
| all policies | 1,296 | 26.7 | 26.7 | 26.7 | 26.7 |
| Pareto-efficient (sparsity ↑, val accuracy ↑) | 19 | 10.0 | 30.5 | 46.3 | 60.0 |
| dominated | 1,277 | 26.9 | 26.6 | 26.4 | 26.2 |
| top 10% by val accuracy within 5-pp sparsity bins | 139 | 1.9 | 13.5 | 19.0 | 27.3 |

(mean pruning ratio, %)

- **A.** High performers prune the sensitive layer far less: features.0
  averages 10% on the Pareto front and 1.9% among the within-bin top 10%,
  against 26.7% overall.
- **C.** Pareto minus dominated mean ratio: features.0 −16.9 pp, features.3
  +3.9, features.6 +19.9, classifier.1 +33.8.
- **D.** Spearman(sensitivity, mean ratio) = −1.0 in the Pareto set and in the
  within-bin top set (exact p = 0.083, the minimum possible), +1.0 among
  dominated policies. Per-policy Spearman averages −0.87 on the Pareto front,
  −0.57 in the within-bin top set, and 0.00 over all policies.

A and D are also exactly what "prune the big layers" would produce, because
size and sensitivity are perfectly confounded here. The within-bin top set
controls for total sparsity and still prunes features.0 least, which favours
sensitivity.

## E. What each mask makes reachable (before any PPO)

| Mask | Policies admitted | Pareto-efficient admitted (of 19) | Max sparsity | Reward-optimal inside (λ_s 0.01 / 0.02), global rank |
|---|---|---|---|---|
| accuracy, correct | 450 | 7 | 56.7% | `[0,2,4,5]`, rank 1 / rank 1 |
| accuracy, shuffled (2 distinct) | 450 | 0 | 41.6% / 38.1% | ranks 139–209 |
| loss, correct | 300 | 7 | 56.7% | `[0,2,4,5]`, rank 1 / rank 1 |
| loss, shuffled (2 distinct) | 300 | 0 | 41.3% / 36.9% | ranks 139–230 |

- **The correct masks keep the global reward optimum and 7 of the 19
  validation-Pareto policies. Every shuffled mask excludes the whole Pareto
  front and caps sparsity near 37–42%,** because every value-changing
  permutation moves a restrictive mask onto classifier.1.
- Below ~35% sparsity, correct and shuffled masks reach the same best
  validation accuracy; they differ only in whether high sparsity is reachable
  at all.

**Implication for the constrained-PPO comparison (flagged before any run).**
Shuffled-mask PPO cannot reach the sparsity of correct-mask PPO, so its raw
accuracy is expected to be *higher*, simply because it prunes less. The
pre-registered criterion 2 (C2 − C3 on test accuracy) therefore favours the
shuffled control by construction; criterion 3 (excess over matched global
magnitude) is the comparison that can separate correct allocation from
pruning less.
