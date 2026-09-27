# Soft Sensitivity-Prior PPO — pre-registration

Written 2026-09-27, before any implementation, check or run of this
experiment. It is hashed and committed before anything else is done. Nothing
here may change after results are visible; any deviation is reported as one.

This is the final pre-registered sensitivity experiment. All earlier results
(original and refined sensitivity ablations, constrained PPO, reward sweep,
matched-sparsity baselines, PPO multi-seed, latency/storage, predictions) are
preserved unchanged.

**Hypothesis.** Layer sensitivity is useful as a *soft prior* over PPO pruning
actions: aggressive pruning of sensitive layers is discouraged but never
forbidden.

## Protocol (unchanged)

Baseline `checkpoints/cnn_baseline_FIXED.pth` (sha256 `ca28f834…`, test
77.03%). Split 45,000 / 5,000 / 10,000, seed 42. Validation only for
sensitivity, reward, policy selection and convergence diagnostics; test only
for evaluating frozen final policies, once per run. Test data are never used
to choose β, the definition, the reward, policies, stopping or models.
Prunable layers features.0, features.3, features.6, classifier.1; actions
0/10/20/30/40/60%.

## Sensitivity (already computed; no new estimator)

From `results/refined_sensitivity_values.csv` (registered 2026-09-27 01:24,
sha256 `3efd306f…`), min-max normalised to [0, 1]:

| Layer | **Loss (primary)** | Accuracy (secondary) |
|---|---|---|
| features.0 | 1.000000 | 1.000000 |
| features.3 | 0.032278 | 0.051062 |
| features.6 | 0.013358 | 0.029824 |
| classifier.1 | 0.000000 | 0.000000 |
| mean | 0.261409 | 0.270221 |

a_norm = ratio / 60: 0, 0.167, 0.333, 0.5, 0.667, 1.0.

## The soft prior

For the current layer ℓ and action a:

    adjusted_logit(ℓ, a) = raw_logit(ℓ, a) − β · S_ℓ · a_norm(a)

Applied inside the policy, before the categorical distribution is formed, so
it shapes sampling during rollouts, the log-probabilities and entropy used in
the PPO update, and the deterministic final policy alike. No action is ever
masked, clipped, remapped or penalised through the reward. The current layer
is read from the observation's layer-index input (index/3 ∈ {0, ⅓, ⅔, 1}).

**A-priori magnitude (a property of the fixed inputs, not a result).** With the
loss vector, the prior acts almost only on features.0: at β = 0.5 the largest
penalty on any other layer is 0.5 · 0.032 = 0.016 logits. At uniform raw
logits it lowers P(features.0 = 60%) from 0.167 to 0.125 (β = 0.5) or 0.091
(β = 1.0).

## β (fixed)

**Primary β = 0.50. Robustness β = 1.00.** No other values. The conclusion
rests on β = 0.50; β = 1.00 cannot replace it. If both fail, stop.

## Conditions

| | State sensitivity | Prior | β |
|---|---|---|---|
| P0 no prior (control) | 0 | none | 0 |
| P1 correct prior (proposed) | 0 | correct S | 0.5 |
| P2 shuffled prior | 0 | S permuted among layers | 0.5 |
| P3 constant prior | 0 | every layer = mean(S) | 0.5 |
| P4 state only | correct S | none | 0 |
| P5 correct prior + state | correct S | correct S | 0.5 |

P2: a seed-deterministic permutation (`numpy.random.default_rng(seed)`) in
which every layer receives another layer's value (the four values are
distinct, so every layer's value changes); same multiset; mapping recorded.
P3: identical mean penalty to P1 by construction (mean over layers and
actions of β·S·a_norm).

Primary set: P0–P5 with the loss definition, β = 0.5, λ_s = 0.01.
Robustness: P1, P2, P3 with β = 1.0 (P0 has β = 0 and is shared).
Secondary definition: P1, P2, P3 with the accuracy definition, β = 0.5.
That is 12 runs per seed.

## PPO and reward (identical everywhere)

stable_baselines3 PPO with a policy class that differs from MlpPolicy only by
the prior; with β = 0 it must reproduce standard PPO exactly. MlpPolicy
network [64, 64] tanh, orthogonal init, Adam lr 3e-4, n_steps 64, batch 32,
n_epochs 10, γ 0.99, GAE λ 0.95, clip 0.2, ent_coef 0.01, vf_coef 0.5,
max_grad_norm 0.5, 2,000 timesteps requested (2,048 trained, 512 episodes).
Reward R = 1.5 · acc_probe/77.32 + 0.01 · S(%) + 0.05 · distinct actions
(λ_s = 0.01, unchanged). Final policy = deterministic policy after training.
Terminal evaluations memoised by policy with the verified random-stream
preservation from the constrained experiment.

**Seeds: 42, 1, 2, …, 19 (20 seeds), identical for every condition, none
removed.** Rationale: memoised runs took about 33 s each at 4 workers in the
constrained experiment, so 240 runs project to about 2.5 hours. If a
measurement before launch projects more than 6 hours, the first 10 seeds
(42, 1, …, 9) are used instead, as a whole.

## Primary metric and criteria

**Primary metric M** = test accuracy − test accuracy of global magnitude
pruning over the same four layers at the run's exact total sparsity ("excess
over matched GM"). This makes every primary comparison a comparable-sparsity
comparison. Raw test accuracy and sparsity differences are reported
alongside.

Primary family (loss definition, β = 0.5), paired by seed, exact two-sided
sign-flip test on the mean paired difference, Holm over the three:

1. **P1 − P0 on M > 0**, Holm p < 0.05 (beats no prior at comparable sparsity).
2. **P1 − P2 on M > 0**, Holm p < 0.05 (correct beats shuffled).
3. **P1 − P3 on M > 0**, Holm p < 0.05 (layer-specific beats generic).

Consistency (required for Level A): in each of the three comparisons, more
seeds favour P1 than oppose it.

## Secondary outcomes (P1 vs P0; not substitutes for the primary criteria)

Holm over this family of seven:

- (a) across-seed SD of test accuracy (exact paired permutation test on the SD difference);
- (b) harmful final policies, features.3 = 60% or features.0 ≥ 20% (exact McNemar over seeds);
- (c) final policy's global reward rank at λ_s = 0.01 in the 1,296-policy validation landscape (sign-flip);
- (d) time to first reach the validation target: the sampled episode's policy has full-validation accuracy ≥ 76.0% **and** sparsity ≥ 55%. Runs that never reach it are censored, never imputed. Paired comparison: per seed, which condition reached it earlier (both censored = tie; one censored = the other is earlier); exact sign test on non-tied pairs;
- (e) trajectory AUC of full-validation accuracy of the sampled policies (mean over the 512 episodes; sign-flip);
- (f) trajectory AUC of global reward rank (sign-flip);
- (g) trajectory AUC of utility U = 1.5 · val_acc/77.32 + 0.01 · S (sign-flip).

Worst-seed accuracy is reported descriptively. Full-validation accuracy per
sampled policy comes from the existing 1,296-policy validation landscape.

Other comparisons, reported with the same statistics but outside both
families: P1 vs P4, P1 vs P5 (does the state add anything beyond the prior),
and every primary comparison at β = 1.0 and for the accuracy definition.

Every comparison reports mean, median and SD of the paired differences,
bootstrap and t 95% CIs, exact sign-flip p, exact Wilcoxon p and Cohen's d_z.

## Classification (fixed now)

- **Level A:** criteria 1, 2 and 3 all hold, with the consistency condition.
- **Level B:** not Level A, and at least one secondary outcome favours P1 over
  P0 with Holm p < 0.05, **and** on that same outcome P1 is at least as good
  as both P2 and P3 in mean (otherwise the improvement is not due to correct,
  layer-specific sensitivity).
- **Level C:** otherwise. Then sensitivity redesign stops: no new β values,
  transformations, masks, rewards or definitions.
