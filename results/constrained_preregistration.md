# Sensitivity-Aware Constrained PPO — pre-registration

Written 2026-09-27, **before** the action-level sensitivity curves, the masks,
the predictive-validity analysis or any constrained PPO run were computed.
Nothing below may be changed after results are seen; any deviation is
reported as a deviation.

Hypothesis: layer sensitivity is more useful as a constraint on the PPO action
space than as a state feature. Previous negative results (original and refined
sensitivity ablations) are unchanged and remain valid.

## Data protocol

Baseline `checkpoints/cnn_baseline_FIXED.pth` (sha256 `ca28f834…`, test 77.03%).
Split: 45,000 train / 5,000 validation / 10,000 test, seed 42 (as before).
Validation folds: the validation split in its fixed order, cut into five
disjoint 1,000-image folds, `val[0:1000]` … `val[4000:5000]`; fold 0 is the RL
probe. Validation only for sensitivity, masks, reward, policy selection and
diagnostics. Test only for the final evaluation of frozen policies.

## Sensitivity measurement

Each prunable layer (features.0, features.3, features.6, classifier.1) alone,
L1-unstructured at 10, 20, 30, 40, 60% (the non-zero action levels), from a
fresh baseline copy each time. Evaluated on the full validation split and on
each fold: accuracy, absolute/relative accuracy drop, cross-entropy loss,
absolute/relative loss increase.

## Action-mask rules

**Primary (accuracy).** For layer ℓ and ratio r, fold drops
d_k = acc(base, fold k) − acc(ℓ at r, fold k), k = 1..5, in pp.
One-sided 95% upper confidence bound UCB = mean(d) + t₀.₉₅,₄ · sd(d)/√5 with
t₀.₉₅,₄ = 2.132. Ratio r is **safe** iff UCB < 1.0 pp (non-inferiority margin
1.0 pp, fixed by the protocol). r*(ℓ) = the largest safe ratio (0 if none).
Allowed actions = every action whose ratio ≤ r* (monotonic closure; 0% is
always allowed).

**Secondary (loss).** Same procedure on relative loss increase
e_k = (loss(ℓ at r, fold k) − loss(base, fold k)) / loss(base, fold k).
Margin δ = 1.0 / (100 − 77.32) = 4.41% relative increase in validation
cross-entropy. Rationale: the accuracy margin allows the baseline's validation
error rate (22.68%) to rise by 1.0 pp, a 4.41% relative increase; the loss rule
allows the same relative degradation of the other error measure,
cross-entropy. The margin depends only on the protocol's 1.0 pp and the
baseline's validation accuracy, not on any pruning result.

## Sensitivity state values

Accuracy definition: mean over the five ratios of the relative accuracy drop
on the full validation split; loss definition: mean relative loss increase.
Each min-max normalised across the four layers to [0, 1] (all equal → 0).

## Conditions (all PPO conditions use the same `sb3_contrib.MaskablePPO`)

| | State sensitivity | Action mask |
|---|---|---|
| C0 unconstrained | zeros | all valid |
| C1 state-only | correct | all valid |
| C2 correct constrained (proposed) | correct | correct |
| C3 shuffled constrained | correct | masks permuted among layers |
| C4 correct mask, zero state | zeros | correct |
| C5 sensitivity-only heuristic | — (no PPO) | each layer at its r* |

C3: a seed-deterministic permutation (`numpy.random.default_rng(seed)`) of the
four masks in which every layer receives a mask that differs, as a set of
allowed actions, from its own. The multiset of masks and total valid-action
count are unchanged. The mapping is recorded per run.

Each condition is run for both definitions (accuracy mask + accuracy state;
loss mask + loss state). Conditions whose inputs do not depend on the
definition (C0; and C4 if the two masks coincide) are shared across
definitions, as in the refined analysis.

**Primary definition: accuracy** (the primary rule above). The loss version is
reported in full as the robustness version.

## Reward and PPO

R = 1.5 · acc_probe / 77.32 + λ_s · S(%) + 0.05 · distinct actions, unchanged.
**Primary λ_s = 0.01; robustness λ_s = 0.02.** No other λ_s.

MaskablePPO with the original hyperparameters: MlpPolicy, lr 3e-4, n_steps 64,
batch 32, n_epochs 10, γ 0.99, GAE λ 0.95, clip 0.2, ent_coef 0.01, vf_coef
0.5, max_grad_norm 0.5, 2,000 timesteps requested (2,048 trained). Reported
policy: the final deterministic policy (masked). Masks are applied by
MaskablePPO's masked categorical distribution; masked actions have zero
probability. The runner asserts that no masked action is ever taken.

Terminal-step evaluation may be memoised by policy (the reward is a
deterministic function of the four actions). This is used only after a run
with memoisation reproduces a recorded original run exactly, including every
training episode.

**Seeds:** 42, 1, …, 9 at minimum. If a measured single-run time projects the
complete design to ≤ 6 hours of wall time, seeds 42, 1, …, 19 (20 seeds) are
used for every condition. No seed is ever dropped.

## Success criteria (fixed now)

Primary family, λ_s = 0.01, accuracy definition, metric = test accuracy,
paired by seed, exact two-sided sign-flip test, Holm over the three:
C2 − C0, C2 − C3, C2 − C4.

1. **C2 beats unconstrained:** mean(C2 − C0) > 0 and Holm p < 0.05.
2. **C2 beats shuffled:** mean(C2 − C3) > 0 and Holm p < 0.05.
3. **Not explained by pruning less:** for each of 1 and 2 that holds, the
   paired difference in *excess over matched global magnitude* (test accuracy
   minus global magnitude over the same four layers at the run's exact
   sparsity) is also > 0 with sign-flip p < 0.05.

Secondary support, reported with tests where meaningful:

4. fewer harmful decisions: features.3 at 60%; features.0 at ≥ 20%; any layer
   above its primary-rule r*;
5. lower across-seed SD (exact paired permutation test);
6. better worst seed;
7. better convergence: rank of the final policy in the validation reward
   landscape, and distance to the best policy reachable under the condition's
   own mask;
8. C2 beats the heuristic C5 (sign-flip on C2 − C5).

C2 − C4 answers whether the state value adds anything beyond the mask.

**Classification.** Level A: criteria 1, 2 and 3 all hold and the effect is
stable across seeds. Level B: 1 or 2 fails on accuracy but correct masks
improve stability or harmful-decision frequency relative to both C0 and C3.
Level C: correct masks do not beat shuffled masks, or do not beat
unconstrained PPO, or any advantage disappears at matched sparsity. After a
Level C outcome no further masks or thresholds are tried.

All other statistics (median difference, bootstrap and t 95% CIs, Cohen's
d_z, exact Wilcoxon) are reported alongside, as in the refined analysis.
