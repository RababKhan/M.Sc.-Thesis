# Sensitivity feature diagnostic

Diagnosis of the **original** sensitivity feature used by the PPO agent, run on
2026-09-27 before any refined experiment. Read-only: nothing was retrained and
no existing result was changed. Validation data only; the test set was not
loaded.

Source: `evaluation/sensitivity_diagnostic.py` → `sensitivity_diagnostic_data.json`,
`sensitivity_noise_folds.csv`, `sensitivity_state_usage_original.csv`,
`sensitivity_perturbation_original.csv`, `sensitivity_input_scale_original.csv`.

---

## 1. How the feature is computed

```
sensitivity(layer) = acc(baseline, probe) − acc(baseline with only `layer` L1-pruned by 10%, probe)
```

| Item | Value |
|---|---|
| Probe pruning ratio | 10% (L1 unstructured, one layer at a time, fresh copy each time) |
| Images | 1,000: the first 1,000 of the 5,000-image validation split (the RL probe) |
| Units | percentage points; resolution 0.1 pp = one image |
| Baseline probe accuracy | 76.10% |
| Computed | once, before training; the same values for every seed and condition |

## 2. Raw values

| Layer | Sensitivity (pp) |
|---|---|
| features.0 | 0.6 |
| features.3 | 0.2 |
| features.6 | 0.1 |
| classifier.1 | 0.0 |

Min 0.0, max 0.6, mean 0.225, SD 0.263 (sample), range 0.6.

## 3. Noise

**Image-level view (probe, 1,000 images).** Each value is a net count of images
whose prediction changed:

| Layer | Images lost | Images gained | Drop (pp) | Bootstrap 95% CI (pp) | Drop on all 5,000 validation images (pp) |
|---|---|---|---|---|---|
| features.0 | 11 | 5 | 0.6 | [−0.1, 1.4] | 0.04 |
| features.3 | 2 | 0 | 0.2 | [0.0, 0.5] | −0.04 |
| features.6 | 2 | 1 | 0.1 | [−0.2, 0.4] | 0.06 |
| classifier.1 | 0 | 0 | 0.0 | [0.0, 0.0] | −0.02 |

On the full validation split, pruning any single layer by 10% changes accuracy
by at most ±0.06 pp: at this ratio no layer is measurably sensitive.

**Same probe on the five disjoint 1,000-image validation folds** (fold 0 is the RL probe):

| Layer | Fold 0 | Fold 1 | Fold 2 | Fold 3 | Fold 4 |
|---|---|---|---|---|---|
| features.0 | 0.6 | −0.5 | 0.0 | −0.2 | 0.3 |
| features.3 | 0.2 | −0.3 | 0.4 | 0.0 | −0.5 |
| features.6 | 0.1 | −0.2 | 0.4 | −0.1 | 0.1 |
| classifier.1 | 0.0 | 0.0 | 0.2 | −0.1 | −0.2 |

The five folds give **four different layer rankings**; fold 1's is the exact
reverse of the probe's. The ordering the agent received is a property of which
1,000 images were used, not of the layers.

## 4. State vector

Order (dimension 0–6), from `state_to_vector`, and values at the start of an episode:

| Dim | Feature | features.0 | features.3 | features.6 | classifier.1 | Range |
|---|---|---|---|---|---|---|
| 0 | layer index / 3 | 0.000 | 0.333 | 0.667 | 1.000 | 0–1 |
| 1 | log(1 + param count) | 6.763 | 9.822 | 11.208 | 13.170 | 6.8–13.2 |
| 2 | param count / 524,288 | 0.00165 | 0.0352 | 0.1406 | 1.000 | 0.002–1 |
| 3 | mean \|weight\| | 0.1153 | 0.0539 | 0.0512 | 0.0292 | 0.03–0.12 |
| 4 | weight SD | 0.1355 | 0.0690 | 0.0657 | 0.0439 | 0.04–0.14 |
| 5 | cumulative pruning so far | 0 | 0 | 0 | 0 | 0–1.8 (varies with earlier actions) |
| 6 | sensitivity | 0.6 | 0.2 | 0.1 | 0.0 | 0–0.6 |

## 5. Normalisation

None, anywhere.

- The sensitivity values enter the state raw (pp).
- Stable-Baselines3's `preprocess_obs` returns `obs.float()` for a non-image
  `Box` space; this was also checked numerically (identity).
- No `VecNormalize` wrapper is used.
- Policy: SB3 `MlpPolicy` defaults, separate policy/value MLPs [64, 64], tanh,
  orthogonal initialisation.

## 6. Constancy and redundancy

- The sensitivity input is **constant for a given layer**: the same value at
  every step for that layer, in every episode and every seed. Checked by
  stepping an episode.
- Dimensions 0–4 are also constant per layer, and **each of them alone takes
  four distinct values**, so each already identifies the layer uniquely. Only
  dimension 5 (cumulative pruning) varies within the task.
- Consequently, in this fixed four-layer setting, sensitivity carries **no
  information the policy cannot already obtain from the layer's identity**. At
  most it can act as a prior (a starting bias) on which layers to protect.

## 7. Relative scale inside the policy network

Share of the first policy layer's input contribution (mean over hidden units
and layers of |W₁[:,d]·x_d|); the weight columns have similar norms (1.5–1.8),
so shares mostly reflect input magnitudes:

| Input | Trained agents (28) | Untrained, same seeds (4) |
|---|---|---|
| log(1 + param count) | **84.0%** | **89.7%** |
| layer index | 6.0% | 4.5% |
| param count ratio | 3.5% | 2.6% |
| cumulative pruning | 2.7% | 0.0% |
| **sensitivity** | **2.3%** | **2.0%** |
| weight SD | 0.8% | 0.7% |
| mean \|weight\| | 0.7% | 0.6% |

25–27% of first-layer tanh units are saturated (|pre-activation| > 2), driven
by the log-parameter input.

## 8. Does the trained policy ignore sensitivity?

**No.** For the 24 trained agents that saw a non-zero sensitivity input:

| Input (value swapped with another layer's) | Chosen action changes | Mean change in action distribution (TVD) |
|---|---|---|
| layer index | 22.2% | 0.185 |
| param count ratio | 12.5% | 0.114 |
| **sensitivity** | **11.8%** | **0.083** |
| log(1 + param count) | 10.4% | 0.074 |
| cumulative pruning | 8.3% | 0.093 |
| weight SD | 2.8% | 0.007 |
| mean \|weight\| | 2.4% | 0.007 |

Changing **only** the sensitivity input (to 0 or to another layer's value)
changed the chosen action in 42 of 360 cases (11.7%), in 20 of 24 agents, most
often at the features.3 position (28%). The policy does react to the input;
it uses it as one of several redundant cues for which layer it is on. Because
the values themselves are noise (§3), reacting to them cannot add information.

This is a local, descriptive analysis of the trained networks, not causal proof.

## 9. Answers to the four hypotheses

| Hypothesis | Verdict | Evidence |
|---|---|---|
| 1. Signal too noisy | **Yes, the main cause.** | ±0.06 pp on 5,000 images; bootstrap CIs include 0; 4 rankings in 5 folds (§3). |
| 2. Numeric scale too small | **Partly.** | Only ~2% of the first-layer input, dwarfed by log(param count) (§7). But the trained policies still respond to it (§8), so it is not simply ignored. |
| 3. Single probe ratio unreliable | **Yes.** | 10% is below the point where any layer degrades. On the full validation set, degradation starts at 20% for features.0 and only at 60% for features.3 and features.6 (`sensitivity_probe_curves.csv`). |
| 4. Reward dominates the state | **Yes, structurally.** | Sensitivity is a per-layer constant, redundant with five identity inputs (§6). The policy's per-layer choice is learned from the terminal reward either way; sensitivity can only bias the start. Separately, no run's final policy equalled the highest-reward configuration it sampled during training (experiment log), so optimisation, not state information, limits the outcome. |

**Implication for the refined experiment.** A multi-ratio estimator on the
full validation split can fix (1) and (3), and normalisation to [0, 1] partly
addresses (2). None of these removes the redundancy in (4). The refined
experiment therefore tests whether a real, well-scaled signal helps **as a
prior**; the redundancy predicts at most a small effect.

---

## 10. Refined estimator (registered before any refined PPO run)

`evaluation/refined_sensitivity.py probe`, 2026-09-27 01:24 (+06:00). Full
5,000-image validation split, probe ratios 10/20/40/60%, one layer at a time
from a fresh baseline copy.

| Layer | Val acc at 10 / 20 / 40 / 60% | Accuracy def. raw → normalised | Loss def. raw → normalised |
|---|---|---|---|
| features.0 | 77.28 / 76.58 / 68.94 / 42.72 | 0.1415 → 1.000 | 0.7839 → 1.000 |
| features.3 | 77.36 / 77.40 / 77.00 / 75.76 | 0.0057 → 0.051 | 0.0243 → 0.032 |
| features.6 | 77.26 / 77.32 / 77.20 / 76.68 | 0.0027 → 0.030 | 0.0094 → 0.013 |
| classifier.1 | 77.34 / 77.40 / 77.44 / 77.60 | −0.0016 → 0.000 | −0.0011 → 0.000 |

- Accuracy def.: mean over ratios of (acc₀ − acc_r)/acc₀. Loss def.: mean over
  ratios of (loss_r − loss₀)/loss₀. Min-max across the four layers to [0, 1].
- **Stability across the five validation folds:** the loss definition gives the
  same ranking in all five; the accuracy definition gives two (features.3 /
  features.6 / classifier.1 swap in fold 1). The original gave four.
- **Both definitions agree on the ranking** and produce nearly the same
  normalised vector.
- **Min-max is dominated by features.0.** Its 60% probe costs 34.6 pp, so the
  1.56 pp cost of pruning features.3 to 60% becomes 0.05 (accuracy) or 0.03
  (loss) after normalisation. This follows from the pre-registered
  normalisation and was not changed after seeing it.
- Pruning classifier.1 slightly *raises* validation accuracy (77.60% at 60% vs
  77.32%), so its raw values are negative.

---

## 11. Addendum (after the refined runs): what the reward itself prefers

`evaluation/reward_landscape.py` → `reward_landscape.csv` scores all 6⁴ = 1,296
policies with the training reward (validation probe only).
`learned_policy_reward_rank.csv` places every learned policy (28 original +
100 refined runs) in that ranking.

| λ_s | Reward-optimal policy | Most common learned policy (rank of 1,296) | Median rank, learned | Median rank, best episode sampled in training |
|---|---|---|---|---|
| 0.01 | `[0,2,4,5]` | `[0,0,5,5]` (88) | 88 | 1 |
| 0.02 | `[0,2,4,5]` | `[0,0,5,5]` (58) | 58 | 1 |
| 0.04 | `[0,4,5,5]` | `[0,5,5,5]` / `[1,1,5,5]` / `[0,0,5,5]` (18 / 33 / 36) | 34.5 | 1 |

- **The reward does not separate the "harmful" policy from the safe one.** At
  λ_s = 0.02, `[0,0,5,5]` scores 2.7204, `[1,1,5,5]` 2.7208, `[0,5,5,5]` 2.7193
  and `[1,5,5,5]` 2.7215: within 0.0022 of each other, although pruning
  features.3 to 60% costs about 1.5 pp of test accuracy. At λ_s = 0.04 the
  reward *prefers* `[1,5,5,5]` (rank 16) to `[0,0,5,5]` (rank 36).
- **The diversity bonus shapes the optimum.** `[0,2,4,5]` wins at λ_s 0.01–0.02
  partly because four distinct actions earn +0.20, more than a 10 pp accuracy
  change is worth in the accuracy term (+1 pp ≈ +0.019).
- **PPO does not converge to the reward optimum.** In nearly every run the best
  policy *sampled* during training is the optimum (median rank 1), but the
  final deterministic policy settles far from it (median rank 58–88 at
  λ_s 0.01–0.02). The median rank is identical for correct, zeroed and shuffled
  sensitivity.

**Consequence.** Which of several near-tied, sub-optimal policies PPO ends on
is decided by optimisation details: seed, initialisation, and any input that
nudges the network. A state feature cannot steer the agent away from a policy
that the reward scores as equal or better. This is hypothesis 4 made
quantitative, and it bounds what any sensitivity feature can achieve under this
reward.
