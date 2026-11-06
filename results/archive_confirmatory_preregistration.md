# Archive-confirmatory experiment — pre-registration

**Sensitivity-Guided PPO with Best-Policy Archive** (experiment family
`archive_confirmatory`). Written 2026-09-28 before any implementation check or
PPO run of this experiment; committed and hashed before training. Not edited
after training starts; any necessary correction would be a separately
committed amendment made before any rerun.

## Hypothesis

The sensitivity-guided PPO agent explores better pruning policies, and a
predefined best-policy archive converts this exploration advantage into higher
**final** model accuracy at matched sparsity. All earlier experiments
(including the Level-A exploration confirmation) are unchanged.

## Fixed inputs (created before this file; hashes locked here)

| Item | Value |
|---|---|
| Baseline | `checkpoints/cnn_baseline_FIXED.pth`, sha256 `ca28f8345c23365ee7b92686e780ae0a9befa76c9d6bb9b3bb0b9e42272e111c` (verified); test 77.03%; not retrained |
| Split | the existing seed-42 CIFAR-10 split: 45,000 train / 5,000 validation / 10,000 test |
| Validation partition | `results/archive_validation_split.npz`, sha256 `55b7ff40a25b9b24f6c89fae1ca41f309a66f8b834160c32c07d732920f0b511`. Stratified by class, deterministic (split seed 20260928): for each class, a `numpy.random.default_rng(20260928)` permutation of its validation positions, the first ⌊0.6·n_c⌋ (plus largest-remainder rounding to total 3,000) → **V_RL (3,000)**, the rest → **V_SELECT (2,000)**. Class counts: validation [488, 512, 532, 471, 471, 514, 507, 500, 504, 501]; V_RL [293, 307, 319, 283, 283, 308, 304, 300, 302, 301]; V_SELECT [195, 205, 213, 188, 188, 206, 203, 200, 202, 200] |
| Baseline accuracy | V_RL 76.9667% (reward denominator); V_SELECT 77.85% |
| Sensitivity | `results/archive_sensitivity_vector.json`, sha256 `de30b20ccd865db3bf69c7b0fdb267fdad58b76809e227afd2a4749c31d280e6`. The refined multi-level loss algorithm unchanged (each layer alone at 10/20/40/60% from a fresh baseline copy; mean relative cross-entropy increase; min-max to [0, 1]) computed on **V_RL only**. Raw [0.768939, 0.023166, 0.006029, −0.001171]; **normalised S = [1.0, 0.03160260491301242, 0.009350119670520027, 0.0]**, layers features.0, features.3, features.6, classifier.1 |

## PPO, prior, reward (identical across conditions except the prior)

- **Prior:** adjusted_logit(ℓ, a) = raw_logit(ℓ, a) − β · S_ℓ · a_norm(a),
  a_norm = ratio/60. Applied before the categorical distribution is formed
  (`SoftPriorPolicy`, `evaluation/soft_prior.py`, unmodified). **β = 0.50.**
  All six actions always available; no masking, clipping, replacement,
  remapping or reward penalty.
- **Reward:** R = 1.5 · acc_{V_RL} / 76.9667 + 0.01 · S(%) + 0.05 · distinct
  actions. The coefficients are unchanged; accuracy and its baseline are
  measured on V_RL. This is a consequence of the partition: the earlier probe
  overlaps V_SELECT.
- **PPO:** stable-baselines3 2.8.0 PPO, MlpPolicy network [64, 64] tanh,
  orthogonal init, Adam lr 3e-4, n_steps 64, batch 32, n_epochs 10, γ 0.99,
  GAE λ 0.95, clip 0.2, ent_coef 0.01, vf_coef 0.5, max_grad_norm 0.5;
  2,000 timesteps requested → 2,048 trained = 512 episodes; `seed` = run seed;
  state sensitivity dimension = 0; unstructured L1 magnitude pruning;
  memoised environment of `evaluation/constrained_ppo.py` (unmodified), with
  V_RL as its evaluation data.
- **Execution:** 4 worker processes × 1 torch thread, Intel i5-6400.

## Seeds (locked)

**200, 201, …, 239 (N = 40).** Verified before this file against every
earlier run record, the experiment log, all scripts and all earlier
pre-registrations: none was used. No seed is removed, replaced or added.

## Conditions (exactly four; archive enabled in all)

| | β | Prior vector |
|---|---|---|
| A0 no prior | 0 | — |
| A1 correct prior (proposed) | 0.50 | S |
| A2 shuffled prior | 0.50 | S permuted among layers, per seed, below |
| A3 constant prior | 0.50 | every layer = mean(S) = 0.26023818114588315 |

A2 permutations, fixed now (`soft_prior.shuffled`: a `default_rng(seed)`
permutation in which every layer receives another layer's, different, value):

| Seed | Prior vector (f.0, f.3, f.6, c.1) | Mapping (layer<-source) |
|---|---|---|
| 200 | 0.009350, 1.000000, 0.000000, 0.031603 | f.0<-f.6, f.3<-f.0, f.6<-c.1, c.1<-f.3 |
| 201 | 0.009350, 0.000000, 1.000000, 0.031603 | f.0<-f.6, f.3<-c.1, f.6<-f.0, c.1<-f.3 |
| 202 | 0.031603, 0.009350, 0.000000, 1.000000 | f.0<-f.3, f.3<-f.6, f.6<-c.1, c.1<-f.0 |
| 203 | 0.031603, 0.000000, 1.000000, 0.009350 | f.0<-f.3, f.3<-c.1, f.6<-f.0, c.1<-f.6 |
| 204 | 0.009350, 0.000000, 0.031603, 1.000000 | f.0<-f.6, f.3<-c.1, f.6<-f.3, c.1<-f.0 |
| 205 | 0.000000, 0.009350, 1.000000, 0.031603 | f.0<-c.1, f.3<-f.6, f.6<-f.0, c.1<-f.3 |
| 206 | 0.009350, 1.000000, 0.000000, 0.031603 | f.0<-f.6, f.3<-f.0, f.6<-c.1, c.1<-f.3 |
| 207 | 0.031603, 0.009350, 0.000000, 1.000000 | f.0<-f.3, f.3<-f.6, f.6<-c.1, c.1<-f.0 |
| 208 | 0.009350, 0.000000, 0.031603, 1.000000 | f.0<-f.6, f.3<-c.1, f.6<-f.3, c.1<-f.0 |
| 209 | 0.009350, 0.000000, 0.031603, 1.000000 | f.0<-f.6, f.3<-c.1, f.6<-f.3, c.1<-f.0 |
| 210 | 0.000000, 1.000000, 0.031603, 0.009350 | f.0<-c.1, f.3<-f.0, f.6<-f.3, c.1<-f.6 |
| 211 | 0.000000, 0.009350, 1.000000, 0.031603 | f.0<-c.1, f.3<-f.6, f.6<-f.0, c.1<-f.3 |
| 212 | 0.031603, 0.009350, 0.000000, 1.000000 | f.0<-f.3, f.3<-f.6, f.6<-c.1, c.1<-f.0 |
| 213 | 0.000000, 0.009350, 0.031603, 1.000000 | f.0<-c.1, f.3<-f.6, f.6<-f.3, c.1<-f.0 |
| 214 | 0.000000, 1.000000, 0.031603, 0.009350 | f.0<-c.1, f.3<-f.0, f.6<-f.3, c.1<-f.6 |
| 215 | 0.031603, 0.000000, 1.000000, 0.009350 | f.0<-f.3, f.3<-c.1, f.6<-f.0, c.1<-f.6 |
| 216 | 0.009350, 1.000000, 0.000000, 0.031603 | f.0<-f.6, f.3<-f.0, f.6<-c.1, c.1<-f.3 |
| 217 | 0.000000, 0.009350, 1.000000, 0.031603 | f.0<-c.1, f.3<-f.6, f.6<-f.0, c.1<-f.3 |
| 218 | 0.031603, 1.000000, 0.000000, 0.009350 | f.0<-f.3, f.3<-f.0, f.6<-c.1, c.1<-f.6 |
| 219 | 0.031603, 1.000000, 0.000000, 0.009350 | f.0<-f.3, f.3<-f.0, f.6<-c.1, c.1<-f.6 |
| 220 | 0.031603, 1.000000, 0.000000, 0.009350 | f.0<-f.3, f.3<-f.0, f.6<-c.1, c.1<-f.6 |
| 221 | 0.009350, 1.000000, 0.000000, 0.031603 | f.0<-f.6, f.3<-f.0, f.6<-c.1, c.1<-f.3 |
| 222 | 0.000000, 0.009350, 1.000000, 0.031603 | f.0<-c.1, f.3<-f.6, f.6<-f.0, c.1<-f.3 |
| 223 | 0.000000, 0.009350, 0.031603, 1.000000 | f.0<-c.1, f.3<-f.6, f.6<-f.3, c.1<-f.0 |
| 224 | 0.000000, 1.000000, 0.031603, 0.009350 | f.0<-c.1, f.3<-f.0, f.6<-f.3, c.1<-f.6 |
| 225 | 0.009350, 0.000000, 1.000000, 0.031603 | f.0<-f.6, f.3<-c.1, f.6<-f.0, c.1<-f.3 |
| 226 | 0.031603, 0.009350, 0.000000, 1.000000 | f.0<-f.3, f.3<-f.6, f.6<-c.1, c.1<-f.0 |
| 227 | 0.031603, 0.009350, 0.000000, 1.000000 | f.0<-f.3, f.3<-f.6, f.6<-c.1, c.1<-f.0 |
| 228 | 0.009350, 0.000000, 1.000000, 0.031603 | f.0<-f.6, f.3<-c.1, f.6<-f.0, c.1<-f.3 |
| 229 | 0.000000, 0.009350, 0.031603, 1.000000 | f.0<-c.1, f.3<-f.6, f.6<-f.3, c.1<-f.0 |
| 230 | 0.000000, 0.009350, 0.031603, 1.000000 | f.0<-c.1, f.3<-f.6, f.6<-f.3, c.1<-f.0 |
| 231 | 0.031603, 0.009350, 0.000000, 1.000000 | f.0<-f.3, f.3<-f.6, f.6<-c.1, c.1<-f.0 |
| 232 | 0.000000, 0.009350, 1.000000, 0.031603 | f.0<-c.1, f.3<-f.6, f.6<-f.0, c.1<-f.3 |
| 233 | 0.009350, 0.000000, 1.000000, 0.031603 | f.0<-f.6, f.3<-c.1, f.6<-f.0, c.1<-f.3 |
| 234 | 0.031603, 1.000000, 0.000000, 0.009350 | f.0<-f.3, f.3<-f.0, f.6<-c.1, c.1<-f.6 |
| 235 | 0.031603, 0.009350, 0.000000, 1.000000 | f.0<-f.3, f.3<-f.6, f.6<-c.1, c.1<-f.0 |
| 236 | 0.031603, 0.009350, 0.000000, 1.000000 | f.0<-f.3, f.3<-f.6, f.6<-c.1, c.1<-f.0 |
| 237 | 0.000000, 0.009350, 0.031603, 1.000000 | f.0<-c.1, f.3<-f.6, f.6<-f.3, c.1<-f.0 |
| 238 | 0.031603, 1.000000, 0.000000, 0.009350 | f.0<-f.3, f.3<-f.0, f.6<-c.1, c.1<-f.6 |
| 239 | 0.000000, 1.000000, 0.031603, 0.009350 | f.0<-c.1, f.3<-f.0, f.6<-f.3, c.1<-f.6 |

(Rounded to six decimals here; exact values are used.)

## Archive and selection (the one rule)

- **Archive:** every unique four-layer policy completed during training, with
  action indices, pruning percentages, exact sparsity, first episode and
  timestep seen, times sampled, and PPO reward. It is frozen at the end of
  training. V_SELECT is never queried during training and does not influence
  PPO, the reward, sensitivity, β, archive membership or stopping.
- **After training:** each unique archived policy is applied to a fresh
  baseline copy and evaluated on V_SELECT (accuracy, mean cross-entropy loss,
  exact sparsity), with no fine-tuning. Evaluations are cached by policy;
  archive membership stays per run.
- **Target band:** 57.5% ≤ sparsity ≤ 58.5% (target 58.0%), fixed.
- **Rule:** if the archive has at least one policy in the band, select the
  highest V_SELECT accuracy. Ties go to sparsity closest to 58.0%, then lower
  V_SELECT loss, then the lexicographically smaller action vector. If there is
  none, select the archived policy closest to 58.0%, then apply the same
  tie-breakers in the same order, and set `target_band_reached = FALSE`; such
  runs remain in all descriptive reporting. PPO reward and test accuracy are
  never used for selection.
- **Freeze, then test:** the selected policy, and for the conversion analysis
  the PPO terminal policy, are written to `seed_<seed>_selected.json` and
  hashed before the test loader can be opened. The test loader raises during
  training and until that file exists. Each frozen policy is applied to a
  fresh baseline copy and evaluated on the 10,000 test images once;
  per-example predictions are saved. The terminal policy is fixed by PPO
  before selection, so its test evaluation cannot affect selection.

## A-priori structural property (from the existing landscape; no accuracies used)

Only 14 of the 1,296 policies lie in the band, all with features.6 = 60% and
classifier.1 = 60%; they differ only in features.0 (0–60%) and features.3
(0–20%). In the earlier exploration study (seeds 100–129) each run sampled
9–10 of the 14 on average. The four conditions may therefore often select the
same policy, which limits the attainable differences. Recorded now, not
acted on.

## Primary endpoint and analysis

**Primary endpoint:** test accuracy of the archive-selected model.

**Primary comparisons:** A1 − A0, A1 − A2, A1 − A3, paired by seed.

**Matched set:** for each comparison, the seeds where both conditions reached
the band. The primary statistics use the matched set; the same statistics on
all 40 seeds are reported as a sensitivity analysis.

For each comparison, reported:

- N matched seeds, each condition's mean and SD;
- mean, median and SD of paired differences;
- **95% t CI** (df = N − 1);
- **two-sided exact sign-flip test** over all 2^N sign patterns (exact
  meet-in-the-middle, as in the previous study);
- d_z = mean/SD;
- A1 wins / ties / losses;
- exact Wilcoxon p (descriptive).

**Holm** over the three primary comparisons; threshold Holm-adjusted p < 0.05.

**Matched-sparsity validity** (per comparison, on the matched set): mean
|paired sparsity difference| ≤ 0.25 pp **and** ≥ 90% of pairs within
0.50 pp. Also reported: paired sparsity differences, their mean, and the
maximum absolute difference.

**McNemar** (supportive, per seed, selected models): counts of seeds
significantly favouring A1, significantly favouring the control, and
non-significant (p ≥ 0.05). Never substituted for the seed-level analysis.

## Classification (precedence: A-FA, then B-FA, then C-FA)

Criteria:

- **C1:** A1 − A0 mean > 0, Holm p < 0.05, 95% t CI excludes 0.
- **C2:** the same for A1 − A2.
- **C3:** the same for A1 − A3.
- **C4:** the matched-sparsity validity requirement holds for all three comparisons.
- **C5:** each comparison has ≥ 30 matched seeds.

**"Explained by pruning less"** for a comparison: mean paired sparsity
difference (A1 − control) < −0.25 pp on its matched set.

- **LEVEL A-FA:** C1–C5 all hold.
- **LEVEL B-FA:** not A-FA, and at least one of the following, none of whose
  comparisons is explained by pruning less:
  - (B1) C1 holds;
  - (B2) all three primary mean differences > 0 and A1 wins more seeds than it
    loses in each;
  - (B3) C1–C3 hold but C4 or C5 fails;
  - (B4) a V_SELECT-only benefit: A1 − A0 selected-policy V_SELECT accuracy
    has sign-flip p < 0.05 and mean > 0 on the matched set while C1 fails, but
    the test difference A1 − A0 is still positive (mean > 0, wins > losses).
- **LEVEL C-FA:** otherwise. This includes: A1 not outperforming A0; A1 similar
  to shuffled/constant; an advantage explained by pruning less; and a
  V_SELECT improvement whose test difference is not positive (failure to
  generalise).

This resolves the specification's two mentions of a V_SELECT-only benefit
(under B-FA and C-FA): it is B-FA only if the test effect is at least
directionally positive, otherwise C-FA. After C-FA: no further sensitivity
redesign without a new hypothesis. Fine-tuning is not part of this
experiment and cannot change the classification.

## Secondary analyses (never substituted for the primary)

- **Archive coverage:** unique policies; % of runs with a band candidate;
  band candidates per seed; best band V_SELECT accuracy; first episode of the
  eventually selected policy.
- **Conversion:** terminal vs archive-selected test accuracy and sparsity;
  archive gain = selected − terminal test accuracy; A1 vs each control on
  gain (paired sign-flip).
- **Discovery time:** first episode, first timestep and times sampled of the
  selected policy; A1 vs each control (paired sign-flip).
- **Archive quality:** unique policies; band policies; max, mean and median
  band V_SELECT accuracy; top-5 mean; share of band policies with V_SELECT
  accuracy ≥ 76.85% (baseline V_SELECT accuracy − 1.0 pp).
- **Global magnitude** at each A1 selected model's exact sparsity (and for
  every condition): difference and McNemar; aggregated. Uniform per-layer at
  matched overall sparsity where feasible.
- **Figure data A–E** as specified. Figure D (cumulative best band V_SELECT
  accuracy available in the archive vs episode) is computed retrospectively
  from first-seen episodes and post-training V_SELECT scores.

## Conduct

- **Stopping:** all 40 × 4 = 160 runs complete before any aggregate
  statistic. Progress output shows only run counts, runtime, crashes and
  hardware state.
- **Failures:** infrastructure failures are retried with the same condition
  and seed, and documented. Numerical/model failures are retained, never
  replaced. No run is removed.
- **Implementation checks:** the 15 specified checks run before training, on
  seed 42 (non-experimental) and synthetic inputs; all must pass.

## Planned outputs (`results/`)

`archive_confirmatory_preregistration.md`, `archive_validation_split.npz`,
`archive_sensitivity_vector.json`, `archive_implementation_checks.md`,
`archive_all_runs.csv`, `archive_unique_policies.csv`,
`archive_selection_scores.csv`, `archive_selected_models.csv`,
`archive_matched_sparsity.csv`, `archive_statistics.csv`,
`archive_mcnemar_summary.csv`, `archive_conversion_analysis.csv`,
`archive_discovery_time.csv`, `archive_quality.csv`,
`archive_global_magnitude.csv`, `archive_summary.csv`,
`archive_classification.json`, `archive_runs/<condition>/…`,
`archive_predictions/<condition>/seed_<seed>.csv`, `archive_fig_A…E_*.csv`,
and a section in `experiment_log.md`.
