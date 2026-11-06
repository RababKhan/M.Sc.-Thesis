# Rock-solid roadmap — master protocol

Frozen at the end of Phase 1 (2026-09-28). It fixes the **rules** every later
phase must follow. It does not state expected results, and it does not
pre-register numerical hypotheses that depend on baselines not yet trained.
Each confirmatory experiment still needs its own pre-registration, committed
before its first run.

## 1. Governance

- **Phase gates.** Work proceeds phase by phase; each phase starts only after
  explicit approval of the previous phase's report.
- **Confirmatory vs exploratory.** Confirmatory experiments are
  pre-registered: hypothesis, design, seeds, endpoints, tests, multiplicity
  and classification rules, committed and hashed before any run, with the
  hash recorded in every run record. Anything else is labelled exploratory
  and cannot support a headline claim.
- **Amendments.** A pre-registration is never edited. A necessary correction
  is a new, separately committed amendment made before any rerun.
- **No outcome-driven iteration.** No new β, threshold, reward, mask, seed
  subset or selection metric after a failed pre-registered test without a
  genuinely new hypothesis.
- **Historical results.** Results marked VERIFIED in
  `rocksolid_experiment_audit.md` are reused as they are; SUPERSEDED results
  are never used in the paper.

## 2. Datasets and splits

| Dataset | Split | Rule |
|---|---|---|
| CIFAR-10 | 45,000 train / 5,000 validation / 10,000 official test | the existing seed-42 permutation of the training set (`rl_env.Splits`, validation index hash `9d3648af…`) |
| CIFAR-100 | 45,000 / 5,000 / 10,000 | the same procedure (`torch.randperm` of the 50,000 training images with generator seed 42; the last 5,000 are validation); index hash recorded at creation |
| V_RL / V_SELECT (both datasets) | 3,000 / 2,000 of validation | stratified by class, deterministic, fixed split seed; indices saved and hashed before any run that uses them |
| CIFAR-10-C / CIFAR-100-C | 19 corruptions × 5 severities | evaluation only (Phase 8) |

Preprocessing is the per-dataset channel mean/std normalisation. Training
augmentation is random crop (padding 4) and horizontal flip. Evaluation data
have no augmentation.

## 3. Data-use rules

- **Train:** weight updates only (dense training, fine-tuning).
- **Validation (V_RL):** sensitivity estimation, PPO reward, training
  diagnostics. In experiments without a V_RL/V_SELECT split, the full
  validation split plays both roles, and this is reported as a limitation.
- **V_SELECT:** post-training choice among already-generated candidates
  (policies, checkpoints, epochs) only. Never queried during training.
- **Test:** read once per frozen model, after the model or policy is written
  to disk and hashed. Test data never enter training, reward, sensitivity,
  selection, early stopping, hyper-parameter choice or classification.
- **Enforcement:** runners use phase guards that raise on out-of-phase
  access (as `rl_env.Splits` and `archive_confirm.ArchiveData` do), and an
  implementation check demonstrates the guards before each experiment.

## 4. Architectures

| Architecture | Definition | Status |
|---|---|---|
| SimpleCNN | `evaluation/common.py` (3 conv blocks + 2 linear) | existing; `cnn_baseline_FIXED.pth` is the C10 reference |
| ResNet-18 (CIFAR) | torchvision ResNet-18 with a 3×3 stride-1 stem conv and no max-pool | Phase 2 |
| MobileNetV2 (CIFAR) | torchvision MobileNetV2 with first-conv stride 1 | Phase 2 |

- **Dense training recipe:** fixed per architecture in the Phase-2
  pre-registration before any training (optimiser, schedule, epochs, weight
  decay, batch size). The checkpoint is chosen on validation only.
- **Reference model:** training seed 0 of each (architecture, dataset). All
  pruning studies on that setting use it. Its hash is recorded in every run.
- **Retraining a reference model** invalidates every result tied to it. It
  is flagged, never silent.

## 5. Methods

- **Prunable-layer set:** all Conv2d/Linear layers except the final
  classifier, unless a pre-registration states otherwise. It is identical
  for every method in a comparison.
- **Unstructured one-shot baselines:** uniform per-layer, global magnitude
  over the prunable set, LAMP, ERK-style layer allocation, random (10 seeds).
  Each is calibrated to the exact total sparsity of the method it is
  compared with.
- **PPO:** the existing formulation (per-layer discrete actions
  {0, 10, 20, 30, 40, 60%}, L1 unstructured, reward
  1.5·acc/base + λ_s·S(%) + 0.05·distinct actions). An architecture-agnostic
  implementation must reproduce the historical SimpleCNN runs exactly before
  it is used.
- **Long networks:** any grouping of layers into decision steps (e.g. per
  MobileNetV2 block) is fixed in the pre-registration.
- **Budget:** 512 episodes per PPO run.
- **Sensitivity conditions** (when a sensitivity claim is tested): no prior,
  correct prior, shuffled prior (per-seed derangement listed in advance),
  constant prior (mean of S). All share one PPO implementation; a zero prior
  must reproduce plain PPO exactly.
- **Iterative magnitude pruning + fine-tuning:** a strong training-based
  baseline under the fine-tuning fairness rule (§8).
- **Structured pruning (Phase 5):** channel/filter pruning with explicit
  dependency handling; structured baselines are uniform channel, global L1
  channel, and network-slimming style.

## 6. Seed policy

- The unit of analysis is the seed: a PPO seed, a training seed or a random-
  baseline seed. Episodes, timesteps and test images are never the N of an
  across-run inference.
- **Registry of used PPO seeds:** 1–19, 42, 100–129, 200–239. Every new
  confirmatory experiment takes a fresh consecutive block (next free block:
  300–), checked against all run records before its pre-registration.
- **Minimum counts:**
  - dense training: 3 seeds per (architecture, dataset), variance reporting
    only; the reference is seed 0;
  - PPO: 20 seeds per condition per setting;
  - random baselines: 10 seeds;
  - fine-tuning of deterministic baselines: 5 seeds.

  Counts are never reduced silently. If compute forces a reduction, it is
  decided before the pre-registration and stated there.
- **No seed** is dropped, replaced or added after results are visible.
  Failed runs:
  - infrastructure failures are retried with the same seed and
    configuration, and documented;
  - genuine numerical or model failures are retained.

## 7. Matched sparsity

- Every accuracy comparison between methods is made at matched total
  sparsity. At minimum, each method is compared with global magnitude
  pruning calibrated to that method's exact achieved sparsity (metric:
  "excess over matched GM").
- Paired PPO-vs-PPO comparisons report per-pair sparsity differences, their
  mean, the maximum, and the share of pairs within 0.5 pp.
- A comparison supports a superiority claim only if the mean |Δsparsity| is
  ≤ 0.25 pp and ≥ 90% of pairs are within 0.5 pp, or if the metric itself is
  sparsity-adjusted.
- An advantage that disappears at matched sparsity is reported as "explained
  by pruning less".

## 8. Fine-tuning fairness

- Every compared model is fine-tuned with the same optimiser, learning rate,
  epochs, batch size, augmentation, training data and epoch-selection rule
  (validation, or V_SELECT where defined). All start from the same dense
  reference.
- Pruning masks stay fixed during fine-tuning (sparsity asserted unchanged
  afterwards).
- A dense model fine-tuned with the same budget is always included as a
  control.
- The test set is read once, after the fine-tuned model is selected and
  frozen.
- Fine-tuning results never change the classification of a
  no-fine-tuning experiment.

## 9. Statistical standards

- **Tests:** paired by seed; two-sided exact sign-flip permutation test on
  the mean paired difference over all 2^N sign patterns (meet-in-the-middle
  implementation, validated against brute force).
- **Reported alongside:** mean, median and SD of the paired differences; a
  95% t-interval; a bootstrap percentile interval; Cohen's d_z; wins / ties
  / losses; exact Wilcoxon p.
- **Multiplicity:** Holm within every pre-specified family (primary family;
  each secondary endpoint family).
- **McNemar per model pair:** supportive only, summarised as counts.
- **Success:** requires the pre-registered direction, corrected significance
  and a confidence interval excluding zero. Effect sizes are always reported;
  non-significant differences are never described as wins.
- **Classification rules** (Level A/B/C-style) are written before the runs
  and computed mechanically by the analysis script.
- **No interim peeking:** no aggregate outcome statistic before all runs of
  a pre-registered experiment finish; progress reports show only run counts,
  runtime, errors and hardware state.

## 10. Efficiency measurement

- **Latency:**
  - run on an otherwise idle machine, checked automatically (CPU load before
    and after);
  - fixed thread count; `eval()` + `inference_mode()`;
  - ≥ 100 warm-up passes, ≥ 600 timed passes, model order interleaved in
    rounds;
  - batches 1 and 128;
  - report median, mean, SD, bootstrap 95% CI of the median and of the
    difference from dense;
  - on GPU, CUDA-synchronise around each timed pass;
  - record the exact CPU/GPU model, driver and library versions.
- **FLOPs, parameters and memory:** measured with one pinned counter.
  Unstructured sparsity is reported as nominal only; no FLOP or latency
  reduction is claimed without measurement on real kernels.
- **Storage:** like-for-like (raw vs raw, compressed vs compressed); sparse
  formats reported as theoretical unless actually implemented.

## 11. Reproducibility

- **Environment:** the CPU environment is pinned in `requirements_locked.txt`
  (Python 3.11.9, torch 2.7.1+cpu, SB3 2.8.0, sb3-contrib 2.8.0,
  gymnasium 1.2.3, numpy 2.2.2, pandas 2.2.3). A GPU environment gets its own
  lock file (CUDA/cuDNN versions included) before first use.
- **Change of hardware or library:** before any new run, a verified
  historical result must reproduce exactly on the new platform (e.g. SimpleCNN
  PPO seed 42), or the platform difference is quantified and reported.
- **Per-run records** hold the sha256 of the baseline checkpoint, data split,
  pre-registration and every script used; the git commit; the environment;
  the hardware; start time and runtime.
- **Determinism:** fixed seeds for Python/numpy/torch; deterministic
  algorithms where available (`cudnn.benchmark = False`). The global torch
  random stream must not be perturbed by caching (the DataLoader
  random-stream issue found in the constrained experiment).
- **Resumability:** every runner is resumable at run granularity, writes its
  record last, and checks file integrity after an interruption. Needed given
  the unstable power.
- **Commits:** all results are committed after each phase. Verified
  historical scripts stay frozen; new work goes into a consolidated library
  that must reproduce historical results before use.

## 12. New dependencies (to be added when their phase starts; not installed)

| Dependency | Needed for | Phase |
|---|---|---|
| CUDA build of torch/torchvision matching 2.7.1 / 0.22.1 | GPU training and evaluation | 2 |
| CIFAR-100 (torchvision download) | datasets | 2 |
| a FLOP/parameter counter (e.g. fvcore or thop; one, pinned) | efficiency | 2 (baseline cards), 6 |
| a dependency-aware channel-pruning tool (e.g. torch-pruning) or equivalent custom code | structured pruning | 5 |
| CIFAR-10-C / CIFAR-100-C | robustness | 8 |
| scipy (optional) | independent cross-check of statistics | any |
