# Rock-solid roadmap — Phase 1 audit

Audit of the repository on 2026-09-28, branch `fix/test-set-leakage` at
`37bd686`. Read-only: no model was trained and no historical result was
modified. Facts below come from the code, the data and the run records
(`evaluation/audit_phase1.py` → `results/rocksolid_audit_data.json`), not from
memory.

Status labels:

- **VERIFIED** — reuse directly.
- **SUPERSEDED** — keep for provenance, never use in the paper.
- **MISSING** — must be implemented.
- **NEEDS VERIFICATION** — a code/result or provenance inconsistency exists.

---

## 1. Repository and provenance

| Item | State |
|---|---|
| Branch / HEAD | `fix/test-set-leakage` at `37bd686`; identical to `origin/fix/test-set-leakage` |
| `origin/main` | `b9bfbee` = merge of PR #1 at `c220caf` (2026-09-26). The 11 later commits (three pre-registrations and every experiment since the leakage fix) are **only on the branch** |
| Uncommitted | `checkpoints/baseline_clean.pth` (modified working copy; see §4). Nothing else |
| Tracked files | 3,198 (includes ~830 PPO agent files, per-run records, training curves) |
| Pre-registrations | `soft_prior` (committed `4a7a5a5` before runs), `exploration_levelA` (`c3a9484`, before runs), `archive_confirmatory` (`1bda844`, before runs, with its split and sensitivity files). `constrained_preregistration.md` was **written and hashed before its runs (2026-09-27 15:27, hash recorded in all 180 run records) but committed afterwards** (`d6fafad`). None has been modified since being added |

## 2. Code inventory

| Component | Location | Notes |
|---|---|---|
| SimpleCNN | `evaluation/common.py`; also redefined in both notebooks | identical architecture; three copies |
| Baseline training | `notebooks/Main code.ipynb` (10 epochs, Adam 1e-3, batch 128, seed 42, **no** validation-based selection; last epoch kept) | only notebook, no script |
| Data split | `rl_env.Splits` (seed-42 permutation; 45k / 5k / 1k probe / 10k test; test locked during training); notebooks build the same split | verified identical (§3) |
| V_RL / V_SELECT | `evaluation/archive_confirm.py` (`ArchiveData`) + `results/archive_validation_split.npz` | used only by the archive experiment |
| PPO environment | `evaluation/rl_env.py` (`CompressionEnvGym`, cell-by-cell copy of Main's); `constrained_ppo.MaskedCompressionEnv` (+ action masks, memoised evaluation with verified random-stream preservation); notebook copies | 4 variants: notebook ×2, rl_env, constrained |
| Pruning utilities | `rl_env.apply_layer_pruning_by_name_inplace`, `common.prune_layerwise`, `common.prune_global_to_total`, `common.prune_uniform_to_total` (all `torch.nn.utils.prune`, L1 unstructured) | |
| Action space / layers | `common.ACTION_TO_PRUNE` {0, 10, 20, 30, 40, 60%}; `FILTERED_LAYERS` = features.0, features.3, features.6, classifier.1 (classifier.3 excluded) | hard-coded to SimpleCNN |
| Reward | `rl_env.reward_components`: 1.5·acc/base + λ_s·S(%) + 0.05·distinct actions; Final-fine-tuned notebook uses a different threshold reward | two reward definitions in the history |
| Sensitivity | (a) Main: 10% probe on 1,000 images (`rl_env.compute_layer_sensitivities`); (b) refined: 10/20/40/60% on 5,000 (`refined_sensitivity.py`); (c) action-level 10–60% + folds (`constrained_masks.py`); (d) V_RL-only refined loss (`archive_confirm.py prepare`) | four estimators, each tied to its experiment |
| Soft prior | `soft_prior.SoftPriorPolicy` (logit prior before the categorical; verified β = 0 ≡ MlpPolicy) | |
| Hard masks | `constrained_ppo.py` (sb3-contrib MaskablePPO) | |
| Archive | `archive_confirm.py` (unique-policy archive, V_SELECT selection, freeze-before-test) | |
| Uniform per-layer baseline | `common.prune_uniform_to_total`; fixed levels in Main | |
| Global magnitude | `common.prune_global_to_total` (4 layers, calibrated to exact sparsity); all-layer sweep in Main | two variants, labelled separately |
| Stronger baselines | matched GM + fine-tune (single run), sensitivity-only heuristic C5 | LAMP, ERK, random, SNIP/GraSP, iterative magnitude pruning: **MISSING** |
| Fine-tuning | `notebooks/Final fine tuned.ipynb`; `finetune_matched_baselines.py` | 3 epochs, Adam 1e-4, validation epoch selection |
| Structured pruning | none | **MISSING** |
| Latency | `latency_benchmark.py` (controlled, CPU); `storage_and_latency.py` (earlier, lighter) | CPU only; no GPU |
| Storage | `storage_and_latency.py` | |
| FLOPs / memory | none | **MISSING** |
| Robustness / calibration | none | **MISSING** |
| Statistics | exact sign-flip, Wilcoxon, Holm, t-CIs, McNemar in `refined_aggregate.py`, `soft_prior_aggregate.py`, `exploration_confirm.py`, `archive_aggregate.py`, `common.mcnemar` | duplicated in 4 files; the `exploration_confirm` versions (meet-in-the-middle, DP) are the validated general ones |

**Duplicated implementations:** SimpleCNN, `evaluate_model`, the reward and
`calculate_sparsity` each exist in `evaluation/` and both notebooks. There are
four PPO environment variants, four sensitivity estimators, and statistics
helpers and t-tables in four analysis scripts. These are not errors: each
experiment's hashed scripts are frozen for provenance. But new work should
use one consolidated library (§ blockers).

**Outdated code:** `storage_and_latency.py` (latency part superseded by
`latency_benchmark.py`). `matched_sparsity_analysis.py` parses PPO results
from the executed Main notebook's stored outputs. The notebooks duplicate the
scripted harness.

## 3. Data protocol verification

| Check | Result |
|---|---|
| Train / validation / test sizes | 45,000 / 5,000 / 10,000 |
| Train ∩ validation | ∅ (union 50,000) |
| Validation index hash | `9d3648af…` (identical in every run record since the harness) |
| RL probe | first 1,000 validation images |
| V_RL / V_SELECT | 3,000 / 2,000, disjoint, union = validation; stratified; labels verified; class counts in `archive_confirmatory_preregistration.md` |
| Old probe vs V_SELECT | the old 1,000-image probe overlaps V_SELECT in 379 images (V_RL in 621). The archive experiment never used the probe |

**Leakage audit — no violation found:**

- **PPO training:** every runner trains inside a guard that raises if the
  test set is requested (`rl_env.Splits.training`, `ArchiveData.training`);
  the notebooks build training environments without a test loader.
- **Sensitivity:** computed on the probe, the full validation split or V_RL
  only.
- **Reward:** probe (validation) or V_RL.
- **Policy selection:** final deterministic policy (no selection) or V_SELECT
  (archive).
- **Early stopping:** the Main baseline has none (fixed 10 epochs);
  fine-tuning selects epochs on validation.
- **Test after freezing:** in every script the test set is read only after
  `agent.learn` returns, on a fresh baseline copy pruned by the fixed policy.
  The archive runner writes and hashes the selection first and verifies
  this.
- **Baseline check:** `ppo_experiments.build_context` evaluates the frozen
  baseline on test once at start-up (assert 77.03%). It selects nothing.
- **Analysis scripts:** they evaluate frozen policies and matched baselines
  on test for reporting only.

**Validation reuse (a limitation, not test leakage):** in the refined,
constrained, soft-prior and exploration studies, the same validation images
supply the reward (probe) and the evaluation (full validation). The
constrained experiment also derived its masks from the full validation
split, and its validation advantage did not reproduce on test. Only the
archive experiment separates the data used for training from the data used
for selection.

## 4. Baselines and checkpoints

| Checkpoint | sha256 (prefix) | Bytes | Val % | Test % | Status |
|---|---|---|---|---|---|
| `cnn_baseline_FIXED.pth` | `ca28f834` | 2,486,069 | 77.32 | **77.03** | **VERIFIED — the only baseline of every current result** (hash recorded in every run since the harness) |
| `baseline_finetuned.pth` | `7c5b700a` | 2,486,069 | 80.02 | 79.91 | VERIFIED — FIXED + 3-epoch fine-tune (Final-fine-tuned re-run) |
| `baseline_clean.pth` (committed) | — | — | 82.74 | 79.57 | SUPERSEDED — April leaky baseline; its validation score is inflated because it was trained on all 50,000 training images |
| `baseline_clean.pth` (working copy, uncommitted) | `a0b89145` | 2,486,005 | 80.12 | 79.56 | SUPERSEDED — abandoned 15-epoch run of 2026-08-25; restore to the committed version or remove; exclude everywhere |
| `cnn_baseline.pth` | `139c658c` | 2,485,973 | 8.64 | **9.78** | SUPERSEDED / **unusable**: chance-level accuracy, although it loads into SimpleCNN; exclude |
| `ppo_final_2000.zip` | `e528b0db` | 149,605 | — | — | VERIFIED — Main PPO seed 42 |
| `ppo_compression_agent.zip`, `ppo_filtered_2000.zip`, `ppo_finetune_agent.zip`, `ppo_finetune_agent_fair.zip`, `manual_and_rl_df_current.csv` | — | — | — | — | SUPERSEDED — April leaky pipeline |
| `checkpoints/{experiments, refined_experiments, constrained_experiments, soft_prior_experiments, exploration_levelA_agents, archive_agents}/` | — | 28 / 100 / 180 / 240 / 120 / 160 files, 124 MB | — | — | VERIFIED — per-run agents |

SimpleCNN: 620,362 parameters (619,872 prunable weights in Conv2d/Linear;
84.6% in classifier.1). Trained 10 epochs (Adam 1e-3, batch 128, seed 42,
last epoch kept).

## 5. PPO configuration (read from an instantiated, untrained agent)

| Item | Value |
|---|---|
| Library | stable-baselines3 2.8.0; sb3-contrib 2.8.0 (MaskablePPO, constrained only) |
| Policy | MlpPolicy: separate policy / value MLPs 7 → 64 → 64, tanh, orthogonal init; action head 64 → 6; value head 64 → 1; Adam |
| Hyper-parameters | lr 3e-4, n_steps 64, batch 32, n_epochs 10, γ 0.99, GAE λ 0.95, clip 0.2 (no value clipping), ent_coef 0.01, vf_coef 0.5, max_grad_norm 0.5, advantage normalisation on, no target KL, no SDE |
| Budget | 2,000 timesteps requested → 2,048 trained (32 rollouts × 64) = 512 four-step episodes |
| State (7) | layer index/3, log(1+params), params/524,288, mean \|w\|, SD(w), cumulative pruning, sensitivity |
| Actions | Discrete(6): 0, 10, 20, 30, 40, 60% L1 unstructured |
| Reward | 1.5 · acc/base + λ_s · S(%) + 0.05 · distinct actions; λ_s 0.04 (Main, ablations), 0.01/0.02 (later studies). Accuracy is measured on the probe (or V_RL); the base is 77.32 (full validation) in Main-derived runs and 76.97 (V_RL) in the archive experiment |
| Final policy | deterministic after training |
| Fine-tuning notebook PPO (different) | n_steps 32, batch 16, n_epochs 5, ent 0.02, 300 timesteps, threshold reward |

## 6. Experiment inventory

| # | Experiment | Data / arch / baseline | Seeds | Protocol | Pruning | Sensitivity / reward | Outputs | Status | Reason |
|---|---|---|---|---|---|---|---|---|---|
| 0 | April pipeline | CIFAR-10 / SimpleCNN / `cnn_baseline.pth`, `baseline_clean.pth` (April) | — | test used for reward, sensitivity, epoch selection | unstructured | 10% probe on test | removed CSVs (history `2f37f37`), April agents, `manual_and_rl_df_current.csv`, old README numbers | **SUPERSEDED** | test-set leakage |
| 1 | Main notebook, clean | C10 / SimpleCNN / FIXED | 42, 1, 2, 3 | train/val/test; test for reporting | L1 unstructured, 4 layers | 10% probe (1k); λ 0.04 | `final_results.csv`, `ppo_multiseed.csv` | **VERIFIED** | reproduced bit-exactly by the scripted harness (policies, sparsity, test, reward) |
| 1b | Main notebook as a document | — | — | — | — | — | `notebooks/Main code.ipynb` | **NEEDS VERIFICATION** | saved state not run top-to-bottom (cells 25/26/44 show no execution); numbers verified independently, notebook should be re-executed before publication |
| 2 | Matched-sparsity analysis | C10 / SimpleCNN / FIXED | same 4 | McNemar on test | uniform & GM matched | — | `matched_sparsity_*.csv`, `per_layer_allocation.csv`, `predictions/test_predictions.csv` | **VERIFIED** | self-check assert against notebook |
| 3 | Storage (+ early latency) | FIXED + seed-42 policy | — | deterministic | — | — | experiment-log section | storage **VERIFIED**; early latency **SUPERSEDED** by #7 | unknown machine, lighter protocol |
| 4 | Fine-tuning arm | C10 / SimpleCNN / FIXED | PPO seed 42 only | val epoch selection, test once | unstructured + 3-epoch FT | threshold reward (different) | `final_results_fair_finetune_holdout.csv`, `baseline_finetuned.pth` | **VERIFIED (single run)** | 1 PPO seed; PPO learned uniform 40% |
| 5 | Matched GM + FT | same | single run | same | GM @ 39.83% + FT | — | `finetune_matched_global_magnitude.csv` | **VERIFIED (single run)** | no replication / no significance test possible |
| 6 | Original sensitivity ablation | C10 / SimpleCNN / FIXED | 42, 1, 2, 3 | harness | unstructured | correct/zeroed/shuffled; λ 0.04 | `sensitivity_ablation*.csv`, `runs/`, `training_curves/` | **VERIFIED** | exploratory (not pre-registered) |
| 7 | Reward sweep | same | 42, 1, 2, 3 | harness | unstructured | λ 0.00–0.08 | `reward_coefficient_sweep*.csv` | **VERIFIED** | exploratory; 4 seeds |
| 8 | Controlled latency | FIXED, seed-42 policy, matched GM | 600 passes | idle-load check | unstructured | — | `latency_results.csv`, `latency_samples.csv` | **VERIFIED** | CPU only (i5-6400); GPU missing |
| 9 | Sensitivity diagnostic | validation only | — | read-only | — | — | `sensitivity_feature_diagnostic.md` + CSVs | **VERIFIED** | |
| 10 | Refined sensitivity | C10 / SimpleCNN / FIXED | 42, 1–9 | harness | unstructured | multi-ratio acc/loss; λ 0.01/0.02 | `refined_*` | **VERIFIED** | values registered (hashed) before runs; analysis plan fixed in code before results; not a formal pre-registration |
| 11 | Reward landscape | validation probe | all 1,296 policies | deterministic | — | λ 0–0.08 | `reward_landscape.csv`, `learned_policy_reward_rank.csv` | **VERIFIED** | |
| 12 | Constrained PPO | C10 / SimpleCNN / FIXED | 42, 1–9 | pre-registered | unstructured + hard masks | action-level masks; λ 0.01/0.02 | `constrained_*`, `sensitivity_action_masks.csv`, `policy_landscape_validation.csv`, `sensitivity_predictive_validity.*` | **VERIFIED** | pre-registration hashed before runs, committed later (§1) |
| 13 | Soft prior | same | 42, 1–19 | pre-registered (`4a7a5a5`) | unstructured + logit prior | refined loss; β 0.5/1.0; λ 0.01 | `soft_prior_*` | **VERIFIED** | Level B |
| 14 | Exploration confirmation | same | 100–129 (fresh) | pre-registered (`c3a9484`), no peeking | same | β 0.5 | `exploration_levelA_*` | **VERIFIED** | Level A (exploration) |
| 15 | Archive confirmation | C10 / SimpleCNN / FIXED; V_RL/V_SELECT | 200–239 (fresh) | pre-registered (`1bda844`), freeze-before-test | same + archive | V_RL refined loss; β 0.5 | `archive_*` | **VERIFIED** | Level C-FA |
| 16 | README | — | — | — | — | — | `README.md` | **NEEDS VERIFICATION (update)** | written before #6–#15; lists the sensitivity ablation as future work; does not reflect later findings |

**MISSING experiments** (from the 12-point plan; see §8): other architectures,
other datasets, stronger baselines, structured pruning, FLOPs and memory,
GPU latency, multi-seed fair fine-tuning, robustness and calibration.

## 7. Evidence strength of current findings

| Finding | Evidence | Strength |
|---|---|---|
| Dense baseline 77.03% test (77.32% val) | reproduced in every run since 2026-09-26 | **STRONG** |
| PPO (λ 0.04, 4 seeds) reaches 57.9–59.7% sparsity at 75.2–76.7% test (75.87 ± 0.82) | reproduced exactly | **STRONG** (descriptive) |
| Uniform per-layer collapses at matched ~58% (≈41%) | McNemar p < 1e-300, 4 seeds | **STRONG** |
| PPO is not better than global magnitude at matched sparsity | 2 seeds n.s., 2 seeds PPO significantly worse | **STRONG** (negative for PPO) |
| Sensitivity as a state feature helps | original (4 seeds) + refined (10 seeds) ablations; correct never better | **NOT SUPPORTED** |
| Hard sensitivity constraints improve final models | pre-registered, Level C | **NOT SUPPORTED** |
| Correct > shuffled constraints at matched sparsity | secondary, p = 0.002 | **MODERATE** |
| Soft prior improves final models | pre-registered, primary failed | **NOT SUPPORTED** |
| Correct soft prior improves exploration quality (layer-specific) | pre-registered fresh-seed confirmation, d_z 0.95–1.42, Holm p ≤ 2e-5 | **STRONG** (exploration only) |
| Archive converts exploration into higher final accuracy | pre-registered, Level C-FA | **NOT SUPPORTED** |
| Correct prior avoids catastrophic archive deployments | secondary, not pre-registered as a hypothesis | **LIMITED** |
| Sensitivity predicts in-context pruning tolerance | validation landscape, 4 layers, confounded with layer size | **MODERATE** |
| λ_s moves PPO along a trade-off | 4 seeds; near switch-like, capped by the 60% action | **MODERATE** |
| Reward's diversity term shapes the optimum; PPO does not converge to it | exhaustive 1,296-policy landscape | **STRONG** |
| No measurable CPU latency gain from unstructured sparsity | controlled benchmark, one CPU | **STRONG** (CPU); GPU unmeasured |
| Raw checkpoint unchanged; gzip −49% | deterministic | **STRONG** |
| Fine-tuning recovers most pruning loss; PPO+FT learned uniform | 1 PPO seed, single runs | **LIMITED** |
| Any result outside SimpleCNN / CIFAR-10 | none | **NOT SUPPORTED** (untested) |

## 8. The 12-point plan

| # | Item | Status | Code needed | Experiment needed | Approx. runs |
|---|---|---|---|---|---|
| 1 | Multiple architectures | **not started** | CIFAR ResNet-18 and MobileNetV2 definitions; architecture-agnostic layer discovery, state vector and environment (`rl_env` is hard-coded to SimpleCNN's 4 layers); layer-grouping rule for long networks | dense baselines, then PPO + baselines per architecture | 12 dense trainings; PPO see Phase 3 |
| 2 | Multiple datasets | **not started** | CIFAR-100 loaders with the same split policy (45k/5k/10k, V_RL/V_SELECT) | CIFAR-100 baselines for each architecture | 9 dense trainings |
| 3 | Stronger baselines | **partial** (uniform, global magnitude, heuristic, single-run GM+FT) | LAMP, ERK-style allocation, random, layer-adaptive magnitude; iterative magnitude pruning with fine-tuning | matched-sparsity evaluations per architecture/dataset | inference-only ~300 evaluations + IMP ~30 training runs |
| 4 | Independent validation / model selection | **partial** (archive experiment only) | generalise `ArchiveData` to all datasets | apply V_RL/V_SELECT to every new experiment | — |
| 5 | Fresh-seed replication | **partial** (SimpleCNN exploration and archive) | none | fresh seeds for every new confirmatory claim | per experiment |
| 6 | Sensitivity mechanism analysis | **partial** (SimpleCNN) | sensitivity curves for new architectures | predictive validity + prior on new architectures | inference ~500 evals; PPO see Phase 4 |
| 7 | Reward-design analysis | **partial** (λ sweep, landscape) | diversity-bonus ablation; scale-free accuracy term | λ × diversity grid | ~200 SimpleCNN runs (CPU) |
| 8 | Structured pruning | **not started** | channel/filter pruning with dependency handling (torch-pruning or custom), structured action space | structured PPO + baselines + fine-tuning | ~200 runs |
| 9 | Real efficiency measurement | **partial** (CPU latency, storage) | FLOP/parameter counter; GPU latency with synchronisation; memory | measure dense/unstructured/structured models | inference-only ~100 measurements |
| 10 | Fair fine-tuning | **partial** (1 seed, SimpleCNN) | one fine-tuning script for all methods, same budget, validation selection | multi-seed FT of every final model | ~300 FT runs |
| 11 | Robustness / calibration | **not started** | CIFAR-10-C / CIFAR-100-C loaders; ECE, NLL, Brier | inference on final models | ~100 model evaluations |
| 12 | Full reproducibility | **partial** (hashes, locks, pre-registrations, resumable runners) | consolidated library; scripted baseline training (replace notebook); GPU/CUDA determinism settings; container or conda file | re-execution check | — |

## 9. Issues to resolve (none blocks inspection; several block Phase 2)

1. **No GPU** on this machine (Intel HD 530; CPU-only torch). ResNet-18,
   MobileNetV2 and CIFAR-100 training is infeasible here (see
   `rocksolid_compute_plan.md`). Blocks Phase 2.
2. `checkpoints/baseline_clean.pth`: uncommitted orphan modification.
   Restore or remove before new work, so the tree is clean.
3. `cnn_baseline.pth` is unusable (chance accuracy). Exclude from everything.
4. `main` lacks the 11 latest commits; a second PR is needed for
   `main` to reflect the verified results.
5. README is outdated relative to experiments #6–#15.
6. `rl_env` / the notebooks are SimpleCNN-specific; scaling needs an
   architecture-agnostic, consolidated library. Historical scripts stay
   frozen.
7. The Main notebook should be re-executed top-to-bottom (scripted) before
   publication.
8. CIFAR-100, CIFAR-10-C and CIFAR-100-C are not downloaded.
9. Missing dependencies (not installed): a FLOP counter, a structured-pruning
   dependency tool, and the CUDA build of torch for a GPU machine.
10. The power supply is unstable (27 unexpected shutdowns in 60 days).
    Every runner must stay resumable, and results must be committed after
    each phase.
