# Phase 6 pre-registration: fair post-pruning fine-tuning and performance recovery

Written 2026-10-08 on branch `phase6-cpu`, after the timing pilot and the implementation checks and
**before any confirmatory fine-tuning run**. Part A (`phase6_design_commitments.md`, commit `74e5f45`)
was committed before the pilot and is not edited; this document completes it and is consistent with it.
Once committed, this file is never edited. Any later change is written as a separate, dated amendment.

**State at freezing.** No Phase-5 policy has been fine-tuned. Phase 6 has not read the test set. The
only models trained so far are pilot models (seed 42, uniform policy `[2,2,2,2]`, 1–2 epochs).

## 1. Question

Do the accuracy differences between sensitivity-aware PPO pruning and conventional pruning persist when
every pruned model receives the same fixed-mask fine-tuning budget at exactly the same sparsity?

This is a controlled comparison. It is not designed to make PPO win, and a result in favour of LAMP or
of no difference is reported as such.

## 2. Frozen inputs

| Input | sha256 (prefix) |
|---|---|
| Part A, `phase6_design_commitments.md` | `7f67b9e25b2e0f25` |
| `phase6_config.json` | `c2907444ca760daa` |
| `phase6_matched_masks.csv` (120 triples) | `ef16fdcf11d85373` |
| `phase6_timing_pilot.csv` | `d603f689bc9cf00e` |
| Phase-5 final policies, `phase5_final_policies.json` | `d04c8556a9500a7c` |
| Phase-5 test evaluations, `phase5_test_evaluations.csv` | `baa6b901291e8684` |
| Split `cifar10_phase5_split.npz` / `cifar100_phase5_split.npz` | `eccd7522d55b4048` / `131cf53457b45ca2` |
| Dense references (C10 SimpleCNN, LeNet-5, ResNet-8; C100 SimpleCNN, LeNet-5, ResNet-8) | `ca28f834`, `5ca6e0fc`, `41fc5ee5`, `d75dd19c`, `8e1b1aee`, `8b84bfad` |
| Code: `run.py`, `phase6_lib.py` | `b06f614fe8651499`, `588b3473ba26b35e` |
| Code: `aggregate.py`, `make_figures.py` | `dd0ef76b00a252e1`, `4226f10f07a375a0` |

- **Data.** 45,000 training images; VAL-RL 2,500 and VAL-SELECT 2,500 (the Phase-5 stratified split);
  the official 10,000-image test set. Only the training images are used to update weights.
- **Phases 1–5** are frozen evidence. Nothing in them is re-trained, re-selected or edited; the
  integrity checks verify this against commits `67f4270`, `a3653fd` and `da061e6`.

## 3. Methods and exact matched sparsity

| Method | Mask |
|---|---|
| **M0 PPO** | the frozen Phase-5 C1 policy of each (setting, policy seed): centred sensitivity prior, accuracy-constrained reward (τ = 0.98), elite archive |
| **M1 LAMP** | the existing `src.pruning` LAMP, applied to the same dense checkpoint at **exactly the PPO model's zero count** |
| **M2 global magnitude** | the existing `src.pruning` global magnitude pruning at the same zero count |
| **Dense control** | no mask; the dense reference fine-tuned with the same protocol and seed |

- **Scope.** The same prunable tensors for all three methods; the output classifier is never pruned.
- **Matching.** `phase6_matched_masks.csv` lists, for each of the 120 (setting, policy seed) pairs, the
  zero count, the per-layer zeros and the mask hash of each method. The zero-count mismatch is 0 in all
  120 triples. No two methods share a mask in any triple.

| Setting | Prunable / total weights | Distinct PPO policies (of 20) | Sparsity, mean (min–max) % |
|---|---|---|---|
| CIFAR-10 SimpleCNN | 617,312 / 619,872 | 14 | 56.93 (53.44–58.78) |
| CIFAR-10 LeNet-5 | 60,930 / 61,770 | 14 | 38.57 (33.49–41.65) |
| CIFAR-10 ResNet-8 | 76,720 / 77,360 | 4 | 24.50 (22.89–25.28) |
| CIFAR-100 SimpleCNN | 617,312 / 642,912 | 13 | 53.09 (49.50–54.39) |
| CIFAR-100 LeNet-5 | 60,930 / 69,330 | 12 | 30.60 (24.09–33.92) |
| CIFAR-100 ResNet-8 | 76,720 / 83,120 | 4 | 15.62 (9.23–16.08) |

Sparsity is the share of zeros among all Conv2d/Linear weights, as in Phases 2–5.

## 4. Fine-tuning protocol (from Part A, identical for every run)

| Item | Value |
|---|---|
| Start | the frozen dense checkpoint with the method's mask applied; a new optimiser for every run |
| Optimiser | Adam, lr 1e-4, betas (0.9, 0.999), eps 1e-8, weight decay 0 |
| Schedule | cosine annealing to 0 over E epochs, stepped per epoch |
| Batch / data | 128; the 45,000 training images; random crop (pad 4) + horizontal flip |
| BatchNorm | train mode during fine-tuning for every method; eval mode for every evaluation; no separate recalibration |
| Mask | fixed, by the PyTorch pruning reparameterisation (effective weight = stored weight × mask); mask hash, masked-weight violations and zero count are asserted after every epoch |
| Selection | **none**: the model after epoch E is the result of every run |
| Threads | 1 per run; 4 worker processes |

**Epoch count: E = 5.** Part A fixed the rule before the pilot: E = 5 if the projected wall time of the
whole experiment is ≤ 30 hours, otherwise E = 3. The projection formula was committed with the pilot
script, and it gives **29.44 hours**, so E = 5.

- The margin is small, and I state it rather than hide it: counting each 10,000-image test pass as two
  validation passes, the same formula gives 29.99 hours. Both are within the limit.
- No accuracy was used for this decision.
- There is no 10-epoch run (projected 58 hours).

## 5. Seeds, pairing and run matrix

- **Policy seeds** 500–519 (the 20 Phase-5 C1 runs of each setting).
- **Fine-tuning seed** = policy seed + 100 (600–619). `torch.manual_seed(seed)`; the data order and the
  augmentation come from `torch.Generator(seed + 10^6)`.
- **Pairing.** The four runs of a (setting, policy seed) share the fine-tuning seed, the minibatch order
  and the augmentation. A hash of the data-stream state is stored per run and compared.

| | Runs |
|---|---|
| Pruned models: 6 settings × 20 policy seeds × 3 methods | 360 |
| Dense controls: 6 settings × 20 fine-tuning seeds | 120 |
| **Confirmatory fine-tuning runs** | **480** |
| Reproduction runs: the PPO run of policy seed 500 in each setting, repeated | 6 |

**Unit of analysis.** One policy seed: an independent Phase-5 PPO training run with its own fine-tuning
seed. n = 20 per setting.

**Repeated policies.** Several policy seeds selected the same policy (see the table in §3; in CIFAR-100
ResNet-8 one policy was selected by 17 of 20 seeds). For these seeds the masks and the pre-fine-tuning
accuracies are identical; the fine-tuned models differ, because each has its own fine-tuning seed. No
run is a duplicate of another.
- The run-level analysis estimates the effect for *the Phase-5 C1 pipeline followed by fine-tuning*.
- A **cluster-level sensitivity analysis** uses one value per distinct PPO policy (the mean over its
  seeds). It is tested by sign-flip only where there are ≥ 6 distinct policies, so not for ResNet-8,
  where it is descriptive.
- A run-level result whose cluster-level mean has the opposite sign is reported as not robust to policy
  clustering.

## 6. Stages and test-set access

| Stage | What | Test set |
|---|---|---|
| 1 | 480 fine-tuning runs and 6 reproduction runs; validation accuracy (VAL-RL, VAL-SELECT) before fine-tuning and after every epoch; final checkpoint saved | not read |
| 2 | `phase6_final_models.json`: state hash, checkpoint hash, mask hash and zero count of the 480 final-epoch models; written once, then committed | not read |
| 3 | test accuracy of every pre-fine-tuning model and of exactly the 480 frozen checkpoints; per-image predictions saved | read |

- The runner refuses stage 3 without the frozen list, refuses a checkpoint whose hash differs from the
  list, and refuses new fine-tuning runs once the list exists.
- The pre-fine-tuning test accuracy must equal the Phase-5 record for every model (asserted).
- A pre-fine-tuning model is evaluated once per distinct mask, not once per seed.

**Disclosure.** The official test sets were read in Phases 2–5 for frozen models, so Phase 6 is not a
fully independent confirmation. No Phase-6 design choice (learning rate, epochs, schedule, methods,
statistics) uses test accuracy; all were fixed in Part A or here.

## 7. Metrics

Per run, on the test set unless stated:
- **A_pre, A_post**: accuracy before and after fine-tuning. **Recovery** = A_post − A_pre (pp).
- **Dense gap** = A_dense − A (before and after), with A_dense the frozen dense reference.
- **G_pre, G_post** = A_PPO − A_baseline before and after, for LAMP and for global magnitude.
  **ΔG** = G_post − G_pre. A positive ΔG means fine-tuning moved the gap in PPO's favour.
- **Retention** = A / A_dense, before and after; the share of runs with retention ≥ 0.98.
  **Retention gap** = retention on VAL-SELECT − retention on test.
- **Pruning-specific recovery** = recovery of the pruned model − gain of the dense control of the same
  pair. It separates recovery from the effect of extra training.

Secondary, per run: sparsity and per-layer zeros before and after; training loss and validation
accuracy per epoch; fine-tuning CPU seconds; peak process memory; checkpoint size raw and gzip.
No latency claim is made for unstructured sparsity.

## 8. Hypotheses and statistics

**Tests.** Exact two-sided sign-flip tests paired by policy seed (n = 20, 2^20 patterns). Each
comparison reports n; mean ± SD and median of both sides; the paired mean difference; the 95% t-CI and
the bootstrap CI; d_z; the raw and Holm-adjusted p; wins, ties and losses. Holm correction is applied
within each family.

**Outcome of one comparison.**
- **Supported:** the mean has the stated direction, Holm p < 0.05 **and** the 95% t-CI excludes 0.
- **Opposite:** the same in the reverse direction.
- **Inconclusive:** otherwise.

**Practical relevance.** Statistical support and size are reported separately. The margin is 0.5 pp,
the margin Phase 5 used for "within reach of LAMP".
- A difference is called *practically relevant* only if |mean| ≥ 0.5 pp.
- Two methods are called *practically equivalent* only if the 95% t-CI lies inside (−0.5, +0.5) pp.
- A supported difference below 0.5 pp is described as statistically detectable but small.

| Family | Comparisons | Size | Direction |
|---|---|---|---|
| **H1** recovery | A_post − A_pre for PPO, LAMP and global, per setting | 18 | + |
| Dense control | A_post − A_pre of the dense control, per setting | 6 | + |
| Pruning-specific recovery | recovery − dense-control gain, per method and setting | 18 | + |
| **H2** PPO vs LAMP | G_post (6) and ΔG (6) | 12 | + = in PPO's favour |
| **H3** PPO vs global | G_post (6) and ΔG (6) | 12 | + = in PPO's favour |
| H2 / H3 reference | G_pre per setting, for LAMP and for global | 6 each | + |
| **H4** architecture dependence | PPO recovery, each pair of architectures within a dataset | 6 | two-sided |
| H4 sensitivity | the same pairs, paired by fine-tuning seed | 6 | two-sided |
| **H6** retention gap | PPO: retention gap after − before, per setting | 6 | − |

- **H1 (recovery).** Fine-tuning improves accuracy over the same fixed-mask model before fine-tuning.
  "Opposite" means fine-tuning significantly reduced accuracy; it is reported, not hidden.
- **H2 (PPO vs LAMP), primary.** Two endpoints in one family: the difference after fine-tuning, and the
  change of the gap. For ΔG: supported = the gap moved toward PPO; opposite = it moved toward LAMP;
  inconclusive = no detectable change. All six mean G_pre are negative in Phase 5 (LAMP ahead by 0.26
  to 0.52 pp), so a positive ΔG is a narrowing unless PPO overtakes LAMP, which is stated explicitly.
- **H3 (PPO vs global).** The same two endpoints against global magnitude pruning.
- **Reference families.** G_pre is deterministic given the policy and was already reported in Phase 5.
  It is restated at run level for the tables and is not new evidence.
- **H4 (architecture dependence).** The settings being compared are different experiments, so the
  primary test is unpaired: a two-sample permutation test (100,000 permutations, seed 0) with a Welch
  95% CI. Runs with the same index share a data stream, so a sign-flip test paired by fine-tuning seed
  is reported as a sensitivity analysis. H4 holds for a dataset if at least one pair differs in either
  direction. Architectures were pruned to different sparsities, so a difference in recovery reflects
  the architecture together with its sparsity and its pre-fine-tuning loss, not the architecture alone.
- **H5 (sparsity preservation).** No test. It holds if, in all 360 pruned runs, the per-layer zero
  counts are identical before and after, the mask hash is unchanged and no masked weight is non-zero.
- **H6 (retention).** Retention is accuracy divided by a constant per setting, so its improvement is
  tested by H1; H6 holds for a method and setting where H1 is supported and H5 holds. The retention-gap
  family tests separately whether fine-tuning reduces the Phase-5 validation-to-test gap of PPO.

**Cross-setting summary.** For each comparison: the number of settings supported, opposite and
inconclusive, out of six. No setting is dropped.

**RQ6 (cost).** Descriptive: CPU minutes per model against recovery. No test.

**Predictions.** Per-image test predictions are saved for every model. No prediction-level test is
pre-registered.

## 9. Criteria (each reported as MET / PARTIALLY MET / NOT MET with the exact numbers)

- **A. Fairness.** MET if all of these hold: one protocol and one epoch count in all 480 runs; the four
  runs of every pair share the fine-tuning seed and the data stream; the zero-count mismatch is 0 in
  all 120 triples; every mask equals the frozen table. Otherwise NOT MET.
- **B. Recovery.**
  - MET: recovery is supported for all three pruning methods in ≥ 2 settings.
  - PARTIALLY MET: not MET, but some method is supported in ≥ 2 settings, or all three in exactly one.
  - NOT MET: otherwise.
- **C. PPO competitiveness** (on ΔG against LAMP).
  - MET: the gap moved toward PPO (supported) in ≥ 2 settings and toward LAMP in none.
  - PARTIALLY MET: supported in ≥ 1 setting, but not MET.
  - NOT MET: otherwise.
  - Reported with it: whether LAMP is still ahead after fine-tuning, and the practical-relevance and
    equivalence labels. C concerns the change of the gap, not who is ahead.
- **D. Architecture generalisation.** MET if all six settings are complete (80 runs each) and appear in
  every per-setting family, table and figure. PARTIALLY MET if all appear but one is incomplete.
  NOT MET if any setting is left out.
- **E. Sparsity preservation.** MET if H5 holds in all 360 pruned runs. PARTIALLY MET if in ≥ 95%.
  NOT MET otherwise.
- **F. Reproducibility.** MET if all 480 runs complete and all 6 reproduction runs are bit-identical
  (weights hash and validation accuracy). PARTIALLY MET if all 480 complete but fewer than 6 are
  identical. NOT MET otherwise.

## 10. Interpretation rules (fixed now)

- PPO is never compared after fine-tuning with a baseline before fine-tuning.
- If LAMP remains ahead after fine-tuning, that is the finding.
- If PPO recovers more but remains below LAMP, recovery and final accuracy are reported separately.
- If the gap moves toward the baseline, the baseline's masks are described as more recoverable.
- Recovery is never attributed to the pruning policy without the dense control beside it.
- A small difference is not called an improvement (§8, practical relevance).
- Architecture dependence is reported if recovery differs across architectures.
- Negative and inconclusive results are kept.

## 11. Conduct and integrity

- **No interim peeking.** Progress reports show run counts, runtimes and failures only. No accuracy is
  aggregated before all runs are complete and the integrity checks have passed.
- **Resumability.** Every run is resumable per epoch and written durably (fsync + atomic rename). An
  interrupted run continues from its last completed epoch with the same seed; the checks below show
  that this is bit-identical to an uninterrupted run. Interruptions are logged.
- **No dropped runs.** A run that fails for an infrastructure reason is repeated with its original
  seed. No run is excluded for its result.
- **Integrity checks** (`phase6_integrity_checks.md`) run after stage 3 and before any statistic:
  - all 480 runs complete, none dropped or added; one protocol and epoch count;
  - shared seed and data stream within every pair; zero-count mismatch 0; masks equal the frozen table;
  - fixed mask in every run;
  - the frozen list equals the stage-1 results, the checkpoint files equal the list, and every test
    evaluation is of a frozen checkpoint;
  - every fine-tuning run finished before the list was frozen, and every test evaluation started after;
  - stage-1 records hold no test accuracy; the evaluated model is the final-epoch model;
  - pre-fine-tuning test accuracy equals the Phase-5 record;
  - pre-registration, configuration, dense checkpoints and splits unchanged; Phases 3–5 unchanged;
  - 1 thread; commit, CPU and package versions recorded; interruptions logged;
  - the 6 reproduction runs (reported; this one decides criterion F and does not block the analysis).
- If any other check fails, no statistic is computed and the failure is reported first.

## 12. Pilot, checks and incidents before freezing

**Pilot** (seed 42, policy `[2,2,2,2]`, test never read): one epoch per setting and method, 4 concurrent
1-thread workers. Training time per epoch: SimpleCNN 158–165 s, LeNet-5 24–29 s, ResNet-8 285–309 s.
Peak process memory ≤ 827 MB. Zero counts unchanged and no mask violation in all 24 pilot runs.

**Implementation checks** (`_pilot_parts/checks.json`, `checks_bn.json`): 11 of 11 passed.
- Fixed mask after 2 epochs for each method: mask hash equal, no violation, zero count unchanged, and
  the stored parameters under the mask never updated.
- Identical zero count for the three methods; classifier dense.
- Bit-identical repeat; bit-identical resume after an interruption (LeNet-5, and ResNet-8 with
  BatchNorm statistics); identical data stream across methods.
- A saved checkpoint reloads into a plain model with identical outputs and zero count.
- 18 Phase-5 pruned models reproduced by VAL-RL accuracy and zero count.

**Self-tests** (`_pilot_parts/selftest_*.json`): 14 of 14 passed.
- The three-stage runner end to end on pilot inputs, with the official test set made unreadable.
- The analysis and figure code on 480 synthetic records with one planted effect, which it recovered
  where planted and nowhere else.

**Incidents and changes before freezing.** None changes Part A.
1. *Power cut during the timing pilot.* The unsaved CIFAR-10 timing rows were recovered from the worker
   logs; the CIFAR-100 rows were measured after the restart. Timing is now saved per setting.
2. *Bug found by the checks and fixed.* After training, PyTorch's derived masked weight is one optimiser
   step behind until the next forward pass. My first version counted zeros on that stale tensor and
   failed when copying the model. The library now recomputes it before every count, check and copy.
   The pilot timing is unaffected.
3. *Test access made stricter.* The draft runner read the test set at the end of each run. The frozen
   runner reads it only after all 480 models are frozen and committed (§6).
4. *Analysis choices fixed here, differing from my draft script.* The practical margin is 0.5 pp (the
   draft had 0.25); H2 and H3 are each one family of 12; the reproduction check does not block the
   analysis. None of these was informed by a result.
5. *Accidental re-run of the Phase-3 aggregation.* A self-test imported the Phase-3 script of the same
   name. It rewrote three Phase-3 files with a new timestamp only (the two timestamp lines and a gzip
   header; every number was regenerated identically). I restored the three files from commit `67f4270`
   and verified that Phases 3–5 have no difference from their frozen commits. Phase-6 scripts are now
   loaded by file path.
6. *Pilot accuracies seen.* The checks printed training losses, and the runner self-test printed the
   VAL-RL accuracy of two pilot models after one epoch. No design element was changed afterwards: the
   protocol was fixed in Part A before the pilot, and E by the timing rule before the self-test ran.

## 13. Budget

| Stage | Process-hours | Wall-hours on 4 workers |
|---|---|---|
| 1: 480 fine-tuning runs | 114 | 28.5 |
| 1: 6 reproduction runs | 1.5 | 0.5 |
| 3: test evaluation | 3 | 0.7 |
| **Total** | **≈ 119** | **≈ 30** |

Per run at E = 5: LeNet-5 2–3 minutes, SimpleCNN about 15 minutes, ResNet-8 25–27 minutes. With power
cuts and normal use of the machine, two to three days of elapsed time are expected.

## 14. Not part of this experiment

- **Secondary methods** (Phase-5 C0 and C2, uniform, ERK) and a **10-epoch schedule** are not run: the
  primary comparison alone uses the 30-hour budget fixed in Part A. If they are run later, they need
  their own amendment committed before their runs, and they are labelled secondary because they would
  be designed after the primary results are known.
- **External data.** No untouched evaluation set is on this machine. CIFAR-10.1 would allow a separate
  confirmation for the three CIFAR-10 settings; it would be registered separately before any evaluation.
- **Structured pruning and latency** belong to later phases.

## 15. Outputs

In `results/phase6/`: this file, `phase6_config.json`, `phase6_environment.json`,
`phase6_timing_pilot.csv`, `phase6_matched_masks.csv`, `phase6_final_models.json`,
`phase6_integrity_checks.md`, `phase6_all_runs.csv`, `phase6_training_curves.csv`,
`phase6_pre_post_accuracy.csv`, `phase6_accuracy_recovery.csv`, `phase6_ppo_vs_lamp.csv`,
`phase6_ppo_vs_global.csv`, `phase6_statistics.csv`, `phase6_cluster_sensitivity.csv`,
`phase6_cross_setting_summary.csv`, `phase6_classification.json`, `phase6_summary.md` (20 sections,
Tables I–IV), and `figures/` with Figs. 1–5 and their source CSV files. Figure styling may be adjusted
after the results; the plotted quantities may not.

Fine-tuned checkpoints are kept in `checkpoints/phase6_finetuned/` (about 0.5 GB) with their hashes in
the frozen list.
