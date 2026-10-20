# Phase 6 summary: fair post-pruning fine-tuning and performance recovery

Branch `phase6-cpu`. Pre-registration `phase6_preregistration.md` (sha256 `cff68be3…`, commit
`b1b5396`); final-model list frozen before any test access (`3181146`). Every number below comes from
the files in `results/phase6/`. The outcome labels follow the pre-registered rule: supported / opposite
/ inconclusive, with Holm correction inside each family and a 95% t-CI that excludes 0.

**Question.** When PPO (Phase-5 C1), LAMP and global magnitude pruning, at exactly the same zero count,
all receive the same 5-epoch fixed-mask fine-tuning, do their accuracy differences persist?

**Answer in brief.**
- Fine-tuning recovers essentially all of the pruning loss, for every method in every setting: on
  average every method ends within 0.14 pp of the dense reference.
- After fine-tuning, the three methods are within 0.26 pp of each other everywhere.
- The PPO–LAMP gap moved in PPO's favour in 5 of 6 settings. LAMP is still slightly ahead in 3
  settings, by 0.14–0.22 pp.
- PPO's large advantage over global magnitude before fine-tuning (up to 6.5 pp) disappears.
- The pruning policy matters before fine-tuning, but at these sparsities it matters very little after.

## 1. Completion and integrity

- **Runs.** 480 of 480 confirmatory runs (360 pruned, 120 dense controls) and 6 of 6 reproduction
  runs completed. None was dropped, added or repeated.
- **Integrity checks.** 20 of 20 passed (`phase6_integrity_checks.md`):
  - every pair shares the protocol, epoch count, fine-tuning seed and data stream;
  - the zero-count mismatch is 0 in all 120 triples, and every mask equals the frozen table;
  - the mask is fixed in every run;
  - the frozen list equals the stage-1 results;
  - every fine-tuning run finished before the freeze, and every test read started after it;
  - stage-1 records contain no test accuracy;
  - the pre-fine-tuning test accuracy equals the Phase-5 record in every run;
  - the pre-registration, configuration, dense checkpoints and splits are unchanged;
  - Phases 3–5 are unchanged (empty `git diff` against `67f4270`, `a3653fd`, `da061e6`);
  - 1 thread in every run.
- **Reproduction.** All 6 repeated runs are bit-identical to the originals in weights and validation
  accuracy.
- **Interruptions.** There was one (`runs/infrastructure_log.txt`). The workers stopped with the
  session on 2026-10-08 after 180 runs. On 2026-10-19 the 180 records were checked read-only (all
  valid), and the 4 partially trained runs continued from their per-epoch states with the original
  seed. Resuming had been verified to be bit-identical in the implementation checks.

## 2. Frozen experimental configuration

**Table I. Experimental configuration**

| Item | Value |
|---|---|
| Datasets | CIFAR-10, CIFAR-100 (train 45,000; VAL-RL 2,500; VAL-SELECT 2,500; test 10,000) |
| Architectures | SimpleCNN, LeNet-5, ResNet-8 (frozen Phase-2 dense references) |
| Methods | M0 PPO (Phase-5 C1 policy); M1 LAMP; M2 global magnitude; dense control |
| Pruning scope | the 4 PPO units of each architecture; the output classifier is never pruned |
| Matching | LAMP and global at exactly the PPO model's zero count |
| Policy seeds / fine-tuning seeds | 500–519 / 600–619 (policy seed + 100); the same data order within each pair |
| Optimiser | Adam, lr 1e-4, betas (0.9, 0.999), eps 1e-8, weight decay 0 |
| Schedule, epochs | cosine annealing to 0, stepped per epoch; **E = 5** (from the pre-registered timing rule) |
| Batch, augmentation | 128; random crop (pad 4) + horizontal flip |
| BatchNorm | train mode during fine-tuning, eval mode for every evaluation |
| Mask | fixed (PyTorch pruning reparameterisation), checked after every epoch |
| Selection | none: the model after epoch 5 |
| Threads / workers | 1 per run / 4 |

## 3. CPU feasibility and computational cost

- **Total.** Fine-tuning used 101 process-hours. The pre-registered projection was 114; the work ran
  on 4 workers at about 4 runs in parallel.
- **Per model** (1 thread, 4 concurrent workers), with no difference between methods:

  | Architecture | CIFAR-10 | CIFAR-100 |
  |---|---|---|
  | LeNet-5 | 1.6 min | 1.7 min |
  | SimpleCNN | 14.0 min | 11.6 min |
  | ResNet-8 | 24.1 min | 23.1 min |

- **Memory.** The peak process working set was at most 1,029 MB.
- **Checkpoints** (gzip): SimpleCNN 1.19–1.33 MB pruned versus 2.31–2.40 MB dense. These are storage
  figures only; no latency claim is made for unstructured sparsity.

## 4. Matched-sparsity verification

- **Zero counts.** Identical for the three methods in all 120 (setting, policy seed) triples; the
  absolute mismatch is 0. No two methods share a mask.
- **After fine-tuning.** Per-layer zero counts are identical before and after in all 360 pruned runs,
  and no masked weight is non-zero, so no pruned weight came back.
- **Mean total sparsity:**

  | Architecture | CIFAR-10 | CIFAR-100 |
  |---|---|---|
  | SimpleCNN | 56.93% | 53.09% |
  | LeNet-5 | 38.57% | 30.60% |
  | ResNet-8 | 24.50% | 15.62% |

  Per-layer zeros and mask hashes are in `phase6_matched_masks.csv`.

## 5. Fine-tuning protocol

Every method used the same protocol (Table I). The pruned and dense models of a pair shared the
fine-tuning seed, the minibatch order and the augmentation; the data-stream hashes are equal within
every pair. The evaluated model is the one after the final epoch in every run, with no checkpoint
selection.

## 6. Pre-fine-tuning results

Pre-fine-tuning models are the Phase-5 models, and their test accuracies reproduce the Phase-5 records
exactly (Table II, "pre").
- **LAMP.** Ahead of PPO in all six settings, by 0.26–0.53 pp (reference family, all opposite).
- **Global magnitude.** Behind PPO in 5 settings (by 0.32–6.51 pp), and level in CIFAR-10 SimpleCNN.

## 7. Post-fine-tuning results

**Table II. Test accuracy (%) before → after fine-tuning (mean of 20 runs) at identical sparsity**

| Setting | Sparsity | Dense ref. | Dense control (FT) | PPO | LAMP | Global |
|---|---|---|---|---|---|---|
| C10 SimpleCNN | 56.93 | 77.03 | 80.08 | 76.45 → 79.73 | 76.94 → 79.95 | 76.54 → 79.89 |
| C10 LeNet-5 | 38.57 | 64.96 | 65.10 | 63.83 → 64.88 | 64.15 → 65.06 | 62.41 → 64.83 |
| C10 ResNet-8 | 24.50 | 78.69 | 78.83 | 77.51 → 78.74 | 78.03 → 78.73 | 71.00 → 78.67 |
| C100 SimpleCNN | 53.09 | 50.50 | 50.60 | 50.07 → 50.53 | 50.36 → 50.58 | 49.75 → 50.51 |
| C100 LeNet-5 | 30.60 | 32.28 | 32.39 | 31.79 → 32.36 | 32.05 → 32.49 | 31.43 → 32.24 |
| C100 ResNet-8 | 15.62 | 43.99 | 44.08 | 42.83 → 44.02 | 43.34 → 44.06 | 39.35 → 44.04 |

- **Every method ends close to the dense reference after fine-tuning.** The mean shortfall is at most
  0.13 pp (global, C10 LeNet-5). Mean test retention after fine-tuning is 0.998–1.038 for every method
  and setting, and every pruned model meets retention ≥ 0.98 on test. Before fine-tuning the share meeting it was 0–100% by method and setting.
- **CIFAR-10 SimpleCNN is a special case.** Its reference is the historical checkpoint, and the dense
  control itself gains +3.05 pp from 5 more epochs. Every model in that setting therefore ends about
  3 pp above the reference.

## 8. Accuracy recovery by method

**Table III. Paired recovery A_post − A_pre (pp, test; mean ± SD, 95% t-CI, Holm p)**

| Setting | PPO | LAMP | Global | Dense control |
|---|---|---|---|---|
| C10 SimpleCNN | 3.28 ± 0.33 [3.13, 3.44] | 3.01 ± 0.10 [2.96, 3.05] | 3.35 ± 0.13 [3.29, 3.41] | 3.05 ± 0.08 [3.01, 3.09] |
| C10 LeNet-5 | 1.04 ± 0.28 [0.91, 1.18] | 0.91 ± 0.26 [0.79, 1.03] | 2.42 ± 0.32 [2.27, 2.56] | 0.14 ± 0.09 [0.10, 0.19] |
| C10 ResNet-8 | 1.23 ± 0.27 [1.11, 1.36] | 0.70 ± 0.10 [0.66, 0.75] | 7.67 ± 0.71 [7.34, 8.00] | 0.14 ± 0.08 [0.10, 0.18] |
| C100 SimpleCNN | 0.46 ± 0.22 [0.35, 0.56] | 0.22 ± 0.11 [0.17, 0.27] | 0.76 ± 0.15 [0.69, 0.83] | 0.10 ± 0.07 [0.07, 0.14] |
| C100 LeNet-5 | 0.57 ± 0.28 [0.44, 0.70] | 0.44 ± 0.15 [0.37, 0.51] | 0.81 ± 0.19 [0.72, 0.90] | 0.11 ± 0.10 [0.07, 0.16] |
| C100 ResNet-8 | 1.20 ± 0.12 [1.14, 1.25] | 0.72 ± 0.18 [0.64, 0.81] | 4.69 ± 1.15 [4.16, 5.23] | 0.09 ± 0.11 [0.04, 0.14] |

- **H1, recovery.** Supported in all 18 method × setting tests. Holm p < 10⁻⁴ for every pruned-method
  test. The dense control's own gain is also supported in 6/6, with Holm
  p ≤ 0.0014.
- **Recovery beyond the dense control** (pruning-specific family):
  - supported in 17 of 18 tests;
  - the exception is LAMP on CIFAR-10 SimpleCNN, which recovers 0.04 pp *less* than the dense control
    gains (opposite);
  - outside CIFAR-10 SimpleCNN, pruning-specific recovery ranges from 0.12 pp (LAMP, CIFAR-100
    SimpleCNN) to 7.53 pp (global, CIFAR-10 ResNet-8).
- **Ordering of recovery.** Global > PPO > LAMP in all six settings. This is the inverse of the
  pre-fine-tuning accuracy: the more a mask lost, the more it recovered.

## 9. PPO versus LAMP

**Table IV. PPO − LAMP test accuracy (pp; mean, 95% t-CI, Holm p within the 12-test H2 family)**

| Setting | G_pre (Phase 5) | G_post | Outcome G_post | ΔG = G_post − G_pre | Outcome ΔG |
|---|---|---|---|---|---|
| C10 SimpleCNN | −0.49 | −0.22 [−0.28, −0.15], p < 10⁻⁴ | opposite (LAMP ahead) | +0.28 [0.12, 0.43], p = 0.003 | supported |
| C10 LeNet-5 | −0.32 | −0.18 [−0.28, −0.09], p = 0.003 | opposite (LAMP ahead) | +0.14 [0.03, 0.24], p = 0.055 | inconclusive |
| C10 ResNet-8 | −0.53 | +0.00 [−0.02, 0.03], p = 0.82 | inconclusive | +0.53 [0.43, 0.63], p < 10⁻⁴ | supported |
| C100 SimpleCNN | −0.29 | −0.05 [−0.09, −0.01], p = 0.055 | inconclusive | +0.24 [0.15, 0.32], p ≤ 10⁻⁴ | supported |
| C100 LeNet-5 | −0.26 | −0.14 [−0.19, −0.08], p < 10⁻³ | opposite (LAMP ahead) | +0.12 [0.04, 0.21], p = 0.027 | supported |
| C100 ResNet-8 | −0.51 | −0.04 [−0.07, −0.01], p = 0.055 | inconclusive | +0.47 [0.42, 0.53], p < 10⁻⁴ | supported |

- **The gap narrowed in every setting.** ΔG is supported in 5 of 6 (C10 LeNet-5 inconclusive at Holm
  p = 0.055), opposite in none. PPO overtakes LAMP nowhere.
- **After fine-tuning LAMP is still ahead in 3 settings** (C10 SimpleCNN, C10 LeNet-5, C100 LeNet-5),
  by 0.14–0.22 pp. The other 3 show no detectable difference.
- **Size.** In all six settings the 95% CI of G_post lies inside ±0.5 pp, so PPO and LAMP are
  *practically equivalent* after fine-tuning by the pre-registered margin. Only one ΔG reaches the
  0.5 pp practical-relevance threshold (CIFAR-10 ResNet-8, +0.53 pp).
- **Robustness to repeated policies.** At cluster level (one value per distinct PPO policy) every
  supported or opposite result keeps its sign; only two inconclusive ResNet-8 G_post rows change
  sign. For ResNet-8 (4 distinct policies) the cluster level is
  descriptive only, and its ΔG means (+0.49, +0.61) agree with the run level.

## 10. PPO versus global magnitude

| Setting | G_pre | G_post (95% CI) | Outcome G_post | ΔG | Outcome ΔG |
|---|---|---|---|---|---|
| C10 SimpleCNN | −0.09 | −0.16 [−0.21, −0.11] | opposite | −0.07 | inconclusive |
| C10 LeNet-5 | +1.42 | +0.05 [−0.05, 0.15] | inconclusive | −1.37 | opposite |
| C10 ResNet-8 | +6.51 | +0.07 [0.03, 0.11] | supported | −6.44 | opposite |
| C100 SimpleCNN | +0.32 | +0.03 [−0.02, 0.07] | inconclusive | −0.30 | opposite |
| C100 LeNet-5 | +0.36 | +0.12 [0.06, 0.18] | supported | −0.24 | opposite |
| C100 ResNet-8 | +3.48 | −0.02 [−0.07, 0.03] | inconclusive | −3.50 | opposite |

- **PPO's advantage largely disappears.** Before fine-tuning PPO was ahead of global in 5 settings. ΔG
  is opposite in those 5: global's much larger recovery closes the gap. On CIFAR-10 SimpleCNN, where
  the pre-fine-tuning gap was already near zero, ΔG is inconclusive.
- **What remains** is a small PPO advantage (+0.07 and +0.12 pp) in 2 settings and a small global
  advantage (0.16 pp) in CIFAR-10 SimpleCNN. Every G_post lies inside ±0.5 pp.
- **RQ4 answer.** The advantage of a particular pruning policy does not persist after fine-tuning at a
  practically relevant size.

## 11. Architecture-dependent recovery

H4 compares PPO recovery between architectures within a dataset (unpaired permutation test, Holm over
6 tests):
- **CIFAR-10.** SimpleCNN recovers more than LeNet-5 (+2.24 pp) and ResNet-8 (+2.05 pp), both
  supported. This is driven by the 3 pp dense-control gain of the historical SimpleCNN reference, not
  by pruning. LeNet-5 versus ResNet-8 is inconclusive.
- **CIFAR-100.** ResNet-8 recovers more than SimpleCNN (+0.74 pp) and LeNet-5 (+0.63 pp). Both are
  significant; they are labelled "opposite" only because the comparisons are written SimpleCNN −
  ResNet-8 and LeNet-5 − ResNet-8. SimpleCNN versus LeNet-5 is inconclusive.
- **The paired sensitivity analysis** gives the same six outcomes.
- **Caveat.** Recovery here reflects each architecture together with its own sparsity and its own
  pre-fine-tuning loss, as pre-registered. It is not an effect of architecture alone.

## 12. Accuracy-retention analysis

- **Retention.** Test retention rises for every method and setting, because accuracy rises (H1). With
  zero counts unchanged (H5), H6 holds in all 18 method × setting cells.
- **Retention ≥ 0.98 on test.** Every pruned model meets it after fine-tuning. Before fine-tuning, PPO
  met it in 100% of runs in 3 settings, 60% in C100 LeNet-5, 45% in C10 LeNet-5 and 0% in C100
  ResNet-8.
- **The Phase-5 validation-to-test retention gap of PPO** (retention on VAL-SELECT minus retention on
  test) shrank:
  - supported in 4 settings, by 0.5–2.2 percentage points of retention;
  - inconclusive in C10 LeNet-5 and C100 SimpleCNN.
- **Note on practical labels.** The retention-gap rows are fractions, so the 0.5 pp practical label
  does not apply to them (it would mean 50 points of retention).
- **Against the fine-tuned dense control of the same pair**, mean retention after fine-tuning is
  0.995–1.003.

## 13. Statistical evaluation

All families use exact two-sided sign-flip tests paired by policy seed (n = 20), except H4, which
uses the unpaired permutation test with a paired sensitivity check. Per-test details are in
`phase6_statistics.csv` (mean ± SD, median, t- and bootstrap CIs, d_z, raw and Holm p, wins/ties/losses).

| Family | Size | Supported | Opposite | Inconclusive |
|---|---|---|---|---|
| H1 recovery | 18 | 18 | 0 | 0 |
| Dense control gain | 6 | 6 | 0 | 0 |
| Pruning-specific recovery | 18 | 17 | 1 | 0 |
| H2 PPO vs LAMP: G_post / ΔG | 6 / 6 | 0 / 5 | 3 / 0 | 3 / 1 |
| H3 PPO vs global: G_post / ΔG | 6 / 6 | 2 / 0 | 1 / 5 | 3 / 1 |
| H4 architecture | 6 | 2 | 2 | 2 |
| H6 retention gap (direction: reduction) | 6 | 4 | 0 | 2 |

**Practical relevance (0.5 pp).**
- Every post-fine-tuning PPO–baseline difference (12 of 12) is practically equivalent: its CI lies
  inside ±0.5 pp.
- Practically relevant differences appear only in recovery (where the effects are large), in the
  PPO–global gap changes on C10 LeNet-5 and both ResNet-8 settings, and in the PPO–LAMP gap change
  on C10 ResNet-8.

## 14. Pre-registered criteria A–F

| Criterion | Status | Basis |
|---|---|---|
| A Fairness | **MET** | identical protocol, epochs, seed and data stream within every pair; zero-count mismatch 0; masks equal the frozen table |
| B Recovery | **MET** | recovery supported for all three methods in 6/6 settings (rule: ≥ 2) |
| C PPO competitiveness | **MET** | ΔG toward PPO supported in 5/6, toward LAMP in 0/6 (rule: ≥ 2 and none opposite). LAMP is still ahead after fine-tuning in 3/6, by 0.14–0.22 pp |
| D Architecture generalisation | **MET** | all six settings complete (80 runs each) and present in every family, table and figure |
| E Sparsity preservation | **MET** | per-layer zero counts and masks unchanged in 360/360 pruned runs |
| F Reproducibility | **MET** | 480/480 runs complete; 6/6 reproduction runs bit-identical |

Criterion C concerns the change of the gap, as pre-registered. It does not mean PPO beats LAMP.

## 15. What improved

- **Pruning loss.** Fixed-mask fine-tuning removes essentially all of it for every method. On average
  every method ends within 0.14 pp of the dense reference, with sparsity unchanged.
- **PPO versus LAMP.** The gap closed from 0.26–0.53 pp to 0.00–0.22 pp, and the two methods are
  practically equivalent after fine-tuning in all six settings.
- **Validation-to-test transfer.** Phase 5's retention transfer problem shrinks: every fine-tuned PPO
  model meets retention ≥ 0.98 on test, against 0% before fine-tuning in C100 ResNet-8.
- **The experiment itself** is complete, exactly matched and reproducible (all criteria MET).

## 16. What did not improve

- **LAMP.** PPO does not overtake it anywhere. LAMP remains slightly but significantly ahead in 3
  settings.
- **Global magnitude.** PPO's pre-fine-tuning advantage over global (up to 6.5 pp on ResNet-8) does not
  survive fine-tuning. The policy-quality advantage seen in Phases 3–5 is largely a pre-fine-tuning
  effect at these sparsities.
- **Recoverability.** The masks that lose more accuracy recover more. Global's masks are the most
  recoverable, PPO's next and LAMP's least, so fine-tuning compresses rather than preserves the
  differences between methods.

## 17. Limitations

- **Test set.** It was read in Phases 2–5, so Phase 6 is not a fully independent confirmation. No
  Phase-6 choice used test accuracy.
- **Budget and sparsity.** One schedule (5 epochs, lr 1e-4) at moderate sparsities (16–57%). Larger
  sparsity or a shorter budget could keep larger differences; neither was tested.
- **Repeated policies.** In ResNet-8 the PPO policies are concentrated: 4 distinct per dataset, and one
  policy in 17/20 seeds on CIFAR-100. The cluster-level analysis there is descriptive only.
- **Historical reference.** The CIFAR-10 SimpleCNN reference is the historical checkpoint and was not
  converged: its dense control gains 3 pp. Recovery in that setting mixes pruning recovery with extra
  training. The dense control and the pruning-specific family separate the two.
- **Architecture comparisons** are confounded with each architecture's sparsity and pre-fine-tuning
  loss.
- **Efficiency.** Unstructured sparsity reduces compressed storage only; no latency benefit is
  claimed.
- **Scope.** Six CIFAR-scale settings on one CPU.

## 18. Thesis-level implications

- **Before fine-tuning,** the sensitivity-aware, accuracy-constrained PPO pipeline finds layer-wise
  allocations that clearly beat global magnitude pruning and come close to LAMP.
- **After a modest, identical fine-tuning budget,** the choice of allocation matters little at these
  sparsities: all three methods converge to within 0.26 pp of each other and within 0.14 pp of the
  dense reference.
- **The thesis's value claims** for PPO should therefore be stated for the no-fine-tuning (one-shot)
  regime, or for regimes where fine-tuning is unavailable or expensive. PPO's merit in this
  comparison is its accuracy before any retraining, not a better final model after retraining.

## 19. Supported publication claims

1. Under identical 5-epoch fixed-mask fine-tuning at exactly matched sparsity, PPO, LAMP and global
   magnitude pruning all recover significantly in 6/6 settings, to within 0.14 pp of the dense
   reference on average, without any loss of sparsity.
2. Fine-tuning narrows the PPO–LAMP gap significantly in 5 of 6 settings. After fine-tuning, PPO and
   LAMP are practically equivalent (95% CI within ±0.5 pp) in all six settings. LAMP keeps a small
   significant edge (0.14–0.22 pp) in three.
3. PPO's pre-fine-tuning advantage over global magnitude pruning (up to 6.5 pp) does not persist
   beyond 0.12 pp after fine-tuning.
4. Fine-tuning removes most of the validation-to-test retention shortfall of the accuracy-constrained
   policies.

**Not supported:**
- PPO outperforms LAMP after fine-tuning.
- The PPO policy's advantage persists after fine-tuning.
- Unstructured sparsity speeds up CPU inference.

## 20. Recommended next step

The phase is complete; its claims are stated above.

- **Scope of PPO claims.** Restrict them to the one-shot regime.
- **Optional follow-up**, which would need its own pre-registration: test whether policy quality
  matters again under a smaller budget (1 epoch) or at higher sparsity, where recovery is incomplete.
- **Paper 3.** The audit recommends structured pruning, because masked models gain no latency. This
  is consistent with the finding here that unstructured masks differ little after fine-tuning.
