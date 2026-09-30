# Phase 5 summary: reward–objective alignment through accuracy-constrained pruning

## 1. Completion and integrity

- **Branch and scale.** Branch `phase5-cpu`, 360/360 confirmatory runs: 6 settings × C0/C1/C2 × seeds
  500–519, 1 thread, 2,048 timesteps each. There were no failures, no dropped seeds and no corrupted
  records.
- **Commit sequence.** Every step was committed before the next one began:

  | Commit | Step |
  |---|---|
  | `aff2f3e` | design commitments (before any Phase-5 data) |
  | `fb9d5fb` | split |
  | `3cbaec1` | pre-registration + frozen τ (before any confirmatory run) |
  | `6c4676f` | VAL-RL archives frozen (before any VAL-SELECT evaluation) |
  | `ddaf72c` | VAL-SELECT selection + final-policy list frozen (before any test access) |

- **Checks.** Implementation checks passed 16/16 on seed 42; integrity checks passed 14/14
  (`phase5_integrity_checks.md`).
- **Frozen evidence.** Phase 3 and Phase 4 are unchanged (zero `git diff`), and every dense checkpoint
  matches its hash.

## 2. Validation split correction

- **Split.** The frozen 5,000-image validation set was split deterministically and stratified (seed
  20260930) into **VAL-RL 2,500 / VAL-SELECT 2,500**.
  - Per class the two halves differ by at most 1 image (CIFAR-10 and CIFAR-100).
  - Hashes: `cifar10_phase5_split.npz` `eccd7522…`, `cifar100_phase5_split.npz` `131cf534…`.
- **Use.**
  - VAL-RL alone drove sensitivity, rewards, training and ranks.
  - VAL-SELECT was used only after the archive freeze. The live candidate evaluation equalled the
    VAL-SELECT landscape.
- **Fixing the Phase-4 bias.**
  - There is no data-driven method selection: C0, C1 and C2 were fixed in advance.
  - No inference uses VAL-SELECT metrics of policies selected on VAL-SELECT.
  - Final quality is judged on test.

## 3. Tau diagnostic and frozen threshold

The rule, committed before the diagnostic: take the largest τ ∈ {0.95, 0.97, 0.98, 0.99} for which
every setting has ≥ 20 feasible policies and ≥ 1 feasible policy with ≥ 10% sparsity.
- 0.99 fails: C100 ResNet-8 has only 11 feasible policies.
- **τ = 0.98 frozen.**

| Setting | Feasible | Max feasible sparsity (VAL-RL) |
|---|---|---|
| C10 SimpleCNN | 515 | 59.1% |
| C10 LeNet-5 | 395 | 51.9% |
| C10 ResNet-8 | 46 | 25.3% |
| C100 SimpleCNN | 254 | 54.4% |
| C100 LeNet-5 | 279 | 34.6% |
| C100 ResNet-8 | 21 | 16.1% |

## 4. Accuracy-constrained reward results

C1 uses R = 1 + ρ + 0.01·q if q ≥ 0.98, else −10·(0.98 − q).
- **Constraint on validation.** C1's selected policies satisfy the constraint on VAL-RL in 100% of runs
  and on VAL-SELECT in 99.2% (1 fallback in C100 LeNet-5).
- **Constraint on test.** Measured as test accuracy / dense test accuracy ≥ 0.98, the constraint holds in
  **67.5%** of C1 runs (C2 75.0%, C0 33.3%). By setting:
  - SimpleCNN ×2 and C10 ResNet-8: 100%;
  - C100 LeNet-5: 60%;
  - C10 LeNet-5: 45%;
  - C100 ResNet-8: 0% (mean test retention 97.4%).

  **The validation-certified constraint does not transfer fully to test.** Two 2,500-image splits carry
  roughly ±1 pp of sampling noise.

## 5. ResNet-8 over-pruning analysis

| Setting | Phase-4 F (reference) | C0 (Phase-4 reward, corrected protocol) | C1 |
|---|---|---|---|
| C10 ResNet-8 | 47.8% sparsity, drop 7.86 pp | 48.8%, drop 7.32 pp | **24.7%, drop 1.18 pp** |
| C100 ResNet-8 | 27.7%, drop 4.77 pp | 23.5%, drop 4.05 pp | **16.1%, drop 1.16 pp** |

- **Gain.** C1 − C0 test accuracy is **+6.14 pp** (95% CI 4.92–7.36; 20/20 seeds) and **+2.89 pp**
  (2.30–3.48; 20/20), both supported (Holm p = 1.1e-5).
- **Where C1 lands.** It sits at or next to the constrained optimum: median selected rank 3 and 1; C100
  ResNet-8 selects the oracle in the median run.
- **Conclusion.** The Phase-4 over-pruning is removed.

## 6. Search-quality results

Ranks are the constrained rank on VAL-RL (1 = oracle).
- **C1 median selected rank:** 10.5 / 15 / 3 / 5.5 / 26 / 1 (C10 SimpleCNN, C10 LeNet-5, C10 ResNet-8,
  C100 SimpleCNN, C100 LeNet-5, C100 ResNet-8). That is ≤ 20 in 5/6 settings.
- **Top-20 reach:** 76.7% of C1 runs select a top-20 policy, and 90.8% *visit* one.
- **C0:** under the constrained yardstick its selected policies rank 12–536 (median). It optimises a
  different objective and is often infeasible (median 244 and 59 on ResNet-8).
- **Search quality is retained**, but not to the pre-registered ≥ 80% selected-top-20 level.

## 7. Sensitivity-prior results (C1 vs C2; the same constrained reward)

- **Primary family (24 tests, Holm):**
  - reward-rank AUC is supported in 5/6 settings (C10 ResNet-8 inconclusive);
  - VAL-RL accuracy AUC is supported in 5/6 (C100 ResNet-8 inconclusive);
  - time to the first top-20 and first top-10 is inconclusive in 6/6 (point estimates favour C1);
  - **0 opposite outcomes.**
- **Secondary:**
  - destructive-action rate is lower in 6/6 (supported);
  - the selected rank is better in 3/6;
  - the best-seen rank is better in 1/6.
- **Final test accuracy (descriptive, not pre-registered as a test).** C1 is 0.05–0.16 pp *below* C2 in
  all six settings.
- **Conclusion.** The centred prior still improves exploration once the reward is corrected. It does
  **not** improve final models.

## 8. Final test models (mean test accuracy; dense in brackets)

| Setting | C0 | C1 | C2 | C1 − C0 (95% CI) | Outcome |
|---|---|---|---|---|---|
| C10 SimpleCNN (77.03) | 76.75 @ 56.1% | 76.45 @ 57.6% | 76.50 @ 56.3% | −0.31 (−0.46, −0.16) | opposite |
| C10 LeNet-5 (64.96) | 63.13 @ 54.0% | 63.83 @ 38.4% | 63.99 @ 36.2% | +0.70 (0.51, 0.89) | supported |
| C10 ResNet-8 (78.69) | 71.37 @ 48.8% | 77.51 @ 24.7% | 77.57 @ 24.1% | +6.14 (4.92, 7.36) | supported |
| C100 SimpleCNN (50.50) | 50.28 @ 52.7% | 50.07 @ 53.7% | 50.13 @ 53.2% | −0.21 (−0.33, −0.09) | opposite |
| C100 LeNet-5 (32.28) | 30.67 @ 46.2% | 31.79 @ 31.0% | 31.86 @ 30.8% | +1.12 (0.95, 1.29) | supported |
| C100 ResNet-8 (43.99) | 39.94 @ 23.5% | 42.83 @ 16.1% | 42.90 @ 16.1% | +2.89 (2.30, 3.48) | supported |

Sparsities are medians.
- **Where C1 is more accurate** (LeNet-5, ResNet-8), it prunes 10–22 pp *less* than C0.
- **SimpleCNN.** C1 prunes slightly more (+0.2 to +0.8 pp, inconclusive) and loses 0.2–0.3 pp: it uses
  the 2% accuracy allowance for sparsity.

## 9. Comparison with LAMP (identical zero count; sparsity mismatch 0)

- **C1 − LAMP:** −0.49 / −0.32 / −0.53 / −0.29 / −0.26 / −0.51 pp. LAMP is **significantly better in
  6/6** settings.
- **Gap relative to C0:**
  - smaller in C100 ResNet-8 (+0.66 pp, supported);
  - larger in C10 SimpleCNN (−0.38) and C100 SimpleCNN (−0.12), both opposite;
  - inconclusive elsewhere.

## 10. Comparison with global magnitude (identical zero count)

- **C1 − global:** +1.42 (C10 LeNet-5), +6.51 (C10 ResNet-8), +0.32 (C100 SimpleCNN), +0.36 (C100
  LeNet-5), +3.48 (C100 ResNet-8), all supported.
- **C10 SimpleCNN:** −0.09, inconclusive.

## 11. Cross-architecture behavior

- **Fragile models (ResNet-8, and LeNet-5 on the high-sparsity side).** The constraint is decisive: large
  accuracy gains at the safe sparsity.
- **Robust models (SimpleCNN).** The Phase-4 reward already chose near-safe policies; the constraint
  converts slack into slightly more sparsity at a small accuracy cost.
- **Safe sparsity is genuinely architecture-dependent:** from 16% (C100 ResNet-8) to 59% (C10 SimpleCNN).

## 12. Pre-registered criteria A–F

- **A. Accuracy safety: PARTIALLY MET.**
  - C1 reduces the accuracy loss in **both** ResNet-8 settings (+6.14, +2.89 pp; supported).
  - But it is significantly worse than C0 in **2** other settings (C10 SimpleCNN −0.31, C100 SimpleCNN
    −0.21), and the criterion allows at most 1.
- **B. Non-trivial compression: MET.**
  - C1's median sparsity is ≥ 10% in 6/6 settings: 57.6 / 38.4 / 24.7 / 53.7 / 31.0 / 16.1%.
  - It is ≥ 20% in 5 settings, where 4 were required; ≥ 20% is infeasible in C100 ResNet-8.
- **C. Search quality: PARTIALLY MET.**
  - The median selected rank is ≤ 20 in 5/6 settings (C100 LeNet-5: 26).
  - But only **76.7%** of runs select a top-20 policy, against the ≥ 80% required; 90.8% visit one.
- **D. Sensitivity contribution: MET.**
  - A primary exploration metric is supported in 6/6 settings (rank AUC 5/6, accuracy AUC 5/6).
  - 0 of 24 outcomes are opposite.
- **E. Final model improvement: PARTIALLY MET.**
  - Supported in 4 settings (LeNet-5 ×2, ResNet-8 ×2).
  - But opposite in 2 (SimpleCNN ×2); the criterion requires no degradation.
- **F. LAMP competitiveness.**
  - **Strong: NOT MET.** C1 never beats LAMP, and LAMP is significantly better in 6/6.
  - **Weak: MET.** C1 is within 0.5 pp in 4/6 settings (−0.26 to −0.49); both ResNet-8 settings are at
    −0.51 and −0.53; none is worse than 1.0 pp.

## 13. What Phase 5 fixed

- **Reward misalignment on fragile architectures.** The ResNet-8 losses of 4–8 pp (Phase 4 / C0) fell
  to about 1.2 pp at safe sparsity.
- **Search quality stays high** under a correctly specified objective: the median selected rank is ≤ 20
  in 5/6 settings.
- **Unbiased protocol:**
  - independent VAL-RL and VAL-SELECT;
  - no data-driven method choice;
  - frozen archives and final lists before VAL-SELECT and before test.
- **Centred sensitivity** still improves exploration after the reward correction.

## 14. What still failed

- **No setting beats LAMP.** At matched sparsity LAMP is 0.26–0.53 pp better everywhere.
- **The constraint does not certify test accuracy.** It holds on test in only 67.5% of C1 runs (0% in
  C100 ResNet-8).
- **Robust models lose a little accuracy** (SimpleCNN, −0.2 to −0.3 pp), because τ = 0.98 permits it.
- **Only 76.7%** of runs select a top-20 policy.
- **Sensitivity** does not translate into better final models; C2 is marginally better on test.

## 15. Thesis-level conclusion

- **Alignment is the main lever.** Once the reward encodes the actual goal ("as sparse as possible while
  keeping ≥ 98% accuracy"), the search improvements of Phase 4 become accuracy gains exactly where the
  Phase-4 reward over-pruned. Reward alignment, not search complexity, was the main remaining lever.
- **Limits.** Validation-based constraints do not guarantee test-level retention. LAMP remains the
  strongest one-shot method at matched sparsity, with an RL gap of about 0.3–0.5 pp.

## 16. Supported paper claims

- **ResNet-8.** An accuracy-constrained RL objective prevents over-pruning of fragile networks:
  +6.14 / +2.89 pp over the Phase-4 reward.
- **Compression.** It keeps non-trivial, architecture-appropriate compression (16–58% median sparsity).
- **Search.** Archive-based PPO reaches near-optimal constrained policies (median rank ≤ 20 in 5/6).
- **Exploration.** Centred sensitivity improves exploration (rank or accuracy AUC in 6/6) and reduces
  destructive actions (6/6) under the constrained reward.
- **Global magnitude.** The constrained RL pipeline beats matched global magnitude in 5/6 settings.
- **LAMP.** It is within 0.5 pp of matched LAMP in 4/6 settings.

## 17. Unsupported claims

- RL beats LAMP.
- The constraint guarantees ≥ 98% accuracy retention on test.
- Sensitivity guidance improves final accuracy.
- The constrained reward improves accuracy on every architecture (SimpleCNN degrades slightly).
- ≥ 80% of runs select a top-20 policy.

## 18. Recommended next step

The most direct remaining weakness is the **validation-to-test transfer of the accuracy constraint**. A
pre-registered next step would be one of two options:
1. **Safety margin.** Select on VAL-SELECT with a margin (e.g. require q_SELECT ≥ τ + δ, with δ fixed
   from the validation sampling error).
2. **Confidence bound.** Use a confidence-bound feasibility test on VAL-SELECT.

Either could run on the existing Phase-5 runs and archives (selection only, no retraining), with test
still read once per frozen list.

Separately, a small fine-tuning study of the selected constrained policies is the natural way to test
whether the remaining 0.3–0.5 pp gap to LAMP closes after brief retraining. It needs a new
pre-registration.
