# QA report: Chapter 4 (Experimental Setup)

- **File:** `chapter_04_experimental_setup.md`.
- **Status:** **FROZEN (v1, 2026-11-05)**, pending the user's review.
- **Support files:** `experimental_design_audit_chapter_04.md`; `tools/audit_ch4.py` (source checks, 100/100 PASS).

## 1. Word count

**9,320** counted words. That is about **7,300 words of prose** plus **about 2,000 words inside tables**, file names
and hashes. The target was 5,000–7,000 "depending on tables". The prose meets it; the table material is the
reproducibility detail requested in task §§18–31.

A first draft of 11,777 words was condensed, keeping every design rule. The estimate is about 22 pages at thesis
formatting.

## 2. Subsections

The 20 sections follow the requested structure:

| § | Title |
|---|---|
| 4.1 | Experimental Design Overview |
| 4.2 | Datasets |
| 4.3 | Dataset Partitioning and Validation Protocol (4.3.1–4.3.3; Tables 4.1, 4.2) |
| 4.4 | Neural Network Architectures (Table 4.3) |
| 4.5 | Dense Reference Models (Table 4.4) |
| 4.6 | Pruning Units and Policy Spaces (4.6.1 masked, 4.6.2 structured, 4.6.3 enumeration) |
| 4.7 | Baseline Pruning Methods (4.7.1 core, 4.7.2 extension, 4.7.3 matching; Table 4.5) |
| 4.8 | PPO Training Configuration (Table 4.6) |
| 4.9 | Experimental Conditions by Phase (Table 4.7; 4.9.1–4.9.8) |
| 4.10 | Sensitivity Conditions and Controls |
| 4.11 | Archive and Selection Protocols |
| 4.12 | Reward Configurations |
| 4.13 | Recovery and Fine-Tuning Protocol |
| 4.14 | Evaluation Metrics (Table 4.8) |
| 4.15 | Statistical Analysis and Decision Criteria (4.15.1–4.15.4; Table 4.9) |
| 4.16 | Hardware and Software Environment (Table 4.10) |
| 4.17 | Structured Pruning and Latency Protocol |
| 4.18 | Energy Measurement Protocol (4.18.1–4.18.5) |
| 4.19 | Reproducibility and Pre-Registration (including test-set governance) |
| 4.20 | Chapter Summary |

**Figures.** None were added. F1 is not duplicated. The split and test-use information is carried by Tables 4.1 and
4.2, so a schematic would add no information.

## 3. Dataset definitions

- CIFAR-10: 10 classes. CIFAR-100: 100 fine labels. Both are 32×32 RGB, with 50,000 training and 10,000 test images.
- Normalisation constants come from `src/data` and were computed on the training file only.
- Augmentation (crop 32 with pad 4, plus horizontal flip) is used for training only.
- Usage by dataset:
  - both datasets: Phases 2–6 and the Paper 3 latency tables;
  - CIFAR-10 only: the Paper 3 accuracy and proxy work, 7A, 7B and the energy confirmation.

## 4. Split definitions by phase

- **Training / validation:** 45,000 / 5,000 (randperm, seed 42), plus the official 10,000 test images. CIFAR-10 and
  CIFAR-100 use the same index lists.
- **Configuration I** (Phases 3–4; records call it V_RL / V_SELECT): 3,000 / 2,000, stratified, seed 20260928.
- **Configuration II** (Phase 5 onward): 2,500 / 2,500, stratified, seed 20260930, per-class imbalance ≤ 1.
- **Why it changed:** the Phase-4 Stage-C selection bias, recorded in the Phase-5 design commitments before any data.
  Of the old V_RL images, 1,521 (CIFAR-10) and 1,515 (CIFAR-100) fall into the new VAL-SELECT, so sensitivity was
  recomputed.

Table 4.2 gives, for every phase:
- which split drives the reward;
- which split selects;
- whether and when the test set is read.

## 5. Architectures

All three are taken from `src/models`:
- **SimpleCNN:** conv 32-64-128, then FC 256 and C; no batch normalisation.
- **LeNet-5, CIFAR adaptation:** 5×5 valid convolutions with 6 and 16 filters, ReLU and max-pooling, then
  400-120-84-C.
- **ResNet-8:** n = 1, with a 16-channel stem and stages of 16, 32 and 64 channels; projection shortcuts in stages
  2–3.

Parameter counts and weight totals were checked against `cpu_baselines.csv` and `cpu_prunable_units.csv`. The
SmallVGG fallback is noted as unused.

## 6. Pruning units

- **Masked:** four per architecture, with the shares from `cpu_prunable_units.csv`.
- **Structured:** SimpleCNN filters and neurons. For ResNet-8:
  - the stem is a **coupled residual pruning group**;
  - stages 1–3 are **block-internal channel groups**.
- §4.6.2 states that the masked and structured ResNet-8 units are different objects.

## 7. Baseline methods

- **Core:** Uniform, Random (10 masks), Global magnitude, ERK and LAMP, with formulas from `src/pruning`.
- **Extension:** Uniform-Structured, LAMP-Structured, old-PPO structured translation and new structured PPO, kept
  separate.
- The methods are not collectively called heuristics.

## 8. Matching rules

Table 4.5 covers ten comparison rows:
- **identical zero count** in Phases 3–6;
- **MAC matching** for Uniform in 7A and 7B;
- **zero count derived from the Phase-6 mask, then structured,** for LAMP in 7A;
- **parameter-count matching** for LAMP in 7B;
- **unmatched** old PPO.

Integer channel counts make every extension match approximate, and the table says so.

## 9. PPO hyper-parameters

Table 4.6 covers Phases 3, 4, 5 and 7B. The SB3 hyper-parameters are identical in all four phases. The differences in
state, prior, reward, cache, extra loss, returned policy and seeds are shown explicitly.

Phase-4 elite behavioural cloning (K = 5, η = 0.1) is documented, with:
- its role: retention during training;
- its verification: η = 0 reproduces standard PPO bit for bit;
- its removal in Phase 5.

## 10. Seeds per phase

| Phase | Seeds | Runs |
|---|---|---|
| 2 (dense) | 0–2 | — |
| 3 | 300–319 | 480 |
| 4 | 400–419 | 1,200 |
| 5 | 500–519 | 360 |
| 6 | fine-tuning seeds 600–619 | 480 + 6 |
| Paper 3, 7A | recovery seed 600 | — |
| 7B | development 900–904; confirmatory 700–719 | 20 confirmatory searches |

Every count was verified by row counts of the frozen CSV files (source check, item 19).

## 11. Recovery protocol

- Adam 1e-4, β = (0.9, 0.999), ε = 1e-8, weight decay 0.
- Cosine annealing to 0 per epoch over E = 5 epochs, set by the timing rule (projected 29.44 h ≤ 30 h).
- Batch 128, crop + flip, BN in train mode, no selection.
- Fixed mask asserted every epoch (Phase 6), or a shape assertion for compacted models.
- Seed 600 for compacted models.
- The proxy variants are distinguished: the Paper 3 1-epoch proxy is the first epoch of the full run; the 7B 2-epoch
  proxy is a separate run.

## 12. Metrics

Table 4.8 gives every metric with a symbol, definition, phases and direction. The groups are:
- predictive, compression, computation, deployment, energy and search;
- search metrics include rank, rank score, constrained rank, best-seen and selected rank, top-k reach, first entry,
  trajectory AUCs, destructive frequency, frontier regret and feasible-sample rate.

§4.14 separates **four notions of compression**, so that "sparsity" is never used as a universal synonym:
1. masked weight sparsity;
2. channel removal ratio;
3. physical parameter reduction;
4. MAC reduction.

## 13. Statistical tests

Every test was verified against its pre-registration:

| Phase | Tests |
|---|---|
| 3–6 | exact sign-flip (2²⁰), 95% t-CI, bootstrap, d_z, Holm per family |
| 3–4 | Wilcoxon p also reported; mechanical classes |
| 3 | McNemar (supportive only) |
| 4 | Stage-A bootstrap; formal one-sided non-inferiority test (0.5 pp) |
| 5 | C1 − global descriptive |
| 6 | ±0.5 pp CI-inclusion rule, **explicitly not a formal equivalence test**; unpaired permutation (100,000) with Welch CI for H4; cluster sensitivity analysis |
| Paper 3 | Spearman correlations and pair fractions; SNR rule |
| 7A / 7B | exact one-sided Wilcoxon (n = 5) plus consistency and noise rules |
| Energy confirmation | Wilcoxon and sign test pre-registered; t-test excluded |

No test was invented.

## 14. Pre-registration rules

Table 4.9 has 12 rows: Phase 2, Phases 3–6, Paper 3 H0, Paper 3 H2 and proxy, Paper 3 ResNet-8, Phase 7A, Phase 7B
Stage A, Phase 7B Stage B, and the energy confirmation. Each row gives the governing file and the commit that first
added it, checked with `git log --diff-filter=A`.

## 15. Hardware and software environments

Table 4.10 compares the two machines:

| Machine | Used for | Details |
|---|---|---|
| Desktop, i5-6400, 4 logical CPUs, Windows 11 build 22621 | Phases 2–6 | from the `phase5/6_environment.json` and Phase-3 integrity records |
| Laptop, i7-1260P, 15.7 GB, Windows 11 Home build 26200 | Paper 3, 7A, 7B, energy confirmation | oneDNN 3.7.1 and MKL 2025.1 from `h2_latency_environment.json` |

**Software.** Python 3.11.9 and torch 2.7.1+cpu on both machines. On the desktop: SB3 2.8.0, sb3-contrib 2.8.0,
Gymnasium 1.2.3 and NumPy 2.2.2.

**Not inferred.** Desktop RAM is "not recorded". The SB3 version for 7B is stated as that of the pinned environment,
not as a recorded run value.

## 16. Energy protocol

- **Backend:** RAPL package counter via the EMI/PDH counter; pWh converted at 3.6e-9 J; PP0 secondary; DRAM reads 0
  and is unused; no TDP model or estimate.
- **Phase 7A:** warm-up 1,000; N = 100,000; 5 sessions; 30 s idle; pilot CV ≤ 3%, met.
- **7B surrogate:** 200 calibration policies; 3 sessions; N = 20,000; pilot ≤ 4%, met; four candidates in order;
  three criteria; candidate 2 accepted.
- **7B confirmatory pilot:** 11.36% at N = 100,000; doubled-N attempt void; rerun 4.01% at N = 200,000; then STOP,
  with no session.
- **Energy confirmation:** 180 s soak, N = 200,000, qualification 19.92% → STOP before any paired measurement.
- **No Phase-7B energy reduction is reported.**

## 17. Test-set governance

- Table 4.2 gives test use by phase, and §4.19 adds the code guard, the freeze order and two disclosures:
  - Phase 6 is not a fully independent confirmation;
  - validation-to-test transfer is measured in Phase 5 (feasibility on three splits) and Phase 6 (retention gap).
- Paper 3, 7A and 7B never read the test set.
- No test-informed selection is implied anywhere.

## 18. Citations

13 distinct references, all verified (0 unknown keys):

| Group | References |
|---|---|
| Datasets | `krizhevsky2009cifar` |
| Architectures | `lecun1998lenet`, `he2016resnet` |
| Baselines | `blalock2020state`, `evci2020rigl`, `lee2021lamp` |
| PPO and tools | `schulman2017ppo`, `schulman2016gae`, `raffin2021sb3`, `paszke2019pytorch` |
| Energy measurement | `khan2018rapl`, `jay2023powermeters`, `raffin2024dissecting` |

- **First used in Chapter 4:** `krizhevsky2009cifar`, `lecun1998lenet` and `he2016resnet`. All three are among the 97
  references verified in Milestone 1. The recent share is low (15%) because a setup chapter cites the foundational
  sources of its datasets, models and tools. `schulman2017ppo` is the allowed arXiv exception.
- **Frozen reference audit untouched.** `reference_audit.md` was not regenerated; the cumulative audit is deferred to
  Stage 10.
- **Candidates to verify at Stage 10.** Holm's step-down procedure, the Wilcoxon signed-rank test, Youden's J and the
  ERK origin are named without new citations. No reference was added outside the verified set. Holm (1979), Wilcoxon
  (1945) and Youden (1950) are candidates for Stage-10 verification.

## 19. Numerical and source checks

`tools/audit_ch4.py` re-reads every Chapter-4 number from the frozen files: **100 checks, 100 PASS, 0 FAIL** (table
below).

**Checked manually against the pre-registration text:**
- the Phase-3 condition definitions;
- the Phase-4 R1a/R1b/R2 forms and the κ rule;
- the Phase-5 C0 selection rule;
- the Phase-6 excluded methods;
- the 7A cases and comparison rule;
- the 7B joint criterion, matching grids, calibration-set design and acceptance order;
- the energy-confirmation validity and confirmation rules.

**Disclosed as values** (fixed design constants or pre-registered stopping outcomes, not results):
- Phase-4 stage outcomes;
- τ = 0.98;
- E = 5 (29.44 h);
- τ̂ = 0.98849, α = 4;
- surrogate acceptance;
- pilot and qualification CVs.

**Kept out** (results for Chapters 5 and 6): Paper 3 correlations and speed-ups, proxy and surrogate statistics, and all
accuracy, latency and energy outcomes.

**Dangerous-phrase scan** (`tools/qa_chapter.py`; every hit inspected):
- **"masked-equivalent model":** a numerical check of logits. Benign.
- **"equivalent" in §4.15.2 and §4.20:** only in explaining that the ±0.5 pp rule is *not* a formal equivalence test.
  Allowed by task §28.
- **"heuristics":** only in the sentence saying the baselines are *not* collectively called heuristics.
- **"lower energy":** inside the definitions of the 7A criteria. Benign.
- **"GREEN-SUCCESS" / "green":** only as the literal pre-registered internal name, introduced once and replaced by
  "joint deployment-efficiency criterion".
- **"carbon":** only in "not a carbon metric".
- **Manual grep for result values:** 22.8, 13.4, 63.4, 36.0, 1.12, 0.844, 1.098, 2.61, 0.948 and 0.996 do not appear.
  There are also no "outperform", "stronger" or "beats" statements.

## 20. Unresolved issues

### A. Chapter-3 wording (reported, not edited)

- **Frozen §3.6, line 261:** "Action 3 on unit 2 prunes a single convolution in SimpleCNN and an entire residual stage
  in ResNet-8 (Table 3.3)".
- **What the code does** (`src/pruning`, `cpu_prunable_units.csv`): the ResNet-8 unit 2 is the stage-1 weight group.
  Action 3 masks 30% of the weights of each of its two convolutions, unstructured. No stage, block or channel is
  removed.
- **Assessment.** The sentence is imprecise rather than a wrong equation or definition. Table 3.3 on the same page
  lists the tensors correctly, and §3.3 says "one action prunes several tensors at once". The word "entire" can still
  be misread as removing the stage.
- **Proposed correction**, awaiting the user's decision:
  > "Action 3 on unit 2 masks 30% of the weights of a single convolution in SimpleCNN, but 30% of the weights of each
  > convolution in the first residual stage of ResNet-8 (Table 3.3)."
- Chapter 4 uses the precise wording ("residual-stage weight group"; "coupled residual pruning group").

### B. Chapter-3 note, minor (no change needed)

§3.18 item 1 says all 1,296 policies are evaluated "on the reward split and once on the selection split" in every
setting. Phase 3 had no VAL-SELECT landscape; it was added in Phase 4. The statement is true of the mature framework
that §3.18 describes. Chapter 4 §4.6.3 gives the per-phase history.

### C. Task-brief clarification

Task §23 calls the first Phase-7B energy pilot "CV = 4.01%". The frozen record shows two pilot results:
- 11.36% at N = 100,000;
- 4.01% at the single permitted N = 200,000 rerun, after one void attempt.

Chapter 4 reports both, as in `result_freeze.md` §9.

### D. Word count

The count exceeds 7,000 only because of table content (see item 1).

### E. Template

Still blocked: no official thesis template, as reported in Milestone 1.

## Consistency with Chapter 3 (task §40)

| Item | Chapter 3 | Chapter 4 | Consistent? |
|---|---|---|---|
| Action space | {0, .1, .2, .3, .4, .6}, 1,296 policies (3.5–3.6) | same (4.6) | yes |
| PPO state | 7 features, slot 0 in Phases 3–5; 8 features in 7B (Table 3.4) | Table 4.6, 4.10 | yes |
| Sensitivity prior | c_l = 0.5·S_l, β = 0.5 (3.10) | 4.9.1 | yes |
| Centred prior | c_l = 1.0·(S_l − S̄) (3.11) | 4.9.2, Table 4.6 | yes |
| Rewards | R0 (3.7), R_c (3.15), R_7B (3.19) | 4.12 (adds the Phase-4 R1a, R1b, R2, which Chapter 3 mentions only in words) | yes |
| Archive rules | (3.12), (3.14), 7B feasible archive | 4.11 | yes |
| Elite behavioural cloning | K = 5, η = 0.1, Phase 4 only (3.13) | 4.8, 4.9.2 | yes |
| Recovery | Phase-6 protocol, E = 5, seed + 100, seed 600 (3.15) | 4.13 | yes |
| Structured semantics | keep max(1, round((1 − r)C)), coupled stem, block-internal stages (3.16.1) | 4.6.2 | yes |
| Data roles | VAL-RL / VAL-SELECT / test (3.18) | 4.3, Table 4.2 | yes; Chapter 4 adds the two configurations promised by Chapter 3 §3.8 |
| ResNet-8 masked unit wording | §3.6 line 261 "entire residual stage" | "residual-stage weight group" | **wording issue A**, reported, not edited |

**No equation or methodological definition contradicts Chapter 3.**

## ResNet-8 wording check (task §41)

| Text | Implementation | Chapter-4 wording |
|---|---|---|
| masked stage unit | L1 unstructured mask at ratio r on each conv tensor of the block (and its 1×1 projection) | "residual-stage weight group … removes no stage, block or channel" |
| structured stem | one shared channel set across every writer and reader of the stage-1 residual stream | "coupled residual pruning group" |
| structured stages | block-internal channels (conv1 outputs + bn1, conv2 inputs) | "block-internal channel groups" |

"Whole/entire residual stage" does not occur in Chapter 4.

## Source-check table (`tools/audit_ch4.py`)

**100 checks, 100 PASS, 0 FAIL.**

| # | Check | Chapter 4 | Source value | Source | Result |
|---|---|---|---|---|---|
| 1 | C100 train/val/V_RL/V_SELECT/test | 45000/5000/3000/2000/10000 | 45000/5000/3000/2000/10000 | `reproducibility/cifar100_split_manifest.json` | PASS |
| 2 | train/val rule seed 42 | 42 | 42 | `cifar100_split_manifest.json split_rule` | PASS |
| 3 | config-I stratified seed | 20260928 | 20260928 | `cifar100_split_manifest.json` | PASS |
| 4 | identical index lists C10/C100 | True | True | `cifar100_split_manifest.json` | PASS |
| 5 | C10 config-I sizes | [2000, 3000] | [2000, 3000] | `archive_validation_split.npz` | PASS |
| 6 | config-II seed | 20260930 | 20260930 | `phase5/splits/phase5_split_manifest.json` | PASS |
| 7 | config-II sizes cifar10 | [2500, 2500] | [2500, 2500] | `phase5_split_manifest.json` | PASS |
| 8 | config-II per-class imbalance cifar10 | 1 | 1 | `phase5_split_manifest.json` | PASS |
| 9 | config-II sizes cifar100 | [2500, 2500] | [2500, 2500] | `phase5_split_manifest.json` | PASS |
| 10 | config-II per-class imbalance cifar100 | 1 | 1 | `phase5_split_manifest.json` | PASS |
| 11 | old V_RL images in new VAL-SELECT (C10 / C100) | 1521 / 1515 | 1521 / 1515 | `phase5_split_manifest.json overlap_with_old_V_RL` | PASS |
| 12 | normalisation (0.4914, 0.4822, 0.4465) | (0.4914, 0.4822, 0.4465) | (0.4914, 0.4822, 0.4465) | `src/data/__init__.py DATASETS` | PASS |
| 13 | normalisation (0.2470, 0.2435, 0.2616) | (0.2470, 0.2435, 0.2616) | (0.2470, 0.2435, 0.2616) | `src/data/__init__.py DATASETS` | PASS |
| 14 | normalisation (0.5071, 0.4865, 0.4409) | (0.5071, 0.4865, 0.4409) | (0.5071, 0.4865, 0.4409) | `src/data/__init__.py DATASETS` | PASS |
| 15 | normalisation (0.2673, 0.2564, 0.2762) | (0.2673, 0.2564, 0.2762) | (0.2673, 0.2564, 0.2762) | `src/data/__init__.py DATASETS` | PASS |
| 16 | Table 4.4 cifar10_simplecnn | ca28f834 77.32 77.03 76.9667 76.88 | ca28f834 77.32 77.03 76.9667 76.88 | `cpu_baselines.csv; phase3_frozen_inputs.json; phase5_config.json` | PASS |
| 17 | Table 4.4 row in chapter cifar10_simplecnn | present | present | `chapter text` | PASS |
| 18 | Table 4.4 cifar10_lenet5 | 5ca6e0fc 64.32 64.96 64.3000 64.20 | 5ca6e0fc 64.32 64.96 64.3000 64.20 | `cpu_baselines.csv; phase3_frozen_inputs.json; phase5_config.json` | PASS |
| 19 | Table 4.4 row in chapter cifar10_lenet5 | present | present | `chapter text` | PASS |
| 20 | Table 4.4 cifar10_resnet8 | 41fc5ee5 78.72 78.69 78.5333 78.92 | 41fc5ee5 78.72 78.69 78.5333 78.92 | `cpu_baselines.csv; phase3_frozen_inputs.json; phase5_config.json` | PASS |
| 21 | Table 4.4 row in chapter cifar10_resnet8 | present | present | `chapter text` | PASS |
| 22 | Table 4.4 cifar100_simplecnn | d75dd19c 49.08 50.50 50.3667 48.96 | d75dd19c 49.08 50.50 50.3667 48.96 | `cpu_baselines.csv; phase3_frozen_inputs.json; phase5_config.json` | PASS |
| 23 | Table 4.4 row in chapter cifar100_simplecnn | present | present | `chapter text` | PASS |
| 24 | Table 4.4 cifar100_lenet5 | 8e1b1aee 32.58 32.28 33.3000 32.76 | 8e1b1aee 32.58 32.28 33.3000 32.76 | `cpu_baselines.csv; phase3_frozen_inputs.json; phase5_config.json` | PASS |
| 25 | Table 4.4 row in chapter cifar100_lenet5 | present | present | `chapter text` | PASS |
| 26 | Table 4.4 cifar100_resnet8 | 8b84bfad 43.48 43.99 43.7667 43.76 | 8b84bfad 43.48 43.99 43.7667 43.76 | `cpu_baselines.csv; phase3_frozen_inputs.json; phase5_config.json` | PASS |
| 27 | Table 4.4 row in chapter cifar100_resnet8 | present | present | `chapter text` | PASS |
| 28 | dense parameters simplecnn | 620,362 / 643,492 | 620,362 / 643,492 | `cpu_baselines.csv params_total` | PASS |
| 29 | weights (prunable) simplecnn | 619,872 / 642,912 (617,312) | 619,872 / 642,912 (617,312) | `cpu_prunable_units.csv` | PASS |
| 30 | unit shares simplecnn | 0.14 2.99 11.94 84.93 | 0.14 2.99 11.94 84.93 | `cpu_prunable_units.csv` | PASS |
| 31 | dense parameters lenet5 | 62,006 / 69,656 | 62,006 / 69,656 | `cpu_baselines.csv params_total` | PASS |
| 32 | weights (prunable) lenet5 | 61,770 / 69,330 (60,930) | 61,770 / 69,330 (60,930) | `cpu_prunable_units.csv` | PASS |
| 33 | unit shares lenet5 | 0.74 3.94 78.78 16.54 | 0.74 3.94 78.78 16.54 | `cpu_prunable_units.csv` | PASS |
| 34 | dense parameters resnet8 | 78,042 / 83,892 | 78,042 / 83,892 | `cpu_baselines.csv params_total` | PASS |
| 35 | weights (prunable) resnet8 | 77,360 / 83,120 (76,720) | 77,360 / 83,120 (76,720) | `cpu_prunable_units.csv` | PASS |
| 36 | unit shares resnet8 | 0.56 6.01 18.69 74.74 | 0.56 6.01 18.69 74.74 | `cpu_prunable_units.csv` | PASS |
| 37 | ResNet-8 masked stage units | stage1 = 2 convs; stage2/3 = 2 convs + 1x1 | stem=conv1; stage1=layer1.conv1;layer1.conv2; stage2=layer2.conv1;layer2.conv2;layer2.shortcut.0; stage3=layer3.conv1;layer3.conv2;layer3.shortcut.0 | `cpu_prunable_units.csv modules` | PASS |
| 38 | runs phase3_all_runs.csv | 480 | 480 | `results/architecture_generalization/phase3_all_runs.csv` | PASS |
| 39 | runs phase4_all_runs.csv | 1200 | 1200 | `results/phase4/phase4_all_runs.csv` | PASS |
| 40 | runs phase5_all_runs.csv | 360 | 360 | `results/phase5/phase5_all_runs.csv` | PASS |
| 41 | runs phase6_all_runs.csv | 480 | 480 | `results/phase6/phase6_all_runs.csv` | PASS |
| 42 | runs ppo_all_seeds.csv | 20 | 20 | `results/phase7b_green_pruning/ppo_all_seeds.csv` | PASS |
| 43 | seeds phase3_all_runs.csv | 300-319 | 300-319 (20) | `results/architecture_generalization/phase3_all_runs.csv` | PASS |
| 44 | seeds phase4_all_runs.csv | 400-419 | 400-419 (20) | `results/phase4/phase4_all_runs.csv` | PASS |
| 45 | seeds phase5_all_runs.csv | 500-519 | 500-519 (20) | `results/phase5/phase5_all_runs.csv` | PASS |
| 46 | seeds ppo_all_seeds.csv | 700-719 | 700-719 (20) | `results/phase7b_green_pruning/ppo_all_seeds.csv` | PASS |
| 47 | Phase 6 fine-tuning seed offset | 100 | 100 | `phase6_config.json` | PASS |
| 48 | Phase 6 E | 5 | 5 | `phase6_config.json` | PASS |
| 49 | Phase 6 projected hours (E = 5) | 29.44 | 29.44 | `phase6_config.json epoch_rule` | PASS |
| 50 | Phase 6 limit (hours) | 30 | 30 | `phase6_config.json` | PASS |
| 51 | Phase 6 matched triples | 120 | 120 | `phase6_matched_masks.csv` | PASS |
| 52 | tau | 0.98 | 0.98 | `phase5_config.json` | PASS |
| 53 | eps, kappa | 0.01, 10.0 | 0.01, 10.0 | `phase5_config.json constants` | PASS |
| 54 | Phase 4 stage winners | R0, P2, S2A | R0, P2, S2A | `phase4_stage_selection.json` | PASS |
| 55 | Phase 4 Stage-B eligible priors | P1, P2 | P1, P2 | `phase4_stage_selection.json (tie -> larger mean d_z)` | PASS |
| 56 | Phase 4 selections: test not accessed | False x3 | False False False | `phase4_stage_selection.json` | PASS |
| 57 | latency-table shapes (4 x 1,296) | 5184 | 5184 | `paper3/h2_latency_table_*.csv` | PASS |
| 58 | table protocol batch/threads/cpu/warm-up/timed | 1/1/[2]/50/300 | 1/1/[2]/50/300 | `h2_latency_freeze.json` | PASS |
| 59 | sessions per table; anchor every | 5; 25 | 5; 25 | `h2_latency_freeze.json` | PASS |
| 60 | rejected session attempts | 4 | 4 | `h2_latency_freeze.json session_log` | PASS |
| 61 | accepted sessions, all drift <= 10% | 20, True | 20, True | `h2_latency_freeze.json` | PASS |
| 62 | table vs H0 dense latency (+16%) | 16 | 16 | `h2_latency_freeze.json failed check detail` | PASS |
| 63 | proxy validation pairs (SimpleCNN / ResNet-8) | 156 / 31 | 156 / 31 | `paper3_proxy_decision.json` | PASS |
| 64 | proxy GO (SimpleCNN / ResNet-8) | False / True | False / True | `paper3_proxy_decision.json` | PASS |
| 65 | ResNet-8 truly feasible pairs (< 3 -> Youden fallback) | <3 | 2 | `paper3_proxy_decision.json` | PASS |
| 66 | ResNet-8 calibration policies | 48 | 48 | `resnet8_threshold_calibration_policies.json` | PASS |
| 67 | H1 checks | 50/50 | 50/50 | `paper3/h1_checks.json` | PASS |
| 68 | H2-C planned / completed | 5,184 / 126 | 5,184 / 126 | `paper3/h2c_stop_record.md` | PASS |
| 69 | 7A warm-up / N / sessions | 1000 / 100000 / 5 | 1000 / 100000 / 5 | `phase7a preregistration.json` | PASS |
| 70 | 7A pilot rule 3% | 3% | 3% | `phase7a preregistration.json` | PASS |
| 71 | 7A pilot met the rule at N = 100,000 | met | met | `energy_pilot/pilot_n100000.json` | PASS |
| 72 | 7A recovery seed / generator | 600 / 1000600 | 600 / 1000600 | `phase7a preregistration.json` | PASS |
| 73 | 7A accuracy rule | 0.5 pp VAL-RL | 0.5 pp VAL-RL | `phase7a preregistration.json` | PASS |
| 74 | 7A models | 8 | 8 | `accuracy_results.csv` | PASS |
| 75 | RAPL unit conversion | 1 pWh = 3.6e-9 J | 1 pWh = 3.6e-9 J | `phase7a preregistration.json energy.backend` | PASS |
| 76 | DRAM domain reads 0 | not used | not used | `phase7a environment.json` | PASS |
| 77 | 7B tau-hat | 0.98849 | 0.98849 | `phase7b_final_preregistration.json` | PASS |
| 78 | 7B alpha, beta, kappa, eps | 4.0, 1.0, 10.0, 0.01 | 4.0, 1.0, 10.0, 0.01 | `phase7b_final_preregistration.json` | PASS |
| 79 | 7B dev / confirmatory seeds | 900-904 / 700-719 | 900-904 / 700-719 | `phase7b_final_preregistration.json` | PASS |
| 80 | 7B surrogate coefficients | -0.117, 13.762 | -0.117, 13.762 | `phase7b_final_preregistration.json` | PASS |
| 81 | 7B energy N / warm-up / idle / pause / dense blocks | 100000/1000/30/5/3 | 100000/1000/30/5/3 | `phase7b_final_preregistration.json` | PASS |
| 82 | 7B pilot max CV | 0.03 | 0.03 | `phase7b_final_preregistration.json` | PASS |
| 83 | surrogate calibration policies | 200 | 200 | `energy_surrogate_metrics.json` | PASS |
| 84 | surrogate candidate 1 passed / accepted model | False / 2 | False / 2 | `energy_surrogate_metrics.json` | PASS |
| 85 | surrogate criteria | 0.95 / 0.05 / 0.05 | 0.95 / 0.05 / 0.05 | `energy_surrogate_metrics.json` | PASS |
| 86 | calibration pilot (N 20,000; limit 4%) | stable, <= 0.04 | stable, 0.0347 <= 0.04 | `_energy_calibration/pilot_n20000.json` | PASS |
| 87 | 7B pilot CV at N = 100,000 | 11.36% | 11.36% | `_energy_final/pilot_n100000.json` | PASS |
| 88 | 7B pilot CV at N = 200,000 (rerun) | 4.01% | 4.01% | `_energy_final/pilot_n200000.json` | PASS |
| 89 | void doubled-N attempt kept separately | present | present | `_energy_final/void_pilot_n200000_20261027/` | PASS |
| 90 | 7B final recovered models | 11 | 11 | `final_summary.csv` | PASS |
| 91 | proxy validation policies; accepted epochs | 49; 2 | 49; 2 | `accuracy_proxy_metrics.json` | PASS |
| 92 | EC qualification CV | 19.92% | 19.92% | `measurement_qualification.json` | PASS |
| 93 | EC runs x blocks; N; warm-up; soak | 3x2; 200000; 2000; 180 | 3x2; 200000; 2000; 180 | `measurement_qualification.json` | PASS |
| 94 | EC decision | not qualified | not qualified | `measurement_qualification.json` | PASS |
| 95 | EC planned sessions | 10 | 10 | `phase7b_energy_confirmation/protocol.json` | PASS |
| 96 | EC verified checkpoints | 5 | 5 | `checkpoint_verification.json` | PASS |
| 97 | desktop CPU | i5-6400 | i5-6400 | `phase5_environment.json` | PASS |
| 98 | desktop packages | torch 2.7.1+cpu, SB3 2.8.0, gymnasium 1.2.3, numpy 2.2.2 | torch 2.7.1+cpu, SB3 2.8.0, gymnasium 1.2.3, numpy 2.2.2 | `phase5_environment.json` | PASS |
| 99 | laptop CPU / RAM / build | i7-1260P / 15.7 GB / 26200 | i7-1260P / 15.7 GB / 26200 | `phase7b_energy_confirmation/environment.json` | PASS |
| 100 | oneDNN 3.7.1, MKL 2025.1 | present | present | `paper3/h2_latency_environment.json` | PASS |

## Automated QA output (`tools/qa_chapter.py`)

### Automated checks for `chapter_04_experimental_setup.md`

- Words (excluding citation keys and code blocks): **9320** (≈ 22.2 pages at ~420 words/page)
- Citation occurrences: 13; distinct references: **13**
- Unverified / unknown keys: **0** 
- Recent (2022–2026): **2**; foundational (pre-2022): **11** → recent share 15%
- arXiv-only references: ['schulman2017ppo']
- Tables referenced: ['3.4', '3.7', '4.1', '4.10', '4.2', '4.3', '4.4', '4.5', '4.6', '4.7', '4.8', '4.9']; figures referenced: none

### Dangerous-phrase scan (each hit inspected manually)

| Pattern | Line | Context |
|---|---|---|
| `equivalen(t|ce)` | 261 | …- logits agree with the masked-equivalent model within 10⁻⁴;… |
| `heuristic` | 313 | …They are referred to by name, not collectively as heuristics.… |
| `lower energy` | 542 | …- **Consistent energy reduction:** lower energy than the same-session dense model in 5 of 5 sessions, **and… |
| `lower energy` | 549 | …ff:** either accuracy within 0.5 pp of the other method and lower energy in 5 of 5 sessions, or… |
| `GREEN-SUCCESS` | 577 | …iciency success criterion.** The pre-registration names it *GREEN-SUCCESS*. A model succeeds only… |
| `green` | 577 | …iciency success criterion.** The pre-registration names it *GREEN-SUCCESS*. A model succeeds only… |
| `carbon` | 740 | … latency) normalised to the same-session dense model; not a carbon metric / 7A, 7B-EC / ↓ /… |
| `equivalen(t|ce)` | 810 | …fidence-interval inclusion rule**, not a formal statistical equivalence test. A formal… |
| `equivalen(t|ce)` | 812 | …pre-registration calls the rule "practically equivalent". The thesis writes "within the ±0.5 pp margin" so that it … |
| `equivalen(t|ce)` | 813 | …not read as a formal equivalence claim.… |
| `equivalen(t|ce)` | 1076 | …  not a formal equivalence test.… |

## Freeze

Chapter 4 v1 is frozen. Later changes are limited to citation-number assembly or factual corrections found during later QA, and will be logged here.
