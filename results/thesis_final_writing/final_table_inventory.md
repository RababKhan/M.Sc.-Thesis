# Final table inventory

Numbered main-text tables: **32**. Each chapter also contains short un-numbered inline tables (summary
grids, criteria lists). These are counted below and should be numbered, or kept as in-text lists, according to
the official template.

| Table | Title | Chapter | Source file | Placement | Numbers audited | In-text references | Status |
|---|---|---|---|---|---|---|---|
| 1.1 | Research questions, objectives, contributions and evidence | Ch. 1 | `chapter_01_introduction.md` | main text | conceptual / mapping (no frozen numbers) | referenced 1× | final |
| 2.1 | Pruning granularity | Ch. 2 | `chapter_02_literature_review.md` | main text | conceptual; Table 2.2 cells checked against abstracts (Milestone 1) | referenced 1× | final |
| 2.2 | Comparison of representative pruning-allocation and deployment-aware pruning works | Ch. 2 | `chapter_02_literature_review.md` | main text | conceptual; Table 2.2 cells checked against abstracts (Milestone 1) | referenced 1× | final |
| 3.1 | Development of the framework across phases | Ch. 3 | `chapter_03_framework.md` | main text | conceptual; equations audited (equation_audit_chapter_03.md) | referenced 1× | final |
| 3.2 | Notation | Ch. 3 | `chapter_03_framework.md` | main text | conceptual; equations audited (equation_audit_chapter_03.md) | referenced 1× | final |
| 3.3 | Pruning units of the core framework (unstructured) | Ch. 3 | `chapter_03_framework.md` | main text | conceptual; equations audited (equation_audit_chapter_03.md) | referenced 3× | final |
| 3.4 | State features by phase | Ch. 3 | `chapter_03_framework.md` | main text | conceptual; equations audited (equation_audit_chapter_03.md) | referenced 3× | final |
| 3.5 | Mature core framework (as frozen after Phase 5, evaluated in Phase 6) | Ch. 3 | `chapter_03_framework.md` | main text | conceptual; equations audited (equation_audit_chapter_03.md) | referenced 1× | final |
| 3.6 | Three levels of deployment efficiency | Ch. 3 | `chapter_03_framework.md` | main text | conceptual; equations audited (equation_audit_chapter_03.md) | referenced 2× | final |
| 3.7 | PPO configuration (identical in Phases 3, 4, 5 and 7B) | Ch. 3 | `chapter_03_framework.md` | main text | conceptual; equations audited (equation_audit_chapter_03.md) | referenced 3× | final |
| 4.1 | Datasets and split configurations | Ch. 4 | `chapter_04_experimental_setup.md` | main text | audit_ch4.py 100/100 | referenced 1× | final |
| 4.2 | Data-split usage by phase | Ch. 4 | `chapter_04_experimental_setup.md` | main text | audit_ch4.py 100/100 | referenced 2× | final |
| 4.3 | Network architectures and pruning units (core framework) | Ch. 4 | `chapter_04_experimental_setup.md` | main text | audit_ch4.py 100/100 | referenced 1× | final |
| 4.4 | Dense reference models (accuracy, %) | Ch. 4 | `chapter_04_experimental_setup.md` | main text | audit_ch4.py 100/100 | referenced 1× | final |
| 4.5 | Baseline methods and matching rules | Ch. 4 | `chapter_04_experimental_setup.md` | main text | audit_ch4.py 100/100 | referenced 1× | final |
| 4.6 | PPO configuration by phase | Ch. 4 | `chapter_04_experimental_setup.md` | main text | audit_ch4.py 100/100 | referenced 2× | final |
| 4.7 | Phase-by-phase experimental design | Ch. 4 | `chapter_04_experimental_setup.md` | main text | audit_ch4.py 100/100 | referenced 2× | final |
| 4.8 | Evaluation metrics | Ch. 4 | `chapter_04_experimental_setup.md` | main text | audit_ch4.py 100/100 | referenced 1× | final |
| 4.9 | Pre-registered decision criteria | Ch. 4 | `chapter_04_experimental_setup.md` | main text | audit_ch4.py 100/100 | referenced 2× | final |
| 4.10 | Hardware and software environments | Ch. 4 | `chapter_04_experimental_setup.md` | main text | audit_ch4.py 100/100 | referenced 2× | final |
| 5.1 | Dense reference models (accuracy %, seed-0 reference) | Ch. 5 | `chapter_05_core_results.md` | main text | audit_ch5.py 182/182 | referenced 2× | final |
| 5.2 | Phase 3: P0 minus baseline, test accuracy (pp, mean of 20 paired runs) | Ch. 5 | `chapter_05_core_results.md` | main text | audit_ch5.py 182/182 | referenced 1× | final |
| 5.4 | Retention: the best policy encountered, the terminal policy (S0) and the archive-selected policy (S2A) | Ch. 5 | `chapter_05_core_results.md` | main text | audit_ch5.py 182/182 | referenced 1× | final |
| 5.5 | ResNet-8 reward alignment across Phases 4 and 5 (test accuracy %, mean of 20 runs; sparsity = mean selected masked sparsity) | Ch. 5 | `chapter_05_core_results.md` | main text | audit_ch5.py 182/182 | referenced 1× | final |
| 5.6 | Phase 5 by setting (test accuracy %, mean of 20 runs; differences in pp) | Ch. 5 | `chapter_05_core_results.md` | main text | audit_ch5.py 182/182 | referenced 1× | final |
| 5.7 | Phase 6: test accuracy before → after equal fine-tuning, and the PPO–LAMP and PPO–Global gaps (pp; mean of 20 paired runs; 95% t-CI) | Ch. 5 | `chapter_05_core_results.md` | main text | audit_ch5.py 182/182 | referenced 1× | final |
| 5.3 | Sensitivity guidance across Phases 3–5: exploration endpoints versus final-model endpoints (settings supported / opposite, of 6) | Ch. 5 | `chapter_05_core_results.md` | main text | audit_ch5.py 182/182 | referenced 4× | final |
| 6.1 | Exhaustive structured latency tables: cost metrics against measured latency gain | Ch. 6 | `chapter_06_deployment_sustainability.md` | main text | audit_ch6.py 142/142 | referenced 3× | final |
| 6.2 | ResNet-8 (CIFAR-10): calibrated accuracy-feasible latency headroom | Ch. 6 | `chapter_06_deployment_sustainability.md` | main text | audit_ch6.py 142/142 | caption only | final |
| 6.3 | Phase 7A: accuracy, compression, latency and measured energy | Ch. 6 | `chapter_06_deployment_sustainability.md` | main text | audit_ch6.py 142/142 | referenced 1× | final |
| 6.4 | Phase 7B: accuracy, memory and computation after the equal 5-epoch recovery (SimpleCNN, CIFAR-10) | Ch. 6 | `chapter_06_deployment_sustainability.md` | main text | audit_ch6.py 142/142 | caption only | final |
| 6.5 | Three-level deployment-efficiency assessment | Ch. 6 | `chapter_06_deployment_sustainability.md` | main text | audit_ch6.py 142/142 | caption only | final |

| Chapter | Un-numbered inline tables |
|---|---|
| Ch. 1 | 1 |
| Ch. 2 | 0 |
| Ch. 3 | 3 |
| Ch. 4 | 6 |
| Ch. 5 | 7 |
| Ch. 6 | 7 |
| Ch. 7 | 2 |
| Ch. 8 | 1 |

## Table QA

| Check | Result |
|---|---|
| Duplicate tables | None. Table 5.1 (dense references, Ch. 5) repeats four columns of Table 4.4 (Ch. 4) by design: Ch. 4 defines them as setup, Ch. 5 reports them as the reference point. At template stage, Table 5.1 could be shortened to a pointer |
| Repeated results | Phase-6 PPO–LAMP gaps appear in Table 5.7 and are summarised in prose in Chapters 7–8; Phase-7A energy appears in Table 6.3 and Table 6.5. The values are identical (global numerical audit PASS) |
| Inconsistent rounding | Accuracies 2 dp, energy J 2 dp in Table 6.3 (3 dp in the appendix), ratios 3 dp. The convention is round-half-up throughout (+31.37). One prose variant (5.1 / 3.2 pp, Ch. 6 §6.8) is a 1-dp rendering of 5.08 / 3.24, accepted |
| Oversized tables moved to the appendix | Seed-level and session-level data and full statistics families are in Appendices B–J. The largest main-text tables (4.6 PPO by phase, 4.8 metrics, 5.7 Phase 6, 6.3 Phase 7A) are 6–14 rows |
| Every numbered table referenced in text | See In-text references. Tables 6.2, 6.4 and 6.5 are introduced by the paragraph directly above them but are not cited by number. Add '(Table 6.x)' at template stage (presentation only) |
| Numbering order | **Table 5.3 is placed in §5.11, after Tables 5.4–5.7** (the chapter explains this: it draws on §§5.4, 5.5 and 5.8). Templates normally require numerical order. At template stage, either move Table 5.3 to the end of §5.5 or renumber it 5.7 (with 5.4–5.7 becoming 5.3–5.6). This is a formatting decision; no content change |
