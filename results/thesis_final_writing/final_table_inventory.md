# Final table inventory (MIST)

Numbered tables: **42** (32 in Chapters 1–8, 10 in Appendices A–E). MIST requires consecutive
numbering; all chapter tables were renumbered in order of appearance by `tools/renumber_tables.py` (mapping in
`tools/_table_renumber_map.json`), which also resolved the former out-of-order Table 5.3 (now Table 5.11, where it
stands). Every previously un-numbered inline table was either converted to an equivalent list/sentence during
compression or given a caption and number. Every table appears in the List of Tables. Table values are unchanged.

| Table | Title | Location | Previous number | In-text references | List |
|---|---|---|---|---|---|
| 1.1 | Research questions, objectives, contributions and evidence | Ch. 1 | 1.1 | referenced 2× | in List of Tables |
| 2.1 | Comparison of representative pruning-allocation and deployment-aware pruning works | Ch. 2 | 2.2 | referenced 1× | in List of Tables |
| 3.1 | Development of the framework across phases | Ch. 3 | 3.1 | referenced 1× | in List of Tables |
| 3.2 | Pruning units of the core framework (unstructured) | Ch. 3 | 3.3 | referenced 4× | in List of Tables |
| 3.3 | State features by phase | Ch. 3 | 3.4 | referenced 2× | in List of Tables |
| 3.4 | Mature core framework (as frozen after Phase 5, evaluated in Phase 6) | Ch. 3 | 3.5 | referenced 1× | in List of Tables |
| 4.1 | Datasets and split configurations | Ch. 4 | 4.1 | referenced 2× | in List of Tables |
| 4.2 | Data-split usage by phase | Ch. 4 | 4.2 | referenced 2× | in List of Tables |
| 4.3 | Network architectures and pruning units (core framework) | Ch. 4 | 4.3 | referenced 1× | in List of Tables |
| 4.4 | Dense reference models (accuracy, %) | Ch. 4 | 4.4 | referenced 1× | in List of Tables |
| 4.5 | Baseline methods and matching rules | Ch. 4 | 4.5 | referenced 1× | in List of Tables |
| 4.6 | PPO configuration by phase | Ch. 4 | 4.6 | referenced 3× | in List of Tables |
| 4.7 | Phase-by-phase experimental design | Ch. 4 | 4.7 | referenced 2× | in List of Tables |
| 4.8 | Reward configurations | Ch. 4 | 4.8 | caption + adjacent text | in List of Tables |
| 4.9 | Evaluation metrics | Ch. 4 | 4.9 | referenced 1× | in List of Tables |
| 4.10 | Pre-registered decision criteria | Ch. 4 | 4.10 | referenced 1× | in List of Tables |
| 4.11 | Hardware and software environments | Ch. 4 | 4.11 | referenced 2× | in List of Tables |
| 5.1 | Dense reference models (accuracy %, seed-0 reference) | Ch. 5 | 5.1 | referenced 2× | in List of Tables |
| 5.2 | Phase 3: P0 minus baseline, test accuracy (pp, mean of 20 paired runs) | Ch. 5 | 5.2 | referenced 1× | in List of Tables |
| 5.3 | Retention: the best policy encountered, the terminal policy (S0) and the archive-selected policy (S2A) | Ch. 5 | 5.3 | referenced 1× | in List of Tables |
| 5.4 | ResNet-8 reward alignment across Phases 4 and 5 (test accuracy %, mean of 20 runs; sparsity = mean selected masked sparsity). Phase-4 and Phase-5 rows share the test set and dense reference | Ch. 5 | 5.5 | referenced 2× | in List of Tables |
| 5.5 | Phase 5 by setting (test accuracy %, mean of 20 runs; differences in pp) | Ch. 5 | 5.6 | referenced 1× | in List of Tables |
| 5.6 | Validation-to-test feasibility of the selected Phase-5 policies (q ≥ 0.98; runs out of 120) | Ch. 5 | 5.8 | caption + adjacent text | in List of Tables |
| 5.7 | Phase 6: test accuracy before → after equal fine-tuning, and the PPO–LAMP and PPO–Global gaps (pp; mean of 20 paired runs; 95% t-CI) | Ch. 5 | 5.9 | referenced 1× | in List of Tables |
| 5.8 | Sensitivity guidance across Phases 3–5: exploration endpoints versus final-model endpoints (settings supported / opposite, of 6) | Ch. 5 | 5.11 | referenced 4× | in List of Tables |
| 6.1 | Exhaustive structured latency tables: cost metrics against measured latency gain | Ch. 6 | 6.1 | referenced 1× | in List of Tables |
| 6.2 | ResNet-8 (CIFAR-10): calibrated accuracy-feasible latency headroom | Ch. 6 | 6.4 | referenced 1× | in List of Tables |
| 6.3 | Phase 7A: accuracy, compression, latency and measured energy | Ch. 6 | 6.5 | referenced 1× | in List of Tables |
| 6.4 | Phase 7B: accuracy, memory and computation after the equal 5-epoch recovery (SimpleCNN, CIFAR-10) | Ch. 6 | 6.6 | referenced 1× | in List of Tables |
| 6.5 | Three-level deployment-efficiency assessment | Ch. 6 | 6.8 | referenced 1× | in List of Tables |
| 7.1 | Negative and partial results and what each establishes | Ch. 7 | 7.1 | referenced 1× | in List of Tables |
| 8.1 | Thesis contributions and their boundaries | Ch. 8 | 8.1 | referenced 2× | in List of Tables |
| A.1 | Pruning units of the three networks: weights, share of prunable weights and MACs per image (CIFAR-10 output layer) | App. A | new (Appendix) | appendix (listed) | in List of Tables |
| A.2 | Frozen dense reference checkpoints (Table 4.4): checkpoint file and SHA-256 prefix | App. A | new (Appendix) | referenced 2× | in List of Tables |
| A.3 | Pre-registration records (Table 4.10) and the commits that first added them | App. A | new (Appendix) | referenced 2× | in List of Tables |
| B.1 | Phase 3 mean test accuracy (%) of PPO (P0) and the five baselines built at each PPO run's zero count (20 runs) | App. B | new (Appendix) | appendix (listed) | in List of Tables |
| B.2 | Normalised loss sensitivity S_l of the four units by setting and the Phase-5 centred prior coefficients | App. B | new (Appendix) | appendix (listed) | in List of Tables |
| C.1 | Phase 6 mean recovery (test accuracy after minus before equal fine-tuning, pp; 20 runs) by setting and method | App. C | new (Appendix) | appendix (listed) | in List of Tables |
| D.1 | H0 calibration: decision quantities of the three candidate measurement configurations | App. D | new (Appendix) | appendix (listed) | in List of Tables |
| D.2 | Exhaustive latency tables: accepted sessions, repeated attempts and anchor drift (limit 10%) | App. D | new (Appendix) | appendix (listed) | in List of Tables |
| E.1 | Phase 7A energy: session summaries (five sessions per architecture) | App. E | new (Appendix) | appendix (listed) | in List of Tables |
| E.2 | Dense-only stability pilots and qualification (energy per 1,000 inferences, J; six dense blocks each) | App. E | new (Appendix) | appendix (listed) | in List of Tables |

