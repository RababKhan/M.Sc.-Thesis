# Final table inventory (MIST)

Numbered tables: **61** (43 in Chapters 1–8, 18 in Appendices A–E). MIST requires consecutive
numbering; all chapter tables were renumbered in order of appearance by `tools/renumber_tables.py` (mapping in
`tools/_table_renumber_map.json`), which also resolved the former out-of-order Table 5.3 (now Table 5.11, where it
stands). Every previously un-numbered inline table was either converted to an equivalent list/sentence during
compression or given a caption and number. Every table appears in the List of Tables. Table values are unchanged.

| Table | Title | Location | Previous number | In-text references | List |
|---|---|---|---|---|---|
| 1.1 | Research questions, objectives, contributions and evidence | Ch. 1 | 1.1 | referenced 1× | in List of Tables |
| 2.1 | Pruning granularity | Ch. 2 | 2.1 | referenced 1× | in List of Tables |
| 2.2 | Comparison of representative pruning-allocation and deployment-aware pruning works | Ch. 2 | 2.2 | referenced 1× | in List of Tables |
| 3.1 | Development of the framework across phases | Ch. 3 | 3.1 | referenced 1× | in List of Tables |
| 3.2 | Notation | Ch. 3 | 3.2 | referenced 1× | in List of Tables |
| 3.3 | Pruning units of the core framework (unstructured) | Ch. 3 | 3.3 | referenced 4× | in List of Tables |
| 3.4 | State features by phase | Ch. 3 | 3.4 | referenced 2× | in List of Tables |
| 3.5 | Mature core framework (as frozen after Phase 5, evaluated in Phase 6) | Ch. 3 | 3.5 | referenced 1× | in List of Tables |
| 3.6 | Three levels of deployment efficiency | Ch. 3 | 3.6 | referenced 2× | in List of Tables |
| 3.7 | PPO configuration (identical in Phases 3, 4, 5 and 7B) | Ch. 3 | 3.7 | referenced 3× | in List of Tables |
| 4.1 | Datasets and split configurations | Ch. 4 | 4.1 | referenced 2× | in List of Tables |
| 4.2 | Data-split usage by phase | Ch. 4 | 4.2 | referenced 2× | in List of Tables |
| 4.3 | Network architectures and pruning units (core framework) | Ch. 4 | 4.3 | referenced 1× | in List of Tables |
| 4.4 | Dense reference models (accuracy, %) | Ch. 4 | 4.4 | referenced 1× | in List of Tables |
| 4.5 | Baseline methods and matching rules | Ch. 4 | 4.5 | referenced 1× | in List of Tables |
| 4.6 | PPO configuration by phase | Ch. 4 | 4.6 | referenced 1× | in List of Tables |
| 4.7 | Phase-by-phase experimental design | Ch. 4 | 4.7 | referenced 2× | in List of Tables |
| 4.8 | Reward configurations | Ch. 4 | new caption | caption + adjacent text | in List of Tables |
| 4.9 | Evaluation metrics | Ch. 4 | 4.8 | referenced 1× | in List of Tables |
| 4.10 | Pre-registered decision criteria | Ch. 4 | 4.9 | referenced 1× | in List of Tables |
| 4.11 | Hardware and software environments | Ch. 4 | 4.10 | referenced 2× | in List of Tables |
| 5.1 | Dense reference models (accuracy %, seed-0 reference) | Ch. 5 | 5.1 | referenced 2× | in List of Tables |
| 5.2 | Phase 3: P0 minus baseline, test accuracy (pp, mean of 20 paired runs) | Ch. 5 | 5.2 | referenced 2× | in List of Tables |
| 5.3 | Retention: the best policy encountered, the terminal policy (S0) and the archive-selected policy (S2A) | Ch. 5 | 5.4 | referenced 1× | in List of Tables |
| 5.4 | Phase 4 pre-registered criteria | Ch. 5 | new caption | caption + adjacent text | in List of Tables |
| 5.5 | ResNet-8 reward alignment across Phases 4 and 5 (test accuracy %, mean of 20 runs; sparsity = mean selected masked sparsity) | Ch. 5 | 5.5 | referenced 3× | in List of Tables |
| 5.6 | Phase 5 by setting (test accuracy %, mean of 20 runs; differences in pp) | Ch. 5 | 5.6 | referenced 1× | in List of Tables |
| 5.7 | Phase 5 pre-registered criteria | Ch. 5 | new caption | caption + adjacent text | in List of Tables |
| 5.8 | Validation-to-test feasibility of the selected Phase-5 policies (q ≥ 0.98; runs out of 120) | Ch. 5 | new caption | caption + adjacent text | in List of Tables |
| 5.9 | Phase 6: test accuracy before → after equal fine-tuning, and the PPO–LAMP and PPO–Global gaps (pp; mean of 20 paired runs; 95% t-CI) | Ch. 5 | 5.7 | referenced 1× | in List of Tables |
| 5.10 | Phase 6 pre-registered criteria | Ch. 5 | new caption | caption + adjacent text | in List of Tables |
| 5.11 | Sensitivity guidance across Phases 3–5: exploration endpoints versus final-model endpoints (settings supported / opposite, of 6) | Ch. 5 | 5.3 | referenced 5× | in List of Tables |
| 6.1 | Exhaustive structured latency tables: cost metrics against measured latency gain | Ch. 6 | 6.1 | referenced 4× | in List of Tables |
| 6.2 | Two CIFAR-10 SimpleCNN policies: parameter reduction, MAC reduction and measured speed-up | Ch. 6 | new caption | caption + adjacent text | in List of Tables |
| 6.3 | Validation of the 1-epoch accuracy proxy against 5-epoch recovery | Ch. 6 | new caption | caption + adjacent text | in List of Tables |
| 6.4 | ResNet-8 (CIFAR-10): calibrated accuracy-feasible latency headroom | Ch. 6 | 6.2 | referenced 1× | in List of Tables |
| 6.5 | Phase 7A: accuracy, compression, latency and measured energy | Ch. 6 | 6.3 | referenced 2× | in List of Tables |
| 6.6 | Phase 7B: accuracy, memory and computation after the equal 5-epoch recovery (SimpleCNN, CIFAR-10) | Ch. 6 | 6.4 | referenced 2× | in List of Tables |
| 6.7 | Phase-7B confirmatory energy pilot (dense model only; limit CV ≤ 3%) | Ch. 6 | new caption | caption + adjacent text | in List of Tables |
| 6.8 | Three-level deployment-efficiency assessment | Ch. 6 | 6.5 | referenced 1× | in List of Tables |
| 7.1 | Negative and partial results and what each establishes | Ch. 7 | new caption | referenced 1× | in List of Tables |
| 7.2 | Stages of the diagnostic framework, the question each isolates, and the thesis evidence | Ch. 7 | new caption | referenced 1× | in List of Tables |
| 8.1 | Thesis contributions and their boundaries | Ch. 8 | new caption | referenced 1× | in List of Tables |
| A.1 | Pruning units, weight shapes, parameters and MACs per image (CIFAR-10 output layer) | App. A | new (Appendix) | appendix (listed) | in List of Tables |
| A.2 | Dense references over three training seeds | App. A | new (Appendix) | appendix (listed) | in List of Tables |
| A.3 | Equal recovery protocol (Phase 6; reused in Paper 3, Phases 7A and 7B) | App. A | new (Appendix) | appendix (listed) | in List of Tables |
| A.4 | Pre-registration and freeze commits | App. A | new (Appendix) | appendix (listed) | in List of Tables |
| A.5 | Software and hardware environments | App. A | new (Appendix) | appendix (listed) | in List of Tables |
| B.1 | Phase 3 mean test accuracy and sparsity by setting and method (20 runs; baselines at each PPO run's zero count) | App. B | new (Appendix) | appendix (listed) | in List of Tables |
| B.2 | Normalised loss sensitivity S_l by setting and the Phase-5 centred prior coefficients | App. B | new (Appendix) | appendix (listed) | in List of Tables |
| C.1 | Phase 4 search quality by variant (VAL-RL (I) ranks among 1,296 policies; medians over 20 runs) | App. C | new (Appendix) | appendix (listed) | in List of Tables |
| C.2 | Phase 5 selected policies by setting and condition (20 runs each) | App. C | new (Appendix) | appendix (listed) | in List of Tables |
| C.3 | Phase 6 test accuracy before and after equal recovery (means of 20 runs) | App. C | new (Appendix) | appendix (listed) | in List of Tables |
| D.1 | H0 calibration: per-configuration decision quantities | App. D | new (Appendix) | appendix (listed) | in List of Tables |
| D.2 | Latency-table session log (anchor drift; limit 10%) | App. D | new (Appendix) | appendix (listed) | in List of Tables |
| D.3 | Accuracy-proxy validation (1-epoch against 5-epoch recovery) | App. D | new (Appendix) | appendix (listed) | in List of Tables |
| D.4 | ResNet-8 threshold calibration and latency-room decision | App. D | new (Appendix) | appendix (listed) | in List of Tables |
| E.1 | Phase 7A energy: session summaries (five sessions per architecture) | App. E | new (Appendix) | appendix (listed) | in List of Tables |
| E.2 | Phase 7B: all recovered final models (5-epoch recovery) | App. E | new (Appendix) | appendix (listed) | in List of Tables |
| E.3 | Dense-only stability pilots and qualification (energy per 1,000 inferences, J) | App. E | new (Appendix) | appendix (listed) | in List of Tables |
| E.4 | Energy-confirmation qualification blocks (descriptive diagnostics) | App. E | new (Appendix) | appendix (listed) | in List of Tables |

