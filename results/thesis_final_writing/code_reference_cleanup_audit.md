# Code-reference cleanup audit

Scope: front matter (English and Bangla Abstracts, abbreviations, notation list), Chapters 1–8, captions, tables
and Appendices A–E. Searched with one detector (used before and after the cleanup) for:
- backticked spans;
- `src/…`, `results/…`, `experiments/…` and `tools/…` paths;
- `.py`, `.json`, `.csv`, `.pt`, `.pth`, `.md`, `.npz` and `.txt` file names;
- CamelCase class names;
- `function()` calls and configuration keys;
- 8-hex-digit commit and hash prefixes.

8-digit split seeds (such as 20260928) are experimental values, not code, and were kept. Every figure source path
inside a markdown image is resolved at build time and never printed, so none appears.

No scientific result, number, equation, experimental condition, claim or method was changed. Page numbers refer to
the final `final_thesis_MIST.pdf` (165 physical pages).

## Results of the search

| Location | References found | After cleanup |
|---|---|---|
| Abstract, Bangla Abstract | 0 | 0 |
| Chapters 1–2 | 0 | 0 |
| Chapter 3 | 15 | **0** |
| Chapter 4 | 41 code references, plus 4 split-seed values | **0** code references; the seeds are kept |
| Chapters 5–8 | 0 | 0 |
| Front-matter lists (abbreviations, notation) | 0 | 0 |
| Appendices B–E and A.1 | 21 (file names in source notes, the generator path, a configuration key) | **0** |
| Appendix A.2 (reproducibility) | n/a | all identifiers collected here (retained) |

## Every reference and its treatment

| # | Location (PDF page) | Reference | Action | Replacement / reason |
|---|---|---|---|---|
| 1 | Ch. 3 Table 3.2 (p. 33) | `features.0`, `features.3`, `features.6`, `classifier.1`, `classifier.3` | removed | The unit names conv1, conv2, conv3, fc1 and output layer stand alone. The module names are retained in A.2 |
| 2 | Ch. 3 §3.5 (p. 34) | `src/rl`, `unit_state` | removed | "The state is a seven-feature vector computed for the current unit l of N_L units" |
| 3 | Ch. 3 §3.8 (p. 36) | `src/sensitivity`, `loss_sensitivity` | removed | The sentence ends at "…never the selection split or the test set" |
| 4 | Ch. 3 §3.9 (p. 37) | `src/rl`, `SoftPriorPolicy` | **rewritten** | "The prior adjustment is applied during every forward pass, both when actions are sampled and when action log-probabilities are evaluated during the PPO update" (the brief's example) |
| 5 | Ch. 3 §3.15 (p. 42) | `experiments/phase6/phase6_lib.py`, `PROTOCOL`, `finetune` | removed | "…applied one fixed protocol to every compared model: Adam with cosine annealing…" |
| 6 | Ch. 3 §3.16.1 (p. 43) | `src/pruning/structured_units.py` | removed | "…does not depend on the order of compaction; removing an output channel…" |
| 7 | Ch. 4 §4.3.1 (p. 49) | `torch.randperm` | removed | "…a fixed permutation of the 50,000 training indices (seed 42)…". The function name is retained in A.2 |
| 8 | Ch. 4 §4.4 (p. 51) | `src/models` | rewritten | "The three networks are:" |
| 9 | Ch. 4 Table 4.4 (p. 53) | six checkpoint SHA-256 prefixes | **moved** | The hash column was removed and the caption now points to Appendix A, Table A.2, where the hashes are listed with their checkpoint files |
| 10 | Ch. 4 §4.7.1 (p. 54) | `src/pruning` | removed | "The baselines are one-shot methods, each calibrated…" |
| 11 | Ch. 4 Table 4.6 (p. 57) | `n_steps` (configuration key) | rewritten | "rollout length; minibatch; epochs per update" |
| 12 | Ch. 4 §4.8 (p. 57) | "a copy of the SB3 PPO update" (implementation detail) | rewritten | "adds η·𝓛_elite (Equation 3.13) to the PPO loss; with η = 0 the update is identical to standard PPO". The implementation note is retained in A.2 |
| 13 | Ch. 4 Table 4.10 (p. 68) | 13 governing-record file names and 13 commit hashes | **moved** | The Source (commit) column was removed and the caption now points to Appendix A, Table A.3, where every record and commit is listed |
| 14 | Ch. 4 §4.16 (p. 69) | `requirements_locked.txt` | rewritten | "Package versions are pinned, and every phase records its software environment and code version (Appendix A.2)" |
| 15 | Ch. 4 §4.17 (p. 69) | `eval()`, `inference_mode()` | rewritten | "evaluation mode with gradient tracking disabled" |
| 16 | Ch. 4 §4.17 (p. 69) | `time.perf_counter_ns()` | rewritten | "a nanosecond-resolution monotonic clock" |
| 17 | Ch. 4 §4.18.1 (p. 70) | `\Energy Meter(RAPL_Package0_PKG)\Energy` | rewritten | "(its package-0 RAPL counter)". The exact counter path is retained in A.2 |
| 18 | Appendix intro (A-1) | `tools/build_appendices.py` | moved | The intro now says the records and identifiers are listed once, in Section A.2 |
| 19 | Table A.1 and its note (A-1) | `cpu_prunable_units.csv`; module unit names | rewritten | The note reads "frozen unit table". Units now use the Table 3.2 names (conv1 … fc1, stage 1–3, output layer) |
| 20 | Notes of Tables B.1, B.2, C.1, D.1, D.2, E.1, E.2 | 12 record file names and the configuration key `session_log` | rewritten | Each note describes its source ("frozen Phase-3 summary record", "frozen latency-table session log", and so on). The file names are listed in A.2 |

## Retained implementation references (all in Appendix A.2, pp. A-1 to A-3)

| Retained item | Reason |
|---|---|
| Repository organisation by phase (folder names) | The single repository statement, which identifies where the per-seed and per-session records are kept |
| Table A.2: checkpoint file names and SHA-256 prefixes | They identify the frozen dense references every pruning result depends on |
| Table A.3: pre-registration record names and commits | They make the pre-registration claims of Chapters 4–8 checkable against the version history |
| "Implementation identifiers" list (modules, classes, functions, the counter path, the requirements file, the appendix generator) | It maps each conceptually described component to the code, for exact reproduction |
| "Source records of the appendix tables" list | Provenance of every appendix value |

## Effect on length and audits

- **Chapters 1–8:** 34,315 → 34,244 words.
- **Appendices:** 1,116 → 1,457 words, because identifiers moved into A.2 and Tables A.2 and A.3 were added.
- **Tables:** 42 numbered (32 chapter, 10 appendix).
- **Length:** 165 physical pages and 154 MIST-countable pages, with a MIST word-equivalent of 50,945 (`mist_length_compliance_audit.md`).

All audits pass after the cleanup:

| Audit | Result |
|---|---|
| Chapter 4 numerical | 100/100. The Table 4.4 check now finds the accuracy values in Chapter 4 and each hash in Appendix A |
| Chapter 5 numerical | 182/182 |
| Chapter 6 numerical | 142/142 |
| Chapter 7 | no new numbers |
| Global numerical | 16/16 |
| Claims | 51 hits, none affirmative |
| Split governance | 6/6 |
| Citations | 189 occurrences, 0 unresolved, 98 references |

Visual QA covered the regenerated PDF, with every changed page inspected.
