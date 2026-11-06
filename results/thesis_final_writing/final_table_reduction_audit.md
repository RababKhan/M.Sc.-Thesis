# Final table-reduction audit (MIST length pass)

Baseline: commit `9366c52a` (61 numbered tables: 43 chapter, 18 appendix). Final: **40 numbered tables** (32 chapter,
8 appendix). No value was changed; every value of a removed or merged table is either carried verbatim into prose,
preserved in a retained table, or kept in the repository's frozen records (named in each row). Numbers below are the
**previous** table numbers; the final numbers are in `final_table_inventory.md` (mapping `tools/_table_renumber_map.json`).

Classes: **A** essential main-text table · **B** useful appendix table · **C** repository-only detail.

## Chapter tables (43 → 32)

| Table (old → new) | Purpose | Claim supported | Main text? | Appendix? | Repository only? | Reason |
|---|---|---|---|---|---|---|
| 1.1 → 1.1 | RQ–objective–contribution–evidence map | structure of the thesis | **A** | — | — | single map of RQs to evidence |
| 2.1 → (prose) | Pruning granularity | unstructured vs structured background | no: one sentence in §2.2 | — | — | three-row definitional table; content fits one sentence |
| 2.2 → 2.1 | Comparison of related works | research gap (§2.14) | **A** | — | — | the literature evidence for the gaps |
| 3.1 → 3.1 | Development of the framework across phases | chronological design (RQ chain) | **A** | — | — | the only overview of when each component entered |
| 3.2 → (front matter) | Notation | — | no: List of Main Notation | — | — | duplicated the MIST notation list (TP.17); symbols `l` and `Ê` added to the list |
| 3.3 → 3.2 | Pruning units | method specification | **A** | — | — | defines the action space |
| 3.4 → 3.3 | State features by phase | method specification | **A** | — | — | defines the state |
| 3.5 → 3.4 | Mature core framework | final framework (Contribution 1) | **A** | — | — | summary of the framework evaluated in Phase 6 |
| 3.6 → (prose) | Three levels of deployment efficiency | RQ6 framing | no: three-item list in §3.16.2 | — | — | small definitional table; Table 6.5 (final) carries the evidence |
| 3.7 → (merged) | PPO configuration | reproducibility | no: merged into Table 4.6 | — | — | duplicated Table 4.6; the two missing items (orthogonal initialisation, Adam lr) were added there |
| 4.1 → 4.1 | Datasets and splits | split governance | **A** | — | — | key methodology |
| 4.2 → 4.2 | Data-split usage by phase | split governance | **A** | — | — | key methodology (split audit) |
| 4.3 → 4.3 | Architectures and units | methodology | **A** | — | — | |
| 4.4 → 4.4 | Dense references (hashes, accuracies) | reference models | **A** | — | — | audited (Ch. 4 audit) |
| 4.5 → 4.5 | Baselines and matching rules | fairness of comparisons | **A** | — | — | |
| 4.6 → 4.6 | PPO configuration by phase | reproducibility | **A** | — | — | now the single PPO configuration table |
| 4.7 → 4.7 | Phase-by-phase design | methodology | **A** | — | — | |
| 4.8 → 4.8 | Reward configurations | RQ4 | **A** | — | — | |
| 4.9 → 4.9 (compressed) | Evaluation metrics | definitions | **A** (25 → 12 rows) | — | — | related metrics merged; every definition kept |
| 4.10 → 4.10 | Pre-registered decision criteria and commits | pre-registration | **A** | — | — | |
| 4.11 → 4.11 | Hardware and software | reproducibility | **A** | — | — | |
| 5.1 → 5.1 | Dense reference performance | reference point for Chapter 5 | **A** | — | — | 3-seed means (not in Table 4.4) |
| 5.2 → 5.2 | Phase 3 PPO minus baselines | RQ1 | **A** | — | — | |
| 5.3 → 5.11 → 5.8 | Sensitivity across Phases 3–5 | RQ2 | **A** | — | — | |
| 5.4 → 5.3 | Retention (best seen / terminal / archive) | RQ3 | **A** | — | — | |
| Phase 4 criteria → (prose) | Phase-4 pre-registered criteria | RQ3–RQ4 verdicts | no: one sentence in §5.7 (all statuses verbatim) | — | — | five status rows |
| 5.5 → 5.4 | ResNet-8 reward alignment | RQ4 | **A** | — | — | |
| 5.6 → 5.5 | Phase 5 by setting | RQ4 | **A** | — | — | |
| Phase 5 criteria → (prose) | Phase-5 pre-registered criteria | RQ4 verdicts | no: one paragraph in §5.8 (statuses and reasons verbatim) | — | — | |
| 5.8 → 5.6 | Validation-to-test feasibility | RQ4 limitation | **A** | — | — | key negative result |
| 5.9 → 5.7 | Phase 6 before/after recovery and gaps | RQ5 | **A** | — | — | |
| Phase 6 criteria → (prose) | Phase-6 pre-registered criteria | RQ5 verdicts | no: one paragraph in §5.10 (all MET, with bases) | — | — | |
| 6.1 → 6.1 | Latency tables: cost metrics vs latency | RQ6 (latency proxies) | **A** | — | — | |
| 6.2 → (prose) | Two SimpleCNN policies | RQ6 example | no: prose in §6.5 (values verbatim) | — | — | two-row example |
| 6.3 → (prose) | 1-epoch proxy validation | RQ6 headroom | no: prose in §6.6 (values verbatim) | — | — | two-row table |
| 6.4 → 6.2 | ResNet-8 latency headroom | RQ6 (1.098× / 1.089×) | **A** | — | — | |
| 6.5 → 6.3 | Phase 7A accuracy, compression, latency, energy | RQ6 (energy with accuracy cost) | **A** | — | — | |
| 6.6 → 6.4 | Phase 7B memory / MAC / accuracy | RQ6 (7B) | **A** | — | — | |
| 6.7 → (prose) | Phase-7B confirmatory pilot | RQ6 (energy unresolved) | no: prose in §6.13 (CVs verbatim) | also in Table E.2 | — | duplicated by Appendix E |
| 6.8 → 6.5 | Three-level assessment | RQ6 answer | **A** | — | — | |
| 7.1 → 7.1 | Negative results and what they establish | negative findings | **A** | — | — | |
| 7.2 → (prose) | Stages of the diagnostic framework | overall contribution | no: one sentence in §7.14 | — | — | duplicated Tables 1.1 and 8.1 |
| 8.1 → 8.1 | Contributions and boundaries | contribution boundaries | **A** | — | — | |

## Appendix tables (18 → 8)

| Table (old → new) | Purpose | Main text? | Appendix? | Repository only? | Reason |
|---|---|---|---|---|---|
| A.1 → A.1 (narrowed) | Pruning units: weights, shares, MACs | — | **B** | weight shapes and module names | verifies parameter/MAC shares cited in Chs 3–4 |
| A.2 Dense references over three seeds | — | — | — | **C** | means ± SD already in Table 5.1; checkpoints and hashes in Table 4.4 |
| A.3 Recovery protocol keys | — | — | — | **C** | protocol stated in §4.13; raw config file in the repository |
| A.4 Pre-registration commits | — | — | — | **C** | identical to Table 4.10 |
| A.5 Software environments | — | — | — | **C** | identical to Table 4.11 |
| B.1 Phase 3 per-setting, per-method means (54 rows) | — | **B** (pivoted: 6 rows, 8 columns → B.1) | full means/SDs | absolute accuracies behind the paired differences of Table 5.2 |
| B.2 → B.2 | Sensitivity vectors | — | **B** | — | verifies the prior coefficients (Eq. 3.11) |
| C.1 Phase 4 search quality by variant (30 rows) | — | — | — | **C** | summarised in Table 5.3 and §5.6 |
| C.2 Phase 5 selected policies (18 rows) | — | — | — | **C** | summarised in Tables 5.4–5.6 |
| C.3 Phase 6 recovery by method (24 rows) | — | **B** (pivoted: recovery gains only → C.1) | before/after per method | before/after values already in Table 5.7 |
| D.1 → D.1 | H0 calibration | — | **B** | — | verifies the measurement-configuration choice |
| D.2 Latency-table session log (24 rows) | — | **B** (summarised per table → D.2) | per-session log | drift limit verification without the full log |
| D.3 Proxy validation | — | — | — | **C** | values in §6.6 prose |
| D.4 ResNet-8 calibration | — | — | — | **C** | identical content in Table 6.2 |
| E.1 → E.1 | Phase 7A energy session summaries | — | **B** | per-block records | verifies 5-of-5 consistency and Wilcoxon p |
| E.2 Phase 7B recovered models (11 rows) | — | — | — | **C** | primary and comparators in Table 6.4 |
| E.3 → E.2 | Stability pilots and qualification | — | **B** | — | verifies the energy-unresolved verdict (CV 19.92% > 3%) |
| E.4 Qualification blocks | — | — | — | **C** | descriptive diagnostics only |

## Summary

| | Before | After |
|---|---|---|
| Chapter tables | 43 | 32 |
| Appendix tables | 18 | 8 |
| **Total** | **61** | **40** |
| Large tables (> ½ page, MIST 550) | 15 | 8 |
| Small tables (MIST 200) | 46 | 32 |
| Tables moved to repository-only (class C) | — | 12 appendix tables (A.2–A.5, C.1, C.2, D.3, D.4, E.2, E.4, plus the full forms of B.1 and C.3) |
| Chapter tables converted to prose or merged (values kept) | — | 11 (2.1, 3.2, 3.6, 3.7, three criteria tables, 6.2, 6.3, 6.7, 7.2) |

The repository statement is in Appendix A.2. Every essential evidence table named in the brief is retained: Phase 3
baselines (5.2), sensitivity (5.8), archive retention (5.3), reward alignment (5.4), Phase 5 (5.5, 5.6), Phase 6
recovery (5.7), Paper 3 latency (6.1, 6.2), Phase 7A energy (6.3) and Phase 7B memory/MAC/accuracy (6.4).
