# Final thesis integration summary

**Thesis:** "A Reinforcement Learning Framework for Layer-Wise Adaptive Neural Network Pruning with Sensitivity-Aware
Optimization"

- Branch: `thesis-final-integration`, base `6b275096`.
- No experiment was run, and no frozen result or earlier branch was modified for this package.
- Every number was re-read from the frozen result files (`result_freeze.md`).

## Package index

| File | Content |
|---|---|
| `result_freeze.md` | Every phase: frozen numbers, verdicts, allowed and forbidden claims, source files, commits; narrative-vs-file corrections (§11) |
| `final_research_questions.md` | RQ1–RQ6; expected relationships H1a–H6c with frozen outcomes; untested questions |
| `final_contributions.md` | C1–C6 with evidence and boundaries; cross-cutting pre-registration contribution |
| `chapter_structure.md` | Thesis story; methodology flow (core framework vs deployment extension); 8 chapters with subsections, phases, tables, figures, pages |
| `results_mapping.md` | Results subsections → RQ, phase, strongest numbers, negative results that must stay visible |
| `table_shortlist.md` | 11 main tables, 11 appendix tables, omitted tables |
| `figure_shortlist.md` | 11 essential and 9 optional figures, with sources; figures not to produce |
| `claim_matrix.md` | 15 claims: evidence, allowed wording, forbidden wording |
| `sustainability_positioning.md` | Three-level positioning (compression / computation / operational energy); recommended conclusion paragraph; terms |
| `limitations_future_work.md` | 19 limitations; 10 future-work items, each tied to a limitation |
| `novelty_statement.md` | Short novelty statement and why it is analytical, not a leaderboard claim |
| `conceptual_conclusion.md` | Conceptual conclusion draft |

## Corrections to the working narrative

These are fixed in all documents; see `result_freeze.md` §11.

1. **Phase 3 sparsity.** Baselines were matched to each PPO run's own sparsity (mean 9.5–55.1%), not to a "high
   sparsity" target. Uniform was beaten in 5/6 settings, not 6/6.
2. **Phase 3 baselines.** There were no "heuristic" baselines: they were Uniform, Random, Global magnitude, ERK and
   LAMP. Global was only MODERATE and ERK LIMITED.
3. **LAMP.** LAMP was *significantly* better before fine-tuning in 5/6, 6/6 and 6/6 settings (Phases 3, 4, 5), and
   still ahead in 3/6 after fine-tuning.
4. **Phase 4 archive.** It improves retention on validation ranking only; the test effect was mixed and criterion D
   was not met. Phase 4's accuracy-based exploration endpoint for sensitivity was reversed.
5. **Phase 6.** It did not include Uniform. "Equivalence" is a CI-inside-±0.5 rule, not a formal test.
6. **Paper 3.** SimpleCNN headroom (2.61×) is unconstrained. The NO-GO causes differ: SimpleCNN failed proxy validity
   (ρ 0.844); ResNet-8 failed latency room (1.098× vs 1.10×).
7. **Phase 7A accuracy losses (5.08 / 3.24 pp).** These are VAL-RL; no method of any kind met ±0.5 pp.
8. **Phase 7B accuracy loss (−1.12 pp).** This is VAL-SELECT, and the comparison with Phase 7A should be made on the
   same split (7A PPO −4.00 pp VAL-SELECT).
9. **Literature review.** It is not in the repository; it is a writing task.
10. **"green" in repository names.** It must not appear in thesis prose.

## Strongest results to emphasise

- PPO vs Uniform: STRONG, up to +31.4 pp at the same sparsity. PPO vs Random: STRONG 6/6.
- The retention diagnosis: terminal top-10 reach 0% vs archive 75–100%.
- The constrained reward: ResNet-8 drop 7.32 → 1.18 pp and 4.05 → 1.16 pp.
- Equal recovery: all Phase-6 criteria MET; PPO–LAMP within ±0.5 pp in 6/6; no regrowth.
- Sparsity is not speed: ρ(params) 0.31–0.55 vs ρ(MACs) 0.89–0.98; 31–39% of pairs sparser but slower.
- Measured energy from structured realisation: −22.8% and −13.4%, 5/5 sessions (Phase 7A).
- Phase 7B: −63.4% memory and −36.0% MACs at −1.12 pp (vs −4.00 pp for structured transfer).

## Negative and partial results that must stay visible

- LAMP is better than PPO before fine-tuning; PPO is never ahead of LAMP after fine-tuning.
- Sensitivity does not improve final accuracy (0/6 in Phase 3), and one Phase-4 exploration endpoint reversed.
- Phase 4 criteria A (reward) and D (final model) were not met; the archive cost 5.66 / 4.00 pp on ResNet-8 test.
- Phase 5 criteria A, C and E were partial, and F-strong was not met; test feasibility was 67.5%.
- Paper 3: no accuracy-feasible headroom for hardware-aware RL, so it was not run.
- Phase 7A: accuracy criterion failed (Case B), and Uniform dominated PPO on SimpleCNN (Case D).
- Phase 7B: GREEN-SUCCESS NO (0/20 seeds); LAMP was the only model within 0.5 pp.
- Phase 7B energy: pilot unstable (CV 4.01%), confirmation unqualified (CV 19.92%), ENERGY-CONFIRMED = NO.

## Main text: tables and figures

- **Tables:** T1 dense references, T2 phase overview, T3 Phase 3 baselines, T4 sensitivity across phases,
  T5 retention, T6 reward alignment and feasibility, T7 Phase 6 PPO–LAMP–Global, T8 Paper 3 correlations,
  T9 Phase 7A, T10 Phase 7B (memory, MACs, accuracy, energy status), T11 three-level sustainability assessment.
- **Figures:** F1 overall framework (new; core and extension separated), F2 sensitivity profile, F3 accuracy vs
  sparsity, F4 allocation heatmap, F5 sensitivity exploration, F6 archive top-k reach, F7 ResNet-8 reward alignment,
  F8 PPO vs LAMP before and after fine-tuning, F9 latency vs parameter and MAC reduction, F10 Phase 7A accuracy vs
  energy, F11 Phase 7B accuracy vs MACs and memory.

## Open items before full writing

1. Literature review (Chapter 2), to be written from scratch.
2. New figures F1 (overall framework), optionally O1 (detailed MDP) and O8 (measurement stability). They are drawn
   from frozen data, with no new experiments.
3. Re-render existing figures in one thesis style; styling only, the plotted data stays the same.
4. Decide whether the thesis cites Paper 3 as the author's own paper or reproduces its analysis. This is an
   authorship and venue question for the author, not a data question.
5. Freeze this package. Full thesis writing starts only after this branch is accepted.
