# QA report: Chapter 1 (Introduction)

- **File:** `chapter_01_introduction.md`.
- **Status:** **FROZEN (v1, 2026-11-05)**, pending the user's review.
- **Basis:** written after Chapters 2–7 were frozen. It describes the completed thesis, not the original plan.

## 1. Word count

**3,427** counted words, of which about **140** are inside two small tables, so there are about **3,290 words of
prose**. That is slightly below the 3,500–5,000 guide. The first draft (about 2,950 words) was extended only with
substantive content:
- why a learned allocator is attractive and what is hard about it;
- the research approach;
- the problem-to-design control table.

The estimate is about 8–9 pages.

## 2. Section structure

| § | Title |
|---|---|
| 1.1 | Background (pruning, masked vs structured, layer-wise allocation, magnitude / LAMP / sensitivity / RL; why a learned allocator and what is hard about it) |
| 1.2 | Motivation (Problems A–E; table of how the design controls each) |
| 1.3 | Problem Statement (with the research approach) |
| 1.4 | Research Gap (six gaps → RQs) |
| 1.5 | Research Questions (RQ1–RQ6; core vs extension; sequential origin) |
| 1.6 | Research Objectives (general objective, O1–O6) |
| 1.7 | Research Contributions (C1–C6, novelty, Table 1.1, result preview) |
| 1.8 | Scope and Boundaries |
| 1.9 | Thesis Organisation |
| 1.10 | Chapter Summary |

## 3. Problem statement

Fixed-rule and single-saliency pruning versus RL allocation. RL's effectiveness depends on four things: search
guidance, policy retention, reward alignment and recovery. Compression metrics do not by themselves establish
deployment efficiency. The thesis therefore studies five follow-on questions: sensitivity, retention, reward,
recovery and deployment.

## 4. Research gap

The six gaps of Chapter 2 §2.14, each mapped to an RQ:
1. adaptive vs deterministic allocation at matched sparsity → RQ1;
2. exploration vs final-model effects of sensitivity → RQ2;
3. retention of strong visited policies → RQ3;
4. incentives for excessive sparsity in rewards → RQ4;
5. comparisons without equal recovery → RQ5;
6. compression vs deployment efficiency, including measurement reliability → RQ6.

The weak claim "no previous study uses PPO for pruning" is explicitly *not* made; the text states that RL has been
applied to pruning [@he2018amc].

## 5. RQ1–RQ6 wording

Reproduced verbatim from the task brief and consistent in meaning with `final_research_questions.md`. RQ2's frozen
version also mentions "as a prior or as a state feature". The brief's shorter wording keeps the meaning. Sensitivity
entered only as a prior in Phases 3–5 and as both a prior and a state feature in Phase 7B (Chapter 3), and nothing in
Chapter 1 contradicts this. RQ1–RQ5 are labelled core framework and RQ6 the later deployment extension. The
sequential origin of the questions is stated in §1.5.

## 6. Objective-to-RQ mapping

| Objective | RQ | Completed experiment |
|---|---|---|
| O1 | RQ1 | Phase 3 (and Phases 4–5 LAMP comparisons) |
| O2 | RQ2 | Phases 3, 4, 5 |
| O3 | RQ3 | Phase 4 |
| O4 | RQ4 | Phases 4, 5 |
| O5 | RQ5 | Phase 6 |
| O6 | RQ6 | Paper 3, Phases 7A, 7B, energy confirmation |

Each objective names its controls or baselines as actually used, for example "shuffled, constant, magnitude-matched
and sign-reversed controls" for O2.

## 7. Contribution-to-evidence mapping

| Contribution | Evidence | Claim boundary in the text |
|---|---|---|
| C1 framework vs 5 baselines | Ch. 5 §5.3; Phases 4–5 | "LAMP remains stronger before recovery" |
| C2 sensitivity: exploration vs outcome | Ch. 5 §§5.4, 5.5, 5.8 | "contribution to final predictive accuracy was not detectable" |
| C3 retention, archive | Ch. 5 §5.6 | "effect on test accuracy is mixed" |
| C4 reward misalignment, constraint | Ch. 5 §§5.7–5.9 | "validation feasibility did not reliably transfer to the test set" |
| C5 equal recovery | Ch. 5 §5.10 | "differences became small", without "equivalent" |
| C6 deployment analysis | Ch. 6 | "reported including its failure" (measurement qualification) |

Table 1.1 in the chapter gives RQ → objective → contribution → evidence chapter.

## 8. Novelty statement

"The novelty lies in **treating adaptive pruning as a sequence of connected optimisation and validation problems,
rather than evaluating only the final compressed model**."
- **Stages:** allocation, exploration, policy retention, reward alignment, recovery, deployment translation and
  hardware validation.
- **What it does not rest on:** PPO beating LAMP, sensitivity improving accuracy, Phase 7B meeting its target, or a
  Phase-7B energy reduction.
- **Pre-registration wording:** "each with decision criteria fixed before the confirmatory data were collected". This
  is consistent with the Chapter-7 refinement (pre-registered criteria across the later experimental stages). Phase
  1, the audit of the earlier pipeline, is not described as pre-registered.

## 9. Result-preview claims

Each claim was checked against Chapters 5–6:

| Preview claim | Source | Check |
|---|---|---|
| PPO ≫ Uniform (up to about 31.4 pp test) and Random | Ch. 5 §5.3 (+31.37 pp, ≈ +31.4) | ✓, using the +31.37 convention ("about 31.4") |
| LAMP stronger before equal recovery | Ch. 5 §§5.3, 5.7, 5.8 | ✓ |
| Sensitivity improves exploration, not final accuracy | Ch. 5 §§5.4–5.5, 5.8 | ✓ |
| Archive restores validation-rank retention; test effect mixed | Ch. 5 §5.6 | ✓ |
| Constrained reward reduces destructive over-pruning; test feasibility about two thirds | Ch. 5 §§5.8–5.9 (67.5%) | ✓ |
| Equal recovery narrows PPO–LAMP differences within the CI margin in every setting; LAMP ahead in 3 | Ch. 5 §5.10 | ✓ |
| Parameter reduction is a weak latency predictor compared with MACs | Ch. 6 §6.5 | ✓ |
| Structured pruning reduced measured CPU package energy, with a substantial accuracy cost; Uniform better on one network | Ch. 6 §§6.7–6.8 | ✓ |
| Phase 7B: memory −63.4%, MACs −36.0%, −1.12 pp; criterion missed; energy unresolved | Ch. 6 §§6.11–6.13 | ✓ |

**Numbers.** All the numbers in Chapter 1 (31.4, 63.4, 36.0, 1.12, 0.5, 1,296 and small counts) occur in the audited
Chapters 2, 4, 5 or 6 (checked by script).

## 10. LAMP positioning

LAMP is introduced in §1.1 as a layer-adaptive deterministic method. It is described as "the stronger method before
equal recovery" (preview) and "a strong baseline, … more accurate than PPO before recovery in nearly every setting"
(§1.8). PPO is not presented as universally better.

## 11. Sensitivity positioning

Sensitivity is "primarily an exploration aid" (§1.8); its contribution is "to the *search*" (C2); and "its
contribution to final predictive accuracy was not detectable".

## 12. Archive positioning

The archive is a "validation-retention mechanism, not a generalisation mechanism" (§1.8). It "restores retention in
terms of validation rank; its effect on test accuracy is mixed" (C3).

## 13. Recovery positioning

The ±0.5 pp rule is a "pre-registered confidence-interval decision rule, and **not** a formal statistical equivalence
test" (§1.8). The differences "became small", and LAMP remained "slightly ahead in three" settings.

## 14. Sustainability positioning

The text uses "sustainability-oriented evaluation of operational inference energy, not an environmental assessment".
Carbon-footprint, lifecycle and environmental-impact claims are explicitly *not* established (§1.8).
- **Phase 7A:** measured energy reduction with a substantial accuracy cost.
- **Phase 7B:** an improved trade-off, with energy unresolved.

There is no "green", "eco-friendly" or "sustainable AI system" wording.

## 15. Phase 7B boundary

- **Achieved:** memory −63.4%, MACs −36.0%, −1.12 pp.
- **Not met:** it missed the pre-registered accuracy criterion.
- **Energy:** direct energy unresolved, because the measurement setup failed its stability check.
- **No reduction claimed:** "No latency or energy reduction is claimed for it" (§1.8).

## 16. Scope

| Included | Not established |
|---|---|
| CIFAR-10/100; SimpleCNN, LeNet-5 (CIFAR adaptation), ResNet-8; PPO; masked core with exhaustive landscapes; structured extension; memory, MACs, batch-1 CPU latency, CPU package energy on one laptop CPU | ImageNet; larger CNNs, transformers, LLMs; GPU, mobile, edge, cloud; whole-system, wall, DRAM and training energy; carbon, lifecycle, environmental impact |

## 17. Thesis organisation

Chapters 1–8 are described concisely in §1.9 and match the frozen chapters. Chapter 8, Conclusion and Future Work, is
not yet written.

## 18. Dangerous-claim audit

**Searched:** beats LAMP, superior, equivalent, sensitivity improves accuracy, archive improves test,
accuracy-preserving, green, environmentally, carbon, Phase-7B energy or latency reduction, state-of-the-art,
unprecedented, groundbreaking.

| Hit | Verdict |
|---|---|
| "carbon-footprint" (§1.8) | in the "not established" list |
| "environmental" (§1.8, two hits) | "not an environmental assessment"; "environmental-impact claim" in the not-established list |
| "do not guarantee" (§1.8) | negation |
| "equivalence" (§1.8) | "**not** a formal statistical equivalence test" |
| "superiority" (§1.10) | "It does not claim superiority over existing methods" |
| Phase-7B energy or latency reduction | only "No latency or energy reduction is claimed for it" |
| "state-of-the-art", "unprecedented", "groundbreaking", "revolutionary" | absent |

**Causal wording.** The only "because" (§1.8, Phase 7B) states protocol logic: the stop rule prevented the
measurement.

## 19. Consistency with Chapters 5–7

- RQ answers implied in the preview match Chapter 5 §5.11 and Chapter 6 §6.15.
- The novelty statement matches the refined Chapter 7 §7.14.
- The boundaries match the Chapter 7 limitations (§7.13) and `claim_matrix.md`.
- Three wording choices follow the corrections made in later chapters:
  - the ResNet-8 units are described generally, with no "entire residual stage" wording;
  - the CI rule is not called equivalence;
  - there is no "at most 1.098×" headroom claim; Chapter 1 does not quote headroom values at all.
- **No contradiction found.**

## 20. Unresolved issues

1. **Prose length.** About 3,290 words, slightly below the 3,500 guide (item 1).
2. **Citation-number assembly.** Chapter 1 is cited first in the final document, so the IEEE numbers will be
   reassigned at Stage 10 (the cumulative reference audit, not rebuilt now).
3. **Template.** Still blocked: no official thesis template.

## Citations

18 references, all verified and all already used in Chapter 2: `desislavov2023trends`, `luccioni2024powerhungry`,
`menghani2023efficient`, `lecun1989obd`, `han2015learning`, `cheng2024pruningsurvey`, `he2024structuredsurvey`,
`hoefler2021sparsity`, `lee2021lamp`, `he2018amc`, `schulman2017ppo`, `engstrom2020implementation`,
`andrychowicz2021onpolicy`, `renda2020rewinding`, `le2021retraining`, `dehghani2022misnomer`,
`pan2022misspecification` and `blalock2020state`. `schulman2017ppo` is the permitted arXiv exception. The reference
audit was not rebuilt.

## Freeze

Chapter 1 v1 is frozen. Later changes are limited to citation-number assembly or factual corrections, and will be
logged here.
