# QA report: Chapter 8 (Conclusion and Future Work)

- **File:** `chapter_08_conclusion.md`.
- **Status:** **FROZEN (v1, 2026-11-05)**, pending the user's review.
- **Basis:** synthesis of the frozen Chapters 5–7. It adds no new evidence, analysis, effect size or reference.

## 1. Word count

**2,708** counted words, of which about **170** are inside the contribution table, so there are about **2,540 words
of prose**. That is within the 2,500–4,000 guide and shorter than the Discussion (about 5,330); about 6–7 pages.

## 2. Section structure

| § | Title |
|---|---|
| 8.1 | Chapter Overview and Summary of the Research (merged, as permitted; the 11-step progression; core vs extension) |
| 8.2 | Conclusions by Research Question (RQ1–RQ6, each with its question and a closing statement) |
| 8.3 | Main Contributions (table: contribution and boundary) |
| 8.4 | Overall Thesis Conclusion (chain of decisions; five examples; implications for practice) |
| 8.5 | Limitations (groups A–G) |
| 8.6 | Future Work (short-, medium- and long-term, 12 items) |
| 8.7 | Final Remarks |

The requested 8.1–8.8 structure is kept, with 8.1 and 8.2 merged; later sections are renumbered accordingly.

## 3–8. RQ conclusions

| RQ | Conclusion | Numbers used (all audited) |
|---|---|---|
| RQ1 | Meaningful non-uniform allocations. Strongly better than Uniform (5/6) and Random (6/6); mixed against Global and ERK; LAMP stronger before equal recovery (5/6, then 6/6). Not universally better | ≈ +31.4 pp (+31.37 convention, C10 SimpleCNN, test) |
| RQ2 | Across Phases 3–5, improves one or more search or exploration measures, without reliable final-accuracy gains. Centring gave stronger evidence of a layer-relative search effect, with one measure favouring a conservative control. No reliable final-accuracy gain. "The main measurable contribution of sensitivity was to exploration rather than to final accuracy" — "not an absence of value" | — |
| RQ3 | Visits but does not retain. Archive restores validation-rank retention (80–100% top-10, 100% top-20), mainly through the archive rule. Test effect mixed. "Solves a retention problem, not a generalisation problem" | 1,296; 0% in five settings, 5% in one; 95–100%; 80–100%; 100% |
| RQ4 | Reward controls aggressiveness. Better search exposed the fixed-rate reward. The constraint reduced over-pruning, but was slightly worse on SimpleCNN, LAMP stayed stronger, and test feasibility was 67.5%. "Aligning the optimisation objective and ensuring generalisation are separate issues" | 7.86; 7.32 → 1.18; 67.5% |
| RQ5 | Equal recovery narrows differences. PPO–LAMP within the ±0.5 pp CI margin in all six settings; **not a formal equivalence test**; LAMP ahead in three; PPO–Global ≤ 0.12 pp. "Compare under equal post-pruning recovery budgets" | 0.26–0.53 → 0.00–0.22; 0.12 |
| RQ6 | Not automatically. Parameter reduction weak and MACs strong as latency proxies; ResNet-8 1.098× proxy-feasible / 1.089× confirmed, neither ≥ 1.10×. Phase 7A: −22.8% / −13.4% energy at −5.08 / −3.24 pp; Uniform better on SimpleCNN. Phase 7B: −63.4% memory, −36.0% MACs, −1.12 pp, 0/20, latency unmeasured, energy unresolved. "Every deployment dimension must be measured separately … interpreted together with its accuracy cost" | 0.31–0.55; 0.89–0.98; 1.098; 1.089; 1.10; 22.8; 13.4; 5.08; 3.24; 63.4; 36.0; 1.12; 0/20 |

**Number check.** A script extracted every numeric token in Chapter 8 (46). All occur in the audited Chapters 4–7, so
no new number was introduced.

## 9. Contribution summary

C1–C6 are presented in a table, each with the requested boundary:

| # | Boundary |
|---|---|
| C1 | not superior to LAMP |
| C2 | no consistent final-accuracy gain |
| C3 | validation-retention benefit, mixed test effect |
| C4 | better control of over-pruning; test feasibility not guaranteed |
| C5 | not formal equivalence |
| C6 | Phase-7A energy savings carry an accuracy cost; Phase-7B energy unresolved |

The pre-registration wording follows the Chapter-7 refinement: "decision criteria were fixed in advance across its
later experimental stages".

## 10. Thesis-level conclusion

Adaptive pruning is "a chain of connected decisions" (six links). "Improvement at one stage does not guarantee success
at the next", illustrated by five examples:
- better search exposed a poor reward;
- validation selection did not transfer to test;
- lower parameter count did not mean lower latency;
- lower energy came with accuracy loss exceeding the pre-registered limit;
- strong compression remained unconfirmed as an energy saving.

The section adds five implications for practice and states the multi-stage view as the central conclusion.

## 11. Limitations

Groups A–G as requested:
- **A. Scale:** CIFAR; no ImageNet; an enumerable space.
- **B. Models:** three small CNNs; no transformers or LLMs; a subset of settings in the extension.
- **C. Optimisation:** cost; reward dependence; LAMP stronger; sensitivity does not improve final accuracy.
- **D. Evaluation:** validation-to-test transfer; threshold and split sensitivity; a single recovery seed in the later
  stages; fixed recovery.
- **E. Hardware:** one CPU; batch 1 and one thread; no GPU, mobile or accelerator.
- **F. Energy:** package only; no whole-system energy; failed Phase-7B qualification; no Phase-7B energy result.
- **G. Sustainability:** no carbon, lifecycle or training and search energy assessment.

The section refers to §7.13 rather than repeating it.

## 12. Short-term future work

1. native structured RL pruning, including residual-coupled groups;
2. hybrid LAMP + RL;
3. better feasibility proxies and selection margins;
4. stable external energy instrumentation (logging wall meter, thermal control, fixed clocks, per-campaign
   qualification).

## 13. Medium-term future work

5. multi-objective Pareto optimisation, compared with deterministic baselines;
6. improved generalisation constraints (multiple validation subsets, uncertainty-aware thresholds, robust
   formulations);
7. larger architectures (ResNet-18/34/50, MobileNet, EfficientNet);
8. larger datasets (ImageNet-scale);
9. edge and mobile deployment (mobile CPU, ARM, embedded GPU, accelerators);
10. GPU evaluation of the parameter/MAC/latency relationships.

The requested "edge/GPU deployment" theme is split into two items.

## 14. Long-term future work

11. hardware-in-the-loop multi-objective RL, only after stable instrumentation;
12. reliable lifecycle and carbon analysis, only after trustworthy direct energy measurement. "Until then, no carbon
    impact should be claimed."

Each future-work item is tied in its text to the finding or limitation that motivates it.

## 15. LAMP boundary

LAMP was stronger before equal recovery (RQ1), remained ahead in three settings after it (RQ5), and stayed more
accurate than the constrained pipeline (RQ4). In §8.7: "simpler deterministic methods remain stronger for ordinary
compression objectives". There is no PPO-superiority claim.

## 16. Sensitivity boundary

"The main measurable contribution of sensitivity was to exploration rather than to final accuracy", explicitly "not an
absence of value". Centring is described as *designed* to distinguish layer-specific guidance, with "stronger
evidence", not isolation.

## 17. Archive boundary

The archive "solves a retention problem. It does not solve, or guarantee, a generalisation problem". Its test effect is
mixed, with ResNet-8 losses explained as retaining the reward's preferred over-pruned policies.

## 18. Phase 6 equivalence-language check

"Equivalence" appears only in "not a formal statistical equivalence test" (RQ5) and in "do not constitute formal
equivalence" (C5). The positive wording is "fell within the pre-registered ±0.5 percentage-point
confidence-interval decision margin". ✓

## 19. Phase 7B energy boundary

"Its latency was not measured"; "its energy remained unresolved, because the measurement setup failed its stability
qualification" (protocol logic). In the conclusion: "strong compression remained unconfirmed as energy saving".
Limitations F: "no direct energy result for the final structured model". No Phase-7B energy or latency reduction is
claimed.

## 20. Sustainability-language audit

- **Terms used:** "deployment efficiency", "directly measured operational energy", "CPU package energy",
  "sustainability-oriented evaluation of operational inference energy, not an environmental assessment".
- **Absent:** "green AI", "environmentally friendly", "carbon efficient", "reduced carbon footprint", "sustainable AI
  system".
- **"Carbon" (4 hits):** only in limitations G (no carbon accounting) and future work 12 (carbon analysis only after
  reliable measurement; no carbon impact claimed).
- **"Environmental" (1 hit):** "not an environmental assessment".

## 21. Dangerous-claim audit

**Searched:** PPO beats LAMP, superior, state-of-the-art, equivalent to LAMP, sensitivity improves accuracy, archive
improves test accuracy, accuracy-preserving, green, environmentally friendly, carbon reduction, Phase-7B energy or
latency reduction, hardware-aware RL succeeded.

| Hit | Verdict |
|---|---|
| "superior" (C1 table; §8.7) | "not superior to LAMP"; "does not establish PPO as a universally superior pruning method" |
| "equivalen…" (RQ5; C5) | negations (see item 18) |
| "guarantee…" (×5) | negations: archive, constraint, "improvement at one stage does not guarantee success at the next" |
| "lower energy" (§8.4) | "Lower energy came with accuracy loss exceeding the pre-registered limit", a Phase-7A negative example |
| "carbon", "environmental" | see item 20 |
| "state-of-the-art", "accuracy-preserving", "green", hardware-aware RL success | absent. Hardware-aware search: "no hardware-aware search was started" |

**Causal wording.** The only "because" (RQ6, Phase 7B) is protocol logic: the stop rule prevented the measurement.

## 22. Unresolved issues

1. **Citations.** Chapter 8 uses no citations. This is intentional: it synthesises the thesis's own findings.
2. **Next stage, not started as instructed.** The Abstract, list of abbreviations, appendices, the cumulative
   reference audit in final citation order, and final assembly. Final assembly remains blocked by the missing official
   thesis template.
3. **Rounding convention.** The +31.37 / ≈ +31.4 convention must also be used in the Abstract.

## Freeze

Chapter 8 v1 is frozen. Later changes are limited to factual corrections found during the final QA, and will be logged
here.

## Correction log

| Date | Location | Change | Authority |
|---|---|---|---|
| 2026-11-05 | §8.2 RQ2 | "It improves the search … In every phase where it was tested, sensitivity guidance improved the search" → "Across Phases 3–5, sensitivity guidance improved one or more search or exploration measures … these improvements did not translate into reliable gains in final predictive accuracy". One Phase-4 accuracy-based exploration measure moved in the opposite direction. | User authorisation (final-stage task, Part A.1) |
| 2026-11-05 | §8.2 RQ6 | Already read "the best confirmed five-epoch-feasible calibration policy reached 1.089×", kept separate from the 1.098× proxy-feasible candidate. No change needed. | Part A.2 (verified) |
| 2026-11-05 | §8.4 | "Lower energy came with unacceptable accuracy loss" → "Lower energy came with accuracy loss exceeding the pre-registered limit". | Part A.3 |

No number changed.
