# QA — Abstract and Keywords

The Abstract (`abstract.md`) was written after re-reading Chapters 1, 5, 6, 7 and 8, as Part B requires. It
contains no citation, no phase name and no new claim. Every statement below is traced to the chapter section that
already makes it.

## 1. Length

| Measure | Value | Target |
|---|---|---|
| Body words (heading "Abstract", bold run-in labels' asterisks and the keywords line excluded; ±, –, % attached to their numbers) | **349** | 250–350 |
| Including heading and keywords line (`tools/assemble.py` count) | 369 | — |

During QA, "each layer unit" was changed to "each layer or layer group" (`terminology_audit.md` §7). This moved the
count from 347 to 349. The Abstract is at the upper end of the range. If the university sets a lower limit (for
example 300 words), the "Findings" bullets on Sensitivity and Retention are the first candidates to merge.

## 2. Required content (Part B, items 1–8)

| # | Element | Where in the Abstract |
|---|---|---|
| 1 | Problem | Paragraph 1: layer-wise allocation is difficult; sparsity alone does not establish deployment benefit |
| 2 | Proposed framework | Paragraph 2: PPO agent assigning per-layer(-group) ratios, guided by sensitivity |
| 3 | Main methodological developments | Paragraph 2 list: centred prior; archive selection; accuracy-constrained reward; equal recovery |
| 4 | Experimental scope | Paragraph 3: CIFAR-10/100, three small CNNs; structured extension with direct CPU latency and energy |
| 5 | Strongest core findings | Findings: PPO vs simple allocation (≈31.4 pp); equal recovery within ±0.5 pp |
| 6 | Deployment findings | Findings: latency proxies; structured energy; deployment-oriented agent (63.4% / 36.0% / 1.12 pp) |
| 7 | Negative result / boundary | Findings: LAMP stronger before recovery; sensitivity not final accuracy; test feasibility about two thirds; 7A accuracy loss beyond limit; 7B criterion missed, energy unresolved |
| 8 | Final conclusion | "Conclusion": pruning as a chain of connected problems, each needing its own evidence |

## 3. Permitted statements (Part B list) → Abstract wording → source

| Permitted statement | Abstract wording | Source |
|---|---|---|
| PPO learns non-uniform policies | "PPO learned non-uniform allocations" | Ch. 5 §5.3; Ch. 8 RQ1 |
| Substantially outperforms Uniform and Random | "substantially outperformed uniform and random allocation" | Ch. 5 §5.3; Ch. 8 RQ1 |
| Max PPO–Uniform ≈ +31.4 pp | "by up to approximately 31.4 percentage points of test accuracy over uniform pruning" | Ch. 5 §5.3, Table 5.2 (C10 SimpleCNN, +31.37; exact 31.365; `global_numerical_audit.md`) |
| LAMP stronger before equal recovery | "LAMP … remained more accurate before equal recovery" | Ch. 5 §§5.3, 5.8; Ch. 8 RQ1 |
| Sensitivity improves exploration, not final accuracy | "improved search and exploration measures, not final accuracy" | Ch. 5 §§5.4–5.5; Ch. 8 RQ2 (Part A wording) |
| Archive improves retention of validation policies | "Archive selection restored their retention on validation ranking" | Ch. 5 §5.6; Ch. 8 RQ3 |
| Constrained reward reduces destructive over-pruning | "reduced destructive over-pruning" | Ch. 5 §5.8; Ch. 8 RQ4 |
| Equal recovery narrows PPO–LAMP differences | "narrowed the PPO–LAMP differences to within a pre-registered ±0.5 percentage-point decision margin in every setting" | Ch. 5 §5.10; Ch. 8 RQ5 ("in all six settings … decision margin") |
| Parameter reduction weak latency proxy vs MACs | "Parameter reduction proved a weak predictor of measured latency compared with reduction in … MACs" | Ch. 6 §6.5; Ch. 8 RQ6 |
| Structured pruning can reduce measured CPU package energy | "reduced directly measured CPU package energy, but with accuracy losses beyond the pre-registered limit" | Ch. 6 §§6.7–6.8; Ch. 8 §8.4 (Part A wording) |
| 7B 63.4% memory, 36.0% MACs, 1.12 pp | "reduced parameter memory by 63.4% and MACs by 36.0% at a 1.12 percentage-point validation-accuracy cost" | Ch. 6 §§6.11–6.12 (VAL-SELECT) |
| 7B direct energy unresolved | "its direct energy remained unresolved because the measurement setup failed its stability qualification" | Ch. 6 §6.13; Ch. 8 RQ6 |

There are two statements beyond the list, both already made in the approved chapters:
- **Test-feasibility transfer, "about two thirds".** 81/120 = 67.5% (Ch. 5 §5.9; Ch. 7 accepted variant; see
  `global_numerical_audit.md` notes).
- **"It missed its accuracy criterion".** Ch. 1 §1.7, Ch. 6 l. 436 ("0 of 20 seeds … lost more than the allowed
  0.5 pp on VAL-SELECT"), Ch. 7 §7.10.

## 4. Forbidden statements (Part B "Do NOT say")

Each was searched for in `abstract.md`, and none occurs:
- PPO beats LAMP;
- PPO is equivalent to LAMP / statistically equivalent;
- sensitivity improves final accuracy;
- archive improves test accuracy;
- Phase 7A accuracy preserving;
- Phase 7B reduces measured energy;
- Phase 7B reduces measured latency;
- green;
- environmentally friendly;
- carbon;
- state-of-the-art.

The "±0.5" sentence calls the margin a "decision margin", not equivalence. The Phase-7B sentence reports memory,
MACs and accuracy only, and says that its energy is unresolved. Latency is not mentioned for 7B. The global claim
audit (`global_claim_audit.md`) includes the Abstract and reports no unbounded hit in it.

## 5. Style checks

| Check | Result |
|---|---|
| Citations | none |
| Long list of phase names | none (no phase is named) |
| Headline numbers (target 4–6) | 5: ≈31.4 pp; about two thirds; ±0.5 pp; 63.4% / 36.0%; 1.12 pp |
| Split named for the 7B accuracy cost | "validation-accuracy cost" (VAL-SELECT in Ch. 6). The split name is omitted in the Abstract to avoid jargon; there is no cross-split comparison in the Abstract |
| Acronyms expanded | PPO (Proximal Policy Optimisation), MACs (multiply–accumulate operations); LAMP is used as a method name, expanded in `list_of_abbreviations.md` |
| Spelling | British ("Optimisation", "centred", "realises"), as in the chapters; see `terminology_audit.md` §1 for the title-spelling advisory |

## 6. Keywords (Part C)

There are eight keywords (target 5–8):
1. neural network pruning;
2. reinforcement learning;
3. proximal policy optimisation;
4. layer-wise pruning;
5. sensitivity-aware pruning;
6. structured pruning;
7. model compression;
8. deployment efficiency.

All are terms that the thesis uses for its own work. "Green AI", "sustainable AI" and "energy-efficient AI" were
deliberately not used.

## 7. Verdict

**PASS.** The Abstract is within length, covers all eight required elements, uses only permitted or
chapter-supported statements, and contains no forbidden claim. It is ready for supervisor review. Its word limit
should be re-checked against the official regulations when they are supplied.
