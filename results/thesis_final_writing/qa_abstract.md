# QA — English and Bangla Abstracts (final MIST stage)

The English Abstract (`abstract.md`) was re-audited after the compression pass and rewritten into the MIST form
(TP.11, §3.3.5): continuous prose paragraphs instead of bullets, with objective and scope, methods, brief results, the
key conclusion and the novelty. No scientific claim was added, removed or strengthened compared with the approved
previous version; every statement is traced below.

## 1. Length

| Measure | Value | MIST limit |
|---|---|---|
| English Abstract, body words (heading and keywords excluded; `tools/wordcount.py` rule) | **390** | about 300–600 words; at most 1.5 pages for a Master's thesis |
| Rendered length (`final_thesis_MIST.pdf`) | see `final_completion_report.md` | ≤ 1.5 pages |

## 2. Required content (MIST §3.3.5 and the final brief)

| Element | Where |
|---|---|
| Objective and scope | Paragraph 1: the allocation problem; the framework developed; CIFAR-10/100, three small CNNs, one laptop CPU |
| Methods | Paragraph 2: PPO agent with sensitivity guidance; centred prior; archive selection; constrained reward; equal recovery; structured deployment extension with direct latency and energy measurement |
| Brief results | Paragraph 3: the headline results, with five numbers (≈31.4 pp; about two thirds; ±0.5 pp; 63.4% / 36.0%; 1.12 pp) |
| Key conclusion | Paragraph 4: an improvement at one stage does not guarantee success at the next; each stage must be validated |
| Novelty / contribution | Paragraph 4: not a new pruning criterion, but treating adaptive pruning as a chain of connected problems, each examined with its own experiment and evidence (Ch. 1 §1.7 novelty statement) |

## 3. Prohibited content (MIST §3.3.5)

| Item | Present? |
|---|---|
| Discussion of results | No (results are stated, not interpreted beyond the Chapter-8 conclusion) |
| Citations or reference numbers | No |
| Tables or figures | No |
| Non-standalone abbreviations | No: PPO, LAMP and the multiply–accumulate operations are written out; CPU and the dataset names CIFAR-10/100 are standalone |

## 4. Statement trace

| Abstract statement | Source |
|---|---|
| Learned allocations substantially outperformed uniform and random allocation, up to ≈31.4 pp over uniform pruning | Ch. 5 §5.3, Table 5.2 (+31.37; exact 31.365); Ch. 8 RQ1 |
| LAMP remained more accurate before equal recovery | Ch. 5 §§5.3, 5.8; Ch. 8 RQ1 |
| Sensitivity improved search and exploration measures, not final accuracy | Ch. 5 §§5.4–5.5; Ch. 8 RQ2 |
| High-ranking policies found but rarely ended on; archive restored retention on validation ranking | Ch. 5 §5.6; Ch. 8 RQ3 |
| Constrained reward reduced destructive over-pruning; validation feasibility held on test in about two thirds of runs | Ch. 5 §§5.8–5.9 (81/120 = 67.5%); Ch. 8 RQ4 |
| Equal five-epoch recovery narrowed PPO–LAMP differences to within the pre-registered ±0.5 pp decision margin in every setting | Ch. 5 §5.10; Ch. 8 RQ5 |
| Parameter reduction a weak predictor of measured latency compared with MAC reduction | Ch. 6 §6.5; Ch. 8 RQ6 |
| Structured pruning reduced directly measured CPU package energy, with accuracy losses beyond the pre-registered limit | Ch. 6 §§6.7–6.8; Ch. 8 §8.4 |
| Deployment-oriented structured agent: 63.4% memory, 36.0% MACs, 1.12 pp validation-accuracy cost; criterion missed; energy unresolved (stability qualification failed) | Ch. 6 §§6.11–6.13; Ch. 8 RQ6 |
| Novelty and conclusion | Ch. 1 §1.7 (novelty); Ch. 8 §8.4 |

The global numerical audit (`global_numerical_audit.md`) includes the Abstract and passes for claims 1, 6 and 14. The
global claim audit includes the Abstract and finds no forbidden phrase in it.

## 5. Bangla Abstract (`abstract_bangla.md`)

| Check | Result |
|---|---|
| Faithfulness | Paragraph-by-paragraph translation of the final English Abstract; same four paragraphs, same claims, same numbers (written in Bangla numerals: ৩১.৪, ±০.৫, ৬৩.৪%, ৩৬.০%, ১.১২) |
| No added or removed claims | Checked sentence by sentence against the English text; boundaries kept ("চূড়ান্ত নির্ভুলতা নয়" for "not final accuracy"; "অমীমাংসিত" for "unresolved"; "নির্ভুলতা-মানদণ্ড পূরণ করেনি" for "missed its accuracy criterion") |
| Technical terms | English retained in parentheses where Bangla has no settled term (MIST §3.3.5, TP.12): neural network pruning, reinforcement learning, proximal policy optimisation (PPO), structured pruning, latency, package energy, multiply–accumulate operations, LAMP, and others |
| Register | Academic and plain, not literary or promotional |
| Title | The thesis title is kept in English on the Bangla page, as in MIST TP.12 |
| Font | Kalpurush (Unicode) at 14 pt, English terms in Times New Roman 12 pt. MIST suggests Sutonny OMJ, which is not installed on this machine (`mist_template_audit.md`) |

## 6. Keywords

The eight keywords are kept after the English Abstract: neural network pruning; reinforcement learning; proximal policy
optimisation; layer-wise pruning; sensitivity-aware pruning; structured pruning; model compression; deployment
efficiency. MIST does not require keywords; they are harmless and can be removed if the department prefers.

## 7. Verdict

**PASS.** Both abstracts meet MIST §3.3.5 and contain only chapter-supported statements.
