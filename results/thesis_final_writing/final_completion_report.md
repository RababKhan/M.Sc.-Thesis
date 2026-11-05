# Final thesis completion report

Thesis: *A Reinforcement Learning Framework for Layer-Wise Adaptive Neural Network Pruning with Sensitivity-Aware
Optimization*. Branch `thesis-final-writing`.

In this stage:
- no experiment was run;
- no model was retrained;
- no frozen result or pre-registered criterion was changed;
- the test set was not read.

The approved chapters were changed only in two ways:
- the authorised Chapter-8 refinements;
- the logged insertion of three statistical-method citations in Chapter 4.

**Content assembly is complete. Final institutional formatting remains blocked until the official thesis template or
formatting specification is supplied.**

## 1. Abstract status

Final. `abstract.md` has 349 body words (target 250–350; 369 including the heading and keywords line). It covers all
eight required elements, contains only permitted or chapter-supported statements, and has no forbidden claim. QA:
`qa_abstract.md` (PASS). One QA correction was made: "each layer unit" became "each layer or layer group".

## 2. Keywords

Eight, saved at the end of `abstract.md`:
1. neural network pruning;
2. reinforcement learning;
3. proximal policy optimisation;
4. layer-wise pruning;
5. sensitivity-aware pruning;
6. structured pruning;
7. model compression;
8. deployment efficiency.

## 3. Abbreviation list

`list_of_abbreviations.md` covers:
- general abbreviations;
- method, model and dataset names;
- experiment labels.

Names are expanded only from their source titles. One label ambiguity is recorded: C1–C6 denote contributions in
Chs 1 and 8, while C0–C2 denote Phase-5 conditions in Chs 4–7.

## 4. Appendix plan

`appendix_plan.md` defines eleven curated appendices, A–K. They are drafted in full in `appendices.md` (about 11,200
words), generated from frozen records by `tools/build_appendices.py`, with the source named under every table. Open
presentation items:
- chapter-to-appendix pointers (an approved-chapter edit, not authorised here);
- landscape pages for the 8-column statistics tables;
- appendix table numbering according to the template.

## 5. Total references

98 cited references, IEEE numbered style, in order of first appearance from Chapter 1 (`references_final.md`). Two
verified but uncited candidates were excluded.

## 6. Percentage of references from 2022–2026

50 of 98 = **51.0%**.

## 7. Number of foundational exceptions

48 pre-2022 foundational sources:
- origins of methods, models, datasets, tools and statistical tests;
- one arXiv-only entry, the original PPO paper, a permitted exception.

The three statistical references (Holm 1979, Wilcoxon 1945, Youden 1950) were added and verified at this stage
(`reference_audit_final.md`).

## 8. Citation audit status

**PASS** (`global_citation_audit.md`). There are no missing, unused, mis-numbered or unsupported citations, and 0
unresolved keys in the assembled source. One citation is noted as partly supporting: Ch. 1 Gap 5 cites Blalock et al.
alone, and the recovery-specific support comes from works cited earlier in Ch. 1. No change was made.

## 9. Numerical audit status

**PASS** (`global_numerical_audit.md`):
- 16 cross-chapter claims recomputed from frozen files, with forbidden-variant searches, all PASS;
- per-chapter audits re-run: Ch. 4 100/100, Ch. 5 182/182, Ch. 6 142/142, Ch. 7 no new numbers;
- accepted rounding variants documented.

## 10. Claim audit status

**PASS** (`global_claim_audit.md`). There were 51 hits for forbidden-claim phrases across the Abstract, Chapters 1–8
and the appendices:

| Classification | Hits |
|---|---|
| Negated or bounded | 41 |
| Literal names | 5 |
| Literature | 1 |
| Benign after manual review | 4 |

There are 0 affirmative occurrences.

## 11. Split-governance audit status

**PASS with one advisory** (`split_governance_audit.md`). Six checks were run: configuration I/II attribution, the
"(I)" labels, the 7A/7B criterion splits, same-split 7A-vs-7B comparisons and test-set governance. The advisory: two
sentences (Ch. 7 §7.10, Ch. 8 RQ6) give the 5.08 pp Phase-7A loss as "validation accuracy" without naming VAL-RL. The
value is correct and there is no cross-split comparison. Inserting one word is recommended, but it was not applied.

## 12. Figure inventory

`final_figure_inventory.md` lists 13 figures (3.1; 5.1–5.8; 6.1–6.4). All 13 are cited in the text by number. Neutral
file names are in `figures/final/`, and no path contains "green". Figures 5.1 and 5.2 were re-rendered at 300 dpi from
the unchanged plotting code; the frozen originals are untouched. Figures 6.1–6.2 are single-column width (3.5 in) and
may need scaling in a one-column thesis template.

## 13. Table inventory

`final_table_inventory.md` lists 32 numbered tables, plus un-numbered inline tables. 29 of 32 are cited by number.

| Issue | Detail | Action |
|---|---|---|
| Not cited by number | Tables 6.2, 6.4 and 6.5 are introduced by the paragraph above them | Add "(Table 6.x)" at template stage |
| Out of numerical order | Table 5.3 sits after Tables 5.4–5.7, as the chapter explains | Move or renumber if the template requires numerical order |

## 14. Terminology audit

`terminology_audit.md`: there are no meaning-affecting inconsistencies. The 22 Part-N terms are checked, and British
spelling is used throughout the body. Four advisories:
1. the registered title's American "Optimization";
2. the two names of the Phase-7B agent;
3. the C1–C6 label reuse;
4. bare ρ for Spearman correlation in two Ch. 6 sentences (also in `final_notation_inventory.md`).

## 15. RQ completion

All six RQs are complete:

| Stage | Location |
|---|---|
| Introduced | Ch. 1 §1.5 |
| Answered | Ch. 5 §5.11 (RQ1–RQ5), Ch. 6 §6.15 (RQ6) |
| Interpreted | Ch. 7 §§7.2–7.10 |
| Concluded | Ch. 8 §8.2 |

Answers in brief:

| RQ | Answer |
|---|---|
| RQ1 | Yes against Uniform/Random; no against LAMP |
| RQ2 | Improves the search, not the final model |
| RQ3 | Retention problem confirmed; the archive restores validation-rank retention; mixed effect on test accuracy |
| RQ4 | The constrained reward curbs over-pruning; test feasibility is not guaranteed (67.5%) |
| RQ5 | Equal recovery narrows PPO–LAMP to within the ±0.5 pp decision rule; not equivalence |
| RQ6 | Not automatically; each deployment quantity must be measured |

## 16. Objective completion

O1–O6 are each mapped to completed and frozen evidence (Ch. 1 Table 1.1; `final_cross_chapter_qa.md` Q5). For O6,
the Phase-7B energy result is a documented measurement failure (CV 19.92% against a 3% limit). This is reported as
completed evidence of an unresolved result.

## 17. Contribution completion

C1–C6 are each mapped to chapter evidence and appendices. Each has a stated boundary in Ch. 8 §8.3:

| Contribution | Boundary |
|---|---|
| C1 | Not superior to LAMP |
| C2 | Search only |
| C3 | Mixed test effect |
| C4 | Test feasibility not guaranteed |
| C5 | Not formal equivalence |
| C6 | 7A accuracy cost; 7B energy unresolved |

## 18. Total main-text word count

46,689 words for Chapters 1–8, of which 40,467 are prose.

| Chapter | Words | Prose |
|---|---|---|
| 1 | 3,427 | 3,292 |
| 2 | 5,450 | 4,942 |
| 3 | 7,975 | 6,687 |
| 4 | 9,321 | 7,332 |
| 5 | 6,803 | 5,736 |
| 6 | 5,250 | 4,504 |
| 7 | 5,739 | 5,417 |
| 8 | 2,724 | 2,557 |

Abstract: 349 body words. Appendices: about 11,200 words.

## 19. Estimated page count

| Part | Estimated pages |
|---|---|
| Main text | about 130 |
| Front matter | 8–12 |
| References | 8 |
| Appendices | 22–30 |
| **Total** | **about 170–180** |

These figures assume about 420 words per page plus allowances for tables and figures (`final_length_report.md`). No
page target was forced. The main text is longer than the 80–100-page guide used while drafting.

## 20. Remaining blockers

1. **Official thesis template or formatting specification: not supplied.** This blocks:
   - the formatted DOCX/PDF;
   - page layout and heading, caption and citation styling;
   - the final page count;
   - confirmation of the Abstract word limit and spelling convention.

   `template/conference-template-a4.docx` is an IEEE conference template and was **not** used as a thesis template.
2. **Official front matter.** The title page, declaration, approval/certification page and acknowledgements are
   explicit placeholders in `thesis_full_source.md`. No author details, signatures, dates or approval statements were
   created.
3. **Author approval needed for presentation-level advisories** that touch approved chapters
   (`final_cross_chapter_qa.md` §B):
   - "VAL-RL" in two sentences;
   - "Spearman ρ" in two sentences;
   - in-text pointers for Tables 6.2, 6.4 and 6.5;
   - Table 5.3 placement;
   - optional appendix pointers;
   - harmonising the Phase-7B agent name.

None of these changes a number or a claim.

## 21. Files created

All files are in `results/thesis_final_writing/`.

**Front matter and support:**
- `abstract.md`, `qa_abstract.md`;
- `list_of_abbreviations.md`;
- `final_notation_inventory.md`;
- `appendix_plan.md`, `appendices.md`.

**References and audits:**
- `references_final.md`, `reference_audit_final.md`;
- `global_citation_audit.md`;
- `global_numerical_audit.md`;
- `global_claim_audit.md`;
- `split_governance_audit.md`;
- `terminology_audit.md`.

**Assembly and QA:**
- `thesis_full_source.md`;
- `final_figure_inventory.md`, `final_table_inventory.md`;
- `final_toc_map.md`;
- `final_length_report.md`;
- `final_cross_chapter_qa.md`;
- `final_completion_report.md`.

**Figures (`figures/final/`):** 13 neutral-name PNGs, `fig_3_1_overall_framework.png` to
`fig_6_4_phase7b_accuracy_planes.png`.

**Tools (`tools/`):**
- `refs_final.py`;
- `global_numeric_audit.py`;
- `global_claim_audit.py`;
- `split_audit.py`;
- `build_appendices.py`;
- `assemble.py`.

**Modified:**
- `chapter_04_experimental_setup.md` (three citations);
- `qa_chapter_04.md`, `qa_chapter_08.md` (logs);
- `tools/reference_verification.json`, `tools/reference_candidates.json` (three verified entries);
- `chapter_08_conclusion.md` (authorised refinements, earlier commit).

## 22. Branch and commit hash

Branch `thesis-final-writing`. Nothing has been pushed or merged.

| Commit | Content |
|---|---|
| `33850186` | Chapter 8 wording refinements (authorised) |
| `0c6bbb0f` | Abstract, keywords and front-matter support files |
| `b8facd6d` | Global reference rebuild and global audits |
| the commit adding this file (`git log -1 -- results/thesis_final_writing/final_completion_report.md`) | Full-source assembly, figures, inventories and final QA |
