# Final thesis completion report (MIST stage)

**Official title:** *Development of a Sensitivity-Aware Reinforcement Learning Framework for Layer-Wise CNN Pruning*
**Author:** Rabab Khan Rongon. **Degree:** M.Sc. Engineering. **Department of Computer Science and Engineering,
Military Institute of Science and Technology, Dhaka, Bangladesh.** Branch `thesis-final-writing` (not pushed, not merged).

In this stage no experiment was run, no model retrained, no frozen result or pre-registered criterion changed, and the
test set was not read. The official MIST template is `template/Thesis Template MIST.pdf`
(SHA-256 `119ec7510d32112f48b47d1ada86460367a4911a2fea9cfcd00729627a685e4c`); the IEEE conference template was not used.
The previous stage's "template blocker" no longer applies and has been removed.

## 1. Template and formatting

Requirements extracted in `mist_template_audit.md` (cover TP.03, title TP.06, approval TP.08, declaration TP.09,
abstracts TP.11–12, acknowledgements TP.13, TOC/LoF/LoT/notation TP.14–17, headings TP.18, citations TP.19–21,
tables/figures/equations TP.22–25; length §2.2.3/§3.3.2). Implemented in `tools/build_docx.py` + `tools/word_finalize.ps1`:
A4; 12 pt Times New Roman, 1.5 spacing, 12 pt before/after, justified; margins L32/R25/T25/B25 mm (cover L35);
"CHAPTER n / TITLE" centred bold capitals; page numbers centred 19 mm from the bottom (none on cover–declaration; Roman
from the Abstract; Arabic from Chapter 1 through References; A-1, B-1 … in appendices, also in the TOC and List of
Tables); captions "Table n.m:" above / "Fig. n.m:" below; equations in Cambria Math 14 pt indented 19 mm with numbers
19 mm inside the right margin; lists labelled (a)/(i). Presentation-only deviations are listed in the template audit
(8–10 pt table text, list spacing, Kalpurush for Bangla, monogram placeholder, keywords kept).

## 2. Citations and references

Converted from 98 IEEE-numbered references to MIST author-date (Harvard): 189 citation occurrences converted, **0**
numbered citations and **0** unconverted markers left, **0** citations without an entry, **0** uncited entries
(`citation_conversion_audit.md`). Reference list: 98 entries, alphabetical, unnumbered, sentence-case titles, Vol./No./
pp./DOI (`references_mist_final.md`); per-reference audit with verification source, class and chapters cited in
`reference_audit_mist_final.md` (replaces the IEEE `reference_audit_final.md`). 50 of 98 (51.0%) from 2022–2026;
48 foundational. Holm (1979), Wilcoxon (1945) and Youden (1950) retained: each is the original source of a method used
in Chapter 4. Same-surname works distinguished by initials (C. Yu et al., 2022; S. Yu et al., 2022). Ch. 1 Gap 5 fixed:
Blalock et al. (2020) now cited only for inconsistent comparison practice, Le and Hua (2021) for the retraining-schedule
effect, with the open question stated as the thesis's inference.

## 3. Compression (Chapters 1–8)

| | Before | After |
|---|---|---|
| Chapters 1–8 (words incl. table cells) | 46,689 | **40,376** (−13.5%; target 38,000–42,000 met) |
| of which prose | 40,467 | 34,862 |
| Appendices | 11,194 (A–K) | 3,453 (A–E) |

Per chapter: 3,099; 4,998; 6,695; 7,887; 6,300; 4,698; 4,499; 2,200 (`final_length_report.md`). Redundancy removed
between Chs 3/4, 5/7, 6/7 and 7/8; no number, criterion, claim boundary, citation key, section, equation, figure or
numbered table was removed or changed. All audits re-run afterwards (§6).

## 4. MIST length compliance

`mist_length_compliance_audit.md`: estimated MIST word-equivalent **62,031** (prose 34,862 + front matter 2,135 +
references 3,035 + appendix prose 599 + 46 small and 15 large tables 17,450 + 2 listings 400 + 10 small and 3 large
figures 2,550 + 20 equations 1,000) against the 50,000 guideline; formatted length **198 A4 pages** (4 unnumbered,
19 Roman, 157 Arabic incl. References, 18 appendix) against 150. **The guideline is not met.** The guideline states it is
not a strict limit; reaching it would require removing evidence (e.g. moving tables or merging appendices), which was
not done without the author's decision (options in §11).

## 5. Abstracts, keywords, front matter

- English Abstract: rewritten into MIST prose form, **390** body words (300–600 allowed), objective/scope, methods,
  results, conclusion and novelty; no citations, tables, figures or non-standalone abbreviations (`qa_abstract.md`).
- Bangla Abstract: faithful translation (`abstract_bangla.md`), English terms in parentheses; Kalpurush 14 pt.
- Front matter in MIST order: cover, title page, approval, declaration (official TP.09 wording), English and Bangla
  abstracts, acknowledgements (placeholder), TOC, List of Figures, List of Tables, List of Main Notation and
  Abbreviations. No dedication (none supplied).

## 6. Audits (all re-run after compression and conversion)

| Audit | Result |
|---|---|
| Chapter 4 / 5 / 6 numerical audits | 100/100, 182/182, 142/142 PASS |
| Chapter 7 no-new-number check | PASS |
| Global numerical audit (16 claims, incl. Abstract) | PASS |
| Global claim audit (51 hits; §33 guards) | PASS: no affirmative forbidden claim |
| Split-governance audit | PASS, 6/6, no advisory |
| Global citation audit (189 occurrences; 26 re-read manually after compression) | PASS |
| Terminology audit | no meaning-affecting issue; all four previous advisories resolved |
| Notation inventory | all required symbols single-meaning; Spearman ρ labelled |

## 7. Authorised wording fixes applied

VAL-RL named for the 5.08 / 3.24 pp loss (Ch. 7 §7.10, Ch. 8 RQ6); "Spearman ρ" in Ch. 6; Phase-7B agent standardised
to "deployment-oriented structured PPO" with the energy-surrogate statement in the methodology; thesis contributions
written "Contribution 1–6" / TC1–TC6 (Phase-5 conditions C0–C2 unchanged); explicit pointers to the former Tables 6.2,
6.4, 6.5; all tables numbered consecutively by appearance (former Table 5.3 is now Table 5.11 where it stands);
first sections renamed "General" in Chs 3, 5, 6, 7, 8; Ch. 8 "item 4" → "item (iv)" to match MIST list labels; one raw
LaTeX token in Ch. 3 (`\overline{|w_l|}`) written as "mean |w_l|".

## 8. Figures and tables

13 figures (10 small, 3 large; data unchanged) and 61 numbered tables (43 chapter, 18 appendix; 6 on landscape pages:
2.2, 4.7, 6.5, A.1, C.1, E.2), all in the List of Figures / List of Tables (`final_figure_inventory.md`,
`final_table_inventory.md`).

## 9. Visual QA

The PDF was rendered page by page (Windows PDF renderer) and every page inspected over five build iterations. Defects
found and fixed: unreadable narrow table columns (width planner, landscape pages for 6 wide tables); tables splitting
after their first row (rows kept together); captions/lead-ins orphaned from their tables/lists; TOC/LoF/LoT entries
over-spaced and blue/underlined (Word built-in TOC styles redefined; hyperlink formatting removed); appendix page labels
in the lists (A-1 …); italic operator names in equations (argmax, softmax, top, round set upright); underscores shown
literally (math subscripts rendered; data identifiers left literal); a stray LaTeX token; cover spacing. Final check:
margins, fonts, spacing, numbering, headings, equations, figures, Bangla text, appendix numbering and reference
indentation all as specified; no blank pages; no table overflow or figure clipping.

## 10. Remaining intentional placeholders (information not supplied)

`[STUDENT NUMBER]` (title and approval pages), `[MONTH YEAR]` (cover, title page), `[DATE OF EXAMINATION]`,
`[NAME OF SUPERVISOR]`, `[NAME OF MEMBER]`, `[NAME OF EXTERNAL MEMBER]`, `[NAME OF HEAD OF THE DEPARTMENT]`,
`[Designation], [Department]`, `[Institution]` (approval page), `[MIST MONOGRAM 28 mm × 30 mm]` (cover, title page), the
structured acknowledgements placeholder (with `[SUPERVISOR]`), and blank signature lines (no signatures created).
No TODO, TBD, XXXX or `******` remains.

## 11. Unresolved / for the author

1. **Length** (§4): 62,031 word-equivalent / 198 pages vs 50,000 / 150. Options: (a) accept, citing the guideline's
   "no strict limit"; (b) move the 15 large tables' detail to the repository and keep summaries (≈ −5,000 to −8,000);
   (c) a second compression pass of Chs 3–6 to ≈ 30,000 prose words.
2. Degree statement wording on the title page ("Master of Science in Engineering (M.Sc. Engineering) in Computer
   Science and Engineering") should be confirmed against MIST CSE practice.
3. Board of Examination composition (four placeholder blocks) to be adjusted to the actual board.
4. Bangla font: Sutonny OMJ is not installed; replace Kalpurush at print stage if required.
5. The approved title contains the acronym "CNN" and starts with "Development of a", which MIST §3.3.3 discourages;
   kept unchanged as the approved title.
6. If fields are updated manually in Word, re-run `tools/word_finalize.ps1` to restore the appendix page labels in the
   lists.

## 12. Files

Created this stage: `mist_template_audit.md`, `abstract_bangla.md`, `references_mist_final.md`,
`reference_audit_mist_final.md`, `citation_conversion_audit.md`, `mist_length_compliance_audit.md`,
`final_thesis_MIST.docx`, `final_thesis_MIST.pdf`; tools `refs_mist.py`, `citation_audit_mist.py`,
`renumber_tables.py`, `assemble_mist.py`, `build_docx.py`, `figmap.py`, `mist_meta.json`, `wordcount.py`,
`mist_length_audit.py`, `word_finalize.ps1`, `word_measure.ps1`, `render_pages.ps1`, `_citation_manual_review.tsv`,
`_table_renumber_map.json`, `_docx_build_log.json`, `_table_heights.json`. Updated: Abstract, Chapters 1–8,
appendices (A–E) and their builder, appendix plan, abbreviations, notation inventory, terminology/claim/split/
numerical/citation audits, Abstract QA, inventories, TOC map, length report, full source, this report and the
cross-chapter QA. Removed (superseded IEEE outputs): `reference_audit_final.md`, `references_final.md`,
`tools/refs_final.py`, `tools/assemble.py`.
