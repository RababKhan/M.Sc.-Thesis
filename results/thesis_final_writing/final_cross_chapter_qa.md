# Final cross-chapter QA (MIST stage)

Scope: the final, compressed Abstract (English and Bangla), Chapters 1–8, references and Appendices A–E as assembled in
`thesis_full_source.md` and formatted in `final_thesis_MIST.docx` / `final_thesis_MIST.pdf`. All audits were re-run
after compression, table renumbering and author-date conversion.

| # | Question | Answer | Evidence |
|---|---|---|---|
| 1 | Are all RQs introduced in Chapter 1? | **Yes.** RQ1–RQ6 verbatim in §1.5, mapped to O1–O6 (§1.6) and Contributions 1–6 (§1.7, Table 1.1 TC1–TC6) | Ch. 1 |
| 2 | Are all RQs answered in Chapters 5–6? | **Yes.** RQ1–RQ5 in §5.11, RQ6 in §6.15 | Ch. 5–6 |
| 3 | Are all RQs interpreted in Chapter 7? | **Yes.** §§7.2–7.10, one or more sections per RQ; compressed to interpretation with pointers to Chs 5–6 | Ch. 7 |
| 4 | Are all RQs concluded in Chapter 8? | **Yes.** §8.2 keeps one subsection per RQ under its full wording, each with a one-line verdict | Ch. 8 |
| 5 | Does every objective map to completed evidence? | **Yes.** O1–O6 → Phases 3–7B; O6's Phase-7B energy part is a documented measurement failure (unresolved result, not missing evidence) | Table 1.1; Ch. 4 Table 4.7 |
| 6 | Does every contribution map to evidence? | **Yes.** TC1–TC6 each with chapter evidence and its boundary in Table 8.1 | Ch. 8 §8.3 |
| 7 | Are Chapter 1 claims consistent with Chapters 5–8? | **Yes.** Findings preview values recomputed (global numerical audit, claims 1, 6, 7, 14) | `global_numerical_audit.md` |
| 8 | Are Chapter 8 conclusions consistent with Chapter 7? | **Yes.** No conclusion strengthened; limitations A–G and all 12 future-work items retained | Ch. 7 §7.13; Ch. 8 §§8.5–8.6 |
| 9 | Are all repeated numbers consistent? | **Yes.** 16 global claims PASS; Ch. 4 100/100, Ch. 5 182/182, Ch. 6 142/142, Ch. 7 no new numbers | `global_numerical_audit.md` |
| 10 | Are all split names correct? | **Yes, 6/6 checks; no advisory remains.** The 5.08 / 3.24 pp Phase-7A loss now reads "VAL-RL accuracy" in Ch. 7 §7.10 and Ch. 8 RQ6 | `split_governance_audit.md` |
| 11 | Are all major limitations present? | **Yes.** Ch. 7 §7.13 and Ch. 8 §8.5 (scale, models, optimisation, evaluation, hardware, energy, sustainability) | Ch. 7–8 |
| 12 | Are future-work items derived from limitations? | **Yes.** Ch. 8 §8.6 items (i)–(xii), each tied to a finding or limitation; the internal cross-reference now reads "item (iv)" to match the MIST list labels | Ch. 8 |
| 13 | Is LAMP consistently positioned? | **Yes.** Stronger before equal recovery; slightly ahead in three settings after; never "beaten" or "equivalent" | `global_claim_audit.md` |
| 14 | Is sensitivity consistently positioned? | **Yes.** Improves search/exploration measures, not final accuracy | Ch. 5, 7, 8; Abstract |
| 15 | Is archive selection consistently positioned? | **Yes.** Restores validation-rank retention; mixed test effect | Ch. 5 §5.6; Ch. 7 §7.4; Ch. 8 RQ3 |
| 16 | Is the ±0.5 pp rule always a decision rule? | **Yes.** Every occurrence bounded as "not a formal equivalence test" | `global_claim_audit.md` |
| 17 | Is Phase-7A energy always paired with accuracy cost? | **Yes** (Table 6.5, §§6.8, 6.14–6.16, §7.10, RQ6, Abstract) | Ch. 6–8 |
| 18 | Is Phase-7B energy always unresolved? | **Yes.** Search used an energy surrogate (stated in §§3.16.3, 4.9.7, 6.10, 7.10); measured energy unresolved | Ch. 3–8 |
| 19 | Is Phase-7B latency always unmeasured? | **Yes** (§6.11, Ch. 7 §7.10, Ch. 8 RQ6) | Ch. 6–8 |
| 20 | Any unsupported carbon/environment claims? | **No.** | `global_claim_audit.md`; `terminology_audit.md` §5 |
| 21 | Placeholders or TODOs left? | **No TODO/TBD/XXXX.** Only the intentional, visible front-matter placeholders listed in `final_completion_report.md` §21 | PDF text scan |
| 22 | Are references globally consistent? | **Yes.** MIST author-date; 98 entries alphabetical, unnumbered; 0 numbered citations; every citation has an entry and every entry is cited | `citation_conversion_audit.md`; `reference_audit_mist_final.md` |
| 23 | Are all figures/tables referenced and listed? | **Yes.** 13 figures and 61 tables (43 chapter, 18 appendix), numbered consecutively, all in the List of Figures / List of Tables with page numbers; explicit pointers added to the former Tables 6.2, 6.4, 6.5 (now 6.4, 6.6, 6.8) | `final_figure_inventory.md`; `final_table_inventory.md` |
| 24 | Is notation consistent? | **Yes.** All required symbols single-meaning; "Spearman ρ" written wherever ρ means correlation | `final_notation_inventory.md` |
| 25 | Is the thesis ready for submission formatting review? | **Formatted to the MIST template and visually checked.** Outstanding items are author-supplied information (student number, examiners, date, acknowledgements, monogram) and the length decision (`mist_length_compliance_audit.md`) | `final_completion_report.md` |
