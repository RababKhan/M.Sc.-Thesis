# MIST length-compliance audit

Rule (MIST template, Part 2 §2.2.3 and §3.3.2): an M.Sc. Engineering thesis "should not exceed 50,000 words (150 A4
pages in 1.5 spacing)", inclusive of appendices, tables, figures and references; tables, figures and display
equations are counted by their word equivalents. The guideline states that there is no strict limit.

Measured on `final_thesis_MIST.docx` / `final_thesis_MIST.pdf` (Word layout). Prose words exclude table cells,
citation keys and code; every table and figure is counted instead by its MIST equivalent, classified by its rendered
height against half the text block (123.5 mm).

## Inputs

| Item | Count / words | MIST equivalent | Words |
|---|---|---|---|
| Chapter text (Chapters 1–8, prose) | 29,905 words | counted directly | 29,905 |
| Front matter (title page ≈51, English Abstract 409, Bangla Abstract 437, Acknowledgements placeholder 62, notation ≈610, abbreviations 590) | 2,159 words | counted directly | 2,159 |
| References (98 entries) | 3,035 words | counted directly | 3,035 |
| Appendix text (prose outside tables) | 493 words | counted directly | 493 |
| Numbered tables measured (chapters 32, appendices 8) | 40 | | |
| Small tables (≤ ½ page; chapters and appendices) | 32 | 200 each | 6,400 |
| Large tables (> ½ page) | 8 | 550 each | 4,400 |
| Algorithm listings (counted as tables) | 2 small, 0 large | 200 / 550 | 400 |
| Small figures (≤ ½ page) | 10 | 150 each | 1,500 |
| Large figures (> ½ page) | 3 | 350 each | 1,050 |
| Display equations | 20 | 50 each | 1,000 |
| **Estimated MIST word-equivalent** | | | **50,342** |

## Pages (from the PDF page labels)

| Part | Pages |
|---|---|
| Unnumbered (cover, title, approval, declaration) | 4 |
| Preliminary (Roman) | 18 |
| Chapters 1–8 and References (Arabic) | 136 |
| Appendices (A-1 …) | 6 |
| **Total A4 pages** | **164** |
| Not counted by MIST §3.3.2: cover, approval, declaration (3); table of contents (5); list of figures (1); list of tables (2) | −11 |
| **MIST-countable pages** (title page, abstracts, acknowledgements, notation, main text, references, appendices) | **153** |

## Assessment

- Chapters 1–8 total 34,315 words including table cells (29,905 prose).
- Estimated word-equivalent 50,342 against the 50,000 guideline: **342 above** the guideline.
- Formatted length: 164 physical A4 pages; **153 MIST-countable pages** against the 150-page guideline (3 above, 2.0%).
- The MIST body format itself (1.5 line spacing plus 12 pt before and after every paragraph, TP.18) is spacious;
  the page count reflects that format, not a change of content.

## Change in the final length-reduction pass (baseline commit `9366c52a`)

| Measure | Before | After |
|---|---|---|
| Chapters 1–8 words (incl. table cells) | 40,376 | 34,315 |
| Chapters 1–8 prose words | 34,862 | 29,905 |
| Appendix words | 3,453 | 1,116 |
| Numbered tables (chapter + appendix) | 61 (43 + 18) | 40 (32 + 8) |
| Small / large tables | 46 / 15 | 32 / 8 |
| Figures (small / large) | 13 (10 / 3) | 13 (10 / 3) |
| MIST word-equivalent | 62,031 | 50,342 |
| Physical A4 pages | 198 | 164 |
| MIST-countable pages | 186 | 153 |

## Stop condition

Compression stopped here (brief §26, condition B for the remainder). The word-equivalent is within 1% of the
50,000 guideline and the MIST-countable length is within 2% of 150 pages. The remaining large tables are the
methodology and evidence tables the brief requires to stay (literature comparison for the research gap, framework
development, data-split governance, baselines and matching rules, PPO configuration, experimental design,
evaluation metrics, pre-registered criteria), and every remaining results table supports an RQ answer. Removing
the last ~3 countable pages would require deleting one of these evidence or reproducibility tables.

## Large tables (more than half a page, or spanning pages)

| Table | Height (mm) | PDF pages |
|---|---|---|
| Table 2.1 | 255 | 41–42 |
| Table 3.1 | 160 | 52–52 |
| Table 4.2 | 145 | 73–73 |
| Table 4.5 | 148 | 78–78 |
| Table 4.6 | 139 | 79–79 |
| Table 4.7 | 77 | 80–80 |
| Table 4.9 | 247 | 87–88 |
| Table 4.10 | 187 | 90–90 |

## Small tables

Table 1.1 (45 mm), Table 3.2 (29 mm), Table 3.3 (110 mm), Table 3.4 (83 mm), Table 4.1 (47 mm), Table 4.3 (43 mm), Table 4.4 (46 mm), Table 4.8 (53 mm), Table 4.11 (76 mm), Table 5.1 (46 mm), Table 5.2 (65 mm), Table 5.3 (50 mm), Table 5.4 (87 mm), Table 5.5 (46 mm), Table 5.6 (30 mm), Table 5.7 (78 mm), Table 5.8 (71 mm), Table 6.1 (42 mm), Table 6.2 (68 mm), Table 6.3 (59 mm), Table 6.4 (48 mm), Table 6.5 (105 mm), Table 7.1 (90 mm), Table 8.1 (94 mm), Table A.1 (100 mm), Table B.1 (44 mm), Table B.2 (69 mm), Table C.1 (49 mm), Table D.1 (28 mm), Table D.2 (40 mm), Table E.1 (54 mm), Table E.2 (65 mm)

