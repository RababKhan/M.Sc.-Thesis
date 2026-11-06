# MIST thesis template audit

## Template used

| Item | Value |
|---|---|
| Path | `template/Thesis Template MIST.pdf` |
| File name | `Thesis Template MIST.pdf` |
| Size | 4,975,358 bytes |
| SHA-256 | `119ec7510d32112f48b47d1ada86460367a4911a2fea9cfcd00729627a685e4c` |
| Content | MIST templates TP.01–TP.25 (Annex pages Ax-40 to Ax-70) and Parts 2–3 of the MIST thesis-writing guidelines (pp. 4–19) |
| Not used | `template/conference-template-a4.docx` (IEEE conference-paper template; not a thesis template) |

The files in `template/` were read only and were not modified. The text was extracted with `pdftotext -layout` into
the session scratchpad. Some annex pages print two overlapping text layers. Where the extraction garbled a rule, the
rule was read from the legible duplicate layer or from the surrounding examples.

## Requirements for an M.Sc. Engineering thesis (postgraduate)

### A. Degree-level limits (Part 2, §2.2.3; §3.3.2)

1. **Length.** One volume, "should not exceed 50,000 words (150 A4 pages in 1.5 spacing)".
   - The limit is "inclusive of Appendices, Footnotes, Tables, Figures and References" (§2.2).
   - The counted items are the title page, dedication, acknowledgements, list of notation, abstract, main text,
     references, tables and figures (§3.3.2).
   - The guideline states "there are no strict restrictions"; the limit is a guideline against "unnecessary and
     cumbersome reading".
2. **Word equivalents (§3.3.2).**

   | Item | Word equivalent |
   |---|---|
   | Each full 1.5-spaced page of text | 330 words |
   | Large table (more than half an MS Word page) | 550 |
   | Small table (less than half a page) | 200 |
   | Large figure (more than half a page) | 350 |
   | Small figure (less than half a page) | 150 |
   | Display equations, when more than half a page in total | 50 per equation |

3. Only the student's own primary data may be used. This thesis reports only its own experiments.
4. **Language (§2.3 D).** English with a single spelling style (British or American). A Bangla translation of the
   Abstract must follow the English Abstract.

### B. Structure and order (§3.3; table in §3.3.1)

| Part | Item | Page numbering | M.Sc. Engg. |
|---|---|---|---|
| A | Cover page (and spine) | none | required |
| B | Title page | none | required |
| C | Certificate of approval | none | required |
| D | Declaration of originality | none | required |
| E | Dedication | none | optional (single author only) |
| F | Abstract (English, then Bangla) | Roman, starting i | required |
| G | Acknowledgements | Roman | required |
| H | Table of contents | Roman | required |
| I | List of figures | Roman | required for M.Sc. (§3.3.1 table) |
| J | List of tables | Roman | required for M.Sc. |
| K | List of notation (abbreviations and symbols) | Roman | required for M.Sc. |
| L | Main text (chapters) | Arabic, starting 1 | required |
| M | References | Arabic, continuing | required |
| N | List of publications | — | optional |
| O | Appendices | A-1, A-2, …; B-1, … | optional |

The table in §3.3.1 lists the preliminary items in a different order (table of contents before the abstract). The
order applied here follows the lettered sequence A–O of §3.3, which the user also specified: Abstract, Bangla Abstract,
Acknowledgements, Table of Contents, Lists.

### C. Page layout

| Element | Rule (template) |
|---|---|
| Paper | A4, one-sided printing |
| Body margins | left 32 mm, right 25 mm, top 25 mm, bottom 25 mm (TP.18, TP.20, TP.22, TP.25) |
| Body text | 12 pt Times New Roman, 1.5 line spacing, 12 pt before and after each paragraph, justified (TP.18) |
| Page numbers | centred, 19 mm (0.75 in) from the bottom; Roman in preliminary pages, Arabic in the main text and references, A-1, … in appendices |
| Chapter heading (TP.18) | new page; 25 mm from top; "CHAPTER 1", then the title on the next line; centred, bold, all capitals, 12 pt Times New Roman, single spacing, followed by 24 pt |
| Section headings (TP.18) | 12 pt Times New Roman bold, left aligned, 24 pt before, 18 pt after; numbered 1.1, 1.1.1, 1.1.1.1 (maximum four levels) |
| First section of each chapter | a short introductory section designated "General" (§3.3.11) |
| Lists | parenthetical Roman numerals (i), (ii) or letters (a), (b); bullets and Arabic numerals "should preferably be avoided" (§3.3.11) |
| Tables (TP.22, TP.24) | centred; caption above, "Table 4.2: …", single spacing, left aligned, 12 pt before and 9 pt after; cells single-spaced with 3 pt before and after; notes 10 pt |
| Figures (TP.22, TP.23) | centred; caption below, "Fig. 4.2: …", single spacing, 9 pt before and 12 pt after (6 pt between a large figure and its caption, caption centred) |
| Equations (TP.25) | 14 pt Cambria Math, indented 19 mm from the left margin, number in parentheses near the right margin, single spacing with 12 pt before and after |

### D. Preliminary pages

| Page | Rule |
|---|---|
| Cover (TP.03, PG) | Title (≤ 100 characters, capitals) in 18 pt Arial bold; other lines in 16 pt Arial bold; all single and centred; author name only (no student number); "M.Sc. ENGINEERING THESIS"; MIST monogram 28 × 30 mm; department; "MILITARY INSTITUTE OF SCIENCE AND TECHNOLOGY"; "DHAKA, BANGLADESH"; month and year. Left margin 35 mm, right 25 mm. Hard-bound cover colour for M.Sc. Engg.: dark blue (binding stage) |
| Title page (TP.06) | Title in 14 pt Times New Roman capitals; others in 12 pt; author with student number "(SN. …)"; "A Thesis Submitted in Partial Fulfillment of the Requirements for the Degree of …"; monogram; department; MIST; Dhaka; month and year |
| Approval (TP.08) | Title (14 pt); "M.Sc. Engineering Thesis"; "By"; author (SN.); "Approved as to style and content by the Board of Examination on [date]:"; signature blocks for Chairman (Supervisor), Member (External), Member, Member (Ex-officio) / Head of the Department, each with name, designation and institution; foot line "Department of …, MIST, Dhaka" |
| Declaration (TP.09) | Title (14 pt, single); heading "DECLARATION"; the template's declaration text in 1.5 spacing, justified; signature space; PG: single name without student number; department line |
| Dedication (TP.10) | optional; only if the author supplies one |
| Abstract (TP.11, TP.12) | heading "ABSTRACT", then the thesis title with each word capitalised (single spacing), then the text in 1.5 spacing, justified; 12 pt Times New Roman; Bangla version in 14 pt Sutonny OMJ (suggested) |
| Abstract content (§3.3.5) | about 300–600 words (Master's: at most 1.5 pages); objectives and scope, methods, brief results, key conclusions, novelty; no discussion, citations, tables, figures, or non-standalone abbreviations |
| Acknowledgements (TP.13) | 1.5 spacing, 12 pt paragraph spacing; PG: single author (singular) |
| Table of contents (TP.14) | single spacing, 3 pt paragraph spacing; level indents 32 / 38 / 50 / 63 mm from the page edge |
| Lists of figures, tables and notation (TP.15–TP.17) | single spacing, 6 pt paragraph spacing; "Figure 2.1:" or "Table 2.1:" followed by the description and page number; notation at 32 mm, description at 75 mm |

### E. Citations and references (TP.19–TP.21)

| Rule | Form |
|---|---|
| System | customised author-date (Harvard). Every reference has a text citation and every citation a reference |
| One author | (Peck, 2002) or Peck (2002) |
| Two or three authors | (Peck, Henson and Simons, 2004) |
| More than three | (Jones et al., 2012) |
| Several works | (Hussain, 1988; Hawser et al., 1998; Noman, 2000) |
| Same author, same year | 2003a, 2003b |
| Same surname, different authors | initials: A. Kabir (1997) and J. Kabir (1997) |
| Corporate authors | full name at first citation with the abbreviation |
| Reference list | heading "REFERENCES"; alphabetical by author, then year; not numbered; "Surname, I. I., Surname, I., and Surname, I. (Year). Title in sentence case, Source in capitalised-word case, Vol., No., pp., DOI"; seven or more authors: first six, then "et al."; single spacing, 10 pt before and after, 13 mm hanging indent, justified |

### F. Thesis title

The approved title is "Development of a Sensitivity-Aware Reinforcement Learning Framework for Layer-Wise CNN
Pruning", which has 94 characters (limit 100).

§3.3.3 (iv) discourages acronyms in titles, and §3.3.3 (v) discourages waste words such as "A" or "An" at the start.
The title contains the acronym "CNN" and starts with "Development of a". Under §3.3.3 (vii), the title must be
approved through MIST's process. Since it is the approved title, it is used exactly as given and is not changed.

## Compliance decisions recorded for this thesis

| Decision | Reason |
|---|---|
| Preliminary order: Abstract (English, Bangla), Acknowledgements, TOC, LoF, LoT, Notation | lettered order of §3.3 and the user's instruction |
| Bullet lists rendered as (a), (b) …; nested lists as (i), (ii) … | §3.3.11 prefers parenthetical letters or Roman numerals to bullets |
| Chapter first sections renamed "General" where the first section was a chapter overview (Chs 3, 5, 6, 7, 8) | §3.3.11 |
| Bangla font: Kalpurush (Unicode) at 14 pt | Sutonny OMJ is not installed on this machine; Kalpurush renders Unicode Bangla correctly. Replace the font at print stage if the department requires Sutonny OMJ |
| MIST monogram | not available in the repository; a visible placeholder box of 28 × 30 mm is left on the cover and title page |
| Student number, supervisor, examiners, examination date, submission month and year | not supplied; visible placeholders, never invented |

## Deviations from the literal template (presentation only)

| Item | Template | Applied | Reason |
|---|---|---|---|
| Table cell font | 12 pt Times New Roman (notes 10 pt) | 10 pt (≤ 5 columns), 9 pt (6–7 columns), 8 pt (≥ 8 columns); notes 10 pt | data tables of 6–10 columns cannot fit 12 pt within the 153 mm text width without breaking numbers across lines |
| List items | (not specified; body 12 pt before and after) | list paragraphs 1.5-spaced with 2 pt before and after, labelled (a)/(i) with a hanging indent | 12 pt around every list item would separate the items of one list more than paragraphs |
| Bangla font | Sutonny OMJ 14 pt | Kalpurush (Unicode) 14 pt in the working digital build | Sutonny OMJ is not installed. Sutonny OMJ substitution should be applied only if required by the department or final print-office instructions; the Bangla Abstract content is not changed by a font substitution |
| MIST monogram | 28 × 30 mm logo | bordered placeholder box of the same size | the logo file is not in the repository |
| Keywords | not part of TP.11 | kept after the English Abstract | harmless; remove if the department prefers |
| Chapter "General" first section | §3.3.11 | applied to Chapters 3, 5, 6, 7 and 8 (former "Chapter Overview"); Chapters 1, 2 and 4 open with substantive first sections (Background; Neural Network Compression; Experimental Design Overview) | renaming those would misdescribe their content |
