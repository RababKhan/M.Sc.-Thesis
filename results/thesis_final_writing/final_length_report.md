# Final length report (after the final MIST length-reduction pass)

Counts use the same rule throughout (`tools/wordcount.py`): citation keys and code blocks excluded; "prose" excludes
table rows. "Before" is commit `9366c52a` (MIST-formatted build before the final length-reduction pass).

| Part | Before (words) | After (words) | After (prose) | Change |
|---|---|---|---|---|
| Chapter 1 | 3,099 | 2,746 | 2,688 | -353 (-11.4%) |
| Chapter 2 | 4,998 | 4,260 | 3,829 | -738 (-14.8%) |
| Chapter 3 | 6,695 | 5,171 | 4,574 | -1,524 (-22.8%) |
| Chapter 4 | 7,887 | 6,978 | 5,208 | -909 (-11.5%) |
| Chapter 5 | 6,300 | 5,366 | 4,694 | -934 (-14.8%) |
| Chapter 6 | 4,698 | 4,090 | 3,598 | -608 (-12.9%) |
| Chapter 7 | 4,499 | 3,666 | 3,498 | -833 (-18.5%) |
| Chapter 8 | 2,200 | 1,967 | 1,800 | -233 (-10.6%) |
| **Chapters 1–8** | **40,376** | **34,244** | **29,889** | **-6,132 (-15.2%)** |
| Appendices | 3,453 (A–E) | 1,457 (A–E, selective) | 712 | -1,996 (-57.8%) |
| English Abstract (body, keywords excluded; unchanged in this pass) | 390 | 390 | | |

Earlier target (first MIST pass) for Chapters 1–8: 38,000–42,000 words; this pass reduced them further to 34,244 for the 150-page / 50,000 word-equivalent guideline.
The MIST word-equivalent and the page count of the formatted thesis are in `mist_length_compliance_audit.md`.
