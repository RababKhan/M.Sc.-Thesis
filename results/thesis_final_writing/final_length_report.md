# Final length report (after the MIST compression pass)

Counts use the same rule throughout (`tools/wordcount.py`): citation keys and code blocks excluded; "prose" excludes
table rows. "Before" is commit `f01d9636` (previous completion stage).

| Part | Before (words) | After (words) | After (prose) | Change |
|---|---|---|---|---|
| Chapter 1 | 3,427 | 3,099 | 3,041 | -328 (-9.6%) |
| Chapter 2 | 5,450 | 4,998 | 4,490 | -452 (-8.3%) |
| Chapter 3 | 7,975 | 6,695 | 5,624 | -1,280 (-16.1%) |
| Chapter 4 | 9,321 | 7,887 | 6,049 | -1,434 (-15.4%) |
| Chapter 5 | 6,803 | 6,300 | 5,332 | -503 (-7.4%) |
| Chapter 6 | 5,250 | 4,698 | 4,105 | -552 (-10.5%) |
| Chapter 7 | 5,739 | 4,499 | 4,188 | -1,240 (-21.6%) |
| Chapter 8 | 2,724 | 2,200 | 2,033 | -524 (-19.2%) |
| **Chapters 1–8** | **46,689** | **40,376** | **34,862** | **-6,313 (-13.5%)** |
| Appendices | 11,194 (A–K) | 3,453 (A–E) | 599 | -7,741 (-69.2%) |
| English Abstract (body, keywords excluded) | 349 | 390 | | |

Target for Chapters 1–8: 38,000–42,000 words. Achieved: 40,376.
The MIST word-equivalent and the page count of the formatted thesis are in `mist_length_compliance_audit.md`.
