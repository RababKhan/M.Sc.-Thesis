# QA report: Chapter 2 (Literature Review)

- **File:** `chapter_02_literature_review.md`.
- **Status:** **FROZEN (v1, 2026-11-05).**

## 1. Automated checks (`tools/qa_chapter.py`)

- **Length.** About 5,450 words of prose, plus Tables 2.1 and 2.2. That is roughly 13 pages of text, or about 15–16
  pages with the tables at thesis formatting.
- **References.** 167 citation occurrences, **91 distinct references**. All are verified (0 unknown keys).
- **Recent share.** 2022–2026: **50**; foundational: **41**; recent share **55%**, so the majority are recent, as
  required.
- **arXiv-only references.** One, `schulman2017ppo` (original PPO algorithm). It is allowed under the foundational
  exception for "PPO original paper if necessary"; no peer-reviewed version exists.
- **Tables.** 2.1 and 2.2, each referenced in the text before it appears. There are no figures.

## 2. Citation audit

- **Sources.** Every key resolves to `tools/reference_verification.json`, with source URLs listed in
  `reference_audit.md`. Each title was confirmed against Crossref, ML Anthology, PMLR, MLSys, JMLR or arXiv.
- **Content.** Every characterisation of a paper was checked against that paper's abstract
  (`tools/reference_abstracts.json`). Six statements were corrected to match their abstracts:
  - LeGR: no "evolutionary" claim;
  - random channel pruning: random search over configurations, no clear winning criterion;
  - ShrinkBench: "simple magnitude-based pruning" rather than "global magnitude";
  - NAS search-phase evaluation: "on average similar to a random policy";
  - inference energy trends;
  - Ganjdanesh et al.: accuracy reward, no hardware metric.
- **Comparison table.** Table 2.2 lists datasets and models only as stated in each abstract.
- **Format.** IEEE-numbered style. Keys are converted to numbers at assembly, in order of first citation.

## 3. Numerical audit

The chapter contains few numbers. Each one is traced:

| Statement | Source | Check |
|---|---|---|
| "more than three thousand pruning papers … 2020 to 2024" | Cheng et al. abstract | matches |
| quantisation better than pruning except at very high compression | Kuzmin et al. abstract | matches |
| "more than fifty design choices" | Andrychowicz et al. abstract (">50") | matches |
| energy for 1,000 inferences (Luccioni et al.) | abstract | matches |
| four units × six ratios = 1,296 policies | `result_freeze.md` §2 | matches |
| all 1,296 structured configurations of two architectures | Paper 3 (`result_freeze.md` §7) | matches |
| thesis row of Table 2.2 (RAPL measured in 7A, unresolved in 7B; LAMP stronger before recovery) | `result_freeze.md` §§3–10 | matches |
| baselines built at each PPO run's own sparsity | Phase 3 design (`result_freeze.md` §3) | matches (correction A) |
| recovery identical across methods, shared seed and data stream | `phase6_preregistration.md` | matches |

## 4. Claim-boundary audit

Every dangerous-phrase hit was inspected:

- **"Green AI"** appears only as the name of an external agenda, cited to Schwartz et al.
- **"carbon"** appears only in descriptions of the literature and in the explicit statement that the thesis makes no
  carbon claims.
- **"state-of-the-art"** appears only inside a cited paper's finding ("state-of-the-art NAS algorithms performed
  similarly to a random policy").
- **"guarantees"** describes the submodular pruning paper.
- **"Energy-Efficient"** appears only in the required section heading 2.13, which describes literature.
- **"heuristic"** was removed. The LAMP sentence now uses its abstract's own words, and the deterministic baselines
  are not all called heuristics (correction B).

The research gap (Section 2.14) is stated as six specific open questions, not as "no one has combined PPO and
pruning". No claim about the thesis' own results appears in this chapter beyond the Table 2.2 row, which matches the
claim matrix.

## 5. Figure and table audit

- **Table 2.1:** granularity taxonomy. It is conceptual and contains no data.
- **Table 2.2:** comparison of 16 verified works plus the thesis row. Every cell was checked against its abstract or
  the frozen thesis results.

## 6. Freeze

Chapter 2 v1 is frozen. Later changes are allowed only for (a) citation-number assembly or (b) a factual error found
during later QA. Any such change will be logged here.
