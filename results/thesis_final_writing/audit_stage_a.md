# Stage A: repository and integration audit (before any thesis prose)

Date: 2026-11-05. Branch `thesis-final-writing`, created from `thesis-final-integration` at `7669fa80`. No experiment
was run and no frozen result or integration file was modified.

## 1. Inputs read

- All 13 documents in `results/thesis_final_integration/`:
  - `result_freeze.md`, `final_research_questions.md`, `final_contributions.md`;
  - `chapter_structure.md`, `results_mapping.md`;
  - `table_shortlist.md`, `figure_shortlist.md`, `claim_matrix.md`;
  - `sustainability_positioning.md`, `limitations_future_work.md`, `novelty_statement.md`;
  - `conceptual_conclusion.md`, `final_thesis_integration_summary.md`.
- The frozen source files named in `result_freeze.md`, for Phases 2–6, Paper 3, 7A, 7B and 7B-EC:
  - CSV statistics and summaries;
  - classification and decision JSON files;
  - summary Markdown files;
  - pre-registrations, where a method detail had to be confirmed.

## 2. Numerical audit

`tools/numeric_audit.py` re-extracts **174** headline numbers directly from the frozen files and compares them with
`result_freeze.md`. The full table, which is also the claim → phase → file → value source map, is in
`numerical_audit.md`.

Result: **167 match, 7 corrected, 0 unresolved.** The corrections are cases where `result_freeze.md` differs from
the frozen file. The frozen file governs, and the thesis uses its value:

| Item | result_freeze.md | Frozen file | Thesis uses |
|---|---|---|---|
| Phase 4 terminal-policy (S0) top-10 reach | 0% in all 6 settings | 0% in 5/6; 5% in C100 LeNet-5 (`phase4_search_quality.csv`) | "0% in five of six settings and 5% in the sixth" |
| Phase 4 archive (S2A) top-10 reach | 75–100% | 80–100% (75–100% is the archive-only S1 row) | 80–100% for S2A |
| Phase 4 archive Δ test, C10 LeNet-5 | +0.90 pp | +0.8945 pp | +0.89 pp |
| Phase 5 ResNet-8 sparsity C0 / C1 (C10) | 48.8% / 24.7% | mean 46.2% / 24.5%; median 48.8% / 24.7% | mean, with the median noted (drops are means) |
| Phase 5 ResNet-8 sparsity C0 / C1 (C100) | 23.5% / 16.1% | mean 25.9% / 15.6%; median 23.5% / 16.1% | as above |

The Phase-5 summary table itself is consistent with the data: it reports *median* sparsity beside *mean* drops. The
integration package simply did not say "median".

Two method facts were confirmed in the pre-registrations:

- **Phase 3 state and prior.** The 7-feature state has its sensitivity feature set to 0 in every condition.
  Sensitivity enters only as a soft prior on the action logits: `logit − β·prior_u·ratio/60`, with β = 0.5
  (`phase3_preregistration.md`).
- **Phase 6 pairing.** Every PPO/LAMP/global triple shares the fine-tuning seed and the data stream
  (`phase6_preregistration.md`).

## 3. Reference pipeline

- `tools/reference_candidates.json` lists the candidate references. `tools/verify_refs.py` verifies each one against
  an authoritative index:
  - Crossref, for IEEE, ACM, Springer, Elsevier, Nature and ACL DOIs;
  - ML Anthology, which mirrors the official ICLR, NeurIPS, ICML and CVF metadata;
  - the official MLSys and JMLR pages;
  - arXiv, for the PPO paper only.

  A candidate is accepted only if the source returns the same normalised title.
- **Result: 97 of 97 candidates verified.** No metadata comes from memory.
- **Discovery.** Recent work was found by web search. Every recent entry then went through the same verification.
- `tools/bib.py` produces the IEEE entries from the verified metadata. ICML volumes and pages are taken from the
  official PMLR indexes, and official title capitalisation is kept.

## 4. Template check (blocker for the final document)

The repository contains no official university or MIST thesis template. The only template is
`template/conference-template-a4.docx`, an IEEE *conference* template owned by the user. It is untracked and must not
be modified. Per task section 42, no thesis template is invented. Chapters are kept as structured Markdown sources
until the official template is supplied. **The final PDF cannot be produced without it.**

## 5. Writing order (task section 7)

1. Stage 1: Chapter 2.
2. Stage 2: F1.
3. Stages 3–9: Chapters 3, 4, 5, 6, 7, 1 and 8, each drafted, audited and frozen before the next.
4. Stage 10: front matter, appendices and final QA.
