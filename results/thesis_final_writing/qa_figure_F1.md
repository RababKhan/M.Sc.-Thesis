# QA report: Figure F1 (overall thesis framework)

- **Files:** `figures/F1_framework.py` (editable source), plus `F1_framework.svg` and `F1_framework.pdf` (vector and
  editable; fonts kept as text) and `F1_framework.png` (600 dpi, 4560 × 5940 px).
- **Status:** **FROZEN (v1, 2026-11-05).**

## Content checks

| Requirement (task §15) | Implementation | Check |
|---|---|---|
| Core flow from Dense CNN to accuracy–compression evaluation | 11 boxes in the required order, with a PPO-update loop from reward back to the policy | yes |
| Separate deployment extension, from structured pruning to trade-off analysis | 6 boxes in a separate, dashed panel on the right | yes |
| Phase 7B must not look like part of the original core | Extension panel titled "added after the core · RQ6". Each core box is tagged with the phase that introduced it. Colours separate *original design (Phases 2–3)*, *added after diagnostic results (Phases 4–6)* and *deployment extension* | yes |
| Clear arrows | Vertical flow arrows; a dashed hand-over arrow routed through the gap between the panels, labelled "7A: transferred from Phase 5 · 7B: learned directly as structured actions" | yes |

## Accuracy checks against frozen sources

- **Sensitivity box.** Labelled "Sensitivity-aware PPO **policy** — soft prior on action logits; centred S_l − S̄
  from Phase 4". The Phase-3 pre-registration sets the sensitivity *state* feature to 0 in all conditions; sensitivity
  enters through the soft prior only. The draft label "sensitivity-aware state" was therefore corrected.
- **Action set.** {0, .1, .2, .3, .4, .6}, from `result_freeze.md` §2.
- **Constraint and recovery.** τ = 0.98 (Phase 5); 5-epoch fixed-mask recovery (Phase 6).
- **Structured rule.** max(1, round((1 − r)·C)) (Phases 7A and 7B). The 1,296 structured shapes come from Paper 3.
- **Energy box.** Tagged "7A measured; 7B unresolved".
- **"Frozen outcomes" notes.** They state the 7A accuracy cost (VAL-RL) and the 7B accuracy cost (VAL-SELECT) with
  the right split for each, and Phase 7B energy as unresolved (CV 19.92%). The figure therefore cannot be read as a
  success claim.
- **Wording.** There is no "green" wording. "Sustainability" appears only in the panel title, as the user specified.

## Visual checks (rendered PNG inspected)

- No overlapping text.
- No arrows crossing boxes.
- Legend clear of content.
- Loop label inside the canvas.
- Colourblind-safe, distinct hues (blue, purple, orange), each also separated by panel position and border style.
