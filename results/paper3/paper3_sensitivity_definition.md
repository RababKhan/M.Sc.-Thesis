# Paper 3 — structured sensitivity prior (PI decision 2026-10-25)

Committed **before any final sensitivity vector is computed**. This closes the open item of amendment 1
(§7) and amendment 2 (§7).

## Definition: immediate post-pruning structured sensitivity, before any recovery

For unit l of CIFAR-10 ResNet-8 (stem group, stage1, stage2, stage3):

S_l,r = (A_dense − A_l,r) / A_dense

- A_dense is the VAL-RL accuracy (2,500 images) of the frozen dense reference.
- A_l,r is the VAL-RL accuracy immediately after physically compacting **unit l alone** by fraction r
  (`structured_units.slim` with the single-unit action vector).
- There is **no recovery or fine-tuning** of the probe. The dense model is never modified: each probe
  is a fresh compacted copy.

**Probe fraction.** The PI's formula names a single probe A_l,probe without fixing the fraction. For
continuity with the sensitivity of Phases 3–5, which probed {10, 20, 40, 60}%, the probe set is that
same set: **S_l = mean over r ∈ {0.1, 0.2, 0.4, 0.6} of S_l,r**. All four values are themselves
action ratios, so each probe is a policy of the action space. Each per-ratio S_l,r is reported too.

## Prior

The same mapping as the Phase-5 C1 pipeline (centred prior P2, β = 1.0):
- S_norm = min-max normalisation of S over the four units;
- the coefficient of unit l is c_l = β · (S_norm,l − mean(S_norm));
- logits are shifted by −c_l · ratio(a) / 0.6 (`src.rl.SoftPriorPolicy`, unchanged).

## Use and protection

- **Same prior everywhere.** C1-S, M1 and H1 all use this prior, so the reward objective stays the only
  manipulated variable.
- **Data.** VAL-RL only, with no test access. The value is computed once, frozen (sha256), and never
  recomputed after any PPO result.
- **Separation of signals.** The sensitivity measures intrinsic structural fragility; the reward
  measures recoverable full-policy quality through the 1-epoch proxy. The two stay separate.
