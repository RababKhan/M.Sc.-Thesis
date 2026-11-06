# Final notation inventory

This inventory was compiled from Table 3.2 (Chapter 3) and checked against every use of each symbol in Chapters 1–8,
the Abstract and the appendices. Searches used both the LaTeX and the Unicode form of each symbol. It is read-only
with respect to the chapters: no chapter text was changed.

Status codes:
- **OK**: one meaning throughout.
- **OK (context)**: the same glyph carries a second meaning, but the second meaning is always marked by a subscript,
  hat, argument or nearby label.
- **ADVISORY**: the reuse is unmarked in at least one sentence (none remains after the final MIST stage).

## Symbols required by the completion brief

| Symbol | Meaning | First chapter | Ambiguity | Status |
|---|---|---|---|---|
| N_L | number of pruning units; N_L = 4 for every architecture | Ch. 3 (Table 3.2, Eq. 3.1) | Ch. 3 states that L alone is never used, because T is latency | OK |
| r_l | pruning ratio assigned to unit l, r_l = r(a_l) ∈ 𝓡 | Ch. 3 (Eq. 3.1) | none | OK |
| **r** (bold) / r(a) | **r**: the policy vector [r_1, …, r_{N_L}]. r(a): the map from action index to ratio | Ch. 3 (Table 3.2) | bold vector vs function of a; Table 3.2 notes that **r** is equivalent to the action vector **a** | OK (context) |
| S_l | normalised sensitivity of unit l, S_l ∈ [0, 1] | Ch. 3 (Table 3.2; §3.9) | none; Chs 4 and 5 use it with the same meaning | OK |
| S̃_l (centred sensitivity) | S_l − S̄, where S̄ = (1/N_L) Σ S_j (Eq. 3.11); the centred prior uses c_l = 1.0·S̃_l | Ch. 3 (Eq. 3.11) | none. "Centred" is spelled the British way throughout (0 uses of "centered") | OK |
| s_l | state observed by the agent before deciding unit l | Ch. 3 (Table 3.2; §3.5) | ŝ(**r**) in Eq. 3.19 is the *predicted energy saving* of Phase 7B, and s_max its maximum. These are distinguished by the hat, the policy argument and the subscript | OK (context) |
| a_l | action at unit l, a_l ∈ {0, …, 5} | Ch. 3 (Table 3.2) | a = −0.117 is the intercept of the energy surrogate (Eq. 3.18), defined in the same line as a coefficient with units | OK (context) |
| π_θ | PPO policy with parameters θ, π_θ(a \| s) = softmax of the shifted logits (Eq. 3.10) | Ch. 3 (Eq. 3.3) | none | OK |
| A_b | dense-reference accuracy on the reward split (VAL-RL) | Ch. 3 (Table 3.2) | For q_VAL-SELECT (Eq. 3.14) the dense reference is the VAL-SELECT accuracy. This is implicit in "retained accuracy on the selection split", not given a separate symbol | OK (context) |
| A_p | accuracy of the pruned network M_**r** on the same split as A_b | Ch. 3 (Table 3.2) | A_2, A_2,dense (7B proxy), A_1/A_5 (Paper 3 proxy) and A_60 (R2 anchor) are separately subscripted accuracies | OK (context) |
| q | retained accuracy A_p / A_b | Ch. 3 (Table 3.2) | q_VAL-SELECT (selection split, Eq. 3.14) and q̂ = A_2/A_2,dense (Phase-7B 2-epoch proxy) are marked by subscript or hat | OK (context) |
| ρ | masked weight sparsity, % of zero weights over all Conv2d/Linear weights (Eq. 3.2) | Ch. 3 (Eq. 3.2) | Spearman's rank correlation is also written with ρ in Chapter 6 and Appendix D. Every such use is now written "Spearman ρ" (the two formerly bare uses in Chapter 6 were corrected in the final MIST stage); sparsity ρ is never used in Chapter 6 | OK (context) — advisory resolved |
| R | episode reward. Variants: R_0 (Eq. 3.7), R1a/R1b/R2 (Phase-4 Stage A), R_c (Eq. 3.15), R_7B (Eq. 3.19); R̂_t is the PPO return target | Ch. 3 (Table 3.2) | In Algorithm 3.1 (plain-text pseudocode) the ratio set 𝓡 is typed as "R = {0, .1, …}", because the code block cannot show the calligraphic letter. The ratio values identify it | OK (context). Optional: render 𝓡 at template stage |
| τ | retained-accuracy threshold of the constrained reward, τ = 0.98 (Phase 5 and the selection rule of Eq. 3.14) | Ch. 3 (Table 3.2; Eq. 3.15) | τ̂ = 0.98849 is the Phase-7B proxy threshold, marked by the hat (Ch. 3 §3.16.3; Ch. 4) | OK (context) |
| latency, T(**r**) | measured batch-1 CPU latency per inference (ms); T_table(**r**) is the frozen Paper-3 table value used by the 7B surrogate | Ch. 3 (Table 3.2) | none. L is avoided on purpose | OK |
| energy, E(**r**) | directly measured CPU package energy per 1,000 inferences (J) | Ch. 3 (Table 3.2) | Ê(**r**) is the surrogate (predicted) energy, used only inside the Phase-7B search. Every chapter says that it is not a measurement | OK |

## Other symbols found

| Symbol | Meaning | First chapter | Ambiguity | Status |
|---|---|---|---|---|
| 𝓡 | ratio set {0, 0.1, 0.2, 0.3, 0.4, 0.6} | Ch. 3 | see R above | OK |
| M, M_**r** | dense reference network; network pruned by policy **r** | Ch. 3 | none | OK |
| S̄ | mean sensitivity over units | Ch. 3 | none | OK |
| c_l | prior coefficient (P1: 0.5·S_l; centred: 1.0·S̃_l) | Ch. 3 | C_l (capital) is the channel count of unit l. Case distinguishes them, and both are defined in Table 3.2 | OK (context) |
| C_l, k_l | channels before structured pruning; channels kept | Ch. 3 | C with a digit labels only the Phase-5 conditions C0–C2; thesis contributions are now TC1–TC6 / "Contribution 1–6" | OK (context) |
| D | number of distinct actions in **a** | Ch. 3 | none | OK |
| 𝒜 | archive of unique visited policies | Ch. 3 | none | OK |
| 𝒞 | top-K archived candidates (Eq. 3.14), K_c = 10 | Ch. 3 | none | OK |
| ε | accuracy tie-breaker weight in R_c and R_7B (0.01) | Ch. 3 (Eq. 3.15) | **ϵ** (lunate epsilon, 0.2) is the PPO clipping range (Eq. 3.20, Table 3.7). The two glyphs are different and both values are stated where each is used | OK (context). The template font must keep ε and ϵ distinct |
| κ | penalty slope for infeasible policies (κ = 10) | Ch. 3 (Eq. 3.15) | κ_t(θ) is the PPO probability ratio (Eq. 3.20), marked by subscript t and argument θ. κ_a and κ_b are R1a/R1b penalty coefficients (Ch. 4 §4.12, reward table) | OK (context) |
| α | weight of the predicted-saving term in R_7B (α = 4) | Ch. 3 (Eq. 3.19) | none | OK |
| a, b | surrogate intercept and slope (−0.117; 13.762) | Ch. 3 (Eq. 3.18) | see a_l | OK (context) |
| η | weight of the behavioural-cloning (elite) loss, Phase-4 S2 only | Ch. 3 (§3.12, Eq. 3.13) | none | OK |
| Â_t, V_φ, 𝓗, c_v, c_e | PPO advantage, value function, entropy, value and entropy coefficients | Ch. 3 (Eq. 3.20) | Â_t (advantage) vs A_b/A_p (accuracy). Hat and time subscript distinguish them | OK (context) |
| MAC(**r**) | multiply–accumulate operations per image | Ch. 3 | none | OK |
| ρ*_s, ρ_60 | R1b target sparsity; sparsity of the maximal policy [5,5,5,5] (R2) | Ch. 4 (§4.12, reward table) | subscripted variants of sparsity ρ | OK |

## Result

All 16 symbols required by the brief are defined in Chapter 3, Table 3.2 or the equation where they first appear, and
are used with one meaning wherever they appear unmarked. The former advisory (bare ρ for Spearman correlation in
two Chapter-6 sentences) was resolved in the final MIST stage by writing "Spearman ρ"; no equation or value changed.
This inventory is the source of the List of Main Notation in `final_thesis_MIST.docx`.
