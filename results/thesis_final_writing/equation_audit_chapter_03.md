# Equation audit: Chapter 3

Every numbered equation in `chapter_03_framework.md` is tied to the implementation or to a frozen pre-registration.
"Exact" means the equation reproduces the implemented computation, including constants. "Standard" means a published
definition that the library implements, cited in the text. "Conceptual" means a definition stated in prose and
labelled as such in the chapter.

| Eq. | Content | Source file | Function / location | Variables | Status |
|---|---|---|---|---|---|
| 3.1 | policy **r** = [r_1..r_{N_L}], r_l ∈ {0, .1, .2, .3, .4, .6} | `src/pruning/__init__.py` | `ACTION_TO_PRUNE`; `UNIT_SPECS` (4 units per architecture) | r_l, N_L | exact |
| 3.2 | sparsity ρ = 100 · zeros / weights over all Conv2d/Linear tensors (including the excluded classifier) | `src/pruning/__init__.py` | `total_sparsity`; module docstring "Sparsity" | ρ | exact |
| 3.3 | max_θ E[R] | — | — | θ, R | conceptual (PPO objective; labelled "conceptual, not a guaranteed optimum") |
| 3.4 | 7-feature state | `src/rl/__init__.py` | `unit_state` (lines 54–58); `PruningEnv._state`. Sensitivity slot 0: `phase3_preregistration.md` §4 "State", `phase4_preregistration.md` l.49, `phase5_preregistration.md` §3 | l, n_l, w_l, r_j | exact |
| 3.5 | r(a) mapping | `src/pruning/__init__.py` | `ACTION_TO_PRUNE = {0:0.0, …, 5:0.6}` | a, r | exact |
| 3.6 | 6⁴ = 1,296 | `src/pruning/__init__.py` docstring; landscapes of 1,296 rows (`phase5/landscapes`, `phase7b_green_pruning/proxy_landscape.csv`) | — | — | exact |
| 3.7 | R0 = 1.5·A_p/A_b + 0.01·ρ(%) + 0.05·D | `src/rl/__init__.py` | `reward_components` (`ACCURACY_COEF` 1.5, `DIVERSITY_COEF` 0.05); λ_s = 0.01 from `phase3_preregistration.md` §4 and `phase4_preregistration.md` §0.1; `experiments/phase5/phase5_lib.py` `r0` (0.01) | A_p, A_b, ρ, D | exact (library default 0.04 noted as historical) |
| 3.8 | S_raw = mean over {.1, .2, .4, .6} of (L_r − L_0)/L_0 | `src/sensitivity/__init__.py` | `loss_sensitivity`, `LOSS_RATIOS` | L_0, L_{l,r} | exact |
| 3.9 | min–max normalisation | `src/sensitivity/__init__.py` | `min_max` | S_l | exact |
| 3.10 | z̃ = z − c_l · r(a)/0.6 | `src/rl/__init__.py` | `prior_matrix` (B[u,a] = β·S_u·ratio(a)/0.6); `SoftPriorPolicy._get_action_dist_from_latent` (logits − prior[unit]). Phase-3 form `β·prior_u·ratio(a)/60` with ratio in % (`phase3_preregistration.md`) is identical | z, c_l, r(a) | exact (P1: c_l = 0.5·S_l) |
| 3.11 | c_l = 1.0·(S_l − S̄) | `experiments/phase4/phase4_lib.py` | `prior_coefficients` (`BETA_NEW` 1.0; branch P2), `prior_matrix`; `phase4_preregistration.md` §4; `phase5_preregistration.md` §1 "Prior P2" | S_l, S̄ | exact |
| 3.12 | archive top-10 by VAL-RL reward; argmax VAL-SELECT score (ties: higher VAL-RL reward, earlier) | `experiments/phase4/phase4_metrics.py` | lines 82–83 (`uniq = sorted(...)[:ARCHIVE_K]`; `arc = sorted(..., key=(-score_vselect, -reward_vrl, first))[0]`); `ARCHIVE_K = 10` (`phase4_lib.py` l.62) | 𝒜, 𝒞 | exact |
| 3.13 | 𝓛 = 𝓛_PPO + η·𝓛_elite; 𝓛_elite = −mean log π over the 4K elite pairs | `experiments/phase4/phase4_lib.py` | `ElitePPO.train` (l.260: `loss + bc_eta * (-elite_log_prob.mean())`); `ArchiveCallback` (K = 5, active once ≥ 5 unique policies); `ETA = 0.1` | η, K | exact |
| 3.14 | argmax ρ over {q_SELECT ≥ τ}; ties and fallback | `experiments/phase5/select_archive.py` | `select` (rule in the docstring, l.9); `phase5_preregistration.md` §4.2 | ρ, q, τ | exact |
| 3.15 | R_c = 1 + ρ/100 + 0.01·q if q ≥ 0.98, else −10·(0.98 − q) | `experiments/phase5/phase5_lib.py` | `r_constrained` (l.58–62); `EPS, KAPPA = 0.01, 10.0` (l.40); τ = 0.98 in `phase5_config.json` / `phase5_tau_selection.md` | q, ρ, τ, ε, κ | exact |
| 3.16 | η_e = ½η_0(1 + cos(πe/E)) | `experiments/phase6/phase6_lib.py` | `make_optimizer`: `CosineAnnealingLR(T_max=epochs, eta_min=0)`, stepped per epoch; `PROTOCOL` lr 1e-4 | η_0, E | standard (PyTorch closed form with η_min = 0) |
| 3.17 | k_l = max(1, round((1 − r_l)·C_l)) | `src/pruning/structured_units.py` | `keep_count`; L1 ranking in `structured.l1_keep` (docstring) | k_l, C_l, r_l | exact |
| 3.18 | Ê = −0.117 + 13.762·T_table | `results/phase7b_green_pruning/phase7b_final_preregistration.json` | `surrogate.coefficients = [-0.11699…, 13.76184…]`, features "a + b·L_table"; `energy_surrogate_metrics.json` (accepted model 2) | Ê, T_table | exact (coefficients rounded to 3 decimals) |
| 3.19 | R_7B = 1 + α·ŝ/s_max + 0.01·q̂ if q̂ ≥ τ̂, else −10·(τ̂ − q̂); ŝ = 1 − Ê/Ê_dense | `experiments/phase7b/ppo.py` | `Inputs.reward` (l.63–67); `s_max` (l.40); `KAPPA, EPS = 10.0, 0.01` (l.26); α = 4 in the pre-registration (`reward.alpha`) | q̂, τ̂, ŝ, α | exact |
| 3.20 | PPO clipped objective with value and entropy terms | Stable-Baselines3 2.8.0 `PPO.train` (clip 0.2, vf_coef 0.5, ent_coef 0.01) | `src/rl/__init__.py` `PPO_KWARGS`; `phase3_preregistration.md` hyper-parameters | κ_t, Â_t, c_v, c_e | standard [@schulman2017ppo]; constants exact |

## Non-equation definitions checked against code

| Statement in Chapter 3 | Source |
|---|---|
| Reward paid at episode end, 0 at intermediate steps | `src/rl/__init__.py` `PruningEnv.step` |
| Unit pruning = L1 unstructured per tensor with round(r·\|t\|) weights | `src/pruning` `apply_unit_ratio` → `torch.nn.utils.prune.l1_unstructured`; PyTorch `_compute_nparams_toprune` returns `round(amount * tensor_size)` |
| Units per architecture (Table 3.3) | `src/pruning/__init__.py` `UNIT_SPECS`, `EXCLUDED` |
| PPO hyper-parameters (Table 3.7) | `src/rl/__init__.py` `PPO_KWARGS`; `phase3_preregistration.md`; `phase5_preregistration.md` §3; `phase7b_final_preregistration.json` `ppo` |
| Terminal deterministic policy in Phase 3 | `phase3_preregistration.md` "Final policy"; `src/rl` `deterministic_rollout` |
| Phase-5 archive frozen before VAL-SELECT | `phase5_preregistration.md` §4.1; `select_archive.py --freeze-archive` |
| Phase-6 protocol (Adam, β, ε, wd 0, cosine, E = 5, batch 128, crop + flip, BN train mode, no selection, seed + 100, shared stream) | `experiments/phase6/phase6_lib.py` `PROTOCOL`, `FT_SEED_OFFSET`, `finetune`; `phase6_preregistration.md` §4 |
| Structured units and residual coupling (ResNet-8) | `src/pruning/structured_units.py` module docstring |
| 7B state (8 features) | `experiments/phase7b/ppo.py` `StructuredEnv._state` |
| 7B prior β = 1.0 with centred structured sensitivity | `ppo.py` `prior_matrix` (`BETA = 1.0`); `structured_sensitivity.json` `centred` |
| 7B proxy q̂ = A_2/A_2,dense; validated before the search | `phase7b_final_preregistration.json` `proxy` (epochs 2, q = A_E/A_E,dense) |
| 7B selection: lowest Ê, 1% tie band, then q̂, MACs, memory, vector | `ppo.py` `select`, `TIE_BAND = 0.01` |
| 7B sensitivity on 2,500 VAL-RL images, structured probes {.1, .2, .4, .6} | `structured_sensitivity.json` (`split`, `ratios`) |
| 7A: modal Phase-5 C1 policy; equal 5-epoch recovery, seed 600 | `results/phase7a_sustainability/selected_policies.json`, `preregistration.json` |

All 20 equations are tied to code or to a frozen pre-registration. None is unverified.
