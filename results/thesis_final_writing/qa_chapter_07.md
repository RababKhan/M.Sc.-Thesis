# QA report: Chapter 7 (Discussion)

- **File:** `chapter_07_discussion.md`.
- **Status:** **FROZEN (v1, 2026-11-05)**, pending the user's review.
- **Number check:** `tools/audit_ch7_numbers.py`. Every numeric token in Chapter 7 occurs in the audited Chapters 4–6;
  no new number was introduced.

## 1. Word count

**5,648** counted words, of which about **320** are inside two synthesis tables, so there are about **5,330 words of
prose**. That is within the 5,000–7,000 target; about 13–14 pages.

## 2. Subsection structure

| § | Title |
|---|---|
| 7.1 | Chapter Overview |
| 7.2 | What Adaptive Layer-Wise PPO Actually Contributes |
| 7.3 | The Role of Sensitivity Information (includes why centring mattered) |
| 7.4 | Search Quality versus Policy Retention (includes elite behavioural cloning) |
| 7.5 | Reward Design and Over-Pruning (includes the accuracy-constrained reward) |
| 7.6 | Validation-to-Test Transfer |
| 7.7 | Why LAMP Remains a Strong Benchmark, and Why Equal Fine-Tuning Changes the Rankings (7.7.1 LAMP and RL cost; 7.7.2 equal fine-tuning and PPO vs Global) |
| 7.8 | Compression versus Real Deployment Efficiency (masked vs structured; sparsity is not speed; accuracy-feasible headroom) |
| 7.9 | Accuracy–Efficiency Trade-offs and the Three Levels of Efficiency |
| 7.10 | Interpreting the Sustainability Experiments (Phase 7A, Phase 7B, surrogate, measurement failure, wording) |
| 7.11 | What the Negative Results Establish (table) |
| 7.12 | Generalisability |
| 7.13 | Limitations (groups A–F) |
| 7.14 | Overall Scientific Contribution (stage table; how RQ1–RQ6 form a chain) |
| 7.15 | Chapter Summary |

The requested headings are all covered. "Accuracy–Efficiency Trade-offs" is merged with the three-level discussion
(7.9). "What Phase 7B Improved and What It Did Not Solve" is part of 7.10. A section on negative results (7.11) was
added for task §25.

## 3–8. RQ interpretations

| RQ | Interpretation in Chapter 7 |
|---|---|
| RQ1 | PPO's value is flexibility: a learnable allocation adaptable to changing objectives. It is not universal superiority: it is far better than Uniform and Random, mixed against Global and ERK, and LAMP is stronger. RL is an adaptive mechanism for objectives not expressible as one saliency score; it is not claimed to be necessary (7.2, 7.7.1) |
| RQ2 | Sensitivity is a search bias. It changes which policies are sampled, not which one is returned. Centring made it layer-specific; one accuracy-based exploration measure reversed against the conservative control. Not "does not work"; its contribution is exploration guidance (7.3) |
| RQ3 | Search quality and retention are separate problems. Studies should report the best visited policy, the retained policy and the selection procedure. The archive retains validation rank, more reliably than in-training imitation in this experiment. It is not a generalisation mechanism (7.4) |
| RQ4 | A stronger optimiser exposed a misaligned reward; the optimiser optimised what was asked. The constraint is an optimisation constraint, not a generalisation guarantee. The 67.5% transfer is discussed with possible, not isolated, explanations (7.5–7.6) |
| RQ5 | Equal recovery narrowed differences (0.00–0.22 pp PPO–LAMP; ≤ 0.12 pp PPO–Global), within the ±0.5 pp CI rule, which is **not** formal equivalence; LAMP stays ahead in 3. Methodological implication: compare after equal recovery. Bounded to small CNNs and moderate sparsity (7.7.2) |
| RQ6 | Sparsity is not speed. Masked and structured pruning serve different roles. Accuracy-feasible headroom is small: 1.098× proxy-feasible and 1.089× confirmed, neither ≥ 1.10×. Three levels each need evidence. Phase 7A shows real energy savings but not a better model; Phase 7B improved the trade-off without solving the objective (7.8–7.10) |

## 9. LAMP positioning

LAMP is the strongest method before recovery and is described as calibrating, not invalidating, the thesis. Its
possible explanation is labelled as untested. The RL cost comparison is qualitative (512 evaluations per run, seeds,
reward design, archive versus one pass), and the future advantage of RL for complex objectives is explicitly not
claimed as shown (7.7.1).

## 10. Sensitivity positioning

Sensitivity is a search bias, not a final-accuracy predictor. The Phase-3 conservative-bias possibility, the Phase-4
centring and the Phase-4 reversed measure are all included (7.3).

## 11. Archive positioning

The archive improves validation-rank retention; its test effect is mixed (+ in 4 settings, −5.66 / −4.00 pp on
ResNet-8). The chapter describes it as "retention amplifies whatever the reward specifies" and as not a generalisation
mechanism. Behavioural cloning is bounded to the tested configuration (7.4).

## 12. Fine-tuning positioning

The recovery effect within a setting is described as isolated by the design (shared seeds and data, masks fixed). Its
mechanism is described as not isolated. The chapter states it is not formal equivalence (7.7.2).

## 13. Deployment positioning

- masked versus structured pruning, with structured not universally better;
- Paper 3 correlations, with the SimpleCNN example presented as arithmetic;
- accuracy-feasible headroom in the corrected wording (7.8–7.9).

## 14. Sustainability positioning

There is no "green", environmental or carbon claim. §7.10 states explicitly what is *not* established and what is
(measured Phase-7A CPU package energy, with its accuracy cost). Lifecycle, training energy and carbon are out of scope.
"Sustainability-oriented evaluation" and "deployment-oriented structured PPO" are used, as requested.

## 15. Phase 7B boundary

- **Improvements:** memory −63.4% and MACs −36.0%, at −1.12 pp against −4.00 pp on the same VAL-SELECT split.
- **Not solved:**
  - missed the 0.5 pp criterion (0/20 seeds);
  - no measured latency;
  - energy unresolved;
  - no clear advantage over Uniform or LAMP.
- **Surrogate:** useful for search, but not measured energy.
- **Measurement failure:** "evidence neither for nor against the model's energy use" (7.10).

## 16. Limitations covered (7.13)

| Group | Content |
|---|---|
| A. Scale | CIFAR, small CNNs, enumerable space; CIFAR-10 only for 7A/7B; SimpleCNN only for 7B |
| B. Optimisation | PPO cost; sensitivity (search only); LAMP stronger; reward dependence; one behavioural-cloning configuration |
| C. Validation | 67.5% transfer; τ and split size; single recovery seed in the deployment phases; older SimpleCNN recipe; Phase 6 not fully independent; CI rule ≠ equivalence; fixed 5-epoch recovery |
| D. Hardware | one CPU; no GPU, mobile or accelerator; SimpleCNN headroom unknown; H2-C stopped |
| E. Energy | package only; no DRAM or wall measurement; failed qualification; proxy and surrogate are not measurements |
| F. Sustainability | no carbon, lifecycle or training-energy analysis; 7A kWh figures are projections only |

## 17. Causal-language audit

**Searched:** because, causes/caused, leads to/led to, results in/resulted in, proves/proved, demonstrates why, due to,
drives/drove. Every hit was inspected:

| Line (approx.) | Phrase | Causality isolated? | Verdict |
|---|---|---|---|
| §7.6 item 2 | "The constrained reward drives the agent to the *most compressed* feasible policy" | **by construction**: the reward's optimum is the most compressed feasible policy (Eq. 3.15) | kept; definitional, not empirical |
| §7.8 | "no accuracy-feasible headroom could be established at all, because its accuracy proxy failed validation" | **protocol logic**: the pre-registered rule forbade use of a failed proxy | kept |
| §7.10 | "energy remained unresolved, because the measurement setup repeatedly failed its pre-registered stability check" | **protocol logic**: the stop rule prevented any comparison | kept |
| §7.11 | "summarised here because they carry much of its methodological weight" | editorial reason | kept |
| §7.11 | "retention led to the archive; over-pruning led to the constraint; the lossy transfer led to Phase 7B" | **design history**: documented decisions, not causal claims about data | kept |
| §7.11 | "credible partly *because* the negative ones are visible" | flagged in the text itself as "a logical point about evidential weight, not an empirical causal claim" | kept |
| §7.7.2 | "within each setting the narrowing of the gaps is an effect of equal recovery" | **yes**: the same fixed mask is evaluated before and after an identical recovery with shared seeds and data, so recovery is the only intervention. The *mechanism* is explicitly stated as not isolated | kept |
| §7.5 | "Comparing … C1 vs C0 isolates the reward as the manipulated factor" | **yes**: same prior, archive, seeds and splits; only the reward differs | kept |
| §7.9 | "average package power was nearly the same for every model, so energy savings followed latency savings" | **arithmetic** (E = P·t), stated as an observation about this platform, "not a general law" | kept |

**Non-isolated mechanisms are hedged throughout:**
- the retention gap: "a possible mechanism (not isolated)";
- sensitivity: "the thesis can describe but not prove";
- LAMP's strength: "the thesis did not test this explanation";
- the Phase-7A transfer loss: "consistent with Paper 3 but not isolated";
- the validation-to-test failures: "possible explanations (none isolated)";
- the measurement failure: "does not claim to know the cause".

## 18. Dangerous-claim audit

**Searched:** PPO beats LAMP, superior, improves accuracy, improves test, guarantee, equivalent,
accuracy-preserving, green, environmentally, carbon, eco-friendly, Phase-7B energy or latency reduction,
hardware-aware RL succeeded.

| Hit | Verdict |
|---|---|
| "superiority" (×3) | only in "does not establish superiority" / "not universal superiority" |
| "PPO did not beat LAMP" | negative result (table, §7.11) |
| "equivalent" (×4) | "not equivalent" (behavioural cloning vs archive); "not a formal equivalence test" / "not equivalent" (Phase 6, as required by task §15); limitations |
| "guarantee" (×6) | all negations: "not a generalisation guarantee", "does not guarantee latency/energy/test feasibility" |
| "green", "carbon", "environmentally" | only in §7.10's statement of what is **not** established and in limitations |
| "accuracy-preserving", "eco-friendly", "sustainable AI system", "lower carbon footprint" | absent, except the negated "lower in carbon footprint" in §7.10 |
| Phase-7B energy or latency reduction | never claimed; "no measured latency", "energy is unresolved" |
| Hardware-aware RL | only "was NO-GO" |

All positive claims are at or below `claim_matrix.md` rows 1–15.

## 19. New numerical claims

**None.** `tools/audit_ch7_numbers.py` finds all 77 numeric tokens in the audited Chapters 4–6 (or among design
constants). For example, the Phase-6 sparsity range "15.6–56.9%" is taken from Table 5.7. Cross-references are
excluded from the check.

## 20. Unresolved issues

1. **Interpretive claims are labelled.** The LAMP explanation (7.7.1) and the retention mechanism (7.4) are offered as
   interpretations and labelled as untested. They could be tested only by new experiments, which are out of scope.
2. **Template.** Still blocked: no official thesis template.
3. **Next chapters.** Chapter 8 (conclusion and future work) and Chapter 1 follow only after review. The future-work
   items implied in 7.6, 7.7.1, 7.10 and 7.12 should be carried into Chapter 8 consistently.

## Citations

16 references, all verified and all already used in Chapter 2: `he2018amc`, `yu2020evalnas`,
`ecoffet2021goexplore`, `oh2018sil`, `pan2022misspecification`, `skalse2022rewardgaming`, `achiam2017cpo`,
`lee2021lamp`, `renda2020rewinding`, `le2021retraining`, `blalock2020state`, `hoefler2021sparsity`,
`dehghani2022misnomer`, `khan2018rapl`, `jay2023powermeters` and `raffin2024dissecting`. Each characterisation matches
its Chapter-2 use. No new reference was added, and the frozen reference audit was not rebuilt.

## Freeze

Chapter 7 v1 is frozen. Later changes are limited to citation-number assembly or factual corrections, and will be
logged here.
