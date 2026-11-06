# Phase 7B (SimpleCNN, CIFAR-10): interpretation

**Status: confirmatory hardware validation STOPPED by the pre-registered energy-pilot rule.** No measured
latency or energy exists for any final model. Every statement below uses only the rules frozen in
`phase7b_final_preregistration.json`; no threshold was changed after the data were seen. The test set was not
used. Numbers come from `statistics.json`, `final_summary.csv` and `memory_results.csv`, which
`experiments/phase7b/stopped_summary.py` produced. The pre-registered `final_analysis.py` was not run, because it
needs the 5 energy sessions.

## 1. Why the hardware validation stopped

| Pilot | Conditions | E_1000 values, dense (J) | CV | Verdict |
|---|---|---|---|---|
| N = 100,000 (2026-10-27) | AC; gates at 7.4% and 9.2% load; idle package 9.1 W | 13.09, 11.41, 11.20, 15.13, 12.27, 12.27 | 11.36% | unstable |
| N = 200,000, first attempt | interrupted by the CPU-load gate (video encoder and Zoom running) | 18.00, 19.00, 22.12 (rep 0 only) | — | void (user decision) |
| N = 200,000 (2026-11-03) | AC; idle laptop; gates at 3.4% and 2.4%; idle package 1.8–2.2 W | 11.81, 12.07, 11.04, 11.54, 11.03, 11.03 | **4.01%** | **unstable** |

The rule is "CV ≤ 3%; if unstable, N doubles once; still unstable → STOP". The doubled-N pilot was unstable on a
clean, idle machine, so no session was run. Within each child process, block power fell (17.3 → 15.2 W and
16.0 → 15.0 W), which points to a warm-up or thermal transient. That is an observation only. It was not used to
change the protocol.

## 2. What is assessable

Reference: dense SimpleCNN after the same 5-epoch recovery, at VAL-SELECT 80.00% and VAL-RL 80.60%. Its state is
bit-identical to the Phase-7A dense model. Accuracy is preserved when ΔVAL-SELECT ≥ −0.5 pp.

| Model | Policy / widths | VAL-SELECT (Δ pp) | VAL-RL | Param memory | MACs | Surrogate Ê_1000 (prediction) |
|---|---|---|---|---|---|---|
| Dense | [32,64,128,256] | 80.00 (0.00) | 80.60 | 2,481,448 B | 10.85 M | 10.77 J |
| **New PPO** [2,2,2,5] | [26,51,102,102] | 78.88 (**−1.12**) | 80.00 | −63.4% | −36.0% | 8.23 J (−23.6%) |
| Uniform r = 0.208 (MAC-matched) | [25,51,101,203] | 78.76 (−1.24) | 79.76 | −37.3% | −36.2% | — |
| LAMP-structured s = 56.2% (param-matched) | [32,58,103,94] | 79.64 (**−0.36**) | 79.12 | −63.3% | −19.3% | — |
| Old PPO, Phase 7A [0,2,5,5] | [32,51,51,102] | 76.00 (−4.00) | 75.52 | −80.1% | −42.6% | 8.49 J |

The other confirmatory selections ([2,2,2,1], [3,1,3,2], [2,2,2,4], [3,1,3,0], [3,1,3,4], and [3,1,2,5], which is
also the enumeration reference) all land between −1.08 and −1.28 pp VAL-SELECT. Serialized checkpoint sizes follow
parameter memory to within 0.1 pp. "—" marks non-grid ratios, for which the surrogate has no prediction.

## 3. Pre-registered verdicts

- **GREEN-SUCCESS for the new PPO model: NO.** It fails the accuracy criterion (−1.12 pp). GREEN-SUCCESS needs all
  five criteria, so this verdict does not depend on the unmeasured latency and energy.
- **GREEN-SUCCESS frequency across seeds: 0 of 20.** Every seed selected a policy, and every selected policy fails
  accuracy.
- **Case: undetermined between GREEN-C and "none of GREEN-A–D".** Memory and MACs fall and accuracy fails. The
  case would be GREEN-C if energy were reduced above noise, and "none" otherwise. Energy was not measured, so it
  stays open. GREEN-A, GREEN-B and GREEN-D are excluded.
- **Comparisons with Uniform, LAMP and old PPO: no verdict.** The Phase-7A comparison rule needs session energy.
  Descriptively:
  - Against MAC-matched Uniform, accuracy is comparable (+0.12 pp, within ±0.5) and memory is 42% lower.
  - Against parameter-matched LAMP, the new PPO has 0.76 pp lower accuracy and 21% fewer MACs.
  - Against the Phase-7A PPO, the new PPO has 2.88 pp higher accuracy, but 11% more MACs and 84% more memory.
- **Pareto analysis, 2 of 4 planes assessable.**
  - Accuracy–MACs front: Dense, LAMP, [3,1,3,4] and old PPO.
  - Accuracy–memory front: Dense, new PPO, LAMP and old PPO.
  - The accuracy–energy and energy–MACs planes are not assessable.
- **Only one model is accuracy-preserving: LAMP-structured.** It cuts memory by 63% and MACs by 19%. Its latency
  and energy are unmeasured, so its GREEN status is not assessable.

## 4. Reading

1. **The accuracy constraint failed at the final stage.** All seven PPO selections had 2-epoch proxy accuracy
   q = 0.9890–0.9900, just above the frozen threshold τ = 0.98849. The constraint-first reward drives the search to
   the feasibility boundary, which is where the lowest predicted energy lies. Stage A accepted the proxy with a
   precision of 0.655, so about a third of proxy-feasible policies were expected to be truly infeasible. Here,
   every boundary policy recovered to −1.1 to −1.3 pp. The proxy rank correlation was high (ρ = 0.948), but that
   did not make the boundary itself reliable.
2. **Structured PPO is a large improvement over Phase 7A, but not enough.** Accuracy loss fell from −4.00 to
   −1.12 pp at a similar MAC level. The new method does not meet the 0.5 pp bar.
3. **A simple baseline beat it on the accuracy axis.** Parameter-matched LAMP-structured stayed within tolerance
   with the same memory reduction. It achieves this by pruning the classifier-adjacent unit hard and the first
   conv layer barely. The new PPO's extra MAC reduction cost about 0.8 pp of accuracy.
4. **Energy remains unknown.** The surrogate predicts a 24% saving for the new PPO. That figure is a model-based
   prediction from latency tables, not a measurement, so it supports no claim.

## 5. Limitations and disclosures

- Hardware energy and latency of the final models are unmeasured. The run-to-run variation of this laptop's
  package energy (CV 4.0% at N = 200,000) exceeded the pre-registered 3% stability limit.
- The interrupted first doubled-N attempt was declared void by user decision. Reruns used the same frozen rule and
  N. This is logged in `development_log.md`.
- A procedural error before the pre-registration (an early dry run used the real seeds) is disclosed in
  `development_log.md`. The real run reproduced those selections exactly.
- Figure: `figures/fig_accuracy_planes.png` shows the two assessable planes. The dashed line marks the
  accuracy-preservation bound (dense − 0.5 pp). The energy figures pre-registered for `final_analysis.py` do not
  exist.
- ResNet-8 is not started. The instruction was "ResNet-8 only after SimpleCNN is frozen", and that remains a
  separate decision.
