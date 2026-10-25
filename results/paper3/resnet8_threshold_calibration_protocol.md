# Paper 3 — ResNet-8 proxy-threshold calibration protocol (PI decision 2026-10-25)

Committed together with the frozen policy list `resnet8_threshold_calibration_policies.json`,
**before any 5-epoch calibration run**. Code: `experiments/paper3/h2e3.py`.

## Scope decision (PI)

- **SimpleCNN** failed the pre-registered proxy-validity criterion (Spearman ρ = 0.844 < 0.90;
  `paper3_proxy_decision.json`).
  - It is NO-GO for proxy-based PPO. The criterion is not relaxed, and no SimpleCNN 1-epoch landscape
    or PPO is run.
  - SimpleCNN stays in the paper's **hardware-analysis** section, using its frozen exhaustive latency
    table: correlations of parameter and MAC reduction with latency, sparsity/latency inversions,
    fc-heavy versus conv-heavy behaviour, and measured speed-ups.
- **The RL experiment** is CIFAR-10 ResNet-8 only. This narrowing follows from the failed validity
  criterion, not from any PPO result; none exists.

## Sample (selection uses the frozen 1-epoch landscape's q₁ only)

- **Source.** `h2_c10_resnet8_1epoch_landscape.csv`, frozen in `h2_e2_freeze.json`.
- **Bins on q₁:** [0.85, 0.90), [0.90, 0.92), [0.92, 0.94), [0.94, 0.96), [0.96, 0.98), [0.98, ∞).
- **Draw.** 8 policies per bin, uniformly without replacement, with numpy `default_rng(20261025)`, bins
  in that order, each pool sorted by policy string. A bin with fewer than 8 policies gives all it has,
  and the shortfall moves to the nearest lower-q₁ bin.
- **Not used for selection:** latency, any 5-epoch value, or any efficiency measure.
- **Freeze.** The 48 IDs are frozen in `resnet8_threshold_calibration_policies.json` before any 5-epoch
  run.

## 5-epoch recovery

- **Protocol.** Each of the 48 runs from the frozen dense reference, is compacted, and is recovered
  with the original frozen Phase-6 protocol: 5 epochs, seed 600, data generator 600 + 10⁶, 1 thread
  (`h2_recovery.recover_one` with E = 5).
- **Measures.** VAL-RL; no test access.
- **Definitions.**
  - q₅ = A₅ / A₅(identity), with A₅(identity) = 78.40, the 5-epoch identity record of the E2 validation
    (`_h2e2_validation_5epoch`), computed with the same deterministic protocol;
  - q₁ comes from the frozen landscape;
  - δ = q₅ − q₁.

## Threshold rule (fixed now)

- **True feasibility:** q₅ ≥ 0.98.
- **Stop condition.** If fewer than **5** calibration policies are truly feasible: **STOP**. No τ_proxy
  and no PPO; this is reported as insufficient evidence.
- **Choice of t.** t is the **largest** q₁ threshold that satisfies both:
  1. 100% recall of the truly feasible calibration policies, i.e.
     t ≤ min{q₁ : calibration policy with q₅ ≥ 0.98};
  2. at least **20** of the 1,296 landscape policies have q₁ ≥ t.

  Hence t = min(t_recall, the 20th-largest q₁ of the landscape).
- **Fixed safety margin:** **τ_proxy = max(0, t − 0.005)**. A policy is proxy-feasible if q₁ ≥ τ_proxy.
- **Reported:** τ_proxy; the proxy-feasible count; the truly feasible calibration count; recall,
  precision, false positives and false negatives at τ_proxy; the distribution of δ.

## Latency-room rule (after τ_proxy is frozen)

- **Inspected.** Among the proxy-feasible policies of the full landscape: count, latency min and max,
  best table speed-up, parameter-reduction range, MAC-reduction range, and the number of Pareto
  policies.
- **GO to PPO** only if there are ≥ 20 proxy-feasible policies **and** at least one has a frozen-table
  speed-up ≥ **1.10×**. Otherwise **STOP**: the hardware-aware objective would lack optimisation room.

## Then (only if GO)

- The final ResNet-8 structured sensitivity vector (`paper3_sensitivity_definition.md`) is computed and
  frozen.
- `paper3_ppo_preregistration.md` and `paper3_preppo_freeze.json` are written.
- PPO waits for the PI's approval.
