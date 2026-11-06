# Phase 7A: sustainable-computing evaluation of adaptive pruning

**Status: complete and frozen.** Pre-registration `preregistration.json`, commit `b8c2766`, written before
any model was built or measured. Every number below comes from the files in this folder. The rules applied
are the pre-registered ones, without change.

## What was evaluated

**Deployment realization.** The thesis PPO policies are unstructured masks, and masks do not reduce dense
CPU work. In this phase each method's per-unit pruning ratios are realized as physical channel and filter
removal with the validated Paper-3 code. Each model is then recovered for 5 epochs with the frozen Phase-6
protocol, and its CPU package energy is measured. No claim is made about the masked thesis models
themselves.

**Models (CIFAR-10, channel widths):**

- **SimpleCNN** (dense widths [32, 64, 128, 256]):
  - PPO-Structured: action vector [0,2,5,5], the most frequent Phase-5 C1 policy (3 of 20 seeds),
    giving [32, 51, 51, 102].
  - Uniform-Structured: common ratio 0.256, MAC-matched, giving [24, 48, 95, 190].
  - LAMP-Structured: [32, 58, 102, 87].
- **ResNet-8** (dense widths [16, 16, 32, 64]):
  - PPO-Structured: action vector [1,1,1,3], the most frequent policy (9 of 20 seeds), giving [14, 14, 29, 45].
  - Uniform-Structured: common ratio 0.149, giving [14, 14, 27, 54]. Its MACs are 2.9% from PPO's;
    integer channel counts prevent a closer match.
  - LAMP-Structured: [15, 15, 27, 45].

**Approximate ResNet-8 mapping.** The thesis units stage2 and stage3 pruned every weight in their
stage. The structured units cut only block-internal channels and a coupled stage-1 residual width.

**Energy backend.** The primary counter is Intel RAPL package energy
(`\Energy Meter(RAPL_Package0_PKG)`), read through the Windows Energy Meter Interface:

- Units are picowatt-hours, refreshed every 1 ms, and agree with the counter's own power value.
- No TDP figure or software estimate is used.
- The core counter (PP0) and net-of-idle energy are secondary metrics.
- The DRAM domain is not available on this machine.

**Measurement protocol.**

- Hardware and software: Intel Core i7-1260P, Windows 11, PyTorch 2.7.1 CPU build with oneDNN 3.7.1.
- Run conditions: AC power, Balanced power plan, batch 1, one thread, pinned to logical CPU 2.
- Sessions: five fresh-process sessions per architecture, with the dense anchor in every session.
- Each model per session: 1,000 warm-up inferences, then one block of 100,000 measured inferences.
- Order: randomized within each session.
- System CPU load before each session: 2.2–4.3%.
- Stability pilot: dense-only, energy CV 2.30% for SimpleCNN and 1.74% for ResNet-8 (rule: at most 3%).

## Results

| Architecture | Method | Accuracy (%) | ΔAcc (pp) | Params ↓ | MACs ↓ | Latency (ms) | Speed-up | Energy / 1000 inf. (J) | Energy ↓ | Normalized EDP |
|---|---|---|---|---|---|---|---|---|---|---|
| SimpleCNN | Dense | 80.60 | +0.00 | 0.0% | 0.0% | 0.7313 | 1.000× | 11.054 | 0.0% | 1.000 |
| SimpleCNN | PPO-Structured | 75.52 | −5.08 | 80.1% | 42.6% | 0.5711 | 1.280× | 8.541 | 22.8% | 0.603 |
| SimpleCNN | Uniform-Structured | 79.12 | −1.48 | 44.7% | 42.5% | 0.5341 | 1.373× | 7.906 | 27.9% | 0.518 |
| SimpleCNN | LAMP-Structured | 78.84 | −1.76 | 65.5% | 19.7% | 0.6579 | 1.117× | 9.760 | 12.1% | 0.783 |
| ResNet-8 | Dense | 78.40 | +0.00 | 0.0% | 0.0% | 0.6984 | 1.000× | 10.882 | 0.0% | 1.000 |
| ResNet-8 | PPO-Structured | 75.16 | −3.24 | 25.0% | 21.5% | 0.6400 | 1.092× | 9.592 | 13.4% | 0.791 |
| ResNet-8 | Uniform-Structured | 75.60 | −2.80 | 16.1% | 19.3% | 0.6459 | 1.074× | 9.645 | 11.8% | 0.818 |
| ResNet-8 | LAMP-Structured | 75.24 | −3.16 | 25.0% | 18.2% | 0.6490 | 1.068× | 9.819 | 8.9% | 0.828 |

How to read the table:

- Accuracy is VAL-RL after the identical 5-epoch recovery.
- ΔAcc is measured against the recovered dense model.
- Latency, energy, speed-up and EDP are medians over 5 sessions.
- Savings and speed-ups are paired with the same-session dense anchor.

**Accuracy.**

- No pruned model meets the pre-registered ±0.5 pp rule.
- Retained accuracy q for PPO-Structured is 0.937 on SimpleCNN and 0.959 on ResNet-8.
- The recovered dense models are bit-identical to Paper 3's independent identity runs.

**Energy.**

- Every pruned model used less package energy than the dense model in 5 of 5 sessions. The exact
  one-sided Wilcoxon p = 0.031 in every case, which is the smallest value attainable with n = 5.
- All median savings exceed the pre-registered noise threshold: twice the dense between-session CV, which
  is 4.3% for SimpleCNN and 3.7% for ResNet-8.
- The core counter (PP0) and net-of-idle savings agree with the package result to within 2 percentage
  points.
- Package power during inference was about 15 W for every model, so the energy savings follow the latency
  savings almost one for one.

**Variability.** Across the 5 sessions, the coefficient of variation of energy per 1,000 inferences was
1.5–3.6%. PPO's per-session energy saving ranged from 19.5% to 25.7% on SimpleCNN and from 5.6% to 14.5%
on ResNet-8.

## Pre-registered interpretation

**SimpleCNN: Case B, together with Case D.**

- **Case B.** The policy reduces operational energy, by 22.8%, but the saving comes with a measurable
  accuracy trade-off: −5.08 pp.
- **Case D: Uniform-Structured is better.** At almost the same MACs (42.5% vs 42.6% removed), uniform
  structured pruning is 3.60 pp more accurate than PPO-Structured. It also used less energy in all 5
  sessions: 7.91 vs 8.54 J per 1,000 inferences, 27.9% vs 22.8% saving.
- **Why.** The PPO allocation concentrates removal in the fully connected layer and conv3. Its 80%
  parameter reduction therefore buys less time than uniform channel removal does at the same MAC count.
  This repeats the Paper-3 finding that parameter reduction is a weak predictor of CPU cost.
- **PPO vs LAMP-Structured.** Neither dominates. PPO used less energy in 5 of 5 sessions (12.5% lower),
  but was 3.32 pp less accurate.

**ResNet-8: Case B.**

- The policy reduces operational energy, by 13.4%, but with a measurable accuracy trade-off: −3.24 pp.
- **PPO vs Uniform:** no clear difference. Accuracy differs by 0.44 pp, PPO used less energy in only 3 of
  5 sessions, and the median difference is 2.1%.
- **PPO vs LAMP:** no clear difference. Accuracy differs by 0.08 pp, and PPO used less energy in 4 of 5
  sessions.

**Primary success criterion (Case A): not met for either architecture.** The positive sustainable-computing
claim allowed by the pre-registration, that the policy "reduces measured CPU inference energy while
preserving predictive performance", is **not supported** by this experiment.

**What can be claimed.** Physically compacting networks along the layer-wise allocation learned by the
adaptive pruning policy reduced measured CPU package energy per inference: by 22.8% for SimpleCNN and 13.4%
for ResNet-8, consistently across five sessions. After an identical 5-epoch recovery, however, these
models lost 5.1 and 3.2 percentage points of validation accuracy.

- For SimpleCNN, a uniform structured allocation at matched MACs achieved both a larger energy saving and
  higher accuracy.
- For ResNet-8, PPO, uniform and LAMP-derived allocations were indistinguishable in their energy–accuracy
  trade-off.

## Deployment-scale projection (not a direct experiment; no CO₂e conversion)

The projection is the measured median per-inference package saving multiplied by N.

| N inferences | SimpleCNN PPO | SimpleCNN Uniform | ResNet-8 PPO | ResNet-8 Uniform |
|---|---|---|---|---|
| 10⁶ | 0.0007 kWh | 0.0009 kWh | 0.0004 kWh | 0.0003 kWh |
| 10⁷ | 0.0070 kWh | 0.0087 kWh | 0.0036 kWh | 0.0034 kWh |
| 10⁸ | 0.070 kWh | 0.087 kWh | 0.036 kWh | 0.034 kWh |

## Limitations

- **Hardware and run configuration.** One laptop CPU, Windows 11, PyTorch CPU backend, batch 1, one thread.
- **Energy scope.** Package energy (and cores) only. DRAM and whole-system power were not measured, and
  package energy includes the package's idle share.
- **Data.** CIFAR-10 only; two small networks.
- **Accuracy evidence.**
  - Validation accuracy only, from a single recovery seed and 2,500 images, so 1 image = 0.04 pp.
  - PPO's policies were trained and selected on these validation halves, so their validation accuracy may
    if anything be optimistic.
- **Not a test of PPO as trained.** The structured realization of a policy learned for unstructured masks
  is not the setting PPO optimized. These results test transferring the learned allocation to structured
  deployment, not a structure-aware PPO.
- **ResNet-8 mapping.** The structured realization of ResNet-8 is approximate.
- **Recovery budget.** Five recovery epochs may favour less aggressive allocations. Longer recovery was
  not tested.
- **Uniform matching.** The ResNet-8 uniform baseline is matched to PPO's MACs only to within 2.9%.
- **Raw data saved.** Only session-level latency statistics were stored, not every one of the 100,000
  per-inference times.
- **No carbon claim.** No emissions were quantified.
