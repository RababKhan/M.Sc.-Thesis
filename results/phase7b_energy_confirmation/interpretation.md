# Phase 7B energy confirmation: interpretation

**ENERGY-CONFIRMED = NO. Direct energy confirmation remains unresolved because the available measurement setup did
not satisfy the stability requirement.**

## Question and status

Question: does the frozen Phase 7B PPO-Structured SimpleCNN ([2,2,2,5], widths [26,51,102,102]) use less measured CPU
inference energy than the recovered dense SimpleCNN?

This was a separate measurement study. The protocol (`protocol.json`, commit `df233be0`) was frozen before any data.
It did not change the frozen Phase 7B experiment or its verdicts: GREEN-SUCCESS = NO, accuracy −1.12 pp, and a
pilot that failed at CV 4.01% > 3%.

## What was done

1. **Checkpoints verified** (`checkpoint_verification.json`). All 5 frozen models were checked: Dense, Phase 7B PPO,
   Uniform, LAMP and the old Phase-7A PPO. File and state-dict SHA-256 hashes match the Phase 7B / 7A freeze
   records.
2. **Measurement qualification** (Dense only, frozen protocol). Settings:
   - RAPL package energy via the Windows Energy Meter Interface;
   - AC power, Balanced plan, logical CPU 2, 1/1 threads, batch 1, float32;
   - `inference_mode`, preallocated input;
   - 180 s thermal soak, then per block 2,000 warm-up and 200,000 measured inferences;
   - 3 fresh processes × 2 blocks.

| Run | Block | E_1000 (J) | Package W | Core (PP0) W | Non-core W | Median latency (ms) | CPU 2 clock (% nominal) | Thermal zone (°C, start→end) | System load |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 1 | 11.417 | 11.16 | 7.40 | 3.75 | 1.0456 | 130 | 86 → 76 | 8.4% |
| 1 | 2 | 14.278 | 13.76 | 8.64 | 5.12 | 1.0513 | 133 | 77 → 83 | 9.2% |
| 2 | 1 | 12.962 | 12.72 | 8.05 | 4.68 | 1.0479 | 131 | 86 → 84 | 8.9% |
| 2 | 2 | 15.221 | 14.51 | 9.66 | 4.85 | 1.0517 | 135 | 79 → 92 | 10.2% |
| 3 | 1 | 9.514 | 9.40 | 6.91 | 2.48 | 1.0420 | 129 | 76 → 81 | 7.6% |
| 3 | 2 | 9.473 | 9.09 | 6.70 | 2.39 | 1.0430 | 125 | 72 → 73 | 7.9% |

The 6 blocks average 12.14 J per 1,000 inferences, with an SD of 2.42 J and a **CV of 19.92%**. The pre-registered
limit is 3%, so the setup is **not qualified**.

All runs passed the validity rules: the pre-run gates were at 3.7–5.7% load and in-block system load was 7.6–10.2%,
against a 15% limit. The measuring process alone accounts for about 6.25%. Nothing was voided.

## Decision

The protocol allowed one qualification attempt. It is not repeated, and the 3% limit is not relaxed. The chained
session stage refused to start, so **no Dense-vs-PPO measurement exists**. As a result:

- None of items 8–16 of the requested report (Dense/PPO energy, saving, session variability, paired statistics,
  latency, speed-up, EDP) has a measured value.
- No energy figure was produced.
- No surrogate or latency-based estimate is substituted.

ENERGY-CONFIRMED criterion 1 (qualification) fails, so the verdict is NO.

## Why the measurement is unstable (descriptive; used for no decision)

- **The workload was steady.** Median latency varied by only 1% (1.042–1.052 ms) and the CPU 2 clock by about 8%.
- **Package power was not.** It varied from 9.1 to 14.5 W.
- **Both the core and the rest of the package drifted.** Core (PP0) power ranged 6.7–9.7 W and non-core power
  2.4–5.1 W.
- **Temperature moved with it.** The thermal zone ranged from 72 to 92 °C. Run 3, the coolest, gave the lowest
  energy.

This pattern fits power drawn by the rest of the package (uncore, graphics/display, other cores' idle states) and
temperature-dependent power, rather than the inference work itself. A software-only setup on this laptop cannot hold
those constant. Background load stayed within the limits throughout.

## What can and cannot be claimed

- **Remains valid (frozen Phase 7B, unchanged):** the Phase 7B PPO model has 63.4% less parameter memory and 36.0%
  fewer MACs than Dense, at −1.12 pp VAL-SELECT accuracy.
- **Not allowed:**
  - any statement that its measured inference energy or latency is lower;
  - "accuracy-preserving green pruning", GREEN-SUCCESS, "no-loss compression", "environmentally friendly AI" or
    "lower carbon footprint".
- **Allowed statement about energy:** "Direct energy confirmation remains unresolved because the available
  measurement setup did not satisfy the stability requirement."

## Limitations and what would resolve it

- **One laptop, software counter, no external meter.** RAPL package energy on this machine varies by about 20%
  between identical Dense blocks, which is far above the expected model difference.
- **What would resolve it** (not done here, and outside this protocol): an instrument-specific qualification defined
  in advance, on either
  - an external wall-power meter with a logged sampling interface, or
  - a desktop or server CPU with fixed clocks, disabled turbo and stable cooling.
