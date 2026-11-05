# Sustainable-computing positioning

## Principle

The thesis does not claim that its method is "green", "environmentally friendly" or "carbon-reducing". It measures
deployment efficiency on three separate levels. A result at one level is never presented as a result at a higher
level.

| Level | What it means | Metric | Measured how |
|---|---|---|---|
| **A. Compression** | Less memory needed to store and load the model | Parameter bytes (Σ numel × element size), serialised size | Exact, from the model |
| **B. Computational efficiency** | Less arithmetic and less time per inference | MACs (closed form, checked against hooks); CPU latency (batch 1, 1 thread, pinned core) | MACs exact; latency measured |
| **C. Operational energy** | Less energy drawn by the processor per inference | Intel RAPL package energy, J per 1,000 inferences, paired with a dense anchor in the same session; EDP | Direct hardware counter (Windows Energy Meter Interface); no TDP or software estimate |

Each level is necessary for the next, but not sufficient:
- Paper 3 shows that A does not imply B. Parameter reduction correlates with latency gain at only ρ 0.31–0.55, and
  31–39% of policy pairs are sparser but slower.
- MACs track latency closely (ρ 0.89–0.98), so B is better grounded in MACs plus measured latency.
- C needs its own direct measurement and a stable instrument.

## Evidence at each level

| | Level A: compression | Level B: computation | Level C: operational energy | Accuracy (criterion ±0.5 pp) |
|---|---|---|---|---|
| **Paper 3** (structured latency tables) | — | Latency measured for 1,296 shapes × 4 settings; MACs strong proxy | — | ResNet-8 accuracy-feasible headroom ≤ 1.098× |
| **Phase 7A** (structured transfer) | SimpleCNN −80.1%; ResNet-8 −25.0% (parameters) | MACs −42.6% / −21.5%; latency speed-up 1.28× / 1.09× | **Measured:** −22.8% / −13.4%, 5/5 sessions; EDP 0.603 / 0.791 | **Failed:** −5.08 / −3.24 pp (VAL-RL). Uniform dominated on SimpleCNN. |
| **Phase 7B** (energy-aware structured PPO, SimpleCNN) | **−63.4%** parameter memory | **MACs −36.0%**; latency **not measured** | **Unresolved:** pilot CV 4.01%; confirmation CV 19.92% (limit 3%) | **Failed:** −1.12 pp (VAL-SELECT), but much better than 7A's −4.00 pp (both VAL-SELECT) |

## Positioning statements (for Chapter 6.6 and the conclusion)

1. **Structured realisation is what makes compression reach the hardware.** Masked (unstructured) sparsity alone
   gives no reliable CPU benefit. Physically removing channels and neurons reduced MACs, latency and measured package
   energy in Phase 7A.
2. **An energy saving bought with material accuracy loss is not a sustainability result.** In Phase 7A the energy
   came with a 5.1 / 3.2 pp accuracy cost. On SimpleCNN, plain MAC-matched uniform pruning was both more accurate
   and more energy-efficient than the PPO allocation.
3. **Designing the policy for the deployed structure narrows the accuracy cost.** Phase 7B's structured,
   energy-aware PPO kept 63.4% memory and 36.0% MAC reductions while cutting the accuracy loss from 4.00 to 1.12 pp
   (both VAL-SELECT). That still fails the strict 0.5 pp criterion.
4. **Measurement reliability is a first-class requirement.** The same software energy counter was stable in Phase 7A
   (CV 2.30% / 1.74%) and unstable in Phase 7B and 7B-EC (CV 4.01%, 19.92%). During qualification, latency held
   constant while package power drifted 9.1–14.5 W with temperature. Without a qualified instrument, no energy claim
   is made.

## Recommended conclusion paragraph

> The experiments demonstrate that structured pruning can reduce operational CPU energy: a physically structured
> realisation of the learned layer-wise allocation lowered measured package energy per inference by 22.8% and 13.4%.
> They also show that energy savings alone are insufficient when they are accompanied by material accuracy loss, and
> that a simple uniform allocation can be the better deployment choice. The redesigned energy-aware structured PPO
> substantially improved the memory–computation–accuracy trade-off (−63.4% parameter memory, −36.0% MACs, −1.12 pp
> validation accuracy), but it missed the strict accuracy criterion, and a stable direct energy measurement was not
> available to confirm its operational energy.

## Terms

- **Use:** "deployment efficiency", "computational efficiency", "measured CPU package energy", "operational
  inference energy (measured)", "energy-aware objective".
- **Do not use:** "green AI", "environmentally friendly", "sustainable model", "carbon footprint", "CO₂ savings",
  "energy-efficient" for Phase 7B.
- Repository names containing "green" are identifiers only.

## Out of scope (state explicitly)

- Lifecycle and embodied carbon, and training energy.
- Whole-system or wall-plug power, and DRAM energy (it read 0 on this platform).
- Data-centre, GPU or edge-device energy.
- Carbon-intensity conversion. Phase 7A kWh figures are a labelled projection only.
