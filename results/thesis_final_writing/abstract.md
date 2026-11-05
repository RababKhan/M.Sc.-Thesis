# Abstract

Neural-network pruning reduces model size and computation, but deciding how much each layer should lose is
difficult, and weight sparsity alone does not establish deployment benefit.

This thesis develops and evaluates a reinforcement-learning framework for layer-wise adaptive pruning. In it, a
Proximal Policy Optimisation (PPO) agent assigns a pruning ratio to each layer or layer group, guided by layer sensitivity. Successive
experiments with pre-specified decision criteria added:
- a centred sensitivity prior;
- archive-based selection of the policies visited during search;
- an accuracy-constrained reward;
- an equal post-pruning recovery protocol.

A later deployment extension realises pruning structurally and measures CPU latency and energy directly. Experiments
used CIFAR-10 and CIFAR-100 with three small convolutional networks.

**Findings.**
- **PPO against simple allocation.** PPO learned non-uniform allocations that substantially outperformed uniform and
  random allocation, by up to approximately 31.4 percentage points of test accuracy over uniform pruning.
- **LAMP.** The deterministic LAMP method remained more accurate before equal recovery.
- **Sensitivity.** Guidance improved search and exploration measures, not final accuracy.
- **Retention.** PPO found high-ranking policies but rarely ended on them. Archive selection restored their retention
  on validation ranking.
- **Reward.** An accuracy-constrained reward reduced destructive over-pruning, although feasibility certified on
  validation data held on the test set in only about two thirds of runs.
- **Equal recovery.** Equal five-epoch recovery narrowed the PPO–LAMP differences to within a pre-registered ±0.5
  percentage-point decision margin in every setting.
- **Latency.** Parameter reduction proved a weak predictor of measured latency compared with reduction in
  multiply–accumulate operations (MACs).
- **Energy of structured pruning.** Structured pruning reduced directly measured CPU package energy, but with accuracy
  losses beyond the pre-registered limit.
- **Deployment-oriented structured agent.** It reduced parameter memory by 63.4% and MACs by 36.0% at a 1.12
  percentage-point validation-accuracy cost. It missed its accuracy criterion, and its direct energy remained
  unresolved because the measurement setup failed its stability qualification.

**Conclusion.** Adaptive pruning is a chain of connected problems: allocation, search guidance, policy retention,
reward alignment, recovery and hardware validation. An improvement at one stage does not guarantee success at the
next. Each must be evaluated with its own evidence before a compression result can be called a deployment-efficiency
result.

**Keywords:** neural network pruning; reinforcement learning; proximal policy optimisation; layer-wise pruning;
sensitivity-aware pruning; structured pruning; model compression; deployment efficiency
