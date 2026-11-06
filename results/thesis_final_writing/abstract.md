# Abstract

Neural-network pruning reduces model size and computation, but deciding how much each layer should lose is
difficult, and weight sparsity alone does not establish a deployment benefit. This thesis develops and evaluates a
sensitivity-aware reinforcement-learning framework for the layer-wise pruning of convolutional neural networks. The
study covers the CIFAR-10 and CIFAR-100 datasets, three small convolutional networks and one laptop CPU.

A Proximal Policy Optimisation (PPO) agent assigns a pruning ratio to each layer or layer group, guided by layer
sensitivity. Successive experiments with pre-specified decision criteria added a centred sensitivity prior,
archive-based selection of the policies visited during search, an accuracy-constrained reward and an equal
post-pruning recovery protocol. A later deployment extension realised the pruning structurally, measured CPU latency
and package energy directly, and trained a deployment-oriented structured agent.

The learned allocations substantially outperformed uniform and random allocation, by up to approximately 31.4
percentage points of test accuracy over uniform pruning. The deterministic layer-adaptive magnitude-based pruning
(LAMP) method nevertheless remained more accurate before equal recovery. Sensitivity guidance improved search and
exploration measures, not final accuracy. The agent found high-ranking policies but rarely ended on them, and archive
selection restored their retention on validation ranking. The accuracy-constrained reward reduced destructive
over-pruning, although feasibility certified on validation data held on the test set in only about two thirds of
runs. Equal five-epoch recovery narrowed the differences between PPO and LAMP to within a pre-registered ±0.5
percentage-point decision margin in every setting. In the deployment analysis, parameter reduction proved a weak
predictor of measured latency compared with the reduction in multiply–accumulate operations. Structured pruning
reduced directly measured CPU package energy, but with accuracy losses beyond the pre-registered limit. The
deployment-oriented structured agent reduced parameter memory by 63.4% and multiply–accumulate operations by 36.0% at
a 1.12 percentage-point validation-accuracy cost. It missed its accuracy criterion, and its direct energy remained
unresolved because the measurement setup failed its stability qualification.

The novelty of the thesis lies not in a new pruning criterion but in treating adaptive pruning as a chain of connected
problems (allocation, search guidance, policy retention, reward alignment, recovery and hardware validation) and in
examining each with its own experiment and evidence. The results show that an improvement at one stage does not
guarantee success at the next. Each stage must therefore be validated before a compression result can be called a
deployment-efficiency result.

**Keywords:** neural network pruning; reinforcement learning; proximal policy optimisation; layer-wise pruning;
sensitivity-aware pruning; structured pruning; model compression; deployment efficiency
