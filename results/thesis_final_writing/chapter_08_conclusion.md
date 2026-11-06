# Chapter 8 — Conclusion and Future Work

## 8.1 General

This chapter answers the six research questions, states the contributions and the thesis-level conclusion,
summarises the limitations and sets out future work. It introduces no new evidence: every number below was reported
and audited in Chapters 5 and 6.

The thesis set out to learn layer-wise pruning allocations with Proximal Policy Optimisation (PPO), guided by layer
sensitivity, and to find out what such a framework needs in order to produce a better and cheaper model. The research
proceeded as a sequence of experiments, each designed in response to what the previous one revealed. The core
framework, which used masked pruning and answered RQ1–RQ5, comprised: adaptive layer-wise PPO pruning on three small
CNNs and two datasets, against five one-shot baselines at identical sparsity; sensitivity-guided exploration, first
with an absolute and then with a centred prior; a policy-retention analysis against exhaustive ground truth; archive
selection on held-out validation data; a reward-misalignment diagnosis; an accuracy-constrained reward; and equal
fine-tuning of every compared model. The deployment extension, added afterwards and answering RQ6, comprised an
exhaustive CPU latency analysis of structured policies, a direct energy evaluation of the structurally realised
allocation, a deployment-oriented structured PPO that chose physical channel allocations using an accuracy proxy and
a latency-based energy surrogate, and a pre-registered attempt to confirm the final model's energy directly.

## 8.2 Conclusions by Research Question

### RQ1. Does PPO learn non-uniform layer-wise pruning allocations that outperform simple allocation strategies, and how does it compare with a strong deterministic method such as LAMP?

**Yes against simple strategies; no against LAMP.** PPO learned meaningful non-uniform allocations, pruning large,
low-sensitivity layers heavily and protecting small early layers. At the sparsity it chose, it was much more accurate
than uniform pruning in five of six settings, by up to approximately +31.4 percentage points of test accuracy
(CIFAR-10 SimpleCNN), and more accurate than random pruning in all six. The comparison with global magnitude and ERK
was mixed. Before equal recovery, LAMP was significantly more accurate than PPO in five of six settings in the first
comparison and in all six for the later pipelines. The learned allocator is effective, but not universally better than LAMP.

### RQ2. Does layer-sensitivity information improve the pruning search process, and does that improvement translate into higher final predictive accuracy?

**It improves search measures, not the final model.** Across Phases 3–5, sensitivity guidance improved one or more
search or exploration measures: better-ranked policies were sampled earlier, and destructive policies less often.
These improvements did not translate into reliable gains in final predictive accuracy. The centred form provided
stronger evidence for a layer-relative exploration effect, although one accuracy-based exploration measure favoured a
conservative control, and under the corrected reward the prior-guided pipeline was marginally less accurate on test
than the identical pipeline without the prior. Its role is narrower than often assumed, not absent.

### RQ3. Does PPO retain the strongest pruning policies encountered during search, and can archive-based policy selection improve retention?

**PPO does not retain them; an archive restores retention on validation ranking.** The terminal policy reached the
top 10 of 1,296 policies in 0% of runs in five settings and 5% in the sixth, although the best visited policy was
there in 95–100% of runs. Archive-based selection raised top-10 reach to 80–100%, mainly through the archive rule
itself rather than through imitation during training. Its effect on test accuracy was mixed, with large losses on
ResNet-8, where the archive retained exactly the over-pruned policies the reward preferred. Archive selection solves a
retention problem, not a generalisation problem.

### RQ4. How does reward design influence pruning aggressiveness and the accuracy–compression trade-off?

**Reward design materially controls how aggressively PPO prunes.** The original reward trades accuracy and sparsity at
a fixed rate, and its optimum over-prunes fragile networks; this became visible only once search and retention
improved (on CIFAR-10 ResNet-8, test accuracy fell 7.86 percentage points below dense). Under an otherwise identical
pipeline, the accuracy-constrained reward cut the mean loss on CIFAR-10 ResNet-8 from 7.32 to 1.18 percentage points,
at more moderate sparsity. But it was slightly worse on SimpleCNN, LAMP remained more accurate, and its
validation-certified feasibility held on the test set in only 67.5% of runs. A validation constraint controls the
search; it does not guarantee behaviour on unseen data.

### RQ5. How does equal post-pruning fine-tuning affect comparisons between PPO and deterministic pruning methods?

**Equal five-epoch, fixed-mask recovery substantially narrowed the differences between methods.** The PPO–LAMP
difference fell from 0.26–0.53 to 0.00–0.22 percentage points, within the pre-registered ±0.5 percentage-point
confidence-interval decision margin in all six settings. This margin is a decision rule, not a formal statistical
equivalence test, and LAMP remained significantly, though slightly, ahead in three settings. PPO's advantage over
global magnitude, large before recovery, fell to at most 0.12 percentage points. Methods should therefore be compared under
equal recovery budgets.

### RQ6. Do compression gains translate into real deployment benefits in memory, computation, latency and measured operational energy?

**Not automatically.**
- **Latency proxies.** Across every structured policy measured, parameter reduction was a weak proxy for CPU latency
  (Spearman 0.31–0.55), and MAC reduction a much stronger one (0.89–0.98). Measured latency remained necessary.
- **ResNet-8 headroom.** Under the accuracy constraint, the best proxy-feasible candidate reached a 1.098× speed-up
  and the best confirmed five-epoch-feasible calibration policy reached 1.089×. Neither met the pre-registered 1.10×
  GO rule, so no hardware-aware search was started.
- **Phase 7A, measured energy with an accuracy cost.** The structured realisation of the learned allocation reduced
  directly measured CPU package energy by 22.8% (SimpleCNN) and 13.4% (ResNet-8), consistently across sessions, but
  lost 5.08 and 3.24 percentage points of VAL-RL accuracy; on SimpleCNN, MAC-matched uniform pruning was both more
  accurate and lower in energy.
- **Phase 7B, a better trade-off without an energy result.** The deployment-oriented structured PPO reduced parameter
  memory by 63.4% and MACs by 36.0%, at a validation-accuracy cost of 1.12 percentage points. No seed (0 of 20) met
  the joint deployment-efficiency criterion, its latency was not measured, and its energy remained unresolved because
  the measurement setup failed its stability qualification.

Compression can produce real deployment savings, but each dimension must be measured separately, with its accuracy
cost.

## 8.3 Main Contributions

The six thesis contributions (TC1–TC6) are stated in Table 8.1 in final form, each with its boundary.

**Table 8.1. Thesis contributions and their boundaries.**

| # | Contribution | Boundary |
|---|---|---|
| TC1 | **Adaptive PPO-based layer-wise pruning**, evaluated against Uniform, Random, global magnitude, ERK and LAMP at identical sparsity on two datasets and three architectures | Strongly better than uniform and random allocation; not superior to LAMP |
| TC2 | **An analysis of sensitivity-guided exploration**, with absolute and centred priors against shuffled, constant, magnitude-matched and sign-reversed controls | Improves the search; no consistent final-accuracy gain |
| TC3 | **A diagnosis of policy retention in RL pruning**, against exhaustive ground truth, and evaluation of archive-based selection | Validation-retention benefit; mixed effect on test accuracy |
| TC4 | **A diagnosis of reward misalignment**, and evaluation of an accuracy-constrained, validation-aware reward | Better control of over-pruning; test feasibility not guaranteed |
| TC5 | **An equal-recovery methodology:** identical fixed-mask fine-tuning, shared seeds and data, at identical sparsity | Small post-recovery differences do not constitute formal equivalence |
| TC6 | **A deployment-efficiency analysis** that separates parameter compression, MAC reduction, measured latency and directly measured operational energy, including a qualified energy-measurement protocol and its reported failure | Phase-7A energy savings carry a material accuracy cost; Phase-7B energy remains unresolved |

**Together**, they examine adaptive pruning at each stage from allocation to hardware validation, with decision
criteria fixed in advance across the later experimental stages.

## 8.4 Overall Thesis Conclusion

**Adaptive neural-network pruning is not only a problem of selecting which weights or layers to remove.** It is a chain
of connected decisions: how pruning is allocated across layers, how the search is guided, whether good policies are
retained, whether the reward matches the true objective, how post-pruning recovery is controlled, and whether
compression translates into hardware benefit.

**Improvement at one stage does not guarantee success at the next.** Better search exposed a poor reward. Better
validation selection transferred to the test set only partly. Lower parameter count did not mean lower latency.
**Lower energy came with accuracy loss exceeding the pre-registered limit**, and a simpler allocation did better. And
the structured model with the best trade-off reduced memory and computation substantially, but its energy could not
be measured reliably.

**Implications for practice.** Compare methods only after an equal recovery budget; report the best visited policy
separately from the returned one; write the acceptable accuracy loss into the objective; confirm any constraint on
held-out data; and treat memory, computation, latency and energy as separate claims, each measured and reported with
its accuracy cost.

**This multi-stage view is the central conclusion of the thesis.** A pruning result becomes a deployment-efficiency
result only when every link of the chain has its own evidence and accuracy is reported alongside each efficiency gain.
The negative results are part of that conclusion, not exceptions to it: each identified the next link to control.

## 8.5 Limitations

The limitations (Section 7.13) are summarised by group.
- **A. Scale.** All experiments used CIFAR-10 and CIFAR-100, with no ImageNet-scale validation. Each pruning space was
  small enough to enumerate, which enabled exact ground truth but makes the search easier than at realistic scale.
- **B. Models.** Three small CNNs (SimpleCNN, a CIFAR adaptation of LeNet-5 and ResNet-8); no larger CNN, transformer
  or language model. The deployment extension used a subset of these settings, and its final agent SimpleCNN only.
- **C. Optimisation.** PPO's search is far more expensive than a deterministic method; its results depend on reward
  design; LAMP remained stronger before recovery; and sensitivity guidance did not improve final accuracy.
- **D. Evaluation.** Validation-certified constraints transferred imperfectly to the test set; feasibility was
  sensitive to the fixed threshold and the size of the validation splits; the later recovery and deployment
  experiments used a single recovery seed per model; and recovery used one fixed five-epoch protocol.
- **E. Hardware.** All latency and energy results come from one laptop CPU, at batch 1 and one thread, with no GPU,
  mobile, embedded or accelerator evaluation.
- **F. Energy.** Only CPU package energy was measured, with no memory or whole-system energy. The Phase-7B
  measurements failed their stability qualification, so there is no direct energy result for the final structured
  model.
- **G. Sustainability.** No carbon accounting, lifecycle analysis or assessment of training and search energy was
  carried out. The thesis is a sustainability-oriented evaluation of operational inference energy, not an
  environmental assessment.

## 8.6 Future Work

Each direction below follows from a specific finding or limitation of the thesis.

### Short-term

1. **Native structured RL pruning from the beginning.** Translating a masked policy into a structured network after
   the search was costly in accuracy (Phase 7A), and learning in the structured space narrowed that cost (Phase 7B).
   Future frameworks should search in the deployed structure from the start, including residual-coupled channel
   groups.
2. **Hybrid LAMP and RL.** Deterministic layer-adaptive scores could initialise, constrain or bias the RL search,
   combining LAMP's cheap reliability with the agent's ability to pursue objectives that one score cannot express.
3. **Better feasibility proxies.** The validation-to-test gap and the modest precision of the Phase-7B accuracy proxy
   call for calibrated predictors of post-recovery accuracy and for selection margins rather than exact thresholds.
4. **Stable external energy instrumentation.** The laptop's software energy counter failed its stability
   qualification. Future energy studies should use a logging wall-power meter, controlled thermal conditions and fixed
   processor clocks where possible, and qualify the instrument before any model comparison.

### Medium-term

5. **Multi-objective optimisation.** Explicit Pareto-front search over accuracy, memory, MACs, latency and measured
   energy, compared with deterministic baselines on the same fronts, would show trade-offs that a scalar reward hides.
6. **Improved generalisation constraints.** Multiple validation subsets, uncertainty-aware thresholds and more robust
   constraint formulations would address the finding that validation feasibility did not guarantee test feasibility.
7. **Larger architectures.** ResNet-18/34/50, MobileNet- and EfficientNet-class networks, or other CNNs whose pruning
   spaces cannot be enumerated, where a learned search may have more to offer.
8. **Larger datasets.** ImageNet-scale or comparable data, to test whether the retention, reward and recovery findings
   hold when each evaluation is much more costly.
9. **Edge and mobile deployment.** Mobile CPUs, ARM processors, embedded GPUs and accelerators, each with its own
   latency and energy behaviour.
10. **GPU evaluation.** Whether the relationship between parameter reduction, MAC reduction and latency found on one
    CPU holds, weakens or reverses on GPUs.

### Long-term

11. **Hardware-in-the-loop multi-objective RL.** Rewards using directly measured latency and energy rather than
    surrogates, attempted only after stable energy instrumentation (item (iv)) is available and with explicit accuracy
    constraints.
12. **Reliable lifecycle and carbon analysis.** Carbon-intensity conversion and lifecycle accounting become meaningful
    only once trustworthy direct energy measurements exist. Until then, no carbon impact should be claimed; this
    thesis makes no such claim.

## 8.7 Final Remarks

The thesis does not establish PPO as a universally superior pruning method. It shows that adaptive pruning must be
evaluated as a complete pipeline, from search through recovery to deployment: RL pruning is useful as a flexible
allocator when its objective is explicitly aligned with the desired accuracy boundary, simpler deterministic methods
remain stronger for ordinary compression objectives, and hardware-level validation must precede efficiency claims. The value of the work lies as much in the links it found to be weak as
in those it showed to be sound.
