# Chapter 8 — Conclusion and Future Work

## 8.1 General

This chapter answers the six research questions, states the contributions and the conclusion, and sets out the
limitations and future work. Every number below was reported and audited in Chapters 5 and 6.

The thesis set out to learn layer-wise pruning allocations with Proximal Policy Optimisation (PPO), guided by layer
sensitivity, and to find out what such a framework needs in order to produce a better and cheaper model. Each
experiment was designed in response to what the previous one revealed: the masked-pruning core framework (RQ1–RQ5)
added a centred sensitivity prior, archive selection, an accuracy-constrained reward and equal fine-tuning; the
deployment extension (RQ6) added exhaustive CPU latency analysis, direct energy measurement, a deployment-oriented
structured PPO and a pre-registered energy-confirmation attempt.

## 8.2 Conclusions by Research Question

### RQ1. Does PPO learn non-uniform layer-wise pruning allocations that outperform simple allocation strategies, and how does it compare with a strong deterministic method such as LAMP?

**Yes against simple strategies; no against LAMP.** PPO learned meaningful non-uniform allocations that were much
more accurate than uniform pruning in five of six settings, by up to approximately +31.4 percentage points of test
accuracy, and than random pruning in all six; against global magnitude and ERK the comparison was mixed. Before equal
recovery, LAMP was significantly more accurate than PPO in five of six settings in the first comparison and in all six
for the later pipelines. The learned allocator is effective, but not universally better than LAMP.

### RQ2. Does layer-sensitivity information improve the pruning search process, and does that improvement translate into higher final predictive accuracy?

**It improves search measures, not the final model.** Across Phases 3–5, sensitivity guidance improved one or more
search or exploration measures (better-ranked policies earlier, destructive policies less often), but these
improvements did not translate into reliable gains in final predictive accuracy; under the corrected reward the
prior-guided pipeline was marginally less accurate on test than the same pipeline without the prior. Its role is
narrower than often assumed, not absent.

### RQ3. Does PPO retain the strongest pruning policies encountered during search, and can archive-based policy selection improve retention?

**PPO does not retain them; an archive restores retention on validation ranking.** The terminal policy reached the
top 10 of 1,296 policies in 0% of runs in five settings and 5% in the sixth, although the best visited policy was
there in 95–100% of runs. Archive-based selection raised top-10 reach to 80–100%, mainly through the archive rule
itself rather than through imitation during training. Its effect on test accuracy was mixed, with large losses on
ResNet-8, where the archive retained exactly the over-pruned policies the reward preferred. Archive selection solves a
retention problem, not a generalisation problem.

### RQ4. How does reward design influence pruning aggressiveness and the accuracy–compression trade-off?

**Reward design materially controls how aggressively PPO prunes.** The original fixed-rate reward has an optimum that
over-prunes fragile networks, visible only once search and retention improved. Under an otherwise identical pipeline,
the accuracy-constrained reward cut the mean loss on CIFAR-10 ResNet-8 from 7.32 to 1.18 percentage points, but it was
slightly worse on SimpleCNN, LAMP remained more accurate, and validation-certified feasibility held on the test set in
only 67.5% of runs. A validation constraint controls the search; it does not guarantee behaviour on unseen data.

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

Compression can produce real deployment savings, but each dimension must be measured with its accuracy cost.

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

**Together**, they examine adaptive pruning at each stage from allocation to hardware validation.

## 8.4 Overall Thesis Conclusion

**Adaptive neural-network pruning is not only a problem of selecting which weights or layers to remove.** It is a chain
of connected decisions: how pruning is allocated across layers, how the search is guided, whether good policies are
retained, whether the reward matches the true objective, how post-pruning recovery is controlled, and whether
compression translates into hardware benefit.

**Improvement at one stage does not guarantee success at the next.** Better search exposed a poor reward; better
validation selection transferred to the test set only partly; lower parameter count did not mean lower latency;
**lower energy came with accuracy loss exceeding the pre-registered limit**, and a simpler allocation did better; and
the structured model with the best trade-off could not have its energy measured reliably.

**Implications for practice.** Compare methods only after an equal recovery budget; report the best visited policy
separately from the returned one; write the acceptable accuracy loss into the objective; confirm any constraint on
held-out data; and treat memory, computation, latency and energy as separate claims, each measured and reported with
its accuracy cost.

**This multi-stage view is the central conclusion of the thesis.** A pruning result becomes a deployment-efficiency
result only when every link has its own evidence and accuracy is reported alongside each efficiency gain; the negative
results are part of that conclusion, each identifying the next link to control.

## 8.5 Limitations

The limitations (Section 7.13) are summarised by group.
- **A. Scale.** CIFAR-10 and CIFAR-100 only; each pruning space was small enough to enumerate, which enabled exact
  ground truth but makes the search easier than at realistic scale.
- **B. Models.** Three small CNNs; no larger CNN, transformer or language model. The deployment extension used a
  subset of these settings, and its final agent SimpleCNN only.
- **C. Optimisation.** PPO's search is far more expensive than a deterministic method; its results depend on reward
  design; LAMP remained stronger before recovery; and sensitivity guidance did not improve final accuracy.
- **D. Evaluation.** Validation-certified constraints transferred imperfectly to the test set and were sensitive to
  the fixed threshold and split size; deployment experiments used one recovery seed per model; and recovery used one
  fixed five-epoch protocol.
- **E. Hardware.** All latency and energy results come from one laptop CPU, at batch 1 and one thread, with no GPU,
  mobile, embedded or accelerator evaluation.
- **F. Energy.** Only CPU package energy was measured, with no memory or whole-system energy. The Phase-7B
  measurements failed their stability qualification, so there is no direct energy result for the final structured
  model.
- **G. Sustainability.** No carbon accounting, lifecycle analysis or assessment of training and search energy was
  carried out. The thesis is a sustainability-oriented evaluation of operational inference energy, not an
  environmental assessment.

## 8.6 Future Work

Each direction below follows from a finding or limitation of the thesis.

### Short-term

1. **Native structured RL pruning from the beginning.** Translating a masked policy into a structure was costly in
   accuracy (Phase 7A) and learning in the structured space narrowed that cost (Phase 7B), so future frameworks should
   search in the deployed structure from the start, including residual-coupled channel groups.
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
   constraint formulations, because validation feasibility did not guarantee test feasibility.
7. **Larger architectures.** ResNet-18/34/50, MobileNet- and EfficientNet-class networks, or other CNNs whose pruning
   spaces cannot be enumerated, where a learned search may have more to offer.
8. **Larger datasets.** ImageNet-scale data, to test whether the retention, reward and recovery findings hold when
   each evaluation is much more costly.
9. **Edge and mobile deployment.** Mobile CPUs, ARM processors, embedded GPUs and accelerators.
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
evaluated as a complete pipeline: RL pruning is useful as a flexible allocator when its objective is explicitly aligned
with the desired accuracy boundary, simpler deterministic methods remain stronger for ordinary compression objectives,
and hardware-level validation must precede efficiency claims. The value of the work lies as much in the links it found
to be weak as in those it showed to be sound.
