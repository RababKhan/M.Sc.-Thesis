# Chapter 8 — Conclusion and Future Work

## 8.1 Chapter Overview and Summary of the Research

This chapter closes the thesis. It answers the six research questions, states the contributions in their final form
and the thesis-level conclusion, summarises the limitations, and sets out future work that follows from them. It
introduces no new evidence: every number below was reported and audited in Chapters 5 and 6.

The thesis set out to learn layer-wise pruning allocations with Proximal Policy Optimisation (PPO), guided by layer
sensitivity, and to find out what such a framework needs in order to produce a better and cheaper model. The research
proceeded as a sequence of experiments. Each experiment was designed in response to what the previous one revealed:

1. **Adaptive layer-wise PPO pruning.** A PPO agent assigned one of six pruning ratios to each of four units, on three
   small CNNs and two datasets. It was compared with five one-shot baselines at identical sparsity.
2. **Sensitivity-guided exploration.** Layer sensitivity was given to the agent as a soft prior on its actions, first
   in absolute and then in centred form, and tested against uninformative controls.
3. **Policy-retention analysis.** Exhaustive evaluation of every policy showed that the agent found near-optimal
   policies but did not end on them.
4. **Archive selection.** Visited policies were archived and the best was selected on held-out validation data.
5. **Reward-misalignment diagnosis.** With search and retention improved, the original accuracy–sparsity reward drove
   fragile networks to over-prune.
6. **Accuracy-constrained reward.** A constraint-first reward, with selection on held-out data, replaced the fixed
   trade-off.
7. **Equal fine-tuning.** Every compared model received the same fixed-mask recovery, at identical sparsity, with a
   shared data stream.
8. **Hardware latency analysis.** Every structured policy of two networks and two output sizes was timed on a CPU.
9. **Structured energy evaluation.** The learned allocation was realised as a physically smaller network, and its CPU
   package energy was measured directly.
10. **Deployment-oriented structured PPO.** The agent was redesigned to choose physical channel allocations directly,
    with an accuracy proxy and a latency-based energy surrogate.
11. **Direct energy-confirmation attempt.** A separate, pre-registered measurement study tried to confirm the energy
    of the final structured model.

Steps 1–7 form the core framework, which used masked pruning and answered RQ1–RQ5. Steps 8–11 form the deployment
extension, which was added after the core framework had been evaluated and answered RQ6.

## 8.2 Conclusions by Research Question

### RQ1. Does PPO learn non-uniform layer-wise pruning allocations that outperform simple allocation strategies, and how does it compare with a strong deterministic method such as LAMP?

**Yes against simple strategies; no against LAMP.**
- **The allocations.** PPO learned meaningful non-uniform allocations. It pruned large, low-sensitivity layers heavily
  and protected small early layers.
- **Against simple strategies, at the sparsity the agent chose:**
  - much more accurate than uniform pruning in five of six settings, by up to approximately +31.4 percentage points of
    test accuracy (CIFAR-10 SimpleCNN);
  - more accurate than random pruning in all six.
- **Against magnitude criteria:** the comparison with global magnitude and ERK was mixed.
- **Against LAMP:** before equal recovery, LAMP was significantly more accurate than PPO:
  - in five of six settings in the first comparison;
  - in all six for the later pipelines.

The learned allocator is therefore effective, but it is not universally better than a strong deterministic
layer-adaptive method.

### RQ2. Does layer-sensitivity information improve the pruning search process, and does that improvement translate into higher final predictive accuracy?

**It improves the search, not the final model.**
- **Search.** In every phase where it was tested, sensitivity guidance improved the search: better-ranked policies
  were sampled earlier and destructive policies less often.
- **Centred form.** Centring was designed to distinguish layer-specific guidance from a generic "prune less"
  pressure. Its controls provided stronger evidence for a layer-relative exploration effect, although one
  accuracy-based exploration measure favoured a conservative control.
- **Final accuracy.** No phase detected a reliable improvement. Under the corrected reward, the prior-guided pipeline
  was marginally less accurate on test than the identical pipeline without the prior.

The main measurable contribution of sensitivity was to exploration rather than to final accuracy. That is a narrower
role than is often assumed, not an absence of value.

### RQ3. Does PPO retain the strongest pruning policies encountered during search, and can archive-based policy selection improve retention?

**PPO does not retain them; an archive restores retention on validation ranking.**
- **What PPO did.** It repeatedly visited near-optimal policies but did not return them as its terminal policy. The
  terminal policy reached the top 10 of 1,296 policies in 0% of runs in five settings and 5% in the sixth, although
  the best visited policy was there in 95–100% of runs.
- **What the archive did.** Archive-based selection raised top-10 reach to 80–100% and top-20 reach to 100%, mainly
  through the archive rule itself rather than through imitation during training.
- **Effect on test accuracy:** mixed, with large losses on ResNet-8, where the archive retained exactly the
  over-pruned policies the reward preferred.

Archive selection solves a retention problem. It does not solve, or guarantee, a generalisation problem.

### RQ4. How does reward design influence pruning aggressiveness and the accuracy–compression trade-off?

**Reward design materially controls how aggressively PPO prunes.**
- **The fixed-rate reward.** The original accuracy–sparsity reward trades the two at a fixed rate, and its optimum
  over-prunes fragile networks. This became visible only once search and retention improved: on CIFAR-10 ResNet-8,
  test accuracy fell 7.86 percentage points below dense.
- **The accuracy-constrained reward.** Comparing it with the original reward under an otherwise identical pipeline,
  the mean loss on CIFAR-10 ResNet-8 fell from 7.32 to 1.18 percentage points, at more moderate sparsity.
- **Its limits:**
  - it was slightly worse on SimpleCNN;
  - LAMP remained more accurate;
  - its validation-certified feasibility held on the test set in only 67.5% of runs.

Aligning the optimisation objective and ensuring generalisation are separate issues: a validation constraint controls
the search, but it does not guarantee behaviour on unseen data.

### RQ5. How does equal post-pruning fine-tuning affect comparisons between PPO and deterministic pruning methods?

**Equal five-epoch, fixed-mask recovery substantially narrowed the differences between methods.**
- **PPO against LAMP.** The difference fell from 0.26–0.53 to 0.00–0.22 percentage points. In all six settings it
  fell within the pre-registered ±0.5 percentage-point confidence-interval decision margin.
- **Not equivalence.** This margin is a decision rule, not a formal statistical equivalence test, and LAMP remained
  significantly, though slightly, ahead in three settings.
- **PPO against global magnitude.** PPO's advantage, large before recovery, fell to at most 0.12 percentage points.

Much of the visible difference between pruning methods lay before fine-tuning. The main implication is that pruning
methods should be compared under equal post-pruning recovery budgets.

### RQ6. Do compression gains translate into real deployment benefits in memory, computation, latency and measured operational energy?

**Not automatically.**
- **Latency proxies.** Across every structured policy measured, parameter reduction was a weak proxy for CPU latency
  (Spearman 0.31–0.55), and MAC reduction a much stronger one (0.89–0.98). Measured latency remained necessary.
- **ResNet-8 headroom.** Under the accuracy constraint, the best proxy-feasible candidate reached a 1.098× speed-up
  and the best confirmed five-epoch-feasible calibration policy reached 1.089×. Neither met the pre-registered 1.10×
  GO rule, so no hardware-aware search was started.
- **Phase 7A, measured energy with an accuracy cost.** The structured realisation of the learned allocation reduced
  directly measured CPU package energy by 22.8% (SimpleCNN) and 13.4% (ResNet-8), consistently across sessions.
  - It lost 5.08 and 3.24 percentage points of validation accuracy.
  - On SimpleCNN, MAC-matched uniform pruning was both more accurate and lower in energy than PPO.
- **Phase 7B, a better trade-off without an energy result.** The deployment-oriented structured PPO reduced parameter
  memory by 63.4% and MACs by 36.0%, at a validation-accuracy cost of 1.12 percentage points.
  - No seed (0 of 20) met the joint deployment-efficiency criterion.
  - Its latency was not measured.
  - Its energy remained unresolved, because the measurement setup failed its stability qualification.

Compression can produce real deployment savings. But every deployment dimension must be measured separately, and each
must be interpreted together with its accuracy cost.

## 8.3 Main Contributions

The six contributions are stated below in final form, each with its boundary.

| # | Contribution | Boundary |
|---|---|---|
| C1 | **Adaptive PPO-based layer-wise pruning**, evaluated against Uniform, Random, global magnitude, ERK and LAMP at identical sparsity on two datasets and three architectures | Strongly better than uniform and random allocation; not superior to LAMP |
| C2 | **An analysis of sensitivity-guided exploration**, with absolute and centred priors against shuffled, constant, magnitude-matched and sign-reversed controls | Improves the search; no consistent final-accuracy gain |
| C3 | **A diagnosis of policy retention in RL pruning**, against exhaustive ground truth, and evaluation of archive-based selection | Validation-retention benefit; mixed effect on test accuracy |
| C4 | **A diagnosis of reward misalignment**, and evaluation of an accuracy-constrained, validation-aware reward | Better control of over-pruning; test feasibility not guaranteed |
| C5 | **An equal-recovery methodology:** identical fixed-mask fine-tuning, shared seeds and data, at identical sparsity | Small post-recovery differences do not constitute formal equivalence |
| C6 | **A deployment-efficiency analysis** that separates parameter compression, MAC reduction, measured latency and directly measured operational energy, including a qualified energy-measurement protocol and its reported failure | Phase-7A energy savings carry a material accuracy cost; Phase-7B energy remains unresolved |

**Together**, these contributions make the thesis's methodological point. Adaptive pruning is examined at each stage
from allocation to hardware validation, and decision criteria were fixed in advance across its later experimental
stages.

## 8.4 Overall Thesis Conclusion

**Adaptive neural-network pruning is not only a problem of selecting which weights or layers to remove.** The thesis
shows it to be a chain of connected decisions:
- how pruning is allocated across layers;
- how the search for that allocation is guided;
- whether the good policies it finds are retained;
- whether the reward matches the true objective;
- how post-pruning recovery is controlled;
- whether compression translates into hardware benefit.

**Improvement at one stage does not guarantee success at the next.** The experiments showed this repeatedly:
- **Better search exposed a poor reward.** When the agent searched and retained well, it found the over-pruned models
  that the original reward preferred.
- **Better validation selection did not transfer to test.** Archive selection and the accuracy constraint worked on
  validation data, and transferred to the test set only partly.
- **Lower parameter count did not mean lower latency.** Removing parameters from low-compute layers shrank models
  without making them proportionally faster.
- **Lower energy came with unacceptable accuracy loss.** Directly measured energy fell, but the models that saved it
  lost more accuracy than the criterion allowed, and a simpler allocation did better.
- **Strong compression remained unconfirmed as energy saving.** The structured model with the best trade-off reduced
  memory and computation substantially, but its energy could not be measured reliably.

**Implications for practice.** For anyone evaluating or deploying adaptive pruning, the results translate into five
habits:
- compare methods only after an equal recovery budget;
- report the best visited policy separately from the returned one;
- write the acceptable accuracy loss into the objective, rather than trading it at a fixed rate;
- confirm any constraint on held-out data;
- treat memory, computation, latency and energy as separate claims, each requiring its own measurement and each
  reported with its accuracy cost.

**This multi-stage view is the central conclusion of the thesis.** A pruning result becomes a deployment-efficiency
result only when every link of the chain has its own evidence and when accuracy is reported alongside each efficiency
gain. The negative results are part of that conclusion, not exceptions to it. Each one identified the link that the
next experiment had to control.

## 8.5 Limitations

The limitations, discussed fully in Section 7.13, are summarised here by group.

- **A. Scale.**
  - All experiments used CIFAR-10 and CIFAR-100.
  - There is no ImageNet-scale validation.
  - Each network's pruning space was small enough to enumerate. That enabled exact ground truth but makes the search
    problem easier than at realistic scale.
- **B. Models.** Three small CNNs: SimpleCNN, a CIFAR adaptation of LeNet-5, and ResNet-8. No larger CNN, transformer
  or language model was studied. The deployment extension used a subset of these settings, and its final agent used
  SimpleCNN only.
- **C. Optimisation:**
  - PPO's search is far more expensive than a deterministic method;
  - its results depend on reward design;
  - LAMP remained stronger before recovery;
  - sensitivity guidance did not improve final accuracy.
- **D. Evaluation:**
  - validation-certified constraints transferred imperfectly to the test set;
  - feasibility was sensitive to the fixed threshold and to the size of the validation splits;
  - the later recovery and deployment experiments used a single recovery seed per model;
  - recovery used one fixed five-epoch protocol.
- **E. Hardware.** All latency and energy results come from one laptop CPU, at batch 1 and one thread. There was no
  GPU, mobile, embedded or accelerator evaluation.
- **F. Energy.**
  - Only CPU package energy was measured, with no memory or whole-system energy.
  - The Phase-7B measurements failed their stability qualification, so there is no direct energy result for the final
    structured model.
- **G. Sustainability.** No carbon accounting, lifecycle analysis or assessment of training and search energy was
  carried out. The thesis is a sustainability-oriented evaluation of operational inference energy, not an
  environmental assessment.

## 8.6 Future Work

Each direction below follows from a specific finding or limitation of the thesis. The directions are grouped by how
directly they build on the present work.

### Short-term

1. **Native structured RL pruning from the beginning.** Phase 7A showed that translating a masked policy into a
   structured network after the search was costly in accuracy. Learning directly in the structured space narrowed that
   cost substantially (Phase 7B). Future frameworks should search in the deployed structure from the start, including
   residual-coupled channel groups for residual networks.
2. **Hybrid LAMP and RL.** LAMP was cheap and strong throughout. Using deterministic layer-adaptive scores to
   initialise, constrain or bias the RL search may combine LAMP's reliability with the agent's ability to pursue
   objectives that one score cannot express.
3. **Better feasibility proxies.** Both the validation-to-test gap and the modest precision of the Phase-7B accuracy
   proxy limited how reliably feasible policies could be selected. Calibrated predictors of post-recovery accuracy,
   and selection margins rather than exact thresholds, are direct next steps.
4. **Stable external energy instrumentation.** The software energy counter on the available laptop failed its
   stability qualification in the later experiments. Future energy studies should use:
   - a logging wall-power meter;
   - controlled thermal conditions;
   - fixed processor clocks where possible.

   Each campaign should qualify its instrument before any model comparison.

### Medium-term

5. **Multi-objective optimisation.** The thesis combined objectives through constraints and a scalar reward. Explicit
   Pareto-front search over accuracy, memory, MACs, latency and measured energy would show trade-offs that a scalar
   reward hides. It should be compared with deterministic baselines on the same fronts.
6. **Improved generalisation constraints:**
   - multiple validation subsets;
   - uncertainty-aware thresholds;
   - more robust constraint formulations.

   These would address the finding that validation feasibility did not guarantee test feasibility.
7. **Larger architectures.** ResNet-18/34/50, MobileNet- and EfficientNet-class networks, or other CNNs whose pruning
   spaces cannot be enumerated. That is where a learned search may have more to offer than at the present scale.
8. **Larger datasets.** ImageNet-scale or comparable data, to test whether the retention, reward and recovery findings
   hold when the networks and the cost of each evaluation are much larger.
9. **Edge and mobile deployment.** Mobile CPUs, ARM processors, embedded GPUs and accelerators, each with its own
   latency and energy behaviour.
10. **GPU evaluation.** Whether the relationship between parameter reduction, MAC reduction and latency found on one
    CPU holds, weakens or reverses on GPUs.

### Long-term

11. **Hardware-in-the-loop multi-objective RL.** Rewards that use directly measured latency and energy rather than
    surrogates. These should be attempted only after stable energy instrumentation (item 4) is available, and with
    explicit accuracy constraints.
12. **Reliable lifecycle and carbon analysis.** Carbon-intensity conversion and lifecycle accounting can be meaningful
    only once trustworthy direct energy measurements exist. Until then, no carbon impact should be claimed. This
    thesis makes no such claim.

## 8.7 Final Remarks

The thesis does not establish PPO as a universally superior pruning method. Instead, it shows that adaptive pruning
must be evaluated as a complete pipeline, from search through recovery to deployment.

The resulting evidence identifies both where adaptive RL pruning is useful and where simpler deterministic methods
remain stronger:
- RL pruning is useful as a flexible allocator, especially when its objective is explicitly aligned with the desired
  accuracy boundary.
- Simpler deterministic methods remain stronger for ordinary compression objectives.

It also shows why hardware-level validation is necessary before efficiency claims can be made. The value of the work
lies as much in the links it found to be weak as in those it showed to be sound.
