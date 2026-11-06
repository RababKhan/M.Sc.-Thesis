# Chapter 1 — Introduction

## 1.1 Background

**The cost of deployment.** Deep neural networks reach high predictive accuracy on many tasks, but the models that do
so are often large. Deploying them has a cost on four fronts:
- the parameter memory needed to store and load them;
- the arithmetic they perform for every input;
- the latency of each prediction;
- the energy drawn by the processor that runs them.

These costs recur at every inference, and inference energy has become a subject of study in its own right
[@desislavov2023trends; @luccioni2024powerhungry].

**Pruning.** Network compression reduces these costs while keeping accuracy, through pruning, quantisation,
distillation and efficient architecture design [@menghani2023efficient]. *Pruning* removes parts of a
trained network that contribute little to its predictions. It is one of the oldest compression techniques
[@lecun1989obd; @han2015learning], and it remains among the most studied; a recent survey counts thousands of pruning
papers in a few years [@cheng2024pruningsurvey].

**What is removed:**
- **Unstructured (masked) pruning** sets individual weights to zero. The tensors keep their shapes, and the zeros are
  recorded in a mask.
- **Structured pruning** removes whole channels, filters or neurons, so the network becomes physically smaller and
  its layers perform less arithmetic [@he2024structuredsurvey].

Only the second shrinks the dense computation that general-purpose hardware actually executes
[@hoefler2021sparsity].

**How much each layer should lose.** The simplest rule prunes every layer by the same fraction, but layers differ in
their share of parameters and computation and in the accuracy lost when they are pruned. *Layer-wise adaptive*
pruning gives each layer its own ratio, chosen by:
- **Magnitude-based rules,** which remove the smallest weights within each layer or across the whole network.
- **Layer-adaptive scores,** such as LAMP, which rescale weight magnitudes so that a single global threshold produces
  a non-uniform allocation [@lee2021lamp].
- **Sensitivity-aware methods,** which measure how strongly each layer's pruning affects the loss.
- **Learned allocators,** which search for the ratios directly. Reinforcement learning (RL) agents that choose a ratio
  for each layer, rewarded by the accuracy and compression of the resulting network, are a well-known example
  [@he2018amc].

This thesis uses Proximal Policy Optimisation (PPO) [@schulman2017ppo] as such an allocator and studies what it
contributes, what it needs to work, and whether its compression becomes deployment benefit.

**Why a learned allocator, and what is hard about it.** A learned allocator can optimise objectives that are not
differentiable or per-weight scores, such as validation accuracy, a hard accuracy constraint or a measured hardware
cost, and can be re-targeted without redesign. This flexibility has costs:
- **Evaluation cost.** Every candidate must be evaluated, so the search costs far more than a one-pass rule.
- **Sensitivity to design details.** Policy-gradient methods are sensitive to implementation and training choices
  [@engstrom2020implementation; @andrychowicz2021onpolicy].
- **Sensitivity to the reward.** The outcome depends on how the reward is written.

Whether the flexibility is worth the cost has to be measured against strong deterministic alternatives.

## 1.2 Motivation

Five linked problems lie between "a network can be pruned" and "a pruned network is a better deployment choice".

**Problem A: uniform pruning ignores heterogeneous layer sensitivity.** A single ratio treats a small early
convolution, on which every later layer depends, like a large redundant fully connected layer. Strong deterministic
layer-adaptive methods already exist, so a learned allocator must be compared with them under controlled conditions,
not only with uniform pruning.

**Problem B: a search may find strong policies without retaining them.** An RL agent evaluates many allocations but
returns its final policy; a good search can end on a mediocre allocation, and reporting only that policy cannot
distinguish weak search from weak retention.

**Problem C: a reward can encourage excessive pruning.** RL pruning rewards typically trade accuracy against sparsity
at a fixed rate, with no point at which accuracy loss becomes unacceptable; a better optimiser may then find exactly
the over-pruned models the reward implicitly prefers.

**Problem D: comparisons before equal recovery may overstate differences.** Pruned networks are usually fine-tuned
before use, and fine-tuning can repair much of the damage [@renda2020rewinding; @le2021retraining]. A method that
looks better immediately after pruning may simply damage the network less before repair; its advantage after equal
recovery is a separate question.

**Problem E: sparsity does not necessarily mean speed or energy savings.** A masked network still runs dense
computations on a CPU. Even under structured pruning, the parameters removed and the computation saved need not
coincide, and cost indicators can disagree [@dehghani2022misnomer]. Speed and energy must be measured, with a reliable
instrument.

A study that solves only the first can still return poor allocations, optimise the wrong trade-off, overstate its
advantage or claim efficiency it does not deliver. The design controls each problem:
- **A:** every baseline is built at exactly the sparsity the agent chose;
- **B:** every policy the agent visits is recorded and ranked against an exhaustive landscape;
- **C:** rewards are compared on the same landscapes, and the accuracy boundary is made explicit;
- **D:** every method receives the same fixed-mask recovery with a shared data stream;
- **E:** pruning is realised physically and measured on hardware, with the measuring instrument qualified before any
  comparison.

## 1.3 Problem Statement

Existing pruning methods often optimise compression and accuracy through fixed rules or single saliency criteria.
Reinforcement learning offers a flexible mechanism for learning non-uniform layer-wise allocations, but its
effectiveness depends on four things beyond the learning algorithm itself:
- the quality of the guidance given to the search;
- whether good policies found during the search are retained;
- whether the reward encodes the intended accuracy–compression trade-off;
- how the pruned models are recovered before comparison.

Nor does the fraction of zero weights establish deployment efficiency. This thesis therefore investigates whether a
PPO agent can learn adaptive layer-wise pruning policies, what sensitivity, retention, reward design and equal recovery
contribute, and whether the compression becomes memory, computation, latency and measured energy benefits.

**Research approach.** The investigation was a sequence of phases, each with conditions, seeds, metrics and decision
criteria committed before its confirmatory data were collected (Chapter 4). Four design choices make the results
auditable:
- **Exhaustive ground truth.** Four pruning units with six ratios give 1,296 policies per setting, all evaluated, so
  the rank of any visited or returned policy is known exactly.
- **Strict data governance.** Training images update weights only; one validation split drives the reward, a second,
  disjoint split only selects among candidates; the test set is read only for frozen models.
- **Matched comparisons.** Every baseline is built at exactly the zero count of the PPO model it is compared with, and
  every compared model receives an identical recovery protocol.
- **A separate extension.** Pruning is realised physically and latency and energy are measured directly, under
  pre-registered stability checks for the instrument.

## 1.4 Research Gap

Chapter 2 identifies six gaps. RL has been applied to pruning [@he2018amc]; the gaps concern how such methods are
evaluated and what their results establish.

1. **Adaptive allocation against strong deterministic allocation.** Layer-specific allocation matters, but learned
   allocators are rarely compared with uniform, random, magnitude-based and layer-adaptive baselines built at exactly
   the same sparsity. → RQ1.
2. **Exploration effects versus final-model effects of sensitivity.** Sensitivity's effect on search efficiency is
   seldom separated from its effect on final accuracy, or compared with uninformative controls. → RQ2.
3. **Retention of strong policies found during search.** RL pruning is usually judged by its final policy. Whether
   stronger policies were encountered and lost has not been examined systematically for pruning. → RQ3.
4. **Incentives for excessive sparsity in pruning rewards.** Hand-weighted accuracy–sparsity rewards can be exploited
   by capable optimisers [@pan2022misspecification]. Whether better pruning search exposes this, and whether an
   explicit accuracy constraint controls it, has not been tested directly. → RQ4.
5. **Comparisons without equal recovery.** Pruning methods are often compared under inconsistent experimental
   conditions [@blalock2020state], and the retraining schedule alone can change which pruned network performs better
   [@le2021retraining]. Whether differences between a learned allocator and deterministic methods, measured before
   recovery or under different recovery budgets, survive identical post-pruning fine-tuning therefore remains to be
   tested. → RQ5.
6. **Compression versus deployment efficiency.** Several questions remain empirical:
   - how well parameter and MAC reductions predict measured CPU latency across a whole pruning space;
   - how much latency headroom remains under an accuracy constraint;
   - whether a structured realisation of a learned allocation reduces directly measured energy without material
     accuracy loss.

   Energy measurement on general-purpose CPUs adds its own reliability problem. → RQ6.

## 1.5 Research Questions

**RQ1–RQ5** concern the **core framework**, which uses masked (unstructured) pruning. **RQ6** concerns the
**deployment extension**, added after the core framework had been completed and evaluated.

> **RQ1.** Does PPO learn non-uniform layer-wise pruning allocations that outperform simple allocation strategies, and
> how does it compare with a strong deterministic method such as LAMP?
>
> **RQ2.** Does layer-sensitivity information improve the pruning search process, and does that improvement translate
> into higher final predictive accuracy?
>
> **RQ3.** Does PPO retain the strongest pruning policies encountered during search, and can archive-based policy
> selection improve retention?
>
> **RQ4.** How does reward design influence pruning aggressiveness and the accuracy–compression trade-off?
>
> **RQ5.** How does equal post-pruning fine-tuning affect comparisons between PPO and deterministic pruning methods?
>
> **RQ6.** Do compression gains translate into real deployment benefits in memory, computation, latency and measured
> operational energy?

The questions were not all fixed at the outset; each answer exposed the next problem (Chapter 3).

## 1.6 Research Objectives

**General objective.** To develop and systematically evaluate a sensitivity-aware reinforcement-learning framework for
adaptive layer-wise neural-network pruning, including its search behaviour, recovery characteristics and deployment
implications.

**Specific objectives:**

- **O1.** Develop a PPO-based framework for non-uniform layer-wise pruning, and evaluate it against uniform, random,
  global-magnitude, ERK and LAMP allocations at identical sparsity.
- **O2.** Evaluate whether sensitivity guidance improves the pruning search and the final model quality, using
  shuffled, constant, magnitude-matched and sign-reversed controls.
- **O3.** Investigate policy-retention behaviour, and evaluate archive-based selection of the policies visited during
  search.
- **O4.** Analyse reward misalignment, and evaluate an accuracy-constrained reward with validation-based selection.
- **O5.** Assess how an equal, fixed-mask post-pruning recovery protocol affects comparative conclusions between PPO
  and deterministic methods.
- **O6.** Evaluate whether pruning-related compression translates into memory, computation, latency and directly
  measured operational-energy improvements when realised as physically structured networks.

Each objective was addressed by completed, pre-registered experiments (Chapter 4); Table 1.1 maps objectives to
research questions, contributions and evidence.

## 1.7 Research Contributions

The thesis makes six contributions. None of them rests on a claim that PPO is the best pruning method.

**Contribution 1. An adaptive layer-wise pruning framework, evaluated against strong baselines.** A PPO agent assigns
one of six ratios to each of four units, evaluated against uniform, random, global-magnitude, ERK and LAMP allocations
at exactly its own sparsity, on two datasets and three architectures. Its non-uniform allocations strongly outperform
uniform and random allocation; LAMP remains stronger before recovery.

**Contribution 2. A separation of the exploration effect and the final-model effect of sensitivity guidance.** Across
three phases and against uninformative controls, sensitivity's measurable contribution is to the *search*
(better-ranked policies earlier, fewer destructive ones); its contribution to final accuracy was not detectable.

**Contribution 3. Identification of a policy-retention problem in RL pruning, and evaluation of archive-based
selection.** Using exhaustive ground-truth landscapes, the thesis shows that the agent visits near-optimal policies
but rarely ends on them. An archive of visited policies, with selection on held-out validation data, restores
retention in terms of validation rank. Its effect on test accuracy is mixed.

**Contribution 4. Identification of reward misalignment, and evaluation of an accuracy-constrained reward.** Improving
search and retention exposed that a fixed-rate accuracy–sparsity reward drives fragile networks to over-prune. A
constraint-first reward, with selection on held-out data, substantially reduced the most destructive over-pruning.
Validation feasibility, however, did not reliably transfer to the test set.

**Contribution 5. A controlled equal-recovery analysis.** Every compared model received an identical fixed-mask
fine-tuning protocol, with shared seeds and data order, at identical sparsity; the differences between methods became
small, which grounds the thesis's main methodological recommendation.

**Contribution 6. A deployment analysis that keeps four quantities apart:** parameter compression, MAC reduction,
measured latency and directly measured operational energy. It comprises:
- an exhaustive latency analysis of every structured policy of two networks and two output sizes;
- direct CPU package-energy measurement of a structured realisation of the learned allocation;
- a deployment-oriented structured PPO that learns physical channel allocations directly;
- pre-registered qualification of the energy instrument, reported including its failure.

**Novelty.** The novelty lies not in a new pruning criterion or a claim that PPO outperforms existing methods, but in
**treating adaptive pruning as a sequence of connected optimisation and validation problems, rather than evaluating
only the final compressed model**: allocation, exploration, policy retention, reward alignment, recovery, deployment
translation and hardware validation, each examined with its own experiment and decision criteria fixed before the
confirmatory data were collected. It shows how a weakness at one stage can hide or reverse improvements at another.

**Table 1.1. Research questions, objectives, contributions and evidence.**

| RQ | Objective | Contribution | Evidence |
|---|---|---|---|
| RQ1 | O1 | TC1 | Chapter 5 (Phase 3; Phases 4–5 LAMP comparisons) |
| RQ2 | O2 | TC2 | Chapter 5 (Phases 3, 4, 5) |
| RQ3 | O3 | TC3 | Chapter 5 (Phase 4) |
| RQ4 | O4 | TC4 | Chapter 5 (Phases 4, 5) |
| RQ5 | O5 | TC5 | Chapter 5 (Phase 6) |
| RQ6 | O6 | TC6 | Chapter 6 (Paper 3, Phases 7A, 7B, energy confirmation) |

TC1–TC6 denote Contributions 1–6. The labels C0, C1 and C2 used in later chapters denote Phase-5 experimental
conditions, not contributions.

**Preview of the main findings** (Chapters 5–7):
- PPO substantially outperformed uniform (by up to about 31.4 percentage points of test accuracy) and random pruning,
  but LAMP remained stronger before equal recovery.
- Sensitivity guidance improved exploration but not final accuracy; archive selection restored validation-rank
  retention, with a mixed test effect.
- The accuracy-constrained reward substantially reduced destructive over-pruning, but its validation-certified
  feasibility held on the test set in only about two thirds of runs.
- Equal recovery narrowed the PPO–LAMP differences to within a pre-registered confidence-interval margin in every
  setting, although LAMP remained slightly ahead in three.
- Parameter reduction was a weak predictor of CPU latency compared with MAC reduction. Structured pruning reduced
  directly measured CPU package energy (Phase 7A) at a substantial accuracy cost; on one network, uniform allocation
  was the better choice.
- The redesigned agent (Phase 7B) reduced parameter memory by 63.4% and MACs by 36.0% at a validation-accuracy cost of
  1.12 percentage points; it missed the pre-registered accuracy criterion, and its direct energy could not be
  confirmed.

## 1.8 Scope and Boundaries

**Included in the scope:**
- **Datasets:** CIFAR-10 and CIFAR-100 image classification.
- **Networks:** three small convolutional networks (SimpleCNN, a CIFAR adaptation of LeNet-5, and ResNet-8), each
  with four pruning units and six pruning ratios.
- **Method:** PPO; masked (unstructured) pruning in the core framework, with all 1,296 policies per setting
  evaluated as ground truth; physically structured pruning on a subset of settings in the deployment extension.
- **Hardware evaluation:** parameter memory, MACs, batch-1 CPU latency, and CPU package energy measured through the
  processor's energy counters, on one laptop CPU.

**Not established by this thesis:**
- behaviour at ImageNet scale, or for larger CNNs, transformers or large language models;
- performance on GPUs, mobile processors, edge accelerators or cloud inference;
- whole-system or wall-plug energy, memory energy, or training energy;
- any carbon-footprint, lifecycle or environmental-impact claim.

The energy evaluation concerns operational inference energy; it is not an environmental assessment.

**Boundaries that the reader should keep in mind throughout:**
- PPO is **not** presented as universally better than deterministic pruning. LAMP remains a strong baseline, and was
  more accurate than PPO before recovery in nearly every setting.
- Sensitivity guidance is, on the evidence, primarily an **exploration aid**.
- Archive selection is a **validation-retention mechanism**, not a generalisation mechanism.
- Validation-based accuracy constraints **do not guarantee** feasibility on the test set.
- The ±0.5 percentage-point rule used after equal recovery is a pre-registered confidence-interval decision rule, and
  **not** a formal statistical equivalence test.
- **Phase 7A** directly measured energy reductions, but every pruned model lost more accuracy than the criterion
  allowed.
- **Phase 7B's** direct energy remains **unresolved**, because the measurement setup failed its stability check. No
  latency or energy reduction is claimed for it.

## 1.9 Thesis Organisation

Chapter 2 reviews the literature and ends with the research gaps. Chapter 3 presents the framework in the order in
which it developed, with the deployment-oriented structured extension separately. Chapter 4 gives the experimental
setup. Chapter 5 reports the core-framework results (Phases 3–6, RQ1–RQ5) and Chapter 6 the deployment analysis
(RQ6). Chapter 7 interprets the findings and states the limitations; Chapter 8 concludes and outlines future work.

## 1.10 Chapter Summary

This chapter argued that adaptive layer-wise pruning is a chain of connected problems rather than a single choice of
ratio, and set out six research questions: **RQ1–RQ5** for a PPO-based, sensitivity-aware core framework using masked
pruning, and **RQ6** for a structured deployment extension with direct hardware measurement. The thesis does not
claim superiority over existing methods, and its scope is limited to small CNNs, CIFAR data and one CPU platform.
