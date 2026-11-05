# Chapter 1 — Introduction

## 1.1 Background

**The cost of deployment.** Deep neural networks reach high predictive accuracy on many tasks, but the models that do
so are often large. Deploying them has a cost on four fronts:
- the parameter memory needed to store and load them;
- the arithmetic they perform for every input;
- the latency of each prediction;
- the energy drawn by the processor that runs them.

These costs matter most at inference time, which is repeated for every query. Inference energy and its trends have
become a subject of study in their own right [@desislavov2023trends; @luccioni2024powerhungry].

**Pruning.** Network compression seeks to keep a model's accuracy while reducing these costs, through pruning,
quantisation, distillation and efficient architecture design [@menghani2023efficient]. *Pruning* removes parts of a
trained network that contribute little to its predictions. It is one of the oldest compression techniques
[@lecun1989obd; @han2015learning], and it remains among the most studied; a recent survey counts thousands of pruning
papers in a few years [@cheng2024pruningsurvey].

**What is removed:**
- **Unstructured (masked) pruning** sets individual weights to zero. The tensors keep their shapes, and the zeros are
  recorded in a mask.
- **Structured pruning** removes whole channels, filters or neurons, so the network becomes physically smaller and
  its layers perform less arithmetic [@he2024structuredsurvey].

The first gives fine-grained control. Only the second shrinks the dense computation that general-purpose hardware
actually executes [@hoefler2021sparsity].

**How much each layer should lose.** The simplest rule prunes every layer by the same fraction. Layers, however,
differ: in their share of the parameters, in their share of the computation, and in how much accuracy is lost when
they are pruned. This motivates *layer-wise adaptive* pruning, in which each layer receives its own ratio.
Approaches to choosing the ratios include:
- **Magnitude-based rules,** which remove the smallest weights within each layer or across the whole network.
- **Layer-adaptive scores,** such as LAMP, which rescale weight magnitudes so that a single global threshold produces
  a non-uniform allocation [@lee2021lamp].
- **Sensitivity-aware methods,** which measure how strongly each layer's pruning affects the loss.
- **Learned allocators,** which search for the ratios directly. Reinforcement learning (RL) agents that choose a ratio
  for each layer, rewarded by the accuracy and compression of the resulting network, are a well-known example
  [@he2018amc].

This thesis uses Proximal Policy Optimisation (PPO) [@schulman2017ppo] as such an allocator. It studies what that
allocator contributes, what it needs in order to work, and whether its compression becomes real deployment benefit.

**Why a learned allocator, and what is hard about it.** A learned allocator is attractive for a specific reason. The
quantity it optimises need not be differentiable or expressible as a per-weight score. It can be validation accuracy
after pruning, a hard accuracy constraint, or a measured hardware cost, and the same agent can be pointed at a
different objective without redesigning the method.

This flexibility comes with difficulties of its own:
- **Evaluation cost.** Each candidate allocation must be evaluated, so the search is far more expensive than a
  one-pass rule.
- **Sensitivity to design details.** Policy-gradient methods are sensitive to implementation and training choices
  [@engstrom2020implementation; @andrychowicz2021onpolicy].
- **Sensitivity to the reward.** The outcome depends on how the reward is written.

Whether the flexibility is worth the cost cannot be assumed. It has to be measured against strong deterministic
alternatives.

## 1.2 Motivation

The thesis is motivated by five linked problems. Each concerns a different step between "a network can be pruned" and
"a pruned network is a better deployment choice".

**Problem A: uniform pruning ignores heterogeneous layer sensitivity.** A single ratio for all layers treats a small
early convolution, which every later layer depends on, the same as a large fully connected layer with many redundant
weights. An adaptive allocation should do better, but the question is *how much* better, and against *which*
alternatives. Strong deterministic layer-adaptive methods already exist. A learned allocator has to be compared with
them under controlled conditions, not only with uniform pruning.

**Problem B: a search may find strong policies without retaining them.** An RL agent evaluates many candidate
allocations during training, but what it returns is its final policy. If its distribution drifts away from the best
candidates it has seen, a good search can still end on a mediocre allocation. Reporting only the final policy cannot
distinguish weak search from weak retention.

**Problem C: a reward can encourage excessive pruning.** RL pruning rewards typically trade accuracy against sparsity
at a fixed rate. Such a reward has no point at which accuracy loss becomes unacceptable. A more effective optimiser
may then find exactly the over-pruned models the reward implicitly prefers.

**Problem D: comparisons before equal recovery may overstate differences.** Pruned networks are usually fine-tuned
before use, and fine-tuning can repair much of the damage [@renda2020rewinding; @le2021retraining]. A method that
looks better immediately after pruning may simply damage the network less before repair. Its advantage after equal
recovery is a separate question.

**Problem E: sparsity does not necessarily mean speed or energy savings.** A masked network with many zeros still runs
dense computations on a CPU. Even when pruning is structured, the parameters removed and the computation saved need
not coincide, and different cost indicators can disagree [@dehghani2022misnomer]. Whether a compressed model is faster
or uses less energy has to be measured, and the measurement must itself be reliable.

These problems form a chain. A study that solves only the first can still return poor allocations, optimise the wrong
trade-off, overstate its advantage, or claim efficiency it does not deliver. The thesis follows the chain from
allocation to measured hardware energy. Each problem is controlled in the design:

| Problem | How the design controls it |
|---|---|
| A | Every baseline is built at exactly the sparsity the agent chose |
| B | Every policy the agent visits is recorded and ranked against an exhaustive landscape |
| C | Rewards are compared on the same landscapes, and the accuracy boundary is made explicit |
| D | Every method receives the same fixed-mask recovery with a shared data stream |
| E | Pruning is realised physically and measured on hardware, with the measuring instrument qualified before any comparison |

## 1.3 Problem Statement

Existing pruning methods often optimise compression and predictive accuracy through fixed rules or single saliency
criteria. Reinforcement learning offers a flexible mechanism for learning non-uniform layer-wise allocations. Its
effectiveness, however, depends on four things beyond the learning algorithm itself:
- the quality of the guidance given to the search;
- whether good policies found during the search are retained;
- whether the reward encodes the intended accuracy–compression trade-off;
- how the pruned models are recovered before comparison.

Moreover, conventional compression metrics, such as the fraction of zero weights, do not by themselves establish
deployment efficiency.

This thesis therefore investigates whether a PPO agent can learn adaptive layer-wise pruning policies, and also:
- what layer-sensitivity information contributes to that learning;
- whether the strongest policies encountered during search are retained;
- how reward design affects pruning aggressiveness;
- how equal post-pruning fine-tuning changes comparisons with deterministic methods;
- whether the resulting compression translates into memory, computation, latency and measured energy benefits.

**Research approach.** The investigation was carried out as a sequence of experimental phases. Each phase's
conditions, seeds, metrics and decision criteria were fixed and committed before its confirmatory data were collected
(Chapter 4). Four design choices make the results auditable:
- **Exhaustive ground truth.** The pruning space of each setting is small enough to enumerate: four pruning units with
  six ratios each give 1,296 policies. Every policy was evaluated once, so the rank of any policy the agent visits or
  returns is known exactly. This allows the thesis to measure search and retention against ground truth, rather than
  only report the final model.
- **Strict data governance.** The training images update weights only. One validation split drives the reward, and a
  second, disjoint split is used only to select among candidates already produced. The test set is read only for
  frozen models.
- **Matched comparisons.** In the core framework, every baseline is built at exactly the number of zeroed weights of
  the PPO model it is compared with. After pruning, every compared model receives an identical recovery protocol.
- **A separate extension.** The deployment questions are addressed by a later extension. It realises pruning
  physically and measures latency and energy directly, under pre-registered stability checks for the measuring
  instrument.

## 1.4 Research Gap

Chapter 2 reviews the literature on pruning criteria, layer-wise allocation, RL-based compression, sensitivity,
recovery, and hardware and energy efficiency. It identifies six narrower gaps. The claim is not that RL has never been
applied to pruning; it has [@he2018amc]. The gaps concern how such methods are evaluated and what their results
establish.

1. **Adaptive allocation against strong deterministic allocation.** Layer-specific allocation matters, but learned
   allocators are rarely compared with uniform, random, magnitude-based and layer-adaptive baselines built at exactly
   the same sparsity. → RQ1.
2. **Exploration effects versus final-model effects of sensitivity.** Sensitivity is widely used to guide pruning, but
   its effect on the efficiency of a search is seldom separated from its effect on final accuracy, or compared with
   uninformative controls. → RQ2.
3. **Retention of strong policies found during search.** RL pruning is usually judged by its final policy. Whether
   stronger policies were encountered and lost has not been examined systematically for pruning. → RQ3.
4. **Incentives for excessive sparsity in pruning rewards.** Hand-weighted accuracy–sparsity rewards can be exploited
   by capable optimisers [@pan2022misspecification]. Whether better pruning search exposes this, and whether an
   explicit accuracy constraint controls it, has not been tested directly. → RQ4.
5. **Comparisons without equal recovery.** Differences between pruning methods measured before recovery, or under
   different recovery budgets, may not survive identical post-pruning fine-tuning [@blalock2020state]. → RQ5.
6. **Compression versus deployment efficiency.** Several questions remain empirical:
   - how well parameter and MAC reductions predict measured CPU latency across a whole pruning space;
   - how much latency headroom remains under an accuracy constraint;
   - whether a structured realisation of a learned allocation reduces directly measured energy without material
     accuracy loss.

   Energy measurement on general-purpose CPUs adds its own reliability problem. → RQ6.

## 1.5 Research Questions

The thesis addresses six research questions. **RQ1–RQ5** concern the **core framework**, which uses masked
(unstructured) pruning. **RQ6** concerns the **deployment extension**, which was added after the core framework had
been completed and evaluated.

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

The questions were not all fixed at the outset. As Chapter 3 describes, each answer exposed the next problem, and the
later questions were posed in response to the earlier results:
- improving the search revealed a retention problem;
- retention revealed reward misalignment;
- a corrected reward made a fair recovery comparison meaningful;
- the deployment question followed.

## 1.6 Research Objectives

**General objective.** To develop and systematically evaluate a sensitivity-aware reinforcement-learning framework for
adaptive layer-wise neural-network pruning, including its search behaviour, its recovery characteristics and its
deployment implications.

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

Each objective was addressed by completed, pre-registered experiments (Chapter 4). Table 1.1 maps the objectives to
the research questions, contributions and evidence.

## 1.7 Research Contributions

The thesis makes six contributions. None of them rests on a claim that PPO is the best pruning method.

**C1. An adaptive layer-wise pruning framework, evaluated against strong baselines.**
- **What it is.** A PPO agent that assigns one of six pruning ratios to each of four units per network. It was
  evaluated against uniform, random, global-magnitude, ERK and LAMP allocations built at exactly the sparsity the
  agent chose, across two datasets and three architectures.
- **What it shows.** The agent learns non-uniform allocations that strongly outperform uniform and random allocation.
  The comparison also shows that the deterministic LAMP method remains stronger before recovery.

**C2. A separation of the exploration effect and the final-model effect of sensitivity guidance.** Across three
experimental phases, and against several uninformative controls, the analysis shows that sensitivity's main
measurable contribution is to the *search*: better-ranked policies earlier and fewer destructive ones. Its
contribution to final predictive accuracy was not detectable.

**C3. Identification of a policy-retention problem in RL pruning, and evaluation of archive-based selection.** Using
exhaustive ground-truth landscapes, the thesis shows that the agent visits near-optimal policies but rarely ends on
them. An archive of visited policies, with selection on held-out validation data, restores retention in terms of
validation rank. Its effect on test accuracy is mixed.

**C4. Identification of reward misalignment, and evaluation of an accuracy-constrained reward.**
- **The diagnosis.** Improving search and retention exposed that a fixed-rate accuracy–sparsity reward drives fragile
  networks to over-prune.
- **The correction.** A constraint-first reward, with selection on held-out data, substantially reduced the most
  destructive over-pruning.
- **The limitation.** Validation feasibility did not reliably transfer to the test set.

**C5. A controlled equal-recovery analysis.** Every compared model received an identical fixed-mask fine-tuning
protocol, with shared seeds and data order, at identical sparsity. This shows how much of the difference between
pruning methods remains after recovery. The differences became small. The thesis's main methodological
recommendation for pruning comparisons follows from this analysis.

**C6. A deployment analysis that keeps four quantities apart:** parameter compression, MAC reduction, measured
latency and directly measured operational energy.
- **Latency.** An exhaustive latency analysis of every structured policy of two networks and two output sizes.
- **Energy of a transferred allocation.** Direct CPU package-energy measurement of a structured realisation of the
  learned allocation.
- **A redesigned agent.** A deployment-oriented structured PPO that learns physical channel allocations directly.
- **Measurement qualification.** Pre-registered qualification of the energy instrument, reported including its
  failure.

**Novelty.** The novelty does not lie in a new pruning criterion, or in a claim that PPO outperforms existing methods.
It lies in **treating adaptive pruning as a sequence of connected optimisation and validation problems, rather than
evaluating only the final compressed model.** These problems are:
1. allocation;
2. exploration;
3. policy retention;
4. reward alignment;
5. recovery;
6. deployment translation;
7. hardware validation.

The thesis examines each with its own experiment, and each with decision criteria fixed before the confirmatory data
were collected. The contribution is methodological and analytical: it shows how a weakness at one stage can hide or
reverse the effect of improvements at another.

**Table 1.1. Research questions, objectives, contributions and evidence.**

| RQ | Objective | Contribution | Evidence |
|---|---|---|---|
| RQ1 | O1 | C1 | Chapter 5 (Phase 3; Phases 4–5 LAMP comparisons) |
| RQ2 | O2 | C2 | Chapter 5 (Phases 3, 4, 5) |
| RQ3 | O3 | C3 | Chapter 5 (Phase 4) |
| RQ4 | O4 | C4 | Chapter 5 (Phases 4, 5) |
| RQ5 | O5 | C5 | Chapter 5 (Phase 6) |
| RQ6 | O6 | C6 | Chapter 6 (Paper 3, Phases 7A, 7B, energy confirmation) |

**Preview of the main findings.** Chapters 5–7 report and discuss the results in full. In brief:
- **PPO against simple allocation.** PPO's learned allocations substantially outperformed uniform pruning (by up to
  about 31.4 percentage points of test accuracy) and random pruning in the studied settings.
- **LAMP.** LAMP remained the stronger method before equal recovery.
- **Sensitivity.** Sensitivity guidance improved exploration but not final accuracy.
- **Archive.** Archive selection restored validation-rank retention. Its test effect was mixed.
- **Reward.** The accuracy-constrained reward substantially reduced destructive over-pruning. Its validation-certified
  feasibility, however, held on the test set in only about two thirds of runs.
- **Equal recovery.** It greatly narrowed the differences between PPO and LAMP. They fell within a pre-registered
  confidence-interval margin in every setting, although LAMP remained slightly ahead in three.
- **Latency proxies.** On the deployment side, parameter reduction was a weak predictor of CPU latency compared with
  MAC reduction.
- **Measured energy (Phase 7A).** Physically structured pruning reduced directly measured CPU package energy, but with
  a substantial accuracy cost. On one network, a simple uniform allocation was the better choice.
- **The redesigned agent (Phase 7B).** It reduced parameter memory by 63.4% and MACs by 36.0% at a validation-accuracy
  cost of 1.12 percentage points. It missed the pre-registered accuracy criterion, and its direct energy could not be
  confirmed.

## 1.8 Scope and Boundaries

**Included in the scope:**
- **Datasets:** CIFAR-10 and CIFAR-100 image classification.
- **Networks:** three small convolutional networks (SimpleCNN, a CIFAR adaptation of LeNet-5, and ResNet-8), each
  with four pruning units and six pruning ratios.
- **Learning algorithm:** PPO.
- **Core framework:** masked (unstructured) pruning, with exhaustive evaluation of all 1,296 policies per setting as
  ground truth.
- **Deployment extension:** physically structured pruning on a subset of these settings.
- **Hardware evaluation:** parameter memory, MACs, batch-1 CPU latency, and CPU package energy measured through the
  processor's energy counters, on one laptop CPU.

**Not established by this thesis:**
- behaviour at ImageNet scale, or for larger CNNs, transformers or large language models;
- performance on GPUs, mobile processors, edge accelerators or cloud inference;
- whole-system or wall-plug energy, memory energy, or training energy;
- any carbon-footprint, lifecycle or environmental-impact claim.

The energy evaluation is a sustainability-oriented evaluation of operational inference energy, not an environmental
assessment.

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

The thesis is organised as follows:
- **Chapter 1, Introduction:** the background, motivation, problem, research gap, questions, objectives,
  contributions and scope.
- **Chapter 2, Literature Review:** neural-network compression; unstructured and structured pruning; layer-wise and
  layer-adaptive allocation, including LAMP; reinforcement learning for compression and PPO; sensitivity-aware
  pruning; search and policy selection; post-pruning recovery; hardware-aware compression; and energy measurement. It
  ends with the six research gaps.
- **Chapter 3, Proposed Framework:** the framework, presented in the order in which it developed. It covers the
  layer-wise pruning representation, the RL formulation, sensitivity and the two sensitivity priors, archive
  selection, the accuracy-constrained reward and the equal-recovery protocol. The later deployment-oriented structured
  extension is presented separately.
- **Chapter 4, Experimental Setup:** datasets and the two validation configurations, architectures and dense
  references, pruning units, baselines and matching rules, PPO configuration, the conditions of every phase, metrics,
  statistical decision rules, hardware and software, the latency and energy protocols, and reproducibility.
- **Chapter 5, Results of the Core Framework:** Phases 3–6, answering RQ1–RQ5.
- **Chapter 6, Deployment and Sustainability Analysis:** the exhaustive latency analysis, the structured transfer with
  measured energy, the deployment-oriented structured PPO and the energy-confirmation attempt, answering RQ6.
- **Chapter 7, Discussion:** interpretation of the findings across phases, the limitations and the thesis's overall
  contribution.
- **Chapter 8, Conclusion and Future Work:** a summary of the answers to the research questions, and the future work
  they motivate.

## 1.10 Chapter Summary

**The problem.** This chapter introduced the problem of compressing neural networks so that they are cheaper to
deploy. It argued that adaptive layer-wise pruning is a chain of connected problems rather than a single choice of
ratio:
- deciding where to prune;
- guiding and retaining the search;
- encoding the intended trade-off;
- comparing methods after equal recovery;
- confirming that compression becomes real hardware savings.

**The thesis.** It addresses these problems through six research questions and six objectives, with six
contributions:
- **RQ1–RQ5** concern a PPO-based, sensitivity-aware core framework using masked pruning.
- **RQ6** concerns a later deployment extension using structured pruning and direct hardware measurement.

**The framing.** The thesis's novelty is methodological. It treats pruning as a sequence of optimisation and
validation problems, each evaluated on its own evidence and including its negative results. It does not claim
superiority over existing methods, and its scope is limited to small CNNs, CIFAR data and one CPU platform.

Chapter 2 reviews the literature on which these questions build.
