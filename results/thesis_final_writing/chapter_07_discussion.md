# Chapter 7 — Discussion

## 7.1 General

Chapters 5 and 6 reported what happened in the core pruning experiments (Phases 3–6) and when the pruned models were
realised and measured on hardware (Paper 3, Phases 7A and 7B, and the energy confirmation study). This chapter
interprets that evidence across phases, places it against the literature reviewed in Chapter 2, and states where the
proposed framework succeeds and where it does not.

No new numbers are introduced; only the few audited numbers that an argument needs are repeated. Causal language is
reserved for effects the experimental design isolated, such as the C1-versus-C2 contrast of Phase 5, or for relations
that hold by construction; where only an association was observed, the text says so. The discussion follows the order
in which the thesis encountered its problems: the agent and sensitivity (Sections 7.2–7.3); retention, reward and
recovery (Sections 7.4–7.7); deployment and energy (Sections 7.8–7.11); and generalisability, limitations and the
overall contribution (Sections 7.12–7.15).

## 7.2 What Adaptive Layer-Wise PPO Actually Contributes

**The evidence (Section 5.3).** The PPO agent learned visibly non-uniform allocations, pruning large, low-sensitivity
layers heavily and keeping small early layers almost intact (Figure 5.2). At the sparsity it chose, these allocations
were far more accurate than uniform allocation (5 of 6 settings, up to about +31.4 pp) and random allocation (6 of 6);
against global magnitude and ERK the picture was mixed; and LAMP was significantly more accurate than PPO before
recovery in 5 of 6 settings in Phase 3, and in 6 of 6 for the later pipelines.

**What this establishes.**
- **The agent solves the allocation problem it was given.** It discovers, without any hand-written allocation rule,
  that layers should be pruned differently. In this sense the thesis confirms the premise behind RL-based compression
  [@he2018amc]: layer-wise allocation is learnable from a scalar reward.
- **It does not establish superiority.** The thesis does not show that a learned allocation is better than the best
  deterministic allocation rule. The comparison with LAMP, a cheap and deterministic method, points the other way at
  the sparsities studied.

**Where the agent's value lies.** One interpretation, supported by the later phases, is that PPO's value lies in its
*flexibility*: Phases 4, 5 and 7B changed the prior, the reward and the action space, while the agent and its training
procedure stayed the same, whereas a fixed rule such as LAMP optimises one implicit criterion. This evidence is
indirect. The thesis shows that PPO *can* be pointed at a new objective and will pursue it (in Phase 5 this paid off
for ResNet-8), not that this flexibility yields a better final model than a tailored deterministic method for any
objective tested here. RL pruning should therefore be presented as an adaptive optimisation mechanism for objectives
that are hard to express as a single saliency score, not as a replacement for strong deterministic pruning under
ordinary magnitude-based compression. Nothing in the thesis implies that PPO is necessary for pruning in general.

## 7.3 The Role of Sensitivity Information

**The pattern (Sections 5.4, 5.5 and 5.8).** Across three phases, sensitivity guidance improved the *search*, but not
the *final model*. In Phase 3 the prior sampled better-ranked policies earlier, with fewer destructive ones; the
primary accuracy-AUC endpoint improved in 3 of 6 settings, and no final-accuracy effect was found in any setting. In
Phase 4 the centred prior beat both its shuffled control and a magnitude-matched conservative control on reward-rank
AUC, and beat the sign-reversed prior in every setting, but did not improve the final rank. In Phase 5, under the
corrected reward, it again improved exploration in 5 of 6 settings. Yet C1's test accuracy was 0.05–0.16 pp *below*
C2's in every setting (descriptive).

**The strongest defensible reading.** Sensitivity is useful as a **search bias**: its measurable contribution lies
in exploration guidance, not in predicting which final model will be most accurate. A natural explanation, which the
thesis can describe but not prove, is that the prior shapes *which* policies are sampled early, whereas the returned
policy is decided by the reward landscape and the selection rule. In these small spaces, where every setting's best-seen policy already had a median rank of 1–4.5, the search
finds near-optimal regions with or without the prior. A better route to the same destination need not change the
destination.

**Why centring mattered.** In Phase 3 the shuffled and constant priors did as well as the correct one on LeNet-5,
and all three Phase-3 priors share the same total pressure towards lighter pruning, so part of any benefit could be
generic conservatism. Centring removes that net conservative pressure by construction. Its advantage over permuted and
magnitude-matched controls on reward-rank AUC therefore provides stronger evidence for a layer-relative exploration
effect than Phase 3 did, although not complete causal isolation of layer-specific information. On the accuracy-based
exploration measure, however, the *conservative* control was significantly better than the centred prior in 4 of 6
settings, consistent with aggressive actions on insensitive units raising the reward rank of sampled policies while
lowering their average accuracy. Centring made the prior more layer-specific; it did not make it a better predictor of
final accuracy.

**Implication for the literature.** Sensitivity is often treated as a direct proxy for how much a layer can be
pruned. The dissociation found here suggests evaluating sensitivity-based methods separately on whether they improve
the *search* and whether they improve the *outcome*; the thesis found the first, but not the second. This does not mean
sensitivity "does not work"; its contribution is of a different kind than often assumed.

## 7.4 Search Quality versus Policy Retention

**The finding (Section 5.6).** One of the clearest results of the thesis is that **finding a good policy and ending on
it are different problems.** In Phase 4, PPO's best visited policy reached the top 10 of 1,296 in 95–100% of runs,
whereas the terminal policy did so in 0% of runs in five settings and 5% in the sixth; the same pattern was present in
Phase 3. Archive selection restored retention on validation ranking (80–100% top-10 reach).

A possible mechanism, which was not isolated, is that a stochastic policy-gradient method updating on noisy,
high-variance estimates may sample an excellent policy early and then settle on a region of high average reward that
does not contain it.

**What the finding establishes is methodological.** A study that reports only the terminal policy of an RL pruning
agent measures retention together with search. Had the thesis reported only terminal policies, it would have
concluded that PPO searches poorly; the exhaustive landscapes showed instead that it searches well and retains poorly. This mirrors the gap between the search phase and the final selection in
architecture search [@yu2020evalnas], and the value of archives of visited states in exploration
[@ecoffet2021goexplore]. RL pruning studies should therefore report the best visited policy, the retained policy and
the selection procedure that links them, separately.

**Elite behavioural cloning.** An imitation term towards the best policies seen so far, in the spirit of
self-imitation learning [@oh2018sil], improved the final-selected rank in only 3 of 6 settings, against 6 of 6 for the
archive rule alone. Pulling the actor towards elite policies during training was thus not equivalent to keeping the
elite candidates outside the agent; with one strength (η = 0.1) and one elite size (K = 5) tested, this does not show
that behavioural cloning is ineffective in general.

**The boundary.** Archive selection improves *validation-rank* retention. Its effect on test accuracy was mixed: small
gains in four settings, and losses of 5.66 and 4.00 pp on ResNet-8, where the archive retained exactly what the reward
ranked highest, an over-pruned model. Archive selection is therefore a retention mechanism, not a generalisation
mechanism. Retention amplifies whatever the reward specifies, which leads directly to the next section.

## 7.5 Reward Design and Over-Pruning

**A stronger optimiser exposes the reward (Section 5.7).** Once search and retention improved in Phase 4, the agent
reliably reached the optimum of the reward R0. That optimum is set by R0's fixed exchange rate between accuracy and
sparsity, and on ResNet-8 it is an accuracy-losing policy: with the improved pipeline, ResNet-8 test accuracy fell by
7.86 pp (CIFAR-10) and 4.77 pp (CIFAR-100) below dense. The optimiser did not fail; it optimised what the reward asked
for, whereas a weaker search had hidden the misalignment. This is a pruning instance of the observation that more
capable optimisers exploit misspecified objectives more strongly
[@pan2022misspecification; @skalse2022rewardgaming]. Search performance must therefore be judged together with
objective quality, and a pruning reward must encode the acceptable accuracy–compression boundary explicitly, because a
trade-off reward has no region in which accuracy loss becomes unacceptable.

**The accuracy-constrained reward as a correction (Section 5.8).** Phase 5 made accuracy a constraint rather than a
traded quantity, in the spirit of constrained reinforcement learning [@achiam2017cpo]. Comparing the constrained
reward with R0 under the same prior and archive (C1 vs C0) isolates the reward as the manipulated factor: the ResNet-8
accuracy loss fell from 7.32 to 1.18 pp (CIFAR-10) and from 4.05 to 1.16 pp (CIFAR-100), at markedly more moderate
sparsity. On SimpleCNN it was slightly *worse* than R0, LAMP remained more accurate in all six settings, and the
constraint held on test in only 67.5% of runs. A validation-based accuracy constraint is therefore an effective
**optimisation constraint**: it controls over-pruning during search. It is not a **generalisation guarantee**.

## 7.6 Validation-to-Test Transfer

**The gap (Section 5.9).** 119 of 120 constrained policies passed the constraint on the selection split, but only 81
of 120 remained feasible on the test set, and in CIFAR-100 ResNet-8 none of 20 did. Three explanations are plausible,
although none was isolated by the experiments:
1. **Validation sampling variability.** The sampling noise of a 2,500-image validation split is of the same order as
   the 2% relative margin.
2. **Selection at the boundary.** The constrained reward drives the agent to the *most compressed* feasible policy,
   which sits close to the threshold by design.
3. **Architecture and dataset dependence.** The failures concentrated in the LeNet-5 settings and in CIFAR-100
   ResNet-8, where the feasible region is small and the dense accuracy is low.

Constraint-based pruning therefore needs margins, uncertainty-aware constraints or larger selection splits before a
validation-certified constraint can be read as a promise about unseen data. Equal recovery removed most of the gap for
this pipeline (Phase 6), but that is a property of recovery, not of the constraint.

## 7.7 Why LAMP Remains a Strong Benchmark, and Why Equal Fine-Tuning Changes the Rankings

### 7.7.1 LAMP

LAMP [@lee2021lamp], deterministic and search-free, was the strongest method before recovery throughout the core
framework. Its strength calibrates the thesis rather than invalidating it. One untested interpretation is that LAMP's
rescaled magnitude score, which sets layer-wise sparsity automatically [@lee2021lamp], already captures much of what a
learned allocation can achieve under an ordinary compression objective.

For simple compression objectives, a strong deterministic layer-adaptive baseline can therefore be difficult to beat,
and RL must justify its additional cost: repeated pruned-model evaluations (512 per run in this thesis), many seeds,
reward design, prior calibration, archive management and selection splits, against one pass over the weights for
LAMP. An adaptive search becomes the more natural tool when the objective includes accuracy constraints, deployment
metrics or several competing goals; the thesis does not show that this advantage materialises, and Phase 7B does not
prove it (Section 7.10).

### 7.7.2 Equal fine-tuning

**What changed (Section 5.10).** Phase 6 changed the comparison more than any other experiment. Before recovery, LAMP
led PPO by 0.26–0.53 pp and PPO led global magnitude by up to 6.51 pp. After an identical five-epoch, fixed-mask
recovery, the PPO–LAMP difference fell to 0.00–0.22 pp, within the pre-registered ±0.5 pp confidence-interval margin in
all six settings, and PPO's advantage over global magnitude fell to at most 0.12 pp. This is a decision rule, not a
formal equivalence test, and LAMP remained slightly but significantly ahead in three settings. The methods are
therefore not equivalent; their differences became small.

**One interpretation.** Much of the apparent advantage of one pruning rule over another reflects how much accuracy the
*initial* masked model preserves: recovery appears to compensate most where the initial loss was largest (global
magnitude recovered the most and LAMP the least). Because the methods differ only in their masks, with shared seeds
and data streams, the narrowing within each setting is an effect of equal recovery; the mechanism inside fine-tuning
is not isolated.

**This is the thesis's most important methodological implication for pruning comparisons.** Methods should be compared
after equal post-pruning recovery budgets, not only immediately after pruning; otherwise the comparison partly measures
how much each method damages the network before the damage is repaired. This is consistent with work
showing that retraining choices can change pruning conclusions [@renda2020rewinding; @le2021retraining;
@blalock2020state]. The claim holds for small CNNs at the moderate sparsities PPO selected under the accuracy
constraint (mean 15.6–56.9% in Phase 6), with a five-epoch budget. It does not extend to very high sparsity,
ImageNet-scale models, transformers or large models, where recovery may compensate far less.

## 7.8 Compression versus Real Deployment Efficiency

**Masked versus structured pruning.** Masked pruning suits search and analysis, because every policy is a masked copy
of the same network and recovery can preserve the mask exactly. But masked zeros stay inside dense tensors: without
sparse kernels, a masked model is not smaller or faster on a CPU [@hoefler2021sparsity], and the thesis makes no
latency or energy claim for its masked models. Only structured pruning, which changes tensor shapes, lets parameters,
MACs, latency and energy fall physically. It is coarser, however, and in Phase 7A the structured realisation of the
masked allocation cost several points of accuracy.

**Sparsity is not speed (Sections 6.4–6.5).** Paper 3's exhaustive tables showed that parameter reduction correlated
only weakly with measured CPU latency gain and MAC reduction strongly, and that in roughly a third of policy pairs or
more the more parameter-sparse policy was slower. In SimpleCNN the reason is arithmetic rather than inference: the
hidden fully connected layer holds most of the weights but very little of the computation, and the early convolutions
are the reverse. This is consistent with the observation that cost indicators can disagree [@dehghani2022misnomer]; the thesis
quantifies the disagreement exhaustively for small structured CNNs on one CPU.

**Accuracy-feasible headroom (Section 6.6).** Raw latency headroom existed, but under an accuracy constraint the useful
headroom shrank sharply. For ResNet-8, the best proxy-feasible candidate reached 1.098× and the best confirmed
five-epoch-feasible calibration policy reached 1.089×; neither satisfied the pre-registered 1.10× GO criterion, and the
1.098× candidate was never confirmed feasible. For SimpleCNN none could be established, because its accuracy
proxy failed validation. **Unconstrained speed-ups say little about the speed-ups available to a
model that must keep its accuracy.**

## 7.9 Accuracy–Efficiency Trade-offs and the Three Levels of Efficiency

The deployment results are best read through the three levels of Chapter 3: compression and memory (Level 1),
computation and latency (Level 2), and measured operational energy (Level 3). **Each level needs its own evidence:**
1. **Memory does not guarantee latency.** In Phase 7A, the SimpleCNN PPO model removed 80.1% of the parameters but was
   slower than a uniform model that removed far fewer, and Paper 3 shows the same weak link across 5,184 shapes.
2. **MACs predict latency better, but do not replace measurement.** The MAC–latency correlation is strong but below 1,
   and at nearly equal MACs Uniform was still faster than PPO in Phase 7A.
3. **Latency does not mathematically guarantee energy.** Energy is power times time, so a faster model saves energy
   only if its average power does not rise to compensate. In Phase 7A average package power was nearly the same for
   every model, so energy savings followed latency savings; that is an observation about this platform, not a law.

**Accuracy must be reported at every level.** The pre-registered criteria of Phases 7A and 7B made this explicit by
requiring accuracy within 0.5 pp of dense. **No pruned model met that requirement in either phase, except
LAMP-Structured in Phase 7B, whose energy could not be measured.** Every deployment saving in the thesis therefore
comes with an accuracy cost, and is reported with it.

## 7.10 Interpreting the Sustainability Experiments

**Phase 7A: real energy savings, but not a better model (Sections 6.7–6.8).** Phase 7A firmly establishes that
physically structured pruning reduced **directly measured** CPU package energy per inference, in all five sessions.
But energy savings alone are not evidence of a better model: the SimpleCNN PPO model lost 5.08 pp of VAL-RL accuracy,
and a MAC-matched uniform model was both more accurate and lower in energy. It shows that structural compression reaches the hardware, not
successful sustainability-oriented optimisation by PPO, and that a transferred masked allocation is a poor structured
allocation, plausibly because the masked policy concentrated
pruning in the hidden layer, where it costs little accuracy *as a mask* but is neither accurate nor computationally
efficient as channel removal (consistent with Paper 3, not isolated).

**Phase 7B: what improved (Sections 6.10–6.12).** Phase 7B used a **deployment-oriented structured PPO**: its actions
were physical channel ratios, and its search was guided by a validated energy surrogate rather than by measured
energy. It substantially improved the memory–computation–accuracy trade-off: parameter memory
fell by 63.4% and MACs by 36.0%, and the selection-split accuracy loss was 1.12 pp, against 4.00 pp for the old policy
on the same split, supporting the design lesson that **an allocation should be
learned in the space where it will be deployed.**

**Phase 7B: what it did not solve.** It lost more than the 0.5 pp the pre-registration allowed, and 0 of 20 seeds met
the joint deployment-efficiency criterion. Its latency was not measured, and its energy is unresolved. Against the
MAC-matched uniform and parameter-matched LAMP structured baselines it showed no clear advantage overall (Section
6.12). Phase 7B improved part of the deployment objective; it did not solve the full objective.

**The energy surrogate.** The surrogate ranked calibration policies very well (Spearman 0.996), which made the search
affordable. But surrogate validity does not turn predicted energy into measured energy: its error for a particular
final model is unknown, and the final model's energy was never measured. Surrogates can reduce search cost; final
deployment claims require direct validation on the final model.

**The measurement failure (Section 6.13).** Phase 7B's energy remained unresolved because the measurement setup
repeatedly failed its pre-registered stability check (3% limit), while latency stayed stable within about one percent;
the thesis does not claim to know the cause. In the present setup, package-energy readings were therefore
substantially less stable than latency measurements; the thesis does not generalise this beyond the tested system.
Because the same counter on the same machine had passed its check in Phase 7A, qualification must be repeated for
every measurement campaign. This is a limitation of the experimental environment. **It is evidence neither for nor
against the Phase-7B model's energy use**: no model comparison was made, and only the dense model was measured. Studies
of software energy meters likewise report that readings need careful protocol design and validation [@khan2018rapl; @jay2023powermeters; @raffin2024dissecting].

**On sustainability wording.** The thesis does not establish that any of its models is environmentally beneficial,
lower in carbon footprint or suitable for "green" deployment. It establishes measured operational CPU energy savings
for the Phase-7A structured models, with their accuracy costs, and, for Phase 7B, memory and computation reductions
with energy unresolved. Lifecycle, training-energy and carbon accounting were outside its scope.

## 7.11 What the Negative Results Establish

The negative and partial results, which carry much of the thesis's methodological weight, are summarised in
Table 7.1.

**Table 7.1. Negative and partial results and what each establishes.**

| Negative or partial result | What it establishes |
|---|---|
| PPO did not beat LAMP before recovery (5 of 6, then 6 of 6 settings) | RL must justify its cost through flexibility or richer objectives |
| Sensitivity did not improve final accuracy in any phase | Search improvement and outcome improvement must be evaluated separately |
| Archive selection had mixed test effects | Retention is not generalisation; it amplifies whatever the reward specifies |
| Validation feasibility transferred to test in 67.5% of runs | An optimisation constraint is not a generalisation guarantee |
| Hardware-aware RL was NO-GO | Accuracy-feasible latency room can be too small to justify a latency-driven search |
| Uniform beat PPO on SimpleCNN in Phase 7A | A transferred masked allocation is a poor structured allocation; deployment metrics can reverse rankings |
| Phase 7B missed the 0.5 pp criterion (0 of 20 seeds) | Learning in the deployment space narrows, but does not close, the accuracy gap at these compression levels |
| Phase-7B energy remained unresolved | Energy claims require a qualified instrument; MAC reductions and surrogates do not substitute for measurement |

**Each of these results came from an experiment whose decision criteria were fixed in advance, and each was kept as
observed.** Several changed the design of the next phase: retention led to the archive, over-pruning to the
constraint, and the lossy transfer to Phase 7B. The positive findings are credible partly *because* the negative ones
are visible; that is a logical point about evidential weight, not an empirical causal claim.

## 7.12 Generalisability

**The evidence base.** The core framework used CIFAR-10 and CIFAR-100, three small CNNs, four pruning units per
network and an action space of 1,296 policies; the deployment extension used a subset of these settings on one CPU
platform (an Intel Core i7-1260P laptop, batch 1, one thread).

**Within that scope**, the main qualitative results recur across both datasets and all three architectures; the
deployment findings are specific to one platform. **Outside
that scope** the thesis provides no evidence, whether for ImageNet-scale data, larger CNNs, transformers or large
language models, other hardware, or other batch sizes or thread counts. At larger scale the policy space can no longer
be enumerated, so retention and search quality cannot be measured against ground truth and the case for a learned
search may become stronger; recovery may compensate less at high sparsity; and latency and energy relations are
hardware-specific. Larger-scale and multi-platform validation is future work (Chapter 8).

## 7.13 Limitations

The limitations below bound every claim in Chapters 5–7. They are grouped by kind.

**A. Model and dataset scale.** CIFAR-scale data and small CNNs only. Four pruning units and six ratios per network:
the space was small enough to enumerate, which made exhaustive ground truth possible but makes the RL problem easier
than at realistic scale. The sustainability phases used CIFAR-10 only; Phase 7B, SimpleCNN only.

**B. Optimisation.** PPO's search cost is substantial. Sensitivity guidance improved exploration but not final
accuracy, and one Phase-4 exploration measure reversed. LAMP remained stronger before recovery and slightly ahead in
three settings after it. The results depend on reward design: no
Phase-4 redesigned reward met its criterion, and the constrained reward made SimpleCNN slightly worse. Elite
behavioural cloning was tested at one configuration only.

**C. Validation methodology.** Validation-to-test transfer of the accuracy constraint was imperfect (67.5% test
feasibility, 0% in CIFAR-100 ResNet-8), and the fixed relative threshold (τ = 0.98) and the small selection splits make
feasibility sensitive to sampling noise. The deployment phases used a single recovery seed per model. One dense
reference (CIFAR-10 SimpleCNN) used an older training recipe. Phase 6 was not a fully independent test confirmation,
since the test sets had been read for earlier frozen models. The ±0.5 pp rule is a confidence-interval decision rule,
not a formal equivalence test. Recovery used one fixed five-epoch protocol; longer or tuned recovery might change the
rankings.

**D. Hardware.** One CPU platform, one batch size and one thread configuration; no GPU, mobile, embedded or
accelerator evaluation. The SimpleCNN accuracy proxy failed validation, so its accuracy-feasible latency headroom is
unknown, and the exhaustive 5-epoch recovered landscape was stopped for compute cost.

**E. Energy.** CPU package energy only, with no DRAM, whole-system or external power measurement. The Phase-7B
measurement setup failed its stability qualification repeatedly, so no Phase-7B energy or latency claim is made. The
energy surrogate and the accuracy proxy are model-selection tools, not measurements.

**F. Sustainability.** No carbon accounting, no lifecycle or embodied-energy assessment, and no analysis of training
or search energy. Phase-7A deployment-scale energy figures are projections, not results.

## 7.14 Overall Scientific Contribution

**The central contribution.** The novelty of this thesis does not lie in a claim that PPO beats existing pruning
methods; it does not. It lies in **an integrated, systematically audited diagnostic framework, with pre-specified and
pre-registered decision criteria across its later experimental stages**. It follows one pruning approach from
allocation to measured hardware energy, isolating a separate question at each stage (Table 7.2).

**Table 7.2. Stages of the diagnostic framework, the question each isolates, and the thesis evidence.**

| Stage | Question | Thesis evidence | Research question |
|---|---|---|---|
| 1. Adaptive layer-wise allocation | Can a learned agent decide *where* to prune? | Yes: far better than uniform or random; not better than LAMP | RQ1 |
| 2. Sensitivity-guided exploration | Does layer sensitivity help the search? | Search yes, outcome no | RQ2 |
| 3. Policy retention | Does the agent keep what it finds? | No, unless an archive retains it (validation ranking) | RQ3 |
| 4. Reward alignment | Does the reward encode the desired trade-off? | A trade-off reward over-prunes; a constraint controls it, without a generalisation guarantee | RQ4 |
| 5. Equal recovery | Do method differences survive identical fine-tuning? | Largely not, at these sparsities | RQ5 |
| 6. Deployment translation | Does compression become latency? | Through MACs and structure, not through parameter count | RQ6 |
| 7. Direct energy validation | Does it become measured energy, at what accuracy cost? | Yes in Phase 7A, with material accuracy loss; unresolved in Phase 7B | RQ6 |

The six research questions therefore form a chain rather than a list of independent results: each answer changed
what the next question had to control for, from learnability (RQ1) through guidance, retention, reward and recovery to
re-measurement at each hardware level (RQ6). Effective pruning is **not a single optimisation problem** but a chain of
coupled problems, and a weakness at any link can hide or reverse the effect of improvements elsewhere: a good search
under a misaligned reward produced worse models; a retained policy did not generalise; an allocation that was good as
a mask was poor as a structure; and a strong MAC reduction remained unconfirmed as an energy saving. **This
multi-stage view, supported at each stage by experiments whose decision criteria were fixed in advance, and reported
including their negative results, is the thesis's central scientific contribution.** Its practical implications for
pruning research are drawn together in Section 8.4.

## 7.15 Chapter Summary

PPO contributes a flexible, learnable layer-wise allocation that clearly beats uniform and random allocation but not
the strong deterministic LAMP baseline; its value lies in adaptability to changing objectives rather than universal
superiority. Sensitivity acts as a search bias that improves exploration but not the final model. An external archive
restores validation-rank retention without improving generalisation. A stronger optimiser exposed a misaligned trade-off reward; an accuracy constraint controlled
over-pruning but did not guarantee test feasibility. Equal recovery narrowed method differences to small values. In
deployment, sparsity is not speed, accuracy-feasible headroom is much smaller than unconstrained headroom, Phase 7A's
measured energy savings carried a material accuracy cost, and the deployment-oriented structured PPO of Phase 7B
improved the memory–computation–accuracy trade-off but missed the accuracy criterion, with its energy unresolved.
Chapter 8 concludes the thesis and sets out the future work these findings motivate.
