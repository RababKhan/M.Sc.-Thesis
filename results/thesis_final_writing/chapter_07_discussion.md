# Chapter 7 — Discussion

## 7.1 Chapter Overview

Chapters 5 and 6 reported what happened:
- in the core pruning experiments (Phases 3–6);
- when the pruned models were realised and measured on hardware (Paper 3, Phases 7A and 7B, and the energy
  confirmation study).

This chapter asks what those results mean. It interprets the evidence across phases, places it against the
literature reviewed in Chapter 2, and states where the proposed framework succeeds and where it does not.

**Ground rules:**
- **No new numbers.** Every number used here was reported and audited in Chapters 5 and 6, and the discussion adds no
  new measurement.
- **Causal wording only where earned.** Causal language is reserved for effects the experimental design isolated,
  such as the C1-versus-C2 contrast of Phase 5, or for relations that hold by construction.
- **Hedged wording elsewhere.** Where only an association was observed, the text says so ("is consistent with", "may
  reflect").

The discussion proceeds in the order in which the thesis encountered its problems:
- what the RL agent contributes (Sections 7.2–7.3);
- retention, reward design and recovery (Sections 7.4–7.7);
- deployment and measured energy (Sections 7.8–7.11);
- generalisability, limitations and the thesis's overall contribution (Sections 7.12–7.15).

## 7.2 What Adaptive Layer-Wise PPO Actually Contributes

**The evidence.**
- **Allocations.** Across six architecture–dataset settings, the PPO agent learned allocations that were visibly
  non-uniform. It pruned large, low-sensitivity layers heavily and kept small early layers almost intact
  (Figure 5.2).
- **Against simple allocation.** At the sparsity it chose itself, these allocations were far more accurate than
  uniform allocation (STRONG, 5 of 6 settings, up to about +31.4 pp on CIFAR-10 SimpleCNN) and random allocation
  (6 of 6).
- **Against magnitude criteria.** Against global magnitude and ERK the picture was mixed (MODERATE and LIMITED).
- **Against LAMP.** LAMP was significantly more accurate than PPO before recovery in 5 of 6 settings in Phase 3, and
  in 6 of 6 for the later pipelines of Phases 4 and 5.

**What this establishes.**
- **The agent solves the allocation problem it was given.** It discovers that layers should be pruned differently,
  and it does so without any hand-written allocation rule. In this sense the thesis confirms the premise behind
  RL-based compression [@he2018amc]: layer-wise allocation is learnable from a scalar reward.
- **It does not establish superiority.** The thesis does not show that a learned allocation is better than the best
  deterministic allocation rule. The comparison with LAMP, a cheap and deterministic method, points the other way at
  the sparsities studied.

**Where the agent's value lies.**
- **Not universal superiority.** One interpretation, supported by the later phases, is that PPO's value lies in its
  *flexibility*: it optimises whatever objective it is given.
- **The phases changed the objective repeatedly, and the agent adapted:**
  - Phase 4 changed the prior;
  - Phase 5 changed the reward to a constraint;
  - Phase 7B changed the action space to physical channels and the objective to a deployment surrogate.

  The agent and its training procedure stayed the same. That property is not shared by a fixed rule such as LAMP,
  which optimises one implicit criterion.
- **The evidence for flexibility is indirect.** The thesis shows that PPO *can* be pointed at a new objective and will
  pursue it. In Phase 5 this paid off for ResNet-8. The thesis does not show that this flexibility yields a better
  final model than a tailored deterministic method for any objective tested here.

**Implication.** RL pruning should be presented as an adaptive optimisation mechanism for objectives that are hard to
express as a single saliency score, not as a replacement for strong deterministic pruning when the objective is
ordinary magnitude-based compression. Nothing in the thesis implies that PPO is necessary for pruning in general.

## 7.3 The Role of Sensitivity Information

**The pattern.** Across three phases, sensitivity guidance improved the *search*, but not the *final model*:
- **Phase 3, prior against no prior:**
  - better-ranked policies were sampled earlier, with fewer destructive policies (STRONG on three secondary search
    endpoints);
  - the primary accuracy-AUC endpoint improved in 3 of 6 settings;
  - no final-accuracy effect was found in any setting.
- **Phase 4, centred prior:** it beat both its shuffled control and a magnitude-matched conservative control on
  reward-rank AUC (STRONG), and beat the sign-reversed prior in every setting. It did not improve the final rank.
- **Phase 5, under the corrected reward:** the centred prior again improved exploration in 5 of 6 settings, with none
  of 24 primary tests opposite. Yet C1's test accuracy was 0.05–0.16 pp *below* C2's in every setting (descriptive).

**The strongest defensible reading.** Sensitivity is useful as a **search bias**. Its measurable contribution lies
primarily in exploration guidance, not in predicting which final model will be most accurate.

The pattern has a natural explanation that the thesis can describe but not prove:
- **What the prior acts on.** It changes the probability of each action. It pushes the agent away from aggressive
  ratios on sensitive units and, once centred, towards them on insensitive units. That shapes *which* policies are
  sampled early.
- **What decides the outcome.** The returned policy is decided by the reward landscape and the selection rule. In
  these small spaces, where every setting's best-seen policy already had a median rank of 1–4.5, the search finds
  near-optimal regions with or without the prior. A better route to the same destination need not change the
  destination.

**Why centring mattered.** The Phase-3 controls carried an important warning:
- **The LeNet-5 controls.** The shuffled and constant priors did as well as the correct one there.
- **A shared pressure.** All three Phase-3 priors share the same total pressure towards lighter pruning. Part of any
  benefit could therefore be generic conservatism rather than layer-specific knowledge.
- **What centring separates.** It distinguishes "more sensitive than average" from "less sensitive than average", and
  removes the net conservative pressure by construction.

What Phase 4 then showed is more informative than a simple success:
- **The layer-specific effect survives centring.** The centred prior's advantage over permuted and
  magnitude-matched controls on reward-rank AUC shows that the effect is not only generic conservatism.
- **The reversed measure.** On the accuracy-based exploration measure, the *conservative* control was significantly
  better than the centred prior in 4 of 6 settings.
  - This is consistent with how the two priors act. The centred prior pushes aggressive actions onto insensitive
    units, which can raise the reward rank of sampled policies while lowering their average accuracy.
  - Reward rank and accuracy are different targets under a trade-off reward. Centring made the prior more
    layer-specific and more interpretable; it did not make it a better predictor of final accuracy.

**Implication for the literature.** Sensitivity is often treated as a direct proxy for how much a layer can be
pruned. The dissociation found here suggests evaluating sensitivity-based methods on two separate axes:
- does the information improve the *search*?
- does it improve the *outcome*?

The thesis found the first, but not the second. This does not mean sensitivity "does not work". It means its
contribution is of a different kind than often assumed.

## 7.4 Search Quality versus Policy Retention

**The finding.** One of the clearest results of the thesis is that **finding a good policy and ending on it are
different problems.**
- **Finding.** In Phase 4, PPO's best visited policy reached the top 10 of 1,296 in 95–100% of runs, with a median
  rank of 1–2.5.
- **Ending.** The terminal policy reached the top 10 in 0% of runs in five settings and 5% in the sixth (median rank
  65–173). The same pattern was present in Phase 3.
- **Retaining.** Archive selection restored retention on validation ranking: 80–100% top-10 reach and 100% top-20
  reach for the final Phase-4 pipeline.

**A possible mechanism (not isolated).** A stochastic policy-gradient method updates its distribution on noisy,
high-variance estimates. It may sample an excellent policy early and then settle on a region of high average reward
that does not contain it. The thesis observed the gap; it did not isolate its mechanism.

**What the finding establishes is methodological.** A study that reports only the terminal policy of an RL pruning
agent measures retention together with search, and can mistake one for the other. Had the thesis reported only
terminal policies, it would have concluded that PPO searches poorly. The exhaustive landscapes showed instead that it
searches well and retains poorly.

This mirrors observations in neighbouring fields:
- the gap between the search phase and the final selection in architecture search [@yu2020evalnas];
- the value of archives of visited states in exploration [@ecoffet2021goexplore].

**Recommendation.** RL pruning studies should report, separately:
- the best visited policy;
- the retained (terminal or selected) policy;
- the selection procedure that links them.

**Elite behavioural cloning.** Phase 4 also tested whether retention could be improved *inside* training, by adding
an imitation term towards the best policies seen so far (in the spirit of self-imitation learning [@oh2018sil]).
- **Results:** it improved the final-selected rank in only 3 of 6 settings and the best-seen rank in 1 of 6. The
  archive rule alone improved the selected rank in 6 of 6.
- **What this supports:** in these experiments, pulling the actor towards elite policies during training was not
  equivalent to keeping the elite candidates outside the agent and choosing among them afterwards.
- **What it does not support:** a claim that behavioural cloning is ineffective in general. Only one strength
  (η = 0.1) and one elite size (K = 5) were tested, and the term was dropped after Phase 4.

**The boundary.** Archive selection improves *validation-rank* retention. Its effect on test accuracy was mixed:
- small gains in four settings;
- losses of 5.66 and 4.00 pp on ResNet-8.

The archive retained exactly what the reward ranked highest, and on ResNet-8 that was an over-pruned model. Retention
amplifies whatever the reward specifies, which leads directly to the next section. Archive selection is therefore a
retention mechanism, not a generalisation mechanism.

## 7.5 Reward Design and Over-Pruning

**A stronger optimiser exposes the reward.** Phase 4 provides the thesis's clearest reinforcement-learning lesson.
- **What changed.** Once search and retention improved, the agent reliably reached the optimum of the reward R0.
- **Where that optimum lay.** R0's optimum is set by its fixed exchange rate between accuracy and sparsity, and lies
  at 23.5–56.4% sparsity depending on the setting.
- **What it cost on ResNet-8.** The optimum is an accuracy-losing policy. With the improved pipeline, ResNet-8
  sparsity rose and test accuracy fell by 7.86 pp (CIFAR-10) and 4.77 pp (CIFAR-100) below dense.

**The optimiser did not fail.** It optimised what the reward asked for. A weaker search had hidden the misalignment by
rarely reaching the reward's optimum. This is a pruning instance of a general observation: more capable optimisers
exploit misspecified objectives more strongly [@pan2022misspecification; @skalse2022rewardgaming].

**The practical consequences:**
1. Search performance must be judged together with objective quality. A better rank under a misaligned reward can be
   a worse model.
2. A pruning reward must encode the acceptable accuracy–compression boundary explicitly. A trade-off reward has no
   region in which accuracy loss becomes unacceptable, so on fragile architectures its optimum can be a model no
   practitioner would deploy.

**The accuracy-constrained reward as a correction.** Phase 5 made accuracy a constraint rather than a traded quantity,
in the spirit of constrained reinforcement learning [@achiam2017cpo].
- **Effect on ResNet-8.** Comparing the constrained reward with R0 under the same prior and archive (C1 vs C0)
  isolates the reward as the manipulated factor. The ResNet-8 accuracy loss fell from 7.32 to 1.18 pp (CIFAR-10) and
  from 4.05 to 1.16 pp (CIFAR-100), at markedly more moderate sparsity.
- **Effect elsewhere.** On SimpleCNN the constrained reward was slightly *worse* than R0. There the agent used the 2%
  allowance for a little more sparsity.
- **LAMP was still stronger.** It remained more accurate than the constrained pipeline in all six settings.
- **The constraint did not transfer fully.** It held on test in only 67.5% of runs.

The evidence therefore supports a clear separation. A validation-based accuracy constraint is an effective
**optimisation constraint**: it controls over-pruning during search. It is not a **generalisation guarantee**.

## 7.6 Validation-to-Test Transfer

The gap is large enough to discuss on its own:
- 119 of 120 constrained policies passed the constraint on the selection split, but only 81 of 120 remained feasible on
  the test set;
- in CIFAR-100 ResNet-8, none of 20 did.

**Possible explanations** (none isolated by the experiments):
1. **Validation sampling variability.** Each validation split has 2,500 images. Its sampling noise is of the same order
   as the 2% relative margin, so a policy that passes by a small margin on validation can fail by a small margin on
   new data.
2. **Selection at the boundary.** The constrained reward drives the agent to the *most compressed* feasible policy, so
   the selected policies sit close to the threshold by design. A tight threshold then turns small accuracy
   differences into binary feasibility failures.
3. **Architecture and dataset dependence.** The failures concentrated in the LeNet-5 settings and in CIFAR-100
   ResNet-8, where the feasible region is small and the dense accuracy is low.

**Implication.** Constraint-based pruning needs either margins, uncertainty-aware constraints or larger selection
splits before a validation-certified constraint can be read as a promise about unseen data. Phase 6 showed that equal
recovery removed most of the gap for this pipeline. That is a property of recovery, not of the constraint.

## 7.7 Why LAMP Remains a Strong Benchmark, and Why Equal Fine-Tuning Changes the Rankings

### 7.7.1 LAMP

LAMP [@lee2021lamp] was the strongest method before recovery throughout the core framework. It is deterministic and
requires no search. Its strength need not invalidate the thesis; it calibrates it.

One interpretation is that LAMP's layer-adaptive score already captures much of what a learned allocation can achieve
under an ordinary objective. That objective is to remove as many small-magnitude weights as possible while keeping
the network's function. Its rescaled magnitude score sets layer-wise sparsity automatically [@lee2021lamp], which is
the decision a sensitivity-guided search tries to learn. The thesis did not test this explanation.

**The lesson about cost.** For simple compression objectives, a strong deterministic layer-adaptive baseline can be
difficult to beat, and RL must justify its additional cost:
- **PPO's cost:** repeated pruned-model evaluations (512 per run in this thesis); many seeds; reward design; prior
  calibration; archive management; selection splits.
- **LAMP's cost:** one pass over the weights.

**Where an adaptive search may become useful.** When the objective is simple, deterministic pruning is likely
preferable. When the objective includes accuracy constraints, layer sensitivity, deployment metrics or several
competing goals, an adaptive search becomes the more natural tool. The thesis does not show that this future
advantage materialises, and Phase 7B does not prove it (Section 7.10).

### 7.7.2 Equal fine-tuning

**What changed.** Phase 6 changed the comparison more than any other experiment:
- **Before recovery,** differences between methods were clearly visible: LAMP led PPO by 0.26–0.53 pp, and PPO led
  global magnitude by up to 6.51 pp.
- **After an identical five-epoch, fixed-mask recovery,** the PPO–LAMP difference fell to 0.00–0.22 pp, and PPO's
  advantage over global magnitude to at most 0.12 pp.
- **Every pruned model recovered,** and the post-recovery PPO–LAMP differences fell within the pre-registered ±0.5 pp
  confidence-interval margin in all six settings.

This is a decision rule, not a formal equivalence test, and LAMP remained slightly but significantly ahead in three
settings. The methods are therefore not equivalent. Their differences became small.

**One interpretation.** Much of the apparent advantage of one pruning rule over another reflects how much accuracy
the *initial* masked model preserves before retraining. Recovery appears to compensate most where the initial loss was
largest: global magnitude recovered the most and LAMP the least.

The Phase-6 design isolates the recovery budget. Methods differ only in their masks; seeds and data streams are
shared. So within each setting the narrowing of the gaps is an effect of equal recovery. The exact mechanism inside
fine-tuning is not isolated, and the interpretation above remains an interpretation.

**This is the thesis's most important methodological implication for pruning comparisons.**
- **Compare after equal recovery.** Methods should be compared after equal post-pruning recovery budgets, and not only
  immediately after pruning. Otherwise the comparison is partly a comparison of how much each method damages the
  network before the damage is repaired.
- **Agreement with the literature.** This is consistent with work showing that retraining choices can change pruning
  conclusions [@renda2020rewinding; @le2021retraining; @blalock2020state].

**The boundary of the claim.** It holds for small CNNs at the moderate sparsities PPO selected under the accuracy
constraint (mean 15.6–56.9% in Phase 6), with a five-epoch budget. It does not extend to very high sparsity, ImageNet-scale models,
transformers or large models, where recovery may compensate far less.

## 7.8 Compression versus Real Deployment Efficiency

**Masked versus structured pruning.** The core framework's masked pruning and the extension's structured pruning
serve different purposes.
- **Masked pruning** is well suited to search and analysis. Every policy is a masked copy of the same network, the
  full policy space can be evaluated, and recovery can preserve the mask exactly. But masked zeros stay inside dense
  tensors. Without sparse kernels, a masked model is not smaller or faster on a CPU [@hoefler2021sparsity]. The
  thesis makes no latency or energy claim for its masked models.
- **Structured pruning** changes tensor shapes. Only then can parameters, MACs, latency and energy fall physically.
  Structured pruning is not universally better, however. Removing whole channels is coarser than removing weights,
  and in Phase 7A the structured realisation of the masked allocation cost several points of accuracy.

**Sparsity is not speed.** Paper 3's exhaustive tables give a clear deployment lesson:
- **Parameter reduction** correlated only weakly with measured CPU latency gain (Spearman 0.31–0.55).
- **MAC reduction** correlated strongly (0.89–0.98).
- **Inversions are common.** In roughly a third of policy pairs or more, the more parameter-sparse policy was slower.

The SimpleCNN example shows the reason, which here is arithmetic rather than inference. In that network:
- the hidden fully connected layer holds most of the weights but very little of the computation;
- the early convolutions are the reverse.

So removing parameters from low-compute layers can shrink the model a great deal while barely changing its runtime,
and pruning expensive convolutional paths can deliver a large speed-up with few parameters removed. This is
consistent with the broader observation that cost indicators can disagree [@dehghani2022misnomer]. The thesis
quantifies the disagreement exhaustively for small structured CNNs on one CPU.

**Accuracy-feasible headroom.** Raw latency headroom existed: the most aggressive structured policies were up to 2.61×
faster on SimpleCNN. Under an accuracy constraint, the useful headroom shrank sharply:
- for ResNet-8, the best proxy-feasible candidate reached 1.098×;
- the best confirmed five-epoch-feasible calibration policy reached 1.089×;
- neither satisfied the pre-registered 1.10× GO criterion, and the 1.098× candidate was never confirmed feasible.

For SimpleCNN no accuracy-feasible headroom could be established at all, because its accuracy proxy failed
validation. **Unconstrained speed-ups say little about the speed-ups available to a model that must keep its
accuracy.**

## 7.9 Accuracy–Efficiency Trade-offs and the Three Levels of Efficiency

The deployment results are best read through the three levels of Chapter 3:
- **Level 1:** compression and memory;
- **Level 2:** computation and latency;
- **Level 3:** measured operational energy.

**Each level needs its own evidence:**
1. **Memory does not guarantee latency.**
   - In Phase 7A, the SimpleCNN PPO model removed 80.1% of the parameters but was slower than a uniform model that
     removed far fewer.
   - Paper 3 shows the same weak link across 5,184 shapes.
2. **MACs predict latency better, but do not replace measurement.**
   - The MAC–latency correlation is strong but below 1.
   - At nearly equal MACs, Uniform was still faster than PPO in Phase 7A.
3. **Latency does not mathematically guarantee energy.** Energy is power times time, so a faster model saves energy
   only if its average power does not rise to compensate.
   - In Phase 7A, average package power was nearly the same for every model, so energy savings followed latency
     savings.
   - That is an observation about this platform and workload, not a general law.

**Accuracy must be reported at every level.** A deployment saving obtained with a material accuracy loss is a
different result from a saving at preserved accuracy. The pre-registered criteria of Phases 7A and 7B made this
explicit by requiring accuracy within 0.5 pp of dense.

**No pruned model met that requirement in either phase, except LAMP-Structured in Phase 7B, whose energy could not be
measured.** Every deployment saving in the thesis therefore comes with an accuracy cost, and is reported with it.

## 7.10 Interpreting the Sustainability Experiments

**Phase 7A: real energy savings, but not a better model.** Phase 7A establishes one positive fact firmly:
- physically structured pruning reduced **directly measured** CPU package energy per inference;
- every pruned model used less energy than dense in all five sessions.

The learned allocation's structured form saved 22.8% (SimpleCNN) and 13.4% (ResNet-8).

**But energy savings alone are not evidence of a better model:**
- **The accuracy cost was large.** The SimpleCNN PPO model lost 5.08 pp of validation accuracy.
- **A simpler method did better.** A MAC-matched uniform model was both more accurate and lower in energy.

Phase 7A therefore does not show successful sustainability-oriented optimisation by PPO. It shows that structural
compression reaches the hardware, and that a transferred masked allocation is a poor structured allocation.

The most plausible reading, consistent with Paper 3 but not isolated, is that:
- the masked policy concentrated pruning in the hidden layer, where it costs little accuracy *as a mask*;
- as channel removal followed by recovery, that allocation was neither accurate nor computationally efficient;
- the ResNet-8 mapping was additionally approximate.

**Phase 7B: what improved.** Phase 7B is described here as a **deployment-oriented structured PPO**: its actions were
physical channel ratios, and its reward used a validated energy surrogate. Relative to the transferred policy, it
substantially improved the memory–computation–accuracy trade-off:
- parameter memory fell by 63.4% and MACs by 36.0%;
- the selection-split accuracy loss was 1.12 pp, against 4.00 pp for the old policy on the same split.

This is evidence for the design lesson of Section 7.8: **an allocation should be learned in the space where it will be
deployed.**

**Phase 7B: what it did not solve:**
- **Accuracy criterion missed.** It lost more than the 0.5 pp the pre-registration allowed, and 0 of 20 seeds met the
  joint deployment-efficiency criterion.
- **No measured latency.** Latency was not measured.
- **No measured energy.** Energy is unresolved.
- **No advantage over the baselines.** Against the MAC-matched uniform and parameter-matched LAMP structured
  baselines it showed no clear advantage overall:
  - roughly equal accuracy to Uniform, with lower memory;
  - lower accuracy than LAMP, with fewer MACs.

Phase 7B improved part of the deployment objective. It did not solve the full objective.

**The energy surrogate.**
- **Why it was useful.** It ranked policies very well (Spearman 0.996) with a small median error (1.05%), which made
  energy-aware search affordable.
- **What it cannot be.** Surrogate validity does not turn predicted energy into measured energy. The surrogate was
  validated on calibration policies. Its error for a particular final model is unknown, and the final model's energy
  was never measured.
- **The principle this illustrates:** surrogates can reduce search cost, but final deployment claims require direct
  validation on the final model.

**The measurement failure.** Phase 7B's energy remained unresolved, because the measurement setup repeatedly failed its
pre-registered stability check:
- the first pilot gave a coefficient of variation of 11.36%;
- the permitted rerun gave 4.01%;
- the separate confirmation study gave 19.92%;
- the limit was 3% throughout.

**What was observed during qualification.** Latency stayed stable within about one percent, while package power varied
substantially. The thesis does not claim to know the cause.

**The interpretation it supports** is limited but useful:
- On commodity laptop hardware, package energy measurement appears much more sensitive to system state than latency
  measurement.
- The same counter, on the same machine, had passed its stability check in Phase 7A, so qualification must be
  repeated for every measurement campaign.
- This is a limitation of the experimental environment. **It is evidence neither for nor against the Phase-7B model's
  energy use.** No model comparison was made, and only the dense model was measured.
- This agrees with studies of software energy meters, which report that their readings need careful protocol design
  and validation [@khan2018rapl; @jay2023powermeters; @raffin2024dissecting].

**On sustainability wording.** The thesis does not establish that any of its models is environmentally beneficial,
lower in carbon footprint or suitable for "green" deployment. It establishes measured operational CPU energy savings
for the Phase-7A structured models, with their accuracy costs. For Phase 7B it establishes memory and computation
reductions, with energy unresolved. Lifecycle, training-energy and carbon accounting were outside its scope.

## 7.11 What the Negative Results Establish

The thesis reports several negative or partial results. They are summarised here because they carry much of its
methodological weight.

| Negative or partial result | What it establishes |
|---|---|
| PPO did not beat LAMP before recovery (5 of 6, then 6 of 6 settings) | RL must justify its cost through flexibility or richer objectives, not small accuracy gains under ordinary compression |
| Sensitivity did not improve final accuracy in any phase | Search improvement and outcome improvement must be evaluated separately |
| Archive selection had mixed test effects | Retention is not generalisation; it amplifies whatever the reward specifies |
| Validation feasibility transferred to test in 67.5% of runs | An optimisation constraint is not a generalisation guarantee |
| Hardware-aware RL was NO-GO | Accuracy-feasible latency room can be too small to justify a latency-driven search |
| Uniform beat PPO on SimpleCNN in Phase 7A | A transferred masked allocation is not a good structured allocation; deployment metrics can reverse method rankings |
| Phase 7B missed the 0.5 pp criterion (0 of 20 seeds) | Learning in the deployment space narrows, but does not close, the accuracy gap at these compression levels |
| Phase-7B energy remained unresolved | Energy claims require a qualified instrument; strong MAC reductions and accurate surrogates do not substitute for measurement |

**Each of these results was pre-registered as a possible outcome and kept as observed.** Several of them changed the
design of the next phase:
- retention led to the archive;
- over-pruning led to the constraint;
- the lossy transfer led to Phase 7B.

This sequence is the evidence that the thesis's conclusions are not tuned to a favourable result. The positive
findings are credible partly *because* the negative ones are visible. That is a logical point about evidential
weight, not an empirical causal claim.

## 7.12 Generalisability

**The evidence base:**
- **Core framework:** CIFAR-10 and CIFAR-100; three small CNNs (SimpleCNN, a CIFAR LeNet-5 and ResNet-8); four
  pruning units per network; one laptop-scale action space of 1,296 policies.
- **Deployment extension:** a subset of these settings, on one CPU platform (an Intel Core i7-1260P laptop, batch 1,
  one thread).

**Within that scope**, the core findings show some robustness. The main qualitative results recur across both
datasets and all three architectures, with architecture-dependent sizes:
- PPO beats Uniform and Random;
- LAMP is stronger before recovery;
- sensitivity helps search but not outcome;
- the retention gap;
- the narrowing under equal recovery.

The deployment findings are specific to that one platform and run configuration.

**Outside that scope** the thesis provides no evidence. It does not support generalisation to:
- ImageNet-scale data;
- larger CNNs, transformers or large language models;
- GPUs, mobile processors, edge accelerators or cloud inference;
- other batch sizes or thread counts.

**What could change at larger scale.** Several findings may change, in either direction:
- **The policy space** can no longer be enumerated. Retention and search quality can then not be measured against
  ground truth, and the case for a learned search may become stronger.
- **Recovery** may compensate less at high sparsity.
- **Latency and energy relations** are hardware-specific.

Larger-scale and multi-platform validation is future work (Chapter 8).

## 7.13 Limitations

The limitations below bound every claim in Chapters 5–7. They are grouped by kind.

**A. Model and dataset scale**
- CIFAR-scale data and small CNNs only.
- Four pruning units and six ratios per network. The space was small enough to enumerate, which made exhaustive
  ground truth possible but makes the RL problem easier than at realistic scale.
- The sustainability phases used CIFAR-10 only, and Phase 7B used SimpleCNN only.

**B. Optimisation**
- **PPO's search cost** is substantial compared with deterministic methods.
- **Sensitivity guidance** improved exploration but not final accuracy, and one Phase-4 exploration measure reversed.
- **LAMP remained stronger** before recovery and was still slightly ahead in three settings after it.
- **The results depend on reward design.** No Phase-4 redesigned reward met its criterion, and the constrained reward
  made SimpleCNN slightly worse.
- **Elite behavioural cloning** was tested at one configuration only.

**C. Validation methodology**
- Validation-to-test transfer of the accuracy constraint was imperfect (67.5% test feasibility, 0% in CIFAR-100
  ResNet-8).
- The fixed relative threshold (τ = 0.98) and the small selection splits make feasibility sensitive to sampling
  noise.
- The deployment phases used a single recovery seed per model.
- One dense reference (CIFAR-10 SimpleCNN) used an older training recipe.
- Phase 6 was not a fully independent test confirmation, since the test sets had been read for earlier frozen models.
- The ±0.5 pp rule is a confidence-interval decision rule, not a formal equivalence test.
- Recovery used one fixed five-epoch protocol. Longer or tuned recovery might change the rankings.

**D. Hardware**
- One CPU platform, one batch size and one thread configuration.
- No GPU, mobile, embedded or accelerator evaluation.
- Latency relations are architecture- and platform-specific.
- The SimpleCNN accuracy proxy failed validation, so its accuracy-feasible latency headroom is unknown.
- The exhaustive 5-epoch recovered landscape was stopped for compute cost.

**E. Energy**
- CPU package energy only. No DRAM, whole-system or wall-plug measurement, and no external power meter.
- The Phase-7B measurement setup failed its stability qualification repeatedly, so no Phase-7B energy or latency claim
  is made.
- The energy surrogate and the accuracy proxy are model-selection tools, not measurements.

**F. Sustainability**
- No carbon accounting.
- No lifecycle or embodied-energy assessment.
- No analysis of training or search energy.

The deployment-scale energy figures computed in Phase 7A are labelled projections and are not used as results.

## 7.14 Overall Scientific Contribution

**The central contribution.** The novelty of this thesis does not lie in a claim that PPO beats existing pruning
methods; it does not. It lies in an **integrated, pre-registered diagnostic framework**. The framework follows one
pruning approach from layer-wise allocation to measured hardware energy, isolating a separate question at each stage:

| Stage | Question | Thesis evidence | Research question |
|---|---|---|---|
| 1. Adaptive layer-wise allocation | Can a learned agent decide *where* to prune? | Yes: far better than uniform or random; not better than LAMP | RQ1 |
| 2. Sensitivity-guided exploration | Does layer sensitivity help the search? | Search yes, outcome no | RQ2 |
| 3. Policy retention | Does the agent keep what it finds? | No, unless an archive retains it (validation ranking) | RQ3 |
| 4. Reward alignment | Does the reward encode the desired trade-off? | A trade-off reward over-prunes; a constraint controls it, without a generalisation guarantee | RQ4 |
| 5. Equal recovery | Do method differences survive identical fine-tuning? | Largely not, at these sparsities | RQ5 |
| 6. Deployment translation | Does compression become latency? | Through MACs and structure, not through parameter count | RQ6 |
| 7. Direct energy validation | Does it become measured energy, at what accuracy cost? | Yes in Phase 7A, with material accuracy loss; unresolved in Phase 7B | RQ6 |

**How the research questions fit together.** The six questions form a chain, not a list of independent results. Each
answer changed what the next question had to control for:
- **RQ1 to RQ2.** RQ1 established that the allocation is learnable, which raised the question of guidance.
- **RQ2 to RQ3.** RQ2's search-versus-outcome dissociation pointed to the returned policy rather than the search.
- **RQ3 to RQ4.** RQ3's retention fix then exposed the reward, which became RQ4.
- **RQ4 to RQ5.** RQ4's corrected models made the fair comparison of RQ5 meaningful.
- **RQ5 to RQ6.** RQ5's small post-recovery differences moved the question from accuracy to deployment.
- **Within RQ6.** RQ6 showed that compression must be re-measured at each hardware level.

**The thesis-level conclusion.** Effective pruning is **not a single optimisation problem**. It is a chain of coupled
problems:
- where to prune;
- how to search;
- how to guide the search;
- how to retain good policies;
- how to encode the desired trade-off;
- how much recovery is allowed;
- whether compression translates into hardware savings;
- whether those savings survive an accuracy constraint.

A weakness at any link can hide or reverse the effect of improvements elsewhere:
- a good search under a misaligned reward produced worse models;
- a retained policy did not generalise;
- an allocation that was good as a mask was poor as a structure;
- a strong MAC reduction remained unconfirmed as an energy saving.

**This multi-stage view, supported at each stage by pre-registered evidence including negative results, is the
thesis's central scientific contribution.**

**Implications for future pruning research:**
1. **Comparisons** should be made after equal recovery.
2. **RL studies** should separate search quality from retention.
3. **Rewards** should encode accuracy boundaries explicitly.
4. **Deployment claims** should be grounded in structured realisation and direct measurement, with accuracy stated
   alongside.
5. **Energy claims** should be made only after the measuring instrument has been qualified.

## 7.15 Chapter Summary

- **PPO.** It contributes a flexible, learnable layer-wise allocation that clearly beats uniform and random
  allocation. It does not beat the strong deterministic LAMP baseline, so its value lies in adaptability to changing
  objectives rather than universal superiority.
- **Sensitivity.** It acts as a search bias. It improves exploration in every phase, but not the final model.
  Centring made the prior layer-specific, but did not make it predictive of final accuracy.
- **Retention.** Search quality and policy retention are separate problems. An external archive retained good
  policies on validation ranking, more reliably than imitation inside training. It did not improve generalisation.
- **Reward.** A stronger optimiser exposed a misaligned trade-off reward. An accuracy constraint controlled
  over-pruning but did not guarantee test feasibility.
- **Equal recovery.** It narrowed method differences to small values. Pruning comparisons should be made under equal
  recovery budgets.
- **Deployment.** Sparsity is not speed: parameter reduction is a weak latency proxy and MACs a strong one. Measured
  latency and energy remain necessary, and accuracy-feasible hardware headroom is much smaller than unconstrained
  headroom.
- **Phases 7A and 7B.**
  - Structured pruning reduced measured CPU energy in Phase 7A, at a material accuracy cost, and Uniform was the
    better choice on SimpleCNN.
  - Deployment-oriented structured PPO (Phase 7B) improved the memory–computation–accuracy trade-off but missed the
    accuracy criterion, and its energy remains unresolved.
- **Measurement.** Energy measurement on commodity hardware is fragile and must be qualified for each campaign.
- **Contribution.** The thesis shows pruning to be a chain of coupled problems, each needing its own evidence. That is
  its central contribution.

Chapter 8 concludes the thesis and sets out the future work these findings motivate.
