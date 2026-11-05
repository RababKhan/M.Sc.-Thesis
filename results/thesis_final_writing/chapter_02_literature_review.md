# Chapter 2 — Literature Review

This chapter reviews the research on which the thesis builds and identifies the questions it leaves open. It begins
with neural network compression in general (Section 2.1) and the distinction between unstructured and structured
pruning (Section 2.2). It then turns to the central problem of the thesis: how much of each layer to prune
(Sections 2.3–2.5). Sections 2.6–2.9 review reinforcement learning (RL) for compression, the PPO algorithm, the use of
sensitivity information, and the search-and-selection problem that any learned pruning policy faces. Section 2.10
covers post-pruning recovery. Sections 2.11–2.13 cover the deployment side: hardware-aware compression, the
relationship between cost metrics and real latency, and energy measurement. Section 2.14 draws these threads together
into the research gap that motivates the thesis.

Citations in this draft are written as bracketed citation keys. Each key resolves to an entry in
`results/thesis_final_writing/reference_audit.md`, and the keys are converted to numbered IEEE references when the
thesis is assembled.

## 2.1 Neural Network Compression

The accuracy gains of deep neural networks have come with growing parameter counts, arithmetic cost and memory
traffic. This makes deployment expensive on servers and difficult on resource-limited devices
[@menghani2023efficient; @hoefler2021sparsity]. Neural network compression aims to reduce one or more of these costs
while keeping predictive quality close to that of the original model. The main families are:

- **pruning**, which removes parameters or structures;
- **quantisation**, which lowers numerical precision;
- **knowledge distillation**, which trains a smaller model to imitate a larger one;
- **efficient architecture design**, sometimes automated through neural architecture search.

Several recent surveys organise this field [@menghani2023efficient; @cheng2024pruningsurvey; @he2024structuredsurvey].

Pruning is among the oldest of these ideas. Early work removed weights according to an estimate of how much the
training loss would increase without them, using second-order information [@lecun1989obd]. Later work showed that
simply removing small-magnitude weights and retraining could reduce the parameter count of large convolutional
networks many times over with little accuracy loss [@han2015learning]. The lottery ticket hypothesis then argued that
dense networks contain sparse subnetworks that train to comparable accuracy, which renewed interest in why pruning
works [@frankle2019lth].

The field is now very large. One recent survey counts more than three thousand pruning papers published between 2020
and 2024 [@cheng2024pruningsurvey]. Pruning and quantisation are also often compared directly. A systematic study
found that quantisation usually gives a better accuracy–compression trade-off than pruning, except at very high
compression ratios [@kuzmin2023pruningvsquant]. This thesis restricts itself to pruning, because its research
questions concern *where* capacity should be removed. Quantisation is complementary and is not studied.

Two observations from this literature shape the thesis.

1. **"Efficiency" is not a single quantity.** A pruned model may need less storage, fewer arithmetic operations, less
   time per inference or less energy, and these do not necessarily move together [@dehghani2022misnomer;
   @hoefler2021sparsity]. Sections 2.11–2.13 return to this point.
2. **Reported compression results are hard to compare.** Studies differ in architectures, datasets, training budgets
   and fine-tuning protocols, and these differences can be larger than the differences between pruning methods
   [@blalock2020state; @liu2023sparsitymaycry]. This motivates the controlled comparisons in this thesis.

## 2.2 Unstructured and Structured Pruning

Pruning methods are commonly classified by the *granularity* of what they remove (Table 2.1).

**Table 2.1. Pruning granularity.**

| Granularity | What is removed | Resulting model | Typical deployment benefit |
|---|---|---|---|
| Unstructured (weight-level) | Individual weights | Same tensor shapes, with a sparse mask | Smaller compressed storage; speed-up only with sparse kernels or hardware support |
| Semi-structured (e.g. N:M) | Fixed patterns of weights within small blocks | Same shapes, regular sparsity pattern | Speed-up on hardware that supports the pattern |
| Structured (channel, filter, neuron) | Whole channels, filters or neurons | Physically smaller dense tensors | Fewer MACs and smaller dense kernels, usable by standard libraries |

**Unstructured pruning** removes individual weights. Because the choice of which weights to keep is unconstrained, it
can reach high sparsity at a given accuracy [@han2015learning; @hoefler2021sparsity]. The resulting model, however,
keeps its original tensor shapes, with zeros stored and computed unless sparse storage formats and sparse kernels are
used. On general-purpose hardware such kernels are effective only at high sparsity and favourable sparsity patterns, so
weight-level sparsity often brings little or no reduction in latency [@hoefler2021sparsity; @he2024structuredsurvey].

**Structured pruning** removes whole structures such as convolutional filters, channels or fully connected neurons
[@li2017filters; @he2017channel; @wen2016ssl]. The pruned network is a smaller dense network: it can be executed by
standard dense libraries and directly reduces MACs and memory traffic. That is why the structured-pruning survey of He
and Xiao describes structured pruning as the route to "realistic acceleration" [@he2024structuredsurvey]. The price is
a coarser search space. Removing a channel removes all of its weights at once, and accuracy typically degrades faster
than with unstructured pruning at the same parameter reduction.

**Recent work on structured pruning** has concentrated on two problems.

- **Dependency handling.** In residual and attention architectures, removing a channel in one layer forces matching
  removals elsewhere. DepGraph models these dependencies as a graph so that coupled parameters are pruned together
  across arbitrary architectures [@fang2023depgraph]. Isomorphic pruning compares the importance of sub-structures
  only within groups of the same computational topology, because heterogeneous structures have incomparable
  parameter scales [@fang2024isomorphic].
- **Pruning during training.** CHEX repeatedly prunes and regrows channels during training instead of pruning a
  fully trained model once [@hou2022chex]. OTOv2 trains and compresses a model in a single pass without separate
  fine-tuning [@chen2023otov2]. Submodular structured pruning offers guarantees in a limited-data setting
  [@halabi2022submodular].

**Random structure as a baseline.** A benchmark of channel pruning found that random search over channel configurations works well
compared with other channel-pruning methods, and that no clear winner emerged among channel-importance criteria in
that setting [@li2022randomchannel]. This is a reminder that *how much* to prune in each layer, together with the training
protocol, can matter as much as *which* channels are removed.

**Relevance to this thesis.** The core framework (Chapters 3–5) uses magnitude-based *unstructured* masks within each
layer and learns the per-layer ratio. This choice made exhaustive evaluation of the policy space possible. The
deployment extension (Chapter 6) realises the same kind of per-layer allocation as *physically structured* networks,
so that memory, MACs, latency and energy can be measured. The gap between the two realisations is itself one of the
findings of Chapter 6.

## 2.3 Layer-Wise Adaptive Pruning

A pruning method must decide not only *which* parameters to remove but also *how many* to remove from each layer: the
layer-wise sparsity allocation. The simplest choice applies the same ratio to every layer (uniform pruning). Layers
differ, however, in size, position and redundancy. Early layers of a convolutional network are small and often
fragile, while large fully connected or late convolutional layers often contain many redundant parameters. A uniform
allocation therefore over-prunes some layers and under-prunes others. A sizeable literature shows that the allocation
itself is a first-order factor in pruned-model quality [@lee2021lamp; @liu2022randompruning; @gadhikar2023random].

Approaches to the allocation problem fall into four broad classes.

1. **Rule-based allocations** fix the ratios from layer properties. An example is the Erdős–Rényi-Kernel (ERK)
   allocation, which scales density with the layer's fan-in and fan-out and was popularised in sparse training
   [@evci2020rigl].
2. **Global ranking** applies one importance threshold across all layers, so that allocation emerges from where
   low-importance weights happen to lie. Global magnitude pruning is the simplest such method.
3. **Score-normalised global ranking** rescales importance scores so that a single global threshold produces a
   better allocation. LAMP is the best-known example (Section 2.5) [@lee2021lamp].
4. **Search- or learning-based allocation** treats the per-layer ratios as variables to optimise, using RL
   [@he2018amc; @camci2022qlp], learned global filter ranking [@chin2020legr], fast sub-network evaluation
   [@li2020eagleeye] or bi-level optimisation [@zhang2022bip].

Recent results have sharpened the importance of allocation. In sparse training, randomly pruned networks can be
competitive provided that the per-layer ratios are sensible; the layer-wise ratio, rather than the precise mask, was a
key factor [@liu2022randompruning]. A theoretical analysis complements this, explaining when random masks with
appropriate layer-wise densities can approximate target networks [@gadhikar2023random]. The same question has
reappeared in large language models. One-shot LLM pruning methods traditionally prune every layer to the same
sparsity, but OWL allocates higher sparsity to layers with fewer outlier activations and improves markedly at high
sparsity [@yin2024owl]. BESA learns block-wise sparsity allocations under a global budget [@xu2024besa]. Other work
proposes a sparsity-based compressibility index (PQ Index) to guide adaptive pruning [@diao2023sparsityperspective].

**What remains less studied** is how a *learned* allocation behaves compared with cheap deterministic allocations
once confounding factors are controlled. Those factors are the sparsity level, the recovery budget and the selection
procedure. Learned allocators are usually reported against published numbers of other methods, obtained under
different protocols. The comparison is rarely made at the exact sparsity chosen by the learned method and under
identical recovery. This is the setting of Research Question 1.

## 2.4 Magnitude-Based and Deterministic Pruning

**Magnitude pruning** removes the weights of smallest absolute value. Despite its simplicity it remains one of the
strongest and most widely used baselines [@han2015learning; @blalock2020state]. It requires no data, no gradients and
no search, and it combines easily with any allocation rule.

Variants differ in the *scope* over which magnitudes are compared:

- **Layer-wise magnitude pruning** removes a fixed fraction within each layer, so it needs an external allocation.
- **Global magnitude pruning** compares magnitudes across all layers. Because weight scales differ between layers, it
  can prune some layers almost completely, which is especially harmful in small layers.

A recurring finding of systematic evaluations is that carefully tuned simple methods are hard to beat. ShrinkBench
documented that many papers compare against weak or inconsistently configured baselines, and that simple
magnitude-based pruning remains competitive with many more complex methods when evaluated consistently
[@blalock2020state]. Re-examinations of
structured pruning found that training a pruned architecture from scratch can match fine-tuning an inherited one,
which questions how much the specific inherited weights matter [@liu2019rethinking]. Benchmarks of sparse networks
on harder tasks have also shown that apparent gains can vanish under more demanding evaluation
[@liu2023sparsitymaycry].

**Deterministic methods** (magnitude pruning, global ranking, ERK, LAMP) have three practical advantages over learned
allocators: they are cheap, reproducible and free of search hyperparameters. Any learned method must therefore
justify its additional cost against them. The thesis compares PPO against Uniform, Random, global magnitude, ERK and
LAMP at the same sparsity as each PPO run, rather than against numbers reported elsewhere.

## 2.5 LAMP and Related Layer-Adaptive Methods

Layer-Adaptive Magnitude-based Pruning (LAMP) rescales each weight's magnitude by the magnitudes of the weights in
the same layer that would survive if it were pruned. This yields a score that reflects the model-level ℓ2 distortion
caused by pruning [@lee2021lamp]. A single global threshold on the LAMP score then produces a layer-wise allocation
automatically, without hyperparameters and with negligible computation. In its original evaluation, LAMP
consistently outperformed popular existing schemes for layer-wise sparsity selection under various
image-classification setups. It also
remained strong under weight rewinding, where the connectivity-oriented allocation, the strongest baseline overall,
fell behind even simple global magnitude pruning [@lee2021lamp].

LAMP matters to this thesis for two reasons. First, it is a strong *deterministic* answer to exactly the problem a
learned allocator addresses: how to distribute sparsity across layers. Second, it costs almost nothing to compute,
whereas an RL search requires thousands of pruned-model evaluations. LAMP is therefore the natural benchmark against
which the practical value of RL-based allocation must be judged.

Related layer-adaptive methods include:

- **Learned global ranking**, which learns a single ranking of filters across layers, from which architectures with
  different accuracy–latency trade-offs follow by pruning the lowest-ranked filters [@chin2020legr];
- **Outlier-based allocations for LLMs** [@yin2024owl];
- **Block-wise learned allocations** [@xu2024besa].

**Limitation shared by these comparisons.** The original comparisons of these methods, like those of most allocators,
were made under each paper's own training and fine-tuning protocol. Chapter 5 of this thesis compares PPO and LAMP
both before and after identical recovery, so that the contribution of allocation can be separated from the
contribution of fine-tuning.

## 2.6 Reinforcement Learning for Neural Network Compression

Compression decisions can be framed as sequential decisions. An agent visits the layers of a network one by one,
chooses a compression ratio for each, and receives a reward that reflects the accuracy and cost of the compressed
model. This view connects compression to neural architecture search, where an RL controller was first used to
generate architectures and was rewarded by their validation accuracy [@zoph2017nas].

**AMC** made the formulation concrete for pruning [@he2018amc]. A DDPG agent receives an embedding of each layer and
outputs a continuous sparsity ratio. The reward combines accuracy with resource constraints. AMC reported better
accuracy than hand-crafted compression at a given FLOPs reduction, and measured speed-ups on a GPU and on mobile
hardware [@he2018amc]. Platform-aware methods such as NetAdapt and HAQ extended the idea by placing measured latency
or hardware feedback inside the optimisation loop [@yang2018netadapt; @wang2019haq].

Recent RL-based pruning research has explored several directions (Table 2.2):

- **Q-learning allocation.** QLP uses deep Q-learning to choose layer-wise sparsity ratios. It can revisit layers and
  prune them gradually instead of making a single first-to-last pass [@camci2022qlp].
- **Lookahead search.** RL guided by a lookahead search has been used to select the filters to remove in channel
  pruning [@wang2022lookahead].
- **Per-channel agents.** DECORE assigns a lightweight agent to each channel and trains the agents with a policy
  gradient method, so that channel retention is learned jointly with the network [@alwani2022decore].
- **Topology embeddings.** GNN-RL embeds the network's computational graph with a graph neural network and uses RL to
  find a compression policy; it reports results on ResNet, VGG, MobileNet and ShuffleNet models [@yu2022gnnrl].
- **Joint training and pruning.** An RL agent sets layer pruning ratios while the network is being trained, rather
  than after a full pre-training run [@ganjdanesh2024agent].

These methods show that RL can find competitive allocations and can incorporate non-differentiable objectives. Three
common features of their evaluation, however, leave questions open.

1. **Final-policy evaluation.** They are usually judged by the quality of the final policy, typically after a
   method-specific fine-tuning procedure. Whether the agent *retained* the best policy it encountered during search is
   rarely examined, even though the search may visit many policies that are better than the one it ends on.
2. **Hand-built rewards.** The rewards combine accuracy and cost terms with hand-chosen coefficients. How those
   coefficients shape pruning aggressiveness, and whether better search exploits weaknesses in the reward, is
   seldom analysed systematically.
3. **Comparisons under different protocols.** Comparisons with deterministic allocators are often made against
   published results obtained under different recovery budgets, which makes it hard to attribute gains to the
   allocation itself.

The thesis addresses these three points directly (Research Questions 3, 4 and 5).

**Table 2.2. Comparison of representative pruning-allocation and deployment-aware pruning works.** "Struct." =
structured pruning; "Sens." = an explicit sensitivity or saliency signal drives allocation; "HW metric" = a measured
or table-based hardware metric (latency, throughput) enters the method or its evaluation; "Energy" = inference energy
measured on hardware. Datasets and models are as stated in each paper's abstract. Limitations are stated relative to
the questions of this thesis.

| Work | Year | Pruning type | Selection / optimisation | Layer-adaptive | Sens. | RL | Struct. | HW metric | Energy | Datasets / models | Main limitation (for this thesis' questions) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Energy-aware pruning [@yang2017energyaware] | 2017 | Weight | Energy-model-guided greedy | Yes | No | No | No | Energy model | Estimated (model calibrated on hardware) | ImageNet CNNs (AlexNet, GoogLeNet) | Energy is modelled, not measured per inference |
| AMC [@he2018amc] | 2018 | Weight / channel | DDPG agent per layer | Yes | No | Yes | Both | Latency (GPU, mobile) | No | ImageNet VGG-16, MobileNet | Final-policy evaluation; reward coefficients fixed by hand |
| NetAdapt [@yang2018netadapt] | 2018 | Filter | Iterative, measured-latency-driven | Yes | No | No | Yes | Measured latency | No | ImageNet MobileNet V1/V2 | Platform-specific measurement loop at each step |
| LAMP [@lee2021lamp] | 2021 | Weight | Rescaled-magnitude global threshold | Yes | Distortion score | No | No | No | No | Image-classification CNNs | Unstructured: no direct latency benefit |
| QLP [@camci2022qlp] | 2022 | Weight | Deep Q-learning, multi-pass | Yes | No | Yes | No | No | No | CIFAR-10 ResNet-32/56; ImageNet ResNet-50, MobileNet-v1 | Unstructured masks; deployment effect not measured |
| Lookahead RL [@wang2022lookahead] | 2022 | Channel | RL with lookahead search | Yes | No | Yes | Yes | No | No | MNIST, CIFAR-10, ImageNet | Final-policy evaluation |
| DECORE [@alwani2022decore] | 2022 | Channel | Per-channel policy-gradient agents | Yes | No | Yes | Yes | FLOPs / memory constraints | No | CIFAR-10 ResNet-110, VGG; ImageNet ResNet-50 | FLOPs used as cost, not measured latency |
| GNN-RL [@yu2022gnnrl] | 2022 | Channel | GNN embedding + RL | Yes | Topology | Yes | Yes | FLOPs-constrained | No | ResNet, VGG-16, MobileNet-v1/v2, ShuffleNet | Cost measured as FLOPs |
| HAP [@yu2022hap] | 2022 | Structured | Second-order (Hessian) sensitivity | Yes | Yes | No | Yes | No | No | CIFAR-10 PreResNet29; ImageNet ResNet-50 | Sensitivity used for selection; no search or retention question |
| HALP [@shen2022halp] | 2022 | Channel | Latency-constrained knapsack | Yes | Saliency | No | Yes | Latency lookup table | No | ImageNet ResNet-50/101; VOC detection | Needs per-device latency tables; energy not measured |
| SMCP [@humble2022smcp] | 2022 | Channel | Soft masking under a cost constraint | Yes | Saliency | No | Yes | Cost (latency) constraint | No | ImageNet, PASCAL VOC | Energy not measured |
| CHEX [@hou2022chex] | 2022 | Channel | Prune-and-regrow during training | Yes | No | No | Yes | FLOPs | No | ImageNet ResNet-50; detection, segmentation | FLOPs as cost |
| DepGraph [@fang2023depgraph] | 2023 | Structured (any) | Dependency graph + norm criterion | Yes | No | No | Yes | No | No | CNNs, ViT, GNN, LSTM | Grouping, not allocation search |
| OTOv2 [@chen2023otov2] | 2023 | Structured | Train-once optimiser | Yes | No | No | Yes | No | No | VGG, ResNet, ConvNeXt, DenseNet; CIFAR-10/100, ImageNet | Allocation implicit in training |
| OWL [@yin2024owl] | 2024 | Weight | Outlier-ratio layer allocation | Yes | Outlier statistic | No | No | Inference speed (sparse engine) | No | LLaMA, Vicuna, OPT, Mistral | LLM-specific signal |
| RL agent + joint training [@ganjdanesh2024agent] | 2024 | Channel | RL agent sets layer ratios during training; accuracy reward | Yes | No | Yes | Yes | No | No | CIFAR-10, ImageNet; ResNets, MobileNets | Final-policy evaluation |
| **This thesis** | — | Weight (core); channel (extension) | PPO + centred sensitivity prior + archive + constrained reward | Yes | Yes | Yes | Extension | Measured CPU latency | RAPL (7A); unresolved (7B) | CIFAR-10/100; SimpleCNN, LeNet-5, ResNet-8 | Small models; one CPU platform; LAMP remains stronger before recovery |

## 2.7 PPO-Based Optimisation

Proximal Policy Optimisation (PPO) is an on-policy policy-gradient algorithm. It stabilises learning by limiting how
far each update can move the policy, using a clipped surrogate objective [@schulman2017ppo]. It is commonly combined
with generalised advantage estimation [@schulman2016gae]. PPO is popular because it is comparatively insensitive to
hyperparameters, handles discrete and continuous actions, and has mature, well-tested implementations such as
Stable-Baselines3 [@raffin2021sb3], which this thesis uses.

Large empirical studies have qualified this picture.

- **Implementation details.** Code-level choices that are not part of the algorithm's description (for example
  value-function clipping and reward normalisation) can account for a large share of PPO's reported advantage over
  related methods [@engstrom2020implementation].
- **Design choices.** A study of more than fifty design choices in on-policy actor-critic methods found that many
  settings matter considerably and offered practical recommendations [@andrychowicz2021onpolicy].
- **Multi-agent settings.** PPO has proved surprisingly strong in cooperative multi-agent games, where it had been
  assumed to be less sample-efficient than off-policy methods [@yu2022mappo].

The conclusion relevant here is that PPO is a reasonable and well-understood optimiser. Its behaviour must still be
examined empirically rather than assumed, especially in an unusual setting such as pruning allocation.

The pruning task differs from typical RL benchmarks in three ways.

1. **Short episodes.** An episode consists of a handful of decisions, one per prunable unit, so the problem resembles
   a contextual combinatorial search more than long-horizon control.
2. **Expensive, noisy reward.** The reward requires evaluating a pruned model on held-out data, and its noise comes
   from a finite validation set.
3. **Small, enumerable space in this thesis.** With four units and six ratios there are 1,296 policies, so the full
   reward landscape can be computed and each PPO run compared against the exact optimum.

The last feature is unusual in the RL-for-compression literature. It allows the thesis to measure search quality
(how highly ranked are the policies PPO visits?) separately from final-policy quality (how highly ranked is the policy
it ends on?).

## 2.8 Sensitivity-Aware Pruning

**Sensitivity** describes how much a model's loss or accuracy changes when a parameter, structure or layer is pruned.
It is one of the oldest ideas in pruning.

- **Second-order estimates.** Optimal Brain Damage estimated weight saliency from second-order derivatives of the
  loss [@lecun1989obd]. Modern second-order structured methods capture correlations across structures with
  Hessian-vector products [@nonnenmacher2022sosp]. Hessian-aware pruning uses second-order sensitivity to decide which
  structures to prune and which to replace [@yu2022hap]. Global structural pruning of vision transformers uses a
  Hessian-aware saliency to redistribute parameters across blocks [@yang2023globalvit].
- **First-order (Taylor) estimates.** These approximate the loss change caused by removing a filter or neuron from
  gradients [@molchanov2017taylor; @molchanov2019importance].
- **Connection sensitivity at initialisation.** SNIP scores connections by their effect on the loss before training
  [@lee2019snip].
- **Post-training compression for large models.** Optimal Brain Compression and SparseGPT build on second-order
  reconstruction for accurate one-shot compression [@frantar2022obc; @frantar2023sparsegpt]. Wanda shows that a
  simple weight-times-activation score is a strong alternative [@sun2024wanda].

In most of this work, sensitivity is used *directly as the pruning criterion*: it decides which elements to remove or
how many. A different use is to provide sensitivity as *guidance to a search procedure* (a feature in the agent's
state or a prior over its actions), while the search itself decides the allocation. In that role, sensitivity can
help in two distinct ways:

- it can make the search more efficient, finding good policies sooner or avoiding destructive ones;
- it can lead to a better final model.

These need not coincide. A search may be made faster without being made better at its end point, especially if
another component (selection, reward or recovery) is the bottleneck. Studies that add sensitivity to an RL or search
pipeline usually report the final accuracy of the combined method, which does not reveal which of the two effects
occurred. Research Question 2 separates them. It measures exploration quality (rank-based and accuracy-based areas
under the search trajectory, and the frequency of destructive policies) separately from final accuracy. It also
compares a correct sensitivity prior with shuffled and constant controls.

## 2.9 Search and Policy Selection in Pruning

Any search-based pruning method has to answer a question that is easy to overlook: *which* of the policies
encountered during search is returned? An RL agent visits many policies during training. The one it ends on, the
terminal or greedy policy, is not necessarily the best one it saw.

The issue is well known in neighbouring fields.

- **Neural architecture search.** An evaluation of the NAS search phase found that state-of-the-art NAS algorithms performed, on
  average, similarly to a random selection policy, and that weight sharing degraded the ranking of candidates. This
  reveals a gap between the search process and the final selection [@yu2020evalnas].
- **Reinforcement learning.** Self-imitation learning reuses the agent's own best past trajectories to steer learning
  towards them [@oh2018sil]. Go-Explore keeps an explicit archive of states it has reached and returns to promising
  ones before exploring further, which overcomes the failure of agents to revisit good states
  [@ecoffet2021goexplore].
- **Pruning.** Evaluating sub-networks is costly, so proxies are needed. EagleEye showed that a fast evaluation step
  using adaptive batch-normalisation statistics correlates much better with final accuracy than direct evaluation of
  the pruned weights [@li2020eagleeye]. In NAS, reduced-training and "zero-cost" proxies are evaluated by how well they
  preserve the ranking of candidates [@abdelfattah2021zerocost].

Two lessons follow for RL-based pruning:

- what the agent *retains* can differ substantially from what it *finds*;
- the selection step itself needs validation data separate from the data that drives the search, or it will favour
  candidates that overfit the search signal.

The thesis studies both. It measures how often the terminal PPO policy is among the top-ranked policies, compared with
the best policy PPO visited. It then introduces archive-based selection, in which the best visited policies are kept
and re-ranked on a held-out selection split (Research Question 3). It also documents where this improves validation
ranking without improving test accuracy.

## 2.10 Post-Pruning Recovery and Fine-Tuning

Pruning usually reduces accuracy, which is then partly or wholly recovered by further training. The recovery
procedure is not a detail.

- **Rewinding.** Weight rewinding and learning-rate rewinding can recover more accuracy than conventional
  low-learning-rate fine-tuning [@renda2020rewinding]. Later work traced much of this effect to the learning-rate
  schedule used during retraining [@le2021retraining].
- **Inherited weights.** Some structured-pruning results suggest that the inherited weights matter less than the
  pruned architecture, since training from scratch can match fine-tuning [@liu2019rethinking].
- **Benchmarks.** ShrinkBench identified inconsistent fine-tuning budgets as a major source of incomparable results
  [@blalock2020state].
- **Recovery budgets in large models.** Retraining is costly enough that some methods avoid it entirely
  [@kwon2022posttraining; @frantar2023sparsegpt]. Others use a small amount of recovery training [@ma2023llmpruner]
  or substantial continued pre-training after structured pruning [@xia2024sheared].
- **Effects beyond accuracy.** Pruning can affect a model in ways that test accuracy does not reveal: changes in
  behaviour under distribution shift [@liebenwein2021lost], disparate effects across groups [@tran2022disparate], and
  transfer performance [@iofinova2022transfer].

The consequence for method comparison is direct. If two pruning methods are compared before recovery, the comparison
measures the immediate damage done by pruning. If they are compared after recovery with different budgets, it
partly measures the budgets. Only a comparison under **identical** recovery isolates the effect of the allocation or
criterion. The thesis therefore holds the recovery protocol fixed across methods (same optimiser, learning rate,
schedule, epochs and data order) and keeps the pruning mask fixed. It compares methods both before and after recovery
(Research Question 5).

## 2.11 Hardware-Aware Neural Network Compression

Hardware-aware methods put a hardware cost, rather than a proxy such as parameter count, into the optimisation.

- **Measured or table-based latency.** NetAdapt adapts a network to a target platform using measured latency
  [@yang2018netadapt]. HAQ uses hardware feedback to choose per-layer quantisation precision [@wang2019haq].
  Platform-aware NAS includes measured device latency in the search objective [@tan2019mnasnet]. Once-for-all
  trains a single network from which specialised sub-networks are drawn for different hardware constraints
  [@cai2020ofa].
- **Structured pruning under a latency budget.** HALP formulates pruning as a knapsack problem that maximises
  retained saliency subject to a latency budget estimated from per-device lookup tables [@shen2022halp]. SMCP extends
  this with soft masks that let pruned channels return during training while respecting a cost constraint
  [@humble2022smcp].

These methods share a premise: the relationship between what is removed and what is gained on hardware is not simple
enough to be captured by parameter counts. They also share a cost. Hardware-aware optimisation needs a hardware
model or measurement loop for each target device. Its benefit depends on there being enough *accuracy-feasible*
headroom, meaning cheaper configurations that still meet the accuracy requirement. The thesis treats that headroom
as an empirical question to be answered before any hardware-aware optimisation is attempted (Chapter 6). It measures
the latency of every structured configuration in its search space and checks whether accuracy-feasible
configurations offer enough room for a hardware-aware RL agent to exploit.

## 2.12 Parameter Count vs MACs/FLOPs vs Real Latency

The cost of a neural network can be described at three levels:

1. **parameter count**, or parameter memory;
2. **arithmetic count**, as multiply–accumulate operations (MACs) or floating-point operations (FLOPs);
3. **measured runtime** on specific hardware.

These levels are related but not interchangeable.

- **FLOPs are not speed.** ShuffleNet V2 showed early that networks with similar FLOPs can differ markedly in speed,
  because memory access cost and degree of parallelism also matter [@ma2018shufflenetv2].
- **The efficiency misnomer.** Dehghani et al. argued that common cost indicators — parameter count, FLOPs and
  throughput — are often assumed to be correlated. In fact they can disagree, so reporting only one can give a
  misleading picture of efficiency [@dehghani2022misnomer].
- **Sparsity does not translate into speed.** Unstructured sparsity reduces parameter count but not necessarily
  runtime, unless the hardware and kernels exploit it [@hoefler2021sparsity]. Even structured reductions translate
  into speed-ups that depend on layer shapes and library kernels [@he2024structuredsurvey; @shen2022halp].
- **Model-level latency studies.** NetAdapt showed that optimising indirect metrics such as MACs or weights need not
  reduce direct metrics such as latency and energy [@yang2018netadapt]. Recent LLM work reports speed-ups only when
  a sparse inference engine supports the sparsity pattern [@yin2024owl].

**What is less common** is an *exhaustive* measurement within one search space: how well parameter reduction and MAC
reduction predict measured latency across *all* configurations a pruning method could choose. Such a measurement
shows how often a sparser configuration is actually slower, and how much latency headroom exists among
configurations that meet an accuracy requirement. Chapter 6 reports such an analysis for all 1,296 structured
configurations of two architectures.

## 2.13 Energy-Efficient and Sustainable Neural Network Inference

**The energy cost of machine learning** has become a research topic in its own right. Early analyses drew attention
to the energy and carbon cost of training large NLP models [@strubell2019energy]. Researchers were urged to report
efficiency alongside accuracy, with the label "Green AI" used for this agenda [@schwartz2020greenai]. Frameworks for
the systematic reporting of energy and carbon footprints followed [@henderson2020reporting]. Industrial analyses have
discussed the environmental implications of AI across the full model lifecycle [@wu2022sustainable]. They have also
argued that best practices can substantially reduce training energy and emissions [@patterson2022plateau]. Carbon
estimates for large models depend on lifecycle assumptions as much as on measured energy [@luccioni2023bloom].

**Inference** has received growing attention because it is repeated for every query.

- Once hardware efficiency gains and consolidated algorithms are taken into account, inference energy has grown much
  more slowly than parameter-count trends would suggest [@desislavov2023trends].
- A systematic comparison of inference cost, measured as the energy needed for 1,000 inferences, found
  multi-purpose generative models to be orders of magnitude more expensive than task-specific models
  [@luccioni2024powerhungry].
- Benchmarks of LLM inference energy on two GPU generations and multi-GPU configurations show how strongly energy
  depends on hardware and deployment configuration [@samsi2023wordswatts].
- Fine-tuning, although cheap per run, contributes to energy use because it is performed so often
  [@wang2023bertenergy].
- MLPerf Power defines a methodology for measuring the energy efficiency of ML systems from microwatt devices to
  data-centre clusters, and emphasises the role of system-level interactions [@tschand2025mlperfpower].

**Energy-aware pruning.** Within pruning, energy-aware pruning used an energy model, calibrated from hardware
measurements, to guide which layers to prune. It showed that reductions in model size or computation did not
necessarily reduce energy [@yang2017energyaware].

**Measurement is central and difficult.**

- **Software power meters.** On CPUs, Intel's Running Average Power Limit (RAPL) interface exposes energy counters for
  the processor package and its domains without additional hardware. Early experience reports discussed its accuracy
  and practical pitfalls [@khan2018rapl]. A comparison of software-based power meters distinguished tools that
  *measure* through hardware interfaces from tools that *estimate* from models such as thermal design power (TDP)
  [@jay2023powermeters]. A recent comparative analysis of RAPL-based measurement mechanisms found that they differ
  mainly in usability and robustness, and that RAPL is more accurate than statistical estimation models
  [@raffin2024dissecting].
- **Package scope.** RAPL measures the *package*, not the program. Any other activity in the package (other cores,
  integrated graphics, uncore components, thermal state) enters the measurement. Stable measurement therefore
  requires controlled conditions, repeated runs and an explicit stability criterion.

Two distinctions from this literature shape the thesis' treatment of sustainability (Chapter 6):

- **Measured versus estimated energy.** The thesis reports only directly measured package energy, never TDP-based or
  model-based estimates. Where measurement could not be qualified as stable, it reports energy as unresolved.
- **Operational energy versus environmental impact.** Inference energy measured on one processor is one component of
  a model's environmental footprint. Converting it into carbon or lifecycle impact needs further data and
  assumptions [@luccioni2023bloom; @wu2022sustainable]. The thesis therefore makes no environmental or carbon claims.
  It uses "sustainability" only in the narrow, measurable sense of reduced memory, computation and measured
  operational energy.

## 2.14 Research Gap

The literature reviewed above establishes that layer-wise allocation is important, that RL can produce competitive
allocations, that sensitivity carries useful information, and that compression metrics do not straightforwardly
become deployment efficiency. Six narrower questions remain open. Together they define the gap this thesis
addresses.

1. **Adaptive allocation versus strong deterministic allocation, under controlled conditions.** Layer-specific
   allocation clearly matters [@lee2021lamp; @liu2022randompruning; @yin2024owl], but learned allocators are usually
   compared with deterministic ones under different protocols. It is not clear how a learned PPO allocation compares
   with Uniform, Random, global magnitude, ERK and LAMP when every baseline is built at exactly the sparsity the agent
   chose. → *Research Question 1.*

2. **The separate contributions of sensitivity to exploration and to final accuracy.** Sensitivity is widely used as a
   pruning criterion [@molchanov2019importance; @yu2022hap; @nonnenmacher2022sosp]. When it is used to *guide* a
   search, its effect on search efficiency is rarely separated from its effect on final accuracy, or compared with
   uninformative controls. → *Research Question 2.*

3. **Retention of the best policies found during search.** RL-based pruning is usually judged by its final policy
   [@he2018amc; @camci2022qlp; @alwani2022decore; @ganjdanesh2024agent]. Whether the optimiser actually retains the
   best policies it encounters, a failure mode documented in neighbouring fields [@yu2020evalnas;
   @ecoffet2021goexplore], has not been examined systematically for pruning. → *Research Question 3.*

4. **Hidden incentives in pruning rewards.** Pruning rewards combine accuracy and sparsity with hand-chosen weights.
   RL research shows that more capable optimisers exploit misspecified rewards more strongly [@pan2022misspecification;
   @skalse2022rewardgaming], and constrained formulations exist for this reason [@achiam2017cpo; @gu2024saferl].
   Whether better pruning search exposes such incentives, and whether an explicit accuracy constraint controls
   them, has not been tested directly. → *Research Question 4.*

5. **Comparisons before equal recovery.** Recovery procedures strongly affect pruned accuracy [@renda2020rewinding;
   @le2021retraining; @blalock2020state]. Differences between pruning methods measured before recovery, or under
   different recovery budgets, may therefore overstate the differences that remain after identical recovery.
   → *Research Question 5.*

6. **Compression metrics versus deployment efficiency.** Parameter sparsity, MACs, latency and energy can disagree
   [@dehghani2022misnomer; @ma2018shufflenetv2; @yang2018netadapt; @yang2017energyaware]. How strongly parameter and
   MAC reductions predict measured CPU latency across a whole pruning search space, how much accuracy-feasible
   latency headroom exists, and whether structured realisation of a learned allocation reduces *directly measured*
   energy without material accuracy loss all remain empirical questions. Energy measurement on general-purpose CPUs
   adds its own reliability problem [@khan2018rapl; @raffin2024dissecting]. → *Research Question 6.*

The thesis addresses these six questions in sequence. Each answer exposed the next problem. Improving search revealed
a retention problem, solving retention revealed reward misalignment, and controlling the reward raised the question of
how much of the remaining difference between methods survives equal recovery. The deployment questions came last.
Chapter 3 describes the framework that emerged from this sequence, and indicates which components were part of the
original design and which were introduced in response to diagnostic results.
