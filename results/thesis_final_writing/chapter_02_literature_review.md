# Chapter 2 — Literature Review

This chapter reviews compression and pruning granularity (Sections 2.1–2.2), the allocation problem (2.3–2.5),
reinforcement learning (RL), PPO, sensitivity and policy selection (2.6–2.9), recovery (2.10), and hardware-aware
compression, cost metrics and energy measurement (2.11–2.13); Section 2.14 states the research gap.

## 2.1 Neural Network Compression

The accuracy gains of deep neural networks have come with growing parameter counts, arithmetic cost and memory
traffic, making deployment expensive on servers and difficult on resource-limited devices
[@menghani2023efficient; @hoefler2021sparsity]. Compression reduces these costs while keeping predictive quality
close to that of the original model, through pruning, quantisation, knowledge distillation or efficient architecture
design [@menghani2023efficient; @cheng2024pruningsurvey; @he2024structuredsurvey].

Pruning is among the oldest of these ideas. Early work removed weights by a second-order estimate of the loss
increase [@lecun1989obd]; later, removing small-magnitude weights and retraining reduced the parameter count of large
convolutional networks many times over with little accuracy loss [@han2015learning]; the lottery ticket hypothesis
argued that dense networks contain sparse subnetworks that train to comparable accuracy [@frankle2019lth]. One survey
counts more than three thousand pruning papers between 2020 and 2024 [@cheng2024pruningsurvey]. A systematic study
found that quantisation usually gives a better accuracy–compression trade-off than pruning, except at very high
compression ratios [@kuzmin2023pruningvsquant]; this thesis studies pruning because its questions concern *where*
capacity is removed. Two observations shape the thesis.

1. **"Efficiency" is not a single quantity.** A pruned model may need less storage, fewer arithmetic operations, less
   time per inference or less energy, and these do not necessarily move together [@dehghani2022misnomer;
   @hoefler2021sparsity]. Sections 2.11–2.13 return to this point.
2. **Reported compression results are hard to compare.** Studies differ in architectures, datasets, training budgets
   and fine-tuning protocols, and these differences can be larger than the differences between pruning methods
   [@blalock2020state; @liu2023sparsitymaycry]. This motivates the controlled comparisons in this thesis.

## 2.2 Unstructured and Structured Pruning

Pruning methods are classified by the *granularity* of what they remove: *unstructured* (individual weights; same
tensor shapes with a sparse mask; smaller compressed storage, speed-up only with sparse kernels or hardware support),
*semi-structured* (fixed patterns such as N:M within small blocks; speed-up on hardware that supports the pattern) and
*structured* (whole channels, filters or neurons; physically smaller dense tensors with fewer MACs, usable by standard
libraries).

**Unstructured pruning** can reach high sparsity at a given accuracy because the choice of weights is unconstrained
[@han2015learning; @hoefler2021sparsity], but zeros are stored and computed unless sparse formats and kernels are
used, and on general-purpose hardware these are effective only at high sparsity and favourable patterns, so
weight-level sparsity often brings little or no latency reduction [@hoefler2021sparsity; @he2024structuredsurvey].

**Structured pruning** removes filters, channels or neurons [@li2017filters; @he2017channel; @wen2016ssl], giving a
smaller dense network that directly reduces MACs and memory traffic, the route to "realistic acceleration"
[@he2024structuredsurvey]. The price is a coarser search space: accuracy typically degrades faster than with
unstructured pruning at the same parameter reduction.

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

**Random structure as a baseline.** A channel-pruning benchmark found that random search over channel configurations
works well compared with other channel-pruning methods, with no clear winner among channel-importance criteria
[@li2022randomchannel]. *How much* to prune in each layer, together with the training protocol, can matter as much as
*which* channels are removed.

**Relevance to this thesis.** The core framework (Chapters 3–5) uses magnitude-based *unstructured* masks and learns
the per-layer ratio, which made exhaustive evaluation possible; the deployment extension (Chapter 6) realises the
allocations as *physically structured* networks so that memory, MACs, latency and energy can be measured.

## 2.3 Layer-Wise Adaptive Pruning

A pruning method must also decide *how many* parameters to remove from each layer: the layer-wise sparsity
allocation. Uniform pruning applies one ratio everywhere, but small early layers are often fragile while large late
layers are redundant, so it over-prunes some layers and under-prunes others; the allocation itself is a first-order
factor in pruned-model quality [@lee2021lamp; @liu2022randompruning; @gadhikar2023random].

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

**What remains less studied** is how a *learned* allocation compares with cheap deterministic allocations at the
exact same sparsity and under identical recovery (Research Question 1).

## 2.4 Magnitude-Based and Deterministic Pruning

**Magnitude pruning** removes the weights of smallest absolute value. Despite its simplicity it remains one of the
strongest and most widely used baselines [@han2015learning; @blalock2020state]. It requires no data, no gradients and
no search, and it combines easily with any allocation rule.

Variants differ in the *scope* over which magnitudes are compared:

- **Layer-wise magnitude pruning** removes a fixed fraction within each layer, so it needs an external allocation.
- **Global magnitude pruning** compares magnitudes across all layers. Because weight scales differ between layers, it
  can prune some layers almost completely, which is especially harmful in small layers.

Systematic evaluations repeatedly find that carefully tuned simple methods are hard to beat. ShrinkBench documented
that many papers compare against weak or inconsistently configured baselines, and that simple magnitude pruning
remains competitive with more complex methods when evaluated consistently [@blalock2020state]. Re-examinations of
structured pruning found that training a pruned architecture from scratch can match fine-tuning an inherited one,
which questions how much the specific inherited weights matter [@liu2019rethinking]. Benchmarks of sparse networks
on harder tasks have also shown that apparent gains can vanish under more demanding evaluation
[@liu2023sparsitymaycry].

**Deterministic methods** (magnitude pruning, global ranking, ERK, LAMP) are cheap, reproducible and free of search
hyperparameters, so a learned method must justify its cost against them, at the same sparsity rather than against
numbers reported elsewhere.

## 2.5 LAMP and Related Layer-Adaptive Methods

Layer-Adaptive Magnitude-based Pruning (LAMP) rescales each weight's magnitude by the magnitudes of the weights in
the same layer that would survive if it were pruned. This yields a score that reflects the model-level ℓ2 distortion
caused by pruning [@lee2021lamp]. A single global threshold on the LAMP score then produces a layer-wise allocation
automatically, without hyperparameters and with negligible computation. In its original evaluation LAMP consistently
outperformed popular layer-wise sparsity schemes on image classification, and remained strong under weight rewinding,
where the strongest other baseline fell behind even global magnitude pruning [@lee2021lamp]. As a near-free
*deterministic* answer to the problem a learned allocator addresses, LAMP is the natural benchmark for RL-based
allocation, which requires thousands of pruned-model evaluations.

Related layer-adaptive methods include:

- **Learned global ranking**, which learns a single ranking of filters across layers, from which architectures with
  different accuracy–latency trade-offs follow by pruning the lowest-ranked filters [@chin2020legr];
- **Outlier-based allocations for LLMs** [@yin2024owl];
- **Block-wise learned allocations** [@xu2024besa].

Like most allocators, these were compared under each paper's own fine-tuning protocol; Chapter 5 compares PPO and
LAMP before and after identical recovery.

## 2.6 Reinforcement Learning for Neural Network Compression

Compression can be framed as sequential decisions: an agent visits the layers one by one, chooses a ratio for each,
and is rewarded by the accuracy and cost of the compressed model. This connects compression to neural architecture
search, where an RL controller generated architectures rewarded by their validation accuracy [@zoph2017nas].

**AMC** made the formulation concrete for pruning [@he2018amc]. A DDPG agent receives an embedding of each layer and
outputs a continuous sparsity ratio. The reward combines accuracy with resource constraints. AMC reported better
accuracy than hand-crafted compression at a given FLOPs reduction, and measured speed-ups on a GPU and on mobile
hardware [@he2018amc]. Platform-aware methods such as NetAdapt and HAQ extended the idea by placing measured latency
or hardware feedback inside the optimisation loop [@yang2018netadapt; @wang2019haq].

Recent RL-based pruning research has explored several directions (Table 2.1):

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

These methods show that RL can find competitive allocations and handle non-differentiable objectives, but they are
usually evaluated on the final policy after method-specific fine-tuning, with hand-built rewards and against published
results under different recovery budgets (questions taken up in Section 2.14).

**Table 2.1. Comparison of representative pruning-allocation and deployment-aware pruning works.** "Struct." =
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
hyperparameters, handles discrete and continuous actions, and has mature implementations such as Stable-Baselines3
[@raffin2021sb3], which this thesis uses. Large empirical studies have qualified this picture:

- **Implementation details.** Code-level choices that are not part of the algorithm's description (for example
  value-function clipping and reward normalisation) can account for a large share of PPO's reported advantage over
  related methods [@engstrom2020implementation].
- **Design choices.** A study of more than fifty design choices in on-policy actor-critic methods found that many
  settings matter considerably and offered practical recommendations [@andrychowicz2021onpolicy].
- **Multi-agent settings.** PPO proved surprisingly strong in cooperative multi-agent games, where it had been assumed
  less sample-efficient than off-policy methods [@yu2022mappo].

PPO is thus a reasonable optimiser whose behaviour must still be examined empirically. Pruning allocation differs from
typical RL benchmarks: episodes are a handful of decisions, one per unit; the reward needs a noisy held-out
evaluation of a pruned model; and here the space is small enough to enumerate (1,296 policies), so search quality (how
highly ranked are the policies PPO visits?) can be measured separately from final-policy quality (how highly ranked is
the policy it ends on?).

## 2.8 Sensitivity-Aware Pruning

**Sensitivity**, how much a model's loss or accuracy changes when a parameter, structure or layer is pruned, is one of
the oldest ideas in pruning:

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

In most of this work sensitivity is *the pruning criterion*. Used instead as *guidance to a search* (a state feature
or an action prior), it can make the search more efficient or lead to a better final model, and these need not
coincide when selection, reward or recovery is the bottleneck. Studies usually report only final accuracy; Research
Question 2 measures the two effects separately, against shuffled and constant controls.

## 2.9 Search and Policy Selection in Pruning

Any search-based pruning method must decide *which* of the policies encountered during search is returned. The
policy an RL agent ends on, the terminal or greedy policy, is not necessarily the best one it saw. The issue is well
known in neighbouring fields:

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

For RL pruning, what the agent *retains* can differ from what it *finds*, and selection needs validation data separate
from the search signal. The thesis measures the rank of the terminal policy against the best visited one and
introduces archive-based selection on a held-out split (Research Question 3).

## 2.10 Post-Pruning Recovery and Fine-Tuning

Pruning usually reduces accuracy, which is then partly or wholly recovered by further training. The recovery
procedure is not a detail:

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

A comparison before recovery measures immediate damage, and one after recovery with different budgets partly
measures the budgets; only **identical** recovery isolates the allocation, so the thesis fixes the recovery protocol
and mask across methods (Research Question 5).

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

These methods need a hardware model or measurement loop per device, and their benefit depends on enough
*accuracy-feasible* headroom. The thesis treats that headroom as an empirical question to answer before any
hardware-aware optimisation (Chapter 6), checking whether accuracy-feasible configurations offer enough room for a
hardware-aware RL agent to exploit.

## 2.12 Parameter Count vs MACs/FLOPs vs Real Latency

Network cost can be described at three related but not interchangeable levels:

1. **parameter count**, or parameter memory;
2. **arithmetic count**, as multiply–accumulate operations (MACs) or floating-point operations (FLOPs);
3. **measured runtime** on specific hardware.

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

**What is less common** is an *exhaustive* measurement, across *all* configurations of one search space, of how well
parameter and MAC reductions predict latency and how much accuracy-feasible headroom exists; Chapter 6 reports this for
all 1,296 structured configurations of two architectures.

## 2.13 Energy-Efficient and Sustainable Neural Network Inference

**The energy cost of machine learning** has become a research topic in its own right. Early analyses drew attention
to the energy and carbon cost of training large NLP models [@strubell2019energy]. Researchers were urged to report
efficiency alongside accuracy, with the label "Green AI" used for this agenda [@schwartz2020greenai]. Frameworks for
the systematic reporting of energy and carbon footprints followed [@henderson2020reporting]. Industrial analyses have
discussed the environmental implications of AI across the full model lifecycle [@wu2022sustainable]. They have also
argued that best practices can substantially reduce training energy and emissions [@patterson2022plateau]. Carbon
estimates for large models depend on lifecycle assumptions as much as on measured energy [@luccioni2023bloom].

**Inference** has received growing attention because it is repeated for every query:

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

The thesis therefore reports only directly measured package energy, never TDP- or model-based estimates, and reports
energy as unresolved where measurement could not be qualified. Inference energy on one processor is only one component
of a model's environmental footprint, and converting it into carbon or lifecycle impact needs further data and
assumptions [@luccioni2023bloom; @wu2022sustainable]; the thesis makes no environmental or carbon claims and uses
"sustainability" only for reduced memory, computation and measured operational energy.

## 2.14 Research Gap

The literature establishes that layer-wise allocation is important, that RL can produce competitive allocations,
that sensitivity carries useful information, and that compression metrics do not straightforwardly become deployment
efficiency. Six narrower questions remain open.

1. **Adaptive allocation versus strong deterministic allocation, under controlled conditions.** Allocation matters
   [@lee2021lamp; @liu2022randompruning; @yin2024owl], but it is unclear how a learned PPO allocation compares with
   Uniform, Random, global magnitude, ERK and LAMP built at exactly the sparsity the agent chose.
   → *Research Question 1.*

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

Chapter 3 describes the framework that emerged, distinguishing original components from those introduced in response
to diagnostic results.
