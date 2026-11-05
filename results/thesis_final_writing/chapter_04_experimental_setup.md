# Chapter 4 — Experimental Setup

## 4.1 Experimental Design Overview

Chapter 3 defined the framework. This chapter specifies how it was evaluated, in enough detail for every experiment
to be reproduced and audited. It reports no results. A fixed design constant or a pre-registered stopping outcome is
stated where the design cannot be understood without it. Outcomes appear in Chapter 5 (core framework, Phases 3–6)
and Chapter 6 (deployment extension).

**Pre-registration.** Every phase committed its conditions, seeds, metrics, statistical tests and decision criteria to
the repository before any confirmatory run. Each phase treats the earlier phases as frozen evidence, read only and
verified by hash.

**Preparatory phases.** Two phases precede the experiments reported here:
- **Phase 1** audited an earlier pipeline. It found that the test set had been used for reward, sensitivity and epoch
  selection, and labelled that pipeline superseded. All reported experiments follow the leakage-guarded protocol
  adopted after the audit (Section 4.3).
- **Phase 2** fixed the splits, trained the dense references and verified the shared code library.

**Two experimental blocks:**
1. **Core framework (Phases 3–6).** Mask-based pruning on six architecture–dataset settings, run on a desktop CPU.
2. **Deployment extension (Paper 3, Phases 7A and 7B, and an energy confirmation study).** Physically compacted
   networks, measured on a laptop CPU.

## 4.2 Datasets

The experiments use CIFAR-10 and CIFAR-100 [@krizhevsky2009cifar].

**Data.** Both datasets consist of 32×32 RGB images: 50,000 official training images and 10,000 official test
images. CIFAR-10 has 10 classes. CIFAR-100 is used with its 100 fine labels.

**Preprocessing.** Images are scaled to [0, 1] and normalised per channel. The constants are computed on the official
50,000-image training file; no test image enters them.

| Dataset | Mean (R, G, B) | Standard deviation (R, G, B) |
|---|---|---|
| CIFAR-10 | 0.4914, 0.4822, 0.4465 | 0.2470, 0.2435, 0.2616 |
| CIFAR-100 | 0.5071, 0.4865, 0.4409 | 0.2673, 0.2564, 0.2762 |

**Augmentation.** All training uses random 32×32 crops with 4-pixel zero padding and random horizontal flips. This
covers dense training (Phase 2) and every fine-tuning or recovery run (Phases 6, 7A, 7B and Paper 3). Evaluation is
never augmented. Pruning without fine-tuning (Phases 3–5) involves no training.

**Which experiments use which dataset:**
- **Both datasets:** Phases 2–6, and the Paper 3 latency tables.
- **CIFAR-10 only:** the Paper 3 accuracy and proxy experiments, Phase 7A, Phase 7B and the energy confirmation study.

## 4.3 Dataset Partitioning and Validation Protocol

### 4.3.1 Training, validation and test

For both datasets, a fixed permutation of the 50,000 training indices (`torch.randperm`, seed 42) assigns:
- 45,000 images to **training**;
- the last 5,000 to **validation**.

The official 10,000 images are the **test** set. CIFAR-10 and CIFAR-100 use identical index lists, recorded with
hashes in split manifests.

### 4.3.2 Two validation configurations

The validation set is split into two class-stratified, disjoint parts with fixed roles:
- **VAL-RL** (reward split) drives sensitivity, rewards and PPO training.
- **VAL-SELECT** (selection split) only chooses among candidates the search has already produced.

The sizes changed once (Table 4.1). The change was declared in the Phase-5 design commitments before any Phase-5
data existed.

- **Configuration I (Phases 3–4): 3,000 / 2,000.** Each class gives about 60% of its validation images to VAL-RL.
  Per-class sizes use largest-remainder rounding, and positions are drawn by per-class permutation (seed 20260928).
  The phase records call these splits V_RL and V_SELECT.
- **Configuration II (Phase 5 onward): 2,500 / 2,500.** The same rule with seed 20260930. Per class, the two halves
  differ by at most one image.

**Why the configuration changed.** The Phase-5 design review found a bias in Phase 4: its final search variant had
been chosen by the VAL-SELECT rank of policies that the archive had itself selected on VAL-SELECT. Phase 5 removed the
bias in three ways:
1. a new balanced split;
2. conditions fixed in advance, so no method was chosen from data;
3. no inference about conditions from the VAL-SELECT metrics of VAL-SELECT-selected policies.

**Overlap with configuration I.** Of the 3,000 configuration-I VAL-RL images, 1,521 (CIFAR-10) and 1,515 (CIFAR-100)
fall into the new VAL-SELECT. Sensitivity was therefore recomputed rather than reused.

**Naming.** Throughout the thesis, VAL-RL and VAL-SELECT name *roles*, and the configuration is stated with every
accuracy. Accuracies from different configurations, or from different splits, are never compared.

**Table 4.1. Datasets and split configurations.**

| Item | CIFAR-10 | CIFAR-100 |
|---|---|---|
| Image size / classes | 32×32 RGB / 10 | 32×32 RGB / 100 (fine) |
| Training / full validation | 45,000 / 5,000 | 45,000 / 5,000 |
| Configuration I (Phases 3–4): VAL-RL / VAL-SELECT | 3,000 / 2,000 (seed 20260928) | 3,000 / 2,000 (seed 20260928) |
| Configuration II (Phase 5 on): VAL-RL / VAL-SELECT | 2,500 / 2,500 (seed 20260930) | 2,500 / 2,500 (seed 20260930) |
| Test | official 10,000 | official 10,000 |

### 4.3.3 Use of each split by phase

Table 4.2 gives the role of every split in every phase. Three rules hold throughout:
- training images only update weights;
- the reward split never selects the returned policy;
- the test set never informs a search, selection or design choice.

**Table 4.2. Data-split usage by phase.**

| Phase | Training | VAL-RL | VAL-SELECT | Test |
|---|---|---|---|---|
| 2 Dense references | weight updates | — (full validation selects the epoch) | — | once per hashed reference |
| 3 | not used | I: sensitivity, reward, landscape, exploration metrics | not used | once per frozen final policy and matched baseline; confirmatory |
| 4 | not used | I: reward, training, search metrics, Stage-A/B selections | I: archive re-ranking, held-out ranks, Stage-C selection | locked until all selections and final policies were frozen; confirmatory |
| 5 | not used | II: sensitivity, choice of τ, reward, training, archive candidates | II: selection among frozen candidates only | after the final-policy list was committed; confirmatory |
| 6 | fine-tuning | II: recorded per epoch; no selection | II: recorded per epoch; no selection | after the 480 final checkpoints were committed; confirmatory |
| Paper 3 | recovery | II: recovered accuracy, proxy validation, threshold calibration | II: stored, not analysed | not read |
| 7A | recovery | II: **primary** accuracy criterion | II: secondary | not used |
| 7B | proxy and final recovery | II: sensitivity, proxy, reward, archive | II: held out; **final** accuracy criterion | never read |
| 7B energy confirmation | — | — | 7B values, read only | — |

**Two consequences for interpretation:**
- **Phase 7A.** The transferred policy was trained on VAL-RL rewards and selected on VAL-SELECT in Phase 5. Its
  validation accuracy may therefore be optimistically biased relative to the baselines. The Phase-7A pre-registration
  records this bias.
- **Phases 7A and 7B.** No frozen test protocol existed for the new structured models, so these phases report
  validation accuracy only.

## 4.4 Neural Network Architectures

The three networks are defined in the shared library (`src/models`). Where an implementation differs from the
canonical design, the implementation is described.

**SimpleCNN** (the historical network of the thesis):
- three 3×3 convolutions (padding 1, with bias) with 32, 64 and 128 filters, each followed by ReLU and 2×2
  max-pooling;
- flattening to 2,048 features;
- a 256-unit fully connected hidden layer with ReLU;
- an output layer of C units.

It has no batch normalisation and no dropout. The module names match the historical code, so the original checkpoint
loads unchanged.

**LeNet-5, adapted to CIFAR.** The LeNet-5 topology [@lecun1998lenet] for three-channel 32×32 input:
- two 5×5 valid convolutions with 6 and 16 filters, each followed by ReLU and 2×2 max-pooling;
- fully connected layers 400 → 120 → 84 with ReLU;
- an output layer of C units.

The adaptation uses ReLU and max-pooling instead of the original sigmoidal units and trainable subsampling.

**ResNet-8.** The CIFAR residual network [@he2016resnet] with n = 1, i.e. 6n + 2 = 8 weight layers:
- a 16-channel 3×3 stem convolution with batch normalisation;
- three single-BasicBlock stages with 16, 32 and 64 channels and strides 1, 2 and 2. Each block has two 3×3
  convolutions with batch normalisation;
- shortcuts: identity in stage 1, and a 1×1 projection with batch normalisation in stages 2 and 3;
- global average pooling and an output layer of C units.

Convolutions have no bias and use Kaiming-normal (fan-out) initialisation. A pre-registered fallback network for an
over-budget ResNet-8 was not needed.

**Table 4.3. Network architectures and pruning units (core framework).** Counts are CIFAR-10 / CIFAR-100. "Weights"
are the Conv2d and Linear weights, the denominator of sparsity (Equation 3.2).

| Architecture | Datasets | Pruning units (share of prunable weights) | Dense parameters | Weights (prunable) | Main role |
|---|---|---|---|---|---|
| SimpleCNN | C10, C100 | conv1 0.14%, conv2 2.99%, conv3 11.94%, fc1 84.93% | 620,362 / 643,492 | 619,872 / 642,912 (617,312) | all phases; the only network of Phase 7B |
| LeNet-5 | C10, C100 | conv1 0.74%, conv2 3.94%, fc1 78.78%, fc2 16.54% | 62,006 / 69,656 | 61,770 / 69,330 (60,930) | core phases 3–6 only |
| ResNet-8 | C10, C100 | stem 0.56%, stage 1 6.01%, stage 2 18.69%, stage 3 74.74% | 78,042 / 83,892 | 77,360 / 83,120 (76,720) | core phases; Paper 3; Phase 7A |

In every architecture, the output layer, biases and batch-normalisation parameters are never pruned.

## 4.5 Dense Reference Models

Each setting has one dense **reference model**, the starting point of every pruning method in that setting. The
references were trained in Phase 2 under a protocol committed before any training run:
- Adam, learning rate 10⁻³, β = (0.9, 0.999), ε = 10⁻⁸, no weight decay;
- cosine annealing to zero over 30 epochs, stepped per epoch;
- batch 128, cross-entropy loss, the augmentation of Section 4.2;
- seeds 0, 1 and 2. The seed sets the initial weights; a generator with seed + 10⁶ sets data order and augmentation;
- the epoch with the highest full-validation accuracy is kept. Its weights are hashed, and only then is the test set
  read, once.

**Reference rule.** The seed-0 model is the reference, fixed by rule before training. Seeds 1 and 2 only measure
dense-training variability.

**Exception.** The CIFAR-10 SimpleCNN reference is the historical checkpoint: Adam at 10⁻³ for 10 epochs, final epoch.
It was reused rather than retrained, and this recipe difference is disclosed.

**Limitation.** Epoch selection used the full validation set, which was later split into VAL-RL and VAL-SELECT. This
affects every method of a setting identically.

**Use.** Every later phase asserts the reference hash before use (Table 4.4). In Phases 6, 7A and 7B, a dense model
fine-tuned with the same protocol as the pruned models is an additional reference, separating recovery from the
effect of extra training. Its accuracy is measured within each phase and reported with that phase's results.

**Table 4.4. Dense reference models (accuracy, %).** The VAL-RL columns are the denominators A_b of retained accuracy
in Phases 3–4 (configuration I) and Phase 5 (configuration II).

| Setting | Checkpoint | SHA-256 | Full validation | Test | VAL-RL (I) | VAL-RL (II) |
|---|---|---|---|---|---|---|
| CIFAR-10 SimpleCNN | historical, 10 epochs | `ca28f834` | 77.32 | 77.03 | 76.9667 | 76.88 |
| CIFAR-10 LeNet-5 | seed 0 | `5ca6e0fc` | 64.32 | 64.96 | 64.3000 | 64.20 |
| CIFAR-10 ResNet-8 | seed 0 | `41fc5ee5` | 78.72 | 78.69 | 78.5333 | 78.92 |
| CIFAR-100 SimpleCNN | seed 0 | `d75dd19c` | 49.08 | 50.50 | 50.3667 | 48.96 |
| CIFAR-100 LeNet-5 | seed 0 | `8e1b1aee` | 32.58 | 32.28 | 33.3000 | 32.76 |
| CIFAR-100 ResNet-8 | seed 0 | `8b84bfad` | 43.48 | 43.99 | 43.7667 | 43.76 |

Paper 3, Phase 7A and Phase 7B start from the CIFAR-10 SimpleCNN and ResNet-8 references above. The Paper 3 latency
tables also use the CIFAR-100 references.

## 4.6 Pruning Units and Policy Spaces

### 4.6.1 Masked pruning units (core framework)

Each architecture has four units (Table 4.3), each consisting of one or more weight tensors. An action applies its
ratio r ∈ {0, 0.1, 0.2, 0.3, 0.4, 0.6} as **L1 unstructured** pruning to *each* tensor of the unit: the round(r·|t|)
smallest-magnitude weights of tensor t are masked to zero, and the shapes are unchanged (Section 3.3).

| Architecture | Units |
|---|---|
| SimpleCNN | three convolutions, and the 256-unit hidden layer |
| LeNet-5 | two convolutions, and two hidden fully connected layers |
| ResNet-8 | the stem convolution, and three **residual-stage weight groups** |

The ResNet-8 stage groups are defined as follows:
- stage 1: the two block convolutions;
- stages 2 and 3: the two block convolutions plus the 1×1 projection.

A stage action masks the same fraction of weights in each of these tensors. It removes no stage, block or channel.

**Destructive action.** For the exploration endpoints, a policy is *destructive* if it prunes the setting's most
sensitive unit by 40% or more. The most sensitive unit has the highest raw sensitivity (Equation 3.8), with ties going
to the earliest unit. The rule was fixed before any PPO run.

### 4.6.2 Physically compacted structured units (deployment extension)

Here a ratio removes whole channels or neurons. A unit keeps k = max(1, round((1 − r)·C)) channels, those with the
largest L1 norm on the dense model (Equation 3.17), and the network is rebuilt with smaller layers.

**SimpleCNN** (C = 32, 64, 128, 256):
- conv1–conv3 remove output filters. The next layer loses the matching inputs; after conv3, the first fully connected
  layer loses 16 flattened columns per channel.
- fc1 removes hidden neurons, and the output layer loses the matching inputs.

**ResNet-8** (C = 16, 16, 32, 64):
- **Stem: a coupled residual pruning group.** It sets the stage-1 residual width. One shared channel set cuts every
  tensor that writes or reads the residual stream before stage 2:
  - writers: the stem convolution and the second stage-1 convolution, each with its batch normalisation;
  - the identity shortcut of stage 1;
  - readers: the inputs of the first stage-1 convolution, the first stage-2 convolution and the stage-2 projection.

  Channels are ranked by the summed L1 norms of the two writing filters.
- **Stages 1–3: block-internal channel groups.** These are the outputs of the block's first convolution and its batch
  normalisation, and the matching inputs of the block's second convolution.
- The stage-2 and stage-3 output widths and the output layer are not pruned.

**The masked and structured ResNet-8 units are different objects.** For example, the masked stage-2 unit covers three
convolutions, whereas the structured stage-2 unit removes only block-internal channels. A masked ResNet-8 policy
therefore maps onto structured units only approximately; Phase 7A records this as a limitation. For SimpleCNN the
mapping is direct.

**Integrity checks on every compacted model:**
- no masks remain;
- parameter and MAC counts equal the analytic values;
- logits agree with the masked-equivalent model within 10⁻⁴;
- outputs are identical after reloading.

### 4.6.3 Policy spaces and enumeration

Every setting has 6⁴ = 1,296 policies. The thesis uses this space in three distinct ways:

1. **PPO search.** One run samples 512 episodes, so it evaluates at most 512 policies, about 40% of the space. The
   agent sees only the rewards of the policies it samples.
2. **Exhaustive diagnostic enumeration** of masked policies without fine-tuning. All 1,296 policies were evaluated on:
   - configuration-I VAL-RL (Phase 3);
   - configuration-I VAL-SELECT (Phase 4);
   - both configuration-II splits (Phase 5).

   These **landscapes** are an exact cache for PPO's terminal evaluations and the reference that defines each policy's
   rank. They are never computed on the test set.
3. **Structured enumerations.**
   - Paper 3 timed all 1,296 structured shapes for SimpleCNN and ResNet-8, each with 10 and 100 output classes: four
     tables, 5,184 shapes.
   - Phase 7B recovered all 1,296 structured SimpleCNN policies with a 2-epoch protocol, as a proxy cache.

**Not enumerated:**
- **Fully recovered accuracy.** An exhaustive 5-epoch landscape planned in Paper 3 (5,184 policies) was stopped for
  compute after 126 CIFAR-10 SimpleCNN policies, which serve only as pilot data. Fully recovered accuracy is therefore
  known only for sampled or selected policies.
- **Energy.** Measured only for the models named in Section 4.18.

## 4.7 Baseline Pruning Methods

### 4.7.1 Core-framework baselines

The baselines are one-shot methods from `src/pruning`. Each is calibrated to an exact number K of zeroed weights,
taken from the prunable tensors only, so the output layer stays dense. Integer per-tensor counts use largest-remainder
rounding so that they sum to K.

- **Uniform:** every prunable tensor loses the fraction K/N of its smallest-magnitude weights, where N is the number of
  prunable weights. Deterministic.
- **Random:** uniform allocation, with the removed weights drawn at random inside each tensor. Ten mask seeds
  (1000–1009) are used, summarised by their mean accuracy.
- **Global magnitude:** the K smallest-magnitude weights over all prunable tensors. Deterministic. It is a standard
  reference point in pruning evaluations [@blalock2020state].
- **ERK** [@evci2020rigl]: tensor t receives density d_t = min(1, ε·(sum of its dimensions)/n_t), with ε solved so
  that N − K weights are kept. Saturated tensors are fixed dense and ε is re-solved. Within each tensor the
  smallest-magnitude weights are removed. Deterministic.
- **LAMP** [@lee2021lamp]: each weight receives the score w_(u)² / Σ_{v≥u} w_(v)² within its magnitude-sorted tensor,
  and the K smallest scores network-wide are removed. Every tensor keeps its largest weight. Deterministic.

The methods differ in mechanism:
- LAMP and ERK are layer-adaptive allocation rules;
- uniform and random are fixed allocations;
- global magnitude is one network-wide threshold.

They are referred to by name, not collectively as heuristics.

### 4.7.2 Deployment-extension comparators

These comparators are kept separate from the core baselines:
- **Uniform-Structured:** one common channel ratio for all four units.
- **LAMP-Structured:** a LAMP allocation translated into per-unit structured ratios.
- **Old PPO (Phase 7B only):** the frozen Phase-7A structured transfer of the Phase-5 policy.
- **New structured PPO (Phase 7B only):** the policy learned directly in the structured space.

### 4.7.3 Matching rules

The matching rule differs between comparisons (Table 4.5). In the core framework, "matched" always means an
**identical zero count**, which is the same as identical sparsity ρ. In the extension, MACs or parameter counts are
matched, and only approximately, because channel counts are integers.

**Table 4.5. Baseline methods and matching rules.**

| Comparison | Methods | Matching rule | Exactness |
|---|---|---|---|
| Phase 3 | P0 vs Uniform, Random, Global, ERK, LAMP | zero count of each PPO run's final model; PPO chooses its own sparsity, so there is no common target | exact, asserted |
| Phase 3, descriptive | the five baselines | fixed sparsity grid 10–60% | curves only; never treated as matched |
| Phase 4 | F, K vs LAMP, Global; F, K vs Uniform, ERK, Random | zero count of each distinct selected model | exact |
| Phase 5 | C0, C1, C2 vs LAMP, Global | zero count of each selected policy | exact (mismatch 0) |
| Phase 6 | PPO vs LAMP, Global, dense control | zero count per (setting, policy seed) | exact (mismatch 0, 120/120 triples) |
| Phase 7A | PPO vs Uniform-Structured | **MACs**: common ratio on a 0.001 grid minimising \|MACs − MACs_PPO\| (ties: smaller ratio) | approximate; residual gap recorded |
| Phase 7A | PPO vs LAMP-Structured | Phase-6 LAMP mask at PPO's zero count; per-unit sparsity realised as structured ratios | approximate; not MAC-matched |
| Phase 7B | new PPO vs Uniform-Structured | **MACs**, 0.001 grid over [0, 0.9] | approximate |
| Phase 7B | new PPO vs LAMP-Structured | **parameter count**: LAMP sparsity on a 0.1% grid over [0, 90]%, realised structurally; closest parameter count (ties: smaller sparsity) | approximate |
| Phase 7B | new PPO vs old PPO | none (frozen Phase-7A checkpoint) | not matched |

## 4.8 PPO Training Configuration

Every agent uses Stable-Baselines3 2.8.0 [@raffin2021sb3]. It runs PPO [@schulman2017ppo] with generalised advantage
estimation [@schulman2016gae] and the hyper-parameters of Table 3.7, identical in every phase. Table 4.6 shows what
differs.

**Fixed in every phase:**
- **Threads.** One thread per run and per evaluation. In Phases 3–6 the runs were spread over four worker processes.
- **Evaluation cache.** Terminal evaluations come from the exact landscape. The environment still creates the
  evaluation data loader at each terminal step, so the random stream matches an uncached run; the implementation
  checks verified this.

**Table 4.6. PPO configuration by phase.**

| Item | Phase 3 | Phase 4 | Phase 5 | Phase 7B |
|---|---|---|---|---|
| Network; lr; n_steps; batch; epochs | MLP [64, 64] tanh; 3×10⁻⁴; 64; 32; 10 | same | same | same |
| γ; GAE λ; clip; entropy; value; grad norm | 0.99; 0.95; 0.2; 0.01; 0.5; 0.5 | same | same | same |
| Budget; threads | 2,048 steps = 512 episodes; 1 | same | same | same |
| Environment; state | masked; 7 features, sensitivity slot 0 | same | same | structured SimpleCNN; 8 features with centred sensitivity |
| Prior | P0–P3, uncentred, β = 0.5 | P0–P5; P2 centred, β = 1.0 | P2 (C0, C1); none (C2) | centred structured, β = 1.0 |
| Reward | R0 | R0, R1a, R1b, R2 | R0 (C0); R_c, τ = 0.98 (C1, C2) | R_7B, α = 4 |
| Evaluation cache | VAL-RL (I) landscape | same | VAL-RL (II) landscape | 2-epoch proxy landscape |
| Extra loss | none | elite behavioural cloning in S2 (K = 5, η = 0.1) | none | none |
| Returned policy | terminal | terminal (S0, S2) or archive (S1, S2A) | archive | feasible archive |
| Seeds | 300–319 | 400–419 | 500–519 | 900–904 (development); 700–719 (confirmatory) |

**Seeds.** The Phase 3–5 seeds were verified unused by any earlier PPO run. The one exception is a 64-timestep smoke
test with seed 300 in the Phase-2 checks, which had no experimental purpose and whose output was not recorded. Within a
phase, seeds are shared across conditions and settings, so comparisons are paired. No seed was removed, replaced or
added.

**Elite behavioural cloning (Phase 4).** S2 runs use a copy of the SB3 PPO update with one added term, η·𝓛_elite
(Equation 3.13). Its parameters are:
- the K = 5 best unique VAL-RL-reward policies seen so far;
- η = 0.1;
- activation once five unique policies exist.

**Methodological role.** It tested whether pulling the actor towards its best visited policies improves retention
*during* training, as a complement to selecting from an archive *after* training. The PPO network remains the decision
maker. With η = 0, the implementation is bit-identical to standard PPO.

**Scope.** It belongs to the final Phase-4 pipeline (S2A) and was dropped from Phase 5 onward.

## 4.9 Experimental Conditions by Phase

Table 4.7 summarises the design of each phase. The subsections define the conditions exactly.

**Table 4.7. Phase-by-phase experimental design.**

| Phase | Research purpose | Datasets | Architectures | Conditions | Seeds | Runs | Primary output |
|---|---|---|---|---|---|---|---|
| 3 | PPO vs baselines; sensitivity prior | C10, C100 | all three | P0, P1, P2, P3 | 300–319 | 480 | test accuracy vs matched baselines; exploration endpoints |
| 4 | reward, centred prior, retention | C10, C100 | all three | Stage A: 4 rewards; B: 6 priors; C: 4 search variants | 400–419 | 1,200 | stage selections; search quality; final pipeline vs control and LAMP |
| 5 | accuracy-constrained reward | C10, C100 | all three | C0, C1, C2 | 500–519 | 360 | constraint, search and accuracy endpoints |
| 6 | equal fixed-mask recovery | C10, C100 | all three | PPO, LAMP, global, dense control | fine-tuning 600–619 | 480 (+6 repeats) | accuracy before and after equal fine-tuning |
| Paper 3 | structured CPU latency | C10, C100 (latency); C10 (accuracy) | SimpleCNN, ResNet-8 | 1,296 shapes per table | recovery seed 600 | 5,184 shapes | latency tables; GO/NO-GO for hardware-aware PPO |
| 7A | structured transfer, measured energy | C10 | SimpleCNN, ResNet-8 | Dense, PPO-, Uniform-, LAMP-Structured | recovery seed 600 | 8 models; 5 sessions per architecture | accuracy, memory, MACs, latency, energy |
| 7B | structured, energy-aware PPO | C10 | SimpleCNN | new PPO; Dense, Uniform-, LAMP-Structured, old PPO | 900–904; 700–719 | 20 searches; 11 recovered models | accuracy, memory, MACs; planned latency and energy |
| 7B energy confirmation | direct energy of frozen 7B models | C10 | SimpleCNN | Dense vs new PPO | — | 6 qualification blocks; 10 sessions planned | qualification; planned paired energy |

### 4.9.1 Phase 3

Phase 3 compared PPO with the baselines across six settings (2 datasets × 3 architectures) under four conditions:

| Condition | Prior |
|---|---|
| **P0, plain PPO** | none |
| **P1, correct prior** | c_l = 0.5·S_l |
| **P2, shuffled** | S permuted per seed so that every unit receives another unit's value |
| **P3, constant** | every unit receives the mean of S |

With 20 seeds, the design has 480 runs. The returned policy is the terminal deterministic policy. Models are pruned
without fine-tuning, and the baselines are built after all PPO runs finish.

**Four separate questions:**
- **A.** PPO vs uniform.
- **B.** PPO vs global magnitude, LAMP, ERK and random.
- **C.** Does the correct prior improve exploration? The primary endpoint is the VAL-RL accuracy AUC.
- **D.** Does the correct prior improve the final model? The endpoint is test accuracy in excess of matched global
  magnitude, compared as P1 − P0.

**Label note.** "P2" is the *shuffled* prior in Phase 3 and the *centred* prior in Phases 4–5. The phase is always
stated.

### 4.9.2 Phase 4

Phase 4 used a staged design with mechanical, validation-only selections, on seeds 400–419.

- **Stage A, reward.** R0, R1a, R1b and R2 (Section 4.12), with P0 and the terminal policy: 480 runs.
  - A redesigned reward qualified only if it at least halved the cross-setting spread of final sparsity (with a
    seed-bootstrap upper bound below 1), **and** kept VAL-RL retention ≥ 0.95 in at least five settings.
  - Otherwise R0 was kept.
- **Stage B, prior.** 600 new runs under the Stage-A reward:

  | Prior | Coefficient |
  |---|---|
  | P0 | none |
  | P1 | c_l = 0.5·S_l (the Phase-3 prior) |
  | P2 | c_l = 1.0·(S_l − S̄) (centred) |
  | P3 | P2 permuted across units |
  | P4 | c = 1.0·mean_l \|S_l − S̄\| for every unit (pure conservatism, magnitude-matched) |
  | P5 | c_l = −1.0·(S_l − S̄) (sign-reversed) |

  P1 and P2 were each tested against P0 on reward-rank AUC. A prior was eligible if it was supported in at least three
  settings with none opposite. The eligible prior supported in more settings was carried forward (tie: larger mean
  effect size d_z). If neither was eligible, P0 was.
- **Stage C, search.** 120 new runs:
  - **S0:** terminal policy;
  - **S1:** archive rule on the S0 runs;
  - **S2:** elite behavioural cloning with the terminal policy;
  - **S2A:** elite behavioural cloning with the archive rule.

  The winner had the lowest mean, over settings, of the median VAL-SELECT rank of the selected policy.

**Stage outcomes.** The selections kept R0 (no redesigned reward qualified), chose prior P2 and chose search variant
S2A. They define:
- the final pipeline **F = (R0, P2, S2A)**;
- the control **K = (R0, P0, S0)**, the Phase-3 configuration on fresh seeds.

### 4.9.3 Phase 5

Phase 5 used configuration II and fixed three conditions in advance:
- **C0:** R0 + P2 + archive, selecting by the highest R0 score on VAL-SELECT. This is the Phase-4 pipeline without
  behavioural cloning.
- **C1 (proposed):** R_c with τ = 0.98, P2, and constraint-aware selection (Equation 3.14).
- **C2:** as C1, without a prior.

With seeds 500–519 the design has 360 runs. C1 − C2 is the sensitivity ablation, and C1 − C0 is the reward
comparison.

**Choice of τ.** τ was chosen before any run, by a rule applied to the VAL-RL landscapes only: the largest value in
{0.95, 0.97, 0.98, 0.99} giving every setting at least 20 feasible policies, including one with ≥ 10% sparsity. The
rule gave τ = 0.98.

### 4.9.4 Phase 6

Phase 6 is a controlled recovery experiment. It tests whether accuracy differences persist when every pruned model
receives the same fine-tuning budget at exactly the same sparsity.

**Methods:**
- **PPO:** the frozen Phase-5 C1 policy of each (setting, policy seed).
- **LAMP and global magnitude:** applied at PPO's exact zero count.
- **Dense control:** fine-tuned with the same protocol.

**Phase 6 contains no uniform, ERK or random method, and no Phase-5 C0 or C2 policy.** The pre-registration excluded
them to keep the primary comparison within the fixed compute budget.

**Run matrix.**
- 360 pruned runs plus 120 dense controls.
- The four runs of each (setting, policy seed) share the fine-tuning seed (policy seed + 100), the minibatch order and
  the augmentation. A stored data-stream hash checks this.
- Six reproduction runs repeat seed 500's PPO run in each setting.
- Masks are fixed and cannot regrow, and no checkpoint is selected (Section 4.13).

**Unit of analysis.** The policy seed, with n = 20 per setting. Several seeds selected the same policy, so a
cluster-level sensitivity analysis uses one value per distinct policy. It is tested only where there are at least six
distinct policies.

### 4.9.5 Paper 3

Paper 3 tested whether structured compression becomes CPU latency, through a sequence of pre-registered decisions:

- **H0, calibration.** Six SimpleCNN shapes were timed under three configurations (Section 4.17). The rule selected the
  primary configuration.
- **H1, validation.** 50 checks of structured compaction, MAC and FLOP counters, and structured sensitivity. The
  Phase-2 library checks were rerun on the laptop.
- **H2-A, latency tables.** All 1,296 structured shapes for SimpleCNN and ResNet-8, each with 10 and 100 output classes:
  4 tables, 5,184 shapes.
- **Recovered accuracy (CIFAR-10).**
  - A 1-epoch recovery proxy was validated against 5-epoch recovery on frozen samples: 156 SimpleCNN and 31 ResNet-8
    policy pairs. An architecture could proceed only if the Spearman correlation was ≥ 0.90.
  - The protocol's Youden-J threshold rule required at least three truly feasible validation pairs. ResNet-8 had
    fewer, which triggered the pre-registered fallback to a decision by the principal investigator. A pre-committed
    calibration on 48 frozen ResNet-8 policies then set the threshold, by a recall-first rule with a 0.005 margin.
  - A latency-room rule then required at least 20 proxy-feasible policies and one with a speed-up ≥ 1.10×.

**Stopping outcomes** (values in Chapter 6):
- SimpleCNN did not meet the proxy-validity criterion.
- ResNet-8 met it and passed calibration, but not the latency-room rule.

The rules therefore stopped hardware-aware PPO before any run, and **no hardware-aware reinforcement learning was
run.** Paper 3 contributes the latency tables and the cost-metric analysis built on them.

### 4.9.6 Phase 7A

Phase 7A uses CIFAR-10, SimpleCNN and ResNet-8, with four methods per architecture:
- **Dense:** the reference, recovered with the same protocol.
- **PPO-Structured:** the most frequent Phase-5 C1 selected policy over seeds 500–519 (ties: smaller action vector).
  Its action indices are read as structured ratios.
- **Uniform-Structured:** MAC-matched to PPO.
- **LAMP-Structured:** derived from the Phase-6 LAMP mask.

Every model receives the equal 5-epoch recovery with seed 600, and its energy is then measured in five sessions
(Section 4.18).

**Fixed criteria:**
- **Accuracy:** |A_method − A_dense,recovered| ≤ 0.5 pp on VAL-RL.
- **Consistent energy reduction:** lower energy than the same-session dense model in 5 of 5 sessions, **and** a median
  saving above a noise threshold of twice the dense model's between-session coefficient of variation.
- **Cases:**
  - **A:** both criteria hold. This is the primary success.
  - **B:** consistent energy reduction with more than 0.5 pp loss.
  - **C:** accuracy within 0.5 pp but no consistent reduction.
  - **D:** a baseline has the better trade-off.
- **Better trade-off:** either accuracy within 0.5 pp of the other method and lower energy in 5 of 5 sessions, or
  accuracy higher by more than 0.5 pp with median energy not higher.

### 4.9.7 Phase 7B

Phase 7B uses SimpleCNN on CIFAR-10 only.

**Stage A (development).** Completed before the final pre-registration:
- implementation;
- validation of the accuracy proxy and the energy surrogate (Section 4.18.3);
- the 2-epoch proxy landscape;
- tuning of α ∈ {0.5, 1, 2, 4} on seeds 900–904. The rule took the lowest median predicted energy of the selected
  policies, ties going to the smaller α, and gave α = 4.

β = 1.0, κ = 10, ε = 0.01, the 1% tie band and the selection rule were fixed in the design and not tuned.

**Stage B (confirmatory).** 20 seeds (700–719), all reported.
- **Agent:**
  - structured actions;
  - the eight-feature state of Table 3.4;
  - the centred prior from structured sensitivity, computed on VAL-RL with probes at {0.1, 0.2, 0.4, 0.6}.
- **Feasibility:** the 2-epoch proxy q̂ = A_2/A_2,dense with τ̂ = 0.98849. The proxy's target was 5-epoch VAL-RL
  accuracy within 0.5 pp of the recovered dense model.
- **Reward:** Equation (3.19).
- **Selection:** from the feasible archive (Section 4.11).
- **Recovery:** every final model receives the equal 5-epoch recovery: dense, the primary new-PPO policy, the other
  distinct per-seed selections, and Uniform- and LAMP-Structured. The old PPO checkpoint is reused read only.

**Joint deployment-efficiency success criterion.** The pre-registration names it *GREEN-SUCCESS*. A model succeeds only
if all five conditions hold at once:
1. VAL-SELECT accuracy at least the recovered dense VAL-SELECT accuracy minus 0.5 pp;
2. parameter-memory reduction > 0;
3. MAC reduction > 0;
4. latency reduced: the median is below dense, and the model is faster in 5 of 5 sessions;
5. measured energy reduced beyond noise: lower in 5 of 5 sessions, and a median saving above twice the dense model's
   between-session coefficient of variation.

There is no override. An empty archive counts as failure, and the success frequency over the 20 seeds is also
defined. The thesis calls this rule the **joint deployment-efficiency criterion**.

### 4.9.8 Phase 7B energy confirmation

This separate protocol was written after Phase 7B had closed. It does not alter Phase 7B, its verdicts or its failed
pilot criterion.

**Sequence:**
1. The five checkpoint hashes are verified.
2. The instrument is qualified on the dense model (Section 4.18.5).
3. Only after qualification: ten paired Dense and new-PPO sessions, five in each order, assigned by a seeded
   permutation.

**Energy confirmation rule.** All four conditions must hold:
1. the setup qualified;
2. the PPO model used less energy in at least 8 of the 10 sessions;
3. the median saving exceeded the noise threshold;
4. a leave-one-session-out check held for every omitted session.

## 4.10 Sensitivity Conditions and Controls

Sensitivity enters the core framework only through the prior: the state slot is 0 throughout Phases 3–5
(Section 3.5). The controls separate three explanations of an effect of the prior:

| Explanation | Control | Phase |
|---|---|---|
| layer-specific information | priors that keep the overall pressure but scramble its unit assignment: shuffled P2; permuted P3 | P2 in Phase 3; P3 in Phase 4 |
| generic conservatism | pressure without sensitivity information: constant P3; magnitude-matched P4 | P3 in Phase 3; P4 in Phase 4 |
| direction | sign-reversed P5 | Phase 4 |

The centred P2 of Phase 4 removes net conservatism by construction. The sensitivity-specific statement held in a
setting only if P2 beat **both** P3 and P4 there.

Phase 5 compares C1 with C2 under the same reward. Phase 7B ran no prior-free control.

**Sensitivity vectors:**
- configuration-I VAL-RL, frozen for Phases 3–4;
- recomputed on configuration-II VAL-RL in Phase 5;
- structured probes on configuration-II VAL-RL in Phase 7B.

None uses VAL-SELECT or test data.

## 4.11 Archive and Selection Protocols

**Phase 3.** The terminal deterministic policy. No archive is kept.

**Phase 4.** The archive records every unique sampled policy, with its VAL-RL reward and first episode.
- S0 and S2 return the terminal policy.
- S1 and S2A take the top 10 by VAL-RL reward and return the highest VAL-SELECT score under the same reward
  (Equation 3.12).

**Phase 5.** The steps follow a fixed order:
1. The archive of all 360 runs was frozen and hashed before any VAL-SELECT evaluation.
2. The candidates were then evaluated on VAL-SELECT, and each result had to equal the VAL-SELECT landscape.
3. Selection:
   - **C1 and C2:** Equation (3.14), with its tie-break and reported fallback;
   - **C0:** the highest R0 score on VAL-SELECT.
4. The final-policy list was committed before the test set was read.

**Phase 7B.** The archive holds every distinct proxy-feasible policy visited.
- **Per seed:** the lowest predicted energy is selected. Policies within 1% count as tied; ties go to higher proxy
  accuracy, then fewer MACs, then lower parameter memory, then the smaller action vector.
- **Primary new-PPO model:** the most frequent selection over the 20 seeds.
- **Enumeration reference:** the same rule applied to every proxy-feasible landscape policy. It is reported, but it is
  not a method.

VAL-SELECT plays no part in Phase-7B selection.

## 4.12 Reward Configurations

All rewards are computed on VAL-RL from the pruned network without fine-tuning; Phase 7B uses its 2-epoch proxy.

| Reward | Definition | Used in |
|---|---|---|
| R0 | 1.5·A_p/A_b + 0.01·ρ(%) + 0.05·D (Equation 3.7) | Phases 3 and 4; Phase 5 C0 |
| R1a | 1.5·A_p/A_b − κ_a·\|ρ − 50\| + 0.05·D | Phase 4 Stage A |
| R1b | 1.5·A_p/A_b − κ_b·\|ρ − ρ*_s\| + 0.05·D, with ρ*_s the largest sparsity retaining ≥ 95% of VAL-RL accuracy | Phase 4 Stage A |
| R2 | 1.5·(A_p − A_60)/(A_b − A_60) + 1.5·ρ/ρ_60 + 0.05·D, anchored on the maximal policy [5,5,5,5] | Phase 4 Stage A |
| R_c | constraint-first, τ = 0.98 (Equation 3.15) | Phase 5 C1, C2 |
| R_7B | constraint-first on the proxy, predicted energy saving, α = 4 (Equation 3.19) | Phase 7B |

κ_a and κ_b were each set before any run: the smallest value in {0.005, 0.01, 0.02, 0.04, 0.08} whose VAL-RL reward
optimum lies within ±5 pp of its target in all six settings.

Only R0 and R_c enter the mature core framework.

## 4.13 Recovery and Fine-Tuning Protocol

The Phase-6 protocol (Section 3.15) is used unchanged in every later phase:
- Adam at 10⁻⁴, β = (0.9, 0.999), ε = 10⁻⁸, weight decay 0, with a new optimiser per run;
- cosine annealing to 0, stepped per epoch;
- batch 128 on the 45,000 training images, with crop and flip;
- batch normalisation in training mode during fine-tuning and evaluation mode for every evaluation;
- one thread;
- no checkpoint selection: the final epoch is the result.

**E = 5.** It was fixed before any confirmatory run by a timing rule: E = 5 if the projected Phase-6 wall time was at
most 30 hours, otherwise E = 3. The projection, from a pilot that used no accuracy, was 29.44 hours. No 10-epoch
schedule was run.

**Masked models (Phase 6).** The mask is fixed by PyTorch's pruning reparameterisation [@paszke2019pytorch]. Three
things are asserted after every epoch: the mask hash, the absence of non-zero masked weights, and the per-layer zero
counts.

**Compacted models (Paper 3, 7A, 7B).**
- There is no mask. Instead, the tensor shapes after recovery must equal those before.
- Every model uses recovery seed 600 with data generator 600 + 10⁶, so all models of a phase see the same minibatch
  sequence.

**Proxies.**
- **Paper 3.** The 1-epoch proxy is exactly the first epoch of the 5-epoch protocol, because the schedule is stepped
  per epoch.
- **Phase 7B.** The 2-epoch proxy is a separate run with the cosine schedule over two epochs, on the same seed and data
  stream.

Final accuracy claims always use 5-epoch models.

## 4.14 Evaluation Metrics

Table 4.8 lists the metrics. Accuracy always names its split, and every reduction names its reference.

**"Sparsity" is not a synonym for all compression measures.** Four notions are kept apart:
1. **Masked weight sparsity ρ** (core framework). The percentage of zero weights over all Conv2d/Linear weights,
   including the never-pruned output layer (Equation 3.2). Shapes are unchanged.
2. **Channel removal ratio r_l** (extension). The fraction of a unit's channels removed, leaving k_l kept
   (Equation 3.17).
3. **Physical parameter reduction** (extension). 1 − P/P_dense over all parameters, including biases and batch
   normalisation. Phase 7B reports it as parameter memory in bytes.
4. **MAC reduction** (extension). 1 − MAC/MAC_dense.

The per-unit sparsity (zeros divided by the unit's weights) is used only to translate LAMP allocations into structured
ratios.

**Table 4.8. Evaluation metrics.** ↑ higher is better; ↓ lower is better.

| Group | Metric (symbol) | Definition | Phases | Better |
|---|---|---|---|---|
| Predictive | accuracy (A) | top-1 accuracy (%) on a named split | all | ↑ |
| | retained accuracy (q) | A / A_ref, with A_ref the dense or recovered dense model on the same split | 3–7B | ↑ |
| | accuracy difference (Δ) | A − A_ref, or method minus baseline (pp) | all | ↑ |
| | recovery; gap change ΔG | A_post − A_pre; (A_PPO − A_base) after minus before fine-tuning | 6 | ↑ |
| | retention gap | VAL-SELECT retention minus test retention | 6 | ↓ |
| | proxy accuracy (q̂) | A_E / A_E,dense on VAL-RL after an E-epoch recovery (E = 1 in Paper 3, 2 in 7B) | Paper 3, 7B | ↑ |
| Compression | masked sparsity (ρ) | Equation 3.2 | 3–6 | context |
| | parameter reduction | 1 − P/P_dense | Paper 3, 7A, 7B | ↑ |
| | parameter memory | Σ numel × element size (bytes, float32) | 7B | ↓ |
| | serialised size | saved state-dict bytes | 6, 7A, 7B | ↓ |
| Computation | MACs; FLOPs | Conv2d/Linear multiply–accumulates per image, closed form checked against hooks; FLOPs = 2 × MACs | Paper 3, 7A, 7B | ↓ |
| Deployment | latency (T) | median per-inference batch-1 time (ms) | Paper 3, 7A, 7B-EC | ↓ |
| | speed-up; latency gain (g_T) | T_ref / T; 1 − T/T_identity within one table | Paper 3, 7A | ↑ |
| Energy | E | package J / N × 1,000 for a block of N inferences | 7A, 7B-EC | ↓ |
| | energy per inference | package J / N | 7A, 7B-EC | ↓ |
| | energy saving | 100·(1 − E/E_dense), same-session dense model | 7A, 7B-EC | ↑ |
| | EDP | (J/inference × latency) normalised to the same-session dense model; not a carbon metric | 7A, 7B-EC | ↓ |
| | predicted energy (Ê) | Equation 3.18; inside the search only, never a result | 7B | — |
| Search | reward rank; rank score | rank among the 1,296 landscape policies (1 = best; ties get the minimum rank); (1,296 − rank)/1,295 | 3–5 | ↓; ↑ |
| | constrained rank | rank under R_c on the VAL-RL landscape (common yardstick for C0–C2) | 5 | ↓ |
| | best-seen and selected rank | minimum rank over the 512 sampled policies; rank of the returned policy | 4, 5 | ↓ |
| | top-k reach; first entry | best-seen or selected policy within the top k (100, 50, 20, 10, 5, optimum); first episode reaching it (censored at 513) | 4, 5 | ↑; ↓ |
| | trajectory AUC | trapezoidal integral over episodes of the sampled policy's rank score, VAL-RL accuracy or utility, normalised by the episode span | 3–5 | ↑ |
| | destructive frequency | share of sampled policies that are destructive (Section 4.6.1) | 3–5 | ↓ |
| | frontier regret; feasible-sample rate | best accuracy at equal or higher sparsity minus own accuracy; share of samples with q ≥ τ | 4; 5 | ↓; ↑ |

## 4.15 Statistical Analysis and Decision Criteria

### 4.15.1 Paired tests (Phases 3–6)

**Pairing and test.** Comparisons are paired by seed, with n = 20 per setting. The primary test is the **exact
two-sided sign-flip test** over all 2²⁰ sign patterns.

**Reported with every comparison:**
- mean, standard deviation and median;
- the 95% t-interval (df 19);
- a bootstrap 95% interval (10,000 resamples in Phase 3);
- the effect size d_z;
- the exact Wilcoxon p (Phases 3 and 4);
- wins, ties and losses.

**Multiplicity.** Holm's step-down correction is applied within each pre-registered family.

**Outcome of one comparison:**
- **supported:** the hypothesised direction, a Holm-adjusted p < 0.05, **and** a 95% t-interval excluding 0;
- **opposite:** the same criteria in reverse;
- **inconclusive:** otherwise.

**Cross-setting classes (Phases 3–4):**
- **STRONG:** supported in ≥ 5 of 6 settings and opposite in none;
- **MODERATE:** supported in 3–4 and opposite in none;
- **LIMITED:** supported in 1–2, or in ≥ 3 with one opposite;
- **NOT SUPPORTED:** supported in none.

A statement opposite in ≥ 3 settings is also labelled REVERSED. No subset result is described as general.

**Families by phase:**
- **Phase 3:** families A (6), B (24), C (6), C-specificity (12), one family per C-secondary endpoint, and D (6).
  McNemar counts are supportive only.
- **Phase 4:** reward, prior, search and final-model families. It also includes a seed bootstrap for Stage A and a
  **one-sided non-inferiority test** of F against LAMP with a 0.5 pp margin: a Holm-corrected sign-flip test on
  d + 0.5, plus a lower t-bound above −0.5.
- **Phase 5:**
  - C1 − C0 on test accuracy (6);
  - C1 − C2 on four primary exploration endpoints (24): reward-rank AUC, VAL-RL accuracy AUC, and the first episodes
    in the top 20 and top 10;
  - C1 − C2 secondary endpoints (18);
  - C1 − LAMP and its gap change (6 each).

  C1 − global magnitude was pre-registered as **descriptive**.
- **Phase 6:**
  - recovery, dense-control and pruning-specific recovery families;
  - PPO vs LAMP and PPO vs global (12 each);
  - retention gap.

  Architecture dependence compares different experiments, so it uses an unpaired permutation test (100,000
  permutations) with a Welch interval. A seed-paired version is a sensitivity analysis.

### 4.15.2 The ±0.5 pp rule of Phase 6

Phase 6 reports statistical support and practical size separately, with a fixed 0.5 pp margin:
- a difference is *practically relevant* only if |mean| ≥ 0.5 pp;
- two methods lie **within the ±0.5 pp margin** if the 95% t-interval of their paired difference lies inside
  (−0.5, +0.5) pp;
- a supported difference below 0.5 pp is *statistically detectable but small*.

This is a **pre-registered confidence-interval inclusion rule**, not a formal statistical equivalence test. A formal
test, such as two one-sided tests with a justified margin and stated error rate, was not pre-registered. The
pre-registration calls the rule "practically equivalent". The thesis writes "within the ±0.5 pp margin" so that it is
not read as a formal equivalence claim.

### 4.15.3 Deployment-extension statistics

**Paper 3** is descriptive and rank-based:
- Spearman rank correlations of latency gain with parameter reduction and with MAC reduction, over all 1,296 shapes;
- the fraction of policy pairs in which the policy with more parameter reduction is slower;
- the H0 signal-to-noise rule.

Proxy validity uses Spearman correlation, error, bias, top-10 overlap and a confusion matrix.

**Phases 7A and 7B:**
- summaries over sessions: median, mean, standard deviation and interquartile range;
- each method against dense: an **exact one-sided Wilcoxon signed-rank test** on five paired sessions. Its minimum
  attainable p is 0.031, so p-values are secondary to the 5-of-5 consistency rule and the noise threshold;
- between methods: exact two-sided Wilcoxon tests, reported descriptively.

**Energy confirmation.** An exact Wilcoxon test and a sign test on ten pairs were pre-registered; t-tests were
excluded.

No other significance test was used.

### 4.15.4 Pre-registered decision rules

**Table 4.9. Pre-registered decision criteria.** The commit is the one that first added the governing file.

| Experiment | Frozen rule | Purpose | Source (commit) |
|---|---|---|---|
| Phase 2 | seed-0 reference; best full-validation epoch; test once after hashing | dense references | `cpu_dense_training_protocol.md` (`366d21ae`) |
| Phase 3 | outcome and class rules; four separate questions | PPO vs baselines; prior | `phase3_preregistration.md` (`2dbcc770`) |
| Phase 4 | mechanical Stage A–C selections; criteria A–E (reward, specific prior, search, final model, LAMP) | reward, prior, retention | `phase4_preregistration.md` (`f52e87d9`) |
| Phase 5 | τ rule; criteria A–F (accuracy safety, compression, search, sensitivity, final model, LAMP) | constrained reward | `phase5_preregistration.md` (`3cbaec1a`) |
| Phase 6 | E timing rule; criteria A–F (fairness, recovery, competitiveness, generalisation, sparsity preservation, reproducibility); ±0.5 pp rule | equal recovery | `phase6_preregistration.md` (`b1b5396f`) |
| Paper 3 H0 | SNR ≥ 5, reduction ≥ 10%, positive in every session; order A, C, B | primary latency configuration | `h0_protocol.md` (`c2dcb22a`) |
| Paper 3 H2, proxy | table protocol; proxy GO iff Spearman ≥ 0.90; Youden threshold with fallback | latency tables; proxy | `paper3_preregistration.md` (`45936846`); `paper3_proxy_protocol.md` (`ec7a7f27`) |
| Paper 3 ResNet-8 | ≥ 5 feasible calibration policies; threshold rule; latency room (≥ 20 feasible, one ≥ 1.10×) | GO/NO-GO for hardware-aware PPO | `resnet8_threshold_calibration_protocol.md` (`5b127804`) |
| Phase 7A | ±0.5 pp on VAL-RL; consistent energy reduction; Cases A–D; pilot CV ≤ 3% | transfer and energy | `preregistration.json` (`b8c27665`) |
| Phase 7B Stage A | proxy acceptance; surrogate acceptance; calibration pilot CV ≤ 4%; α rule | development | `phase7b_design.md` (`f5abe039`) |
| Phase 7B Stage B | joint deployment-efficiency criterion; comparison rule; pilot CV ≤ 3%, N doubled once, then STOP | confirmatory | `phase7b_final_preregistration.json` (`0e101d55`) |
| 7B energy confirmation | qualification CV ≤ 3%, one attempt; energy confirmation rule | direct energy | `protocol.json` (`df233be0`) |

## 4.16 Hardware and Software Environment

All experiments ran on CPUs with CPU-only PyTorch builds; no GPU was used (Table 4.10).

**Table 4.10. Hardware and software environments.**

| Item | Core framework (Phases 2–6) | Extension (Paper 3, 7A, 7B, energy confirmation) |
|---|---|---|
| CPU | Intel Core i5-6400 @ 2.70 GHz, 4 logical CPUs (desktop) | Intel Core i7-1260P (laptop): 4 performance cores × 2 threads (logical 0–7), 8 efficiency cores (8–15) |
| Memory | not recorded | 15.7 GB |
| OS | Windows 11, build 22621 | Windows 11 Home, build 26200 |
| Python; PyTorch | 3.11.9; 2.7.1+cpu (torchvision 0.22.1+cpu) | 3.11.9; 2.7.1+cpu with oneDNN 3.7.1 and oneMKL 2025.1 |
| Other libraries | Stable-Baselines3 2.8.0, sb3-contrib 2.8.0, Gymnasium 1.2.3, NumPy 2.2.2, pandas 2.2.3, ptflops 0.7.5, psutil 7.2.2 | Stable-Baselines3 2.8.0 in the pinned environment; the 7B records do not store package versions separately |
| Threads | 1 per run (dense training: 4) | 1 intra-op, 1 inter-op |
| Power | mains | AC required; Balanced plan |

Package versions are pinned in `requirements_locked.txt`. Every phase records its environment and commit, and the
integrity checks verify them.

**Results do not transfer between machines.** No desktop latency or energy value is used. Every latency and energy
number comes from the laptop.

## 4.17 Structured Pruning and Latency Protocol

**Measurement core** (Paper 3, reused in Phases 7A and 7B):
- **Inference.** Batch 1, float32, NCHW, `eval()` and `inference_mode()`, on one preallocated, seeded input tensor.
- **Threads and affinity.** One intra-op and one inter-op thread. The affinity and the OpenMP and MKL thread variables
  are set before PyTorch is imported. The process is pinned to logical CPU 2, a performance-core thread other than
  core 0, which avoids the scheduling noise of the hybrid design.
- **Timing.** `time.perf_counter_ns()` per pass, with garbage collection disabled. Model construction, loading and data
  handling are never timed.
- **Gates.** AC power is required. System load must be ≤ 15% over 5 s before and after each session, and no other
  experiment may run.

**H0 calibration.** Six SimpleCNN shapes were timed under three configurations:

| Configuration | Batch | Threads | Warm-up / timed passes |
|---|---|---|---|
| A | 1 | 1, on CPU 2 | 50 / 300 |
| B | 1 | 4, one per performance core | 50 / 300 |
| C | 128 | 4 | 10 / 50 |

Passes were split into five rounds, each opening with a dense anchor block, over five fresh-process sessions. The rule
of Section 4.9.5 selected configuration A.

**H2-A latency tables:**
- **Passes.** Configuration A: 50 warm-up and 300 timed passes per shape.
- **Sessions.** Five fresh-process sessions per table. Each times all 1,296 shapes once, in a seeded random order.
- **Anchors.** A dense anchor block precedes the first shape and follows every 25 shapes.
- **Drift criterion.** A session whose anchors drift by more than 10% from their median is rejected and repeated once.
  Four attempts were repeated, and all 20 accepted sessions met the criterion.
- **Aggregation.** A shape's latency is the median over sessions of its per-session median. Also stored: the 95th
  percentile, interquartile range, throughput, speed-up and latency gain against the identity shape of the same table,
  and an anchor-normalised latency as a secondary column.
- **Freeze.** The tables were hashed and committed before use. Latency depends on shape only.

**Calibration-day versus table latency.** On the table days, the absolute dense latency was about 16% higher than on
the H0 calibration day. This cross-occasion check is kept as a recorded failure. Within each table, the identity shape
agrees closely with the dense anchors, so relative latency within a table is the analysed quantity. Later sessions
compare against a dense model measured in the same session, never against absolute table values. The Phase-7B
surrogate uses table latency as a feature fitted to energies measured on the same machine.

**Phase 7A latency.** The median per-inference time over the same N inferences as the energy block. Speed-up is
computed within a session.

## 4.18 Energy Measurement Protocol

### 4.18.1 Backend

Energy is read from the **Intel RAPL package counter** [@khan2018rapl] through the **Windows Energy Meter Interface**:
- performance counter `\Energy Meter(RAPL_Package0_PKG)\Energy`, read as raw PDH values;
- picowatt-hours, converted with 1 pWh = 3.6 × 10⁻⁹ J;
- refresh interval of about 1 ms;
- read immediately before and after each block.

**Scope.**
- Package level. The core (PP0) domain is recorded as secondary.
- The DRAM domain reads zero on this machine and is not used.
- No TDP model, software estimate or surrogate replaces a measurement.
- No external meter was available.

Software energy counters need careful protocol design and stability checks [@jay2023powermeters;
@raffin2024dissecting]. This motivates the gates and qualification rules below.

### 4.18.2 Phase 7A

**Workload.** Per model: 1,000 warm-up inferences, then one block of N batch-1 inferences, giving
E = package J / N × 1,000.

**Sessions.** Five fresh-process sessions per architecture, each with the four models (dense included) in a seeded
random order. 30 s of idle power are measured at the start of each session. The gates of Section 4.17 apply, and the
power plan must be identical in every session.

**Pilot.** A dense-only pilot of 2 processes × 3 blocks. If the coefficient of variation of the six values was ≤ 3%
for both architectures, N = 100,000; otherwise N was doubled once, and then the phase stopped. The rule was met at
N = 100,000.

### 4.18.3 Phase 7B energy surrogate

The surrogate made the search affordable. Its target was **directly measured RAPL energy**.

**Calibration set.** 200 structured policies, fixed with seed 20261028:
- dense and the H0 policies (7);
- 160 policies stratified over eight MAC-reduction bins;
- similar-MAC pairs (MACs within 2%, widths far apart), to separate MACs from shape.

**Measurement.**
- Three fresh-process sessions on CPU 2.
- Per policy: 200 warm-up inferences and N = 20,000.
- A dense anchor every 25 policies; sessions drifting by more than 10% are repeated.
- The target is the three-session median.
- A dense pilot with a 4% limit came first and met the limit at N = 20,000.

**Candidate predictors,** tried in order; the first to pass is accepted:
1. MACs;
2. table latency T_table;
3. MACs, parameters and T_table;
4. a degree-2 ridge model in the normalised widths plus MACs, parameters and T_table.

**Acceptance criteria,** using 5-fold cross-validation (seed 20261029) on out-of-fold predictions:
- Spearman correlation ≥ 0.95;
- median absolute percentage error ≤ 5%;
- the same error ≤ 5% within the lowest-energy quartile.

Candidate 1 failed. Candidate 2 passed and was accepted, giving Equation (3.18). Its statistics appear in Chapter 6.
**Predicted energy is never reported as measured energy.**

### 4.18.4 Phase 7B confirmatory measurement and its stopping rule

**Planned sessions.** Five fresh-process sessions, each with 30 s idle, 1,000 warm-up inferences and N = 100,000 per
block, and 5 s pauses. Dense runs three times per session, and every other model once, in a seeded order.

**Pilot rule.** As in Phase 7A, with a 3% limit; N is doubled once, then STOP.

**Course of the pilot:**
1. **N = 100,000:** coefficient of variation 11.36%.
2. **First doubled-N attempt:** halted by the load gate during foreground activity. By recorded user decision it was
   declared void, as an earlier AC-gate refusal had been.
3. **Permitted rerun at N = 200,000, laptop idle:** 4.01%, above the 3% limit.

Under the frozen rule, hardware validation **stopped**. No confirmatory session was run, so the latency and energy
conditions of the joint criterion could not be assessed by measurement. **No Phase-7B energy reduction is reported.**

### 4.18.5 Energy confirmation study

**Protocol.** A new protocol, committed before any qualification data. Its workload:
- a fixed 180 s thermal soak before the first block;
- 2,000 warm-up inferences and N = 200,000 per block, with 5 s pauses;
- a fresh child process per run;
- all models loaded and hash-checked before the soak.

**Recorded per block:** package and core energy, latency, system load, CPU-2 processor performance, thermal-zone
temperature and AC status.

**Validity.** A run is void, kept and repeated if load exceeds 15% during any block or the laptop leaves AC. Runs are
never voided on energy or latency values. More than three voids in a stage is a stop.

**Qualification.** On the dense model only: 3 fresh runs × 2 blocks. The setup is qualified if the coefficient of
variation of the six values is ≤ 3%. There is one attempt, and the limit is not relaxed.

**Outcome.** The coefficient of variation was 19.92%, so the setup was **not qualified**. The study stopped **before any
paired Dense-versus-PPO measurement**, and energy confirmation is NO. The stop reflects the instability of the
measurement setup; no comparison of the two models was made.

## 4.19 Reproducibility and Pre-Registration

**Pre-registration and freezing.** Each protocol was committed before its confirmatory data (Table 4.9) and never
edited. A necessary change was a separately committed amendment made before the affected runs. Later phases verify
that frozen phases show no difference from their commits.

**Conduct rules:**
- **No interim peeking.** Progress reports showed counts, runtimes and failures only.
- **No result-based exclusion.** Infrastructure failures were repeated with the same seed and logged.
- **Durable records.** Runs were resumable and written atomically, and resumption was verified to be bit-identical.
- **Checks.** Implementation checks used only seed 42 and synthetic inputs, before each phase. Integrity checks ran
  before any statistic. They covered:
  - complete run matrices;
  - unchanged hashes;
  - no test access in training;
  - the freeze order of archives, final lists and test evaluations;
  - exact matching;
  - thread counts and environments.

**Determinism and artefacts.**
- **Seeds.** Listed in Table 4.6 and Section 4.9; dense training used seeds 0–2.
- **Hashes.** Split index files, dense references, frozen final-model lists, fine-tuned and recovered checkpoints, and
  latency tables are stored with SHA-256 hashes, including state-dictionary hashes for models.
- **Reproduction checks.** The six Phase-6 reproduction runs test bit-identical reproduction. Paper 3 verified that
  recovery does not depend on the core that runs it.
- **Stored artefacts.** Per-run records and aggregates are saved as CSV and JSON, with per-image test predictions for
  paired analyses.
- **Provenance.** Each phase has its own branch with frozen commits; the governing commits are listed in Table 4.9.

**Test-set governance.** Code enforces Table 4.2: a guard raises on any test access during training, or before a
phase's final-model list is frozen.
- **Phases 3–5:** the test set was read once per distinct frozen model and matched baseline.
- **Phase 6:** read only for the 480 frozen checkpoints and the pre-fine-tuning masks. Each pre-fine-tuning accuracy had
  to equal the Phase-5 record.
- **Paper 3, 7A and 7B:** the test set was never read.
- **Design choices:** none used test accuracy.

**Two disclosures:**
1. **Phase 6 is not a fully independent confirmation.** The test sets had been read in Phases 2–5 for earlier frozen
   models.
2. **Validation-to-test transfer is measured, not assumed.** Phase 5 records each selected policy's feasibility
   separately on VAL-RL, VAL-SELECT and test, with q_test = test / dense test. Phase 6 tests whether fine-tuning
   changes the gap between VAL-SELECT and test retention.

## 4.20 Chapter Summary

This chapter fixed the evaluation design:
- **Data.** A 45,000 / 5,000 / 10,000 partition of CIFAR-10 and CIFAR-100. Validation is split into VAL-RL and
  VAL-SELECT in two declared configurations: 3,000 / 2,000 in Phases 3–4 and 2,500 / 2,500 from Phase 5. The test set
  is read only for frozen models.
- **Models.** Three hashed dense references per dataset.
- **Pruning units.** Four per network: masked in the core framework, and physically compacted in the extension, with
  a coupled residual pruning group for ResNet-8.
- **Baselines.** Matched in zero count in the core framework, and approximately in MACs or parameters in the
  extension.
- **PPO.** One configuration throughout.
- **Recovery.** One 5-epoch protocol, identical across methods.
- **Statistics.** Pre-registered paired tests and decision rules, including a ±0.5 pp confidence-interval rule that is
  not a formal equivalence test.
- **Measurement.** Latency and RAPL energy protocols whose stability gates stopped the Phase-7B measurements before
  any paired comparison.

Chapters 5 and 6 report the results.
