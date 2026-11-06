# Chapter 4 — Experimental Setup

## 4.1 Experimental Design Overview

This chapter specifies how the framework of Chapter 3 was evaluated, in enough detail for every experiment to be
reproduced and audited; it reports no results. Every phase committed its conditions, seeds, metrics, tests and decision
criteria before any confirmatory run, treating earlier phases as frozen evidence verified by hash. Phase 1 audited an
earlier pipeline whose test set had been used for reward, sensitivity and epoch selection, and labelled it superseded;
all reported experiments follow the leakage-guarded protocol adopted after it (Section 4.3). Phase 2 fixed the splits,
trained the dense references and verified the shared library. The **core framework** (Phases 3–6) uses mask-based
pruning on six architecture–dataset settings on a desktop CPU; the **deployment extension** (Paper 3, Phases 7A and
7B, an energy confirmation study) uses physically compacted networks measured on a laptop CPU.

## 4.2 Datasets

The experiments use CIFAR-10 and CIFAR-100 [@krizhevsky2009cifar] (Table 4.1), scaled to [0, 1] and normalised per
channel with constants from the official 50,000-image training file: means (R, G, B) 0.4914, 0.4822, 0.4465 for
CIFAR-10 and 0.5071, 0.4865, 0.4409 for CIFAR-100, and standard deviations 0.2470, 0.2435, 0.2616 and
0.2673, 0.2564, 0.2762. All training uses random 32×32 crops with 4-pixel zero padding and random horizontal flips; evaluation is
never augmented. CIFAR-100 is used only in Phases 2–6 and the Paper 3 latency tables.

## 4.3 Dataset Partitioning and Validation Protocol

### 4.3.1 Training, validation and test

For both datasets, a fixed permutation of the 50,000 training indices (`torch.randperm`, seed 42) assigns 45,000
images to **training** and the last 5,000 to **validation**; the official 10,000 images are the **test** set. Both
datasets use identical index lists, recorded with hashes in split manifests.

### 4.3.2 Two validation configurations

Validation is split into two class-stratified, disjoint parts: **VAL-RL** (reward split) drives sensitivity, rewards
and PPO training, and **VAL-SELECT** (selection split) only chooses among already-produced candidates. The sizes
changed once (Table 4.1), as declared before any Phase-5 data existed: **configuration I** (Phases 3–4, 3,000 / 2,000;
about 60% of each class to VAL-RL, largest-remainder rounding, per-class permutation with seed 20260928; recorded as
V_RL and V_SELECT) and **configuration II** (Phase 5 onward, 2,500 / 2,500; seed 20260930; per class the halves differ
by at most one image).

The change removed a Phase-4 bias: the final Phase-4 search variant had been chosen by the VAL-SELECT rank of policies
that the archive had itself selected on VAL-SELECT. Phase 5 therefore used a new balanced split and conditions fixed in
advance, and drew no inference about conditions from VAL-SELECT metrics of VAL-SELECT-selected policies. Of the 3,000
configuration-I VAL-RL images, 1,521 (CIFAR-10) and 1,515 (CIFAR-100) fall into the new VAL-SELECT, so sensitivity was
recomputed. VAL-RL and VAL-SELECT name *roles*; the configuration is stated with every accuracy, and accuracies from
different configurations or splits are never compared.

**Table 4.1. Datasets and split configurations.**

| Item | CIFAR-10 | CIFAR-100 |
|---|---|---|
| Image size / classes | 32×32 RGB / 10 | 32×32 RGB / 100 (fine) |
| Training / full validation | 45,000 / 5,000 | 45,000 / 5,000 |
| Configuration I (Phases 3–4): VAL-RL / VAL-SELECT | 3,000 / 2,000 (seed 20260928) | 3,000 / 2,000 (seed 20260928) |
| Configuration II (Phase 5 on): VAL-RL / VAL-SELECT | 2,500 / 2,500 (seed 20260930) | 2,500 / 2,500 (seed 20260930) |
| Test | official 10,000 | official 10,000 |

### 4.3.3 Use of each split by phase

In every phase (Table 4.2) the reward split never selects the returned policy, and the test set never informs a
search, selection or design choice.

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

The Phase-7A policy was trained on VAL-RL and selected on VAL-SELECT in Phase 5, so its validation accuracy may be
optimistically biased relative to the baselines (recorded in the Phase-7A pre-registration). Without a frozen test
protocol for the structured models, Phases 7A and 7B report validation accuracy only.

## 4.4 Neural Network Architectures

The networks are defined in `src/models`:
- **SimpleCNN** (the historical network): three 3×3 convolutions (padding 1, with bias) with 32, 64 and 128 filters,
  each with ReLU and 2×2 max-pooling; flattening to 2,048 features; a 256-unit hidden layer with ReLU; an output layer
  of C units. No batch normalisation or dropout; module names match the historical checkpoint.
- **LeNet-5, adapted to CIFAR** [@lecun1998lenet]: two 5×5 valid convolutions with 6 and 16 filters, each with ReLU and
  2×2 max-pooling (replacing the original sigmoidal units and trainable subsampling); fully connected layers
  400 → 120 → 84 with ReLU; an output layer of C units.
- **ResNet-8** [@he2016resnet] (n = 1, 6n + 2 = 8 weight layers): a 16-channel 3×3 stem convolution with batch
  normalisation; three single-BasicBlock stages with 16, 32 and 64 channels and strides 1, 2 and 2, each with two 3×3
  convolutions and batch normalisation; identity shortcuts in stage 1 and 1×1 projections with batch normalisation in
  stages 2 and 3; global average pooling and an output layer of C units. Convolutions have no bias and Kaiming-normal
  (fan-out) initialisation. A pre-registered fallback for an over-budget ResNet-8 was not needed.

**Table 4.3. Network architectures and pruning units (core framework).** Counts are CIFAR-10 / CIFAR-100. "Weights"
are the Conv2d and Linear weights, the denominator of sparsity (Equation 3.2).

| Architecture | Datasets | Pruning units (share of prunable weights) | Dense parameters | Weights (prunable) | Main role |
|---|---|---|---|---|---|
| SimpleCNN | C10, C100 | conv1 0.14%, conv2 2.99%, conv3 11.94%, fc1 84.93% | 620,362 / 643,492 | 619,872 / 642,912 (617,312) | all phases; the only network of Phase 7B |
| LeNet-5 | C10, C100 | conv1 0.74%, conv2 3.94%, fc1 78.78%, fc2 16.54% | 62,006 / 69,656 | 61,770 / 69,330 (60,930) | core phases 3–6 only |
| ResNet-8 | C10, C100 | stem 0.56%, stage 1 6.01%, stage 2 18.69%, stage 3 74.74% | 78,042 / 83,892 | 77,360 / 83,120 (76,720) | core phases; Paper 3; Phase 7A |

The output layer, biases and batch-normalisation parameters are never pruned.

## 4.5 Dense Reference Models

Each setting has one dense **reference model**, the starting point of every pruning method, trained in Phase 2 under
a protocol committed before any training run: Adam, learning rate 10⁻³, β = (0.9, 0.999), ε = 10⁻⁸, no weight decay;
cosine annealing to zero over 30 epochs, stepped per epoch; batch 128, cross-entropy loss and the augmentation of
Section 4.2; seeds 0, 1 and 2, where the seed sets the initial weights and a generator with seed + 10⁶ sets data order
and augmentation. The epoch with the highest full-validation accuracy is kept and hashed, and only then is the test set
read, once.

The seed-0 model, fixed by rule, is the reference (seeds 1 and 2 only measure variability), except the disclosed
CIFAR-10 SimpleCNN historical checkpoint (Adam at 10⁻³ for 10 epochs, final epoch). Epoch selection used the full
validation set, later split into VAL-RL and VAL-SELECT, which affects every method of a setting identically. Later
phases assert the reference hash (Table 4.4), and Phases 6, 7A and 7B add a dense model fine-tuned with the pruned
models' protocol, separating recovery from extra training.

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

The extension starts from the CIFAR-10 SimpleCNN and ResNet-8 references.

## 4.6 Pruning Units and Policy Spaces

### 4.6.1 Masked pruning units (core framework)

The four units per architecture are those of Tables 3.3 and 4.3, with ResNet-8's three **residual-stage weight
groups** (stage 1: the two block convolutions; stages 2 and 3: plus the 1×1 projection). An action applies its ratio
as **L1 unstructured** pruning to *each* tensor of the unit (Section 3.3); a ResNet-8 stage action removes no stage,
block or channel. For the exploration endpoints, a policy is **destructive** if it prunes the setting's most sensitive
unit (highest raw sensitivity, Equation 3.8; ties to the earliest unit) by 40% or more, a rule fixed before any PPO
run.

### 4.6.2 Physically compacted structured units (deployment extension)

A ratio removes whole channels or neurons: a unit keeps the k = max(1, round((1 − r)·C)) channels with the largest L1
norm on the dense model (Equation 3.17), and the network is rebuilt with smaller layers.

In **SimpleCNN** (C = 32, 64, 128, 256) conv1–conv3 remove output filters and the next layer loses the matching
inputs (16 flattened columns per channel of the first fully connected layer after conv3); fc1 removes hidden neurons
and the output layer the matching inputs. In **ResNet-8** (C = 16, 16, 32, 64) the **stem is a coupled residual pruning
group** setting the stage-1 residual width: one shared channel set, ranked by the summed L1 norms of the two writing
filters, cuts the writers (the stem convolution and the second stage-1 convolution, each with its batch
normalisation), the stage-1 identity shortcut, and the readers (the inputs of the first stage-1 convolution, the first
stage-2 convolution and the stage-2 projection). **Stages 1–3 are block-internal channel groups** (the outputs of the
block's first convolution and its batch normalisation, and the matching inputs of its second convolution); the stage-2
and stage-3 output widths and the output layer are not pruned.

**The masked and structured ResNet-8 units are different objects** (the masked stage-2 unit covers three
convolutions, the structured one only block-internal channels), so a masked ResNet-8 policy maps onto structured units
only approximately, a recorded Phase-7A limitation; for SimpleCNN the mapping is direct. Every compacted model is
checked: no masks remain, parameter and MAC counts equal the analytic values, logits agree with the masked-equivalent
model within 10⁻⁴, and outputs are identical after reloading.

### 4.6.3 Policy spaces and enumeration

Every setting has 6⁴ = 1,296 policies. A PPO run samples 512 episodes, so it evaluates at most 512 policies (about 40%
of the space). **Exhaustive diagnostic landscapes** of masked policies without fine-tuning exist on configuration-I
VAL-RL (Phase 3), configuration-I VAL-SELECT (Phase 4) and both configuration-II splits (Phase 5); they are an exact
cache for PPO's terminal evaluations, define each policy's rank, and are never computed on the test set. **Structured
enumerations:** Paper 3 timed all 1,296 structured shapes for SimpleCNN and ResNet-8, each with 10 and 100 output
classes (four tables, 5,184 shapes), and Phase 7B recovered all 1,296 structured SimpleCNN policies with a 2-epoch
protocol as a proxy cache. A planned exhaustive 5-epoch Paper 3 landscape (5,184 policies) was stopped for compute
after 126 CIFAR-10 SimpleCNN policies, used only as pilot data, so fully recovered accuracy is known only for sampled
or selected policies; energy was measured only for the models named in Section 4.18.

## 4.7 Baseline Pruning Methods

### 4.7.1 Core-framework baselines

The baselines are one-shot methods from `src/pruning`, each calibrated to an exact number K of zeroed weights from the
prunable tensors only (the output layer stays dense), with largest-remainder rounding of per-tensor counts. All are
deterministic except Random:
- **Uniform:** every prunable tensor loses the fraction K/N of its smallest-magnitude weights (N prunable weights).
- **Random:** uniform allocation with weights removed at random inside each tensor; ten mask seeds (1000–1009),
  summarised by their mean accuracy.
- **Global magnitude:** the K smallest-magnitude weights over all prunable tensors, a standard reference point in
  pruning evaluations [@blalock2020state].
- **ERK** [@evci2020rigl]: tensor t receives density d_t = min(1, ε·(sum of its dimensions)/n_t), with ε solved so that
  N − K weights are kept (saturated tensors fixed dense, ε re-solved), removing the smallest-magnitude weights within
  each tensor.
- **LAMP** [@lee2021lamp]: each weight receives the score w_(u)² / Σ_{v≥u} w_(v)² within its magnitude-sorted tensor,
  and the K smallest scores network-wide are removed; every tensor keeps its largest weight.

LAMP and ERK are layer-adaptive allocation rules, uniform and random fixed allocations, and global magnitude a single
network-wide threshold; each is referred to by name, not as a "heuristic".

### 4.7.2 Deployment-extension comparators

**Uniform-Structured** uses one common channel ratio for all four units and **LAMP-Structured** a LAMP allocation
translated into per-unit structured ratios. In Phase 7B, **old PPO** is the frozen Phase-7A transfer of the Phase-5
policy and **new PPO** the policy learned by the deployment-oriented structured PPO.

### 4.7.3 Matching rules

In the core framework, "matched" always means an **identical zero count**, i.e. identical sparsity ρ. In the
extension, MACs or parameter counts are matched approximately, because channel counts are integers (Table 4.5).

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

Every agent uses Stable-Baselines3 2.8.0 [@raffin2021sb3], running PPO [@schulman2017ppo] with generalised advantage
estimation [@schulman2016gae] and the hyper-parameters of Table 3.7; Table 4.6 shows what differs by phase (Phases 3–6
spread the single-thread runs over four worker processes). The evaluation data loader is still created at each
cached terminal step, so the random stream matches an uncached run.

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

The Phase 3–5 seeds were verified unused by any earlier PPO run, except an unrecorded 64-timestep smoke test with seed
300 in the Phase-2 checks. Seeds are shared across conditions and settings within a phase, so comparisons are paired;
no seed was removed, replaced or added. Elite behavioural cloning (Phase 4, S2) adds η·𝓛_elite (Equation 3.13) to a
copy of the SB3 PPO update, bit-identical to standard PPO at η = 0.

## 4.9 Experimental Conditions by Phase

Table 4.7 summarises each phase.

**Table 4.7. Phase-by-phase experimental design.**

| Phase | Research purpose | Datasets | Architectures | Conditions | Seeds | Runs | Primary output |
|---|---|---|---|---|---|---|---|
| 3 | PPO vs baselines; sensitivity prior | C10, C100 | all three | P0, P1, P2, P3 | 300–319 | 480 | test accuracy vs matched baselines; exploration endpoints |
| 4 | reward, centred prior, retention | C10, C100 | all three | Stage A: 4 rewards; B: 6 priors; C: 4 search variants | 400–419 | 1,200 | stage selections; search quality; final pipeline vs control and LAMP |
| 5 | accuracy-constrained reward | C10, C100 | all three | C0, C1, C2 | 500–519 | 360 | constraint, search and accuracy endpoints |
| 6 | equal fixed-mask recovery | C10, C100 | all three | PPO, LAMP, global, dense control | fine-tuning 600–619 | 480 (+6 repeats) | accuracy before and after equal fine-tuning |
| Paper 3 | structured CPU latency | C10, C100 (latency); C10 (accuracy) | SimpleCNN, ResNet-8 | 1,296 shapes per table | recovery seed 600 | 5,184 shapes | latency tables; GO/NO-GO for hardware-aware PPO |
| 7A | structured transfer, measured energy | C10 | SimpleCNN, ResNet-8 | Dense, PPO-, Uniform-, LAMP-Structured | recovery seed 600 | 8 models; 5 sessions per architecture | accuracy, memory, MACs, latency, energy |
| 7B | deployment-oriented structured PPO | C10 | SimpleCNN | new PPO; Dense, Uniform-, LAMP-Structured, old PPO | 900–904; 700–719 | 20 searches; 11 recovered models | accuracy, memory, MACs; planned latency and energy |
| 7B energy confirmation | direct energy of frozen 7B models | C10 | SimpleCNN | Dense vs new PPO | — | 6 qualification blocks; 10 sessions planned | qualification; planned paired energy |

### 4.9.1 Phase 3

Phase 3 ran the four conditions of Section 3.9 (P0, no prior; P1, c_l = 0.5·S_l; P2, S permuted per seed so that every
unit receives another unit's value; P3, the mean of S for every unit) across six settings with 20 seeds: 480 runs,
returning the terminal deterministic policy, pruned without fine-tuning, with baselines built after all PPO runs. Four
separate questions were posed: (A) PPO vs uniform; (B) PPO vs global magnitude, LAMP, ERK and random; (C) whether the
correct prior improves exploration (primary endpoint: VAL-RL accuracy AUC); and (D) whether it improves the final model
(test accuracy in excess of matched global magnitude, P1 − P0). "P2" is the *shuffled* prior in Phase 3 and the
*centred* prior in Phases 4–5; the phase is always stated.

### 4.9.2 Phase 4

Phase 4 used a staged design with mechanical, validation-only selections, on seeds 400–419:
- **Stage A, reward.** R0, R1a, R1b and R2 (Section 4.12), with P0 and the terminal policy: 480 runs. A redesigned
  reward qualified only if it at least halved the cross-setting spread of final sparsity (with a seed-bootstrap upper
  bound below 1) **and** kept VAL-RL retention ≥ 0.95 in at least five settings; otherwise R0 was kept.
- **Stage B, prior.** 600 new runs under the Stage-A reward with six priors: P0, none; P1, c_l = 0.5·S_l; P2,
  c_l = 1.0·(S_l − S̄) (centred); P3, P2 permuted across units; P4, c = 1.0·mean_l |S_l − S̄| for every unit
  (magnitude-matched pure conservatism); P5, c_l = −1.0·(S_l − S̄) (sign-reversed). P1 and P2 were each tested against
  P0 on reward-rank AUC; a prior supported in at least three settings with none opposite was eligible, the eligible
  prior supported in more settings was carried forward (tie: larger mean d_z), and otherwise P0 was.
- **Stage C, search.** 120 new runs: **S0**, terminal policy; **S1**, archive rule on the S0 runs; **S2** and **S2A**,
  elite behavioural cloning with the terminal policy or the archive rule. The winner had the lowest mean, over
  settings, of the median VAL-SELECT rank of the selected policy.

The selections kept R0 (no redesigned reward qualified), chose prior P2 and chose search variant S2A, defining the final
pipeline **F = (R0, P2, S2A)** and the control **K = (R0, P0, S0)**, the Phase-3 configuration on fresh seeds.

### 4.9.3 Phase 5

Phase 5 used configuration II and three conditions fixed in advance: **C0**, R0 + P2 + archive, selecting by the
highest R0 score on VAL-SELECT (the Phase-4 pipeline without behavioural cloning); **C1 (proposed)**, R_c with
τ = 0.98, P2 and constraint-aware selection (Equation 3.14); and **C2**, as C1 without a prior. With seeds 500–519 the
design has 360 runs; C1 − C2 is the sensitivity ablation and C1 − C0 the reward comparison. τ was chosen before any
run, on the VAL-RL landscapes only, as the largest value in {0.95, 0.97, 0.98, 0.99} giving every setting at least 20
feasible policies, including one with ≥ 10% sparsity: τ = 0.98.

### 4.9.4 Phase 6

Phase 6 tests whether accuracy differences persist under the same fine-tuning budget at exactly the same sparsity. The
methods are **PPO** (the frozen Phase-5 C1 policy of each setting and policy seed), **LAMP** and **global magnitude**
at PPO's exact zero count, and a **dense control**. **Phase 6 contains no uniform, ERK or random method, and no Phase-5
C0 or C2 policy**, which the pre-registration excluded to fit the fixed compute budget. There are 360 pruned runs plus
120 dense controls; the four runs of each (setting, policy seed) share the fine-tuning seed (policy seed + 100),
minibatch order and augmentation, checked by a data-stream hash, and six reproduction runs repeat seed 500's PPO run
in each setting (Section 4.13). The unit of analysis is the policy seed (n = 20 per setting); because seeds often
selected the same policy, a cluster-level sensitivity analysis uses one value per distinct policy where there are at
least six.

### 4.9.5 Paper 3

Paper 3 tested whether structured compression becomes CPU latency, through pre-registered decisions: **H0**, a
calibration selecting the primary timing configuration (Section 4.17); **H1**, 50 checks of structured compaction,
MAC and FLOP counters and structured sensitivity, with the Phase-2 library checks rerun on the laptop; and **H2-A**,
the four latency tables (5,184 shapes). For CIFAR-10 recovered accuracy, a 1-epoch recovery proxy was validated
against 5-epoch recovery on frozen samples (156 SimpleCNN and 31 ResNet-8 policy pairs), and an architecture could
proceed only if the Spearman correlation was ≥ 0.90. The protocol's Youden-J [@youden1950index] threshold rule needed
at least three truly feasible validation pairs; ResNet-8 had fewer, triggering the pre-registered fallback to a
decision by the principal investigator, and a pre-committed calibration on 48 frozen ResNet-8 policies set the
threshold by a recall-first rule with a 0.005 margin. A latency-room rule then required at least 20 proxy-feasible
policies and one with a speed-up ≥ 1.10×.

**Stopping outcomes** (values in Chapter 6). SimpleCNN did not meet the proxy-validity criterion; ResNet-8 met it and
passed calibration, but not the latency-room rule. Hardware-aware PPO was therefore stopped before any run, and **no
hardware-aware reinforcement learning was run**; Paper 3 contributes the latency tables and their cost-metric analysis.

### 4.9.6 Phase 7A

Phase 7A uses CIFAR-10 SimpleCNN and ResNet-8 with four methods each: **Dense**; **PPO-Structured**, the most frequent
Phase-5 C1 selected policy over seeds 500–519 (ties: smaller action vector), read as structured ratios;
**Uniform-Structured**, MAC-matched to PPO; and **LAMP-Structured**, from the Phase-6 LAMP mask. All receive the equal
5-epoch recovery with seed 600 and five energy sessions (Section 4.18). The fixed criteria are **accuracy**,
|A_method − A_dense,recovered| ≤ 0.5 pp on VAL-RL, and **consistent energy reduction**, lower energy than the
same-session dense model in 5 of 5 sessions **and** a median saving above a noise threshold of twice the dense model's
between-session coefficient of variation. **Case A** (both hold) is the primary success; **B** is consistent reduction
with more than 0.5 pp loss; **C** accuracy within 0.5 pp without consistent reduction; **D** a baseline with the better
trade-off, i.e. accuracy within 0.5 pp of the other method and lower energy in 5 of 5 sessions, or accuracy higher by
more than 0.5 pp with median energy not higher.

### 4.9.7 Phase 7B

Phase 7B uses SimpleCNN on CIFAR-10 only. Its agent, the deployment-oriented structured PPO (Section 3.16.3), was
rewarded during the search with the *surrogate (predicted)* energy saving of Equation (3.18); no measured energy
entered the search, and the name does not imply a measured energy result.

**Stage A (development)**, before the final pre-registration, validated the proxy and surrogate (Section 4.18.3),
built the 2-epoch proxy landscape and tuned α ∈ {0.5, 1, 2, 4} on seeds 900–904 by the lowest median predicted energy
of the selected policies (ties: smaller α), giving α = 4; β = 1.0, κ = 10, ε = 0.01, the 1% tie band and the selection
rule were fixed, not tuned. **Stage B (confirmatory)** ran 20 seeds (700–719), all reported, with structured
sensitivity on VAL-RL (probes at {0.1, 0.2, 0.4, 0.6}) and τ̂ = 0.98849, whose target was 5-epoch VAL-RL accuracy
within 0.5 pp of the recovered dense model. All final models (dense, the primary new-PPO policy, the other distinct
per-seed selections, Uniform- and LAMP-Structured) receive the equal 5-epoch recovery; old PPO is reused read only.

**Joint deployment-efficiency criterion** (named *GREEN-SUCCESS* in the pre-registration). A model succeeds only if
all five conditions hold at once: (1) VAL-SELECT accuracy at least the recovered dense VAL-SELECT accuracy minus
0.5 pp; (2) parameter-memory reduction > 0; (3) MAC reduction > 0; (4) latency reduced, with the median below dense and
the model faster in 5 of 5 sessions; and (5) measured energy reduced beyond noise, lower in 5 of 5 sessions with a
median saving above twice the dense model's between-session coefficient of variation. There is no override, an empty
archive counts as failure, and the success frequency over the 20 seeds is also defined.

### 4.9.8 Phase 7B energy confirmation

This protocol, written after Phase 7B had closed, does not alter Phase 7B or its verdicts. After hash verification and
qualification of the instrument on the dense model (Section 4.18.5), ten paired Dense and new-PPO sessions would run,
five in each order (seeded permutation). Confirmation requires that the setup qualified, the PPO model used less energy
in at least 8 of the 10 sessions, the median saving exceeded the noise threshold, and a leave-one-session-out check
held for every omitted session.

## 4.10 Sensitivity Conditions and Controls

Sensitivity enters the core framework only through the prior (state slot 0; Section 3.5). **Layer-specific
information** is tested by priors that scramble the unit assignment of the same pressure (shuffled P2 in Phase 3;
permuted P3 in Phase 4); **generic conservatism** by pressure without sensitivity information (constant P3 in Phase
3; magnitude-matched P4 in Phase 4); and **direction** by the sign-reversed P5 (Phase 4). The centred P2 of Phase 4 removes net conservatism by construction, and the
sensitivity-specific statement held in a setting only if P2 beat **both** P3 and P4 there. Phase 5 compares C1 with C2;
Phase 7B ran no prior-free control. The vectors are computed on configuration-I VAL-RL (Phases 3–4), configuration-II
VAL-RL (Phase 5) and, with structured probes, configuration-II VAL-RL (Phase 7B); none uses VAL-SELECT or test data.

## 4.11 Archive and Selection Protocols

**Phase 3** returns the terminal deterministic policy without an archive. In **Phase 4**, S0 and S2 return the
terminal policy and S1 and S2A apply Equation (3.12) to the archive (Section 3.12).

**Phase 5** follows a fixed order:
1. The archive of all 360 runs was frozen and hashed before any VAL-SELECT evaluation.
2. The candidates were evaluated on VAL-SELECT, and each result had to equal the VAL-SELECT landscape.
3. C1 and C2 selected by Equation (3.14), with its tie-break and reported fallback; C0 by the highest R0 score on
   VAL-SELECT.
4. The final-policy list was committed before the test set was read.

In **Phase 7B** the archive holds every distinct proxy-feasible policy visited. Per seed, the lowest predicted energy is
selected; policies within 1% count as tied, and ties go to higher proxy accuracy, then fewer MACs, then lower parameter
memory, then the smaller action vector. The primary new-PPO model is the most frequent selection over the 20 seeds. The
same rule applied to every proxy-feasible landscape policy gives an enumeration reference, reported but not a method.
VAL-SELECT plays no part in Phase-7B selection.

## 4.12 Reward Configurations

All rewards are computed on VAL-RL from the pruned network without fine-tuning; Phase 7B uses its 2-epoch proxy.

**Table 4.8. Reward configurations.**

| Reward | Definition | Used in |
|---|---|---|
| R0 | 1.5·A_p/A_b + 0.01·ρ(%) + 0.05·D (Equation 3.7) | Phases 3 and 4; Phase 5 C0 |
| R1a | 1.5·A_p/A_b − κ_a·\|ρ − 50\| + 0.05·D | Phase 4 Stage A |
| R1b | 1.5·A_p/A_b − κ_b·\|ρ − ρ*_s\| + 0.05·D, with ρ*_s the largest sparsity retaining ≥ 95% of VAL-RL accuracy | Phase 4 Stage A |
| R2 | 1.5·(A_p − A_60)/(A_b − A_60) + 1.5·ρ/ρ_60 + 0.05·D, anchored on the maximal policy [5,5,5,5] | Phase 4 Stage A |
| R_c | constraint-first, τ = 0.98 (Equation 3.15) | Phase 5 C1, C2 |
| R_7B | constraint-first on the proxy, predicted energy saving, α = 4 (Equation 3.19) | Phase 7B |

κ_a and κ_b were each set before any run as the smallest value in {0.005, 0.01, 0.02, 0.04, 0.08} whose VAL-RL reward
optimum lies within ±5 pp of its target in all six settings. Only R0 and R_c enter the mature core framework.

## 4.13 Recovery and Fine-Tuning Protocol

The Phase-6 protocol (Section 3.15) is used unchanged in every later phase: Adam at 10⁻⁴, β = (0.9, 0.999), ε = 10⁻⁸,
weight decay 0, a new optimiser per run; cosine annealing to 0, stepped per epoch; batch 128 on the 45,000 training
images, with crop and flip; one thread; no checkpoint selection. **E = 5** was fixed before any confirmatory run by a
timing rule (E = 5 if the projected Phase-6 wall time was at most 30 hours, otherwise E = 3); the projection, from a
pilot that used no accuracy, was 29.44 hours, and no 10-epoch schedule was run.

In Phase 6 the mask is fixed by PyTorch's pruning reparameterisation [@paszke2019pytorch], with the mask hash, the
absence of non-zero masked weights and per-layer zero counts asserted after every epoch. Compacted models (Paper 3, 7A,
7B) must keep their tensor shapes through recovery and all use recovery seed 600 with data generator 600 + 10⁶, so all
models of a phase see the same minibatch sequence. The Paper-3 1-epoch proxy is exactly the first epoch of the 5-epoch
protocol; the Phase-7B 2-epoch proxy is a separate two-epoch cosine run on the same seed and data stream. Final accuracy
claims always use 5-epoch models.

## 4.14 Evaluation Metrics

Table 4.9 lists the metrics; accuracy always names its split, and every reduction its reference. Masked sparsity ρ,
the channel removal ratio r_l (Equation 3.17), physical parameter reduction over all parameters including biases and
batch normalisation, and MAC reduction are kept apart; per-unit sparsity is used only to translate LAMP allocations
into structured ratios.

**Table 4.9. Evaluation metrics.** ↑ higher is better; ↓ lower is better.

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

Comparisons are paired by seed (n = 20 per setting) with the **exact two-sided sign-flip test** over all 2²⁰ sign
patterns as primary test, reporting the mean, standard deviation and median, the 95% t-interval (df 19), a bootstrap
95% interval (10,000 resamples in Phase 3), d_z, the exact Wilcoxon signed-rank p [@wilcoxon1945individual] (Phases 3
and 4), and wins, ties and losses. Holm's step-down correction [@holm1979sequential] is applied within each
pre-registered family. A comparison is **supported** with the hypothesised direction, a Holm-adjusted p < 0.05 **and**
a 95% t-interval excluding 0; **opposite** under the same criteria in reverse; otherwise **inconclusive**. Across
settings (Phases 3–4) a statement is **STRONG** if supported in ≥ 5 of 6 settings and opposite in none, **MODERATE** if
supported in 3–4 and opposite in none, **LIMITED** if supported in 1–2 (or ≥ 3 with one opposite), and **NOT
SUPPORTED** otherwise; opposite in ≥ 3 settings is also labelled REVERSED. No subset result is described as general.

**Families.** Phase 3: A (6), B (24), C (6), C-specificity (12), one per C-secondary endpoint, and D (6), with McNemar
counts supportive only. Phase 4: reward, prior, search and final-model families, a seed bootstrap for Stage A, and a
**one-sided non-inferiority test** of F against LAMP with a 0.5 pp margin (a Holm-corrected sign-flip test on d + 0.5,
plus a lower t-bound above −0.5). Phase 5: C1 − C0 on test accuracy (6); C1 − C2 on four primary exploration endpoints
(24: reward-rank AUC, VAL-RL accuracy AUC, first episodes in the top 20 and top 10) and on secondary endpoints (18);
C1 − LAMP and its gap change (6 each); C1 − global magnitude **descriptive**. Phase 6: recovery, dense-control and
pruning-specific recovery families; PPO vs LAMP and PPO vs global (12 each); the retention gap; and, because
architecture dependence compares different experiments, an unpaired permutation test (100,000 permutations) with a
Welch interval, plus a seed-paired sensitivity version.

### 4.15.2 The ±0.5 pp rule of Phase 6

Phase 6 reports statistical support and practical size separately, with a fixed 0.5 pp margin:
- a difference is *practically relevant* only if |mean| ≥ 0.5 pp;
- two methods lie **within the ±0.5 pp margin** if the 95% t-interval of their paired difference lies inside
  (−0.5, +0.5) pp;
- a supported difference below 0.5 pp is *statistically detectable but small*.

This is a **pre-registered confidence-interval inclusion rule**, not a formal statistical equivalence test. A formal
test, such as two one-sided tests with a justified margin and stated error rate, was not pre-registered. The
pre-registration calls the rule "practically equivalent"; the thesis writes "within the ±0.5 pp margin" so that it is
not read as a formal equivalence claim.

### 4.15.3 Deployment-extension statistics

**Paper 3** is descriptive and rank-based: Spearman correlations of latency gain with parameter and MAC reduction over
all 1,296 shapes, the fraction of policy pairs in which the policy with more parameter reduction is slower, and the H0
signal-to-noise rule; proxy validity uses Spearman correlation, error, bias, top-10 overlap and a confusion matrix.
**Phases 7A and 7B** summarise sessions by median, mean, standard deviation and interquartile range, and test each
method against dense with an **exact one-sided Wilcoxon signed-rank test** on five paired sessions, whose minimum
attainable p of 0.031 makes p-values secondary to the 5-of-5 rule and the noise threshold; two-sided tests between
methods are descriptive. The **energy confirmation** pre-registered an exact Wilcoxon and a sign test on ten pairs. No
other significance test was used.

### 4.15.4 Pre-registered decision rules

**Table 4.10. Pre-registered decision criteria.** The commit is the one that first added the governing file.

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

All experiments ran on CPUs with CPU-only PyTorch builds; no GPU was used (Table 4.11).

**Table 4.11. Hardware and software environments.**

| Item | Core framework (Phases 2–6) | Extension (Paper 3, 7A, 7B, energy confirmation) |
|---|---|---|
| CPU | Intel Core i5-6400 @ 2.70 GHz, 4 logical CPUs (desktop) | Intel Core i7-1260P (laptop): 4 performance cores × 2 threads (logical 0–7), 8 efficiency cores (8–15) |
| Memory | not recorded | 15.7 GB |
| OS | Windows 11, build 22621 | Windows 11 Home, build 26200 |
| Python; PyTorch | 3.11.9; 2.7.1+cpu (torchvision 0.22.1+cpu) | 3.11.9; 2.7.1+cpu with oneDNN 3.7.1 and oneMKL 2025.1 |
| Other libraries | Stable-Baselines3 2.8.0, sb3-contrib 2.8.0, Gymnasium 1.2.3, NumPy 2.2.2, pandas 2.2.3, ptflops 0.7.5, psutil 7.2.2 | Stable-Baselines3 2.8.0 in the pinned environment; the 7B records do not store package versions separately |
| Threads | 1 per run (dense training: 4) | 1 intra-op, 1 inter-op |
| Power | mains | AC required; Balanced plan |

Package versions are pinned in `requirements_locked.txt`, and every phase records its environment and commit, verified
by the integrity checks. **Results do not transfer between machines:** no desktop latency or energy value is used.

## 4.17 Structured Pruning and Latency Protocol

**Measurement core** (Paper 3, reused in Phases 7A and 7B): batch 1, float32, NCHW, `eval()` and `inference_mode()`
on one preallocated, seeded input tensor; one intra-op and one inter-op thread, with affinity and OpenMP/MKL thread
variables set before PyTorch is imported and the process pinned to logical CPU 2, a performance-core thread other than
core 0, avoiding hybrid-scheduling noise; each pass timed with `time.perf_counter_ns()` with garbage collection
disabled, never timing model construction, loading or data handling; AC power required, system load ≤ 15% over 5 s
before and after each session, and no other experiment running.

**H0 calibration.** Six SimpleCNN shapes were timed under configurations **A** (batch 1, 1 thread on CPU 2, 50 / 300
warm-up / timed passes), **B** (batch 1, 4 threads, one per performance core, 50 / 300) and **C** (batch 128, 4
threads, 10 / 50), in five rounds opening with a dense anchor block over five fresh-process sessions; the rule selected
configuration A.

**H2-A latency tables.** Configuration A in five fresh-process sessions per table, each timing all 1,296 shapes once in
a seeded random order, with a dense anchor block before the first shape and after every 25. A session whose anchors
drift by more than 10% from their median is repeated once; four attempts were repeated, and all 20 accepted sessions
met the criterion. A shape's latency is the median over sessions of its per-session median (the 95th percentile,
interquartile range, throughput, speed-up, latency gain against the identity shape and an anchor-normalised latency
are also stored), and the tables were hashed and committed before use. The absolute dense latency on the table days
was about 16% higher than on the H0 calibration day, a cross-occasion check kept as a recorded failure; within each
table the identity shape agrees closely with the dense anchors, so relative within-table latency is analysed, later
sessions compare against a same-session dense model, and the Phase-7B surrogate fits table latency to energies measured
on the same machine. **Phase 7A latency** is the median per-inference time over the energy block's N inferences, with
speed-up computed within a session.

## 4.18 Energy Measurement Protocol

### 4.18.1 Backend

Energy is read from the **Intel RAPL package counter** [@khan2018rapl] through the **Windows Energy Meter Interface**
(counter `\Energy Meter(RAPL_Package0_PKG)\Energy`): raw PDH values in picowatt-hours (1 pWh = 3.6 × 10⁻⁹ J),
refreshed about every 1 ms, read immediately before and after each block. The core (PP0) domain is secondary; the DRAM
domain reads zero on this machine and is not used. No TDP model, software estimate or surrogate replaces a measurement,
and no external meter was available. Software energy counters need careful protocol design and stability checks
[@jay2023powermeters; @raffin2024dissecting], hence the gates and qualification rules below.

### 4.18.2 Phase 7A

Per model, 1,000 warm-up inferences precede one block of N batch-1 inferences (E = package J / N × 1,000), in five
fresh-process sessions per architecture, each with the four models in a seeded random order after 30 s of idle-power
measurement, under the gates of Section 4.17 and an identical power plan. A dense-only pilot of 2 processes × 3 blocks
set N: N = 100,000 if the coefficient of variation of the six values was ≤ 3% for both architectures, otherwise N was
doubled once and then the phase stopped. The rule was met at N = 100,000.

### 4.18.3 Phase 7B energy surrogate

The surrogate's target was **directly measured RAPL energy** of 200 structured policies fixed with seed 20261028:
dense and the H0 policies (7), 160 policies stratified over eight MAC-reduction bins, and similar-MAC pairs (MACs
within 2%, widths far apart) separating MACs from shape. They were measured in three fresh-process sessions on CPU 2
(200 warm-up inferences, N = 20,000 per policy, a dense anchor every 25 policies, sessions drifting by more than 10%
repeated), taking the three-session median; a dense pilot with a 4% limit met the limit at N = 20,000. Candidate
predictors were tried in order, the first to pass being accepted: (1) MACs; (2) table latency T_table; (3) MACs,
parameters and T_table; (4) a degree-2 ridge model in the normalised widths plus MACs, parameters and T_table.
Acceptance required, on 5-fold cross-validated (seed 20261029) out-of-fold predictions, Spearman ≥ 0.95, a median
absolute percentage error ≤ 5%, and the same error ≤ 5% within the lowest-energy quartile. Candidate 1 failed;
candidate 2 was accepted, giving Equation (3.18) (statistics in Chapter 6). **Predicted energy is never reported as
measured energy.**

### 4.18.4 Phase 7B confirmatory measurement and its stopping rule

Five fresh-process sessions were planned (30 s idle, 1,000 warm-up inferences, N = 100,000 per block, 5 s pauses;
dense three times per session, every other model once, in a seeded order), preceded by the Phase-7A pilot rule with a
3% limit (N doubled once, then STOP). The pilot gave a coefficient of variation of 11.36% at N = 100,000; the first
doubled-N attempt was halted by the load gate during foreground activity and, by recorded user decision, declared void
(as an earlier AC-gate refusal had been); the permitted rerun at N = 200,000 with the laptop idle gave 4.01%, above the
3% limit. Under the frozen rule, hardware validation **stopped**: no confirmatory session was run, so the latency and
energy conditions of the joint criterion could not be assessed. **No Phase-7B energy reduction is reported.**

### 4.18.5 Energy confirmation study

A new protocol, committed before any qualification data, used a 180 s thermal soak, 2,000 warm-up inferences and
N = 200,000 per block with 5 s pauses, a fresh child process per run, and all models hash-checked before the soak,
recording package and core energy, latency, system load, CPU-2 processor performance, thermal-zone temperature and AC
status per block. A run is void, kept and repeated if load exceeds 15% during a block or the laptop leaves AC (never on
energy or latency values); more than three voids in a stage is a stop.

**Qualification** used the dense model only (3 fresh runs × 2 blocks; coefficient of variation of the six values ≤ 3%;
one attempt, no relaxation). The coefficient of variation was 19.92%, so the setup was **not qualified**, and the study
stopped **before any paired Dense-versus-PPO measurement**: energy confirmation is NO. The stop reflects the
instability of the measurement setup; no comparison of the two models was made.

## 4.19 Reproducibility and Pre-Registration

Each protocol was committed before its confirmatory data (Table 4.10) and never edited (necessary changes were separate
amendments committed before the affected runs). There was no interim peeking and no result-based exclusion;
infrastructure failures were repeated with the same seed and logged, and resumption was verified to be bit-identical.
Implementation checks used only seed 42 and synthetic inputs; integrity checks before any statistic covered complete
run matrices, unchanged hashes, no test access in training, the freeze order of archives, final lists and test
evaluations, exact matching, and thread counts and environments. Splits, references, final-model lists, checkpoints and
latency tables are stored with SHA-256 hashes; six Phase-6 reproduction runs test bit-identical reproduction, and
Paper 3 verified that recovery does not depend on the core that runs it.

**Test-set governance.** Code enforces Table 4.2: a guard raises on any test access during training, or before a
phase's final-model list is frozen. In Phases 3–5 the test set was read once per distinct frozen model and matched
baseline; in Phase 6 only for the 480 frozen checkpoints and the pre-fine-tuning masks, each pre-fine-tuning accuracy
having to equal the Phase-5 record; in Paper 3, 7A and 7B never. No design choice used test accuracy.

**Two disclosures.** (1) **Phase 6 is not a fully independent confirmation**, because the test sets had been read in
Phases 2–5 for earlier frozen models. (2) **Validation-to-test transfer is measured, not assumed:** Phase 5 records each
selected policy's feasibility separately on VAL-RL, VAL-SELECT and test, with q_test = test / dense test, and Phase 6
tests whether fine-tuning changes the gap between VAL-SELECT and test retention.

## 4.20 Chapter Summary

The design fixes two validation configurations, a test set read only for frozen models, exact zero-count matching, one
PPO configuration and one 5-epoch recovery protocol, pre-registered tests including a ±0.5 pp confidence-interval rule
that is not a formal equivalence test, and measurement gates that stopped the Phase-7B measurements before any paired
comparison.
