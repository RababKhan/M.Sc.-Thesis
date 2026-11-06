# Chapter 6 — Deployment and Sustainability Analysis

## 6.1 General

Chapter 5 evaluated masked pruning, where compression is measured as weight sparsity. This chapter asks the question
the core framework cannot answer:

> **RQ6. Does model compression translate into real deployment benefit?**

The evidence comes from four later studies (Chapter 4, Table 4.7): **Paper 3**, which measures the CPU latency of every
structured policy (Sections 6.4–6.6); **Phase 7A**, which realises the learned Phase-5 allocation as a physically
smaller network and measures its CPU package energy (Sections 6.7–6.8); **Phase 7B**, which searches directly in the
structured space (Sections 6.9–6.12); and the **energy confirmation** study of the frozen Phase-7B model
(Section 6.13).

Deployment benefit is analysed at three levels (Table 3.6): compression and memory; computation and latency; and
operational energy. The central message is that **an improvement at one level does not guarantee an
improvement at the next.** Each level needs its own measurement, and the accuracy cost must be stated alongside it.

**Reporting conventions:**
- Accuracies are validation accuracies on configuration II (VAL-RL / VAL-SELECT, 2,500 images each), after the
  equal 5-epoch recovery. No test-set number exists for these phases (Section 4.3).
- Phase 7A's pre-registered accuracy split is **VAL-RL**; Phase 7B's is **VAL-SELECT**. Both are reported, and the
  split is always named.
- Latency and energy were measured on the laptop CPU of Table 4.11 (Sections 4.17–4.18).

## 6.2 From Compression to Deployment Efficiency

The core framework produced masked models. A masked network keeps every tensor at its dense shape and stores zeros
in place of removed weights. On a general-purpose CPU with dense kernels, those zeros are still loaded and multiplied
unless sparse formats and kernels are used [@hoefler2021sparsity]. Weight sparsity is therefore a statement about the
model, not about its cost on hardware, and cost indicators such as parameter count, FLOPs and measured speed can
disagree with one another [@dehghani2022misnomer]. This thesis makes no latency or energy claim for its masked models.

To reach the hardware, the extension realises each policy **structurally**: whole channels and neurons are removed,
and a smaller dense network is built [@li2017filters; @he2017channel; @he2024structuredsurvey]. The three levels then
become measurable: parameter count, parameter memory and serialised size exactly from the model (Level 1); MACs in
closed form, checked against hooks, and measured batch-1 CPU latency (Level 2); and CPU package energy per 1,000
inferences from the RAPL hardware counter [@khan2018rapl], with no TDP model or estimate (Level 3).

## 6.3 Structured Physical Pruning Setup

The structured search space is the core space realised physically (Section 4.6.2):
- **Action set and policy count:** the same six ratios per unit, so there are 1,296 policies per network.
- **Keep rule:** each unit keeps max(1, round((1 − r)·C)) channels, those with the largest L1 norm on the dense model.
- **SimpleCNN units:** the three convolutions' filters and the hidden layer's neurons.
- **ResNet-8 units:** the stem is a **coupled residual pruning group**, which sets the stage-1 residual width across
  every tensor that writes or reads it; the three stage units remove **block-internal channels**.
- **Integrity:** every compacted model was checked to contain no masks, to have analytic parameter and MAC counts,
  and to reproduce the masked-equivalent logits within 10⁻⁴.

**Recovery.** Accuracy always follows the equal 5-epoch Phase-6 protocol, with recovery seed 600 for every model.

**Latency protocol** (Section 4.17). Calibration chose configuration A (batch 1, one thread, one performance core);
its dense-minus-aggressive latency difference had a signal-to-noise ratio of 12.3 and a relative reduction of 61.6%,
passing the thresholds of 5 and 10%. Each table was timed in five fresh-process sessions with dense anchors and a 10%
drift limit.

## 6.4 Exhaustive CPU Latency Analysis

Paper 3 timed **all 1,296 physically compacted policies** in each of four tables: SimpleCNN and ResNet-8, each with
10 outputs (CIFAR-10) and with 100 outputs (CIFAR-100). That is **5,184 structured policies** in total. Latency
depends only on shape, so the tables need no trained weights and no accuracy evaluation. Four session attempts
exceeded the drift limit and were repeated; all 20 accepted sessions met it.

**Measurement stability.** The median between-session coefficient of variation per shape was 1.2–1.6%; the Spearman
correlation of latency gains between any two sessions was 0.96–0.995; and raw and anchor-normalised gains agreed at
Spearman ≥ 0.997.

**Relative latency.** The absolute dense latency on the table days was about 16% higher than on the calibration day,
while the dense anchors within each table agreed with its identity shape to within 1%. The analysis therefore uses
latency *relative to the identity shape of the same table*.

**Table 6.1. Exhaustive structured latency tables: cost metrics against measured latency gain** (Spearman rank
correlation over 1,296 policies per table; latency gain g_T = 1 − T/T_identity).

| Table | Policies | Parameter reduction vs latency gain | MAC reduction vs latency gain | Pairs where the more parameter-sparse policy is slower | Maximum speed-up (unconstrained) |
|---|---|---|---|---|---|
| SimpleCNN, CIFAR-10 | 1,296 | 0.32 | 0.89 | 39.0% | 2.61× |
| SimpleCNN, CIFAR-100 | 1,296 | 0.31 | 0.89 | 39.4% | 2.59× |
| ResNet-8, CIFAR-10 | 1,296 | 0.55 | 0.98 | 30.6% | 1.67× |
| ResNet-8, CIFAR-100 | 1,296 | 0.54 | 0.98 | 30.8% | 1.72× |

The maximum speed-up in every table belongs to the most aggressive policy [5, 5, 5, 5]. These maxima describe the
latency space only. They are **not** accuracy-preserving speed-ups: no accuracy constraint is applied in Table 6.1
(Section 6.6).

![Figure 6.1](../paper3/paper_figures_revised/fig1_latency_vs_reduction.png)

**Figure 6.1. Measured latency reduction against (a) physical parameter reduction and (b) MAC reduction, for all
1,296 structured policies of the CIFAR-10 SimpleCNN and ResNet-8 tables (Paper 3).** The CIFAR-100 tables (Table 6.1)
behave the same way.

## 6.5 Parameter Reduction versus MAC Reduction versus Latency

**Result.** Across all four tables (Table 6.1), the Spearman correlation of latency gain was 0.31–0.55 (weak) with
parameter reduction and 0.89–0.98 (strong) with MAC reduction. In **39.0–39.4%** of SimpleCNN policy pairs and
**30.6–30.8%** of ResNet-8 pairs, the policy with more parameter reduction was *slower*.

**Example (CIFAR-10 SimpleCNN table).** Two policies show why the location of pruning matters more than the parameter
count.

**Table 6.2. Two CIFAR-10 SimpleCNN policies: parameter reduction, MAC reduction and measured speed-up.**

| Policy | What it prunes | Parameter reduction | MAC reduction | Speed-up |
|---|---|---|---|---|
| [5, 5, 1, 0] | the first two convolutions at 60% | 18.7% | 69.3% | **2.08×** |
| [0, 0, 0, 5] (fc-heavy) | only the 256-unit hidden layer at 60% | 51.1% | 2.9% | **1.03×** |

In SimpleCNN, the hidden layer holds most of the weights but little of the arithmetic, while the early convolutions
have few weights operating at high spatial resolution; the fc-heavy policy therefore removes almost three times the
parameters for almost no time. The examples describe cost only: pruning the first convolutions at 60% is destructive
in this network (Section 5.4).

ResNet-8 correlates more strongly with parameters (0.54–0.55 vs 0.31–0.32) but still has roughly one inverted pair
in three.

**Interpretation.** Parameter sparsity is a weak proxy for CPU latency, and MAC reduction a strong one. Measured
latency remains the deployment-grounded quantity, because even MACs leave part of the ordering unexplained
(Spearman ρ < 1). The same mechanism reappears in Phase 7A (Section 6.8).

**Claim boundary.** These are batch-1, single-thread results on one laptop CPU. They do not transfer to other CPUs,
thread counts, batch sizes or accelerators. No correlation is reported other than Spearman's.

## 6.6 Accuracy-Feasible Latency Headroom

Table 6.1 shows how much faster a structured network *can* be; deployment needs the speed-up of policies that also
keep accuracy.

**Feasibility criterion.** Paper 3 used the Phase-5 criterion, q₅ ≥ 0.98 in retained VAL-RL accuracy after the
5-epoch recovery. The exhaustive 5-epoch landscape was stopped for computational cost (Section 4.6.3), so a 1-epoch
accuracy proxy was validated first and accepted only if its Spearman correlation with 5-epoch recovery was ≥ 0.90.

**Table 6.3. Validation of the 1-epoch accuracy proxy against 5-epoch recovery.**

| Architecture | Validation pairs | Spearman ρ(A₁, A₅) | Decision |
|---|---|---|---|
| SimpleCNN | 156 | 0.844 | **rejected** (NO-GO) |
| ResNet-8 | 31 | 0.980 | accepted (GO) |

Because the SimpleCNN proxy was rejected, **no accuracy-feasibility claim is made for the SimpleCNN latency tables.**
Its 2.61× maximum is an unconstrained figure only.

**ResNet-8 calibration.** The ResNet-8 validation pairs contained too few truly feasible policies for the
pre-registered threshold rule, so its fallback applied. A pre-committed calibration then recovered 48 frozen ResNet-8
policies for the full 5 epochs, and the proxy threshold was set by a recall-first rule with a fixed 0.005 margin. The
resulting accuracy-feasible latency headroom is given in Table 6.4.

**Table 6.4. ResNet-8 (CIFAR-10): calibrated accuracy-feasible latency headroom.**

| Item | Value |
|---|---|
| Calibration policies (5-epoch recovered) | 48 |
| Truly 5-epoch feasible (q₅ ≥ 0.98) | 14 |
| Proxy threshold τ_proxy | 0.960 |
| Recall / precision at τ_proxy | 1.00 / 0.875 (14 true positives, 2 false positives) |
| Proxy-feasible policies in the 1,296 landscape | 52 |
| Their parameter / MAC reduction | 0–28.9% / 0–18.4% |
| **Best table speed-up among proxy-feasible policies** | **1.098×** |
| Best speed-up among the 14 truly feasible calibration policies | 1.089× ([0, 3, 1, 1]) |
| Proxy-feasible policies with speed-up ≥ 1.10× | 0 |

The 52 proxy-feasible policies are *candidates*, not all truly feasible: only the 48 calibration policies were
confirmed by 5-epoch recovery.

![Figure 6.2](../paper3/paper_figures_revised/fig3_resnet8_latency_room.png)

**Figure 6.2. ResNet-8 (CIFAR-10): table speed-up against retained validation accuracy (Paper 3).** Grey: the 1-epoch
proxy q₁ for all 1,296 policies. Orange: 5-epoch q₅ for the 48 calibration policies. The dotted line is q = 0.98; the
dashed line is the 1.10× latency-room criterion.

**Decision.** The pre-registered latency-room rule required at least 20 proxy-feasible policies **and** at least one
with a speed-up ≥ 1.10×. The first condition held (52); the second did not (best 1.098×).
**Because the latency-room requirement was not met, no hardware-aware PPO search was started.**

The gap between 1.098× and 1.10× is a protocol decision threshold fixed in advance, not a scientifically meaningful
discontinuity. The substantive finding concerns the evidence available under the 2% accuracy constraint: the best
proxy-feasible candidate reached a 1.098× speed-up, while the best confirmed five-epoch-feasible calibration policy
reached 1.089×. Neither satisfied the pre-registered 1.10× GO criterion. The accuracy-feasible latency room found on
this CPU is therefore small, and for SimpleCNN it could not be established at all, because its proxy failed.

**Claim boundary.** No hardware-aware reinforcement-learning result exists. Paper 3 contributes a hardware analysis,
not an RL success.

## 6.7 Phase 7A — Structured Transfer and Energy Evaluation

**Question.** Does the learned PPO allocation, realised structurally, reduce directly measured CPU energy, and at what
accuracy cost?

**Design** (Section 4.9.6). CIFAR-10, SimpleCNN and ResNet-8, four models each:
- **Dense:** the reference, recovered identically.
- **PPO-Structured:** the most frequent Phase-5 C1 policy, transferred to structured units: SimpleCNN [0, 2, 5, 5]
  (3 of 20 seeds), widths [32, 51, 51, 102]; ResNet-8 [1, 1, 1, 3] (9 of 20 seeds), widths [14, 14, 29, 45]. The
  ResNet-8 mapping is approximate.
- **Uniform-Structured:** MAC-matched to PPO, giving common ratios 0.256 and 0.149.
- **LAMP-Structured:** from the Phase-6 LAMP mask at PPO's zero count.

**Measurement** (Section 4.18). After the identical 5-epoch recovery, RAPL package energy was measured over five
fresh-process sessions per architecture, with the dense anchor in every session and 100,000 batch-1 inferences per
model. The stability pilot passed its 3% rule (dense CV 2.30% for SimpleCNN and 1.74% for ResNet-8).

**Table 6.5. Phase 7A: accuracy, compression, latency and measured energy.** Accuracy after the equal 5-epoch
recovery; Δ in pp against the recovered dense model on VAL-RL, the pre-registered split. Latency, energy and EDP are
medians over 5 sessions, paired with the same-session dense anchor.

| Architecture | Method | VAL-RL (Δ) | VAL-SELECT | Params ↓ | MACs ↓ | Latency (ms) | Speed-up | J / 1,000 inf. | Energy ↓ | EDP |
|---|---|---|---|---|---|---|---|---|---|---|
| SimpleCNN | Dense | 80.60 (0) | 80.00 | — | — | 0.7313 | 1.00× | 11.05 | — | 1.000 |
| | PPO-Structured | 75.52 (−5.08) | 76.00 | 80.1% | 42.6% | 0.5711 | 1.28× | 8.54 | 22.8% | 0.603 |
| | Uniform-Structured | 79.12 (−1.48) | 78.48 | 44.7% | 42.5% | 0.5341 | 1.37× | 7.91 | 27.9% | 0.518 |
| | LAMP-Structured | 78.84 (−1.76) | 79.44 | 65.5% | 19.7% | 0.6579 | 1.12× | 9.76 | 12.1% | 0.783 |
| ResNet-8 | Dense | 78.40 (0) | 78.52 | — | — | 0.6984 | 1.00× | 10.88 | — | 1.000 |
| | PPO-Structured | 75.16 (−3.24) | 74.72 | 25.0% | 21.5% | 0.6400 | 1.09× | 9.59 | 13.4% | 0.791 |
| | Uniform-Structured | 75.60 (−2.80) | 76.00 | 16.1% | 19.3% | 0.6459 | 1.07× | 9.64 | 11.8% | 0.818 |
| | LAMP-Structured | 75.24 (−3.16) | 75.72 | 25.0% | 18.2% | 0.6490 | 1.07× | 9.82 | 8.9% | 0.828 |

**Energy results.**
- **Every pruned model used less package energy than the dense model in 5 of 5 sessions.** For each, the exact
  one-sided Wilcoxon signed-rank test gives p = 0.031, the smallest value attainable with five sessions.
- **The p-value is secondary.** The pre-registered decision rests on the 5-of-5 consistency and on the median saving
  exceeding the noise threshold of twice the dense between-session coefficient of variation (4.3% for SimpleCNN; 3.7%
  for ResNet-8). All six pruned models met both conditions.
- PPO's per-session saving ranged from 19.5% to 25.7% on SimpleCNN and from 5.6% to 14.5% on ResNet-8, and the core
  counter and net-of-idle energy agreed with the package result to within 2 percentage points.

**Latency and EDP.** Every pruned model was faster than the same-session dense model, and speed-ups track MAC
reduction far more closely than parameter reduction (Table 6.5); at nearly equal MACs Uniform was still faster than
PPO, so MACs do not explain the whole difference either. Median package power was 14.6–15.6 W for every model. The
normalised energy–delay product (EDP; lower is better) combines energy and latency relative to the same-session dense
model. It is not a carbon metric.

## 6.8 Phase 7A — Accuracy–Energy Trade-off

**Accuracy cost.** After identical recovery, every pruned model lost VAL-RL accuracy (Table 6.5).
**The pre-registered ±0.5 pp accuracy-preservation condition was not met by any pruned method.** On VAL-SELECT, PPO's
losses were 4.00 pp (SimpleCNN) and 3.80 pp (ResNet-8).

**Pre-registered verdicts:**
- **Primary success, Case A** (accuracy preserved and energy consistently reduced): **not met** for either
  architecture.
- **SimpleCNN:** Case B (energy reduced, with a measurable accuracy trade-off) **and Case D** (a baseline offers the
  better trade-off).
- **ResNet-8:** Case B.

**PPO against Uniform on SimpleCNN.** At almost identical MACs (42.6% vs 42.5% removed), MAC-matched Uniform-Structured
was **3.60 pp more accurate** than PPO-Structured on VAL-RL. It also **used less energy in 5 of 5 sessions** (7.91 vs
8.54 J per 1,000 inferences; median per-session difference 8.0%). **PPO does not dominate Uniform in Phase 7A: on
SimpleCNN, Uniform was better on both axes.** PPO concentrated its 80.1% parameter removal in conv3 and the hidden
layer, which bought less time than Uniform's broad channel removal at the same MAC count (latency 0.571 vs 0.534 ms),
the Section-6.5 mechanism.

**Other comparisons** (between-method tests are two-sided exact Wilcoxon tests on five session pairs, minimum
attainable p = 0.0625, so they are descriptive only):
- **PPO vs LAMP on SimpleCNN:** neither dominates. PPO used less energy in 5 of 5 sessions (12.5% median difference)
  but was 3.32 pp less accurate.
- **ResNet-8:** PPO, Uniform and LAMP showed **no clear difference** in the energy–accuracy trade-off. Against
  Uniform, accuracy differed by 0.44 pp and PPO used less energy in only 3 of 5 sessions; against LAMP, accuracy
  differed by 0.08 pp and PPO used less energy in 4 of 5.

![Figure 6.3](../phase7a_sustainability/figures/fig_A_accuracy_vs_energy.png)

**Figure 6.3. Phase 7A: VAL-RL accuracy after recovery against measured package energy per 1,000 inferences.**
Points: medians. Horizontal bars: range over the five sessions. Shaded band: dense accuracy ± 0.5 pp; no pruned model
lies inside it.

**Interpretation.** Structured realisation made the compression real at the hardware: parameters, MACs, latency and
directly measured package energy all fell, consistently across sessions. The learned masked allocation, however, did
not transfer well: its structured form lost 5.1 and 3.2 pp of VAL-RL accuracy, and on SimpleCNN a plain uniform
allocation was both more accurate and more energy-efficient.

**Claim boundary.**
- Phase 7A does not show accuracy-preserving energy reduction.
- It does not show that PPO is more energy-efficient than Uniform.
- It says nothing about the masked Phase-5/6 models themselves.
- The energy is CPU package energy on one laptop, not whole-system or lifecycle energy.

## 6.9 Motivation for Phase 7B

Phase 7A exposed a design problem, not a measurement problem: a policy learned for *masked* pruning, realised as
channel removal and recovered, was neither accurate nor efficient enough, and for ResNet-8 the masked units do not
match the structured units.

Phase 7B was therefore designed **after** Phase 7A, as a separate variant whose agent chooses structured actions
directly. It was not part of the original framework.

## 6.10 Phase 7B — Deployment-Oriented Structured PPO

**Design** (Sections 3.16.3 and 4.9.7; summarised here). The deployment-oriented structured PPO searches directly in
the structured space. Its search reward used a *surrogate* (predicted) energy saving, so the name implies no measured
energy result.
- **Scope, actions and state:** SimpleCNN on CIFAR-10 only; structured ratios per unit, each a physical width
  configuration; an eight-feature state including centred structured sensitivity, plus the centred sensitivity prior
  (β = 1.0).
- **Accuracy proxy:** feasibility judged by a 2-epoch recovery proxy on VAL-RL, accepted before the search (Spearman
  0.948 against 5-epoch recovery; threshold τ̂ = 0.98849; recall 1.00 and precision 0.655 on the validation sample).
- **Energy surrogate:** Ê = −0.117 + 13.762·T_table (J per 1,000 inferences, T in ms), fitted to directly measured
  energy of 200 calibration policies and accepted by cross-validation (Spearman 0.996, median absolute percentage error
  1.05%). The first candidate in the fixed order, MACs alone, had failed (Spearman 0.946 against the required 0.95;
  lowest-energy-quartile error 8.55% against 5%), so measured latency, not MACs, was the accepted predictor.
- **Reward and selection:** a constraint-first reward on the proxy, with α = 4 fixed on development seeds; a feasible
  archive with lowest-predicted-energy selection; 20 confirmatory seeds (700–719).
- **Recovery and accuracy criterion:** the equal 5-epoch recovery for every final model; accuracy judged on
  VAL-SELECT, which is held out from the search and selection.

**Status of the surrogate.** The surrogate supported the search only. **It is not measured energy, and it is not
evidence for the energy of any final Phase-7B model.**

**Selection outcome.** Every one of the 20 seeds selected a policy. The primary policy, [2, 2, 2, 5], was selected in
**5 of 20** seeds; its widths **[26, 51, 102, 102]** remove 20% of the three convolutions' channels and 60% of the
hidden neurons. The six other policies, each selected by 1–4 seeds, were also recovered and evaluated. The primary
policy's proxy retention, 0.9890, lies just above the threshold of 0.98849: the selections sit at the proxy
feasibility boundary.

## 6.11 Phase 7B — Memory and Computation Results

Table 6.6 compares the recovered Phase-7B model with the dense model, matched baselines and old Phase-7A model.

**Table 6.6. Phase 7B: accuracy, memory and computation after the equal 5-epoch recovery (SimpleCNN, CIFAR-10).**
Memory is parameter memory, Σ numel × element size (float32). Latency and energy were **not measured** (Section 6.13).

| Model | Widths | VAL-SELECT (Δ pp) | VAL-RL (Δ pp) | Parameter memory | Memory ↓ | MACs (M) | MACs ↓ |
|---|---|---|---|---|---|---|---|
| Dense | [32, 64, 128, 256] | 80.00 (0) | 80.60 (0) | 2,481,448 B | — | 10.85 | — |
| **New PPO [2, 2, 2, 5]** | [26, 51, 102, 102] | **78.88 (−1.12)** | 80.00 (−0.60) | 908,916 B | **63.4%** | 6.94 | **36.0%** |
| Uniform, r = 0.208 | [25, 51, 101, 203] | 78.76 (−1.24) | 79.76 (−0.84) | 1,555,908 B | 37.3% | 6.93 | 36.2% |
| LAMP, s = 56.2% | [32, 58, 103, 94] | 79.64 (−0.36) | 79.12 (−1.48) | 909,932 B | 63.3% | 8.76 | 19.3% |
| Old PPO (7A) [0, 2, 5, 5] | [32, 51, 51, 102] | 76.00 (−4.00) | 75.52 (−5.08) | 493,836 B | 80.1% | 6.23 | 42.6% |

**Level 1, memory.** The new PPO model needs **63.4% less parameter memory** than the dense model (908,916 vs 2,481,448
bytes). Its serialised checkpoint is 63.3% smaller.

**Level 2, computation.** It needs **36.0% fewer MACs** (6.94 M vs 10.85 M per image).

**Spread across seeds.** The seven distinct per-seed selections all removed between 34.1% and 37.5% of the MACs, but
their parameter-memory reduction ranged from 29.5% to 63.4%. The search rewarded predicted energy, which tracks
latency, and therefore MACs; the selections differ mainly in how many hidden-layer neurons they remove, which changes
memory a great deal and MACs very little (Section 6.5).

**Matched baselines** (descriptive). MAC-matched Uniform removes about the same MACs but much less memory;
parameter-matched LAMP removes about the same memory but far fewer MACs (Table 6.6).

**Latency.** Latency was not measured for Phase 7B, so Level 2 is established for MACs only. The Paper-3 table
latency of the policy's shape was a surrogate input, not a Phase-7B measurement.

## 6.12 Phase 7B — Accuracy Trade-off

**Result on the pre-registered split (VAL-SELECT).** The new PPO model reached 78.88% against the recovered dense
model's 80.00%, a loss of **1.12 pp**. Every other per-seed selection lost between 1.08 and 1.28 pp.

**Comparison with the Phase-7A transfer.** Measured on the same split and protocol, the old Phase-7A PPO model lost
**4.00 pp**, so the redesign reduced the loss by 2.88 pp. Phase 7A's widely quoted 5.08 pp loss is on VAL-RL, Phase
7A's criterion split; comparing it directly with the 1.12 pp VAL-SELECT figure would mix splits. On VAL-RL the new
model's loss was 0.60 pp, against 5.08 pp for the old model.

**Joint deployment-efficiency success criterion.** The pre-registration names it GREEN-SUCCESS (Section 4.9.7).
- **Result: met by 0 of 20 seeds.** Every selected policy lost more than the allowed 0.5 pp on VAL-SELECT, which alone
  decides the outcome.
- The latency and energy conditions could not be assessed either, because no measurement session took place
  (Section 6.13).

**Comparative boundaries** (descriptive; no energy verdict is possible for any model):
- **Against Uniform:** the new PPO model is 0.12 pp more accurate on VAL-SELECT, at similar MACs (36.0% vs 36.2%
  removed) and with much lower memory (63.4% vs 37.3% removed).
- **Against LAMP:** the new PPO model is 0.76 pp less accurate, removes more MACs (36.0% vs 19.3%), and removes about
  the same parameter memory (63.4% vs 63.3%).
- **LAMP-Structured** is the only pruned model within 0.5 pp of dense on VAL-SELECT (−0.36 pp). Its joint criterion is
  not assessable, because energy was never measured.

![Figure 6.4](../phase7b_green_pruning/figures/fig_accuracy_planes.png)

**Figure 6.4. Phase 7B: VAL-SELECT accuracy after the equal 5-epoch recovery against MACs per image (left) and
parameter memory (right).** Dashed line: dense accuracy − 0.5 pp, the accuracy condition of the joint criterion. Grey
points: the other per-seed selections and the enumeration reference.

**Interpretation.** Learning directly in the structured space improved the memory–computation–accuracy trade-off
substantially over transferring a masked policy, but did not reach the 0.5 pp criterion. That the selections sit at the
proxy feasibility boundary, with proxy precision 0.655 in validation, is consistent with the shortfall, although the
data do not isolate the proxy as its cause.

**Claim boundary.**
- Phase 7B shows no measured latency or energy reduction.
- It is not accuracy-preserving.
- It does not beat Uniform or LAMP.
- The joint deployment-efficiency criterion was not achieved.

## 6.13 Direct Energy Confirmation and Measurement Failure

Two independent attempts were made to measure Phase-7B energy directly. Both stopped at the pre-registered stability
check, before any comparison between models.

**Attempt 1, the Phase-7B confirmatory pilot.** N may be doubled once, then STOP.

**Table 6.7. Phase-7B confirmatory energy pilot (dense model only; limit CV ≤ 3%).**

| Step | N per block | CV of six dense blocks | Decision |
|---|---|---|---|
| First pilot | 100,000 | **11.36%** | unstable; N doubled |
| Doubled-N attempt | 200,000 | — | voided by recorded user decision after the CPU-load gate stopped it during foreground activity |
| Permitted rerun, laptop idle | 200,000 | **4.01%** | unstable → **hardware validation STOPPED** |

No confirmatory energy session was run.

**Attempt 2, the separate energy confirmation study.**
- **Protocol.** Written and committed before any data, for the frozen Phase-7B models; all five checkpoint hashes
  were verified. The workload added a fixed 180 s thermal soak and used 200,000 inferences per block.
- **Qualification.** Dense model only, 3 fresh runs × 2 blocks = **6 blocks**. Energy per 1,000 inferences ranged from
  9.47 to 15.22 J (mean 12.14 J). The coefficient of variation was **19.92%**, against the **3%** criterion. All runs
  passed the validity gates, with in-block system load 7.6–10.2%, below the 15% limit.
- **Outcome.** The setup was **not qualified**. There was one permitted attempt, and the limit was not relaxed.
  **No Dense-versus-PPO paired session ran.**

**Result.** **Direct energy confirmation for Phase 7B remains unresolved.** No estimate from latency or from the
surrogate is substituted.

**What the qualification recorded** (descriptive). The workload was steady: median latency stayed within
1.042–1.052 ms across the six blocks. Package power was not: it ranged from 9.1 to 14.5 W (core PP0 6.7 to 9.7 W, the
rest of the package 2.4 to 5.1 W), and the thermal-zone reading ranged from 72 to 92 °C, with the coolest run giving the lowest
energy. The instability therefore lay in the package's power draw under identical work, not in the timing of the inference.
The measurements do not identify its source, and only the dense model was measured.

**The contrast with Phase 7A.** The same software counter on the same laptop had passed its pilot in Phase 7A, with CV
2.30% and 1.74%. The thesis records this contrast without claiming a cause. A software counter on a laptop is not a
guaranteed-stable instrument, which is why each study had to qualify it [@jay2023powermeters; @raffin2024dissecting].

## 6.14 Three-Level Sustainability Assessment

Table 6.8 brings the three levels of Section 6.2 together for Phases 7A and 7B, with the accuracy cost of each.

**Table 6.8. Three-level deployment-efficiency assessment.**

| Level | Metric | Phase 7A evidence (structured transfer) | Phase 7B evidence (deployment-oriented structured PPO) | Conclusion |
|---|---|---|---|---|
| **1. Compression / memory** | parameter count / parameter memory; serialised size | parameters −80.1% (SimpleCNN) and −25.0% (ResNet-8) | parameter memory −63.4%; serialised −63.3% | **Established** in both phases |
| **2. Computation / latency** | MACs; measured batch-1 latency | MACs −42.6% / −21.5%; **measured** speed-up 1.28× / 1.09× | MACs −36.0%; latency **not measured** | MACs established in both; latency measured for 7A only. Paper 3: MACs are a strong latency proxy, parameters a weak one |
| **3. Operational energy** | RAPL package J per 1,000 inferences; EDP | **Measured:** −22.8% / −13.4%, lower in 5/5 sessions; EDP 0.603 / 0.791 | **Unresolved:** pilot CV 11.36% → 4.01%; confirmation CV 19.92% (limit 3%) | Real energy reduction demonstrated in 7A only |
| **Accuracy** (criterion ±0.5 pp) | recovered validation accuracy | **Failed:** VAL-RL −5.08 / −3.24 pp; no pruned method within 0.5 pp; Uniform dominated PPO on SimpleCNN | **Failed:** VAL-SELECT −1.12 pp; 0/20 seeds met the joint criterion | The accuracy trade-off was too large in 7A and reduced, but not removed, in 7B |

Three readings follow. **Level 1 does not imply Level 2:** the Phase-7A SimpleCNN PPO model removed 80.1% of the
parameters but only 42.6% of the MACs, and MAC-matched Uniform, with far fewer parameters removed, ran faster.
**Level 2 tracked Level 3 in Phase 7A,** because package power was nearly constant across models. **Level 3 must be
measured:** in Phase 7B the instrument could not be qualified, and neither the MAC reduction nor the energy surrogate
substitutes for a measurement. A deployment saving bought with a material accuracy loss is not presented as a
sustainability result.

## 6.15 Answer to RQ6

Compression can translate into real deployment savings, but not automatically, and not at every level at once.
1. **Masked sparsity alone gives no hardware evidence.** Physical structured pruning is needed for dense-kernel CPU
   cost to fall.
2. **Even structurally, parameter reduction is a weak guide to latency.** Across 5,184 measured structured policies,
   Spearman ρ was 0.31–0.55 for parameter reduction and 0.89–0.98 for MAC reduction. In 31–39% of policy pairs the
   sparser policy was slower.
3. **Accuracy-feasible headroom is small.** For ResNet-8, the best proxy-feasible candidate reached a 1.098×
   speed-up, while the best confirmed five-epoch-feasible calibration policy reached 1.089×. Neither satisfied the
   pre-registered 1.10× GO criterion, so the hardware-aware search was not started.
4. **Phase 7A: real energy savings, with material accuracy loss.** Directly measured CPU package energy fell by 22.8%
   (SimpleCNN) and 13.4% (ResNet-8), with VAL-RL losses of 5.08 and 3.24 pp; on SimpleCNN, MAC-matched uniform
   allocation was both more accurate and more energy-efficient.
5. **Phase 7B: a much better trade-off, but neither criterion met.** Parameter memory fell by 63.4% and MACs by 36.0%
   at a 1.12 pp VAL-SELECT loss (4.00 pp for the transferred policy). No seed met the 0.5 pp accuracy criterion, and
   direct energy confirmation failed its stability qualification.

**Therefore** deployment benefit must be established separately for memory, computation, latency and energy, each
with its accuracy cost. An improvement at one level is a hypothesis about the next, not evidence for it.

## 6.16 Chapter Summary

Paper 3 showed that parameter reduction is a weak latency proxy (Spearman ρ 0.31–0.55) and MAC reduction a strong one
(Spearman ρ 0.89–0.98), and that the accuracy-feasible latency headroom of ResNet-8 was too small for a hardware-aware
search. Phase 7A showed that structured pruning reduces directly measured CPU package energy, but with VAL-RL losses
beyond the pre-registered limit and with Uniform better than PPO on SimpleCNN. Phase 7B, designed in response,
improved the memory–computation–accuracy trade-off, but no seed met the joint deployment-efficiency criterion, and its
energy remains unresolved because both measurement attempts failed the stability check. Chapter 7 discusses these
results together with those of Chapter 5.
