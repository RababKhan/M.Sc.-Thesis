"""Revised Paper-3 manuscript (submission version) in the IEEE conference template.

  python experiments/paper3/paper_build_revised.py <unpacked template dir> <out document.xml>
Reuses the verified element builders of paper_build.py (template styles and XML, unchanged). Content only from frozen
results in results/paper3/. Citations are written as [@key] and numbered by order of first appearance (IEEE).
Page structure: page 1 title, authors, abstract, keywords, Introduction; Related Work starts page 2; Experimental
Setup starts page 3 (page breaks before both headings).
"""
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import paper_build as PB  # noqa: E402  (helpers only: runs, para, equation, figure, tbl)

TITLE = "Sparsity Is Not Speed: An Exhaustive CPU-Latency Analysis of Structured Neural Network Pruning"
AUTHOR = PB.AUTHOR
ABSTRACT = (
    "Pruned networks are often compared by parameter sparsity or multiply–accumulate (MAC) count, yet deployment "
    "cost is paid in latency. We test how well these proxies predict measured CPU latency after structured pruning. "
    "For SimpleCNN and ResNet-8 on CIFAR-10 and CIFAR-100, we physically compact all 1,296 policies of a four-unit, "
    "six-action layer-wise pruning space and measure batch-1, single-thread latency on an Intel Core i7-1260P under a "
    "pre-registered protocol. Parameter reduction correlates weakly with latency gain (Spearman ρ = 0.31–0.55) "
    "and MAC reduction strongly (ρ = 0.89–0.98); in 31–39% of policy pairs, the sparser policy is the slower "
    "one. Where pruning happens matters: removing 51% of SimpleCNN’s parameters from its fully connected layer "
    "yields a 1.03× speed-up, whereas a policy that prunes convolution filters and removes 19% of the parameters "
    "yields 2.08×. We then estimate the latency headroom that survives a 98% recovered-accuracy constraint for "
    "ResNet-8, using a one-epoch recovery proxy validated (ρ = 0.98) and calibrated against five-epoch recovery. "
    "The maximum speed-up in this calibrated proxy-feasible region is 1.098×. Accuracy-feasible latency headroom "
    "should therefore be measured before expensive hardware-aware optimization is attempted.")
KEYWORDS = ("structured pruning, CPU inference latency, neural network compression, hardware-aware inference, MACs, "
            "accuracy–latency trade-off")

B = []
PAGE = "<w:pageBreakBefore/>"


def h1(t, newpage=False): B.append(("Heading1", t, PAGE if newpage else ""))
def h2(t): B.append(("Heading2", t, ""))
def p(t): B.append(("BodyText", t, ""))
def eq(t, n): B.append(("equation", t, n))
def fig(n, cap): B.append(("figure", n, cap))
def table(cap, header, rows, widths, foot=None): B.append(("table", cap, header, rows, widths, foot))


# ------------------------------------------------------------------ I. Introduction (page 1, prose only)
h1("Introduction")
p("Deep neural networks are increasingly deployed on CPUs in laptops, edge servers and embedded devices, where "
  "inference cost is paid in time rather than in parameters. Pruning reduces this cost by removing weights, channels "
  "or neurons from a trained network [@han2015], [@li2017], [@hoefler2021], and pruned models are commonly compared "
  "by the fraction of parameters removed or by the floating-point operations or multiply–accumulates (MACs) "
  "saved [@blalock2020]. On a CPU serving interactive or streaming requests, inference often runs at batch size one, "
  "where the latency of a single request rather than throughput determines responsiveness.")
p("These counts are convenient, but they are proxies. Unstructured sparsity leaves tensor shapes unchanged, and dense "
  "CPU kernels do not skip zeroed weights, so a highly sparse model can run no faster than its dense parent "
  "[@hoefler2021]. In convolutional networks, moreover, most parameters often sit in fully connected layers while "
  "most computation sits in convolutions [@li2017], so the number of parameters removed says little about where "
  "time is saved. Even when computation is physically removed, latency also depends on memory access, parallelism "
  "and per-operator work that arithmetic counts do not capture [@ma2018], [@zhang2021nnmeter].")
p("Structured pruning removes whole filters, channels or neurons and therefore shrinks tensors physically [@li2017], "
  "[@fang2023], [@he2024survey]. It is the natural setting in which to ask whether compression proxies track "
  "deployment cost, because every structured policy yields a smaller dense network whose latency can be measured "
  "directly, as platform-aware methods do [@yang2018], [@cai2019], [@wu2019].")
p("This paper examines four questions on one commodity CPU: whether parameter sparsity reliably predicts inference "
  "latency after structured pruning; whether MAC reduction predicts it better; how much latency headroom remains when "
  "the pruned network must be retrained and keep 98% of its recovered accuracy; and whether a hardware-aware search "
  "should be attempted before such headroom has been verified. We study two CIFAR-scale networks, SimpleCNN and "
  "ResNet-8, on CIFAR-10 and CIFAR-100, with a four-unit, six-action layer-wise pruning space that is small enough "
  "to enumerate, so that every policy can be physically compacted and timed on an Intel Core i7-1260P laptop CPU. "
  "We characterize the latency landscape of a finite "
  "structured pruning space exhaustively and evaluate accuracy-feasible latency headroom using a pre-registered "
  "recovery proxy calibrated against five-epoch recovery. All protocols and decision rules were committed to version "
  "control before the corresponding data were collected.")
p("This work makes three contributions. First, it provides an exhaustive latency characterization of all 1,296 "
  "physically compacted structured pruning policies in each of four network–dataset settings, showing that "
  "parameter sparsity and measured CPU latency can diverge substantially. Second, it compares parameter reduction, "
  "MAC reduction and measured latency across these policies, finding a weak parameter–latency correlation, a "
  "strong MAC–latency correlation, and frequent cases in which a more parameter-sparse policy is slower. Third, "
  "it reports a pre-registered recovery and feasibility analysis indicating that, for ResNet-8 on this platform, the "
  "region associated with 98% five-epoch accuracy retention remains below the pre-registered 1.10× speed-up "
  "criterion, which can leave too little headroom to justify an expensive hardware-aware search. The study proposes no new pruning algorithm, and its conclusions "
  "are specific to the measured CPU and networks. Section II reviews related work, Section III describes the "
  "experimental setup, Section IV reports the results, and Sections V and VI discuss and conclude.")

# ------------------------------------------------------------------ II. Related Work (page 2, prose only)
h1("Related Work", newpage=True)
p("Modern magnitude-based pruning removes individual weights of small absolute value and retrains the remaining ones, "
  "reaching high sparsity with little loss of accuracy [@han2015]. The resulting irregular sparsity reduces storage "
  "more readily than time: on general-purpose hardware it becomes faster only with specialized sparse kernels or "
  "hardware support [@hoefler2021]. Structured pruning instead removes entire filters, channels or neurons. Li "
  "*et al.* rank convolution filters by their L1 norm and remove the smallest, producing a smaller dense network that "
  "standard libraries execute without modification [@li2017]. Liu *et al.* showed that for many structured methods, "
  "training the pruned architecture from scratch can match fine-tuning the inherited weights, which suggests that the "
  "resulting architecture is the main outcome of structured pruning [@liu2019]. In the same spirit, Li *et al.* found "
  "that randomly sampled channel configurations, trained under a common protocol, can perform comparably to "
  "sophisticated pruning criteria [@li2022random]. Modern architectures complicate removal, because channels joined by "
  "residual additions or concatenations must be removed together; DepGraph models these couplings explicitly and "
  "groups the parameters that have to be pruned jointly [@fang2023]. A recent survey organizes structured pruning "
  "methods by granularity, importance criterion and search strategy [@he2024survey].")
p("Because layers differ in size and sensitivity, the fraction removed from each layer is itself a design decision. "
  "Rule-based allocation derives per-layer budgets from weight statistics or tensor shapes. LAMP rescales magnitude "
  "scores so that a single global threshold yields layer-adaptive sparsity [@lee2021], and the "
  "Erdős–Rényi-Kernel rule used in sparse training assigns densities that scale with the dimensions "
  "of each layer [@evci2020]. Such rules are inexpensive but do not consult the target hardware. Learning-based methods search the allocation "
  "instead and can include a hardware cost directly in their objective. AMC formulates layer-wise "
  "compression as a reinforcement-learning problem in which an agent selects per-layer compression ratios under either "
  "a resource constraint or an accuracy guarantee [@he2018amc], and policy-gradient methods such as proximal policy "
  "optimization [@schulman2017] are common choices for such agents. In these formulations the reward or constraint is "
  "typically a count of parameters or FLOPs, or a latency value supplied by a model of the hardware, so the fidelity "
  "of that cost signal directly shapes the policies that the search can find.")
p("Hardware-aware methods bring the target platform into the loop. NetAdapt iteratively simplifies a pretrained "
  "network and, at each step, selects the layer reduction that best preserves accuracy while meeting a latency budget "
  "obtained from empirical measurements on the target device [@yang2018]. In neural architecture search, "
  "ProxylessNAS includes hardware latency in its objective through a latency prediction model, making expected "
  "latency differentiable [@cai2019], and FBNet uses a lookup table of operator latencies measured on a target phone "
  "[@wu2019]. HALP casts structural pruning as a knapsack problem that maximizes accumulated importance under a "
  "latency budget estimated from a latency lookup table [@shen2022halp]. Benchmarks such as HW-NAS-Bench provide "
  "measured or estimated hardware costs for entire architecture search spaces on several devices [@li2021hwnas], and "
  "predictors such as nn-Meter estimate model latency on edge devices at the level of fused kernels, so that not "
  "every candidate has to be measured [@zhang2021nnmeter]. NetAdapt and HALP both obtain layer latencies from "
  "lookup tables built by measuring the target device and approximate the latency of a network by summing the "
  "entries of its layers [@yang2018], [@shen2022halp]; NetAdapt applies them in a greedy, layer-by-layer "
  "simplification, whereas HALP allocates channels across all layers at once. Lookup tables and predictors make "
  "large searches affordable, but their estimation error enters the objective directly, and when candidate policies "
  "differ by only a few percent in latency, a small error can reorder them. Benchmarks supply precomputed costs for "
  "the architectures they cover, but these form fixed search spaces rather than the pruned variants of a particular "
  "trained network. Measured latency is itself sensitive to the measurement "
  "protocol: batch size, thread count, warm-up and the operator library all change absolute timings, so "
  "comparisons are meaningful only under fixed and reported conditions.")
p("These efforts rest on the observation that arithmetic counts are indirect measures of speed. Ma *et al.* showed "
  "that networks with similar FLOPs can differ markedly in speed, because memory access cost, the degree of "
  "parallelism and element-wise operations are not counted, and they recommended evaluating speed directly on the "
  "target platform [@ma2018]. nn-Meter attributes a similar gap on edge devices to operator fusion and to "
  "device-specific kernel implementations [@zhang2021nnmeter]. Blalock *et al.* found that many pruning papers report "
  "only parameter or FLOP reductions, often under inconsistent experimental conditions, which makes their efficiency "
  "claims difficult to compare [@blalock2020]. Parameter count adds a further distortion in convolutional networks, "
  "whose parameters often reside in fully connected layers that contribute little computation, so the layers that "
  "dominate model size and those that dominate run time need not coincide [@li2017]. At batch size one, part of "
  "each inference may also be spent in work that does not shrink with channel width, which limits how much of a MAC "
  "reduction appears as a latency reduction.")
p("Existing methods therefore often optimize or report sparsity, FLOPs or MACs, and hardware-aware methods typically "
  "measure latency for the candidates that their search visits or estimate it with a predictor. Fewer studies "
  "characterize an entire structured pruning space with measured CPU latency while also testing how much of that "
  "latency survives when the pruned networks must recover a fixed share of their accuracy. For a small, finite policy "
  "space such a characterization is feasible, and it answers a question that precedes the choice of any search "
  "method: whether an accuracy-feasible region with meaningful latency savings exists at all. It also separates "
  "two causes of a weak search outcome, an optimizer that fails to find good policies and a problem that offers "
  "none, because the full table shows where the fastest feasible policies lie. Proxy quality matters for the same "
  "reason on the accuracy side: when recovering every policy is too expensive, a cheaper recovery proxy decides "
  "which policies count as feasible, and a proxy that ranks policies poorly near the feasibility boundary can "
  "mislead a latency-aware search as much as an inaccurate latency model can. Rather than measuring or predicting "
  "latency only for the candidates that a search visits, this paper measures every policy of a finite structured "
  "space directly and pairs the latency table with an accuracy proxy validated and calibrated against five-epoch "
  "recovery. The resulting characterization covers two CIFAR-scale networks on one CPU; it does not claim that the findings transfer to other "
  "hardware, larger models or other accuracy tolerances.")

# ------------------------------------------------------------------ III. Experimental Setup (page 3 onward)
h1("Experimental Setup", newpage=True)
h2("Networks and Data")
p("We use CIFAR-10 and CIFAR-100 [@krizhevsky2009]. The 50,000 training images are split once into 45,000 for "
  "training and 5,000 for validation, and the validation images are divided into two stratified halves of 2,500: "
  "VAL-RL, used for every accuracy reported here, and VAL-SELECT, which is held back. The official test set is not "
  "used. SimpleCNN has three 3×3 convolutions with 32, 64 and 128 filters, a 256-unit hidden layer and a linear "
  "classifier. ResNet-8 is a CIFAR residual network [@he2016] with a 16-channel stem and three basic blocks of 16, 32 "
  "and 64 channels. The CIFAR-10 variants contain 620,362 and 78,042 parameters, respectively, and the CIFAR-100 "
  "variants 643,492 and 83,892. Each pruned model starts from a frozen, hash-verified dense reference trained in an "
  "earlier stage of this work, and all code uses PyTorch 2.7.1 [@paszke2019].")
h2("Structured Action Space")
p("Each network has four pruning units, and each unit takes one of six actions that remove *r* ∈ {0, 10, 20, "
  "30, 40, 60}% of its channels. A unit with *C* channels keeps")
eq("*k* = max(1, round((1 − *r*)*C*))", 1)
p("channels with the largest L1 norm [@li2017], scored on the dense model so that the result does not depend on the "
  "order of the units. For SimpleCNN the units are the output filters of the three convolutions and the hidden "
  "neurons of the first fully connected layer; the following layer loses the matching inputs, including the 16 "
  "flattened columns of each removed channel, and the classifier is never pruned. For ResNet-8 the first unit is the "
  "stage-1 residual width: one channel set is removed from the stem convolution and its batch normalization, from the "
  "outputs of the second convolution of block 1, and from the inputs of every layer that reads this residual stream, "
  "including the identity and projection shortcuts [@fang2023]. These channels are ranked by the summed L1 norm of "
  "the two filters that write them. The other three units are the internal channels of the three blocks.")
p("All 1,296 action vectors give distinct physical shapes in both networks. Slimmed models and the equivalent masked "
  "models give the same logits (maximum difference 9.5×10^{−6}), parameter and MAC counts equal "
  "closed-form values for all 1,296 policies, and the FLOPs reported by PyTorch’s FlopCounter equal twice the "
  "MACs.")
h2("Latency Measurement")
p("All timing runs on a laptop with an Intel Core i7-1260P (four performance and eight efficiency cores), Windows 11 "
  "and the PyTorch 2.7.1 CPU build with oneDNN 3.7.1, on AC power. The primary configuration uses batch size 1, one "
  "intra-op and one inter-op thread, and a process pinned to one hyper-thread of a performance core before PyTorch is "
  "imported. Each model is timed in evaluation and inference mode on a single preallocated float32 input, and model "
  "construction is never timed. Every shape receives 50 warm-up and 300 timed passes, measured with a nanosecond "
  "counter.")
p("A latency table covers all 1,296 shapes of one network and is measured in five fresh-process sessions, each in a "
  "seeded random order. A block of dense-model passes after every 25 shapes monitors drift, and a session whose "
  "anchors drift by more than 10% is repeated. The latency *L* of a shape is the median over sessions of its "
  "per-session median, and its gain is")
eq("*g*_{L} = 1 − *L* / *L*_{dense}", 2)
p("where *L*_{dense} is the unpruned shape of the same table.")
h2("Recovery Protocol and Pre-Registered Rules")
p("Removing channels without retraining is destructive: removing 10% of every unit of ResNet-8 leaves 21% "
  "validation accuracy. Every evaluated policy is therefore recovered with one fixed protocol inherited unchanged "
  "from earlier experiments of this work: Adam with learning rate 10^{−4}, a cosine schedule over the recovery "
  "budget, batch size 128, random crop and horizontal flip, no weight decay, and one fixed seed that gives every "
  "policy the same minibatch order. We report the retained accuracy")
eq("*q* = *A* / *A*_{dense}", 3)
p("where *A* is the VAL-RL accuracy after recovery and *A*_{dense} that of the dense model after the same recovery. "
  "A complete five-epoch recovered-accuracy landscape across all four settings, comprising 5,184 policies, was "
  "estimated from measured run times to require about 130 hours of continuous computation on this laptop and was "
  "therefore not completed. "
  "The exhaustive ResNet-8 recovery landscape uses one-epoch recovery (*q*_{1}), while five-epoch recovery "
  "(*q*_{5}) is reserved for proxy validation and the pre-registered boundary calibration.")
p("Three rules were committed before their data existed. First, the one-epoch proxy is accepted for a network only "
  "if the Spearman correlation between one-epoch and five-epoch accuracy on a validation sample is at least 0.90. "
  "Second, the proxy threshold τ is the largest *q*_{1} that retains every policy with *q*_{5} ≥ 0.98 in a "
  "stratified calibration sample of 48 policies and leaves at least 20 policies in the full table, minus a fixed "
  "margin of 0.005. Third, a hardware-aware search proceeds only if at least 20 policies pass τ and at least one "
  "of them is at least 1.10× faster than dense in the latency table.")

# ------------------------------------------------------------------ IV. Results
h1("Results")
h2("Choice of the Latency Metric")
p("Before building the tables, we timed six SimpleCNN calibration models in three configurations: batch 1 with one "
  "thread (A), batch 1 with four threads (B), and batch 128 with four threads (C). A configuration was usable if the "
  "latency difference between the dense and the most slimmed model was at least five times the between-session "
  "spread, at least 10% of the dense latency, and positive in every session; the first usable configuration in the "
  "order A, C, B became the primary metric. Configuration A gave a signal-to-noise ratio of 12.3, C 9.6 and B 4.3, so "
  "batch-1, single-thread latency is the primary metric, with a between-session coefficient of variation of 2.1% for "
  "the dense model (Table I). Already in this small set, location matters: the fc-heavy model removes 51% of the "
  "parameters but only 3% of the MACs and is 1.03× faster, whereas the conv-heavy model removes 63% of the "
  "parameters and 65% of the MACs and is 1.47× faster.")
table("Latency Calibration on CIFAR-10 SimpleCNN",
      ["Model", "Params ↓", "MACs ↓", "Median ms", "Speedup"],
      [["Dense", "—", "—", "0.680", "1.00×"],
       ["Light", "19.1%", "17.5%", "0.609", "1.12×"],
       ["Moderate", "50.7%", "49.5%", "0.454", "1.50×"],
       ["Aggressive", "83.9%", "81.7%", "0.261", "2.60×"],
       ["FC-heavy", "51.1%", "2.9%", "0.657", "1.03×"],
       ["Conv-heavy", "62.6%", "65.2%", "0.465", "1.47×"]],
      [60, 45, 45, 45, 45],
      "Light, moderate and aggressive remove 10%, 30% and 60% of every unit; FC-heavy removes 60% of fc1; conv-heavy "
      "removes 60% of conv2 and conv3. Configuration A, five sessions.")
h2("Exhaustive Latency Tables")
p("Table II summarizes the four tables. SimpleCNN shapes span 0.30–0.79 ms and ResNet-8 shapes 0.46–0.77 "
  "ms. Across sessions, the median per-shape coefficient of variation was 1.2–1.6%, and the rank correlation of "
  "latency gains between any two sessions was at least 0.96. The absolute dense latency differed by 16% between the "
  "calibration day and the table measurements while relative gains were stable, so only gains relative to a dense "
  "model measured in the same sessions are interpreted.")
p("Parameter reduction is a weak predictor of latency gain (Fig. 1a), whereas MAC reduction predicts it well (Fig. "
  "1b). Across all pairs of policies with different parameter reductions, the sparser policy is the slower one in 39% "
  "of SimpleCNN pairs and 31% of ResNet-8 pairs; for MAC reduction the inversion rate drops to 14% and 5–6%. The "
  "SimpleCNN policy [5, 5, 1, 0] removes only 18.7% of the parameters but 69.3% of the MACs and runs 2.08× "
  "faster, while removing 60% of the fully connected layer, which holds most parameters but few MACs, removes 51% of "
  "the parameters for about a 1.03× speed-up. A linear fit of latency on MACs has a positive intercept of 0.26 ms "
  "for SimpleCNN (R^{2} = 0.81) and 0.33 ms for ResNet-8 (R^{2} = 0.97). The positive intercept suggests a "
  "substantial latency component that is not removed by channel pruning, which limits the attainable speed-up.")
table("Exhaustive Latency Tables (1,296 Policies Each)",
      ["Network", "Max speedup", "ρ_{P}", "ρ_{M}", "Sparser, slower"],
      [["SimpleCNN, C10", "2.61×", "0.32", "0.89", "39.0%"],
       ["SimpleCNN, C100", "2.59×", "0.31", "0.89", "39.4%"],
       ["ResNet-8, C10", "1.67×", "0.55", "0.98", "30.6%"],
       ["ResNet-8, C100", "1.72×", "0.54", "0.98", "30.8%"]],
      [76, 46, 34, 34, 50],
      "ρ_{P}, ρ_{M}: Spearman correlation of parameter and MAC reduction with latency gain. Last column: "
      "share of policy pairs in which the policy with the larger parameter reduction has the higher latency.")
fig(1, "Latency reduction of all 1,296 physically compacted policies of CIFAR-10 SimpleCNN and ResNet-8 against "
       "(a) physical parameter reduction and (b) MAC reduction (batch 1, one thread, i7-1260P).")
h2("Recovery and Proxy Validity")
p("Five epochs of recovery restore most of the accuracy lost to channel removal: ResNet-8 with 10% removed from every "
  "unit improves from 20.96% to 76.84% VAL-RL accuracy, and the conv-heavy SimpleCNN from 25.84% to 73.36%. The "
  "one-epoch proxy was validated on 156 SimpleCNN policies (126 lightly pruned pilot policies and 30 random ones) and "
  "on 31 ResNet-8 policies (the dense policy and 30 random ones), each recovered for one and for five epochs (Fig. 2, "
  "Table III). The proxy underestimates the five-epoch accuracy in every pair. For ResNet-8 it preserves the ranking "
  "and was accepted. For SimpleCNN the overall ρ of 0.844 is below the pre-registered 0.90: among the lightly "
  "pruned pilot policies, whose five-epoch accuracies lie within a few tenths of a point of each other, the one-epoch "
  "ranking is unreliable (ρ = 0.77), although it is accurate across the random policies (ρ = 0.97). "
  "SimpleCNN’s one-epoch proxy therefore failed the pre-registered acceptance threshold and was not used to build "
  "an exhaustive proxy accuracy landscape; SimpleCNN contributes only its latency evidence.")
table("Validation of the One-Epoch Recovery Proxy",
      ["Metric", "SimpleCNN", "ResNet-8"],
      [["Policy pairs", "156", "31"],
       ["Spearman ρ (A_{1}, A_{5})", "0.844", "0.980"],
       ["Pearson r", "0.973", "0.990"],
       ["Mean bias A_{1} − A_{5} (pp)", "−1.29", "−5.68"],
       ["Top-10 overlap", "3", "10"],
       ["Decision (ρ ≥ 0.90)", "Rejected", "Accepted"]],
      [120, 60, 60],
      "A_{1}, A_{5}: VAL-RL accuracy after one and five recovery epochs (CIFAR-10).")
fig(2, "VAL-RL accuracy after the one-epoch proxy recovery against the five-epoch recovery for the validation "
       "policies of both networks. The dashed line marks equality.")
h2("Latency Room under an Accuracy Constraint")
p("For ResNet-8, the one-epoch proxy was computed for all 1,296 policies. The stratified calibration sample of 48 "
  "policies, eight per bin of *q*_{1}, contained 14 policies with *q*_{5} ≥ 0.98 (Table IV). The pre-registered "
  "rule gave τ = 0.960, with recall 1.00 and precision 0.875 on the sample. The calibrated proxy identifies 52 "
  "candidate policies associated with the five-epoch feasibility region. The fastest of them is 1.098× faster "
  "than dense, and the fastest truly feasible calibration policy is 1.089× faster; every faster calibration "
  "policy, up to 1.22×, has a five-epoch retention below 0.98 (Fig. 3). The "
  "pre-registered 1.10× requirement was therefore not met, and no hardware-aware search was started. The gap "
  "between 1.098× and 1.10× is not scientifically meaningful in itself; the stop is a pre-registered protocol "
  "decision, and the substantive finding is that the region identified by the calibrated proxy offers speed-ups of about 10% "
  "at most.")
table("ResNet-8 Proxy-Calibrated Latency Room at 98% Retention",
      ["Quantity", "Value"],
      [["Calibration policies (five epochs)", "48"],
       ["Calibration policies with q_{5} ≥ 0.98", "14"],
       ["Proxy threshold τ", "0.960"],
       ["Recall / precision at τ (calibration)", "1.00 / 0.875"],
       ["Proxy-feasible policies, q_{1} ≥ τ (of 1,296)", "52"],
       ["Latency of proxy-feasible policies", "0.702–0.770 ms"],
       ["Best proxy-feasible speed-up (rule: ≥ 1.10×)", "1.098×"],
       ["Best five-epoch-feasible calibration policy", "1.089×"]],
      [170, 70],
      "Proxy-feasible policies are candidates identified by the calibrated one-epoch proxy; only the 48 calibration "
      "policies have five-epoch accuracy.")
fig(3, "ResNet-8 speed-up versus retained accuracy. Grey points show one-epoch proxy values for all 1,296 "
       "policies, while orange points show five-epoch values for the 48-policy calibration set. The proxy-feasible "
       "set reaches at most 1.098× speed-up, and the best truly feasible calibration policy reaches 1.089×.")

# ------------------------------------------------------------------ V. Discussion
h1("Discussion")
h2("What the Cost Metric Should Measure")
p("On this CPU, a cost based on parameter count would favor pruning the fully connected layer of SimpleCNN, which "
  "holds 85% of its parameters but 5% of its MACs; such policies save memory but little time. MAC reduction ranks "
  "policies almost as measured latency does (ρ up to 0.98) and is a sound, hardware-independent first "
  "approximation. Measured latency captures shape-specific and non-MAC effects that arithmetic counts miss, and it "
  "remains the deployment-grounded metric. For a finite policy space an exhaustive latency table is inexpensive: the four "
  "tables took about 2.5 hours and remove timing noise from any downstream objective.")
h2("Why ResNet-8 Leaves Little Room")
p("ResNet-8 is compact, and its residual stages are sensitive to channel removal. Its latency–MAC fit has a "
  "positive intercept of 0.33 ms relative to a dense latency of about 0.77 ms, consistent with a substantial latency "
  "component that is not eliminated by channel pruning. The validated proxy indicates that policies compatible with 98% five-epoch retention "
  "are concentrated in the lightly pruned region, where MAC reductions reach at most 18% among the proxy-feasible "
  "policies and latency falls by at most 9%. The evidence therefore points to the accuracy constraint, rather than "
  "search quality alone, as the principal limiter of latency headroom in this setting: an exhaustive enumeration of "
  "the proxy table finds no proxy-feasible policy at 1.10×, although the threshold retained every truly feasible "
  "calibration policy.")
h2("Implications for Hardware-Aware Optimization")
p("The accuracy-feasible latency headroom of a pruning problem can be measured before an optimizer is built. Here an "
  "exhaustive latency table was combined with an accuracy proxy validated against five-epoch recovery and a "
  "pre-registered feasibility check. Under the calibrated proxy and five-epoch boundary validation, the "
  "accuracy-feasible region available to a latency-aware objective, whether pursued by reinforcement learning, "
  "evolutionary search or a greedy method, was narrow and remained below the pre-registered 1.10× criterion. Larger networks, a looser "
  "accuracy tolerance, longer recovery or batched inference, where configuration C reached a 2.76× speed-up in "
  "calibration, are settings in which hardware-aware optimization may have more room to matter.")
h2("Limitations")
p("All latencies come from one laptop CPU (Intel Core i7-1260P, Windows 11, PyTorch CPU backend with oneDNN), with "
  "batch size one and a single thread as the primary metric; absolute latency shifted by 16% between days, so only "
  "relative gains are interpreted, and no GPU or energy measurements were made. The study covers two CIFAR-scale "
  "networks, SimpleCNN and ResNet-8. All accuracies are validation accuracies from a single recovery seed, and the "
  "test set was not used. No exhaustive five-epoch accuracy landscape was computed: the ResNet-8 analysis rests on a "
  "validated one-epoch proxy and a 48-policy calibration, and the SimpleCNN proxy was rejected. Because the "
  "pre-registered latency-room criterion was not met, no reinforcement-learning or other hardware-aware optimization "
  "was run. The 0.98 tolerance and the 1.10× threshold are conventions fixed in advance.")

# ------------------------------------------------------------------ VI. Conclusion
h1("Conclusion")
p("Exhaustive measurement of 5,184 physically compacted structured pruning policies on one CPU shows that parameter "
  "sparsity is not a reliable deployment-cost metric, while MAC count is substantially better, with rank correlations "
  "of 0.89–0.98 against measured latency gain. Measured latency nonetheless remains the deployment-grounded "
  "metric. For ResNet-8, the validated proxy and five-epoch calibration indicate less than 10% latency headroom in "
  "the region associated with 98% recovered-accuracy retention. Accuracy-feasible latency headroom should therefore "
  "be established before expensive hardware-aware optimization is attempted.")

# ------------------------------------------------------------------ references (verified entries; see the report)
REFS = {}
exec(open(os.path.join(HERE, "paper_refs_revised.py"), encoding="utf-8").read())   # defines REFS (key -> IEEE string)


# ------------------------------------------------------------------ citations
def number_citations(blocks, extra_texts):
    order = []
    texts = [ABSTRACT] + [t for b in blocks for t in _texts(b)] + extra_texts
    for t in texts:
        for k in re.findall(r"\[@([\w]+)\]", t):
            if k not in order:
                order.append(k)
    num = {k: i + 1 for i, k in enumerate(order)}

    def group(m):
        keys = re.findall(r"\[@([\w]+)\]", m.group(0))
        ns = sorted(num[k] for k in keys)
        parts, i = [], 0
        while i < len(ns):
            j = i
            while j + 1 < len(ns) and ns[j + 1] == ns[j] + 1:
                j += 1
            parts.append(f"[{ns[i]}]–[{ns[j]}]" if j - i >= 2 else ", ".join(f"[{n}]" for n in ns[i:j + 1]))
            i = j + 1
        return ", ".join(parts)
    return order, lambda t: re.sub(r"\[@[\w]+\](?:, \[@[\w]+\])*", group, t)


def _texts(b):
    if b[0] in ("figure",):
        return [b[2]]
    if b[0] == "table":
        return [b[1]] + [c for r in b[3] for c in r] + ([b[5]] if b[5] else [])
    if b[0] == "equation":
        return [b[1]]
    return [b[1]]


def build(unpacked, out_path):
    sys.path.insert(0, os.path.join(unpacked, ".."))
    from split import load
    head, els, tail = load()
    order, cite = number_citations(B, [])
    missing = [k for k in order if k not in REFS]
    assert not missing, f"no reference entry for {missing}"
    unused = [k for k in REFS if k not in order]
    assert not unused, f"reference entries never cited: {unused}"
    ppr0 = re.search(r"<w:pPr>.*?</w:pPr>", els[0]).group(0)
    title = (f'<w:p>{ppr0}<w:r><w:rPr><w:kern w:val="48"/></w:rPr><w:t xml:space="preserve">{PB.escape(TITLE)}</w:t>'
             f'</w:r></w:p>')
    rpr = '<w:sz w:val="18"/><w:szCs w:val="18"/>'
    alines = []
    for i, line in enumerate(AUTHOR):
        it, t = line.startswith("*"), line.strip("*")
        alines.append(f'<w:r><w:rPr>{"<w:i/>" if it else ""}{rpr}</w:rPr>{"<w:br/>" if i else ""}'
                      f'<w:t xml:space="preserve">{PB.escape(t)}</w:t></w:r>')
    author = (f'<w:p><w:pPr><w:pStyle w:val="Author"/><w:spacing w:before="5pt" w:beforeAutospacing="1"/>'
              f'<w:rPr>{rpr}</w:rPr></w:pPr>{"".join(alines)}</w:p>')
    sect8 = els[8].replace('w:num="3"', 'w:num="1"')
    sect9 = re.sub(r'<w:r><w:br w:type="column"/></w:r>', "", els[9]).replace('w:num="3"', 'w:num="1"')
    body = [f'<w:p><w:pPr><w:pStyle w:val="Abstract"/></w:pPr><w:r><w:rPr><w:i/><w:iCs/></w:rPr><w:t>Abstract</w:t></w:r>'
            f'<w:r><w:t>—</w:t></w:r>{PB.runs(ABSTRACT)}</w:p>',
            f'<w:p><w:pPr><w:pStyle w:val="Keywords"/></w:pPr><w:r><w:t>Keywords—</w:t></w:r>{PB.runs(KEYWORDS)}</w:p>']
    for b in B:
        if b[0] == "equation":
            body.append(PB.equation(b[1], b[2]))
        elif b[0] == "figure":
            body.append(PB.figure(b[1], cite(b[2])))
        elif b[0] == "table":
            cap, header, rows, widths, foot = b[1:]
            body.append(PB.tbl(cite(cap), header, [[cite(c) for c in r] for r in rows], widths, cite(foot) if foot else None))
        else:
            body.append(PB.para(b[0], cite(b[1]), b[2]))
    body.append(PB.para("Heading5", "References"))
    body += [PB.para("references", REFS[k], '<w:ind w:start="17.70pt" w:hanging="17.70pt"/>') for k in order]
    end2 = re.sub(r"<w:r\b.*?</w:r>", "", els[90], flags=re.S)
    new = [title, els[2], els[3], author, sect8, sect9] + body + [end2, "<w:p/>", els[92]]
    open(out_path, "w", encoding="utf-8").write(head + "".join(new) + tail)
    words = len(re.findall(r"\S+", ABSTRACT))
    print(f"wrote {out_path}; {len(order)} references; abstract {words} words; citation order: {order}")


if __name__ == "__main__":
    build(sys.argv[1], sys.argv[2])
