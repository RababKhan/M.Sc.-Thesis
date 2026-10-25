"""Build the Paper-3 manuscript into the IEEE conference template (template/conference-template-a4.docx, unchanged).

  python experiments/paper3/paper_build.py <unpacked template dir> <out document.xml>
Every element reuses the template's own styles and XML: title, author block, Abstract, Keywords, Heading1/2,
BodyText, bulletlist, equation, tablehead + table (template table properties), tablefootnote, figurecaption,
references. Template guidance text, the funding text box and the figure text box are removed; the template's
section breaks (title, author block, two-column body) are kept. Figures are left as [[FIGn]] markers and inserted
by Word (paper_word.ps1). Inline markup: **bold**, *italic*, _{sub}, ^{sup}.
"""
import re
import sys
from xml.sax.saxutils import escape

sys.path.insert(0, sys.argv[1] + "/..")

# ------------------------------------------------------------------ content
TITLE = "Sparsity Is Not Speed: An Exhaustive CPU-Latency Analysis of Structured Pruning for Hardware-Aware RL"
AUTHOR = ["[Author Name]", "*[Department]*", "*[Organization]*", "[City, Country]", "[e-mail address]"]
ABSTRACT = (
    "Reinforcement-learning (RL) pruning methods usually reward sparsity, assuming that fewer parameters mean faster "
    "inference. We test this assumption exhaustively on a commodity CPU before training any agent. For CIFAR-10/100 "
    "SimpleCNN and ResNet-8, we physically compact all 1,296 policies of a four-unit, six-action structured layer-wise "
    "pruning space and measure batch-1, single-thread latency on an Intel Core i7-1260P with a pre-registered protocol "
    "(performance-core pinning, five fresh-process sessions, dense anchors). Parameter reduction is a weak predictor of "
    "latency gain (Spearman ρ = 0.31–0.55), whereas MAC reduction is a strong one (ρ = 0.89–0.98); "
    "in 31–39% of policy pairs, the sparser policy is the slower one. A SimpleCNN policy that removes 51% of the "
    "parameters is only 1.03× faster, while one that removes 19% is 2.08× faster. We then measure how much "
    "latency an accuracy-constrained search could gain. With a fixed 5-epoch recovery and a pre-registered 1-epoch "
    "proxy (accepted for ResNet-8, ρ = 0.98; rejected for SimpleCNN, ρ = 0.84), the fastest ResNet-8 policy "
    "predicted to keep 98% of the recovered validation accuracy is only 1.098× faster than the dense model. Hardware-aware pruning rewards should "
    "therefore use measured or MAC-based cost, and the latency headroom of a pruning problem should be measured before "
    "an RL search is built on it.")
KEYWORDS = ("structured pruning, CPU inference latency, hardware-aware neural network compression, reinforcement "
            "learning, proxy validation, reproducibility")

B = []                                     # body blocks


def h1(t): B.append(("Heading1", t))
def h2(t): B.append(("Heading2", t))
def p(t): B.append(("BodyText", t))
def bullet(t): B.append(("bulletlist", t))
def eq(t, n): B.append(("equation", t, n))
def fig(n, cap): B.append(("figure", n, cap))
def table(cap, header, rows, widths, foot=None): B.append(("table", cap, header, rows, widths, foot))


h1("Introduction")
p("Pruning removes weights, channels or neurons from a trained network to make inference cheaper [1]–[3]. "
  "Layer-wise methods decide how much to remove from each layer, and several of them search this allocation with "
  "reinforcement learning (RL) [5] or derive it from layer-adaptive scores [6], [7]. The search is usually driven by "
  "a proxy for efficiency, such as the fraction of removed parameters or the number of floating-point operations. "
  "These proxies are convenient, but what a deployed system pays for is time.")
p("Unstructured pruning makes the gap between proxy and time explicit: zeroed weights stay inside dense tensors, and "
  "dense CPU kernels do not skip them, so latency is unchanged [11], [12]. Structured pruning removes whole filters or "
  "neurons, so tensors physically shrink [2], [4]. Even then, the parameter count does not determine latency. "
  "Parameters and arithmetic are distributed differently across layers, and every inference also pays a fixed "
  "framework overhead. Platform-aware methods therefore measure latency directly or through lookup tables [5], "
  "[8]–[10].")
p("This paper asks two questions that should be answered before a latency-aware RL pruning agent is trained. First, "
  "on a commodity CPU, how well do parameter sparsity and multiply–accumulate (MAC) count predict the measured "
  "latency of structured pruning policies? Second, once the pruned network is retrained under a fixed recovery budget "
  "and must keep a fixed share of its accuracy, is enough latency left to optimize? We answer both exhaustively for a "
  "finite space of 6^{4} = 1,296 layer-wise policies on CIFAR-scale networks. Every protocol and decision rule was "
  "committed to version control before the corresponding data were collected. Our contributions are:")
bullet("an exhaustive, pre-registered latency study of all 1,296 physically compacted policies of four network and "
       "dataset pairs, timed at batch size 1 on a pinned performance core in five fresh-process sessions;")
bullet("evidence that parameter reduction is a poor latency proxy (ρ = 0.31–0.55) and MAC reduction a good "
       "one (ρ = 0.89–0.98), with 31–39% of policy pairs inverted for parameters;")
bullet("a validated proxy design, in which a 1-epoch recovery proxy is accepted for ResNet-8 (ρ = 0.980 against "
       "5-epoch recovery) and rejected for SimpleCNN (ρ = 0.844) by a pre-registered rule;")
bullet("a pre-registered feasibility analysis showing that ResNet-8 policies predicted to keep 98% of the recovered "
       "accuracy are at most 1.098× faster, so a latency-aware search has little room to act on this platform.")

h1("Related Work")
p("*Pruning granularity.* Magnitude pruning of individual weights yields high sparsity [1], but irregular sparsity "
  "needs specialized kernels to become faster [12]. Filter pruning ranks whole filters, for example by their L1 norm, "
  "and removes them [2]; Liu *et al.* showed that the resulting architecture matters more than the inherited weights "
  "[3]. In residual networks, channels joined by an addition must be removed together, and DepGraph formalizes these "
  "dependency groups [4].")
p("*Layer-wise allocation.* AMC trains an RL agent to choose per-layer compression ratios under a resource-constrained "
  "or accuracy-guaranteed reward [5]. LAMP derives layer-adaptive sparsity from normalized magnitude scores [6], and "
  "ERK sets layer densities from tensor shapes [7]. Proximal policy optimization [13] is a common choice for such "
  "agents.")
p("*Hardware awareness.* NetAdapt adapts a network to a target platform with empirical latency measurements [8]. "
  "ProxylessNAS and FBNet use measured or table-based latency inside neural architecture search [9], [10]. Blalock "
  "*et al.* criticize pruning results that report only parameter or FLOP reductions [11]. Our study does not propose a "
  "search method. It measures, for an entire finite policy space, how much latency the common proxies explain and how "
  "much latency remains under an accuracy constraint.")

h1("Experimental Setup")
h2("Networks and Data")
p("We use CIFAR-10 and CIFAR-100 [15]. The 50,000 training images are split once into 45,000 for training and 5,000 "
  "for validation. The validation images are divided into two stratified halves of 2,500: VAL-RL, used for every "
  "accuracy reported here, and VAL-SELECT, which is held back. The official test set is not used in this study. The "
  "networks are SimpleCNN (three 3×3 convolutions with 32, 64 and 128 filters, a 256-unit hidden layer and a "
  "linear classifier; 620,362 parameters) and a CIFAR ResNet-8 [14] (a 16-channel stem and three basic blocks with 16, "
  "32 and 64 channels; 78,042 parameters). Each pruned model starts from a frozen, hash-verified dense reference "
  "trained in an earlier stage of this work. All code uses PyTorch 2.7.1 [16].")
h2("Structured Action Space")
p("Each network has four pruning units, and each unit takes one of six actions that remove *r* ∈ {0, 10, 20, "
  "30, 40, 60}% of its channels. A unit with *C* channels keeps")
eq("*k* = max(1, round((1 − *r*)*C*))", 1)
p("channels with the largest L1 norm [2], scored on the dense model so that the result does not depend on the order "
  "of the units. For SimpleCNN the units are the output filters of the three convolutions and the hidden neurons of "
  "the first fully connected layer. The following layer loses the matching inputs, including the 16 flattened "
  "columns of each removed channel, and the classifier is never pruned. For ResNet-8 the first unit is the stage-1 "
  "residual width. One channel set is removed from the stem convolution and its batch normalization, from the "
  "outputs of the second convolution of block 1, and from the inputs of every layer that reads this residual stream, "
  "including the identity and projection shortcuts [4]. These channels are ranked by the summed L1 norm of the two "
  "filters that write them. The other three units are the internal channels of the three blocks.")
p("All 1,296 action vectors give distinct physical shapes in both networks. Slimmed models and the equivalent "
  "masked models give the same logits (maximum difference 9.5×10^{−6}). Parameter and MAC counts equal "
  "closed-form values for all 1,296 policies, and the FLOPs reported by PyTorch's FlopCounter equal twice the MACs.")
h2("Latency Measurement")
p("All timing runs on a laptop with an Intel Core i7-1260P (four performance and eight efficiency cores), Windows 11 "
  "and the PyTorch 2.7.1 CPU build with oneDNN 3.7.1, on AC power. The primary configuration uses batch size 1, one "
  "intra-op and one inter-op thread, and a process pinned to one hyper-thread of a performance core before PyTorch "
  "is imported. Each model is timed in evaluation and inference mode on a single preallocated float32 input; model "
  "construction is never timed. Every shape receives 50 warm-up passes and 300 timed passes, measured with a "
  "nanosecond counter.")
p("A latency table covers all 1,296 shapes of one network and is measured in five fresh-process sessions, each in a "
  "seeded random order. A block of dense-model passes after every 25 shapes monitors drift, and a session whose "
  "anchors drift by more than 10% is repeated. The latency *L* of a shape is the median over sessions of its "
  "per-session median, and its gain is")
eq("*g*_{L} = 1 − *L* / *L*_{dense}", 2)
p("where *L*_{dense} is the unpruned shape of the same table.")
h2("Recovery and Pre-Registered Decision Rules")
p("Removing channels without retraining is destructive: removing 10% of every unit of ResNet-8 leaves 21% "
  "validation accuracy. Every policy is therefore recovered with one fixed protocol, inherited unchanged from earlier "
  "experiments of this work: Adam with learning rate 10^{−4}, cosine annealing over the budget, batch size 128, "
  "random crop and horizontal flip, no weight decay, and one fixed seed that gives every policy the same minibatch "
  "order. We report the retained accuracy")
eq("*q* = *A* / *A*_{dense}", 3)
p("where *A* is the VAL-RL accuracy after recovery and *A*_{dense} is that of the dense model after the same "
  "recovery. A full 5-epoch recovery of every policy would take about 130 CPU-hours on this laptop. We therefore use "
  "the first recovery epoch as a search proxy (*q*_{1}) and the full five epochs (*q*_{5}) for every claim. Three "
  "rules were committed before their data existed:")
bullet("the proxy is accepted for a network only if the Spearman correlation between 1-epoch and 5-epoch accuracy is "
       "at least 0.90;")
bullet("the proxy threshold τ is the largest *q*_{1} that keeps every policy with *q*_{5} ≥ 0.98 in a "
       "stratified calibration sample of 48 policies and leaves at least 20 policies in the full table, minus a "
       "margin of 0.005;")
bullet("a latency-aware search proceeds only if at least 20 policies pass τ and at least one of them is at "
       "least 1.10× faster than dense.")

h1("Results")
h2("Choice of the Latency Metric")
p("Before building the tables, we timed six SimpleCNN calibration models in three configurations: batch 1 with one "
  "thread (A), batch 1 with four threads (B), and batch 128 with four threads (C). A configuration was usable if the "
  "latency difference between the dense model and the most slimmed model was at least five times the between-session "
  "spread, at least 10% of the dense latency, and positive in every session. The first usable configuration in the "
  "order A, C, B became the primary metric. Configuration A gave a signal-to-noise ratio of 12.3, C 9.6 and B 4.3, so "
  "batch-1, single-thread latency is the primary metric (Table I). The between-session coefficient of variation of "
  "the dense model was 2.1%. Even in this small set, the fc-heavy model removes 51% of the parameters but only 3% of "
  "the MACs and is just 1.03× faster. The conv-heavy model removes 63% of the parameters and 65% of the MACs and "
  "is 1.47× faster.")
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
      "removes 60% of conv2 and conv3. Configuration A (batch 1, one thread on a pinned performance core), five sessions. "
      "Signal-to-noise: A 12.3, C 9.6, B 4.3.")
h2("Exhaustive Latency Tables")
p("Table II summarizes the four tables. SimpleCNN shapes span 0.30–0.79 ms (up to 2.61× faster than "
  "dense), and ResNet-8 shapes span 0.46–0.77 ms (up to 1.67×). Across sessions, the median per-shape "
  "coefficient of variation was 1.2–1.6%, and the rank correlation of latency gains between any two sessions "
  "was at least 0.96. The absolute dense latency differed by 16% between the calibration day and the table "
  "measurements, while relative gains were stable. We therefore interpret only gains relative to a dense model "
  "measured in the same sessions.")
p("Parameter reduction is a weak predictor of latency gain (Fig. 1a): Spearman ρ is 0.31–0.32 for "
  "SimpleCNN and 0.54–0.55 for ResNet-8. MAC reduction predicts it well (Fig. 1b; ρ = 0.89 and 0.98). "
  "Across all pairs of policies with different parameter reductions, the sparser policy is the slower one in 39% of "
  "SimpleCNN pairs and 31% of ResNet-8 pairs. For MAC reduction, the inversion rate drops to 14% and 5–6%.")
p("The extremes are instructive. The SimpleCNN policy [5, 5, 1, 0] removes only 18.7% of the parameters but 69.3% of "
  "the MACs and runs 2.08× faster. Removing 60% of the fully connected layer, which holds most parameters but "
  "few MACs, removes 51% of the parameters for a 1.03× speed-up. A linear fit of latency on MACs gives an "
  "intercept of 0.26 ms for SimpleCNN (R^{2} = 0.81) and 0.33 ms for ResNet-8 (R^{2} = 0.97). This fixed "
  "per-inference cost is not reduced by any channel removal, and it caps the attainable speed-up.")
table("Exhaustive Latency Tables (1,296 Policies Each)",
      ["Network", "Max speedup", "ρ_{P}", "ρ_{M}", "Sparser, slower"],
      [["SimpleCNN, C10", "2.61×", "0.32", "0.89", "39.0%"],
       ["SimpleCNN, C100", "2.59×", "0.31", "0.89", "39.4%"],
       ["ResNet-8, C10", "1.67×", "0.55", "0.98", "30.6%"],
       ["ResNet-8, C100", "1.72×", "0.54", "0.98", "30.8%"]],
      [76, 46, 34, 34, 50],
      "ρ_{P}, ρ_{M}: Spearman correlation of parameter and MAC reduction with latency gain. Last column: share of policy pairs in which the policy with "
      "the larger parameter reduction has the higher latency.")
fig(1, "Latency reduction of all 1,296 physically compacted policies of CIFAR-10 SimpleCNN and ResNet-8 against "
       "(a) physical parameter reduction and (b) MAC reduction (batch 1, one thread, i7-1260P).")
h2("Recovery and Proxy Validity")
p("Five epochs of recovery restore most of the accuracy lost to channel removal. ResNet-8 with 10% removed from every "
  "unit improves from 20.96% to 76.84% VAL-RL accuracy, and the conv-heavy SimpleCNN improves from 25.84% to 73.36%. "
  "The 1-epoch proxy was validated on 156 SimpleCNN policies (126 lightly pruned pilot policies and 30 random ones) and "
  "on 31 ResNet-8 policies (the dense policy and 30 random ones), each recovered for one and for five epochs "
  "(Fig. 2, Table III).")
p("The proxy underestimates the 5-epoch accuracy in every pair, by 1.29 percentage points on average for SimpleCNN "
  "and 5.68 points for ResNet-8. For ResNet-8 it preserves the ranking (ρ = 0.980) and was accepted. For "
  "SimpleCNN the overall ρ is 0.844, below the pre-registered 0.90, and the proxy was rejected. Among the lightly "
  "pruned pilot policies, whose 5-epoch accuracies lie within a few tenths of a point of each other, the 1-epoch "
  "ranking is unreliable (ρ = 0.77), although it is accurate across the random policies (ρ = 0.97). We "
  "therefore did not build a SimpleCNN accuracy table, and SimpleCNN contributes to this paper only through its "
  "latency evidence.")
table("Validation of the 1-Epoch Recovery Proxy",
      ["Metric", "SimpleCNN", "ResNet-8"],
      [["Policy pairs", "156", "31"],
       ["Spearman ρ (A_{1}, A_{5})", "0.844", "0.980"],
       ["Pearson r", "0.973", "0.990"],
       ["Mean bias A_{1} − A_{5} (pp)", "−1.29", "−5.68"],
       ["Top-10 overlap", "3", "10"],
       ["Decision (ρ ≥ 0.90)", "Rejected", "Accepted"]],
      [120, 60, 60],
      "A_{1}, A_{5}: VAL-RL accuracy after 1 and 5 recovery epochs (CIFAR-10).")
fig(2, "VAL-RL accuracy after the 1-epoch proxy recovery against the full 5-epoch recovery for the validation "
       "policies of both networks. The dashed line marks equality.")
h2("Latency Room under an Accuracy Constraint")
p("For ResNet-8, we computed the 1-epoch proxy for all 1,296 policies. The calibration sample of 48 policies, eight per "
  "bin of *q*_{1}, contained 14 policies with *q*_{5} ≥ 0.98 (Table IV). The rule gave τ = 0.960, with recall "
  "1.00 and precision 0.875 on the sample. In the full table, 52 policies satisfy *q*_{1} ≥ τ, but the "
  "fastest of them is only 1.098× faster than dense, and the fastest truly feasible calibration policy is "
  "1.089× faster. Speed-ups of 1.13–1.22× occur only among calibration policies whose 5-epoch retention "
  "is below 0.98 (Fig. 3). The pre-registered requirement of at least one feasible policy with a 1.10× speed-up "
  "is therefore not met, and the latency-aware RL search was not started.")
table("ResNet-8 Latency Room at 98% Retained Accuracy",
      ["Quantity", "Value"],
      [["Calibration policies", "48"],
       ["With q_{5} ≥ 0.98", "14"],
       ["Proxy threshold τ", "0.960"],
       ["Recall / precision at τ", "1.00 / 0.875"],
       ["Policies with q_{1} ≥ τ (of 1,296)", "52"],
       ["Latency of these policies", "0.702–0.770 ms"],
       ["Best speed-up (required ≥ 1.10×)", "1.098×"],
       ["Best truly feasible (calibration)", "1.089×"]],
      [150, 90])
fig(3, "ResNet-8 speed-up against retained accuracy: proxy q_{1} for all 1,296 policies (grey) and 5-epoch q_{5} for "
       "the 48 calibration policies (orange). No policy at q ≥ 0.98 reaches the 1.10× line.")

h1("Discussion")
h2("What the Reward Should Measure")
p("On this CPU, a pruning reward based on parameter count would steer an agent toward the fully connected layer of "
  "SimpleCNN, which holds 85% of the parameters but 5% of the MACs. Such policies save memory but not time. MAC "
  "reduction ranks policies almost as measured latency does (ρ up to 0.98), so a MAC-based reward is a sound, "
  "hardware-independent first approximation. Measured latency adds the fixed overhead and shape-specific effects that "
  "MACs miss. Because the action space is finite, an exhaustive latency table costs little (the four tables took about "
  "2.5 hours) and removes timing noise from the reward entirely.")
h2("Why ResNet-8 Leaves Little Room")
p("ResNet-8 is already compact, with 78,042 parameters. About 43% of its dense latency is fixed overhead (0.33 of "
  "0.77 ms), and its residual stages are fragile under channel removal. Under a 2% accuracy tolerance after five "
  "recovery epochs, only lightly pruned policies survive. Their MAC reductions, at most 18% among the proxy-feasible "
  "policies, translate into at most 9% less latency. The bottleneck is the accuracy constraint, not the search "
  "algorithm: an exhaustive search of the proxy table finds no proxy-feasible policy at 1.10×, although the proxy "
  "threshold kept every truly feasible calibration policy.")
h2("Implications for Hardware-Aware RL")
p("The latency room of a pruning problem can be measured before any agent is trained, and it should be. Here, an "
  "exhaustive latency table, an accuracy proxy validated against the full recovery, and a feasibility check showed "
  "that a latency-aware agent would compete with simpler objectives over a band of less than 10%. Larger networks, a "
  "looser accuracy tolerance, longer recovery, or batched inference (where configuration C reached a 2.76× "
  "speed-up in calibration) are the settings in which hardware-aware RL has room to matter.")
h2("Limitations")
p("All latencies come from one laptop CPU in one thread configuration under the default Windows power plan, and the "
  "absolute latency shifted by 16% between days, so only relative gains are interpreted. The networks are small and "
  "CIFAR-scale. All accuracies are validation accuracies with a single recovery seed; the test set was deliberately "
  "not used. The 0.98 tolerance and the 1.10× threshold are conventions fixed in advance, and other values would "
  "change the conclusion quantitatively.")

h1("Conclusion")
p("Exhaustive measurement of 5,184 physically compacted pruning policies shows that parameter sparsity is a poor proxy "
  "for CPU latency, while MAC count is a good one. It also shows that structured pruning of a compact ResNet-8 leaves "
  "less than 10% of latency to gain once 98% of the recovered accuracy must be kept. Hardware-aware pruning should "
  "therefore reward measured or MAC-based cost, and the latency headroom of a pruning problem should be established "
  "before an RL search is built on it.")

REFERENCES = [
    "S. Han, J. Pool, J. Tran, and W. J. Dally, “Learning both weights and connections for efficient neural "
    "networks,” in *Proc. Adv. Neural Inf. Process. Syst. (NeurIPS)*, 2015, pp. 1135–1143.",
    "H. Li, A. Kadav, I. Durdanovic, H. Samet, and H. P. Graf, “Pruning filters for efficient ConvNets,” in "
    "*Proc. Int. Conf. Learn. Represent. (ICLR)*, 2017.",
    "Z. Liu, M. Sun, T. Zhou, G. Huang, and T. Darrell, “Rethinking the value of network pruning,” in "
    "*Proc. Int. Conf. Learn. Represent. (ICLR)*, 2019.",
    "G. Fang, X. Ma, M. Song, M. B. Mi, and X. Wang, “DepGraph: Towards any structural pruning,” in *Proc. "
    "IEEE/CVF Conf. Comput. Vis. Pattern Recognit. (CVPR)*, 2023, pp. 16091–16101.",
    "Y. He, J. Lin, Z. Liu, H. Wang, L.-J. Li, and S. Han, “AMC: AutoML for model compression and acceleration on "
    "mobile devices,” in *Proc. Eur. Conf. Comput. Vis. (ECCV)*, 2018, pp. 784–800.",
    "J. Lee, S. Park, S. Mo, S. Ahn, and J. Shin, “Layer-adaptive sparsity for the magnitude-based pruning,” in "
    "*Proc. Int. Conf. Learn. Represent. (ICLR)*, 2021.",
    "U. Evci, T. Gale, J. Menick, P. S. Castro, and E. Elsen, “Rigging the lottery: Making all tickets "
    "winners,” in *Proc. Int. Conf. Mach. Learn. (ICML)*, 2020, pp. 2943–2952.",
    "T.-J. Yang, A. Howard, B. Chen, X. Zhang, A. Go, M. Sandler, V. Sze, and H. Adam, “NetAdapt: Platform-aware "
    "neural network adaptation for mobile applications,” in *Proc. Eur. Conf. Comput. Vis. (ECCV)*, 2018, "
    "pp. 285–300.",
    "H. Cai, L. Zhu, and S. Han, “ProxylessNAS: Direct neural architecture search on target task and "
    "hardware,” in *Proc. Int. Conf. Learn. Represent. (ICLR)*, 2019.",
    "B. Wu *et al.*, “FBNet: Hardware-aware efficient ConvNet design via differentiable neural architecture "
    "search,” in *Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit. (CVPR)*, 2019, pp. 10734–10742.",
    "D. Blalock, J. J. Gonzalez Ortiz, J. Frankle, and J. Guttag, “What is the state of neural network "
    "pruning?” in *Proc. Mach. Learn. Syst. (MLSys)*, 2020.",
    "T. Hoefler, D. Alistarh, T. Ben-Nun, N. Dryden, and A. Peste, “Sparsity in deep learning: Pruning and growth "
    "for efficient inference and training in neural networks,” *J. Mach. Learn. Res.*, vol. 22, no. 241, "
    "pp. 1–124, 2021.",
    "J. Schulman, F. Wolski, P. Dhariwal, A. Radford, and O. Klimov, “Proximal policy optimization "
    "algorithms,” arXiv:1707.06347, 2017.",
    "K. He, X. Zhang, S. Ren, and J. Sun, “Deep residual learning for image recognition,” in *Proc. IEEE "
    "Conf. Comput. Vis. Pattern Recognit. (CVPR)*, 2016, pp. 770–778.",
    "A. Krizhevsky, “Learning multiple layers of features from tiny images,” Univ. Toronto, Toronto, ON, "
    "Canada, Tech. Rep., 2009.",
    "A. Paszke *et al.*, “PyTorch: An imperative style, high-performance deep learning library,” in *Proc. "
    "Adv. Neural Inf. Process. Syst. (NeurIPS)*, 2019, pp. 8024–8035.",
]


# ------------------------------------------------------------------ XML helpers
def runs(text, base_rpr=""):
    """Markup -> runs. **bold**, *italic*, _{sub}, ^{sup}."""
    out = []
    pattern = re.compile(r"(\*\*.+?\*\*|\*.+?\*|_\{.+?\}|\^\{.+?\})")
    for tok in pattern.split(text):
        if not tok:
            continue
        rpr = base_rpr
        if tok.startswith("**"):
            tok, rpr = tok[2:-2], rpr + "<w:b/><w:bCs/>"
            out.append(_run(tok, rpr))
            continue
        if tok.startswith("*") and tok.endswith("*") and len(tok) > 2:
            inner = tok[1:-1]
            # italics may contain sub/superscripts
            for sub in re.split(r"(_\{.+?\}|\^\{.+?\})", inner):
                if not sub:
                    continue
                out.append(_scripted(sub, rpr + "<w:i/><w:iCs/>"))
            continue
        out.append(_scripted(tok, rpr))
    return "".join(out)


def _scripted(tok, rpr):
    if tok.startswith("_{"):
        return _run(tok[2:-1], rpr + '<w:vertAlign w:val="subscript"/>')
    if tok.startswith("^{"):
        return _run(tok[2:-1], rpr + '<w:vertAlign w:val="superscript"/>')
    return _run(tok, rpr)


def _run(t, rpr):
    rp = f"<w:rPr>{rpr}</w:rPr>" if rpr else ""
    return f'<w:r>{rp}<w:t xml:space="preserve">{escape(t)}</w:t></w:r>'


def para(style, text, ppr_extra=""):
    return f'<w:p><w:pPr><w:pStyle w:val="{style}"/>{ppr_extra}</w:pPr>{runs(text)}</w:p>'


TNR = '<w:rFonts w:ascii="Times New Roman" w:hAnsi="Times New Roman" w:cs="Times New Roman"/>'


def equation(text, n):
    """The template's equation style uses the Symbol font; every run is set to Times New Roman, as in its sample."""
    return (f'<w:p><w:pPr><w:pStyle w:val="equation"/></w:pPr><w:r><w:tab/></w:r>{runs(text, TNR)}'
            f'<w:r><w:tab/></w:r><w:r><w:rPr>{TNR}</w:rPr><w:t xml:space="preserve">({n})</w:t></w:r></w:p>')


def figure(n, caption):
    holder = (f'<w:p><w:pPr><w:keepNext/><w:spacing w:before="6pt" w:after="3pt"/><w:jc w:val="center"/></w:pPr>'
              f'<w:r><w:t>[[FIG{n}]]</w:t></w:r></w:p>')
    return holder + para("figurecaption", caption)


LEFT = '<w:jc w:val="start"/>'


def tbl(caption, header, rows, widths, foot):
    assert sum(widths) <= 243, "table wider than the column"
    tblpr = ('<w:tblPr><w:tblW w:w="0pt" w:type="dxa"/><w:jc w:val="center"/><w:tblBorders>'
             '<w:top w:val="single" w:sz="2" w:space="0" w:color="auto"/><w:start w:val="single" w:sz="2" w:space="0" w:color="auto"/>'
             '<w:bottom w:val="single" w:sz="2" w:space="0" w:color="auto"/><w:end w:val="single" w:sz="2" w:space="0" w:color="auto"/>'
             '<w:insideH w:val="single" w:sz="2" w:space="0" w:color="auto"/><w:insideV w:val="single" w:sz="2" w:space="0" w:color="auto"/>'
             '</w:tblBorders><w:tblLayout w:type="fixed"/><w:tblLook w:firstRow="0" w:lastRow="0" w:firstColumn="0" '
             'w:lastColumn="0" w:noHBand="0" w:noVBand="0"/></w:tblPr>')
    grid = "<w:tblGrid>" + "".join(f'<w:gridCol w:w="{int(w * 20)}"/>' for w in widths) + "</w:tblGrid>"

    def row(cells, style, head, keep=True):
        trpr = '<w:trPr><w:cantSplit/>' + ('<w:tblHeader/>' if head else '') + '<w:jc w:val="center"/></w:trPr>'
        tcs = "".join(f'<w:tc><w:tcPr><w:tcW w:w="{w}pt" w:type="dxa"/><w:vAlign w:val="center"/></w:tcPr>'
                      f'{para(style, c, ("<w:keepNext/>" if keep else "") + ("" if (head or i) else LEFT))}</w:tc>'
                      for i, (c, w) in enumerate(zip(cells, widths)))
        return f"<w:tr>{trpr}{tcs}</w:tr>"
    body = row(header, "tablecolhead", True) + "".join(row(r, "tablecopy", False, k < len(rows) - 1) for k, r in enumerate(rows))
    out = para("tablehead", caption, "<w:keepNext/>") + f"<w:tbl>{tblpr}{grid}{body}</w:tbl>"
    if foot:
        out += para("tablefootnote", foot, '<w:numPr><w:ilvl w:val="0"/><w:numId w:val="0"/></w:numPr><w:jc w:val="start"/>')
    return out


# ------------------------------------------------------------------ assemble
def build(unpacked, out_path):
    sys.path.insert(0, unpacked + "/..")
    from split import load
    head, els, tail = load()
    keep = {}
    # title: template paragraph properties, one run
    ppr0 = re.search(r"<w:pPr>.*?</w:pPr>", els[0]).group(0)
    title = f'<w:p>{ppr0}<w:r><w:rPr><w:kern w:val="48"/></w:rPr><w:t xml:space="preserve">{escape(TITLE)}</w:t></w:r></w:p>'
    # author block: one author, centred, in a single-column section (template sections kept)
    rpr = '<w:sz w:val="18"/><w:szCs w:val="18"/>'
    alines = []
    for i, line in enumerate(AUTHOR):
        it = line.startswith("*")
        t = line.strip("*")
        alines.append(f'<w:r><w:rPr>{"<w:i/>" if it else ""}{rpr}</w:rPr>{"<w:br/>" if i else ""}<w:t xml:space="preserve">{escape(t)}</w:t></w:r>')
    author = (f'<w:p><w:pPr><w:pStyle w:val="Author"/><w:spacing w:before="5pt" w:beforeAutospacing="1"/>'
              f'<w:rPr>{rpr}</w:rPr></w:pPr>{"".join(alines)}</w:p>')
    sect8 = els[8].replace('w:num="3"', 'w:num="1"')
    sect9 = re.sub(r'<w:r><w:br w:type="column"/></w:r>', "", els[9]).replace('w:num="3"', 'w:num="1"')
    # body
    body = [f'<w:p><w:pPr><w:pStyle w:val="Abstract"/></w:pPr>'
            f'<w:r><w:rPr><w:i/><w:iCs/></w:rPr><w:t>Abstract</w:t></w:r><w:r><w:t>—</w:t></w:r>{runs(ABSTRACT)}</w:p>',
            f'<w:p><w:pPr><w:pStyle w:val="Keywords"/></w:pPr><w:r><w:t>Keywords—</w:t></w:r>{runs(KEYWORDS)}</w:p>']
    for blk in B:
        kind = blk[0]
        if kind == "equation":
            body.append(equation(blk[1], blk[2]))
        elif kind == "figure":
            body.append(figure(blk[1], blk[2]))
        elif kind == "table":
            body.append(tbl(*blk[1:]))
        else:
            body.append(para(kind, blk[1]))
    body.append(para("Heading5", "References"))
    body += [para("references", r, '<w:ind w:start="17.70pt" w:hanging="17.70pt"/>') for r in REFERENCES]
    # the two-column section break of the template, without its red guidance text
    end2 = re.sub(r"<w:r\b.*?</w:r>", "", els[90], flags=re.S)
    last = "<w:p/>"                                         # replaces the template's floating figure-guidance text box
    new = [title, els[2], els[3], author, sect8, sect9] + body + [end2, last, els[92]]
    open(out_path, "w", encoding="utf-8").write(head + "".join(new) + tail)
    print(f"wrote {out_path}: {len(new)} top-level elements, {len(REFERENCES)} references, "
          f"{sum(1 for b in B if b[0] == 'figure')} figures, {sum(1 for b in B if b[0] == 'table')} tables")


if __name__ == "__main__":
    build(sys.argv[1], sys.argv[2])
