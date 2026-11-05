# List of Abbreviations

Only abbreviations actually used in the Abstract and Chapters 1–8 are listed, in alphabetical order. Method, model
and dataset names are given in their published form. They are expanded only where the published title or abstract states the
expansion; no expansion has been invented.

## General abbreviations

| Abbreviation | Meaning |
|---|---|
| AC | Alternating current (mains power) |
| AI | Artificial intelligence |
| ARM | ARM processor architecture |
| AUC | Area under the curve (trajectory AUC of a search measure) |
| CI | Confidence interval |
| CNN | Convolutional neural network |
| CPU | Central processing unit |
| CSV | Comma-separated values |
| CV | Coefficient of variation |
| DRAM | Dynamic random-access memory |
| EDP | Energy–delay product |
| FLOP | Floating-point operation (FLOPs: floating-point operations) |
| GAE | Generalised advantage estimation |
| GPU | Graphics processing unit |
| HW | Hardware (table header, Chapter 2) |
| JSON | JavaScript Object Notation |
| LLM | Large language model |
| MAC | Multiply–accumulate operation (MACs: multiply–accumulate operations) |
| MKL | Math Kernel Library (Intel oneMKL) |
| MLP | Multilayer perceptron |
| NAS | Neural architecture search |
| NCHW | Tensor layout: batch (N), channels (C), height (H), width (W) |
| NLP | Natural language processing |
| OS | Operating system |
| PDH | Performance Data Helper (Windows performance-counter interface) |
| pp | Percentage point(s) |
| PP0 | RAPL power plane 0 (processor cores) |
| PPO | Proximal Policy Optimisation |
| RAPL | Running Average Power Limit (Intel energy-counter interface) |
| ReLU | Rectified linear unit |
| RGB | Red–green–blue colour channels |
| RL | Reinforcement learning |
| RQ | Research question |
| SB3 | Stable-Baselines3 |
| SD | Standard deviation |
| SHA-256 | Secure Hash Algorithm, 256-bit digest |
| SNR | Signal-to-noise ratio |
| TDP | Thermal design power |
| VAL-RL | Validation split used for sensitivity, reward and PPO training (Chapter 4) |
| VAL-SELECT | Validation split used only to select among candidates already produced (Chapter 4) |

## Method, model and dataset names

| Name | Published form or expansion |
|---|---|
| AMC | AutoML for Model Compression |
| BESA | Blockwise Parameter-Efficient Sparsity Allocation |
| CHEX | CHannel EXploration |
| CIFAR-10, CIFAR-100 | Image-classification datasets of the Canadian Institute for Advanced Research [10 and 100 classes] |
| DDPG | Deep Deterministic Policy Gradient |
| DECORE | Deep Compression with Reinforcement Learning |
| ERK | Erdős–Rényi-kernel layer-wise sparsity allocation |
| GNN | Graph neural network |
| HALP | Hardware-Aware Latency Pruning |
| HAP | Hessian-Aware Pruning |
| HAQ | Hardware-Aware Automated Quantization |
| LAMP | Layer-Adaptive Magnitude-based Pruning (score) |
| LeNet-5, ResNet-8, SimpleCNN | Network architectures (Chapter 4) |
| LSTM | Long short-term memory |
| MNIST, PASCAL VOC, ImageNet | Datasets named in the literature review |
| OPT, LLaMA, Vicuna, Mistral, ViT, MobileNet V1/V2 | Model families named in the literature review |
| OWL | Outlier Weighed Layerwise sparsity |
| PQ Index | Compressibility index named in the literature review |
| QLP | Deep Q-Learning for Pruning |
| SMCP | Soft Masking for Cost-Constrained Channel Pruning |
| SNIP | Single-shot Network Pruning |

## Experiment labels (thesis notation)

| Label | Meaning | Defined in |
|---|---|---|
| C10, C100 | CIFAR-10, CIFAR-100 (in tables) | Chapter 5 |
| C0, C1, C2 | Phase-5 conditions (R0 + centred prior; constrained reward + centred prior; constrained reward, no prior) | Chapter 4 §4.9.3 |
| C1–C6 (Chapters 1 and 8) | Thesis contributions 1–6 | Chapter 1 §1.7 |
| F, K | Phase-4 final pipeline (R0, P2, S2A) and control (R0, P0, S0) | Chapter 4 §4.9.2 |
| H0, H1, H2-A | Paper-3 stages: latency calibration, structured-pruning validation, exhaustive latency tables | Chapter 4 §4.9.5 |
| O1–O6 | Specific research objectives | Chapter 1 §1.6 |
| P0–P3 (Phase 3) | No prior; correct prior; shuffled prior; constant prior | Chapter 4 §4.9.1 |
| P0–P5 (Phase 4) | No prior; Phase-3 prior; centred prior; permuted centred; magnitude-matched conservative; sign-reversed | Chapter 4 §4.9.2 |
| R0, R1a, R1b, R2, R_c | Reward functions | Chapter 4 §4.12 |
| RQ1–RQ6 | Research questions | Chapter 1 §1.5 |
| S0, S1, S2, S2A | Phase-4 search variants (terminal; archive; elite cloning; elite cloning + archive) | Chapter 4 §4.9.2 |
| 7A, 7B, 7B-EC | Phase 7A, Phase 7B, Phase-7B energy confirmation study | Chapter 4 §4.9 |

**Label ambiguity, flagged.** "C1" denotes the Phase-5 *condition* in Chapters 4–7, but *Contribution 1* in Chapter 1
§1.7 and Chapter 8 §8.3. The two meanings never occur in the same section, and each chapter defines its use. At
template formatting, the contributions could be written out as "Contribution 1" to remove the ambiguity. This is a
presentation change only.
