# RL-Based Neural Network Pruning with Fine-Tuning

This project tests whether a **reinforcement learning (PPO) agent** can learn
**layer-wise pruning ratios** for a CNN on CIFAR-10, and whether the policy it
learns does better than simple pruning baselines **at the same sparsity**.

Two settings are studied:

1. **Pruning only.** The agent picks a pruning ratio per layer and is rewarded on
   the pruned model's accuracy and sparsity.
2. **Pruning + fine-tuning.** Every pruned model is fine-tuned before it is
   scored, both inside the RL loop and in the final evaluation.

Every number in this README comes from the leakage-free pipeline described under
[Data protocol](#data-protocol). Provenance for each result file is in
[`results/experiment_log.md`](results/experiment_log.md).

---

## Summary of findings

- **Allocation matters a lot.** At about 58% total sparsity, pruning every layer by
  the same ratio drops test accuracy to about 41%. Allocations that spare the
  small convolutional layers keep it at about 76%.
- **PPO learns a good allocation, but not a better one than global magnitude
  pruning.** At matched sparsity, 2 of 4 PPO seeds are statistically
  indistinguishable from global magnitude pruning and 2 are significantly worse
  (−1.20 pp, McNemar p = 4.4e-5). PPO clearly beats uniform per-layer pruning
  (+34 to +36 pp, all seeds).
- **With fine-tuning, the learned policy was a uniform one.** The single PPO run
  in the fine-tuning setting chose `[4,4,4,4]`, 40% on every layer, which is
  identical to one of the fixed baselines. That run shows no layer-wise
  adaptation.
- **Fine-tuning recovers most of the pruning damage on its own.** Uniform 60%
  pruning goes from 40.48% test accuracy without fine-tuning to 78.62% with it.
- **The sparsity does not make the model faster, and does not shrink it without
  compression.** The raw checkpoint size is unchanged and CPU inference latency
  shows no measurable difference. The size reduction appears only when the
  checkpoint is compressed (−49% gzipped vs gzipped).

---

## Approach

1. Train a baseline CNN on CIFAR-10.
2. Measure each layer's **sensitivity**: the validation-accuracy drop when that
   layer alone is pruned by 10%.
3. A **PPO agent** visits the prunable layers in order and picks a pruning ratio
   for each.
4. Apply L1 unstructured pruning layer by layer.
5. Optionally fine-tune the pruned model.
6. Compare against uniform per-layer and global magnitude pruning **at matched
   total sparsity**, using paired McNemar tests on the test set.

---

## Data protocol

| Split | Size | Source | Used for |
|---|---|---|---|
| train | 45,000 | CIFAR-10 train, seeded permutation (seed 42) | all weight updates |
| validation | 5,000 | CIFAR-10 train, held out | epoch selection, layer sensitivity |
| RL probe | 1,000 | first 1,000 of the validation split | PPO reward during training |
| test | 10,000 | CIFAR-10 test | final reported numbers only |

The PPO training environments are built with no access to the test set. Both
notebooks build this split the same way and use the **same baseline model**
(`checkpoints/cnn_baseline_FIXED.pth`). The fine-tuning notebook asserts that it
reproduces that model's recorded accuracy before doing anything else.

---

## Results

### Baseline

SimpleCNN, 10 epochs, Adam (lr 1e-3), seed 42: **77.32% validation, 77.03% test.**

Prunable weights: 619,872. The agent controls the first four layers; the output
layer is excluded.

| Layer | Params | Share of prunable weights |
|---|---|---|
| `features.0` (conv) | 864 | 0.14% |
| `features.3` (conv) | 18,432 | 2.97% |
| `features.6` (conv) | 73,728 | 11.89% |
| `classifier.1` (linear) | 524,288 | **84.58%** |
| `classifier.3` (output, not pruned by the agent) | 2,560 | 0.41% |

Total sparsity is dominated by `classifier.1`, so per-layer ratios are reported
alongside it.

### Setting 1: pruning only (4 PPO seeds)

| Seed | Policy | Per-layer pruning | Sparsity % | Test acc % |
|---|---|---|---|---|
| 42 | `[1,1,5,5]` | 10 / 10 / 60 / 60 | 58.20 | 76.49 |
| 1 | `[0,0,5,5]` | 0 / 0 / 60 / 60 | 57.88 | 76.65 |
| 2 | `[0,5,5,5]` | 0 / 60 / 60 / 60 | 59.67 | 75.16 |
| 3 | `[0,5,5,5]` | 0 / 60 / 60 / 60 | 59.67 | 75.16 |

Mean **75.87 ± 0.82%** test accuracy at 58.9% mean sparsity (sample SD, n = 4).
Seeds 2 and 3 converged to the same policy.

**Matched-sparsity comparison.** Each baseline is calibrated to the same total
sparsity as that seed's policy, over the same four layers, so only the
allocation differs. McNemar exact test on the same 10,000 test images.

| Seed | Sparsity % | PPO % | Global magnitude % | Δ vs global | p | Uniform per-layer % | Δ vs uniform |
|---|---|---|---|---|---|---|---|
| 42 | 58.20 | 76.49 | 76.35 | +0.14 | 0.598 | 41.20 | +35.29 |
| 1 | 57.88 | 76.65 | 76.54 | +0.11 | 0.686 | 40.69 | +35.96 |
| 2 | 59.67 | 75.16 | 76.36 | **−1.20** | **4.4e-5** | 40.68 | +34.48 |
| 3 | 59.67 | 75.16 | 76.36 | **−1.20** | **4.4e-5** | 40.68 | +34.48 |

Every comparison against uniform per-layer pruning is significant
(p < 1e-300).

**Per-layer allocation at seed 42's sparsity (58.20%):**

| Layer | PPO | Global magnitude | Uniform |
|---|---|---|---|
| `features.0` | 10.0% | 9.8% | 58.4% |
| `features.3` | 10.0% | 28.7% | 58.4% |
| `features.6` | 60.0% | 32.0% | 58.4% |
| `classifier.1` | 60.0% | 63.3% | 58.4% |

Both PPO and global magnitude pruning leave the first convolution almost intact.
The policies that also prune `features.3` at 60% (seeds 2 and 3) are the ones
that fall significantly behind global magnitude pruning.

### Setting 2: pruning + fine-tuning (1 PPO seed)

Every pruned model is fine-tuned for 3 epochs (Adam, lr 1e-4) on the train split.
The best epoch is chosen on validation, and the model is then measured once on
test.

| Method | Policy | Test acc % | Sparsity % |
|---|---|---|---|
| Baseline | – | 77.03 | 0.00 |
| Baseline + fine-tune | – | 79.91 | 0.00 |
| Fine-tuned uniform 10% | `[1,1,1,1]` | 79.58 | 9.96 |
| Fine-tuned uniform 20% | `[2,2,2,2]` | 79.75 | 19.92 |
| Fine-tuned uniform 40% | `[4,4,4,4]` | 79.60 | 39.83 |
| Fine-tuned uniform 60% | `[5,5,5,5]` | 78.62 | 59.75 |
| **PPO + fine-tune** | `[4,4,4,4]` | 79.65 | 39.83 |

PPO's policy is identical to fine-tuned uniform 40%. The 0.05 pp gap between
the two rows comes only from randomness in fine-tuning, so it is not a method
difference. This is a single seed; more seeds are needed before drawing
conclusions about this setting.

### Storage and inference cost (seed-42 policy, 58.2% sparsity)

| | Dense | Pruned | Change |
|---|---|---|---|
| `.pth` checkpoint | 2427.1 KB | 2427.1 KB | 0.0% |
| `.pth`, gzipped | 2253.4 KB | 1142.2 KB | **−49.3%** |
| CPU latency, batch 128, median | 75.89 ms | 75.27 ms | no measurable difference |

Unstructured pruning stores the zeros, so the raw checkpoint does not shrink,
and dense kernels do the same work whether or not weights are zero. At this
density a sparse CSR kernel on `classifier.1` was **2.32× slower** than the
dense one.

### Superseded results

An earlier version of this README reported 81.33% for PPO + fine-tune. That
pipeline used CIFAR-10 test images for the RL reward, the sensitivity probe and
in-episode fine-tuning, and selected epochs on test accuracy. Those numbers are
optimistically biased and should not be cited. The files that hold them are
listed as superseded in the experiment log.

---

## RL formulation

### State (7 features per layer)

- normalised layer index
- log parameter count
- parameter count relative to the largest layer
- mean absolute weight
- weight standard deviation
- cumulative pruning so far in the episode
- layer sensitivity (validation-accuracy drop at 10% pruning)

### Action space

Six discrete pruning levels (`ACTION_TO_PRUNE`), applied as L1 unstructured
pruning:

```
[0%, 10%, 20%, 30%, 40%, 60%]
```

There is no 50% level.

### Reward

Pruning only (`Main code.ipynb`):

```
R = 1.5 * (accuracy / baseline_accuracy) + 0.04 * sparsity_percent
      + 0.05 * (number of distinct actions)
```

Pruning + fine-tuning (`Final fine tuned.ipynb`), where
`gain = accuracy − baseline_accuracy`:

```
R = 0.04 * sparsity_percent + 0.05 * (number of distinct actions)
      + (5.0 + gain   if gain >= 0
         2.0 * gain   otherwise)
```

The distinct-actions bonus differs between policies, so reward values are not
used to compare methods; accuracy at matched sparsity is.

### PPO

Stable-Baselines3 `MlpPolicy`.

| | Pruning only | Pruning + fine-tuning |
|---|---|---|
| Timesteps | 2,000 | 300 |
| `n_steps` / `batch_size` / `n_epochs` | 64 / 32 / 10 | 32 / 16 / 5 |
| Learning rate | 3e-4 | 3e-4 |
| Entropy coefficient | 0.01 | 0.02 |
| In-episode fine-tuning | none | 1 epoch on 5,000 train images |

---

## Model

- **Architecture:** SimpleCNN with three conv blocks (32/64/128 channels, 3×3,
  ReLU, max-pool) and a classifier `Linear(2048, 256) → ReLU → Linear(256, 10)`
- **Dataset:** CIFAR-10 with random crop and horizontal flip for training
- **RL algorithm:** PPO (Stable-Baselines3)
- **Hardware:** CPU only

---

## Project structure

```
├── notebooks/
│   ├── Main code.ipynb             # baseline training; PPO pruning only, 4 seeds
│   └── Final fine tuned.ipynb      # PPO with fine-tuning; fine-tuned baselines
│
├── evaluation/
│   ├── common.py                   # shared model, pruning and McNemar code
│   ├── matched_sparsity_analysis.py
│   └── storage_and_latency.py
│
├── checkpoints/
│   ├── cnn_baseline_FIXED.pth      # the baseline every current result uses
│   ├── baseline_finetuned.pth      # that baseline after 3 epochs of fine-tuning
│   ├── ppo_final_2000.zip          # PPO agent, pruning only, seed 42
│   └── ppo_finetune_agent_holdout.zip  # PPO agent, pruning + fine-tuning
│
├── results/
│   ├── experiment_log.md           # every result file, its source, and its status
│   └── *.csv
│
├── data/                           # CIFAR-10 (python version)
└── requirements.txt                # pinned environment
```

Other files in `checkpoints/` and `results/` come from superseded runs; the
experiment log lists which ones.

---

## Reproducing

[`requirements.txt`](requirements.txt) pins the versions every current result was
produced with (CPU builds of torch 2.7.1 and torchvision 0.22.1). Python 3.13
and 3.11 both reproduce the baseline exactly.

```
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

It includes `ipykernel` for running the notebooks from VS Code. Add
`pip install jupyter` if you want the Jupyter interface.

The notebooks are run from `notebooks/` and read data from `../data`. The
scripts are run from the repository root:

```
python evaluation/matched_sparsity_analysis.py   # matched baselines + McNemar tests
python evaluation/storage_and_latency.py         # storage + latency (run on an idle machine)
```

`matched_sparsity_analysis.py` checks that it reproduces the notebook's recorded
seed-42 result exactly before computing anything new.

**Re-running `Main code.ipynb` retrains and overwrites the baseline checkpoint.**
Every result above is tied to the current `cnn_baseline_FIXED.pth`, so retraining
it invalidates all of them, including the significance tests.

---

## Sustainability

Model compression is often motivated by lower energy use and by deploying models
on low-power devices. This project measured what unstructured pruning actually
delivers on that front:

- **Storage:** real, but only with compression (−49% gzipped vs gzipped). A
  bitmask-plus-values format would give about −55% compared with dense float32.
- **Compute and energy:** no measurable latency change on CPU, so no energy
  saving at inference time can be claimed from these experiments.

Turning sparsity into compute savings needs **structured pruning** (removing
whole filters or channels) or hardware and kernels that exploit sparsity at much
higher densities than 58%.

---

## Limitations

- PPO does not beat global magnitude pruning at matched sparsity, a baseline that
  needs no training.
- The pruning + fine-tuning setting has a single PPO seed, and that seed learned
  a uniform policy.
- Unstructured sparsity brings no latency benefit here.
- One small model (SimpleCNN) on one dataset.
- The two settings use different reward functions and PPO budgets, so their PPO
  results are not directly comparable with each other.

---

## Future work

- More PPO seeds in the fine-tuning setting
- Ablation of the sensitivity feature (with / zeroed / shuffled)
- Sweep of the reward's sparsity coefficient
- Structured (filter/channel) pruning, where sparsity does reduce compute
- A latency-aware reward, measured on real hardware
- Larger architectures (ResNet, ViT)

---

## Conclusion

How sparsity is spread across layers matters a great deal: at about 58% total
sparsity, the allocation alone separates about 41% from about 76% test accuracy.
A PPO agent learns allocations in the good regime without being told which
layers are sensitive. However, it does not beat global magnitude pruning, which
reaches the same result without any training. Once fine-tuning is added,
fine-tuning accounts for most of the accuracy recovery, and the one PPO run in
that setting settled on a uniform policy.
