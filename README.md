# RL-Based Neural Network Pruning with Fine-Tuning

This project explores **Reinforcement Learning (RL)-based model compression** using PPO to learn **layer-wise pruning strategies**, combined with **fine-tuning** to recover accuracy.

The goal is to achieve: **Aggressive compression with minimal or no accuracy loss**

---

## Key Idea

Instead of manually pruning all layers equally, this project:

1. Trains a CNN on CIFAR-10  
2. Computes **layer sensitivity**  
3. Uses a **PPO agent** to decide pruning per layer  
4. Applies pruning sequentially  
5. Fine-tunes the pruned model  
6. Evaluates the accuracy–sparsity trade-off  

---

## Motivation

### Traditional pruning:
- Fixed pruning ratios (e.g., 20%, 40%, 60%)
- Treats all layers equally ❌

### Our approach:
- Learns **adaptive pruning policies**
- Preserves important layers
- Prunes less important layers more aggressively ✅

---

## Final Results

> **Superseded.** The table below came from a pipeline in which the RL reward,
> the layer-sensitivity probe and the in-episode fine-tuning all used CIFAR-10
> *test* images, and both training loops selected their best epoch on test
> accuracy. Those numbers are optimistically biased and should not be cited.
> Current, leakage-free results live in `results/` and are indexed in
> `results/experiment_log.md`.

| Method | Accuracy (%) | Sparsity (%) |
|--------|-------------|--------------|
| Baseline | 79.57 | 0 |
| Baseline + Fine-tune | 81.34 | 0 |
| PPO + Fine-tune | **81.33** | ~20–30 |

---

## Interpretation

- PPO + Fine-tune achieves **almost identical accuracy** to the fine-tuned baseline  
- While introducing **meaningful model compression**  
- Demonstrates an effective **accuracy–sparsity trade-off**

---

## Key Contributions

- RL-based **layer-wise pruning policy learning**
- Integration of **fine-tuning within the RL pipeline**
- Proper **fair comparison with fine-tuned baseline**
- Empirical validation that:
  > Not all layers are equally important

---

## Project Structure

```
├── notebooks/
│   ├── Main code.ipynb
│   └── Final fine tuned.ipynb
│
├── checkpoints/
│   ├── baseline_clean.pth
│   ├── baseline_finetuned.pth
│   └── ppo_agent/
│
├── results/
│   └── final_results.csv
│
└── README.md
```
## Model Details

- **Architecture:** Simple CNN (Conv2D + Linear layers)  
- **Dataset:** CIFAR-10  
- **RL Algorithm:** PPO (Stable-Baselines3)  

---

## RL Formulation

### State Representation

Each layer is represented using:

- Layer index  
- Parameter count  
- Weight statistics (mean, std)  
- Cumulative pruning  
- Sensitivity score  

---

### Action Space

Six discrete pruning levels (`ACTION_TO_PRUNE`):

```
[0%, 10%, 20%, 30%, 40%, 60%]
```

There is no 50% level; the action space has six entries, not seven.

---

### Reward Function

The reward balances:

- Accuracy preservation  
- Sparsity maximization  
- Action diversity  

---

## Observations

- Low pruning → negligible accuracy drop  
- Moderate pruning → optimal trade-off  
- High pruning (≥60%) → accuracy degradation  

---

## Key Learning Behavior

The RL agent learns to:

- Protect sensitive layers  
- Aggressively prune robust layers

## Key Insight

Not all layers contribute equally to model performance.  

Reinforcement Learning (RL) automatically discovers this behavior.

---
## Sustainability Impact

This project contributes to **sustainable computing** by improving the efficiency of deep learning models through adaptive pruning.

### Key Contributions

- **Reduced Energy Consumption**  
  Model pruning decreases the number of parameters and computations, which lowers CPU/GPU usage and overall power consumption.

- **Lower Carbon Footprint**  
  Efficient models require less energy during training and inference, helping reduce CO₂ emissions associated with large-scale AI systems.

- **Efficient Resource Utilization**  
  The reinforcement learning agent learns to prune only less important layers while preserving critical ones, avoiding unnecessary computation.

- **Edge Device Deployment**  
  Smaller and compressed models can run on low-power devices such as mobile and embedded systems, reducing reliance on high-energy servers.

---

## Limitations

- RL does not outperform fine-tuned baseline accuracy  
- Improvements come from compression, not raw accuracy gain  
- Model is relatively small (Simple CNN)  

---

## Enhancement Needed: 

- Apply to larger architectures (ResNet, ViT)  
- Use structured pruning (filter/channel pruning)  
- Introduce latency-aware reward  
- Deploy on real hardware for benchmarking  

---

## Conclusion

This work demonstrates that:

Reinforcement learning can learn efficient pruning policies that maintain near-baseline accuracy while reducing model size.
