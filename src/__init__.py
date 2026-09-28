"""Consolidated, architecture-independent library for the CPU-only rock-solid roadmap.

The historical scripts under evaluation/ stay frozen; this package re-implements
their shared pieces once, for any registered architecture and dataset, and
experiments/phase2/verify_library.py checks that it reproduces the historical
SimpleCNN results exactly before anything new is built on it.

Subpackages
  utils        seeding, hashing, provenance
  data         CIFAR-10/100 arrays, deterministic splits, stratified V_RL/V_SELECT,
               cached evaluation tensors, augmentation stream, phase guards
  models       SimpleCNN, LeNet-5 (CIFAR), ResNet-8, SmallVGG (pre-registered fallback)
  pruning      pruning units, L1 unit pruning, sparsity, one-shot baselines,
               structured channel removal (prepared, not used in Phase 2)
  sensitivity  accuracy-drop and multi-ratio loss sensitivity
  rl           generalised PPO environment, reward, soft-prior policy
  evaluation   accuracy, predictions, loss, dense training loop, efficiency
  statistics   exact paired tests, Holm, intervals, McNemar
"""
