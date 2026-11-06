# Confirmatory exploration study — implementation checks

Run 2026-09-28T09:15:03+06:00 with `python evaluation/exploration_confirm.py check`, after the pre-registration commit and before any fresh-seed run. Only seed 42 (non-experimental, used in earlier studies) and synthetic data are used; no fresh-seed outcome exists or is inspected.
**Overall: ALL CHECKS PASS**


## 1. beta = 0 reproduces E0 exactly (standard PPO) — PASS

E0 (SoftPriorPolicy, beta 0) vs SB3 MlpPolicy, seed 42: all 512 training episodes identical = **True**. E0 vs the recorded soft-prior P0 run (seed 42): identical = **True**.

## 2. The correct prior reproduces the previously implemented P1 mechanism — PASS

E1 prior matrix equals soft-prior P1's (beta 0.5, loss S) = **True**. E1 on seed 42 reproduces the recorded soft-prior P1 run, all 512 episodes = **True**.

## 3. Shuffled prior uses exactly the same sensitivity-value multiset — PASS

For all 30 locked seeds: the E2 vector is a permutation of S (sorted vectors equal), every layer's value differs from its correct value, and it matches the vector listed in the pre-registration.

## 4. Constant prior uses exactly the mean sensitivity — PASS

E3 vector = [0.26140902642330055, 0.26140902642330055, 0.26140902642330055, 0.26140902642330055]; mean(S) = np.float64(0.26140902642330055); mean penalty E3 = 0.058090894761, E1 = 0.058090894761.

## 5. All actions keep non-zero probability — PASS

Minimum action probability over the four layer observations, untrained policies built with the experimental seeds' priors (no training, no outcome): E0/seed 100: 0.1644, E0/seed 129: 0.1649, E1/seed 100: 0.1254, E1/seed 129: 0.1259, E2/seed 100: 0.1256, E2/seed 129: 0.1260, E3/seed 100: 0.1557, E3/seed 129: 0.1565.

## 6. Test set is inaccessible during training — PASS

Inside `Splits.training()` (wrapping every `agent.learn`) the test loader raises: **True**.

## 7. Evaluation checkpoints are identical across conditions — PASS

All four seed-42 check runs (E0–E3) have exactly 512 checkpoints at timesteps 4, 8, …, 2048; `trajectory()` asserts this for every run and fails otherwise.

## 8. Trajectory metrics are computed identically — PASS

A single function (`trajectory` → `run_metrics`) computes every endpoint for every condition. Synthetic checks: constant y → AUC = y; linear y = t → AUC = midpoint; triangle → 0.5. Shuffling a curve's rows gives identical metrics (rows are ordered by episode first).

## 9. AUC implementation is deterministic — PASS

Recomputing all endpoints for the four check runs gives bit-identical values.

## 10. Statistics use seeds, not episodes, as observations; exact tests are exact — PASS

`paired()` requires exactly one row per (condition, seed) and exactly the 30 locked seeds, and raises on episode-level or duplicated input (verified). The meet-in-the-middle sign-flip p equals brute-force enumeration for n = 6, 10, 14, 16 (20 random cases), and matches a 400,000-draw Monte Carlo estimate for n = 30 within 0.005. The dynamic-programming Wilcoxon p equals brute-force enumeration on the same cases.
