# Soft Sensitivity-Prior PPO — implementation checks

Run 2026-09-27T19:12:21+06:00 with `python evaluation/soft_prior.py check`, before any experimental run.
**Overall: ALL CHECKS PASS**


## 0. Penalty algebra: beta*S*a_norm is 0 at a=0, 0 at S=0, non-decreasing in S and in a — PASS

Checked on S in {0, 0.05, ..., 1} x all six actions for beta 0.5 and 1.0. a_norm = [0.0, 0.1667, 0.3333, 0.5, 0.6667, 1.0].

## 1. beta = 0 reproduces standard PPO — PASS

(a) P0 configuration (zero state, lambda_s 0.01, seed 42): SoftPriorPolicy with beta 0 vs SB3 MlpPolicy — all 512 training episodes identical: **True**; final policies [1, 1, 5, 5] vs [1, 1, 5, 5].

(b) SoftPriorPolicy with beta 0 on the ORIGINAL configuration (original sensitivity state, lambda_s 0.04, seed 42) reproduces the recorded original run `results/training_curves/sens-correct_coef-0.04_seed-42.csv`, all 512 episodes: **True**; final policy [1, 1, 5, 5] (recorded [1, 1, 5, 5]).

## 2. S = 0 (with beta 0.5) produces exactly the original logits — PASS

Two policies with identical weights, one with prior S = [0,0,0,0] and beta 0.5, one without: distribution logits bit-identical on the four layer observations and four perturbed ones: **True**.

## 3. All six actions remain possible for every layer under every prior — PASS

Initial policy (seed 7), layer observations at the start of an episode. The prior subtracts a finite amount from finite logits, so every softmax probability stays > 0.

| Definition | beta | Prior | Min probability over layers x actions | Max prob. change per layer (f.0, f.3, f.6, c.1) |
|---|---|---|---|---|
| loss | 0.5 | correct | 0.1249 | [0.042100001126527786, 0.001500000013038516, 0.0006000000284984708, 0.0] |
| loss | 0.5 | shuffled seed 42 | 0.1249 | [0.0, 0.0006000000284984708, 0.001500000013038516, 0.042100001126527786] |
| loss | 0.5 | constant | 0.1552 | [0.011800000444054604, 0.011800000444054604, 0.011800000444054604, 0.011800000444054604] |
| loss | 1.0 | correct | 0.0910 | [0.07970000058412552, 0.003000000026077032, 0.0012000000569969416, 0.0] |
| loss | 1.0 | shuffled seed 42 | 0.0911 | [0.0, 0.0012000000569969416, 0.003000000026077032, 0.07970000058412552] |
| loss | 1.0 | constant | 0.1439 | [0.023099999874830246, 0.023099999874830246, 0.023099999874830246, 0.023099999874830246] |
| accuracy | 0.5 | correct | 0.1249 | [0.042100001126527786, 0.002400000113993883, 0.00139999995008111, 0.0] |
| accuracy | 0.5 | shuffled seed 42 | 0.1249 | [0.0, 0.00139999995008111, 0.002400000113993883, 0.042100001126527786] |
| accuracy | 0.5 | constant | 0.1548 | [0.012199999764561653, 0.012199999764561653, 0.012199999764561653, 0.012199999764561653] |
| accuracy | 1.0 | correct | 0.0910 | [0.07970000058412552, 0.004699999932199717, 0.0027000000700354576, 0.0] |
| accuracy | 1.0 | shuffled seed 42 | 0.0911 | [0.0, 0.00279999990016222, 0.004699999932199717, 0.07970000058412552] |
| accuracy | 1.0 | constant | 0.1432 | [0.023800000548362732, 0.023800000548362732, 0.023800000548362732, 0.023800000548362732] |

## 4. The correct prior changes probabilities but never removes an action — PASS

Same table: the correct prior changes features.0's distribution (largest penalty) and leaves every probability > 0. On the other layers the change is tiny because S is ≤ 0.051 there.

## 5. Shuffled prior uses exactly the same sensitivity multiset, with every layer's value changed — PASS

All 20 seeds x both definitions checked. Examples: loss, seed 42: {'features.0': 'classifier.1', 'features.3': 'features.6', 'features.6': 'features.3', 'classifier.1': 'features.0'}; loss, seed 1: {'features.0': 'classifier.1', 'features.3': 'features.0', 'features.6': 'features.3', 'classifier.1': 'features.6'}; accuracy, seed 42: {'features.0': 'classifier.1', 'features.3': 'features.6', 'features.6': 'features.3', 'classifier.1': 'features.0'}; accuracy, seed 1: {'features.0': 'classifier.1', 'features.3': 'features.0', 'features.6': 'features.3', 'classifier.1': 'features.6'}.

## 6. Constant-prior control has the same average penalty as the correct prior — PASS

Mean of beta*S*a_norm over layers and actions: loss, beta 0.5: P1 0.058091, P3 0.058091; loss, beta 1.0: P1 0.116182, P3 0.116182; accuracy, beta 0.5: P1 0.060049, P3 0.060049; accuracy, beta 1.0: P1 0.120098, P3 0.120098.

## 7. No test-set call can occur during PPO training — PASS

Every run trains inside `Splits.training()`, where requesting the test loader raises `RuntimeError` (demonstrated: raised = **True**). The test set is used once, after `agent.learn()` returns, on a fresh baseline copy pruned by the frozen policy. The three check runs in item 1 trained under the same guard without error.
