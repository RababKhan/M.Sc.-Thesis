# Phase 5 tau selection

Rule (committed in `phase5_design_commitments.md` before the diagnostic): the largest tau in [0.95, 0.97, 0.98, 0.99] for which every setting has >= 20 feasible policies (q = VAL-RL accuracy / dense VAL-RL accuracy >= tau) and >= 1 feasible policy with total sparsity >= 10%.

| tau | meets the rule in all six settings |
|---|---|
| 0.95 | yes |
| 0.97 | yes |
| 0.98 | yes |
| 0.99 | no |

**Chosen tau = 0.98.**

Per setting at the chosen tau:

| Setting | Feasible policies | Max feasible sparsity | Median feasible sparsity | Oracle policy | Oracle sparsity | Oracle q |
|---|---|---|---|---|---|---|
| cifar10_simplecnn | 515 | 59.09% | 25.26% | [1, 4, 5, 5] | 59.09% | 0.9906 |
| cifar10_lenet5 | 395 | 51.91% | 18.95% | [0, 1, 5, 3] | 51.91% | 0.9807 |
| cifar10_resnet8 | 46 | 25.34% | 10.49% | [1, 2, 1, 3] | 25.34% | 0.9812 |
| cifar100_simplecnn | 254 | 54.39% | 24.62% | [1, 3, 4, 5] | 54.39% | 0.9845 |
| cifar100_lenet5 | 279 | 34.61% | 16.75% | [1, 3, 4, 4] | 34.61% | 0.9817 |
| cifar100_resnet8 | 21 | 16.08% | 7.50% | [0, 1, 1, 2] | 16.08% | 0.9890 |
