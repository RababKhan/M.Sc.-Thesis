# Conceptual conclusion (draft, not the final polished text)

This thesis set out to test whether reinforcement learning can learn layer-wise adaptive pruning allocations, and
what is needed for such allocations to matter in deployment. A PPO agent learned useful non-uniform allocations that
strongly outperformed uniform and random allocation at the same sparsity. Layer-sensitivity guidance made its search
more efficient, but did not raise final accuracy. The analysis then uncovered two problems in RL pruning:

- **Retention.** PPO discovered top-ranked policies without retaining them. Archive selection fixed this on
  validation ranking.
- **Reward alignment.** With retention fixed, the accuracy–sparsity reward reliably drove ResNet-8 to over-prune. A
  validation-aware accuracy-constrained reward largely removed this, although validation feasibility did not
  guarantee test feasibility.

Throughout, LAMP remained a strong benchmark, more accurate than PPO before fine-tuning in nearly every setting.
Under identical fine-tuning, all methods recovered most of their loss and the PPO–LAMP difference fell within
±0.5 pp in every setting. PPO's advantage is therefore mainly an advantage before fine-tuning.

The deployment extension showed that compression metrics are not deployment efficiency. Parameter sparsity is a weak
proxy for CPU latency, whereas MACs and measured latency track the real cost. Physically structured realisation of
the learned allocation reduced directly measured CPU package energy (22.8% and 13.4%), but at an accuracy cost of 5.1
and 3.2 pp, and on SimpleCNN a uniform allocation was the better deployment choice. A PPO redesigned to learn
structured, energy-aware policies directly reduced parameter memory by 63.4% and MACs by 36.0%, with a much smaller
accuracy loss (1.12 pp). It still missed the strict 0.5 pp accuracy criterion, and its operational energy could not
be confirmed because direct energy measurement on the available hardware was not stable.

The overall message: adaptive RL pruning is useful, but search quality alone is not enough. Reward alignment, policy
retention, equal recovery, deployment-aware objectives and reliable hardware measurement are all needed before a
pruning result can be called a deployment-efficiency result.
