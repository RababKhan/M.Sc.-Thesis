# Final contributions

Each contribution is stated in the strength the evidence supports. Each comes with its evidence and a boundary that
the thesis text must not cross. Numbers are in `result_freeze.md`.

## C1. A sensitivity-aware PPO framework for layer-wise pruning allocation, evaluated against baselines under pre-registered rules

**Claim.** A PPO agent chooses one pruning ratio per layer unit, guided by layer sensitivity. Across 6
architecture–dataset settings, the learned non-uniform allocations strongly outperform uniform and random allocation
at the same sparsity. They partly outperform global magnitude, and they do not outperform LAMP.

**Evidence.** Phase 3 (480 runs): Uniform STRONG 5/6, up to +31.4 pp test; Random STRONG 6/6; Global MODERATE; ERK
LIMITED; LAMP NOT SUPPORTED, REVERSED.

**Boundary.** Not "a state-of-the-art pruning method", not "better than LAMP".

## C2. Evidence that sensitivity guidance improves exploration but not final accuracy

**Claim.** A centred layer-sensitivity prior makes PPO's search more efficient: better reward-rank AUC and fewer
destructive policies. This improvement does not translate into higher final accuracy. That dissociation is reported
as a finding in its own right.

**Evidence.**
- Phase 3: secondary exploration endpoints 6/6; primary AUC 3/6; final accuracy 0/6.
- Phase 4: P2 reward-rank AUC STRONG, but V_RL-accuracy AUC reversed.
- Phase 5: C1 vs C2 supported in 5/6 on exploration, with 0 of 24 tests opposite; test accuracy 0.05–0.16 pp lower.

**Boundary.** Not "sensitivity improves pruned-model accuracy".

## C3. A diagnosis of the policy-retention problem in RL pruning and archive-based selection as its remedy

**Claim.** PPO regularly discovers top-ranked policies during training but fails to end on them. A small archive of
the best policies seen, re-ranked on a held-out selection split, restores retention.

**Evidence.**
- Phase 4: the best-seen policy reaches the top 10 in 95–100% of runs, but the terminal policy does in 0% of
  settings.
- With the archive, 75–100% reach the top 10 and 100% reach the top 20.

**Boundary.** Retention is improved on validation ranking. On test the archive had mixed effects (ResNet-8 −5.66 and
−4.00 pp, because the archive also retained over-pruned policies under the misaligned R0 reward), and criterion D was
not met. Do not claim that the archive improves test accuracy.

## C4. Reward-alignment analysis and a validation-aware, accuracy-constrained reward

**Claim.** Better search exposed that a reward trading accuracy against sparsity is misaligned: it drives ResNet-8 to
over-prune. A constraint-first reward fixes this. The reward requires q ≥ 0.98 on VAL-RL; infeasible policies get a
penalty proportional to the shortfall. Under it, the accuracy drop fell from 7.32 to 1.18 pp (CIFAR-10 ResNet-8) and
from 4.05 to 1.16 pp (CIFAR-100 ResNet-8). The thesis also quantifies how far validation feasibility carries over to
test.

**Evidence.**
- Phase 4: criterion A not met; K → F sparsity 29.0% → 47.8% with a 7.86 pp drop.
- Phase 5: C1 − C0 = +6.14 and +2.89 pp; test feasibility 67.5%, against 99.2% on VAL-SELECT.

**Boundary.** Criteria A and E were only PARTIALLY MET: SimpleCNN got slightly worse, and feasibility on test is not
guaranteed.

## C5. A fair equal-recovery evaluation of pruning methods

**Claim.** Under identical fixed-mask 5-epoch fine-tuning, all methods recover most of their pruning loss and the
differences between methods become small. The PPO–LAMP gap falls inside ±0.5 pp in all 6 settings, with LAMP still
ahead in 3. PPO's earlier lead over global magnitude disappears. So PPO's practical advantage lies mainly in
accuracy before fine-tuning.

**Evidence.**
- Phase 6 (480 runs): criteria A–F MET; gap shrank significantly in 5/6; no regrowth (360/360); 6/6 repeats
  bit-identical.

**Boundary.** The equivalence is a CI-inside-margin rule, not a formal equivalence test. It covers 5 epochs, one
optimiser, and CIFAR-scale models only.

## C6. A deployment and sustainability analysis showing that compression metrics do not automatically become deployment efficiency

**Claim.** Four findings together show where compression does and does not reach deployment:

1. Parameter sparsity is a weak proxy for CPU latency (ρ 0.31–0.55; 31–39% of policy pairs sparser but slower),
   while MACs are a strong one (ρ 0.89–0.98) (Paper 3).
2. Physically structured realisation of a pruning allocation reduces directly measured CPU package energy: −22.8% for
   SimpleCNN and −13.4% for ResNet-8, consistent over 5 sessions. In this case the saving came with an accuracy loss
   of 5.08 / 3.24 pp. On SimpleCNN, MAC-matched Uniform pruning dominated (Phase 7A).
3. A PPO redesigned to learn structured policies directly reduced memory by 63.4% and MACs by 36.0%. Its accuracy loss
   was 1.12 pp, against 4.00 pp for the earlier structured transfer at similar MACs (both VAL-SELECT). It still
   missed the 0.5 pp criterion (0/20 seeds) (Phase 7B).
4. Direct energy measurement on a laptop CPU was not stable enough to confirm Phase 7B's energy (CV 4.01% and
   19.92%, against a 3% limit). This is reported as a measurement limitation, not hidden (Phase 7B energy
   confirmation).

**Boundary.** No claim of "green", "environmentally friendly" or carbon reduction. No energy or latency claim for
Phase 7B. No accuracy-preserving claim for Phase 7A or 7B. No claim that PPO is more energy-efficient than Uniform.

## Cross-cutting methodological contribution (state briefly, not as a separate numbered claim)

Every confirmatory phase was pre-registered before its data existed. That covers frozen thresholds, seeds and
decision rules, a leakage-guarded split protocol, hashed models, and integrity checks. Negative and partial verdicts
are reported as they fell. This is what makes C1–C6 credible, and it is part of why the thesis is analytical rather
than a leaderboard result.
