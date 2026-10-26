# Phase 7B design: accuracy-preserving sustainable structured pruning with sensitivity-aware PPO

**Status.** Stage-A design. Written and committed before any Phase-7B model was built, trained, recovered
or measured. It fixes:

- the development procedures and acceptance criteria (proxy, surrogate, sample designs);
- the confirmatory frame (seeds, selection, baselines, GREEN rule).

The confirmatory pre-registration, `phase7b_final_preregistration.json`, will freeze the remaining tuned
quantities: the proxy budget, τ_E, the surrogate model and α. Stage A may tune only those, and only by the
rules written here.

Phases 1–7A, Paper 3 and their result folders are **read only**. The Phase-7A negative result stays
unchanged and motivates this phase.

## 1. Question and the change from Phase 7A

**Question.** Can a sensitivity-aware PPO agent learn a physically structured pruning policy that reduces
parameter memory, MACs and measured CPU inference energy while preserving accuracy?

**What changes from Phase 7A.** Phase 7A transferred an unstructured PPO allocation into structured
pruning. It saved energy but lost 5.1 pp (SimpleCNN) and 3.2 pp (ResNet-8). Here PPO chooses
**structured channel-removal actions directly**:

- every action vector is a physically compacted model;
- its reward uses a validated structured-recovery accuracy proxy and a validated hardware-energy surrogate.

There is no post-hoc conversion.

**Order.** CIFAR-10 SimpleCNN is completed and frozen first. ResNet-8 follows only afterwards, under the
same rules and its own Stage-A validation.

## 2. Structured space and backend (unchanged, validated in Paper 3)

- **Units.** SimpleCNN uses conv1, conv2, conv3 and fc1 hidden units. ResNet-8 uses the coupled stage-1
  residual width and the three block-internal widths.
- **Actions.** r ∈ {0, 0.1, 0.2, 0.3, 0.4, 0.6} per unit, so 6⁴ = 1,296 policies.
- **Channels kept.** k = max(1, round((1 − r)C)), selected by L1 rank on the dense model
  (`src/pruning/structured_units.py`).
- **Counts.** MACs and parameters come from the Paper-3 closed forms, which were verified against the
  hooks and FlopCounter.
- **Integrity checks before any measurement.** No masks; parameters and MACs equal the analytic values;
  slimmed and masked-equivalent logits agree within 1e-4; identical outputs after reload.

## 3. Accuracy

- **Splits.** VAL-RL (2,500 images) is the **search split**: proxy, landscape, reward and archive.
  VAL-SELECT (2,500 images) is **held out from all search and selection**.
- **Final GREEN accuracy criterion.**
  - The 5-epoch recovered **VAL-SELECT** accuracy must lie within 0.5 pp of the 5-epoch recovered dense
    model's VAL-SELECT accuracy.
  - VAL-RL is reported as secondary.
  - Using the held-out split avoids the winner's-curse bias of selecting and evaluating on the same images.
- **Test set.** Not used.
- **Recovery protocol.** The frozen Phase-6 / Paper-3 / Phase-7A protocol: Adam with lr 1e-4, cosine to
  0 stepped per epoch, batch 128, crop + flip, weight decay 0, seed 600, data-generator seed 600 + 10⁶,
  one thread, the model after the final epoch.
- **Final recovery.** E = 5 epochs for every method, Dense included.

## 4. Accuracy proxy (Stage A, validated before any PPO search)

The Paper-3 one-epoch SimpleCNN proxy failed (ρ = 0.844), so it is **not** reused.

**Proxy.** An E-epoch recovery, E ∈ {2, 3}: the same protocol with the cosine schedule over E epochs. It
is a separate run, not a truncation of the 5-epoch run. The proxy retained accuracy is
q_E = A_E / A_E,dense, both on VAL-RL.

**Target.** A policy is 5-epoch feasible if A_5 ≥ A_5,dense − 0.5 pp on VAL-RL.

**Validation sample.** Fixed now: dense plus 48 policies drawn without replacement with seed 20261027,
stratified by MAC reduction:

| MAC-reduction bin | [0, 5%) | [5, 15%) | [15, 25%) | [25, 40%) | [40, 60%) | [60, 100%] |
|---|---|---|---|---|---|---|
| Policies | 10 | 10 | 8 | 8 | 6 | 6 |

The low bins are over-sampled because the ±0.5 pp boundary lies there. Each policy is recovered for 2, 3
and 5 epochs.

**Threshold.** τ_E = (the largest q_E that keeps every 5-epoch-feasible sample policy) − 0.005.

**Acceptance rule for proxy E.** All three must hold:

1. Spearman ρ(A_E, A_5) ≥ 0.90 over the 49 sample policies.
2. Recall = 1.00 for the 5-epoch-feasible sample policies at τ_E. This holds by construction, so the
   precision condition below gives it content.
3. Precision ≥ 0.50 at τ_E.

The validation also needs at least 5 feasible sample policies. Otherwise the proxy cannot be validated,
and the phase stops.

**Order.** The 2-epoch proxy is evaluated first. The 3-epoch proxy is used only if the 2-epoch proxy
fails. If both fail, **STOP**: no PPO search.

Both budgets are run on the sample at the same time for scheduling. The decision rule is fixed above.

## 5. Memory

- **Primary.** Parameter memory = Σ numel × element_size over all parameters (float32), in bytes.
- **Also reported.** The trainable parameter count and the serialized `state_dict` size (`torch.save`,
  bytes).
- MemoryReduction = 1 − M / M_dense.
- Peak RSS is not used, because it is dominated by the framework and is not reliably attributable to the
  model.

## 6. Computation

- **Primary.** MACs from the verified counter, with MACReduction = 1 − MAC / MAC_dense.
- **Also reported.** FLOPs = 2 × MACs, and measured latency.

## 7. Energy

**Backend (Phase 7A, unchanged).** Intel RAPL package energy read through the Windows Energy Meter
Interface (PDH raw values, pWh units). No TDP or software estimate is used anywhere in this phase.

**Calibration set (surrogate dataset).** Fixed now, seed 20261028; 200 distinct policies in total:

- **Named policies (7).** Dense, plus the 6 Paper-3 H0 calibration policies (light, moderate,
  aggressive, fc-heavy, conv-heavy).
- **Stratified random policies (160).** 20 per MAC-reduction bin, over 8 bins:
  [0, 5), [5, 10), [10, 20), [20, 30), [30, 40), [40, 50), [50, 65), [65, 100]%.
- **Similar-MAC pairs (to 200).** 16 anchor policies are drawn at random. Each is paired with the policy
  whose MACs differ by at most 2% and whose width vector is farthest away (L1 on normalized widths). More
  anchors are added, with policies already chosen excluded, until the set holds 200 distinct policies.

**Calibration measurement.**

- 3 fresh-process sessions, all on AC with the Balanced plan, one thread, pinned to logical CPU 2.
- Policy order is randomized per session.
- Each policy is built from the dense reference (weight values do not affect dense CPU work) and run
  through 200 warm-up inferences, then one block of N = 20,000 batch-1 inferences.
- A dense anchor block runs after every 25 policies. A session whose anchors drift by more than 10% is
  repeated.
- The target is the median over the 3 sessions of E_1000 (J per 1,000 inferences).
- A dense-only pilot comes first: 2 processes × 3 blocks of N = 20,000. If the coefficient of variation
  exceeds 4%, N doubles once; if it is still above 4%, STOP.

**Surrogate.** Candidate models, tried in this order; the first that passes is used:

1. E_1000 ~ a + b·MACs
2. E_1000 ~ a + b·L
3. E_1000 ~ a + b·MACs + c·params + d·L
4. Degree-2 polynomial in the normalized widths, plus MACs, params and L (ridge, λ = 1e-3)

Here L is the measured Paper-3 batch-1 latency table (`h2_latency_table_simplecnn_c10.csv`), so every
one of the 1,296 policies has a measured L.

**Validation.** 5-fold cross-validation (seed 20261029), on out-of-fold predictions. All three must hold:

- Spearman ρ ≥ 0.95;
- median absolute percentage error ≤ 5%;
- median absolute percentage error ≤ 5% within the lowest-energy 25% of the calibration policies.

If no candidate passes, **STOP**: no PPO search.

## 8. Proxy landscape (cache)

After the proxy is accepted, all 1,296 policies receive the E-epoch recovery once. Proxy accuracy is
deterministic per policy, so this cache gives exactly the reward that on-line evaluation would. Phases 3–5
used landscape caches the same way.

The cache also defines an **enumeration reference**: the lowest-predicted-energy proxy-feasible policy
over the whole space. It shows the best any search could do under the same proxy. It is reported, but is
not a competing method.

## 9. Structured sensitivity-aware PPO

**Environment.** One decision per unit, in unit order (4 steps). Discrete(6) actions over the ratios
above. The reward is paid at the end of the episode.

**State (8 features for the current unit u).**

1. u / (n − 1)
2. log1p(C_u), the unit's channel count
3. the unit's share of dense parameters
4. the unit's share of dense MACs
5. centred structured sensitivity S̃_u
6. the ratio chosen for the previous unit (0 for u = 0)
7. the mean ratio chosen so far
8. the predicted energy of the partial policy, Ê(chosen so far, 0 for the rest) / Ê_dense

**Sensitivity.**

- **Definition.** `structured_loss_sensitivity` (Paper 3, unit-tested) on the dense reference, using all
  2,500 VAL-RL images and ratios {0.1, 0.2, 0.4, 0.6}, min-max normalized. The centred value is
  S̃ = S − mean(S).
- **Uses.** S̃ appears in the state and in the Phase-4 centred soft prior: the logit of action a for unit
  u is shifted by −β · S̃_u · r(a)/0.6, with β = 1.0 (`src.rl.SoftPriorPolicy`).
- **Not a metric.** Sensitivity is never used as a sustainability measure.

**Reward (constraint first).** With q = A_E / A_E,dense on VAL-RL:

- if q ≥ τ_E (feasible): R = 1 + α · ŝ / ŝ_max + 0.01 · q;
- otherwise: R = −10 · (τ_E − q).

Here ŝ = 1 − Ê / Ê_dense is the predicted energy saving, and ŝ_max is the largest ŝ over the 1,296
policies. Infeasible rewards are ≤ 0 and feasible rewards are ≥ 1, so energy can never compensate for an
accuracy violation.

**PPO.**

- The thesis settings: SB3 PPO with the `src.rl.PPO_KWARGS` hyper-parameters and 512 episodes
  (2,048 steps), one thread.
- The archive is recorded during training.

**Feasible archive.** Every distinct proxy-feasible policy visited during training, recorded with:

- the action vector;
- proxy accuracy and q;
- predicted energy;
- parameter memory and MACs;
- measured-table latency;
- reward;
- the first episode in which it was visited.

**Selection per seed.**

1. The lowest predicted energy among archive policies.
2. Policies within 1% of that energy count as tied. Ties go to higher proxy accuracy, then lower MACs,
   then lower parameter memory, then the smaller action vector.
3. A seed whose archive is empty has no selected policy and is reported as a failure.

**Primary new-PPO model.** The most frequent selected policy over the confirmatory seeds. Ties go to the
lower predicted energy.

## 10. Stage A (development) and Stage B (confirmatory)

**Stage A.**

- Implementation and debugging.
- Proxy and surrogate validation (sections 4 and 7).
- The landscape cache.
- **Tuning α ∈ {0.5, 1, 2, 4}** on development seeds 900–904:
  - choose the α whose seeds' selected policies have the lowest median predicted energy;
  - ties go to the smaller α.
- β = 1.0, κ = 10, ε = 0.01, the tie band and the selection rule are fixed here and are not tuned.

**Stage B.** 20 confirmatory seeds, 700–719, under the frozen `phase7b_final_preregistration.json`. No
redesign afterwards. All seeds are reported.

## 11. Final comparison (Stage B)

**Final models.** Five 5-epoch recovered methods, one row each in the final table, plus a pool of
per-seed models measured alongside them:

| Final-table method | What it is |
|---|---|
| Dense | The dense reference. |
| New Energy-Constrained Structured PPO | The primary new-PPO model (section 9). |
| Uniform-Structured | One common ratio for all units, on a 0.001 grid, minimizing \|MACs − MACs_PPO\| against the primary new-PPO model (Phase-7A rule). |
| LAMP-Structured | The LAMP allocation (unstructured LAMP scores on the dense model) at the global sparsity whose per-unit ratios, realized structurally, give the parameter count closest to the primary new-PPO model's. The per-unit ratios are realized with the same rule. |
| Old PPO-Structured | The Phase-7A model, `[0,2,5,5]`. Its frozen recovered checkpoint (same protocol) is reused read-only; nothing is rerun. |

**Measured alongside (not separate final-table methods):**

- **Per-seed selections.** Every distinct confirmatory-seed selected policy, recovered and measured so
  that GREEN can be counted per seed.
- **Enumeration reference** (section 8).

**Recovery.** One identical protocol for all; nothing receives extra training.

**Hardware validation.** The Phase-7A protocol: AC, Balanced plan, batch 1, one intra-op and one inter-op
thread, pinned to CPU 2, 1,000 warm-up inferences, N = 100,000 measured inferences per block, fresh
process per session, 5 sessions with a randomized order.

**Dense anchor.** Sessions are longer than in Phase 7A, so Dense is measured in **3 blocks** at random
positions in every session. E_dense(session) is the median of the 3 blocks.

**Energy statistics (Phase 7A, unchanged).**

- Saving = 1 − E / E_dense, within the session.
- Noise threshold = 2 × the between-session coefficient of variation of the dense anchor.
- "Energy reduced" means lower than the same-session dense in 5 of 5 sessions **and** a median saving
  above the noise threshold.
- The exact one-sided Wilcoxon test is reported (minimum p = 0.031). Paired tests against Uniform are
  descriptive.

**Latency.** Reduced if the median-over-sessions latency is below dense **and** lower in 5 of 5 sessions.

## 12. GREEN-SUCCESS (never overridden)

A model is **GREEN-SUCCESS = YES** if all of these hold:

- the VAL-SELECT accuracy loss is ≤ 0.5 pp;
- memory reduction > 0;
- MAC reduction > 0;
- latency is reduced (section 11);
- energy is reduced (section 11).

Otherwise it is **NO**.

**Cases**, applied to the primary new-PPO model:

| Case | Condition |
|---|---|
| GREEN-A | GREEN-SUCCESS, and neither Uniform nor LAMP has a better trade-off. |
| GREEN-B | GREEN-SUCCESS, but Uniform or LAMP has a better trade-off (Phase-7A comparison rule, accuracy on VAL-SELECT). |
| GREEN-C | Memory, MACs and energy all improve, but accuracy fails. |
| GREEN-D | Accuracy is preserved, but energy is not reduced. |

The GREEN-SUCCESS frequency over the 20 seeds is reported.

**Pareto analysis.** Over the final candidates, in four planes: accuracy–energy, accuracy–MACs,
accuracy–memory and energy–MACs.

## 13. Stop conditions

STOP and report if any of the following happens:

- a slimming integrity check fails;
- the backend is not the RAPL/EMI hardware counter;
- the energy pilot is unstable;
- the surrogate fails;
- no proxy passes;
- a confirmatory step would require changing a frozen rule;
- measurement noise is comparable to the model differences.

## 14. Estimated cost (SimpleCNN; 8 recovery workers, ~100 s/epoch on P-cores, ~190 s on E-cores)

| Stage-A / B step | Approximate cost |
|---|---|
| Proxy validation (49 policies × 2 + 3 + 5 epochs) | 2.5–3 h |
| Energy calibration (pilot + 3 sessions × 200 policies × 20,000 inferences) | 3 h, needs exclusive CPU |
| Proxy landscape (1,296 × E epochs) | 13–15 h (E = 2) or 20–22 h (E = 3) |
| PPO tuning and confirmatory search on the cache | under 2 h |
| Final recovery (≤ 26 models × 5 epochs) | 1 h |
| Hardware validation (5 sessions of ≤ 30 min) | 2.5 h |
