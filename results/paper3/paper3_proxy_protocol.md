# Paper 3 — pre-registration amendment 2: reduced-compute proxy design (E2)

Written 2026-10-24 on branch `paper3-hardware-aware`, **before any 1-epoch recovery or any new 5-epoch
validation run exists**, and committed before E2 starts.
- **Amends** `paper3_preregistration.md` (amendment 1, `4593684`) where stated. Everything not
  mentioned here is unchanged.
- **Why.** The exhaustive 5-epoch H2-C landscape needs about 130 h of laptop CPU time. It was stopped
  for that reason alone (`h2c_stop_record.md`), not because of any observed accuracy. This redesign is
  likewise motivated by computational feasibility only.
- **Pilot data.** The 126 completed 5-epoch CIFAR-10 SimpleCNN pilot records were **not inspected** to
  define this proxy. Only their policy identifiers were read, to exclude them from the random draw.

## 1. Scope

- **Primary scope:** CIFAR-10 × {SimpleCNN, ResNet-8}, each with the full structured action space of
  1,296 policies (units and action semantics unchanged).
- **CIFAR-100:** leaves the primary scope. Its H2-A latency tables stay frozen and unused.

## 2. Proxy and final accuracy

- **Search accuracy (proxy):** VAL-RL accuracy after a **1-epoch** recovery, the frozen Phase-6
  protocol with E = 1.
  - Everything else is unchanged: Adam lr 1e-4, betas (0.9, 0.999), eps 1e-8, weight decay 0,
    CosineAnnealingLR(T_max = E) stepped per epoch, batch 128, crop + flip, the 45,000 training
    images, seed 600 and data generator 600 + 10⁶, 1 thread, BatchNorm in train mode, no selection.
  - Because the cosine schedule is stepped per epoch, epoch 1 runs at lr 1e-4 in both the E = 1 and
    E = 5 protocols, on the same seed and minibatches. **The proxy model is exactly the state after
    the first epoch of the full protocol.** The proxy definition therefore has no free parameter.
- **Proxy feasibility:** q₁(p) = A₁(p) / A₁(identity), the identity policy recovered with the same
  1 epoch.
- **Final accuracy:** every final selected policy (C1-S, M1, H1), every deterministic baseline policy
  used in the final comparison, and any Pareto-front policy reported receives the original frozen
  **5-epoch** recovery.
  - q₅(p) = A₅(p) / A₅(identity).
  - **All final accuracy claims, and the true constraint q₅ ≥ 0.98, use only these 5-epoch values.**
    The 1-epoch proxy never enters a final claim.

## 3. Proxy validation (before any 1-epoch landscape)

**Validation pairs**, frozen in `paper3_proxy_validation_policies.json` (sha256 `5e3e5384e23d2d52…`,
seed 20261024) before any measurement:

| Architecture | Pairs |
|---|---|
| CIFAR-10 SimpleCNN (156) | the 126 pilot policies (5-epoch records exist), plus 30 policies drawn uniformly from the 1,170 non-pilot policies |
| CIFAR-10 ResNet-8 (31) | the identity policy plus 30 policies drawn uniformly from the 1,295 non-identity policies |

The ResNet-8 pairs are added because the proxy is used for both architectures and the pilot covers
SimpleCNN only. For every pair, both the 1-epoch and the 5-epoch recovery are computed; only the
pilot's 5-epoch values already exist.

**Statistics, per architecture:**
- Spearman ρ between A₁ and A₅ (**primary**); Pearson r (secondary);
- MAE of q (|q₁ − q₅|);
- bias: mean A₁ − A₅ and mean q₁ − q₅, with sign;
- top-10 overlap by A₅ against A₁;
- agreement of q₁ ≥ τ₁ with q₅ ≥ 0.98, as a confusion matrix.

**GO/NO-GO, fixed now and not weakened after seeing the result:**
- The proxy is **GO for an architecture only if Spearman ρ(A₁, A₅) ≥ 0.90** on that architecture's
  validation pairs.
- An architecture with ρ < 0.90 is **STOPPED**: no 1-epoch landscape is run and no PPO uses the proxy
  for it.
- The report goes to the PI before anything else.

## 4. Proxy threshold τ₁ (objective rule, fixed before any 1-epoch value exists)

- **Purpose.** PPO's constrained reward needs a feasibility test on the proxy. τ₁ is set **per
  architecture**, from the validation pairs only, never from the landscape:
  - τ₁ = the value of q₁ that maximises Youden's J = sensitivity + specificity − 1 for predicting
    q₅ ≥ 0.98;
  - candidate thresholds are the observed q₁ values of the validation pairs;
  - ties go to the larger τ₁.
- **Fallback.** If fewer than 3 validation pairs of an architecture have q₅ ≥ 0.98, or fewer than 3
  have q₅ < 0.98, no τ₁ is set for it and the PI decides.
- **The true constraint** q₅ ≥ 0.98 is evaluated only after the 5-epoch confirmation. Ranking and
  search use the proxy.
- **Reporting.** The relationship between q₁ and q₅ is reported in full.

## 5. 1-epoch landscapes

- **Coverage:** only for architectures with proxy GO. All 1,296 policies of CIFAR-10 SimpleCNN and
  of CIFAR-10 ResNet-8, at most 2,592 in total.
- **Method:**
  - the same frozen dense references, action space and importance rule;
  - VAL-RL and VAL-SELECT (VAL-SELECT stored for the later selection step only);
  - exactly 1 epoch, no early stopping, no policy-specific tuning, no test-set access.
- **Shared runs.** The 1-epoch validation runs are members of the landscape and are not repeated.
  Recovery is deterministic, and core-independence was verified in H2-B.
- **Freeze.** The landscapes are frozen and hashed before any PPO run.

## 6. Latency

- **Tables:** the frozen H2-A tables `simplecnn_c10` and `resnet8_c10`, reused unchanged and not
  re-measured.
- **Primary metric:** batch-1, 1-thread latency on the i7-1260P, as relative gain against the identity
  shape of the same table. The final confirmation session reports speed-ups against dense anchors
  measured in the same session.
- **No Phase 1–6 desktop latency is used.**

## 7. Later PPO (not started; separate approval)

- C1-S, M1 and H1 use the same structured action space, the same 1-epoch proxy table, the same τ₁,
  the same sensitivity prior, and the same seeds and PPO settings. Only the efficiency term of the
  reward differs.
- The sensitivity-prior variant is still open (amendment 1, §7) and needs the PI's decision before H3.

## 8. Pilot data

- **Kept unchanged:** the 126 pilot records, the stop records, the 8 incomplete lock files and the
  status marker.
- **Use:** only as 5-epoch values in the proxy validation (§3). Never as part of a final or exhaustive
  landscape.
