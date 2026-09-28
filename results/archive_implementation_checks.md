# Archive-confirmatory experiment — implementation checks

Run 2026-09-28T11:42:33+06:00 with `python evaluation/archive_confirm.py check`, after the pre-registration commit (`1bda844`) and before any experimental run. Training checks use seed 42 only (non-experimental).
**Overall: ALL CHECKS PASS**


## 1. A0 reproduces ordinary PPO when beta = 0 — PASS

A0 with SoftPriorPolicy (beta 0) vs SB3 MlpPolicy on V_RL, seed 42: all 512 training episodes identical = **True**; terminal policies [0, 1, 5, 5] vs [0, 1, 5, 5].

## 2. A1 uses the correct sensitivity vector — PASS

A1 prior = [1.0, 0.03160260491301242, 0.009350119670520027, 0.0] = `archive_sensitivity_vector.json` (sha256 `de30b20ccd865db3…`), beta 0.5; prior matrix = beta·S·a_norm exactly.

## 3. A2 has identical sensitivity values but wrong assignment — PASS

All 40 seeds: A2 vector is a permutation of S, every layer's value differs from its own, and it equals the vector listed in the pre-registration.

## 4. A3 has identical average prior strength — PASS

A3 vector = mean(S) = np.float64(0.26023818114588315) for every layer; mean penalty A1 0.057830706921, A3 0.057830706921.

## 5. All actions remain available — PASS

Minimum action probability of untrained policies (seed-200 priors; no training) over the four layer observations: A0 0.1650, A1 0.1244, A2 0.1246, A3 0.1547. No masking anywhere.

## 6. V_SELECT is inaccessible during PPO training — PASS

Inside `ArchiveData.training()` (wrapping `agent.learn`), evaluating a policy on V_SELECT raises: **True**.

## 7. TEST is inaccessible before archive selection — PASS

TEST raises during training (**True**) and after training until a selection is frozen (**True**).

## 8. The archive stores every unique completed policy — PASS

For both seed-42 check runs: the archive's policy set equals the set of policies completed in the 512 training episodes, with no duplicates, and times_sampled sums to 512.

## 9. Duplicate policies are not repeatedly evaluated — PASS

V_SELECT evaluations performed = 425; cached policies = 425; distinct policies across the three runs' archives and terminals = 425.

## 10. The selection rule is deterministic — PASS

Synthetic archives: out-of-band high accuracy is ignored; equal accuracy is broken by distance to 58.0%, then lower loss ([1,0,5,5] over [0,0,5,5] and [0,1,5,5]); equal everything by the lexicographically smaller action vector; row order does not matter; with no band candidate the policy closest to 58.0% is chosen and flagged.

## 11. The target sparsity interval is fixed — PASS

Constants BAND = (57.5, 58.5), TARGET = 58.0 (as pre-registered). Boundaries are inclusive: 57.5 and 58.5 are in the band, 58.5000001 is not.

## 12. The final selected policy is frozen before test evaluation — PASS

For each check run the selected-policy file exists before the prediction file, and its sha256 equals the hash recorded in the run record.

## 13. The test set is evaluated only after selection — PASS

Every test-loader request during the three check runs came in the selection phase with a frozen policy: [('selection', True), ('selection', True), ('selection', True)].

## 14. Seeds are fresh — PASS

Locked seeds 200–239 vs every earlier run record: overlap [].

## 15. Baseline hash unchanged — PASS

sha256 ca28f8345c23365ee7b92686e780ae0a9befa76c9d6bb9b3bb0b9e42272e111c.
