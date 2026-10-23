# Paper 3 — H0/H1 integrity checks (summary)

| Suite | Script | Result | Record |
|---|---|---|---|
| Hardware (H0) | `experiments/paper3/h0_calibrate.py` | 9/9 PASS | `h0_decision.json`, `h0_calibration_report.md` |
| Structured pruning (H1): SimpleCNN, ResNet-8, MAC/FLOP counters, structured sensitivity | `experiments/paper3/h1_validate.py` | 50/50 PASS | `h1_checks.json`, `h1_validation_report.md` |
| Regression: unchanged Phase-2 implementation checks, rerun on this machine | `experiments/paper3/regression.py` | 69/69 PASS, identical to the frozen record | `regression/cpu_phase2_checks.md` |

**Total: 128/128 passed, no failures.**

- **Test set.** No script here reads it.
- **Frozen files.**
  - The Phase-2 regression writes were redirected; `git status` shows no change under
    `results/architecture_generalization` or `results/reproducibility`.
  - No Phase 1–6 file was modified.
