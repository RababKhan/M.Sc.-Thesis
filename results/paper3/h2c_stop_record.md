# H2-C stop record

**Status: Stopped for computational feasibility; pilot/feasibility data only.**

| Item | Value |
|---|---|
| Timestamp | 2026-10-24, workers stopped 11:30–11:35 (+06:00) |
| Git commit of the runs | `7bfb144` (code, protocol and pre-registration `4593684` unchanged) |
| Planned | 5,184 policies (4 settings × 1,296), fixed 5-epoch recovery |
| Completed | **126**, all CIFAR-10 SimpleCNN; all records valid (5 epochs, shapes unchanged, no test use) |
| Incomplete | 8 (CIFAR-10 SimpleCNN 0330, 0331, 0332, 0333, 0334, 0335, 0340, 0341) |
| Not started | 5,050 (rest of CIFAR-10 SimpleCNN; all of CIFAR-10 ResNet-8, CIFAR-100 SimpleCNN, CIFAR-100 ResNet-8) |

**Incomplete policies.** Each was claimed in the instant before its worker stopped and had run for
about one second. None wrote a record or a work state. Their lock files are kept; a later resume must
remove them first.

**Workers.** All 8 were stopped by the operator immediately after each had written its last record, so
no write was interrupted. No Python process remains. Last completed policy per worker:

| CPU | Core | Last completed |
|---|---|---|
| 2 | P | 0323 |
| 4 | P | 0324 |
| 6 | P | 0325 |
| 8 | E | 0322 |
| 10 | E | 0312 |
| 12 | E | 0320 |
| 14 | E | 0314 |
| 15 | E | 0321 |

**Reason.** The exhaustive 5-epoch design needs about 130 hours (≈ 5 days) of continuous laptop CPU
time, projected from the 126 measured runs. The PI judged this impractical. This is a
computational-feasibility decision.

**Unrelated to outcomes.** The PI decided to stop from runtime alone. During H2-C only run counts,
runtimes and errors were monitored. No recovered-accuracy value from the landscape runs was
summarised, inspected or used, either for this decision or for the runtime estimates, which use run
time and MACs only.

**Use rule.**
- These 126 records are **not** the Paper-3 primary landscape. They may be used only as
  pilot/feasibility data, unless a later committed protocol explicitly permits otherwise.
- The H2-A latency tables (`h2_latency_freeze.json`), H0 and H1 are unchanged and frozen.
- The pre-registration is not modified by this record.

Machine-readable record: `h2c_stop_record.json`. Runs: `_h2_recovery_runs/`.
