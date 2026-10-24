# Paper 3 — E2 integrity checks

**10/10 passed.**

| Status | Check | Detail |
|---|---|---|
| PASS | cifar10_simplecnn: proxy NO-GO, no landscape (by rule) | 0.844211537118283 |
| PASS | cifar10_resnet8: 1,296 complete 1-epoch records, one per policy | 1296 |
| PASS | cifar10_resnet8: frozen H2-A latency table unchanged (sha256) | aeb0d452d0c03c1b |
| PASS | cifar10_resnet8: every record: exactly 1 epoch, seed 600, Phase-6 protocol, 1 thread, no test use, frozen reference, amendment-2 hash | 41fc5ee592b96fa0 |
| PASS | cifar10_resnet8: identical minibatch exposure (one data-stream hash) |  |
| PASS | cifar10_resnet8: identity before recovery = frozen reference (Phase-5 dense VAL-RL record) | 78.92 |
| PASS | cifar10_resnet8: shapes unchanged in every run; widths, parameters and MACs equal the latency table |  |
| PASS | cifar10_resnet8: amendment 2 (ec7a7f2) committed before the first 1-epoch run | 2026-10-24T11:41:32+06:00 < 2026-10-24T14:09:48+06:00 |
| PASS | 5-epoch validation runs: every record 5 epochs, seed 600, shapes unchanged, no test use | 61 |
| PASS | pilot H2-C records unchanged since the stop commit (git diff empty) |  |
