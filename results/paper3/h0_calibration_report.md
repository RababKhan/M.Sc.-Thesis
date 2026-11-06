# Paper 3 — H0 CPU latency calibration report

Protocol: `h0_protocol.md` (committed before this measurement). Machine: 12th Gen Intel(R) Core(TM) i7-1260P, Windows-10-10.0.26200-SP0, Python 3.11.9, torch 2.7.1+cpu; oneDNN available True, MKL True; power plan: Power Scheme GUID: 381b4222-f694-41f0-9685-ff5bb260df2e  (Balanced); battery/AC: {'percent': 86, 'power_plugged': True}.

**H0 decision: GO.** Primary latency configuration: **A**.

## Calibration models (CIFAR-10 SimpleCNN, physically compacted)

| Model | Actions | Widths (conv1, conv2, conv3, fc1) | Parameters | Param. reduction | MACs | MAC reduction |
|---|---|---|---|---|---|---|
| dense | [0, 0, 0, 0] | [32, 64, 128, 256] | 620,362 | 0.0% | 10,848,768 | 0.0% |
| light | [1, 1, 1, 1] | [29, 58, 115, 230] | 501,893 | 19.1% | 8,944,540 | 17.6% |
| moderate | [3, 3, 3, 3] | [22, 45, 90, 179] | 305,850 | 50.7% | 5,481,566 | 49.5% |
| aggressive | [5, 5, 5, 5] | [13, 26, 51, 102] | 99,781 | 83.9% | 1,986,204 | 81.7% |
| fc_heavy | [0, 0, 0, 5] | [32, 64, 128, 102] | 303,276 | 51.1% | 10,531,836 | 2.9% |
| conv_heavy | [0, 5, 5, 0] | [32, 26, 51, 256] | 232,117 | 62.6% | 3,776,896 | 65.2% |

## Configuration A: batch 1, 1 intra-op thread(s), 1 inter-op, CPUs [2]

| Model | Median (ms) | p95 (ms) | IQR (ms) | Throughput (img/s) | Speed-up vs dense | Session medians min–max (ms) | Between-session CV |
|---|---|---|---|---|---|---|---|
| dense | 0.6800 | 0.7420 | 0.0298 | 1,471 | 1.000 | 0.6643–0.6978 | 2.14% |
| light | 0.6092 | 0.6585 | 0.0272 | 1,641 | 1.116 | 0.5935–0.6252 | 2.22% |
| moderate | 0.4537 | 0.4910 | 0.0193 | 2,204 | 1.499 | 0.4432–0.4637 | 1.97% |
| aggressive | 0.2614 | 0.2840 | 0.0134 | 3,826 | 2.597 | 0.2541–0.2701 | 2.62% |
| fc_heavy | 0.6568 | 0.6984 | 0.0316 | 1,523 | 1.034 | 0.6426–0.6770 | 2.38% |
| conv_heavy | 0.4647 | 0.4979 | 0.0219 | 2,152 | 1.466 | 0.4530–0.4770 | 2.49% |

Signal (median over sessions of dense − aggressive): **0.4127 ms** (61.6% of dense); noise (largest between-session range of the two): 0.0335 ms; **SNR = 12.3**; dense − aggressive positive in every session: True; usable: **True**.

## Configuration B: batch 1, 4 intra-op thread(s), 1 inter-op, CPUs [0, 2, 4, 6]

| Model | Median (ms) | p95 (ms) | IQR (ms) | Throughput (img/s) | Speed-up vs dense | Session medians min–max (ms) | Between-session CV |
|---|---|---|---|---|---|---|---|
| dense | 0.3890 | 0.5035 | 0.0293 | 2,571 | 1.000 | 0.3712–0.3999 | 3.23% |
| light | 0.3774 | 0.4911 | 0.0318 | 2,650 | 1.032 | 0.3604–0.3872 | 3.17% |
| moderate | 0.3250 | 0.4174 | 0.0213 | 3,077 | 1.197 | 0.3071–0.3341 | 3.23% |
| aggressive | 0.2675 | 0.3415 | 0.0176 | 3,738 | 1.451 | 0.2616–0.2756 | 2.27% |
| fc_heavy | 0.3853 | 0.4836 | 0.0248 | 2,595 | 1.007 | 0.3685–0.3970 | 3.12% |
| conv_heavy | 0.3236 | 0.4339 | 0.0227 | 3,090 | 1.203 | 0.3123–0.3325 | 2.38% |

Signal (median over sessions of dense − aggressive): **0.1222 ms** (31.8% of dense); noise (largest between-session range of the two): 0.0287 ms; **SNR = 4.3**; dense − aggressive positive in every session: True; usable: **False**.

## Configuration C: batch 128, 4 intra-op thread(s), 1 inter-op, CPUs [0, 2, 4, 6]

| Model | Median (ms) | p95 (ms) | IQR (ms) | Throughput (img/s) | Speed-up vs dense | Session medians min–max (ms) | Between-session CV |
|---|---|---|---|---|---|---|---|
| dense | 24.3152 | 26.2719 | 1.7410 | 5,264 | 1.000 | 23.4315–25.0385 | 2.53% |
| light | 22.7397 | 24.5742 | 1.4926 | 5,629 | 1.069 | 22.0795–23.4159 | 2.52% |
| moderate | 16.6228 | 18.8725 | 1.2443 | 7,700 | 1.472 | 15.8752–17.1911 | 3.39% |
| aggressive | 8.8852 | 10.5808 | 0.6596 | 14,406 | 2.755 | 8.5604–9.1157 | 2.77% |
| fc_heavy | 24.4179 | 28.2120 | 1.8229 | 5,242 | 1.000 | 23.4293–25.2459 | 3.43% |
| conv_heavy | 16.6769 | 18.0154 | 1.0499 | 7,675 | 1.464 | 16.2578–17.0343 | 2.14% |

Signal (median over sessions of dense − aggressive): **15.4531 ms** (63.2% of dense); noise (largest between-session range of the two): 1.6070 ms; **SNR = 9.6**; dense − aggressive positive in every session: True; usable: **True**.

## Decision (rule of `h0_protocol.md`, applied mechanically)

Order A → C → B; usable = SNR ≥ 5, relative reduction ≥ 10% and positive in every session. Usable: A, C. Selected: **A**. Parameters reduced: True; MACs reduced: True. **H0 GO.**

## Hardware checks

| Status | Check | Detail |
|---|---|---|
| PASS | every pruned calibration model has fewer physical parameters than dense |  |
| PASS | every pruned calibration model has fewer MACs than dense |  |
| PASS | FlopCounter FLOPs = 2 x analytic MACs for every calibration model |  |
| PASS | every session process ran with the requested threads, inter-op threads and CPU affinity, oneDNN enabled | 15 session processes |
| PASS | every session used the frozen reference (sha256) and the same input shape |  |
| PASS | CPU load <= 15% before and after every kept session process | 0 repeats |
| PASS | repeated dense timing stable: between-session CV of the dense median <= 5% in every configuration | {'A': 0.0214, 'B': 0.0323, 'C': 0.0253} |
| PASS | dense anchor drift within every session <= 10% | {'A': 0.0161, 'B': 0.0939, 'C': 0.0892} |
| PASS | latency signal larger than noise (SNR >= 5) in the selected configuration | {'A': 12.32, 'B': 4.26, 'C': 9.62} |

Sessions: 5 per configuration, each in a fresh process (15 processes); orders drawn from seed 20261023. Raw passes: `h0_latency_raw.csv`; summary: `h0_latency_summary.csv`; decision: `h0_decision.json`; environment: `h0_environment.json`. All sessions ran back-to-back on one day, so the between-session spread does not include day-to-day variation; the final Paper-3 confirmation session is on a different occasion.
