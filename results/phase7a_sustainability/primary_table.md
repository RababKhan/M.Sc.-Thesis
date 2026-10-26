| Architecture | Method | Accuracy (%) | ΔAcc (pp) | Params ↓ | MACs ↓ | Latency (ms) | Speed-up | Energy / 1000 inf. (J) | Energy ↓ | Normalized EDP |
|---|---|---|---|---|---|---|---|---|---|---|
| SimpleCNN | Dense | 80.60 | +0.00 | 0.0% | 0.0% | 0.7313 | 1.000× | 11.054 | 0.0% | 1.000 |
| SimpleCNN | PPO-Structured | 75.52 | -5.08 | 80.1% | 42.6% | 0.5711 | 1.280× | 8.541 | 22.8% | 0.603 |
| SimpleCNN | Uniform-Structured | 79.12 | -1.48 | 44.7% | 42.5% | 0.5341 | 1.373× | 7.906 | 27.9% | 0.518 |
| SimpleCNN | LAMP-Structured | 78.84 | -1.76 | 65.5% | 19.7% | 0.6579 | 1.117× | 9.760 | 12.1% | 0.783 |
| ResNet-8 | Dense | 78.40 | +0.00 | 0.0% | 0.0% | 0.6984 | 1.000× | 10.882 | 0.0% | 1.000 |
| ResNet-8 | PPO-Structured | 75.16 | -3.24 | 25.0% | 21.5% | 0.6400 | 1.092× | 9.592 | 13.4% | 0.791 |
| ResNet-8 | Uniform-Structured | 75.60 | -2.80 | 16.1% | 19.3% | 0.6459 | 1.074× | 9.645 | 11.8% | 0.818 |
| ResNet-8 | LAMP-Structured | 75.24 | -3.16 | 25.0% | 18.2% | 0.6490 | 1.068× | 9.819 | 8.9% | 0.828 |
