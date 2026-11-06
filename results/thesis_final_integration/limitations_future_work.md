# Limitations and future work

## Limitations

**Scale and scope**
1. **Small models and datasets.** SimpleCNN, LeNet-5 and ResNet-8 on CIFAR-10/100, with 4 prunable units and 6 ratios
   (1,296 policies). The action space is small enough to enumerate. That made exhaustive ground truth possible, but
   it also makes the RL problem easier than at realistic scale.
2. **CPU only, one primary platform.** All latency and energy results come from one laptop (Intel i7-1260P, Windows
   11, batch 1, one thread, one pinned core). No GPU, accelerator, mobile or embedded validation was done.
3. **The sustainability phases use CIFAR-10 only, and Phase 7B uses SimpleCNN only.** ResNet-8 was not repeated under
   the energy-aware structured PPO.

**Method findings that bound the claims**

4. **LAMP remains competitive or stronger.** It was significantly better before fine-tuning in 5/6 (Phase 3) and 6/6
   (Phases 4–5) settings, still ahead in 3/6 after fine-tuning, and the only structured model within 0.5 pp in
   Phase 7B.
5. **Sensitivity helps exploration but not final accuracy.** In Phase 4, the accuracy-based exploration endpoint
   reversed.
6. **Archive selection improves retention on validation ranking only.** On test it had mixed effects, with large
   ResNet-8 losses under the misaligned reward.
7. **Validation-to-test transfer.** Feasible-on-validation policies were feasible on test in only 67.5% of Phase-5 C1
   runs (0% for C100 ResNet-8). Archive and selection splits are small (2,000–2,500 images; 1 image = 0.04 pp).
8. **Reward design remained imperfect.** No Phase-4 redesigned reward met its criterion. The Phase-5 constraint made
   SimpleCNN slightly worse, and its criteria were only partially met.

**Recovery and evaluation**

9. **Recovery is modest and fixed.** All recovery uses 5 epochs, one optimiser (Adam, lr 1e-4, cosine). Longer or
   tuned recovery might change the rankings.
10. **Few recovery seeds in the deployment phases.** Phase 6 has 20 seeds per setting. Phases 7A and 7B use a single
    recovery seed (600) per model.
11. **The equivalence rule is a CI-inside-margin rule.** It is not a formal equivalence test (TOST).
12. **One reference recipe differs.** The CIFAR-10 SimpleCNN dense reference uses an older 10-epoch training recipe.

**Deployment and sustainability**

13. **Structured translation in Phase 7A was lossy.** Transferring the unstructured Phase-5 allocation to channel
    removal cost 5.1 / 3.2 pp. The ResNet-8 mapping was approximate, and its Uniform comparator was matched only
    within 2.9% of MACs.
14. **Energy-measurement instability.** The RAPL package counter on this laptop was stable in Phase 7A but not in
    Phase 7B: pilot CV 11.36% and 4.01%, qualification CV 19.92%, against a 3% limit. The variance tracks package
    temperature and non-core power, which software controls cannot hold fixed.
15. **Phase 7B energy and latency are unresolved.** No measured energy or latency exists for the Phase-7B models.
16. **The Phase-7B proxy and surrogate are model-selection tools.** The 2-epoch accuracy proxy had precision 0.655,
    and selections sat on its feasibility boundary. The energy surrogate is latency-based. Neither replaces
    measurement.
17. **No lifecycle or carbon analysis.** The work covers package-domain energy only, with no DRAM, wall-plug, training
    energy or embodied carbon.
18. **Computational cost of the approach.** PPO search needs thousands of pruned-model evaluations (2,048 timesteps
    per run; exhaustive landscapes for the analysis), whereas LAMP needs none. The cost–benefit balance currently
    favours heuristics at this scale.
19. **Paper 3 compute stop.** The full structured recovery landscape (H2-C) was stopped for compute and replaced by a
    validated 1-epoch proxy design.

## Future work

Each item follows from a finding above; none is presented as done.

1. **Structured RL from the start.** Phase 7B showed that learning on the deployed structure narrows the accuracy cost
   far more than translating masks (−1.12 vs −4.00 pp). Next: a structured action space throughout, including
   residual-coupled units for ResNets. *(Limitations 13, 3.)*
2. **More reliable feasibility proxies.** Proxy precision (0.655) and the validation-to-test gap (67.5%) limited both
   Phase 5 and Phase 7B. Options: calibrated proxies, uncertainty-aware or margin-based constraints, larger selection
   splits. *(7, 16.)*
3. **Multi-objective optimisation.** Optimise accuracy–memory–latency–energy explicitly with Pareto-front search
   rather than a scalarised reward, comparing against LAMP and Uniform on the same fronts. *(4, 8.)*
4. **Hybrid heuristic + RL.** Initialise or constrain PPO with LAMP-derived allocations, since LAMP is cheap and
   strong. *(4, 18.)*
5. **Larger architectures and datasets.** ResNet-18/50 and MobileNet-class models, then ImageNet scale, where the
   policy space can no longer be enumerated and RL search may pay off. *(1, 18.)*
6. **Stable energy instrumentation first.** Use an external logging power meter, or a desktop or server platform
   with fixed clocks, disabled turbo and controlled cooling. Pre-define an instrument-specific qualification
   criterion before any model comparison. *(14, 15.)*
7. **Energy-aware reward with measured energy.** Replace the latency-based surrogate in the reward with hardware
   measurements, but only after step 6 succeeds. *(15, 16.)*
8. **Other platforms.** GPU, accelerator and mobile/edge (ARM) deployment, with each platform's own latency and
   energy behaviour. *(2.)*
9. **Stronger recovery studies.** Recovery length and optimiser sweeps; multiple recovery seeds for the deployment
   phases; formal TOST equivalence tests. *(9, 10, 11.)*
10. **Carbon-aware deployment analysis.** Only after trustworthy energy measurement exists: carbon-intensity
    conversion and lifecycle accounting, clearly separated from measured operational energy. *(17.)*
