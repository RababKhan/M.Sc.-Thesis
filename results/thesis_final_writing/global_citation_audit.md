# Global citation audit (Abstract and Chapters 1–8)

- **Numbering:** IEEE, generated mechanically from citation keys in order of first appearance (Chapter 1 first).
  Citation numbers in the assembled source and the bibliography come from the same key list, so a mismatch between
  them is impossible by construction.
- **Content checks:**
  - Chapter 2's characterisations were checked against each paper's abstract in Milestone 1 (`qa_chapter_02.md`).
  - Every citation in the other chapters was checked here against Chapter 2's characterisation and the stored
    abstracts (`tools/reference_abstracts.json`).

## Summary

| Chapter | Citation occurrences | Distinct references | Largest cluster | Clusters with ≥ 4 references |
|---|---|---|---|---|
| Abstract | 0 | 0 | 0 | 0 |
| Ch. 1 | 15 | 18 | 2 | 0 |
| Ch. 2 | 131 | 91 | 4 | 2 |
| Ch. 3 | 8 | 12 | 2 | 0 |
| Ch. 4 | 15 | 16 | 2 | 0 |
| Ch. 5 | 2 | 2 | 1 | 0 |
| Ch. 6 | 5 | 8 | 3 | 0 |
| Ch. 7 | 12 | 16 | 3 | 0 |
| Ch. 8 | 0 | 0 | 0 | 0 |

Total citation occurrences: 188; distinct references: 98 (all verified; see `reference_audit_final.md`).

## Checks

| Check | Result |
|---|---|
| Missing citations: a key in the text without a bibliography entry | **0**: every cited key is verified and appears in `references_final.md` |
| Bibliography entries never cited | **0**: the bibliography is generated from the cited keys; 2 verified items (`sutton2018rl`, `xia2022cofi`) are uncited and excluded |
| Incorrect citation numbers | **0**: numbers are generated, not typed |
| Placeholder citations (`[@key]` left unresolved, `[?]`, TODO) | **0** after conversion in `thesis_full_source.md` (checked by the assembly script) |
| Large clusters (≥ 4) | 2, both in Chapter 2 §2.14 (gaps 3 and 6). Each lists the specific works that motivate a gap, so their size is justified |
| Citations for claims the paper does not support | **0 unsupported. 1 partly supported** (below) |
| Statistical methods named without citation (Chapter 4 QA item 18) | **resolved**: Holm [OpenAlex], Wilcoxon and Youden [Crossref] added at first mention in Chapter 4 (logged in `qa_chapter_04.md`) |

## Citations outside Chapter 2: claim-support review

| Ch. | Ref. no. | Key(s) | Claim supported (abridged) | Verdict |
|---|---|---|---|---|
| 1 | [1], [2] | desislavov2023trends, luccioni2024powerhungry | …Inference energy and its trends have become a subject of study in their own right | supported (consistent with Ch. 2 and the abstract) |
| 1 | [3] | menghani2023efficient | …cing these costs, through pruning, quantisation, distillation and efficient architecture design | supported (consistent with Ch. 2 and the abstract) |
| 1 | [4], [5] | lecun1989obd, han2015learning | …It is one of the oldest compression techniques | supported (consistent with Ch. 2 and the abstract) |
| 1 | [6] | cheng2024pruningsurvey | …mains among the most studied; a recent survey counts thousands of pruning papers in a few years | supported (consistent with Ch. 2 and the abstract) |
| 1 | [7] | he2024structuredsurvey | …rs or neurons, so the network becomes physically smaller and its layers perform less arithmetic | supported (consistent with Ch. 2 and the abstract) |
| 1 | [8] | hoefler2021sparsity | …Only the second shrinks the dense computation that general-purpose hardware actually executes | supported (consistent with Ch. 2 and the abstract) |
| 1 | [9] | lee2021lamp | …h rescale weight magnitudes so that a single global threshold produces a non-uniform allocation | supported (origin of the named method, model, dataset or tool) |
| 1 | [10] | he2018amc | …er, rewarded by the accuracy and compression of the resulting network, are a well-known example | supported (consistent with Ch. 2 and the abstract) |
| 1 | [11] | schulman2017ppo | …This thesis uses Proximal Policy Optimisation (PPO) | supported (origin of the named method, model, dataset or tool) |
| 1 | [12], [13] | engstrom2020implementation, andrychowicz2021onpolicy | …to design details. Policy-gradient methods are sensitive to implementation and training choices | supported (consistent with Ch. 2 and the abstract) |
| 1 | [14], [15] | renda2020rewinding, le2021retraining | …runed networks are usually fine-tuned before use, and fine-tuning can repair much of the damage | supported (consistent with Ch. 2 and the abstract) |
| 1 | [16] | dehghani2022misnomer | …removed and the computation saved need not coincide, and different cost indicators can disagree | supported (consistent with Ch. 2 and the abstract) |
| 1 | [10] | he2018amc | …The claim is not that RL has never been applied to pruning; it has | supported (consistent with Ch. 2 and the abstract) |
| 1 | [17] | pan2022misspecification | …pruning rewards. Hand-weighted accuracy–sparsity rewards can be exploited by capable optimisers | supported (consistent with Ch. 2 and the abstract) |
| 1 | [18] | blalock2020state | …covery, or under different recovery budgets, may not survive identical post-pruning fine-tuning | **partly supported.** The paper documents inconsistent evaluation setups that confound pruning comparisons. The recovery-specific part of the claim is supported by `renda2020rewinding` and `le2021retraining`, cited for Problem D in the same chapter. Acceptable; no change |
| 3 | [92] | paszke2019pytorch | …gnitude weights of each tensor t are masked to zero, using PyTorch's pruning reparameterisation | supported (origin of the named method, model, dataset or tool) |
| 3 | [17], [89] | pan2022misspecification, skalse2022rewardgaming | …the general observation that more capable optimisers exploit misspecified rewards more strongly | supported (consistent with Ch. 2 and the abstract) |
| 3 | [14], [15] | renda2020rewinding, le2021retraining | …network is normally fine-tuned before use, and recovery can reduce or reverse those differences | supported (consistent with Ch. 2 and the abstract) |
| 3 | [7], [8] | hoefler2021sparsity, he2024structuredsurvey | …tiplied unless sparse kernels are used, so parameter sparsity need not reduce latency or energy | supported (consistent with Ch. 2 and the abstract) |
| 3 | [22], [23] | li2017filters, he2017channel | …ion as a physically smaller dense network, using structured channel (filter) and neuron removal | supported (consistent with Ch. 2 and the abstract) |
| 3 | [54] | raffin2021sb3 | … Phases 3, 4, 5 and 7B).   Setting   Value    --- ---    Implementation   Stable-Baselines3 PPO | supported (origin of the named method, model, dataset or tool) |
| 3 | [11] | schulman2017ppo | … 2,048 timesteps = 512 episodes of 4 steps per run     Threads   1 per run   PPO objective. PPO | supported (origin of the named method, model, dataset or tool) |
| 3 | [53] | schulman2016gae | … the clipped surrogate objective, with advantages estimated by generalised advantage estimation | supported (origin of the named method, model, dataset or tool) |
| 4 | [93] | krizhevsky2009cifar | …## 4.2 Datasets The experiments use CIFAR-10 and CIFAR-100 | supported (origin of the named method, model, dataset or tool) |
| 4 | [94] | lecun1998lenet | …LeNet-5, adapted to CIFAR. The LeNet-5 topology | supported (origin of the named method, model, dataset or tool) |
| 4 | [95] | he2016resnet | …ResNet-8. The CIFAR residual network | supported (origin of the named method, model, dataset or tool) |
| 4 | [18] | blalock2020state | …It is a standard reference point in pruning evaluations | supported (consistent with Ch. 2 and the abstract) |
| 4 | [33] | evci2020rigl | …- ERK | supported (origin of the named method, model, dataset or tool) |
| 4 | [9] | lee2021lamp | …- LAMP | supported (origin of the named method, model, dataset or tool) |
| 4 | [54] | raffin2021sb3 | …int)   not matched   ## 4.8 PPO Training Configuration Every agent uses Stable-Baselines3 2.8.0 | supported (origin of the named method, model, dataset or tool) |
| 4 | [11] | schulman2017ppo | …It runs PPO | supported (origin of the named method, model, dataset or tool) |
| 4 | [53] | schulman2016gae | …It runs PPO [@schulman2017ppo] with generalised advantage estimation | supported (origin of the named method, model, dataset or tool) |
| 4 | [96] | youden1950index | …- The protocol's Youden-J | supported (origin of the named method, model, dataset or tool) |
| 4 | [92] | paszke2019pytorch | …Masked models (Phase 6). The mask is fixed by PyTorch's pruning reparameterisation | supported (origin of the named method, model, dataset or tool) |
| 4 | [97] | wilcoxon1945individual | …terval (10,000 resamples in Phase 3); - the effect size d_z; - the exact Wilcoxon signed-rank p | supported (origin of the named method, model, dataset or tool) |
| 4 | [98] | holm1979sequential | …Multiplicity. Holm's step-down correction | supported (origin of the named method, model, dataset or tool) |
| 4 | [86] | khan2018rapl | …ergy Measurement Protocol ### 4.18.1 Backend Energy is read from the Intel RAPL package counter | supported (origin of the named method, model, dataset or tool) |
| 4 | [87], [88] | jay2023powermeters, raffin2024dissecting | …Software energy counters need careful protocol design and stability checks | supported (consistent with Ch. 2 and the abstract) |
| 5 | [33] | evci2020rigl | …al policy of each run was compared on test accuracy with Uniform, Random, Global magnitude, ERK | supported (origin of the named method, model, dataset or tool) |
| 5 | [9] | lee2021lamp | … compared on test accuracy with Uniform, Random, Global magnitude, ERK [@evci2020rigl] and LAMP | supported (origin of the named method, model, dataset or tool) |
| 6 | [8] | hoefler2021sparsity | …kernels, those zeros are still loaded and multiplied unless sparse formats and kernels are used | supported (consistent with Ch. 2 and the abstract) |
| 6 | [16] | dehghani2022misnomer | …Cost indicators such as parameter count, FLOPs and measured speed can disagree with one another | supported (consistent with Ch. 2 and the abstract) |
| 6 | [7], [22], [23] | li2017filters, he2017channel, he2024structuredsurvey | …licy structurally: whole channels and neurons are removed, and a smaller dense network is built | supported (consistent with Ch. 2 and the abstract) |
| 6 | [86] | khan2018rapl | …Operational energy   CPU package energy per 1,000 inferences, from the RAPL counter | supported (origin of the named method, model, dataset or tool) |
| 6 | [87], [88] | jay2023powermeters, raffin2024dissecting | …er on a laptop is not a guaranteed-stable instrument, which is why each study had to qualify it | supported (consistent with Ch. 2 and the abstract) |
| 7 | [10] | he2018amc | …In this sense the thesis confirms the premise behind RL-based compression | supported (consistent with Ch. 2 and the abstract) |
| 7 | [64] | yu2020evalnas | …uring fields: - the gap between the search phase and the final selection in architecture search | supported (consistent with Ch. 2 and the abstract) |
| 7 | [66] | ecoffet2021goexplore | … architecture search [@yu2020evalnas]; - the value of archives of visited states in exploration | supported (consistent with Ch. 2 and the abstract) |
| 7 | [65] | oh2018sil | … imitation term towards the best policies seen so far (in the spirit of self-imitation learning | supported (consistent with Ch. 2 and the abstract) |
| 7 | [17], [89] | pan2022misspecification, skalse2022rewardgaming | …of a general observation: more capable optimisers exploit misspecified objectives more strongly | supported (consistent with Ch. 2 and the abstract) |
| 7 | [90] | achiam2017cpo | …a constraint rather than a traded quantity, in the spirit of constrained reinforcement learning | supported (consistent with Ch. 2 and the abstract) |
| 7 | [9] | lee2021lamp | … Remains a Strong Benchmark, and Why Equal Fine-Tuning Changes the Rankings ### 7.7.1 LAMP LAMP | supported (origin of the named method, model, dataset or tool) |
| 7 | [9] | lee2021lamp | …Its rescaled magnitude score sets layer-wise sparsity automatically | supported (origin of the named method, model, dataset or tool) |
| 7 | [14], [15], [18] | renda2020rewinding, le2021retraining, blalock2020state | …re. This is consistent with work showing that retraining choices can change pruning conclusions | supported (consistent with Ch. 2 and the abstract) |
| 7 | [8] | hoefler2021sparsity | …Without sparse kernels, a masked model is not smaller or faster on a CPU | supported (consistent with Ch. 2 and the abstract) |
| 7 | [16] | dehghani2022misnomer | …This is consistent with the broader observation that cost indicators can disagree | supported (consistent with Ch. 2 and the abstract) |
| 7 | [86]–[88] | khan2018rapl, jay2023powermeters, raffin2024dissecting | …are energy meters, which report that their readings need careful protocol design and validation | supported (consistent with Ch. 2 and the abstract) |

## Result

- **Citation audit: PASS.** No missing, unused, mis-numbered or unsupported citation was found. One citation is partly
  supported and acceptable as it stands.
- **No thesis claim was changed to fit a citation.** The only edit was the addition of the three statistical-method
  citations to Chapter 4.
