# Paper 3 — H1 structured pruning validation

**50/50 checks passed.** Validation images only (Phase-5 VAL-RL); the test set is never read. Structural checks cover all 1,296 action vectors per architecture.

| Status | Section | Check | Detail |
|---|---|---|---|
| PASS | SimpleCNN | every action gives a valid width (1 <= w <= C) and the 6 widths of each unit are distinct | {'conv1': [32, 29, 26, 22, 19, 13], 'conv2': [64, 58, 51, 45, 38, 26], 'conv3': [128, 115, 102, 90, 77, 51], 'fc1': [256, 230, 205, 179, 154, 102]} |
| PASS | SimpleCNN | 1,296 action vectors -> 1,296 distinct physical shapes | 1296 |
| PASS | SimpleCNN | all 1,296: physical parameters and hook MACs equal the analytic counts; no pruning mask left | [] |
| PASS | SimpleCNN | parameters and MACs physically decrease for every non-zero policy; raising any action never adds parameters and always removes MACs | dense (620362, 10848768), [5,5,5,5] (99781, 1986204) |
| PASS | SimpleCNN | cifar10: [0,0,0,0] reproduces the dense model exactly (logits and state dict) |  |
| PASS | SimpleCNN | cifar10: slimmed logits == masked-equivalent logits (29 policies, 512 VAL-RL images) | max /dlogit/ 4.77e-06, argmax agreement 1.000 |
| PASS | SimpleCNN | cifar10: output dimension (10) correct; no mask after compaction; state dict reloads into the policy skeleton (strict) with identical logits |  |
| PASS | SimpleCNN | cifar10: flatten mapping: classifier.1 keeps exactly columns c*16..c*16+15 of every kept conv3 channel |  |
| PASS | SimpleCNN | cifar10: output classifier never pruned (all 10 outputs and the bias kept; only its inputs follow fc1) |  |
| PASS | SimpleCNN | cifar100: [0,0,0,0] reproduces the dense model exactly (logits and state dict) |  |
| PASS | SimpleCNN | cifar100: slimmed logits == masked-equivalent logits (29 policies, 512 VAL-RL images) | max /dlogit/ 9.54e-06, argmax agreement 1.000 |
| PASS | SimpleCNN | cifar100: output dimension (100) correct; no mask after compaction; state dict reloads into the policy skeleton (strict) with identical logits |  |
| PASS | SimpleCNN | cifar100: flatten mapping: classifier.1 keeps exactly columns c*16..c*16+15 of every kept conv3 channel |  |
| PASS | SimpleCNN | cifar100: output classifier never pruned (all 100 outputs and the bias kept; only its inputs follow fc1) |  |
| PASS | ResNet-8 | every action gives a valid width and the 6 widths of each unit are distinct | {'stem': [16, 14, 13, 11, 10, 6], 'stage1': [16, 14, 13, 11, 10, 6], 'stage2': [32, 29, 26, 22, 19, 13], 'stage3': [64, 58, 51, 45, 38, 26]} |
| PASS | ResNet-8 | 1,296 action vectors -> 1,296 distinct physical shapes | 1296 |
| PASS | ResNet-8 | all 1,296: physical parameters and hook MACs equal the analytic counts; no pruning mask left | [] |
| PASS | ResNet-8 | parameters and MACs physically decrease for every non-zero policy; monotone in every unit | dense (78042, 12501632), [5,5,5,5] (31108, 3586176) |
| PASS | ResNet-8 | cifar10: [0,0,0,0] reproduces the dense model exactly (logits and state dict) |  |
| PASS | ResNet-8 | cifar10: with the stem action 0, the 3 block-internal units equal the verified Phase-2 resnet8_structured exactly |  |
| PASS | ResNet-8 | cifar10: Conv2d input/output channels consistent along the coupled stem group, the three block-internal units and the unpruned widths (31 policies) |  |
| PASS | ResNet-8 | cifar10: BatchNorm num_features, weight, bias, running_mean, running_var, eps, num_batches_tracked = dense values at the kept channels |  |
| PASS | ResNet-8 | cifar10: coupled stem group: the same channel set is cut in stem conv/bn, layer1.conv1 inputs, layer1.conv2/bn2 outputs, layer2.conv1 and layer2.shortcut inputs; layer1 shortcut stays identity |  |
| PASS | ResNet-8 | cifar10: residual branch and shortcut have identical shapes in every block (no broadcasting) |  |
| PASS | ResNet-8 | cifar10: classifier input 64 and the classifier unchanged (10 outputs) |  |
| PASS | ResNet-8 | cifar10: random and real-data forward passes have the right output shape and finite logits |  |
| PASS | ResNet-8 | cifar10: slimmed logits == masked-equivalent logits (31 policies, 512 VAL-RL images) | max /dlogit/ 5.72e-06, argmax agreement 1.000 |
| PASS | ResNet-8 | cifar10: no mask after compaction; state dict reloads into the policy skeleton (strict) with identical logits |  |
| PASS | ResNet-8 | cifar100: [0,0,0,0] reproduces the dense model exactly (logits and state dict) |  |
| PASS | ResNet-8 | cifar100: with the stem action 0, the 3 block-internal units equal the verified Phase-2 resnet8_structured exactly |  |
| PASS | ResNet-8 | cifar100: Conv2d input/output channels consistent along the coupled stem group, the three block-internal units and the unpruned widths (31 policies) |  |
| PASS | ResNet-8 | cifar100: BatchNorm num_features, weight, bias, running_mean, running_var, eps, num_batches_tracked = dense values at the kept channels |  |
| PASS | ResNet-8 | cifar100: coupled stem group: the same channel set is cut in stem conv/bn, layer1.conv1 inputs, layer1.conv2/bn2 outputs, layer2.conv1 and layer2.shortcut inputs; layer1 shortcut stays identity |  |
| PASS | ResNet-8 | cifar100: residual branch and shortcut have identical shapes in every block (no broadcasting) |  |
| PASS | ResNet-8 | cifar100: classifier input 64 and the classifier unchanged (100 outputs) |  |
| PASS | ResNet-8 | cifar100: random and real-data forward passes have the right output shape and finite logits |  |
| PASS | ResNet-8 | cifar100: slimmed logits == masked-equivalent logits (31 policies, 512 VAL-RL images) | max /dlogit/ 9.54e-06, argmax agreement 1.000 |
| PASS | ResNet-8 | cifar100: no mask after compaction; state dict reloads into the policy skeleton (strict) with identical logits |  |
| PASS | MAC/FLOP | analytic MACs = forward-hook MACs; FlopCounter FLOPs = 2 x MACs (representative policies, both architectures) | 20 |
| PASS | MAC/FLOP | ptflops (adds BatchNorm, ReLU, pooling) >= conv/linear MACs and within 10% of them | ptflops / MACs 1.0203-1.0586 |
| PASS | structured sensitivity | simplecnn: the dense model is unchanged after probing (state-dict hash) |  |
| PASS | structured sensitivity | simplecnn: deterministic (two calls identical) |  |
| PASS | structured sensitivity | simplecnn: normalised values in [0, 1] with min 0 and max 1 | [0.913, 1.0, 0.351, 0.0] |
| PASS | structured sensitivity | simplecnn: each probe equals the compacted single-unit policy of the action space (60% probes) |  |
| PASS | structured sensitivity | simplecnn: a 0% probe is the dense model (zero loss change) |  |
| PASS | structured sensitivity | resnet8: the dense model is unchanged after probing (state-dict hash) |  |
| PASS | structured sensitivity | resnet8: deterministic (two calls identical) |  |
| PASS | structured sensitivity | resnet8: normalised values in [0, 1] with min 0 and max 1 | [0.871, 1.0, 0.0, 0.057] |
| PASS | structured sensitivity | resnet8: each probe equals the compacted single-unit policy of the action space (60% probes) |  |
| PASS | structured sensitivity | resnet8: a 0% probe is the dense model (zero loss change) |  |

## MAC/FLOP counters (representative policies)

| Arch | Policy | Analytic MACs | Hook MACs | FlopCounter FLOPs | ptflops MACs |
|---|---|---|---|---|---|
| simplecnn | [0, 0, 0, 0] | 10,848,768 | 10,848,768 | 21,697,536 | 11,136,266 |
| simplecnn | [1, 1, 1, 1] | 8,944,540 | 8,944,540 | 17,889,080 | 9,204,760 |
| simplecnn | [3, 3, 3, 3] | 5,481,566 | 5,481,566 | 10,963,132 | 5,681,153 |
| simplecnn | [5, 5, 5, 5] | 1,986,204 | 1,986,204 | 3,972,408 | 2,102,680 |
| simplecnn | [5, 0, 0, 0] | 7,521,792 | 7,521,792 | 15,043,584 | 7,712,010 |
| simplecnn | [0, 5, 0, 0] | 5,245,440 | 5,245,440 | 10,490,880 | 5,484,298 |
| simplecnn | [0, 0, 5, 0] | 7,694,848 | 7,694,848 | 15,389,696 | 7,957,706 |
| simplecnn | [0, 0, 0, 5] | 10,531,836 | 10,531,836 | 21,063,672 | 10,818,872 |
| simplecnn | [0, 5, 5, 0] | 3,776,896 | 3,776,896 | 7,553,792 | 3,991,114 |
| simplecnn | [5, 3, 3, 1] | 4,373,564 | 4,373,564 | 8,747,128 | 4,527,224 |
| resnet8 | [0, 0, 0, 0] | 12,501,632 | 12,501,632 | 25,003,264 | 12,755,594 |
| resnet8 | [1, 1, 1, 1] | 10,526,848 | 10,526,848 | 21,053,696 | 10,758,922 |
| resnet8 | [3, 3, 3, 3] | 7,424,128 | 7,424,128 | 14,848,256 | 7,620,682 |
| resnet8 | [5, 5, 5, 5] | 3,586,176 | 3,586,176 | 7,172,352 | 3,726,090 |
| resnet8 | [5, 0, 0, 0] | 8,456,832 | 8,456,832 | 16,913,664 | 8,649,354 |
| resnet8 | [0, 5, 0, 0] | 9,552,512 | 9,552,512 | 19,105,024 | 9,775,754 |
| resnet8 | [0, 0, 5, 0] | 10,400,384 | 10,400,384 | 20,800,768 | 10,639,754 |
| resnet8 | [0, 0, 0, 5] | 10,400,384 | 10,400,384 | 20,800,768 | 10,647,050 |
| resnet8 | [5, 5, 0, 0] | 7,350,912 | 7,350,912 | 14,701,824 | 7,512,714 |
| resnet8 | [3, 0, 5, 2] | 7,878,016 | 7,878,016 | 15,756,032 | 8,084,170 |

## Structured sensitivity (unit test on 512 VAL-RL images; not a final vector)

- simplecnn: units ['conv1', 'conv2', 'conv3', 'fc1'], raw [0.6761, 0.7394, 0.2688, 0.0141], normalised [0.913, 1.0, 0.351, 0.0]
- resnet8: units ['stem', 'stage1', 'stage2', 'stage3'], raw [8.6765, 9.6355, 2.2201, 2.6399], normalised [0.871, 1.0, 0.0, 0.057]

## Sanity: VAL-RL accuracy of representative policies (CIFAR-10; 2,500 images; decides nothing)

| Arch | Policy | Actions | Params | MACs | VAL-RL acc. (%) | Retention |
|---|---|---|---|---|---|---|
| simplecnn | dense | [0, 0, 0, 0] | 620,362 | 10,848,768 | 76.88 | 1.0000 |
| simplecnn | light | [1, 1, 1, 1] | 501,893 | 8,944,540 | 74.16 | 0.9646 |
| simplecnn | moderate | [3, 3, 3, 3] | 305,850 | 5,481,566 | 61.04 | 0.7940 |
| simplecnn | aggressive | [5, 5, 5, 5] | 99,781 | 1,986,204 | 17.48 | 0.2274 |
| simplecnn | fc_heavy | [0, 0, 0, 5] | 303,276 | 10,531,836 | 75.84 | 0.9865 |
| simplecnn | conv_heavy | [0, 5, 5, 0] | 232,117 | 3,776,896 | 25.84 | 0.3361 |
| resnet8 | dense | [0, 0, 0, 0] | 78,042 | 12,501,632 | 78.92 | 1.0000 |
| resnet8 | light | [1, 1, 1, 1] | 69,812 | 10,526,848 | 20.96 | 0.2656 |
| resnet8 | moderate | [3, 3, 3, 3] | 53,503 | 7,424,128 | 17.80 | 0.2255 |
| resnet8 | aggressive | [5, 5, 5, 5] | 31,108 | 3,586,176 | 9.76 | 0.1237 |
| resnet8 | stem_only_60 | [5, 0, 0, 0] | 71,652 | 8,456,832 | 10.60 | 0.1343 |
