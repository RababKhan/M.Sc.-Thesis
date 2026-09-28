"""Structured (filter / channel / neuron) pruning with actual removal. PREPARED ONLY.

This file is independent of the unstructured framework in src/pruning/__init__.py
and changes nothing in it. It builds a physically smaller network, so parameter
counts, MACs and CPU latency fall for real (unlike masked unstructured pruning).
Phase 2 only implements and tests it; no structured-pruning experiment is run.

Ranking criterion: L1 norm of each output filter / neuron (Li et al., ICLR 2017),
ties broken by lower index. A unit ratio r keeps max(1, round((1 - r) * C)) of
its C channels.

SimpleCNN (4 structured units, mirroring the unstructured ones)
  features.0 / features.3 / features.6   output filters of that conv; the next
                                         conv loses the matching input channels
  classifier.1                           hidden neurons; classifier.3 loses the
                                         matching input columns
  features.6 feeds classifier.1 through Flatten: channel c owns input columns
  c*16 ... c*16+15 (4x4 spatial map, channel-major order).

ResNet-8 (3 structured units)
  stage1 / stage2 / stage3   the block-internal channels (conv1 outputs, bn1,
                             conv2 inputs). The stem width and the stage output
                             widths are tied together by the residual additions
                             and are left unchanged, the standard dependency-safe
                             choice for residual networks.
"""
import copy

import torch
import torch.nn as nn

STRUCTURED_UNITS = {
    "simplecnn": ["features.0", "features.3", "features.6", "classifier.1"],
    "resnet8": ["stage1", "stage2", "stage3"],
}


def l1_keep(weight, ratio):
    """Indices (sorted) of the filters/neurons to keep: highest L1 norm, count max(1, round((1-r)*C))."""
    c = weight.shape[0]
    keep = max(1, int(round((1.0 - ratio) * c)))
    norms = weight.detach().abs().flatten(1).sum(1)
    order = sorted(range(c), key=lambda i: (-float(norms[i]), i))
    return sorted(order[:keep])


def _conv(src, out_idx, in_idx):
    new = nn.Conv2d(len(in_idx), len(out_idx), src.kernel_size, src.stride, src.padding,
                    bias=src.bias is not None)
    with torch.no_grad():
        new.weight.copy_(src.weight[out_idx][:, in_idx])
        if src.bias is not None:
            new.bias.copy_(src.bias[out_idx])
    return new


def _bn(src, idx):
    new = nn.BatchNorm2d(len(idx), eps=src.eps, momentum=src.momentum)
    with torch.no_grad():
        for name in ("weight", "bias", "running_mean", "running_var"):
            getattr(new, name).copy_(getattr(src, name)[idx])
        new.num_batches_tracked.copy_(src.num_batches_tracked)
    return new


def _linear(src, out_idx, in_idx):
    new = nn.Linear(len(in_idx), len(out_idx), bias=src.bias is not None)
    with torch.no_grad():
        new.weight.copy_(src.weight[out_idx][:, in_idx])
        if src.bias is not None:
            new.bias.copy_(src.bias[out_idx])
    return new


def simplecnn_structured(model, ratios):
    """Physically slimmed copy of a SimpleCNN; ratios for (features.0, features.3, features.6, classifier.1)."""
    m = copy.deepcopy(model).eval()
    f, c = m.features, m.classifier
    k0 = l1_keep(f[0].weight, ratios[0])
    k1 = l1_keep(f[3].weight, ratios[1])
    k2 = l1_keep(f[6].weight, ratios[2])
    kh = l1_keep(c[1].weight, ratios[3])
    spatial = c[1].in_features // f[6].out_channels
    cols = [ch * spatial + s for ch in k2 for s in range(spatial)]
    f[0] = _conv(f[0], k0, list(range(f[0].in_channels)))
    f[3] = _conv(f[3], k1, k0)
    f[6] = _conv(f[6], k2, k1)
    c[1] = _linear(c[1], kh, cols)
    c[3] = _linear(c[3], list(range(c[3].out_features)), kh)
    return m.eval(), {"features.0": k0, "features.3": k1, "features.6": k2, "classifier.1": kh}


def resnet8_structured(model, ratios):
    """Slimmed copy of a ResNet8; ratios for the internal channels of (layer1, layer2, layer3)."""
    m = copy.deepcopy(model).eval()
    kept = {}
    for name, r in zip(("layer1", "layer2", "layer3"), ratios):
        block = getattr(m, name)
        k = l1_keep(block.conv1.weight, r)
        block.conv1 = _conv(block.conv1, k, list(range(block.conv1.in_channels)))
        block.bn1 = _bn(block.bn1, k)
        block.conv2 = _conv(block.conv2, list(range(block.conv2.out_channels)), k)
        kept[name.replace("layer", "stage")] = k
    return m.eval(), kept


def structured(model, arch, ratios):
    if arch == "simplecnn":
        return simplecnn_structured(model, ratios)
    if arch == "resnet8":
        return resnet8_structured(model, ratios)
    raise NotImplementedError(f"structured pruning is prepared for {list(STRUCTURED_UNITS)} only")


def masked_equivalent(model, arch, kept):
    """The dense-shaped model with removed channels zeroed; must compute what the slimmed model computes.

    Used by the implementation checks: removed filters get zero weight AND zero
    bias (or zero BN scale and shift), so their post-ReLU activations are exactly 0.
    """
    m = copy.deepcopy(model).eval()
    with torch.no_grad():
        if arch == "simplecnn":
            for name, mod in (("features.0", m.features[0]), ("features.3", m.features[3]),
                              ("features.6", m.features[6]), ("classifier.1", m.classifier[1])):
                drop = [i for i in range(mod.weight.shape[0]) if i not in set(kept[name])]
                mod.weight[drop] = 0.0
                mod.bias[drop] = 0.0
        elif arch == "resnet8":
            for i in (1, 2, 3):
                block = getattr(m, f"layer{i}")
                drop = [j for j in range(block.conv1.out_channels) if j not in set(kept[f"stage{i}"])]
                block.conv1.weight[drop] = 0.0
                block.bn1.weight[drop] = 0.0
                block.bn1.bias[drop] = 0.0
    return m
