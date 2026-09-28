"""Pruning units, L1 unstructured unit pruning, sparsity and one-shot baselines.

Pruning units (the PPO decision steps)
  Each architecture registers exactly four units, so every setting has the same
  6^4 = 1,296-policy action space as the historical SimpleCNN studies. A unit is
  one or more Conv2d/Linear weight tensors; a PPO action applies one ratio to
  every tensor in the unit (L1 unstructured, per tensor, as
  torch.nn.utils.prune.l1_unstructured). The output classifier is excluded in
  every architecture, as classifier.3 was for SimpleCNN. BatchNorm parameters
  and biases are never pruned.

Sparsity
  Always a PERCENTAGE of zero weights over ALL Conv2d/Linear weight tensors,
  including the excluded classifier (historical calculate_sparsity).

One-shot baselines (calibrated to an exact number K of zeroed weights)
  Given a target total sparsity s (percent) and W weights in all Conv2d/Linear
  tensors, K = round(s * W / 100). Every zero comes from the prunable tensors
  P = {t_1..t_m} (n_t weights each, N = sum n_t); the excluded classifier stays
  dense. Integer per-tensor counts use largest-remainder rounding so they sum
  to K exactly.
  uniform   every prunable tensor loses the same fraction K/N of its
            smallest-magnitude weights
  global    the K smallest |w| over the union of P (torch global_unstructured)
  random    uniform allocation, but the weights removed inside each tensor are
            drawn uniformly at random (explicit generator, seeded per run)
  LAMP      Lee et al. (ICLR 2021): within tensor t with weights sorted so that
            |w_(1)| <= ... <= |w_(n)|, score(u) = w_(u)^2 / sum_{v >= u} w_(v)^2;
            remove the K smallest scores over the union of P. Every tensor keeps
            its largest weight (score 1).
  ERK       Evci et al. (ICML 2020) layer allocation: density
            d_t = min(1, eps * (sum of the dimensions of t) / n_t) with eps
            chosen so that sum_t d_t n_t = N - K (tensors reaching d_t = 1 are
            fixed dense and eps re-solved); within each tensor the
            smallest-magnitude weights are removed.
"""
import copy
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.utils.prune as prune

ACTION_TO_PRUNE = {0: 0.0, 1: 0.1, 2: 0.2, 3: 0.3, 4: 0.4, 5: 0.6}

UNIT_SPECS = {
    "simplecnn": [("features.0", ["features.0"], "conv"), ("features.3", ["features.3"], "conv"),
                  ("features.6", ["features.6"], "conv"), ("classifier.1", ["classifier.1"], "linear")],
    "lenet5": [("conv1", ["features.0"], "conv"), ("conv2", ["features.3"], "conv"),
               ("fc1", ["classifier.1"], "linear"), ("fc2", ["classifier.3"], "linear")],
    "resnet8": [("stem", ["conv1"], "conv"),
                ("stage1", ["layer1.conv1", "layer1.conv2"], "residual stage"),
                ("stage2", ["layer2.conv1", "layer2.conv2", "layer2.shortcut.0"], "residual stage"),
                ("stage3", ["layer3.conv1", "layer3.conv2", "layer3.shortcut.0"], "residual stage")],
    "smallvgg": [("stage1", ["features.0", "features.3"], "conv stage"),
                 ("stage2", ["features.7", "features.10"], "conv stage"),
                 ("stage3", ["features.14", "features.17"], "conv stage"),
                 ("fc1", ["classifier.1"], "linear")],
}
EXCLUDED = {"simplecnn": ["classifier.3"], "lenet5": ["classifier.5"], "resnet8": ["fc"],
            "smallvgg": ["classifier.3"]}


@dataclass
class Unit:
    name: str
    kind: str
    modules: list        # [(module_name, module)]

    @property
    def param_count(self):
        return sum(m.weight.numel() for _, m in self.modules)

    def weights(self):
        """Current (masked) weights of the unit, concatenated in module order."""
        if len(self.modules) == 1:
            return self.modules[0][1].weight.detach().cpu()
        return torch.cat([m.weight.detach().cpu().flatten() for _, m in self.modules])


def weight_layers(model):
    return [(n, m) for n, m in model.named_modules() if isinstance(m, (nn.Conv2d, nn.Linear))]


def get_units(model, arch):
    """Resolve the registered units on `model`; asserts units + excluded cover every weight tensor once."""
    named = dict(model.named_modules())
    units = [Unit(name, kind, [(mn, named[mn]) for mn in mods]) for name, mods, kind in UNIT_SPECS[arch]]
    covered = [mn for u in units for mn, _ in u.modules] + EXCLUDED[arch]
    all_names = [n for n, _ in weight_layers(model)]
    assert sorted(covered) == sorted(all_names) and len(set(covered)) == len(covered), \
        f"{arch}: units + excluded != all Conv2d/Linear layers"
    return units


def prunable_tensors(units):
    return [(mn, m) for u in units for mn, m in u.modules]


# ------------------------------------------------------------------ applying and measuring
def apply_unit_ratio(unit, ratio):
    """L1 unstructured pruning of every tensor in the unit by `ratio` (in place)."""
    for _, m in unit.modules:
        prune.l1_unstructured(m, name="weight", amount=ratio)


def prune_units(model, arch, ratios):
    """Copy of `model` with unit i pruned by ratios[i] (historical prune_layerwise for SimpleCNN)."""
    pruned = copy.deepcopy(model)
    units = get_units(pruned, arch)
    assert len(ratios) == len(units)
    for unit, r in zip(units, ratios):
        apply_unit_ratio(unit, r)
    return pruned


def prune_actions(model, arch, actions):
    return prune_units(model, arch, [ACTION_TO_PRUNE[int(a)] for a in actions])


def total_sparsity(model):
    """Percentage of zero weights across every Conv2d/Linear layer (historical calculate_sparsity)."""
    total = zeros = 0
    for _, m in weight_layers(model):
        total += m.weight.nelement()
        zeros += torch.sum(m.weight == 0).item()
    return 100.0 * zeros / total


def layer_sparsity(model):
    return {n: 100.0 * torch.sum(m.weight == 0).item() / m.weight.nelement() for n, m in weight_layers(model)}


def unit_sparsity(model, arch):
    return {u.name: 100.0 * sum(torch.sum(m.weight == 0).item() for _, m in u.modules) / u.param_count
            for u in get_units(model, arch)}


def weight_total(model):
    return sum(m.weight.nelement() for _, m in weight_layers(model))


def zero_count(model):
    return sum(int(torch.sum(m.weight == 0).item()) for _, m in weight_layers(model))


def make_permanent(model):
    """Fold every pruning mask into its weight (removes weight_orig / weight_mask)."""
    for _, m in weight_layers(model):
        if prune.is_pruned(m) and hasattr(m, "weight_mask"):
            prune.remove(m, "weight")
    return model


# ------------------------------------------------------------------ one-shot baselines
def target_count(model, target_sparsity):
    """K = round(target% of all Conv2d/Linear weights)."""
    return int(round(target_sparsity / 100.0 * weight_total(model)))


def largest_remainder(shares, total):
    """Integers proportional to `shares` that sum to `total` exactly (ties: lower index)."""
    shares = np.asarray(shares, float)
    base = np.floor(shares).astype(int)
    rest = int(total - base.sum())
    order = sorted(range(len(shares)), key=lambda i: (-(shares[i] - base[i]), i))
    for i in order[:rest]:
        base[i] += 1
    return base


def _setup(model, arch, target_sparsity):
    pruned = copy.deepcopy(model)
    units = get_units(pruned, arch)
    tensors = prunable_tensors(units)
    k = target_count(pruned, target_sparsity)
    n = np.array([m.weight.numel() for _, m in tensors])
    if not 0 <= k <= n.sum():
        raise ValueError(f"target {target_sparsity}% needs {k} zeros but only {n.sum()} weights are prunable")
    return pruned, tensors, k, n


def uniform_fraction(model, arch, fraction):
    """Historical uniform pruning: the float fraction on every prunable tensor (torch rounds per tensor)."""
    pruned = copy.deepcopy(model)
    for _, m in prunable_tensors(get_units(pruned, arch)):
        prune.l1_unstructured(m, name="weight", amount=fraction)
    return pruned


def global_fraction(model, arch, fraction):
    """Historical global magnitude pruning with a float fraction of the prunable weights."""
    pruned = copy.deepcopy(model)
    prune.global_unstructured([(m, "weight") for _, m in prunable_tensors(get_units(pruned, arch))],
                              pruning_method=prune.L1Unstructured, amount=fraction)
    return pruned


def uniform(model, arch, target_sparsity):
    pruned, tensors, k, n = _setup(model, arch, target_sparsity)
    counts = largest_remainder(n * k / n.sum(), k)
    for (_, m), c in zip(tensors, counts):
        prune.l1_unstructured(m, name="weight", amount=int(c))
    return pruned


def global_magnitude(model, arch, target_sparsity):
    pruned, tensors, k, _ = _setup(model, arch, target_sparsity)
    prune.global_unstructured([(m, "weight") for _, m in tensors], pruning_method=prune.L1Unstructured,
                              amount=int(k))
    return pruned


def random_uniform(model, arch, target_sparsity, seed):
    pruned, tensors, k, n = _setup(model, arch, target_sparsity)
    counts = largest_remainder(n * k / n.sum(), k)
    g = torch.Generator().manual_seed(int(seed))
    for (_, m), c in zip(tensors, counts):
        mask = torch.ones(m.weight.numel())
        mask[torch.randperm(m.weight.numel(), generator=g)[: int(c)]] = 0.0
        prune.custom_from_mask(m, name="weight", mask=mask.view_as(m.weight))
    return pruned


def lamp_scores(weight):
    """LAMP score of every weight of one tensor (float64, same shape)."""
    w2 = weight.detach().double().flatten() ** 2
    sorted_w2, idx = torch.sort(w2, stable=True)
    suffix = torch.flip(torch.cumsum(torch.flip(sorted_w2, [0]), 0), [0])
    scores = torch.zeros_like(w2)
    scores[idx] = torch.where(suffix > 0, sorted_w2 / suffix, torch.zeros_like(suffix))
    return scores.view_as(weight)


def lamp(model, arch, target_sparsity):
    pruned, tensors, k, n = _setup(model, arch, target_sparsity)
    scores = torch.cat([lamp_scores(m.weight).flatten() for _, m in tensors])
    remove = torch.argsort(scores, stable=True)[:k]
    flat_mask = torch.ones(len(scores))
    flat_mask[remove] = 0.0
    offset = 0
    for (_, m), size in zip(tensors, n):
        prune.custom_from_mask(m, name="weight", mask=flat_mask[offset:offset + size].view_as(m.weight))
        offset += size
    return pruned


def erk_densities(shapes, keep):
    """ERK densities for weight tensors of the given shapes keeping `keep` weights in total."""
    n = np.array([int(np.prod(s)) for s in shapes], float)
    raw = np.array([sum(s) / np.prod(s) for s in shapes], float)
    dense = np.zeros(len(shapes), bool)
    while True:
        eps = (keep - n[dense].sum()) / (raw[~dense] * n[~dense]).sum()
        d = np.where(dense, 1.0, eps * raw)
        over = (~dense) & (d > 1.0)
        if not over.any():
            return d
        dense |= over


def erk(model, arch, target_sparsity):
    pruned, tensors, k, n = _setup(model, arch, target_sparsity)
    keep = int(n.sum() - k)
    d = erk_densities([tuple(m.weight.shape) for _, m in tensors], keep)
    kept = largest_remainder(d * n, keep)
    kept = np.minimum(kept, n)
    assert kept.sum() == keep, "ERK integer allocation"
    for (_, m), size, c in zip(tensors, n, kept):
        prune.l1_unstructured(m, name="weight", amount=int(size - c))
    return pruned


BASELINES = {"uniform": uniform, "global": global_magnitude, "random": random_uniform, "lamp": lamp, "erk": erk}
