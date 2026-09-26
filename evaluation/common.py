"""Shared definitions for the evaluation scripts.

These mirror the definitions in notebooks/Main code.ipynb exactly. Every script
that uses them asserts that it reproduces the notebook's recorded numbers before
reporting anything new, so a silent drift between this file and the notebook
would fail loudly rather than quietly change a result.
"""
import copy
import math

import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

# The four layers the RL action space can prune. classifier.3 (the output layer)
# is deliberately excluded, matching prunable_layers_filtered in the notebook.
FILTERED_LAYERS = ["features.0", "features.3", "features.6", "classifier.1"]

# ACTION_TO_PRUNE from the notebook: six actions, no 50% level.
ACTION_TO_PRUNE = {0: 0.0, 1: 0.1, 2: 0.2, 3: 0.3, 4: 0.4, 5: 0.6}


class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 256),
            nn.ReLU(),
            nn.Linear(256, 10)
        )

    def forward(self, x):
        return self.classifier(self.features(x))


def load_baseline(path, device="cpu"):
    model = SimpleCNN().to(device)
    model.load_state_dict(torch.load(path, map_location=device))
    model.eval()
    return model


def test_loader(root="data", batch_size=500):
    """CIFAR-10 test set, un-augmented, in a fixed order (shuffle=False).

    The fixed order is what makes the paired McNemar tests reproducible.
    """
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.4914, 0.4822, 0.4465),
            std=(0.2470, 0.2435, 0.2616)
        )
    ])
    dataset = torchvision.datasets.CIFAR10(root=root, train=False, download=False, transform=transform)
    return DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)


def predictions(model, loader, device="cpu"):
    """Per-sample predicted labels and ground truth, in loader order."""
    model.eval()
    preds, labels = [], []
    with torch.no_grad():
        for images, targets in loader:
            images = images.to(device)
            _, predicted = torch.max(model(images), 1)
            preds.append(predicted.cpu())
            labels.append(targets)
    return torch.cat(preds), torch.cat(labels)


def accuracy_from(preds, labels):
    """Integer-exact accuracy, matching the notebook's `100 * correct / total`.

    Averaging in float32 instead introduces ~3e-6 of error, which is enough to
    break an exact reproduction check against the notebook's recorded numbers.
    """
    correct = int((preds == labels).sum())
    return 100.0 * correct / len(labels)


def calculate_sparsity(model):
    """Percentage of zero weights across every Conv2d/Linear layer.

    Returns a PERCENTAGE (58.2), not a fraction (0.582) -- the reward function
    consumes this value directly, so the scale matters.
    """
    total = zeros = 0
    for module in model.modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            total += module.weight.nelement()
            zeros += torch.sum(module.weight == 0).item()
    return 100.0 * zeros / total


def per_layer_sparsity(model):
    out = {}
    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            out[name] = 100.0 * torch.sum(module.weight == 0).item() / module.weight.nelement()
    return out


def layer_param_counts(model):
    return {
        name: module.weight.nelement()
        for name, module in model.named_modules()
        if isinstance(module, (nn.Conv2d, nn.Linear))
    }


def prune_layerwise(model, amounts, layers=FILTERED_LAYERS):
    """Prune each named layer independently by its own fraction (L1 unstructured)."""
    pruned = copy.deepcopy(model)
    for layer_name, amount in zip(layers, amounts):
        for name, module in pruned.named_modules():
            if name == layer_name:
                prune.l1_unstructured(module, name="weight", amount=amount)
                break
        else:
            raise ValueError(f"layer not found: {layer_name}")
    return pruned


def prune_uniform_to_total(model, target_total_sparsity, layers=FILTERED_LAYERS):
    """Uniform per-layer pruning calibrated to hit an exact TOTAL sparsity.

    Every layer in `layers` loses the same fraction; that fraction is scaled up
    because the layers outside `layers` stay dense and dilute the total.
    """
    counts = layer_param_counts(model)
    total = sum(counts.values())
    covered = sum(counts[name] for name in layers)
    fraction = (target_total_sparsity / 100.0) * total / covered
    if fraction > 1.0:
        raise ValueError(f"target {target_total_sparsity}% unreachable: needs {fraction:.3f} per layer")
    return prune_layerwise(model, [fraction] * len(layers), layers), fraction


def prune_global_to_total(model, target_total_sparsity, layers=FILTERED_LAYERS):
    """Global magnitude pruning over `layers`, calibrated to an exact TOTAL sparsity.

    Restricting it to the same four layers the agent controls is what makes this
    a like-for-like baseline: both methods get the same weight budget over the
    same parameters, and only the allocation differs.
    """
    pruned = copy.deepcopy(model)
    counts = layer_param_counts(model)
    total = sum(counts.values())
    covered = sum(counts[name] for name in layers)
    fraction = (target_total_sparsity / 100.0) * total / covered

    targets = [
        (module, "weight")
        for name, module in pruned.named_modules()
        if name in layers
    ]
    prune.global_unstructured(targets, pruning_method=prune.L1Unstructured, amount=fraction)
    return pruned, fraction


def _log_binomial_tail(n, k):
    """log( sum_{i=0}^{k} C(n, i) ), computed stably via lgamma.

    Done in log space because the direct form needs 2**n as an exact integer,
    which overflows once the two models disagree on a few thousand samples.
    """
    log_terms = [
        math.lgamma(n + 1) - math.lgamma(i + 1) - math.lgamma(n - i + 1)
        for i in range(k + 1)
    ]
    largest = max(log_terms)
    return largest + math.log(sum(math.exp(t - largest) for t in log_terms))


def mcnemar(correct_a, correct_b):
    """Exact two-sided McNemar test on paired per-sample correctness.

    Returns (b, c, chi2_with_continuity_correction, exact_p) where b counts
    samples only A gets right and c counts samples only B gets right. A p of
    exactly 0.0 means the true value underflowed float64 (< ~1e-308).
    """
    b = int((correct_a & ~correct_b).sum())
    c = int((~correct_a & correct_b).sum())
    n = b + c
    if n == 0:
        return b, c, 0.0, 1.0

    chi2 = ((abs(b - c) - 1) ** 2) / n

    log_p = math.log(2.0) + _log_binomial_tail(n, min(b, c)) - n * math.log(2.0)
    p = min(1.0, math.exp(log_p)) if log_p > -745.0 else 0.0
    return b, c, chi2, p


def format_p(p):
    """McNemar's exact p underflows to 0.0 for large, lopsided disagreements."""
    if p == 0.0:
        return "<1e-300"
    if p < 0.001:
        return f"{p:.2e}"
    return f"{p:.4f}"
