"""Per-unit pruning sensitivity.

accuracy_drop     notebook cell 28 / rl_env.compute_layer_sensitivities: accuracy
                  lost when one unit alone is L1-pruned by 10%
loss_sensitivity  the refined multi-ratio definition used by the soft-prior,
                  exploration and archive studies: mean over ratios {10, 20, 40, 60}%
                  of the relative loss increase (L_r - L_0) / L_0 when one unit alone
                  is pruned, then min-max normalised to [0, 1] across units
Both only ever receive V_RL (or the legacy probe) tensors; never V_SELECT or test.
"""
import copy

import numpy as np

from src.evaluation import evaluate_accuracy, mean_loss_accuracy
from src.pruning import apply_unit_ratio, get_units

LOSS_RATIOS = (0.1, 0.2, 0.4, 0.6)


def _pruned_copy(model, arch, unit_index, ratio):
    m = copy.deepcopy(model)
    apply_unit_ratio(get_units(m, arch)[unit_index], ratio)
    return m


def accuracy_drop(model, arch, loader, probe_amount=0.1):
    base = evaluate_accuracy(model, loader)
    names = [u.name for u in get_units(model, arch)]
    return base, {n: base - evaluate_accuracy(_pruned_copy(model, arch, i, probe_amount), loader)
                  for i, n in enumerate(names)}


def min_max(values):
    lo, hi = min(values), max(values)
    return [0.0] * len(values) if hi - lo <= 0 else [(v - lo) / (hi - lo) for v in values]


def loss_sensitivity(model, arch, x, y, ratios=LOSS_RATIOS):
    loss0, acc0 = mean_loss_accuracy(model, x, y)
    names = [u.name for u in get_units(model, arch)]
    raw, detail = [], {}
    for i, name in enumerate(names):
        rel = []
        for r in ratios:
            loss_r, acc_r = mean_loss_accuracy(_pruned_copy(model, arch, i, r), x, y)
            rel.append((loss_r - loss0) / loss0)
            detail[f"{name}@{int(round(100 * r))}"] = {"loss": loss_r, "accuracy": acc_r}
        raw.append(float(np.mean(rel)))
    return {"units": names, "raw": raw, "normalized": min_max(raw), "baseline_loss": loss0,
            "baseline_accuracy": acc0, "detail": detail, "ratios": list(ratios), "n_images": int(len(x))}
