"""Paper 3: structured (physically compacted) pruning units, action semantics and structured sensitivity.

Builds on src/pruning/structured.py (Phase 2, verified, unchanged): the same L1 filter ranking `l1_keep`
(keep max(1, round((1 - r) * C)) channels with the largest L1 norm; ties: lower index) and the same
layer-rebuilding helpers. Every importance score is computed on the DENSE model, so the result does not
depend on the order in which units are slimmed.

Actions (as in every earlier phase): a in {0..5} -> ratio {0, 10, 20, 30, 40, 60}% of the unit's channels.

SimpleCNN, 4 units (identical to structured.simplecnn_structured)
  conv1, conv2, conv3   output filters of features.0 / .3 / .6; the next conv (or classifier.1 through the
                        channel-major flatten, 16 columns per channel) loses the matching inputs
  fc1                   hidden neurons of classifier.1; classifier.3 loses the matching inputs
  classifier.3 (the output layer) is never pruned.

ResNet-8, 4 units
  stem    the stage-1 residual width (16 channels): every tensor that writes or reads the residual stream
          before stage 2 is cut on the same channel set K:
            writes   conv1 (stem) outputs + bn1;  layer1.conv2 outputs + layer1.bn2
            passes   layer1 identity shortcut (no parameters)
            reads    layer1.conv1 inputs; layer2.conv1 inputs; layer2.shortcut.0 (1x1 conv) inputs
          score(c) = ||conv1.weight[c]||_1 + ||layer1.conv2.weight[c]||_1 (the two filters that write channel c)
  stage1, stage2, stage3
          block-internal channels: layerN.conv1 outputs + layerN.bn1, layerN.conv2 inputs
          (identical to structured.resnet8_structured when the stem action is 0)
  The stage-2/3 output widths and the classifier are not pruned.
"""
import copy

import torch

from src import models as M
from src.pruning import ACTION_TO_PRUNE
from src.pruning import structured as SP

UNITS = {"simplecnn": ["conv1", "conv2", "conv3", "fc1"], "resnet8": ["stem", "stage1", "stage2", "stage3"]}
BASE_WIDTHS = {"simplecnn": [32, 64, 128, 256], "resnet8": [16, 16, 32, 64]}
LOSS_RATIOS = (0.1, 0.2, 0.4, 0.6)


def keep_count(c, ratio):
    """The number of channels kept by structured.l1_keep."""
    return max(1, int(round((1.0 - ratio) * c)))


def widths(arch, actions):
    return [keep_count(c, ACTION_TO_PRUNE[int(a)]) for c, a in zip(BASE_WIDTHS[arch], actions)]


def _ratios(actions):
    assert len(actions) == 4 and all(int(a) in ACTION_TO_PRUNE for a in actions)
    return [ACTION_TO_PRUNE[int(a)] for a in actions]


def _group_keep(scores, ratio):
    c = len(scores)
    keep = keep_count(c, ratio)
    order = sorted(range(c), key=lambda i: (-float(scores[i]), i))
    return sorted(order[:keep])


def resnet8_keep(model, ratios):
    """Kept channel indices of the four ResNet-8 units, all scored on the dense `model`."""
    stem_score = model.conv1.weight.detach().abs().flatten(1).sum(1) + model.layer1.conv2.weight.detach().abs().flatten(1).sum(1)
    kept = {"stem": _group_keep(stem_score, ratios[0])}
    for i, r in zip((1, 2, 3), ratios[1:]):
        kept[f"stage{i}"] = SP.l1_keep(getattr(model, f"layer{i}").conv1.weight, r)
    return kept


def resnet8_slim(model, ratios):
    m = copy.deepcopy(model).eval()
    k = resnet8_keep(model, ratios)
    g = k["stem"]
    allc = lambda conv: list(range(conv.out_channels))  # noqa: E731
    m.conv1 = SP._conv(model.conv1, g, list(range(model.conv1.in_channels)))
    m.bn1 = SP._bn(model.bn1, g)
    l1, l2, l3 = m.layer1, m.layer2, m.layer3
    d1, d2, d3 = model.layer1, model.layer2, model.layer3
    l1.conv1, l1.bn1 = SP._conv(d1.conv1, k["stage1"], g), SP._bn(d1.bn1, k["stage1"])
    l1.conv2, l1.bn2 = SP._conv(d1.conv2, g, k["stage1"]), SP._bn(d1.bn2, g)
    assert len(d1.shortcut) == 0, "layer1 must have an identity shortcut"
    l2.conv1, l2.bn1 = SP._conv(d2.conv1, k["stage2"], g), SP._bn(d2.bn1, k["stage2"])
    l2.conv2 = SP._conv(d2.conv2, allc(d2.conv2), k["stage2"])
    l2.shortcut[0] = SP._conv(d2.shortcut[0], allc(d2.shortcut[0]), g)
    l3.conv1, l3.bn1 = SP._conv(d3.conv1, k["stage3"], list(range(d3.conv1.in_channels))), SP._bn(d3.bn1, k["stage3"])
    l3.conv2 = SP._conv(d3.conv2, allc(d3.conv2), k["stage3"])
    return m.eval(), k


def slim(model, arch, actions):
    """Physically compacted copy of a dense `model` under a 4-unit action vector; returns (model, kept indices)."""
    r = _ratios(actions)
    if arch == "simplecnn":
        m, kept = SP.simplecnn_structured(model, r)
        return m, dict(zip(UNITS[arch], kept.values()))
    if arch == "resnet8":
        return resnet8_slim(model, r)
    raise NotImplementedError(arch)


def masked_equivalent(model, arch, kept):
    """Dense-shaped model with every removed channel silenced (weights and BN scale/shift 0); computes what the
    slimmed model computes (eval mode)."""
    if arch == "simplecnn":
        return SP.masked_equivalent(model, arch, dict(zip(["features.0", "features.3", "features.6", "classifier.1"], kept.values())))
    m = SP.masked_equivalent(model, arch, {k: kept[k] for k in ("stage1", "stage2", "stage3")})
    drop = [c for c in range(m.conv1.out_channels) if c not in set(kept["stem"])]
    with torch.no_grad():
        for conv, bn in ((m.conv1, m.bn1), (m.layer1.conv2, m.layer1.bn2)):
            conv.weight[drop] = 0.0
            bn.weight[drop] = 0.0
            bn.bias[drop] = 0.0
    return m


def skeleton(arch, actions, num_classes):
    """A model with exactly the slimmed shapes (weights arbitrary): the target of load_state_dict for a slimmed checkpoint."""
    return slim(M.build_model(arch, num_classes, seed=0).eval(), arch, actions)[0]


def shape_key(model):
    return tuple((n, tuple(p.shape)) for n, p in model.state_dict().items())


# ------------------------------------------------------------------ analytic counts (independent of the hooks)
def analytic(arch, actions, num_classes):
    """(conv/linear MACs per image, total parameters) of the slimmed model, from the widths alone."""
    w = widths(arch, actions)
    if arch == "simplecnn":
        a, b, c, h = w
        macs = 32 * 32 * a * 3 * 9 + 16 * 16 * b * a * 9 + 8 * 8 * c * b * 9 + c * 16 * h + h * num_classes
        params = (a * 27 + a) + (b * a * 9 + b) + (c * b * 9 + c) + (c * 16 * h + h) + (h * num_classes + num_classes)
        return macs, params
    g, i1, i2, i3 = w
    macs = (32 * 32 * g * 3 * 9 + 32 * 32 * i1 * g * 9 + 32 * 32 * g * i1 * 9 + 16 * 16 * i2 * g * 9 + 16 * 16 * 32 * i2 * 9
            + 16 * 16 * 32 * g + 8 * 8 * i3 * 32 * 9 + 8 * 8 * 64 * i3 * 9 + 8 * 8 * 64 * 32 + 64 * num_classes)
    conv = g * 27 + i1 * g * 9 + g * i1 * 9 + i2 * g * 9 + 32 * i2 * 9 + 32 * g + i3 * 32 * 9 + 64 * i3 * 9 + 64 * 32
    bn = 2 * (g + i1 + g + i2 + 32 + 32 + i3 + 64 + 64)
    return macs, conv + bn + 64 * num_classes + num_classes


# ------------------------------------------------------------------ structured sensitivity
def single_unit_actions(unit, ratio):
    """The 4-unit action vector that removes `ratio` of one unit only (ratio must be an action ratio)."""
    a = {v: k for k, v in ACTION_TO_PRUNE.items()}[ratio]
    return [a if i == unit else 0 for i in range(4)]


def structured_loss_sensitivity(model, arch, x, y, ratios=LOSS_RATIOS):
    """Per-unit structured sensitivity with the same definition as src.sensitivity.loss_sensitivity, but with
    physical channel removal: mean over `ratios` of (L_r - L_0) / L_0 when one unit alone is slimmed (the
    dense model is never modified; each probe is a fresh compacted copy), then min-max normalised."""
    from src.evaluation import mean_loss_accuracy
    from src.sensitivity import min_max
    loss0, acc0 = mean_loss_accuracy(model, x, y)
    raw, detail = [], {}
    for u, name in enumerate(UNITS[arch]):
        rel = []
        for r in ratios:
            m, _ = slim(model, arch, single_unit_actions(u, r))
            loss_r, acc_r = mean_loss_accuracy(m, x, y)
            rel.append((loss_r - loss0) / loss0)
            detail[f"{name}@{int(round(100 * r))}"] = {"loss": loss_r, "accuracy": acc_r}
        raw.append(float(sum(rel) / len(rel)))
    return {"units": UNITS[arch], "raw": raw, "normalized": min_max(raw), "baseline_loss": loss0,
            "baseline_accuracy": acc0, "detail": detail, "ratios": list(ratios), "n_images": int(len(x))}
