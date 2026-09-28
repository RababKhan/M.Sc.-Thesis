"""Sensitivity-Aware Constrained PPO, step 1: curves, action masks, state values.

Implements results/constrained_preregistration.md exactly. Validation data
only; the test set is never loaded here.

  * every prunable layer alone at 10/20/30/40/60% (the non-zero action levels),
    fresh baseline copy per probe, evaluated on the full validation split and
    on its five fixed 1,000-image folds
  * primary mask (accuracy): ratio safe iff the one-sided 95% t upper bound of
    the five fold drops is < 1.0 pp; allowed = all ratios <= largest safe ratio
  * secondary mask (loss): same on relative cross-entropy increase with margin
    1.0 / (100 - 77.32) = 4.41%
  * state values: mean relative accuracy drop / mean relative loss increase
    over the five ratios (full validation), min-max normalised
  * heuristic policy C5: each layer at its largest safe action

Outputs
  results/constrained_sensitivity_curves.csv   layer x ratio x {full, fold 0..4}
  results/sensitivity_action_masks.csv         one row per (rule, layer)
  results/constrained_sensitivity_state.csv    state values per definition

Run from the repository root:  python evaluation/constrained_masks.py
"""
import copy
import datetime
import json
import os
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common  # noqa: E402
import rl_env  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(ROOT, "results")
RATIOS = [0.1, 0.2, 0.3, 0.4, 0.6]
FOLD, N_FOLDS = 1000, 5
T95_DF4 = 2.132                       # one-sided 95%, df = 4
ACC_MARGIN_PP = 1.0
BASELINE_VAL = 77.32
LOSS_MARGIN_REL = ACC_MARGIN_PP / (100.0 - BASELINE_VAL)   # 0.04409...
RATIO_OF_ACTION = rl_env.ACTION_TO_PRUNE


def per_image(model, images, labels):
    model.eval()
    loss_fn = nn.CrossEntropyLoss(reduction="none")
    correct, losses = [], []
    with torch.no_grad():
        for i in range(0, len(images), 128):
            out = model(images[i:i + 128])
            correct.append(out.argmax(1) == labels[i:i + 128])
            losses.append(loss_fn(out, labels[i:i + 128]))
    return torch.cat(correct).numpy().astype(float), torch.cat(losses).numpy()


def ucb(values):
    v = np.asarray(values, float)
    return float(v.mean() + T95_DF4 * v.std(ddof=1) / np.sqrt(len(v)))


def mask_from(safe_by_ratio):
    safe = [r for r in RATIOS if safe_by_ratio[r]]
    r_star = max(safe) if safe else 0.0
    allowed = [a for a, r in RATIO_OF_ACTION.items() if r <= r_star + 1e-12]
    masked = [a for a in RATIO_OF_ACTION if a not in allowed]
    return r_star, allowed, masked


def minmax(values):
    lo, hi = min(values), max(values)
    return [0.0] * len(values) if hi - lo <= 0 else [(v - lo) / (hi - lo) for v in values]


def main():
    model = common.load_baseline(os.path.join(ROOT, "checkpoints", "cnn_baseline_FIXED.pth"))
    splits = rl_env.Splits(os.path.join(ROOT, "data"))
    val_x, val_y = splits._val
    folds = [slice(k * FOLD, (k + 1) * FOLD) for k in range(N_FOLDS)]
    base_c, base_l = per_image(model, val_x, val_y)
    assert round(100 * base_c.mean(), 2) == BASELINE_VAL

    rows = []
    fold_stats = {}
    for layer in rl_env.FILTERED_LAYERS:
        for r in RATIOS:
            pruned = rl_env.apply_layer_pruning_by_name_inplace(copy.deepcopy(model), layer, r)
            c, l = per_image(pruned, val_x, val_y)
            drops, rel_loss = [], []
            for scope, sl in [("full", slice(None))] + [(f"fold{k}", s) for k, s in enumerate(folds)]:
                a0, a = 100 * base_c[sl].mean(), 100 * c[sl].mean()
                l0, l1 = base_l[sl].mean(), l[sl].mean()
                rows.append({"layer": layer, "probe_ratio": r, "scope": scope,
                             "n_images": len(base_c[sl]),
                             "val_accuracy": a, "baseline_val_accuracy": a0,
                             "accuracy_drop_pp": a0 - a, "relative_accuracy_drop": (a0 - a) / a0,
                             "val_loss": l1, "baseline_val_loss": l0,
                             "loss_increase": l1 - l0, "relative_loss_increase": (l1 - l0) / l0})
                if scope != "full":
                    drops.append(a0 - a)
                    rel_loss.append((l1 - l0) / l0)
            fold_stats[(layer, r)] = (drops, rel_loss)
    curves = pd.DataFrame(rows)
    curves.to_csv(os.path.join(RESULTS, "constrained_sensitivity_curves.csv"), index=False)

    # ---------------------------------------------------------- state values
    full = curves[curves["scope"] == "full"]
    acc_raw = [full[full["layer"] == n]["relative_accuracy_drop"].mean() for n in rl_env.FILTERED_LAYERS]
    loss_raw = [full[full["layer"] == n]["relative_loss_increase"].mean() for n in rl_env.FILTERED_LAYERS]
    state = pd.DataFrame({"layer": rl_env.FILTERED_LAYERS,
                          "accuracy_raw": acc_raw, "accuracy_normalized": minmax(acc_raw),
                          "loss_raw": loss_raw, "loss_normalized": minmax(loss_raw)})
    state.to_csv(os.path.join(RESULTS, "constrained_sensitivity_state.csv"), index=False)

    # ----------------------------------------------------------------- masks
    mask_rows = []
    for rule, idx, margin, unit in (("accuracy", 0, ACC_MARGIN_PP, "pp"),
                                     ("loss", 1, LOSS_MARGIN_REL, "relative")):
        for i, layer in enumerate(rl_env.FILTERED_LAYERS):
            bounds = {r: ucb(fold_stats[(layer, r)][idx]) for r in RATIOS}
            means = {r: float(np.mean(fold_stats[(layer, r)][idx])) for r in RATIOS}
            safe = {r: bounds[r] < margin for r in RATIOS}
            r_star, allowed, masked = mask_from(safe)
            mask_rows.append({
                "rule": rule, "layer": layer,
                "sensitivity_raw": (acc_raw if rule == "accuracy" else loss_raw)[i],
                "sensitivity_normalized": (minmax(acc_raw) if rule == "accuracy" else minmax(loss_raw))[i],
                "margin": margin, "margin_unit": unit,
                **{f"mean_fold_degradation_{int(100 * r)}": means[r] for r in RATIOS},
                **{f"ucb95_{int(100 * r)}": bounds[r] for r in RATIOS},
                **{f"safe_{int(100 * r)}": safe[r] for r in RATIOS},
                "safe_max_ratio": r_star,
                "safe_max_action": max(allowed),
                "allowed_actions": str(allowed),
                "allowed_ratios_pct": str([int(100 * RATIO_OF_ACTION[a]) for a in allowed]),
                "masked_actions": str(masked),
                "n_allowed": len(allowed),
                "non_monotone_safety": any(not safe[r] for r in RATIOS if r < r_star),
                "rule_text": (f"one-sided 95% t upper bound over 5 validation folds (t=2.132, df=4) of "
                              f"{'accuracy drop (pp)' if rule == 'accuracy' else 'relative loss increase'} "
                              f"< {margin:.4f}; allowed = ratios <= largest safe ratio"),
            })
    masks = pd.DataFrame(mask_rows)
    masks["computed_at"] = datetime.datetime.now().astimezone().isoformat(timespec="seconds")
    masks.to_csv(os.path.join(RESULTS, "sensitivity_action_masks.csv"), index=False)

    heuristic = {rule: [int(masks[(masks.rule == rule) & (masks.layer == n)]["safe_max_action"].iloc[0])
                        for n in rl_env.FILTERED_LAYERS] for rule in ("accuracy", "loss")}
    with open(os.path.join(RESULTS, "constrained_heuristic_policies.json"), "w", encoding="utf-8") as f:
        json.dump(heuristic, f, indent=1)

    pd.set_option("display.width", 250)
    print(full[["layer", "probe_ratio", "val_accuracy", "accuracy_drop_pp", "val_loss",
                "relative_loss_increase"]].round(4).to_string(index=False))
    print(masks[["rule", "layer", "sensitivity_normalized"] + [f"ucb95_{int(100 * r)}" for r in RATIOS]
                + ["safe_max_ratio", "allowed_actions"]].round(4).to_string(index=False))
    print(state.round(4).to_string(index=False))
    print("heuristic policies (C5):", heuristic)


if __name__ == "__main__":
    main()
