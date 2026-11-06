"""Does layer sensitivity predict pruning tolerance?  (validation only)

Uses results/policy_landscape_validation.csv (all 1,296 policies on the
validation split) and the single-layer sensitivity values/masks.

  A  Among high-performing policies, are sensitive layers pruned less?
     High-performing = (i) validation Pareto-efficient; (ii) top 10% by
     validation accuracy within 5-pp sparsity bins (controls for sparsity).
  B  Is sensitivity negatively associated with tolerated pruning?
     In-context tolerance: for each layer and ratio, the mean validation
     accuracy damage over all 216 configurations of the other three layers
     (the layer's average marginal effect); tolerated ratio = largest ratio
     whose mean damage is <= 1.0 pp. This is independent of layer size.
  C  Do Pareto-efficient policies protect sensitive layers more than dominated ones?
  D  Does sensitivity rank correlate with high-quality policies' pruning ratios?
     Spearman over the four layers (exact permutation p over 24 orderings).
  E  Mask-level validity: best validation accuracy reachable inside the
     correct mask vs inside each shuffled mask, per sparsity band, and the
     share of Pareto-efficient policies each mask admits.

Confound: in this network the sensitivity order equals the reverse of the
layer-size order, so any analysis driven by total sparsity (A-i, C, D) cannot
separate sensitivity from size. B and A-ii are the size-independent tests.

Outputs: results/sensitivity_predictive_validity.csv (long format)
Run from the repository root:  python evaluation/predictive_validity.py
"""
import itertools
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import rl_env  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(ROOT, "results")
LAYERS = rl_env.FILTERED_LAYERS
RATIOS = [0, 10, 20, 30, 40, 60]
PARAMS = {"features.0": 864, "features.3": 18432, "features.6": 73728, "classifier.1": 524288}


def spearman(a, b):
    ra, rb = pd.Series(a).rank().to_numpy(), pd.Series(b).rank().to_numpy()
    if ra.std() == 0 or rb.std() == 0:
        return float("nan")
    return float(np.corrcoef(ra, rb)[0, 1])


def exact_p(a, b):
    """Two-sided exact permutation p for Spearman over all orderings of b."""
    obs = spearman(a, b)
    if np.isnan(obs):
        return float("nan")
    rhos = [spearman(a, p) for p in itertools.permutations(b)]
    return float(np.mean([abs(r) >= abs(obs) - 1e-12 for r in rhos]))


def main():
    L = pd.read_csv(os.path.join(RESULTS, "policy_landscape_validation.csv"))
    state = pd.read_csv(os.path.join(RESULTS, "constrained_sensitivity_state.csv")).set_index("layer")
    masks = pd.read_csv(os.path.join(RESULTS, "sensitivity_action_masks.csv"))
    sens = {"accuracy": [state.loc[n, "accuracy_raw"] for n in LAYERS],
            "loss": [state.loc[n, "loss_raw"] for n in LAYERS]}
    size = [PARAMS[n] for n in LAYERS]
    rows = []

    def add(question, analysis, **kv):
        rows.append({"question": question, "analysis": analysis, **kv})

    add("confound", "spearman(sensitivity_accuracy, parameter_count)", value=spearman(sens["accuracy"], size))
    add("confound", "spearman(sensitivity_loss, parameter_count)", value=spearman(sens["loss"], size))

    # ---- A: high-performing sets
    L["sparsity_bin"] = (L["total_sparsity"] // 5).astype(int)
    top = L.groupby("sparsity_bin", group_keys=False).apply(
        lambda g: g[g["val_accuracy"] >= g["val_accuracy"].quantile(0.9)] if len(g) >= 10 else g.iloc[0:0])
    sets = {"all": L, "pareto": L[L["pareto_val_accuracy"]], "dominated": L[~L["pareto_val_accuracy"]],
            "top10pct_within_sparsity_bin": top}
    for name, s in sets.items():
        means = [s[f"ratio_{n}"].mean() for n in LAYERS]
        for n, m in zip(LAYERS, means):
            add("A/C", f"mean pruning ratio (%) in set '{name}'", layer=n, value=m, n_policies=len(s))
        for d in ("accuracy", "loss"):
            add("D", f"spearman(sensitivity_{d}, mean ratio in '{name}')",
                value=spearman(sens[d], means), exact_p=exact_p(sens[d], means), n_policies=len(s))
        add("D", f"spearman(parameter_count, mean ratio in '{name}')", value=spearman(size, means),
            exact_p=exact_p(size, means), n_policies=len(s))

    # ---- A-ii detail: within-bin top policies, per bin
    for b, g in top.groupby("sparsity_bin"):
        for n in LAYERS:
            add("A-ii", f"sparsity bin {5 * b}-{5 * b + 5}%: mean ratio of top-10% policies", layer=n,
                value=g[f"ratio_{n}"].mean(), n_policies=len(g))

    # ---- B: in-context marginal damage per layer (size-independent)
    tolerance = {}
    for n in LAYERS:
        others = [m for m in LAYERS if m != n]
        base = L[L[f"ratio_{n}"] == 0].set_index([f"ratio_{m}" for m in others])["val_accuracy"]
        tol = 0
        for r in RATIOS[1:]:
            at = L[L[f"ratio_{n}"] == r].set_index([f"ratio_{m}" for m in others])["val_accuracy"]
            damage = (base - at.reindex(base.index)).mean()
            worst = (base - at.reindex(base.index)).max()
            add("B", "mean in-context validation accuracy damage (pp) over other-layer configurations",
                layer=n, ratio=r, value=float(damage), worst_case=float(worst), n_policies=len(at))
            if damage <= 1.0:
                tol = r
        tolerance[n] = tol
        add("B", "in-context tolerated ratio (largest ratio with mean damage <= 1.0 pp)", layer=n, value=tol)
    tol_vec = [tolerance[n] for n in LAYERS]
    for d in ("accuracy", "loss"):
        add("B", f"spearman(sensitivity_{d}, in-context tolerated ratio)", value=spearman(sens[d], tol_vec),
            exact_p=exact_p(sens[d], tol_vec))
    rule_tol = {r: [masks[(masks.rule == r) & (masks.layer == n)]["safe_max_ratio"].iloc[0] * 100 for n in LAYERS]
                for r in ("accuracy", "loss")}
    for r in ("accuracy", "loss"):
        agree = sum(a == b for a, b in zip(rule_tol[r], tol_vec))
        add("B", f"single-layer {r} mask max ratio vs in-context tolerated ratio: layers agreeing",
            value=agree, detail=json.dumps({"mask": rule_tol[r], "in_context": tol_vec}))

    # ---- C: Pareto vs dominated, per layer
    for n in LAYERS:
        p, d_ = sets["pareto"][f"ratio_{n}"], sets["dominated"][f"ratio_{n}"]
        add("C", "mean ratio Pareto minus dominated (pp)", layer=n, value=float(p.mean() - d_.mean()))

    # ---- D: per-policy correlation distribution
    for name, s in sets.items():
        for d in ("accuracy", "loss"):
            rhos = [spearman(sens[d], [row[f"ratio_{n}"] for n in LAYERS]) for _, row in s.iterrows()]
            rhos = [x for x in rhos if not np.isnan(x)]
            add("D", f"per-policy spearman(sensitivity_{d}, ratios): mean over '{name}'",
                value=float(np.mean(rhos)), n_policies=len(rhos))

    # ---- E: mask-level validity
    for rule in ("accuracy", "loss"):
        m = masks[masks.rule == rule].set_index("layer").loc[LAYERS]
        allowed = [set(json.loads(a)) for a in m["allowed_actions"]]
        perms = [p for p in itertools.permutations(range(4))
                 if all(allowed[p[i]] != allowed[i] for i in range(4))]
        variants = {"correct": allowed}
        seen = set()
        for p in perms:
            key = tuple(tuple(sorted(allowed[p[i]])) for i in range(4))
            if key not in seen:
                seen.add(key)
                variants[f"shuffled#{len(seen)}"] = [allowed[p[i]] for i in range(4)]
        for vname, var in variants.items():
            inside = L[[all(a in var[i] for i, a in enumerate(json.loads(x))) for x in L["actions"]]]
            add("E", f"{rule} mask '{vname}': policies admitted", value=len(inside),
                detail=json.dumps([sorted(v) for v in var]))
            add("E", f"{rule} mask '{vname}': Pareto-efficient policies admitted",
                value=int(inside["pareto_val_accuracy"].sum()), n_policies=int(L["pareto_val_accuracy"].sum()))
            add("E", f"{rule} mask '{vname}': max reachable sparsity", value=float(inside["total_sparsity"].max()))
            for coef in ("0.01", "0.02"):
                bestr = inside.loc[inside[f"reward_{coef}"].idxmax()]
                add("E", f"{rule} mask '{vname}': reward-optimal policy inside mask (lambda {coef})",
                    value=float(bestr[f"reward_{coef}"]), detail=bestr["actions"],
                    val_accuracy=float(bestr["val_accuracy"]), sparsity=float(bestr["total_sparsity"]),
                    global_rank=int(bestr[f"reward_rank_{coef}"]))
            for lo in range(0, 60, 5):
                band = inside[(inside["total_sparsity"] >= lo) & (inside["total_sparsity"] < lo + 5)]
                if len(band):
                    add("E", f"{rule} mask '{vname}': best val accuracy in sparsity band",
                        ratio=lo, value=float(band["val_accuracy"].max()), n_policies=len(band))

    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(RESULTS, "sensitivity_predictive_validity.csv"), index=False)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_colwidth", 90)
    pd.set_option("display.max_rows", 400)
    print(out[out.question != "A-ii"].drop(columns=["detail"], errors="ignore").round(4).to_string(index=False))


if __name__ == "__main__":
    main()
