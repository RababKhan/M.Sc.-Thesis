"""Matched-sparsity baselines and paired significance tests.

Every comparison here holds total sparsity fixed and varies only how that
sparsity is allocated across layers, so the learned policy is compared against
alternatives that spend exactly the same weight budget.

Inputs  : checkpoints/cnn_baseline_FIXED.pth, the executed notebooks/Main code.ipynb
Outputs : results/ppo_multiseed.csv
          results/matched_sparsity_comparison.csv
          results/predictions/test_predictions.csv

Run from the repository root:  python evaluation/matched_sparsity_analysis.py
"""
import ast
import copy
import io
import json
import os
import sys

import pandas as pd
import torch
import torch.nn as nn
import torch.nn.utils.prune as prune

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CKPT = os.path.join(ROOT, "checkpoints", "cnn_baseline_FIXED.pth")
NB = os.path.join(ROOT, "notebooks", "Main code.ipynb")
RESULTS = os.path.join(ROOT, "results")

torch.set_num_threads(2)   # leave headroom for anything else running


# --------------------------------------------------------------- provenance
def ppo_results_from_notebook(path):
    """Pull the recorded PPO runs out of the executed notebook's stored outputs."""
    nb = json.load(io.open(path, encoding="utf-8"))
    found = {}
    for index, cell in enumerate(nb["cells"]):
        if cell["cell_type"] != "code":
            continue
        for output in cell.get("outputs", []):
            text = "".join(output.get("text", "")) if "text" in output else ""
            if not text and "data" in output:
                text = "".join(output["data"].get("text/plain", ""))
            for line in text.splitlines():
                start = line.find("{'Method'")
                if start == -1:
                    continue
                try:
                    record = ast.literal_eval(line[start:])
                except (ValueError, SyntaxError):
                    continue
                if "Actions" in record and record["Actions"] != "-":
                    record["source_cell"] = index
                    found[record["Method"]] = record
    return found


recorded = ppo_results_from_notebook(NB)
if not recorded:
    raise SystemExit("No PPO results found in the notebook - has it been executed?")

SEED_OF = {"PPO_Final_2000": 42, "PPO_Seed_1": 1, "PPO_Seed_2": 2, "PPO_Seed_3": 3}
runs = []
for method, record in recorded.items():
    if method not in SEED_OF:
        continue
    runs.append({
        "Seed": SEED_OF[method],
        "Method": method,
        "Actions": record["Actions"],
        "Prune fractions": str([common.ACTION_TO_PRUNE[a] for a in ast.literal_eval(record["Actions"])]),
        "Accuracy (%)": record["Accuracy (%)"],
        "Sparsity (%)": record["Sparsity (%)"],
        "Reward": record["Reward"],
        "Source": f"notebooks/Main code.ipynb cell {record['source_cell']}",
    })
runs.sort(key=lambda r: r["Seed"])
multiseed = pd.DataFrame(runs)

os.makedirs(RESULTS, exist_ok=True)
multiseed.to_csv(os.path.join(RESULTS, "ppo_multiseed.csv"), index=False)

print("== PPO runs recovered from the executed notebook ==")
print(multiseed.to_string(index=False))
identical = multiseed.duplicated(subset=["Actions", "Accuracy (%)"], keep=False)
if identical.any():
    dupes = multiseed.loc[identical, "Seed"].tolist()
    print(f"\n  NOTE: seeds {dupes} converged to the same policy and therefore produce "
          f"identical deterministic test accuracy. Reported as-is, not deduplicated.")

# ------------------------------------------------------------------- setup
model = common.load_baseline(CKPT)
loader = common.test_loader(root=os.path.join(ROOT, "data"))

base_preds, labels = common.predictions(model, loader)
base_correct = base_preds == labels
baseline_accuracy = common.accuracy_from(base_preds, labels)
print(f"\n== Baseline == {baseline_accuracy:.2f}% test accuracy, "
      f"{common.calculate_sparsity(model):.2f}% sparsity")

counts = common.layer_param_counts(model)
total_params = sum(counts.values())
print("\n== Prunable parameter budget ==")
for name, n in counts.items():
    marker = "" if name in common.FILTERED_LAYERS else "   (excluded from the action space)"
    print(f"  {name:<14} {n:>8,}  {100*n/total_params:5.2f}% of prunable weights{marker}")
print(f"  {'TOTAL':<14} {total_params:>8,}")
print(f"\n  classifier.1 alone holds {100*counts['classifier.1']/total_params:.1f}% of all prunable "
      f"weights, so total sparsity is dominated by that one layer's ratio.")

# --------------------------------------------------- self-check vs notebook
seed42 = multiseed[multiseed["Seed"] == 42].iloc[0]
seed42_actions = ast.literal_eval(seed42["Actions"])
check_model = common.prune_layerwise(model, [common.ACTION_TO_PRUNE[a] for a in seed42_actions])
check_sparsity = common.calculate_sparsity(check_model)
check_preds, _ = common.predictions(check_model, loader)
check_accuracy = common.accuracy_from(check_preds, labels)

print("\n== Self-check: this script vs the notebook's recorded seed-42 run ==")
print(f"  sparsity  script {check_sparsity:.6f}  notebook {seed42['Sparsity (%)']:.6f}")
print(f"  accuracy  script {check_accuracy:.2f}       notebook {seed42['Accuracy (%)']:.2f}")
assert abs(check_sparsity - seed42["Sparsity (%)"]) < 1e-6, "sparsity does not reproduce"
assert abs(check_accuracy - seed42["Accuracy (%)"]) < 1e-6, "accuracy does not reproduce"
print("  reproduces exactly")

# ------------------------------------------------------------ comparisons
rows = []
prediction_columns = {"true_label": labels.numpy(), "baseline": base_preds.numpy()}


def record(method, model_obj, actions, notes=""):
    preds, _ = common.predictions(model_obj, loader)
    accuracy = common.accuracy_from(preds, labels)
    sparsity = common.calculate_sparsity(model_obj)
    rows.append({
        "Method": method,
        "Actions": actions,
        "Accuracy (%)": round(accuracy, 2),
        "Sparsity (%)": sparsity,
        "Notes": notes,
    })
    return preds, (preds == labels), accuracy, sparsity


print("\n== Matched-sparsity comparison, per seed ==")
comparisons = []
for _, run in multiseed.iterrows():
    seed = int(run["Seed"])
    actions = ast.literal_eval(run["Actions"])
    target = run["Sparsity (%)"]

    ppo_model = common.prune_layerwise(model, [common.ACTION_TO_PRUNE[a] for a in actions])
    ppo_preds, ppo_correct, ppo_acc, ppo_sp = record(
        f"PPO seed {seed}", ppo_model, str(actions), "learned policy")
    prediction_columns[f"ppo_seed{seed}"] = ppo_preds.numpy()

    gm_model, gm_fraction = common.prune_global_to_total(model, target)
    gm_preds, gm_correct, gm_acc, gm_sp = record(
        f"Global magnitude @ {target:.2f}%", gm_model, "-",
        f"matched to PPO seed {seed}; {gm_fraction:.4f} of the four prunable layers")

    uni_model, uni_fraction = common.prune_uniform_to_total(model, target)
    uni_preds, uni_correct, uni_acc, uni_sp = record(
        f"Uniform per-layer @ {target:.2f}%", uni_model, "-",
        f"matched to PPO seed {seed}; {uni_fraction:.4f} per layer")

    if seed == 42:
        prediction_columns["global_magnitude_matched"] = gm_preds.numpy()
        prediction_columns["uniform_matched"] = uni_preds.numpy()

    print(f"\n  seed {seed}: policy {actions} at {ppo_sp:.2f}% sparsity")
    for label, acc, correct in (("global magnitude", gm_acc, gm_correct),
                                ("uniform per-layer", uni_acc, uni_correct)):
        b, c, chi2, p = common.mcnemar(ppo_correct, correct)
        verdict = "no significant difference" if p > 0.05 else "significant difference"
        print(f"    vs {label:<18} {ppo_acc:.2f}% vs {acc:.2f}%  "
              f"(delta {ppo_acc-acc:+.2f} pp)  discordant {b}/{c}  "
              f"chi2={chi2:.3f}  p={common.format_p(p)}  -> {verdict}")
        comparisons.append({
            "Seed": seed, "PPO policy": str(actions), "Sparsity (%)": round(ppo_sp, 4),
            "Baseline method": label, "PPO accuracy (%)": round(ppo_acc, 2),
            "Baseline accuracy (%)": round(acc, 2), "Delta (pp)": round(ppo_acc - acc, 2),
            "Discordant PPO-only": b, "Discordant baseline-only": c,
            "McNemar chi2": round(chi2, 4), "McNemar p": p,
            "Significant at 0.05": bool(p <= 0.05),
        })

# ------------------------------------------- reference (unmatched) baselines
print("\n== Reference baselines at their own sparsity levels ==")
for level, actions in ((0.2, [2, 2, 2, 2]), (0.4, [4, 4, 4, 4]), (0.6, [5, 5, 5, 5])):
    m = common.prune_layerwise(model, [level] * 4)
    _, _, acc, sp = record(f"Uniform per-layer {int(level*100)}%", m, str(actions),
                           "fixed action-space level, sparsity not matched to PPO")
    print(f"  uniform per-layer {int(level*100):>2}%  {acc:.2f}%  @ {sp:.2f}% sparsity")

# Global magnitude over ALL Conv2d/Linear layers, reproducing the notebook's
# original sweep (which also prunes classifier.3, unlike the matched baselines
# above). Kept as a separate row set so the two are never conflated.
for level in (0.1, 0.2, 0.4, 0.6, 0.8):
    m = copy.deepcopy(model)
    targets = [(mod, "weight") for _, mod in m.named_modules() if isinstance(mod, (nn.Conv2d, nn.Linear))]
    prune.global_unstructured(targets, pruning_method=prune.L1Unstructured, amount=level)
    _, _, acc, sp = record(f"Global magnitude {int(level*100)}%", m, "-",
                           "all Conv2d/Linear incl. classifier.3, as in the notebook sweep")
    print(f"  global magnitude {int(level*100):>2}%   {acc:.2f}%  @ {sp:.2f}% sparsity")

# ------------------------------------------------- per-layer allocation table
print("\n== Per-layer allocation at seed 42's sparsity ==")
ppo42 = common.prune_layerwise(model, [common.ACTION_TO_PRUNE[a] for a in seed42_actions])
gm42, _ = common.prune_global_to_total(model, seed42["Sparsity (%)"])
uni42, _ = common.prune_uniform_to_total(model, seed42["Sparsity (%)"])
ppo_layers, gm_layers, uni_layers = (common.per_layer_sparsity(x) for x in (ppo42, gm42, uni42))

allocation = pd.DataFrame([
    {"Layer": name,
     "Params": counts[name],
     "Share of prunable (%)": round(100 * counts[name] / total_params, 2),
     "PPO (%)": round(ppo_layers[name], 1),
     "Global magnitude (%)": round(gm_layers[name], 1),
     "Uniform (%)": round(uni_layers[name], 1)}
    for name in counts
])
print(allocation.to_string(index=False))

# ------------------------------------------------------------------ outputs
comparison_df = pd.DataFrame(comparisons)
comparison_df.to_csv(os.path.join(RESULTS, "matched_sparsity_comparison.csv"), index=False)

results_df = pd.DataFrame(rows).drop_duplicates(subset=["Method"])
results_df.to_csv(os.path.join(RESULTS, "matched_sparsity_methods.csv"), index=False)

allocation.to_csv(os.path.join(RESULTS, "per_layer_allocation.csv"), index=False)

os.makedirs(os.path.join(RESULTS, "predictions"), exist_ok=True)
pd.DataFrame(prediction_columns).to_csv(
    os.path.join(RESULTS, "predictions", "test_predictions.csv"), index=False)

print(f"\nWrote:")
for name in ("ppo_multiseed.csv", "matched_sparsity_comparison.csv",
             "matched_sparsity_methods.csv", "per_layer_allocation.csv",
             os.path.join("predictions", "test_predictions.csv")):
    print(f"  results/{name}")
