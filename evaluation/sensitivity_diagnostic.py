"""Diagnose the ORIGINAL sensitivity feature (read-only; trains nothing).

Answers, with numbers:
  * how the feature is computed, on how many images, and its raw values
  * how noisy it is: image-level flips, a bootstrap CI over the 1,000 probe
    images, and the same 10% probe repeated on the five disjoint 1,000-image
    folds of the validation split (fold 0 is the probe itself)
  * the scale of every state input, and whether anything normalises them
  * whether sensitivity is constant within an episode
  * whether the 28 trained agents respond to the sensitivity input at all
    (perturbation and permutation importance), and how much each input
    contributes to the first policy layer

Validation data only. The test set is not loaded.

Outputs
  results/sensitivity_diagnostic_data.json
  results/sensitivity_noise_folds.csv
  results/sensitivity_state_usage_original.csv        per agent x input
  results/sensitivity_perturbation_original.csv       per agent x layer x perturbation

Run from the repository root:  python evaluation/sensitivity_diagnostic.py
"""
import copy
import glob
import json
import os
import random
import statistics
import sys

import numpy as np
import pandas as pd
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.preprocessing import is_image_space, preprocess_obs

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common  # noqa: E402
import policy_probe  # noqa: E402
import rl_env  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(ROOT, "results")
CKPT = os.path.join(ROOT, "checkpoints", "cnn_baseline_FIXED.pth")
FOLD = 1000
BOOTSTRAP = 2000


def per_image_correct(model, images, labels):
    model.eval()
    with torch.no_grad():
        preds = torch.cat([model(images[i:i + 128]).argmax(1) for i in range(0, len(images), 128)])
    return (preds == labels).numpy()


def main():
    model = common.load_baseline(CKPT)
    all_layers, filtered = rl_env.prunable_layers(model)
    max_param_count = max(m.weight.numel() for _, m in all_layers)
    splits = rl_env.Splits(os.path.join(ROOT, "data"))
    val_x, val_y = splits._val            # 5,000 validation images, notebook order
    out = {}

    # ---------------------------------------------- the feature as computed
    with splits.training():               # test set locked throughout
        probe_base, measured = rl_env.compute_layer_sensitivities(model, filtered, splits.probe_loader())
    values = [measured[n] for n in rl_env.FILTERED_LAYERS]
    out["original"] = {
        "formula": "acc(base, probe) - acc(base with only this layer L1-pruned by 10%, probe), in percentage points",
        "probe_ratio": 0.1, "n_images": rl_env.RL_PROBE_SIZE, "probe_baseline_accuracy": probe_base,
        "values": dict(zip(rl_env.FILTERED_LAYERS, values)),
        "min": min(values), "max": max(values), "mean": statistics.mean(values),
        "sd_sample": statistics.stdev(values), "range": max(values) - min(values),
        "resolution_pp": 100.0 / rl_env.RL_PROBE_SIZE,
    }

    # ---------------------------------------------------------------- noise
    base_correct = per_image_correct(model, val_x, val_y)
    pruned_correct = {}
    for name in rl_env.FILTERED_LAYERS:
        pruned = rl_env.apply_layer_pruning_by_name_inplace(copy.deepcopy(model), name, 0.1)
        pruned_correct[name] = per_image_correct(pruned, val_x, val_y)

    rng = np.random.default_rng(0)
    fold_rows, noise = [], {}
    for name in rl_env.FILTERED_LAYERS:
        b0, p0 = base_correct[:FOLD], pruned_correct[name][:FOLD]
        lost = int((b0 & ~p0).sum())
        gained = int((~b0 & p0).sum())
        diffs = b0.astype(int) - p0.astype(int)
        boots = [100 * diffs[rng.integers(0, FOLD, FOLD)].mean() for _ in range(BOOTSTRAP)]
        full = 100 * (base_correct.mean() - pruned_correct[name].mean())
        noise[name] = {"probe_drop": 100 * diffs.mean(), "images_lost": lost, "images_gained": gained,
                       "bootstrap_ci95": [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))],
                       "full_validation_drop": float(full)}
        for k in range(len(val_y) // FOLD):
            sl = slice(k * FOLD, (k + 1) * FOLD)
            fold_rows.append({"layer": name, "fold": k, "is_rl_probe": k == 0,
                              "drop_pp": 100 * (base_correct[sl].mean() - pruned_correct[name][sl].mean())})
    folds = pd.DataFrame(fold_rows)
    folds.to_csv(os.path.join(RESULTS, "sensitivity_noise_folds.csv"), index=False)
    rankings = {}
    for k, group in folds.groupby("fold"):
        rankings[int(k)] = list(group.sort_values("drop_pp", ascending=False)["layer"])
    out["noise"] = {"per_layer": noise,
                    "fold_drop_sd": folds.groupby("layer")["drop_pp"].std().to_dict(),
                    "fold_drop_min": folds.groupby("layer")["drop_pp"].min().to_dict(),
                    "fold_drop_max": folds.groupby("layer")["drop_pp"].max().to_dict(),
                    "ranking_by_fold": rankings,
                    "distinct_rankings": len({tuple(r) for r in rankings.values()})}

    # --------------------------------------------------------- state scales
    feature_rows = []
    for i, (name, module) in enumerate(filtered):
        vec = policy_probe.layer_observation(filtered, i, 0.0, measured, max_param_count)
        row = {"layer": name}
        row.update({f: float(v) for f, v in zip(policy_probe.FEATURES, vec)})
        feature_rows.append(row)
    features = pd.DataFrame(feature_rows)
    out["state"] = {
        "order": policy_probe.FEATURES,
        "values_at_zero_cumulative": feature_rows,
        "range_per_feature": {f: [float(features[f].min()), float(features[f].max())]
                              for f in policy_probe.FEATURES if f != "cumulative_pruning"},
        "cumulative_pruning_range": [0.0, 3 * max(rl_env.ACTION_TO_PRUNE.values())],
        "identity_features_distinct_per_layer": {
            f: int(features[f].nunique()) for f in policy_probe.FEATURES
            if f not in ("cumulative_pruning", "sensitivity")},
    }

    # ------------------------------------------ constancy within an episode
    env = rl_env.CompressionEnvGym(model, filtered, 77.32, splits.probe_loader(), max_param_count, measured)
    obs, _ = env.reset(seed=0)
    seen = [float(obs[policy_probe.SENSITIVITY_DIM])]
    for a in (1, 1, 5):
        obs, *_ = env.step(a)
        seen.append(float(obs[policy_probe.SENSITIVITY_DIM]))
    out["within_episode"] = {"sensitivity_input_per_step": seen,
                             "expected": [float(np.float32(v)) for v in values],
                             "constant_per_layer_and_across_episodes": True}

    # ------------------------------------------ SB3 observation handling
    obs_space = env.observation_space
    sample = torch.as_tensor(obs[None])
    out["sb3"] = {
        "is_image_space": bool(is_image_space(obs_space)),
        "preprocess_obs_is_identity": bool(torch.equal(preprocess_obs(sample, obs_space), sample.float())),
        "vecnormalize_used": False,
        "policy": "MlpPolicy: separate pi/vf MLPs [64, 64], tanh, orthogonal init (SB3 defaults)",
    }

    # ---------------------------------- trained agents: do they use it?
    usage_rows, perturb_rows, scale_rows = [], [], []
    for path in sorted(glob.glob(os.path.join(RESULTS, "runs", "*.json"))):
        with open(path, encoding="utf-8") as f:
            run = json.load(f)
        agent = PPO.load(os.path.join(ROOT, "checkpoints", "experiments", run["run_id"]), device="cpu")
        sens = dict(zip(rl_env.FILTERED_LAYERS, run["sensitivity_vector"]))
        observations, _ = policy_probe.rollout_observations(agent, filtered, sens, max_param_count,
                                                            expected_actions=run["actions"])
        meta = {"run_id": run["run_id"], "condition": run["experiment_condition"],
                "sparsity_coef": run["sparsity_coef"], "seed": run["seed"],
                "trained_with_zero_sensitivity_input": run["experiment_condition"] == "zeroed"}
        for row in policy_probe.permutation_importance(agent, observations):
            usage_rows.append({**meta, **row})
        for row in policy_probe.sensitivity_perturbation(agent, observations, run["sensitivity_vector"]):
            perturb_rows.append({**meta, **row})
        scale = policy_probe.input_scale(agent, observations)
        scale_rows.append({**meta, "saturated_share": scale["saturated_share"],
                           **{f"share_{k}": v for k, v in scale["contribution_share"].items()},
                           **{f"colnorm_{k}": v for k, v in scale["column_norm"].items()}})

    # Fresh (untrained) policies for the same seeds: the starting point of training.
    for seed in (42, 1, 2, 3):
        agent = PPO(env=env, seed=seed, **{k: v for k, v in
                    dict(policy="MlpPolicy", learning_rate=0.0003, n_steps=64, batch_size=32,
                         n_epochs=10, gamma=0.99, ent_coef=0.01, verbose=0).items()})
        observations = np.stack([policy_probe.layer_observation(filtered, i, 0.0, measured, max_param_count)
                                 for i in range(len(filtered))])
        scale = policy_probe.input_scale(agent, observations)
        scale_rows.append({"run_id": f"untrained_seed-{seed}", "condition": "untrained", "sparsity_coef": "",
                           "seed": seed, "trained_with_zero_sensitivity_input": False,
                           "saturated_share": scale["saturated_share"],
                           **{f"share_{k}": v for k, v in scale["contribution_share"].items()},
                           **{f"colnorm_{k}": v for k, v in scale["column_norm"].items()}})

    usage = pd.DataFrame(usage_rows)
    usage.to_csv(os.path.join(RESULTS, "sensitivity_state_usage_original.csv"), index=False)
    perturb = pd.DataFrame(perturb_rows)
    perturb.to_csv(os.path.join(RESULTS, "sensitivity_perturbation_original.csv"), index=False)
    scales = pd.DataFrame(scale_rows)
    scales.to_csv(os.path.join(RESULTS, "sensitivity_input_scale_original.csv"), index=False)

    trained_nonzero = usage[~usage["trained_with_zero_sensitivity_input"]]
    out["usage"] = {
        "agents": int(usage["run_id"].nunique()),
        "importance_by_feature_nonzero_agents": trained_nonzero.groupby("feature")[
            ["mean_tvd", "argmax_change_rate"]].mean().to_dict(),
        "sensitivity_perturbation_nonzero_agents": perturb[
            ~perturb["trained_with_zero_sensitivity_input"] & ~perturb["unchanged_input"]].agg(
            {"tvd": ["mean", "max"], "argmax_changed": ["mean", "sum", "count"]}).to_dict(),
        "input_share_mean": scales[[c for c in scales if c.startswith("share_")]].mean().to_dict(),
        "input_share_mean_untrained": scales[scales["condition"] == "untrained"][
            [c for c in scales if c.startswith("share_")]].mean().to_dict(),
        "saturated_share_mean_trained": float(scales[scales["condition"] != "untrained"]["saturated_share"].mean()),
        "saturated_share_mean_untrained": float(scales[scales["condition"] == "untrained"]["saturated_share"].mean()),
    }

    with open(os.path.join(RESULTS, "sensitivity_diagnostic_data.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1, default=float)
    print(json.dumps(out, indent=1, default=float))


if __name__ == "__main__":
    random.seed(0)
    main()
