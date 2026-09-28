"""Phase-1 audit: read-only fact collection (no training, no result is modified).

Collects, from code and data rather than memory:
  * data split integrity (45k/5k/10k; V_RL/V_SELECT partition)
  * checkpoint inventory: sha256, size, git status; architecture, parameter
    count, validation and test accuracy of every CNN checkpoint (inference only)
  * the PPO configuration, read from instantiated (untrained) PPO/MaskablePPO objects
  * run-record inventory per experiment and pre-registration provenance
  * duplicated definitions across the codebase

Output: results/rocksolid_audit_data.json
Run from the repository root:  python evaluation/audit_phase1.py
"""
import glob
import hashlib
import json
import os
import re
import subprocess
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common  # noqa: E402
import rl_env  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
R = os.path.join(ROOT, "results")


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def git(*args):
    return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True).stdout.strip()


def main():
    out = {}
    torch.set_num_threads(4)

    # ------------------------------------------------ data protocol
    splits = rl_env.Splits(os.path.join(ROOT, "data"))
    tr, va = set(splits.train_indices), set(splits.val_indices)
    test_loader = splits.test_loader()
    n_test = len(test_loader.dataset)
    npz = np.load(os.path.join(R, "archive_validation_split.npz"))
    rl, sel = set(npz["v_rl_positions"].tolist()), set(npz["v_select_positions"].tolist())
    val_labels = splits._val[1].numpy()
    out["data"] = {
        "train": len(tr), "validation": len(va), "test": n_test,
        "train_val_disjoint": not (tr & va), "train_plus_val": len(tr | va),
        "val_indices_sha256": hashlib.sha256(json.dumps(splits.val_indices).encode()).hexdigest(),
        "rl_probe_is_first_1000_of_validation": splits.probe_indices == splits.val_indices[:1000],
        "V_RL": len(rl), "V_SELECT": len(sel), "V_RL_V_SELECT_disjoint": not (rl & sel),
        "V_RL_union_V_SELECT_is_validation": (rl | sel) == set(range(5000)),
        "V_RL_cifar_indices_subset_of_validation": set(npz["v_rl_cifar_train_indices"].tolist()) <= va,
        "V_SELECT_cifar_indices_subset_of_validation": set(npz["v_select_cifar_train_indices"].tolist()) <= va,
        "V_RL_labels_match": bool(np.array_equal(val_labels[npz["v_rl_positions"]], npz["v_rl_labels"])),
        "class_counts_validation": np.bincount(val_labels, minlength=10).tolist(),
        "class_counts_V_RL": np.bincount(npz["v_rl_labels"], minlength=10).tolist(),
        "class_counts_V_SELECT": np.bincount(npz["v_select_labels"], minlength=10).tolist(),
        "probe_overlap_with_V_SELECT": len(set(range(1000)) & sel),
        "probe_overlap_with_V_RL": len(set(range(1000)) & rl),
    }

    # ------------------------------------------------ checkpoints
    cks = []
    for p in sorted(glob.glob(os.path.join(ROOT, "checkpoints", "*.*"))):
        name = os.path.relpath(p, ROOT).replace("\\", "/")
        row = {"file": name, "bytes": os.path.getsize(p), "sha256": sha(p),
               "git_tracked": bool(git("ls-files", name)), "git_modified": bool(git("status", "--porcelain", name)),
               "last_commit": git("log", "-1", "--format=%h %ad %s", "--date=short", "--", name)}
        if p.endswith(".pth"):
            m = common.load_baseline(p)
            row["parameters_total"] = int(sum(t.numel() for t in m.parameters()))
            row["prunable_weights"] = int(sum(mod.weight.numel() for _, mod in rl_env.prunable_layers(m)[0]))
            row["val_accuracy"] = rl_env.evaluate_model(m, splits.val_loader())
            row["test_accuracy"] = rl_env.evaluate_model(m, test_loader)
        cks.append(row)
    for d in sorted(glob.glob(os.path.join(ROOT, "checkpoints", "*/"))):
        files = glob.glob(os.path.join(d, "**", "*"), recursive=True)
        cks.append({"directory": os.path.relpath(d, ROOT).replace("\\", "/"), "files": len(files),
                    "bytes": sum(os.path.getsize(f) for f in files if os.path.isfile(f))})
    out["checkpoints"] = cks

    # ------------------------------------------------ PPO configuration (instantiated, not trained)
    from stable_baselines3 import PPO
    import stable_baselines3
    import sb3_contrib
    import ppo_experiments as base
    model = common.load_baseline(os.path.join(ROOT, "checkpoints", "cnn_baseline_FIXED.pth"))
    all_layers, filtered = rl_env.prunable_layers(model)
    env = rl_env.CompressionEnvGym(model, filtered, 77.32, splits.probe_loader(),
                                   max(mm.weight.numel() for _, mm in all_layers))
    agent = PPO(env=env, seed=0, **base.PPO_KWARGS)
    out["ppo"] = {
        "declared_kwargs": {k: v for k, v in base.PPO_KWARGS.items()},
        "total_timesteps_requested": base.TOTAL_TIMESTEPS,
        "learning_rate": agent.learning_rate, "n_steps": agent.n_steps, "batch_size": agent.batch_size,
        "n_epochs": agent.n_epochs, "gamma": agent.gamma, "gae_lambda": agent.gae_lambda,
        "clip_range": float(agent.clip_range(1.0)), "clip_range_vf": agent.clip_range_vf,
        "ent_coef": agent.ent_coef, "vf_coef": agent.vf_coef, "max_grad_norm": agent.max_grad_norm,
        "normalize_advantage": agent.normalize_advantage, "target_kl": agent.target_kl, "use_sde": agent.use_sde,
        "policy_architecture": str(agent.policy.mlp_extractor).replace("\n", " "),
        "action_net": str(agent.policy.action_net), "value_net": str(agent.policy.value_net),
        "activation": agent.policy.activation_fn.__name__, "ortho_init": agent.policy.ortho_init,
        "optimizer": type(agent.policy.optimizer).__name__,
        "observation_space": str(env.observation_space), "action_space": str(env.action_space),
        "action_to_prune": {str(k): v for k, v in rl_env.ACTION_TO_PRUNE.items()},
        "prunable_layers": rl_env.FILTERED_LAYERS,
        "reward_coefficients": {"accuracy": rl_env.ACCURACY_COEF, "sparsity_default": rl_env.DEFAULT_SPARSITY_COEF,
                                "diversity": rl_env.DIVERSITY_COEF},
        "stable_baselines3": stable_baselines3.__version__, "sb3_contrib": sb3_contrib.__version__,
    }

    # ------------------------------------------------ run records and pre-registrations
    runs = {}
    for folder in ["runs", "refined_runs", "constrained_runs", "soft_prior_runs", "exploration_levelA_runs"]:
        recs = [json.load(open(p, encoding="utf-8")) for p in glob.glob(os.path.join(R, folder, "*.json"))]
        runs[folder] = {"records": len(recs), "seeds": sorted({r["seed"] for r in recs}),
                        "preregistration_sha256": sorted({str(r.get("preregistration_sha256", ""))[:16] for r in recs})}
    arc = [json.load(open(p, encoding="utf-8")) for p in glob.glob(os.path.join(R, "archive_runs", "*", "seed_*_run.json"))]
    runs["archive_runs"] = {"records": len(arc), "seeds": sorted({r["seed"] for r in arc}),
                            "preregistration_sha256": sorted({r["preregistration_sha256"][:16] for r in arc})}
    out["run_records"] = runs
    pre = {}
    for p in sorted(glob.glob(os.path.join(R, "*preregistration*.md"))):
        name = os.path.relpath(p, ROOT).replace("\\", "/")
        pre[name] = {"sha256": sha(p), "added_in": git("log", "--diff-filter=A", "--format=%h %ad", "--date=short", "--", name),
                     "modified_since_added": bool(git("log", "--format=%h", "--diff-filter=M", "--", name))}
    out["preregistrations"] = pre

    # ------------------------------------------------ duplicated definitions
    patterns = {"class SimpleCNN": r"class SimpleCNN", "def evaluate_model": r"def evaluate_model",
                "def compute_reward / reward_components": r"def (compute_reward|reward_components)",
                "def calculate_sparsity": r"def calculate_sparsity",
                "sensitivity estimators": r"def (compute_layer_sensitivities|definitions_from|cmd_probe|cmd_prepare)\b|refined_sensitivity|mean_loss_acc",
                "exact sign-flip tests": r"def sign_flip_p", "Holm": r"def holm", "t-quantile tables": r"T975\s*=",
                "PPO environments": r"class \w*CompressionEnv\w*"}
    files = glob.glob(os.path.join(ROOT, "evaluation", "*.py")) + glob.glob(os.path.join(ROOT, "notebooks", "*.ipynb"))
    dup = {}
    for label, pat in patterns.items():
        hits = [os.path.relpath(f, ROOT).replace("\\", "/") for f in files
                if re.search(pat, open(f, encoding="utf-8").read())]
        dup[label] = hits
    out["definitions"] = dup

    with open(os.path.join(R, "rocksolid_audit_data.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1, default=str)
    print(json.dumps(out, indent=1, default=str))


if __name__ == "__main__":
    main()
