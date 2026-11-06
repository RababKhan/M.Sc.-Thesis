"""Sensitivity ablation and reward sparsity-coefficient sweep for the PPO agent.

Every run trains one PPO agent with the exact settings of notebooks/Main code.ipynb
(cells 49-51 and 63) and changes exactly one variable:

  ablation   the sensitivity feature in the state:
               correct  - the measured per-layer sensitivities
               zeroed   - 0.0 for every layer (state dimension unchanged)
               shuffled - the measured values permuted among layers by a
                          seed-deterministic derangement, so every layer gets
                          another layer's value; the mapping is recorded
  sweep      the sparsity coefficient lambda_s in
               R = 1.5*(acc/base) + lambda_s*S + 0.05*D,  S in percent (0-100)

Seeds 42, 1, 2, 3 for every cell. The ablation's "correct" condition and the
sweep's lambda_s = 0.04 cell are the same configuration, so they share runs.
Those four runs are also a regression test: they must reproduce the policies,
sparsities and test accuracies recorded in the notebook, or the script stops.

Data protocol
  reward during training   1,000-image validation probe (as in the notebook)
  validation accuracy      full 5,000-image validation split, after training
  test accuracy            10,000-image test set, once, after training
The test set is locked while agent.learn() runs (see rl_env.Splits).

Kept exactly as in the notebook, deliberately: the training reward divides
probe accuracy by the baseline's FULL-validation accuracy (77.32%), not by its
probe accuracy (76.1%). Changing that would change the verified experiment.

Outputs (one file per run, so the script can be interrupted and resumed)
  results/runs/<run_id>.json
  results/training_curves/<run_id>.csv
  checkpoints/experiments/<run_id>.zip

Run from the repository root:
  python evaluation/ppo_experiments.py                  # everything, resumable
  python evaluation/ppo_experiments.py --only reproduce # the 4 regression runs
"""
import argparse
import datetime
import hashlib
import json
import os
import platform
import subprocess
import sys
import time

import numpy as np
import pandas as pd
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common  # noqa: E402
import rl_env  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CKPT = os.path.join(ROOT, "checkpoints", "cnn_baseline_FIXED.pth")
RUNS_DIR = os.path.join(ROOT, "results", "runs")
CURVES_DIR = os.path.join(ROOT, "results", "training_curves")
AGENTS_DIR = os.path.join(ROOT, "checkpoints", "experiments")

SEEDS = [42, 1, 2, 3]
CONDITIONS = ["correct", "zeroed", "shuffled"]
COEFFICIENTS = [0.00, 0.01, 0.02, 0.04, 0.08]

# Notebook cells 50 and 63, unchanged for every run.
PPO_KWARGS = dict(policy="MlpPolicy", learning_rate=0.0003, n_steps=64, batch_size=32,
                  n_epochs=10, gamma=0.99, ent_coef=0.01, verbose=0)
TOTAL_TIMESTEPS = 2000

# Recorded by the notebook (cells 51 and 63) and verified in results/ppo_multiseed.csv.
RECORDED = {
    42: {"actions": [1, 1, 5, 5], "sparsity": 58.19572427856073, "test": 76.49},
    1: {"actions": [0, 0, 5, 5], "sparsity": 57.884530999948375, "test": 76.65},
    2: {"actions": [0, 5, 5, 5], "sparsity": 59.66860900314904, "test": 75.16},
    3: {"actions": [0, 5, 5, 5], "sparsity": 59.66860900314904, "test": 75.16},
}
BASELINE_VAL, BASELINE_TEST = 77.32, 77.03


# --------------------------------------------------------------- provenance
def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def git_state():
    def run(*args):
        return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    return {"commit": run("rev-parse", "HEAD"), "dirty": bool(run("status", "--porcelain"))}


def cpu_name():
    if sys.platform == "win32":
        import winreg
        key = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE,
                             r"HARDWARE\DESCRIPTION\System\CentralProcessor\0")
        return winreg.QueryValueEx(key, "ProcessorNameString")[0].strip()
    return platform.processor()


def environment():
    import gymnasium
    import stable_baselines3
    import torchvision
    return {
        "python": platform.python_version(), "torch": torch.__version__,
        "torchvision": torchvision.__version__, "gymnasium": gymnasium.__version__,
        "stable_baselines3": stable_baselines3.__version__, "numpy": np.__version__,
        "pandas": pd.__version__, "cpu": cpu_name(), "logical_cpus": os.cpu_count(),
        "torch_threads": torch.get_num_threads(), "cuda": torch.cuda.is_available(),
        "os": platform.platform(),
    }


# ---------------------------------------------------------- sensitivity views
def derangement(n, seed):
    """Seed-deterministic permutation with no fixed points."""
    rng = np.random.default_rng(seed)
    while True:
        perm = rng.permutation(n)
        if all(perm[i] != i for i in range(n)):
            return [int(p) for p in perm]


def sensitivity_view(condition, measured, seed):
    names = rl_env.FILTERED_LAYERS
    if condition == "correct":
        return dict(measured), None
    if condition == "zeroed":
        return {name: 0.0 for name in names}, None
    if condition == "shuffled":
        perm = derangement(len(names), seed)
        mapping = {names[i]: names[perm[i]] for i in range(len(names))}
        return {layer: measured[source] for layer, source in mapping.items()}, mapping
    raise ValueError(condition)


# ------------------------------------------------------------------ callback
class EpisodeLog(BaseCallback):
    """Records every finished training episode. Read-only: touches no RNG."""

    def __init__(self):
        super().__init__()
        self.rows = []

    def _on_step(self):
        for done, info in zip(self.locals["dones"], self.locals["infos"]):
            if done and "final_accuracy" in info:
                self.rows.append({
                    "episode": len(self.rows) + 1,
                    "timestep": self.num_timesteps,
                    "episode_reward": info["reward"],
                    "accuracy_term": info["accuracy_term"],
                    "sparsity_term": info["sparsity_term"],
                    "diversity_term": info["diversity_term"],
                    "probe_accuracy": info["final_accuracy"],
                    "sparsity": info["final_sparsity"],
                    "actions": str(info["actions"]),
                })
        return True


# ---------------------------------------------------------------- one run
def run_id(condition, coef, seed):
    return f"sens-{condition}_coef-{coef:.2f}_seed-{seed}"


def run_one(ctx, condition, coef, seed):
    rid = run_id(condition, coef, seed)
    out_path = os.path.join(RUNS_DIR, f"{rid}.json")
    if os.path.exists(out_path):
        print(f"  {rid}: already done, skipping", flush=True)
        with open(out_path, encoding="utf-8") as f:
            return json.load(f)

    sensitivities, mapping = sensitivity_view(condition, ctx["measured_sensitivity"], seed)
    started = time.time()
    started_at = datetime.datetime.now().astimezone().isoformat(timespec="seconds")

    train_env = rl_env.CompressionEnvGym(
        base_model=ctx["model"], prunable_layers=ctx["filtered"],
        baseline_accuracy=ctx["baseline_val"],       # notebook: baseline_val_accuracy
        eval_loader=ctx["splits"].probe_loader(),    # notebook: small_val_loader
        max_param_count=ctx["max_param_count"], layer_sensitivities=sensitivities,
        sparsity_coef=coef)

    agent = PPO(env=train_env, seed=seed, **PPO_KWARGS)
    log = EpisodeLog()
    with ctx["splits"].training():
        agent.learn(total_timesteps=TOTAL_TIMESTEPS, callback=log)
    train_seconds = time.time() - started

    # Final deterministic policy, rolled out exactly as evaluate_trained_agent does.
    obs, _ = train_env.reset()
    actions, done = [], False
    while not done:
        action, _ = agent.predict(obs, deterministic=True)
        actions.append(int(action))
        obs, _, terminated, truncated, info = train_env.step(int(action))
        done = terminated or truncated
    probe_accuracy = info["final_accuracy"]

    fractions = [rl_env.ACTION_TO_PRUNE[a] for a in actions]
    pruned = common.prune_layerwise(ctx["model"], fractions)
    sparsity = common.calculate_sparsity(pruned)
    per_layer = common.per_layer_sparsity(pruned)
    val_accuracy = rl_env.evaluate_model(pruned, ctx["splits"].val_loader())
    test_accuracy = rl_env.evaluate_model(pruned, ctx["splits"].test_loader())  # once, after training

    training_reward = rl_env.reward_components(probe_accuracy, ctx["baseline_val"], sparsity,
                                               actions, coef)
    # The notebook's reported reward uses the test baseline and test accuracy
    # (eval_env, cell 49). Kept only for cross-checking against its tables.
    reported_reward = rl_env.reward_components(test_accuracy, ctx["baseline_test"], sparsity,
                                               actions, coef)

    episodes = pd.DataFrame(log.rows)
    best = episodes.loc[episodes["episode_reward"].idxmax()]

    record = {
        "run_id": rid, "experiment_condition": condition, "sparsity_coef": coef, "seed": seed,
        "actions": actions, "prune_percent": [round(100 * f, 1) for f in fractions],
        "total_sparsity": sparsity,
        "per_layer_sparsity": per_layer,
        "probe_accuracy": probe_accuracy,
        "val_accuracy": val_accuracy,
        "test_accuracy": test_accuracy,
        "accuracy_retention": 100.0 * test_accuracy / ctx["baseline_test"],
        "training_reward": training_reward,
        "reported_reward_test_definition": reported_reward,
        "sensitivity_vector": [sensitivities[n] for n in rl_env.FILTERED_LAYERS],
        "sensitivity_measured": [ctx["measured_sensitivity"][n] for n in rl_env.FILTERED_LAYERS],
        "shuffle_mapping": mapping,
        "policy_reported": "final deterministic policy after training",
        "best_training_episode": {
            "episode": int(best["episode"]), "actions": best["actions"],
            "reward": float(best["episode_reward"]),
            "probe_accuracy": float(best["probe_accuracy"]),
            "sparsity": float(best["sparsity"]),
            "note": "selected on validation-probe reward; not test-evaluated",
        },
        "episodes": len(episodes),
        "timesteps_requested": TOTAL_TIMESTEPS,
        "timesteps_trained": int(agent.num_timesteps),
        "ppo": {k: v for k, v in PPO_KWARGS.items() if k != "verbose"},
        "runtime_seconds": round(time.time() - started, 1),
        "train_seconds": round(train_seconds, 1),
        "started_at": started_at,
        **ctx["provenance"],
    }

    os.makedirs(RUNS_DIR, exist_ok=True)
    os.makedirs(CURVES_DIR, exist_ok=True)
    os.makedirs(AGENTS_DIR, exist_ok=True)
    episodes.to_csv(os.path.join(CURVES_DIR, f"{rid}.csv"), index=False)
    agent.save(os.path.join(AGENTS_DIR, rid))
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(record, f, indent=1)

    print(f"  {rid}: actions {actions}  sparsity {sparsity:.2f}%  val {val_accuracy:.2f}%  "
          f"test {test_accuracy:.2f}%  ({record['runtime_seconds']:.0f}s)", flush=True)
    return record


def check_reproduces(record):
    """The correct/0.04 runs must match the notebook's recorded runs exactly."""
    expected = RECORDED[record["seed"]]
    problems = []
    if record["actions"] != expected["actions"]:
        problems.append(f"actions {record['actions']} != {expected['actions']}")
    if abs(record["total_sparsity"] - expected["sparsity"]) > 1e-9:
        problems.append(f"sparsity {record['total_sparsity']} != {expected['sparsity']}")
    if abs(record["test_accuracy"] - expected["test"]) > 1e-9:
        problems.append(f"test {record['test_accuracy']} != {expected['test']}")
    return problems


# ------------------------------------------------------------------- main
def build_context():
    model = common.load_baseline(CKPT)
    all_layers, filtered = rl_env.prunable_layers(model)
    splits = rl_env.Splits(os.path.join(ROOT, "data"))

    baseline_val = rl_env.evaluate_model(model, splits.val_loader())
    baseline_test = rl_env.evaluate_model(model, splits.test_loader())
    assert round(baseline_val, 2) == BASELINE_VAL, f"baseline val {baseline_val} != {BASELINE_VAL}"
    assert round(baseline_test, 2) == BASELINE_TEST, f"baseline test {baseline_test} != {BASELINE_TEST}"

    with splits.training():
        probe_base, measured = rl_env.compute_layer_sensitivities(model, filtered,
                                                                  splits.probe_loader())
    print(f"baseline: val {baseline_val:.2f}%  test {baseline_test:.2f}%  probe {probe_base:.2f}%")
    print("measured sensitivity:", {k: round(v, 4) for k, v in measured.items()})

    here = os.path.dirname(os.path.abspath(__file__))
    provenance = {
        "baseline_checkpoint": "checkpoints/cnn_baseline_FIXED.pth",
        "baseline_sha256": sha256(CKPT),
        "baseline_val_accuracy": baseline_val, "baseline_test_accuracy": baseline_test,
        "baseline_probe_accuracy": probe_base,
        "split": {"seed": rl_env.SPLIT_SEED, "train": 45000, "val": rl_env.VAL_SIZE,
                  "probe": rl_env.RL_PROBE_SIZE, "test": 10000,
                  "val_indices_sha256": hashlib.sha256(
                      json.dumps(splits.val_indices).encode()).hexdigest()},
        "test_set_use": "final evaluation only, after training; locked during agent.learn()",
        "git": git_state(),
        "script_sha256": {name: sha256(os.path.join(here, name))
                          for name in ("ppo_experiments.py", "rl_env.py", "common.py")},
        "environment": environment(),
    }
    return {"model": model, "filtered": filtered, "splits": splits,
            "max_param_count": max(m.weight.numel() for _, m in all_layers),
            "baseline_val": baseline_val, "baseline_test": baseline_test,
            "measured_sensitivity": measured, "provenance": provenance}


def plan(only):
    reproduce = [("correct", 0.04, s) for s in SEEDS]
    ablation = [(c, 0.04, s) for c in ("zeroed", "shuffled") for s in SEEDS]
    sweep = [("correct", k, s) for k in COEFFICIENTS if k != 0.04 for s in SEEDS]
    groups = {"reproduce": reproduce, "ablation": reproduce + ablation,
              "sweep": reproduce + sweep, "all": reproduce + ablation + sweep}
    return groups[only]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", choices=["reproduce", "ablation", "sweep", "all"], default="all")
    args = parser.parse_args()

    ctx = build_context()
    runs = plan(args.only)
    print(f"{len(runs)} runs planned", flush=True)

    for i, (condition, coef, seed) in enumerate(runs, 1):
        print(f"[{i}/{len(runs)}] condition={condition} coef={coef:.2f} seed={seed}", flush=True)
        record = run_one(ctx, condition, coef, seed)
        if condition == "correct" and coef == 0.04:
            problems = check_reproduces(record)
            if problems:
                raise SystemExit(f"REGRESSION FAILURE seed {seed}: " + "; ".join(problems) +
                                 "\nThis harness does not reproduce the notebook. Stopping.")
            print(f"  reproduces the notebook's seed-{seed} run exactly", flush=True)
    print("ALL RUNS DONE", flush=True)


if __name__ == "__main__":
    main()
