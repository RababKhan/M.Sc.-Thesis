"""Refined sensitivity analysis: multi-ratio estimator and a fair PPO ablation.

The ORIGINAL sensitivity ablation (ppo_experiments.py, lambda_s = 0.04, 10%
probe on 1,000 images) is left untouched. This is a separate experiment.

Subcommands
  probe    Measure each layer at probe ratios 10/20/40/60% on the full
           5,000-image validation split (fresh baseline copy per probe; only
           that layer pruned), record accuracy and cross-entropy loss, and fix
           the two refined definitions BEFORE any PPO run:
             A  accuracy:  mean over ratios of (acc_base - acc_r) / acc_base
             B  loss:      mean over ratios of (loss_r - loss_base) / loss_base
           Each is min-max normalised across the four layers to [0, 1]
           (all equal -> zeros). The normalised value enters the state.
  verify   Re-run an ORIGINAL configuration at a given thread count and check
           that it reproduces the recorded run, including its entire training
           curve, before any parallel execution is trusted.
  run      The refined ablation. Conditions:
             correct   normalised refined sensitivity
             zeroed    0.0 in the sensitivity slot (state dimension unchanged)
             shuffled  normalised values permuted among layers by a seed-
                       deterministic derangement in which every layer also
                       receives a DIFFERENT value from its own; mapping recorded
           for definitions {accuracy, loss} x lambda_s {0.01, 0.02} x seeds
           42, 1, ..., 9. The zeroed state does not depend on the definition, so
           one zeroed run per (lambda_s, seed) serves both definitions.
           Everything else is the original harness: same baseline, split,
           reward (accuracy 1.5, diversity 0.05), PPO settings and budget.

Validation data only for sensitivity, reward and selection; test once per run
after training (locked during agent.learn()).

Run from the repository root:
  python evaluation/refined_sensitivity.py probe
  python evaluation/refined_sensitivity.py verify --threads 2 --seed 42
  python evaluation/refined_sensitivity.py run --threads 2 --worker 0 --workers 2
"""
import argparse
import copy
import datetime
import json
import os
import sys
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from stable_baselines3 import PPO

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common  # noqa: E402
import ppo_experiments as base  # noqa: E402  (shared harness pieces; not modified)
import rl_env  # noqa: E402

ROOT = base.ROOT
RESULTS = os.path.join(ROOT, "results")
RUNS_DIR = os.path.join(RESULTS, "refined_runs")
CURVES_DIR = os.path.join(RESULTS, "refined_training_curves")
AGENTS_DIR = os.path.join(ROOT, "checkpoints", "refined_experiments")
VALUES_CSV = os.path.join(RESULTS, "refined_sensitivity_values.csv")
CURVES_CSV = os.path.join(RESULTS, "sensitivity_probe_curves.csv")

PROBE_RATIOS = [0.1, 0.2, 0.4, 0.6]
DEFINITIONS = ["accuracy", "loss"]
COEFFICIENTS = [0.01, 0.02]
SEEDS = [42, 1, 2, 3, 4, 5, 6, 7, 8, 9]
BOOTSTRAP = 2000
FOLD = 1000


# ------------------------------------------------------------------- probe
def per_image(model, images, labels):
    model.eval()
    losses, correct = [], []
    loss_fn = nn.CrossEntropyLoss(reduction="none")
    with torch.no_grad():
        for i in range(0, len(images), 128):
            out = model(images[i:i + 128])
            losses.append(loss_fn(out, labels[i:i + 128]))
            correct.append(out.argmax(1) == labels[i:i + 128])
    return torch.cat(correct).numpy(), torch.cat(losses).numpy()


def definitions_from(base_correct, base_loss, probes, idx):
    """Both refined definitions on the images `idx`, from per-image results."""
    acc0, loss0 = base_correct[idx].mean(), base_loss[idx].mean()
    out = {}
    for layer in rl_env.FILTERED_LAYERS:
        rel_acc = [(acc0 - probes[(layer, r)][0][idx].mean()) / acc0 for r in PROBE_RATIOS]
        rel_loss = [(probes[(layer, r)][1][idx].mean() - loss0) / loss0 for r in PROBE_RATIOS]
        out[layer] = (float(np.mean(rel_acc)), float(np.mean(rel_loss)))
    return out


def minmax(values):
    lo, hi = min(values), max(values)
    if hi - lo <= 0:
        return [0.0 for _ in values]
    return [(v - lo) / (hi - lo) for v in values]


def cmd_probe(_args):
    if os.path.exists(VALUES_CSV):
        raise SystemExit(f"{VALUES_CSV} already exists: the definitions are fixed. "
                         f"Delete it deliberately to recompute.")
    model = common.load_baseline(base.CKPT)
    splits = rl_env.Splits(os.path.join(ROOT, "data"))
    val_x, val_y = splits._val
    everything = np.arange(len(val_y))

    base_correct, base_loss = per_image(model, val_x, val_y)
    probes = {}
    rows = []
    rng = np.random.default_rng(0)
    for layer in rl_env.FILTERED_LAYERS:
        for r in PROBE_RATIOS:
            pruned = rl_env.apply_layer_pruning_by_name_inplace(copy.deepcopy(model), layer, r)
            correct, loss = per_image(pruned, val_x, val_y)
            probes[(layer, r)] = (correct, loss)
            d_acc = base_correct.astype(float) - correct.astype(float)
            d_loss = loss - base_loss
            boot_acc, boot_loss = [], []
            for _ in range(BOOTSTRAP):
                s = rng.integers(0, len(val_y), len(val_y))
                boot_acc.append(100 * d_acc[s].mean())
                boot_loss.append(d_loss[s].mean())
            acc0, loss0 = 100 * base_correct.mean(), base_loss.mean()
            acc, mean_loss = 100 * correct.mean(), loss.mean()
            rows.append({
                "layer": layer, "probe_ratio": r, "n_images": len(val_y),
                "val_accuracy": acc, "baseline_val_accuracy": acc0,
                "accuracy_drop_pp": acc0 - acc,
                "accuracy_drop_pp_ci95_low": np.percentile(boot_acc, 2.5),
                "accuracy_drop_pp_ci95_high": np.percentile(boot_acc, 97.5),
                "relative_accuracy_drop": (acc0 - acc) / acc0,
                "val_loss": mean_loss, "baseline_val_loss": loss0,
                "loss_increase": mean_loss - loss0,
                "loss_increase_ci95_low": np.percentile(boot_loss, 2.5),
                "loss_increase_ci95_high": np.percentile(boot_loss, 97.5),
                "relative_loss_increase": (mean_loss - loss0) / loss0,
            })
    pd.DataFrame(rows).to_csv(CURVES_CSV, index=False)

    full = definitions_from(base_correct, base_loss, probes, everything)
    acc_raw = [full[n][0] for n in rl_env.FILTERED_LAYERS]
    loss_raw = [full[n][1] for n in rl_env.FILTERED_LAYERS]
    with open(os.path.join(RESULTS, "runs", "sens-correct_coef-0.04_seed-42.json"), encoding="utf-8") as f:
        original = json.load(f)["sensitivity_measured"]

    # Stability: the same definitions on each disjoint 1,000-image fold.
    fold_rows = []
    for k in range(len(val_y) // FOLD):
        fold = definitions_from(base_correct, base_loss, probes, np.arange(k * FOLD, (k + 1) * FOLD))
        for name in rl_env.FILTERED_LAYERS:
            fold_rows.append({"fold": k, "layer": name, "accuracy_raw": fold[name][0],
                              "loss_raw": fold[name][1]})
    pd.DataFrame(fold_rows).to_csv(os.path.join(RESULTS, "refined_sensitivity_folds.csv"), index=False)

    registered = datetime.datetime.now().astimezone().isoformat(timespec="seconds")
    values = pd.DataFrame({
        "layer": rl_env.FILTERED_LAYERS,
        "accuracy_raw": acc_raw, "accuracy_normalized": minmax(acc_raw),
        "loss_raw": loss_raw, "loss_normalized": minmax(loss_raw),
        "original_raw_pp": original,
        "probe_ratios": str(PROBE_RATIOS), "n_validation_images": len(val_y),
        "normalization": "min-max across the four layers to [0,1]; all equal -> 0",
        "registered_before_any_refined_ppo_run": registered,
    })
    values.to_csv(VALUES_CSV, index=False)
    print(pd.DataFrame(rows)[["layer", "probe_ratio", "val_accuracy", "accuracy_drop_pp",
                              "accuracy_drop_pp_ci95_low", "accuracy_drop_pp_ci95_high",
                              "val_loss", "relative_loss_increase"]].to_string(index=False))
    print(values.drop(columns=["probe_ratios", "normalization"]).to_string(index=False))


# -------------------------------------------------------------- run helpers
def load_values():
    values = pd.read_csv(VALUES_CSV)
    assert list(values["layer"]) == rl_env.FILTERED_LAYERS
    return values


def shuffle_mapping(normalized, seed):
    """Seed-deterministic derangement where every layer also gets a different VALUE."""
    names = rl_env.FILTERED_LAYERS
    rng = np.random.default_rng(seed)
    for _ in range(10000):
        perm = rng.permutation(len(names))
        if all(perm[i] != i and not np.isclose(normalized[perm[i]], normalized[i])
               for i in range(len(names))):
            return {names[i]: names[int(perm[i])] for i in range(len(names))}
    raise RuntimeError("no value-changing derangement exists for these values")


def plan():
    runs = []
    for coef in COEFFICIENTS:
        for seed in SEEDS:
            runs.append(("shared", "zeroed", coef, seed))
            for definition in DEFINITIONS:
                runs.append((definition, "correct", coef, seed))
                runs.append((definition, "shuffled", coef, seed))
    return runs


def run_id(definition, condition, coef, seed):
    return f"refined_def-{definition}_sens-{condition}_coef-{coef:.2f}_seed-{seed}"


def state_vector(values, definition, condition, seed):
    names = rl_env.FILTERED_LAYERS
    if condition == "zeroed":
        return {n: 0.0 for n in names}, None, None, None
    raw = list(values[f"{definition}_raw"])
    normalized = list(values[f"{definition}_normalized"])
    if condition == "correct":
        return dict(zip(names, normalized)), raw, normalized, None
    mapping = shuffle_mapping(normalized, seed)
    index = {n: i for i, n in enumerate(names)}
    return {n: normalized[index[mapping[n]]] for n in names}, raw, normalized, mapping


def train_and_record(ctx, rid, sensitivities, coef, seed, meta, runs_dir, curves_dir, agents_dir):
    out_path = os.path.join(runs_dir, f"{rid}.json")
    if os.path.exists(out_path):
        print(f"  {rid}: already done, skipping", flush=True)
        with open(out_path, encoding="utf-8") as f:
            return json.load(f)

    started = time.time()
    started_at = datetime.datetime.now().astimezone().isoformat(timespec="seconds")
    env = rl_env.CompressionEnvGym(
        base_model=ctx["model"], prunable_layers=ctx["filtered"], baseline_accuracy=ctx["baseline_val"],
        eval_loader=ctx["splits"].probe_loader(), max_param_count=ctx["max_param_count"],
        layer_sensitivities=sensitivities, sparsity_coef=coef)
    agent = PPO(env=env, seed=seed, **base.PPO_KWARGS)
    log = base.EpisodeLog()
    with ctx["splits"].training():
        agent.learn(total_timesteps=base.TOTAL_TIMESTEPS, callback=log)
    train_seconds = time.time() - started

    obs, _ = env.reset()
    actions, done = [], False
    while not done:
        action, _ = agent.predict(obs, deterministic=True)
        actions.append(int(action))
        obs, _, terminated, truncated, info = env.step(int(action))
        done = terminated or truncated
    probe_accuracy = info["final_accuracy"]

    fractions = [rl_env.ACTION_TO_PRUNE[a] for a in actions]
    pruned = common.prune_layerwise(ctx["model"], fractions)
    sparsity = common.calculate_sparsity(pruned)
    val_accuracy = rl_env.evaluate_model(pruned, ctx["splits"].val_loader())
    test_accuracy = rl_env.evaluate_model(pruned, ctx["splits"].test_loader())  # once, after training
    reward = rl_env.reward_components(probe_accuracy, ctx["baseline_val"], sparsity, actions, coef)
    episodes = pd.DataFrame(log.rows)
    best = episodes.loc[episodes["episode_reward"].idxmax()]

    record = {
        "run_id": rid, **meta, "sparsity_coef": coef, "seed": seed,
        "state_sensitivity_vector": [sensitivities[n] for n in rl_env.FILTERED_LAYERS],
        "actions": actions, "prune_percent": [round(100 * f, 1) for f in fractions],
        "total_sparsity": sparsity, "per_layer_sparsity": common.per_layer_sparsity(pruned),
        "probe_accuracy": probe_accuracy, "val_accuracy": val_accuracy, "test_accuracy": test_accuracy,
        "accuracy_retention": 100.0 * test_accuracy / ctx["baseline_test"],
        "training_reward": reward,
        "best_training_episode": {"episode": int(best["episode"]), "actions": best["actions"],
                                  "reward": float(best["episode_reward"]),
                                  "note": "validation-probe reward; not test-evaluated"},
        "episodes": len(episodes), "timesteps_requested": base.TOTAL_TIMESTEPS,
        "timesteps_trained": int(agent.num_timesteps),
        "ppo": {k: v for k, v in base.PPO_KWARGS.items() if k != "verbose"},
        "runtime_seconds": round(time.time() - started, 1), "train_seconds": round(train_seconds, 1),
        "started_at": started_at, **ctx["provenance"],
    }
    for d in (runs_dir, curves_dir, agents_dir):
        os.makedirs(d, exist_ok=True)
    episodes.to_csv(os.path.join(curves_dir, f"{rid}.csv"), index=False)
    agent.save(os.path.join(agents_dir, rid))
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(record, f, indent=1)
    print(f"  {rid}: {actions} sparsity {sparsity:.2f}% val {val_accuracy:.2f}% "
          f"test {test_accuracy:.2f}% ({record['runtime_seconds']:.0f}s)", flush=True)
    return record


def context(threads):
    torch.set_num_threads(threads)
    ctx = base.build_context()
    here = os.path.dirname(os.path.abspath(__file__))
    ctx["provenance"]["script_sha256"]["refined_sensitivity.py"] = base.sha256(
        os.path.join(here, "refined_sensitivity.py"))
    ctx["provenance"]["environment"]["torch_threads"] = torch.get_num_threads()
    if os.path.exists(VALUES_CSV):
        ctx["provenance"]["refined_values_sha256"] = base.sha256(VALUES_CSV)
    return ctx


def cmd_verify(args):
    """Re-run an original configuration and compare with its recorded run exactly."""
    ctx = context(args.threads)
    rid = base.run_id("correct", 0.04, args.seed)
    tmp = os.path.join(ROOT, "results", "_verify_threads", f"threads-{args.threads}")
    record = train_and_record(ctx, rid, dict(ctx["measured_sensitivity"]), 0.04, args.seed,
                              {"purpose": "thread-count verification"}, tmp, tmp, tmp)
    with open(os.path.join(base.RUNS_DIR, f"{rid}.json"), encoding="utf-8") as f:
        recorded = json.load(f)
    same_curve = pd.read_csv(os.path.join(tmp, f"{rid}.csv")).equals(
        pd.read_csv(os.path.join(base.CURVES_DIR, f"{rid}.csv")))
    same = (record["actions"] == recorded["actions"]
            and record["total_sparsity"] == recorded["total_sparsity"]
            and record["test_accuracy"] == recorded["test_accuracy"]
            and record["val_accuracy"] == recorded["val_accuracy"])
    print(f"VERIFY threads={args.threads} seed={args.seed}: final result identical={same}, "
          f"all 512 training episodes identical={same_curve}, "
          f"train {record['train_seconds']:.0f}s", flush=True)


def cmd_run(args):
    values = load_values()
    ctx = context(args.threads)
    runs = plan()
    mine = [r for i, r in enumerate(runs) if i % args.workers == args.worker]
    print(f"worker {args.worker}/{args.workers}: {len(mine)} of {len(runs)} runs, "
          f"{torch.get_num_threads()} threads", flush=True)
    for i, (definition, condition, coef, seed) in enumerate(mine, 1):
        sensitivities, raw, normalized, mapping = state_vector(values, definition, condition, seed)
        meta = {"experiment": "refined sensitivity analysis", "sensitivity_definition": definition,
                "sensitivity_condition": condition, "raw_sensitivity_vector": raw,
                "normalized_sensitivity_vector": normalized, "shuffle_mapping": mapping,
                "shared_by_definitions": DEFINITIONS if condition == "zeroed" else None}
        print(f"[{args.worker}:{i}/{len(mine)}] def={definition} cond={condition} "
              f"coef={coef:.2f} seed={seed}", flush=True)
        train_and_record(ctx, run_id(definition, condition, coef, seed), sensitivities, coef, seed,
                         meta, RUNS_DIR, CURVES_DIR, AGENTS_DIR)
    print(f"WORKER {args.worker} DONE", flush=True)


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("probe")
    v = sub.add_parser("verify")
    v.add_argument("--threads", type=int, default=4)
    v.add_argument("--seed", type=int, default=42)
    r = sub.add_parser("run")
    r.add_argument("--threads", type=int, default=4)
    r.add_argument("--worker", type=int, default=0)
    r.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()
    {"probe": cmd_probe, "verify": cmd_verify, "run": cmd_run}[args.cmd](args)


if __name__ == "__main__":
    main()
