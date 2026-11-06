"""Sensitivity-Aware Constrained PPO: runner (see results/constrained_preregistration.md).

All PPO conditions use sb3_contrib.MaskablePPO. Masks are genuine categorical
masks: MaskablePPO's MaskableCategorical sets masked logits to -inf, so masked
actions have zero probability during rollouts, updates and deterministic
prediction. Nothing is clipped, remapped or penalised. The unconstrained
conditions pass an all-valid mask through the same code path.

Conditions (per sensitivity definition d in {accuracy, loss}):
  C0  zero state,       all actions valid         (shared by both definitions)
  C1  correct state d,  all actions valid
  C2  correct state d,  correct mask d            (proposed)
  C3  correct state d,  masks permuted among layers (every layer gets a
                        different mask as a set; seed-deterministic)
  C4  zero state,       correct mask d
  C5  heuristic, no PPO (evaluated in constrained_aggregate.py)

MaskedCompressionEnv reproduces rl_env.CompressionEnvGym.step exactly and adds
(1) action_masks() for the current layer and (2) memoisation of the terminal
evaluation by policy. The reward is a deterministic function of the four
actions, so memoisation returns identical values; `verify` proves it by
reproducing a recorded original run, all 512 episodes included.

Run from the repository root:
  python evaluation/constrained_ppo.py verify
  python evaluation/constrained_ppo.py run --worker K --workers 4 [--seeds 20]
"""
import argparse
import datetime
import json
import os
import sys
import time

import numpy as np
import pandas as pd
import torch
from sb3_contrib import MaskablePPO
from stable_baselines3 import PPO

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common  # noqa: E402
import ppo_experiments as base  # noqa: E402  (shared, unmodified harness pieces)
import rl_env  # noqa: E402

ROOT = base.ROOT
RESULTS = os.path.join(ROOT, "results")
RUNS_DIR = os.path.join(RESULTS, "constrained_runs")
CURVES_DIR = os.path.join(RESULTS, "constrained_training_curves")
AGENTS_DIR = os.path.join(ROOT, "checkpoints", "constrained_experiments")
MASKS_CSV = os.path.join(RESULTS, "sensitivity_action_masks.csv")
STATE_CSV = os.path.join(RESULTS, "constrained_sensitivity_state.csv")
PREREG = os.path.join(RESULTS, "constrained_preregistration.md")

DEFINITIONS = ["accuracy", "loss"]
COEFFICIENTS = [0.01, 0.02]
SEEDS_10 = [42, 1, 2, 3, 4, 5, 6, 7, 8, 9]
SEEDS_20 = SEEDS_10 + list(range(10, 20))
N_ACTIONS = len(rl_env.ACTION_TO_PRUNE)
ALL_VALID = [[True] * N_ACTIONS for _ in rl_env.FILTERED_LAYERS]

_EVAL_CACHE = {}


# ------------------------------------------------------------ environment
class MaskedCompressionEnv(rl_env.CompressionEnvGym):
    """rl_env.CompressionEnvGym with per-layer action masks and a memoised terminal evaluation."""

    def __init__(self, *args, layer_masks=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.layer_masks = [np.array(m, dtype=bool) for m in (layer_masks or ALL_VALID)]
        assert len(self.layer_masks) == self.total_layers and all(m[0] for m in self.layer_masks), \
            "every layer must allow action 0"
        self.masked_actions_taken = 0

    def action_masks(self):
        if self.current_layer_index >= self.total_layers:
            return np.ones(N_ACTIONS, dtype=bool)
        return self.layer_masks[self.current_layer_index].copy()

    def step(self, action):
        # Identical to rl_env.CompressionEnvGym.step except for the memoised evaluation.
        action = int(action)
        if not self.layer_masks[self.current_layer_index][action]:
            self.masked_actions_taken += 1
        prune_amount = rl_env.ACTION_TO_PRUNE[action]
        layer_name = self.prunable_layers_info[self.current_layer_index][0]
        self.current_model = rl_env.apply_layer_pruning_by_name_inplace(self.current_model, layer_name,
                                                                        prune_amount)
        self.actions_taken.append(action)
        self.cumulative_pruning += prune_amount
        self.current_layer_index += 1

        terminated = self.current_layer_index >= self.total_layers
        if terminated:
            key = tuple(self.actions_taken)
            if key not in _EVAL_CACHE:
                _EVAL_CACHE[key] = (rl_env.evaluate_model(self.current_model, self.eval_loader),
                                    rl_env.calculate_sparsity(self.current_model))
            else:
                # Creating a DataLoader iterator draws one int64 from the GLOBAL torch
                # generator (torch/utils/data/dataloader.py, _BaseDataLoaderIter.__init__),
                # and PPO samples actions from that same generator. Create it anyway, so a
                # cached evaluation consumes the random stream exactly as a real one does.
                iter(self.eval_loader)
            final_accuracy, final_sparsity = _EVAL_CACHE[key]
            parts = rl_env.reward_components(final_accuracy, self.baseline_accuracy, final_sparsity,
                                             self.actions_taken, self.sparsity_coef)
            reward = parts["total"]
            observation = np.zeros((rl_env.STATE_DIM,), dtype=np.float32)
            info = {"final_accuracy": final_accuracy, "final_sparsity": final_sparsity,
                    "reward": reward, "actions": list(self.actions_taken), **parts}
        else:
            reward = 0.0
            observation = self._get_state()
            info = {"current_layer": layer_name, "prune_amount": prune_amount}
        return observation, reward, terminated, False, info


# ------------------------------------------------------------ design
def load_design():
    masks = pd.read_csv(MASKS_CSV)
    state = pd.read_csv(STATE_CSV)
    design = {}
    for d in DEFINITIONS:
        m = masks[masks["rule"] == d].set_index("layer").loc[rl_env.FILTERED_LAYERS]
        allowed = [json.loads(a) for a in m["allowed_actions"]]
        design[d] = {
            "masks": [[a in al for a in range(N_ACTIONS)] for al in allowed],
            "state": dict(zip(rl_env.FILTERED_LAYERS, state.set_index("layer").loc[
                rl_env.FILTERED_LAYERS][f"{d}_normalized"])),
        }
    return design


def shuffled_masks(masks, seed):
    """Seed-deterministic permutation where every layer gets a mask different (as a set) from its own."""
    names = rl_env.FILTERED_LAYERS
    rng = np.random.default_rng(seed)
    for _ in range(10000):
        perm = rng.permutation(len(names))
        if all(masks[perm[i]] != masks[i] for i in range(len(names))):
            mapping = {names[i]: names[int(perm[i])] for i in range(len(names))}
            return [masks[int(perm[i])] for i in range(len(names))], mapping
    raise RuntimeError("no permutation gives every layer a different mask")


def conditions(design, coef, seed):
    zero = {n: 0.0 for n in rl_env.FILTERED_LAYERS}
    out = [("shared", "C0", zero, ALL_VALID, None)]
    for d in DEFINITIONS:
        sh, mapping = shuffled_masks(design[d]["masks"], seed)
        out += [(d, "C1", design[d]["state"], ALL_VALID, None),
                (d, "C2", design[d]["state"], design[d]["masks"], None),
                (d, "C3", design[d]["state"], sh, mapping),
                (d, "C4", zero, design[d]["masks"], None)]
    return out


def plan(seeds):
    return [(coef, seed) for coef in COEFFICIENTS for seed in seeds]


def run_id(definition, condition, coef, seed):
    return f"constrained_{condition}_def-{definition}_coef-{coef:.2f}_seed-{seed}"


# ------------------------------------------------------------ one run
def masked_rollout(agent, env):
    obs, _ = env.reset()
    actions, done = [], False
    while not done:
        action, _ = agent.predict(obs, action_masks=env.action_masks(), deterministic=True)
        actions.append(int(action))
        obs, _, terminated, truncated, info = env.step(int(action))
        done = terminated or truncated
    return actions, info


def train(ctx, rid, definition, condition, state, masks, mapping, coef, seed, algo=MaskablePPO,
          out=(RUNS_DIR, CURVES_DIR, AGENTS_DIR)):
    runs_dir, curves_dir, agents_dir = out
    path = os.path.join(runs_dir, f"{rid}.json")
    if os.path.exists(path):
        print(f"  {rid}: already done, skipping", flush=True)
        return None
    started = time.time()
    started_at = datetime.datetime.now().astimezone().isoformat(timespec="seconds")
    env = MaskedCompressionEnv(ctx["model"], ctx["filtered"], ctx["baseline_val"],
                               ctx["splits"].probe_loader(), ctx["max_param_count"],
                               layer_sensitivities=state, sparsity_coef=coef, layer_masks=masks)
    agent = algo(env=env, seed=seed, **base.PPO_KWARGS)
    log = base.EpisodeLog()
    with ctx["splits"].training():
        agent.learn(total_timesteps=base.TOTAL_TIMESTEPS, callback=log)
    train_seconds = time.time() - started

    episodes = pd.DataFrame(log.rows)
    violations = sum(1 for a in episodes["actions"] for i, x in enumerate(json.loads(a)) if not masks[i][x])
    assert violations == 0 and env.masked_actions_taken == 0, f"{rid}: {violations} masked actions taken"

    if algo is MaskablePPO:
        actions, info = masked_rollout(agent, env)
    else:
        obs, _ = env.reset()
        actions, done = [], False
        while not done:
            a, _ = agent.predict(obs, deterministic=True)
            actions.append(int(a))
            obs, _, done, _, info = env.step(int(a))
    assert all(masks[i][a] for i, a in enumerate(actions)), f"{rid}: final policy uses a masked action"

    # Frozen policy -> fresh baseline copy -> validation, then test once.
    fractions = [rl_env.ACTION_TO_PRUNE[a] for a in actions]
    pruned = common.prune_layerwise(ctx["model"], fractions)
    sparsity = common.calculate_sparsity(pruned)
    val_accuracy = rl_env.evaluate_model(pruned, ctx["splits"].val_loader())
    preds, labels = common.predictions(pruned, ctx["splits"].test_loader())
    test_accuracy = common.accuracy_from(preds, labels)
    reward = rl_env.reward_components(info["final_accuracy"], ctx["baseline_val"], sparsity, actions, coef)
    best = episodes.loc[episodes["episode_reward"].idxmax()]

    record = {
        "run_id": rid, "experiment": "Sensitivity-Aware Constrained PPO",
        "sensitivity_definition": definition, "condition": condition, "sparsity_coef": coef, "seed": seed,
        "algorithm": algo.__name__,
        "state_sensitivity_vector": [float(state[n]) for n in rl_env.FILTERED_LAYERS],
        "layer_masks": [[int(b) for b in m] for m in masks],
        "mask_shuffle_mapping": mapping,
        "n_valid_actions": int(sum(sum(m) for m in masks)),
        "actions": actions, "prune_percent": [round(100 * f, 1) for f in fractions],
        "total_sparsity": sparsity, "per_layer_sparsity": common.per_layer_sparsity(pruned),
        "probe_accuracy": info["final_accuracy"], "val_accuracy": val_accuracy,
        "test_accuracy": test_accuracy, "accuracy_retention": 100.0 * test_accuracy / ctx["baseline_test"],
        "training_reward": reward,
        "best_training_episode": {"episode": int(best["episode"]), "actions": best["actions"],
                                  "reward": float(best["episode_reward"])},
        "masked_actions_taken_in_training": int(violations),
        "episodes": len(episodes), "timesteps_trained": int(agent.num_timesteps),
        "ppo": {k: v for k, v in base.PPO_KWARGS.items() if k != "verbose"},
        "ppo_defaults": {"gae_lambda": agent.gae_lambda, "clip_range": 0.2, "vf_coef": agent.vf_coef,
                         "max_grad_norm": agent.max_grad_norm},
        "test_predictions": "".join(str(int(p)) for p in preds.tolist()),
        "runtime_seconds": round(time.time() - started, 1), "train_seconds": round(train_seconds, 1),
        "started_at": started_at, **ctx["provenance"],
    }
    for d in (runs_dir, curves_dir, agents_dir):
        os.makedirs(d, exist_ok=True)
    episodes.to_csv(os.path.join(curves_dir, f"{rid}.csv"), index=False)
    agent.save(os.path.join(agents_dir, rid))
    with open(path, "w", encoding="utf-8") as f:
        json.dump(record, f, indent=1)
    print(f"  {rid}: {actions} sparsity {sparsity:.2f}% val {val_accuracy:.2f}% test {test_accuracy:.2f}% "
          f"({record['runtime_seconds']:.0f}s)", flush=True)
    return record


def context(threads):
    torch.set_num_threads(threads)
    ctx = base.build_context()
    here = os.path.dirname(os.path.abspath(__file__))
    import sb3_contrib
    ctx["provenance"]["script_sha256"]["constrained_ppo.py"] = base.sha256(os.path.join(here, "constrained_ppo.py"))
    ctx["provenance"]["environment"]["torch_threads"] = torch.get_num_threads()
    ctx["provenance"]["environment"]["sb3_contrib"] = sb3_contrib.__version__
    for name, p in (("masks_sha256", MASKS_CSV), ("state_sha256", STATE_CSV), ("preregistration_sha256", PREREG)):
        ctx["provenance"][name] = base.sha256(p)
    return ctx


# ------------------------------------------------------------ commands
def cmd_verify(args):
    """(1) memoised env + PPO reproduces the recorded original seed-42 run, all 512 episodes;
    (2) MaskablePPO with all-valid masks on the same config, compared (informational)."""
    ctx = context(args.threads)
    tmp = os.path.join(RESULTS, "_verify_constrained")
    rid = base.run_id("correct", 0.04, 42)
    recorded = pd.read_csv(os.path.join(base.CURVES_DIR, f"{rid}.csv"))
    for algo in (PPO, MaskablePPO):
        _EVAL_CACHE.clear()
        out = (os.path.join(tmp, algo.__name__),) * 3
        rec = train(ctx, rid, "original", "verify", dict(ctx["measured_sensitivity"]), ALL_VALID, None,
                    0.04, 42, algo=algo, out=out)
        same = pd.read_csv(os.path.join(out[1], f"{rid}.csv")).equals(recorded)
        print(f"VERIFY {algo.__name__}: all 512 episodes identical to the recorded PPO run = {same}; "
              f"final {rec['actions']} test {rec['test_accuracy']}; train {rec['train_seconds']:.0f}s, "
              f"{len(_EVAL_CACHE)} distinct policies evaluated", flush=True)


def cmd_run(args):
    design = load_design()
    seeds = SEEDS_20 if args.seeds == 20 else SEEDS_10
    ctx = context(args.threads)
    jobs = []
    for coef, seed in plan(seeds):
        for d, cond, state, masks, mapping in conditions(design, coef, seed):
            jobs.append((d, cond, state, masks, mapping, coef, seed))
    mine = [j for i, j in enumerate(jobs) if i % args.workers == args.worker]
    print(f"worker {args.worker}/{args.workers}: {len(mine)} of {len(jobs)} runs, seeds {seeds}", flush=True)
    for i, (d, cond, state, masks, mapping, coef, seed) in enumerate(mine, 1):
        print(f"[{args.worker}:{i}/{len(mine)}] {cond} def={d} coef={coef:.2f} seed={seed}", flush=True)
        train(ctx, run_id(d, cond, coef, seed), d, cond, state, masks, mapping, coef, seed)
    print(f"WORKER {args.worker} DONE", flush=True)


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)
    v = sub.add_parser("verify")
    v.add_argument("--threads", type=int, default=1)
    r = sub.add_parser("run")
    r.add_argument("--threads", type=int, default=1)
    r.add_argument("--worker", type=int, default=0)
    r.add_argument("--workers", type=int, default=1)
    r.add_argument("--seeds", type=int, choices=[10, 20], default=10)
    args = parser.parse_args()
    {"verify": cmd_verify, "run": cmd_run}[args.cmd](args)


if __name__ == "__main__":
    main()
