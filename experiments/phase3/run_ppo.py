"""Phase 3 PPO runs (results/phase3_preregistration.md). Resumable; 1 thread per worker.

  python experiments/phase3/run_ppo.py --worker K --workers 4        # K = 0..3

Job i of the fixed 480-job order goes to worker i mod workers. A run is skipped only if its record
verifies (complete status, curve hash, unchanged frozen-input and landscape hashes); an invalid
record is logged to phase3_runs/corruption_log.txt and the run is repeated with the same seed and
configuration. Progress lines contain the run id and runtime only (no metrics: no interim peeking).
"""
import argparse
import datetime
import gzip
import io
import json
import os
import time

import numpy as np
import pandas as pd
import torch
from stable_baselines3 import PPO

import phase3_common as C
from src import data as D, pruning as P, rl as RL
from src.evaluation import accuracy_from, predictions
from src.utils import environment, git_state, sha256_file

CURVES = os.path.join(C.RUNS_DIR, "curves")


class Context:
    """Per-worker cache of bundles, reference models, landscapes and hashes."""

    def __init__(self):
        C.set_threads()
        self.inputs = C.frozen_inputs()
        self.bundles, self.models, self.caches, self.land = {}, {}, {}, {}
        self.hashes = {"preregistration_sha256": sha256_file(C.PREREG),
                       "frozen_inputs_sha256": sha256_file(C.INPUTS_JSON),
                       "sensitivity_csv_sha256": sha256_file(C.SENS_CSV),
                       "destructive_rules_sha256": sha256_file(C.DESTRUCTIVE_JSON)}
        here = os.path.dirname(os.path.abspath(__file__))
        self.code = {os.path.relpath(p, C.ROOT).replace("\\", "/"): sha256_file(p)
                     for p in [os.path.join(here, f) for f in ("run_ppo.py", "phase3_common.py")]
                     + [os.path.join(C.ROOT, "src", s, "__init__.py") for s in ("rl", "pruning", "data", "evaluation")]}
        self.env_info = environment()
        self.git = git_state()

    def setting(self, setting):
        if setting not in self.models:
            d, a = C.split(setting)
            self.bundles.setdefault(d, D.DataBundle(d, train_images=False))
            model, sha = C.load_reference(setting)
            assert sha == self.inputs["settings"][setting]["reference_sha256"]
            self.models[setting] = model
            self.caches[setting] = C.landscape_cache(setting)
            assert len(self.caches[setting]) == 1296
            land = json.load(open(C.landscape_json(setting), encoding="utf-8"))
            self.land[setting] = {tuple(r["actions"]): r for r in land["policies"]}
            self.hashes[f"landscape_{setting}_sha256"] = sha256_file(C.landscape_json(setting))
        d, _ = C.split(setting)
        return self.bundles[d], self.models[setting], self.caches[setting], self.land[setting]


def prior_vector(inputs, condition, seed):
    if condition == "P0":
        return None
    if condition == "P1":
        return inputs["sensitivity_normalized"]
    if condition == "P2":
        return inputs["shuffled_priors"][str(seed)]["prior"]
    return inputs["constant_prior"]


def record_path(rid):
    return os.path.join(C.RUNS_DIR, f"{rid}.json")


def verified(ctx, setting, rid):
    path = record_path(rid)
    if not os.path.exists(path):
        return False
    try:
        rec = json.load(open(path, encoding="utf-8"))
        ok = (rec.get("status") == "complete"
              and sha256_file(os.path.join(C.ROOT, rec["curve_file"])) == rec["curve_sha256"]
              and rec["frozen_inputs_sha256"] == ctx.hashes["frozen_inputs_sha256"]
              and rec["landscape_sha256"] == sha256_file(C.landscape_json(setting)))
    except Exception as e:                                   # noqa: BLE001
        ok, rec = False, {"error": repr(e)}
    if not ok:
        with open(os.path.join(C.RUNS_DIR, "corruption_log.txt"), "a", encoding="utf-8") as f:
            f.write(f"{datetime.datetime.now().isoformat(timespec='seconds')} {rid}: invalid record, repeated "
                    f"with the same seed and configuration ({str(rec)[:200]})\n")
    return ok


def durable_write(data, path):
    """Write bytes via a flushed + fsynced temporary file and an atomic rename (power-cut safe)."""
    tmp = f"{path}.{os.getpid()}.tmp"
    with open(tmp, "wb") as f:
        f.write(data)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def write_shared_once(arr, path, attempts=20):
    """Shared deterministic cache shared by 4 workers. Written if absent; if present it must be identical
    (a free determinism check). Retries transient Windows locks (another reader, the virus scanner):
    a first version crashed on such a lock, see phase3_runs/infrastructure_log.txt."""
    for _ in range(attempts):
        if os.path.exists(path):
            try:
                existing = np.load(path)
            except (OSError, ValueError):
                time.sleep(0.5)
                continue
            if not np.array_equal(existing, arr):
                raise RuntimeError(f"non-deterministic predictions for {path}")
            return
        buf = io.BytesIO()
        np.save(buf, arr)
        try:
            durable_write(buf.getvalue(), path)
            return
        except PermissionError:
            tmp = f"{path}.{os.getpid()}.tmp"
            if os.path.exists(tmp):
                os.remove(tmp)
            time.sleep(0.5)
    raise RuntimeError(f"could not write {path}")


def test_predictions(bundle, setting, pruned, actions):
    """Computed for every run from its own frozen model (once, after freezing); stored in a shared cache."""
    d, _ = C.split(setting)
    folder = os.path.join(C.PRED_DIR, setting)
    os.makedirs(folder, exist_ok=True)
    preds, labels = predictions(pruned, bundle.test_loader())
    preds, labels = preds.numpy().astype(np.int16), labels.numpy().astype(np.int16)
    write_shared_once(preds, os.path.join(folder, f"ppo_{C.policy_key(actions)}.npy"))
    write_shared_once(labels, os.path.join(C.PRED_DIR, f"{d}_test_labels.npy"))
    return preds, labels


def rollout_with_entropy(agent, env):
    obs, _ = env.reset()
    actions, entropies, done, info = [], [], False, {}
    while not done:
        with torch.no_grad():
            dist = agent.policy.get_distribution(agent.policy.obs_to_tensor(obs)[0])
            entropies.append(float(dist.entropy().item()))
        a, _ = agent.predict(obs, deterministic=True)
        actions.append(int(a))
        obs, _, done, _, info = env.step(int(a))
    return actions, info, entropies


def run_one(ctx, setting, condition, seed):
    rid = C.run_id(setting, condition, seed)
    started = time.time()
    started_at = datetime.datetime.now().astimezone().isoformat(timespec="seconds")
    bundle, model, cache, land = ctx.setting(setting)
    inputs = ctx.inputs["settings"][setting]
    d, a = C.split(setting)
    prior = prior_vector(inputs, condition, seed)
    env = RL.PruningEnv(model, a, inputs["base_vrl_accuracy"], bundle.rl_loader(), sensitivities=[0.0] * 4,
                        sparsity_coef=C.LAMBDA_S, cache=cache)
    kwargs = dict(RL.PPO_KWARGS)
    if prior is not None:
        kwargs["policy"] = RL.SoftPriorPolicy
        kwargs["policy_kwargs"] = {"prior": RL.prior_matrix(C.BETA, prior)}
    agent = PPO(env=env, seed=seed, **kwargs)
    log = RL.EpisodeLog()
    with bundle.training():                              # test and V_SELECT raise in here
        agent.learn(total_timesteps=C.TOTAL_TIMESTEPS, callback=log)
    train_seconds = time.time() - started
    assert agent.num_timesteps == 2048 and len(log.rows) == C.EPISODES, "budget"
    assert len(cache) == 1296, "cache must not grow (every policy is in the landscape)"
    actions, info, entropies = rollout_with_entropy(agent, env)

    pruned = P.prune_actions(model, a, actions)
    sparsity = P.total_sparsity(pruned)
    row = land[tuple(actions)]
    assert sparsity == row["total_sparsity"] and info["final_accuracy"] == row["val_accuracy"]
    bundle.freeze({"run": rid, "actions": actions})       # frozen policy: the test set unlocks
    preds, labels = test_predictions(bundle, setting, pruned, actions)
    test_acc = accuracy_from(torch.from_numpy(preds), torch.from_numpy(labels))
    reward = RL.reward_components(info["final_accuracy"], inputs["base_vrl_accuracy"], sparsity, actions, C.LAMBDA_S)

    os.makedirs(CURVES, exist_ok=True)
    os.makedirs(C.AGENTS_DIR, exist_ok=True)
    curve_file = os.path.join(CURVES, f"{rid}.csv.gz")
    buf = io.StringIO()
    pd.DataFrame(log.rows).to_csv(buf, index=False, float_format="%.17g")
    durable_write(gzip.compress(buf.getvalue().encode("utf-8"), mtime=0), curve_file)
    agent_path = os.path.join(C.AGENTS_DIR, f"{rid}.zip")
    agent.save(agent_path)
    units = [u["name"] for u in inputs["units"]]
    rec = {
        "run_id": rid, "status": "complete", "setting": setting, "dataset": d, "arch": a, "condition": condition,
        "seed": seed, "beta": 0.0 if prior is None else C.BETA, "prior_vector": prior,
        "prior_mapping": inputs["shuffled_priors"][str(seed)]["mapping"] if condition == "P2" else None,
        "state_sensitivity": [0.0] * 4, "sparsity_coef": C.LAMBDA_S,
        "policy_class": "MlpPolicy" if prior is None else "SoftPriorPolicy",
        "actions": actions, "unit_names": units,
        "unit_ratios_percent": [int(100 * P.ACTION_TO_PRUNE[x]) for x in actions],
        "unit_sparsity": list(P.unit_sparsity(pruned, a).values()), "total_sparsity": sparsity,
        "zero_count": P.zero_count(pruned), "val_accuracy_vrl": info["final_accuracy"], "test_accuracy": test_acc,
        "reward": reward["total"], "reward_components": reward, "reward_rank": row["reward_rank"],
        "pareto_val_accuracy": row["pareto_val_accuracy"], "utility": row["utility"], "destructive": row["destructive"],
        "action_entropy_rollout": entropies, "mean_action_entropy": float(np.mean(entropies)),
        "episodes": len(log.rows), "timesteps_trained": int(agent.num_timesteps),
        "ppo": {k: (v.__name__ if isinstance(v, type) else v) for k, v in kwargs.items() if k not in ("verbose", "policy_kwargs")},
        "ppo_defaults": {"gae_lambda": agent.gae_lambda, "clip_range": float(agent.clip_range(1.0)), "vf_coef": agent.vf_coef,
                         "max_grad_norm": agent.max_grad_norm, "net_arch": "[64, 64] tanh", "optimizer": "Adam"},
        "threads": torch.get_num_threads(), "test_guard": "DataBundle.training() during learn; freeze before test",
        "reference_checkpoint": inputs["reference_checkpoint"], "reference_sha256": inputs["reference_sha256"],
        "base_vrl_accuracy": inputs["base_vrl_accuracy"],
        "sensitivity_normalized": inputs["sensitivity_normalized"],
        "landscape_sha256": ctx.hashes[f"landscape_{setting}_sha256"],
        "curve_file": os.path.relpath(curve_file, C.ROOT).replace("\\", "/"), "curve_sha256": sha256_file(curve_file),
        "agent_file": os.path.relpath(agent_path, C.ROOT).replace("\\", "/"),
        "test_predictions_file": os.path.relpath(os.path.join(C.PRED_DIR, setting, f"ppo_{C.policy_key(actions)}.npy"),
                                                 C.ROOT).replace("\\", "/"),
        "train_seconds": round(train_seconds, 1), "runtime_seconds": round(time.time() - started, 1),
        "started_at": started_at, "finished_at": datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
        **{k: v for k, v in ctx.hashes.items() if not k.startswith("landscape_")},
        "code_sha256": ctx.code, "environment": ctx.env_info, "git": ctx.git,
    }
    durable_write(json.dumps(rec, indent=1).encode("utf-8"), record_path(rid))
    return rec["runtime_seconds"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--worker", type=int, required=True)
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()
    os.makedirs(C.RUNS_DIR, exist_ok=True)
    ctx = Context()
    mine = [(i, j) for i, j in enumerate(C.jobs()) if i % args.workers == args.worker]
    print(f"worker {args.worker}: {len(mine)} jobs", flush=True)
    for i, (setting, condition, seed) in mine:
        rid = C.run_id(setting, condition, seed)
        if verified(ctx, setting, rid):
            continue
        secs = run_one(ctx, setting, condition, seed)
        print(f"done {rid} {secs:.0f}s", flush=True)
    print(f"worker {args.worker}: finished", flush=True)


if __name__ == "__main__":
    main()
