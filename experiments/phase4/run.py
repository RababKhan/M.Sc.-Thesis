"""Phase 4 training runs (resumable, 1 thread per worker). The TEST set is never accessed here.

  python experiments/phase4/run.py --stage A --worker K --workers 4
  python experiments/phase4/run.py --stage B --worker K --workers 4     # needs Stage-A selection
  python experiments/phase4/run.py --stage C --worker K --workers 4     # needs Stage-B selection

A training configuration is (reward, prior, training variant) with variant S0 (standard PPO) or S2 (PPO +
elite behavioural cloning). S1 and S2A are selection rules applied afterwards to S0 / S2 runs. A
configuration shared by two stages is trained once (same seed = identical run). Progress lines show run
ids and runtimes only.
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
from stable_baselines3.common.callbacks import CallbackList

import phase4_lib as L
from src import data as D, rl as RL
from src.utils import environment, git_state, sha256_file


def durable_write(data, path):
    tmp = f"{path}.{os.getpid()}.tmp"
    with open(tmp, "wb") as f:
        f.write(data)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def run_id(setting, reward, prior, variant, seed):
    return f"{setting}_{reward}_{prior}_{variant}_seed{seed}"


def stage_configs(stage):
    """Training configurations (reward, prior, variant) of a stage; later stages need the frozen selection."""
    if stage == "A":
        return [(r, "P0", "S0") for r in L.REWARDS]
    sel = json.load(open(L.SELECTION, encoding="utf-8"))
    r = sel["A"]["winner"]
    if stage == "B":
        return [(r, p, "S0") for p in L.PRIORS]
    p = sel["B"]["winner"]
    return [(r, p, "S0"), (r, p, "S2")]


def jobs(stage):
    return [(s, r, p, v, seed) for s in L.SETTINGS for seed in L.SEEDS for (r, p, v) in stage_configs(stage)]


class Context:
    def __init__(self):
        torch.set_num_threads(L.THREADS)
        assert torch.get_num_threads() == L.THREADS
        self.cfg = json.load(open(L.CONFIG, encoding="utf-8"))
        self.bundles, self.models, self.caches = {}, {}, {}
        here = os.path.dirname(os.path.abspath(__file__))
        self.hashes = {"preregistration_sha256": sha256_file(L.PREREG), "config_sha256": sha256_file(L.CONFIG),
                       "code_sha256": {f: sha256_file(os.path.join(here, f)) for f in ("run.py", "phase4_lib.py")}}
        self.env_info, self.git = environment(), git_state()

    def setting(self, s):
        if s not in self.models:
            d, a = L.P3.split(s)
            self.bundles.setdefault(d, D.DataBundle(d, train_images=False))
            model, sha = L.P3.load_reference(s)
            assert sha == self.cfg["settings"][s]["reference_sha256"]
            assert sha256_file(L.P3.landscape_json(s)) == self.cfg["settings"][s]["phase3_landscape_sha256"]
            self.models[s] = model
            self.caches[s] = L.P3.landscape_cache(s)
        return self.bundles[L.P3.split(s)[0]], self.models[s], self.caches[s]


def verified(ctx, rid):
    path = os.path.join(L.RUNS, f"{rid}.json")
    if not os.path.exists(path):
        return False
    try:
        rec = json.load(open(path, encoding="utf-8"))
        ok = (rec["status"] == "complete" and rec["config_sha256"] == ctx.hashes["config_sha256"]
              and sha256_file(os.path.join(L.ROOT, rec["curve_file"])) == rec["curve_sha256"])
    except Exception as e:                                  # noqa: BLE001
        ok, rec = False, {"error": repr(e)}
    if not ok:
        with open(os.path.join(L.RUNS, "corruption_log.txt"), "a", encoding="utf-8") as f:
            f.write(f"{datetime.datetime.now().isoformat(timespec='seconds')} {rid}: invalid record, repeated ({str(rec)[:200]})\n")
    return ok


def run_one(ctx, setting, reward, prior, variant, seed):
    rid = run_id(setting, reward, prior, variant, seed)
    started = time.time()
    started_at = datetime.datetime.now().astimezone().isoformat(timespec="seconds")
    bundle, model, cache = ctx.setting(setting)
    _, arch = L.P3.split(setting)
    cfg_s = ctx.cfg["settings"][setting]
    prm = ctx.cfg["reward_params"][setting][reward]["vrl"]
    pri = ctx.cfg["priors"][setting][prior]
    coeffs = None if prior == "P0" else (pri[str(seed)]["coefficients"] if prior == "P3" else pri["coefficients"])
    env = L.RewardEnv(model, arch, cfg_s["A_b_vrl"], bundle.rl_loader(), sensitivities=[0.0] * 4,
                      sparsity_coef=L.R0_LAMBDA, cache=cache, reward_kind=reward, reward_params=prm)
    kwargs = dict(RL.PPO_KWARGS)
    if coeffs is not None:
        kwargs["policy"] = RL.SoftPriorPolicy
        kwargs["policy_kwargs"] = {"prior": L.prior_matrix(coeffs)}
    log = RL.EpisodeLog()
    if variant == "S0":
        agent = PPO(env=env, seed=seed, **kwargs)
        callback = log
    else:
        agent = L.ElitePPO(env=env, seed=seed, bc_eta=L.ETA, **kwargs)
        archive = L.ArchiveCallback(env)
        callback = CallbackList([log, archive])
    with bundle.training():                                 # TEST and V_SELECT raise in here
        agent.learn(total_timesteps=L.TOTAL_TIMESTEPS, callback=callback)
    train_seconds = time.time() - started
    assert agent.num_timesteps == 2048 and len(log.rows) == L.EPISODES and len(cache) == 1296
    actions, info = RL.deterministic_rollout(agent, env)

    os.makedirs(L.CURVES, exist_ok=True)
    os.makedirs(L.AGENTS, exist_ok=True)
    curve_file = os.path.join(L.CURVES, f"{rid}.csv.gz")
    buf = io.StringIO()
    pd.DataFrame(log.rows).to_csv(buf, index=False, float_format="%.17g")
    durable_write(gzip.compress(buf.getvalue().encode("utf-8"), mtime=0), curve_file)
    agent_path = os.path.join(L.AGENTS, f"{rid}.zip")
    agent.save(agent_path)
    rec = {"run_id": rid, "status": "complete", "setting": setting, "reward": reward, "prior": prior, "variant": variant,
           "seed": seed, "prior_coefficients": coeffs,
           "prior_permutation": pri[str(seed)]["permutation"] if prior == "P3" else None,
           "reward_params_vrl": prm, "final_actions": actions, "final_vrl_accuracy": info["final_accuracy"],
           "final_sparsity": info["final_sparsity"], "final_reward": info["reward"],
           "final_reward_components": {k: info[k] for k in ("accuracy_term", "sparsity_term", "diversity_term", "total")},
           "episodes": len(log.rows), "timesteps_trained": int(agent.num_timesteps), "threads": torch.get_num_threads(),
           "bc_eta": L.ETA if variant == "S2" else 0.0, "bc_updates": getattr(agent, "bc_updates", 0),
           "final_elite_policies": getattr(agent, "elite_policies", None),
           "ppo": {k: (v.__name__ if isinstance(v, type) else v) for k, v in kwargs.items() if k not in ("verbose", "policy_kwargs")},
           "ppo_defaults": {"gae_lambda": agent.gae_lambda, "clip_range": float(agent.clip_range(1.0)), "vf_coef": agent.vf_coef,
                            "max_grad_norm": agent.max_grad_norm, "net_arch": "[64, 64] tanh", "optimizer": "Adam"},
           "test_set_accessed": False, "guard": "DataBundle.training() during learn(); no test access in this runner",
           "reference_sha256": cfg_s["reference_sha256"], "landscape_sha256": cfg_s["phase3_landscape_sha256"],
           "curve_file": os.path.relpath(curve_file, L.ROOT).replace("\\", "/"), "curve_sha256": sha256_file(curve_file),
           "agent_file": os.path.relpath(agent_path, L.ROOT).replace("\\", "/"),
           "train_seconds": round(train_seconds, 1), "runtime_seconds": round(time.time() - started, 1),
           "started_at": started_at, "finished_at": datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
           **ctx.hashes, "environment": ctx.env_info, "git": ctx.git}
    durable_write(json.dumps(rec, indent=1).encode("utf-8"), os.path.join(L.RUNS, f"{rid}.json"))
    return rec["runtime_seconds"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["A", "B", "C"])
    ap.add_argument("--worker", type=int, required=True)
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()
    os.makedirs(L.RUNS, exist_ok=True)
    ctx = Context()
    mine = [j for i, j in enumerate(jobs(args.stage)) if i % args.workers == args.worker]
    print(f"stage {args.stage} worker {args.worker}: {len(mine)} jobs", flush=True)
    for s, r, p, v, seed in mine:
        rid = run_id(s, r, p, v, seed)
        if verified(ctx, rid):
            continue
        secs = run_one(ctx, s, r, p, v, seed)
        print(f"done {rid} {secs:.0f}s", flush=True)
    print(f"stage {args.stage} worker {args.worker}: finished", flush=True)


if __name__ == "__main__":
    main()
