"""Phase 5 confirmatory PPO runs (C0, C1, C2 x 6 settings x seeds 500-519). Resumable; 1 thread; never reads TEST
or VAL-SELECT (both raise inside DataBundle.training(); nothing here requests them).

  python experiments/phase5/run.py --worker K --workers 4
Progress lines show run ids and runtimes only.
"""
import argparse
import datetime
import gzip
import io
import json
import os
import time

import pandas as pd
import torch
from stable_baselines3 import PPO

import phase5_lib as L
from src import rl as RL
from src.utils import environment, git_state, sha256_file


def durable_write(data, path):
    tmp = f"{path}.{os.getpid()}.tmp"
    with open(tmp, "wb") as f:
        f.write(data)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def run_id(setting, cond, seed):
    return f"{setting}_{cond}_seed{seed}"


def jobs():
    return [(s, c, x) for s in L.SETTINGS for x in L.SEEDS for c in L.CONDITIONS]


class Ctx:
    def __init__(self):
        torch.set_num_threads(L.THREADS)
        assert torch.get_num_threads() == L.THREADS
        self.cfg = json.load(open(L.CONFIG, encoding="utf-8"))
        self.bundles, self.models, self.caches = {}, {}, {}
        here = os.path.dirname(os.path.abspath(__file__))
        self.hashes = {"preregistration_sha256": sha256_file(L.PREREG), "config_sha256": sha256_file(L.CONFIG),
                       "design_commitments_sha256": sha256_file(L.COMMITMENTS),
                       "code_sha256": {f: sha256_file(os.path.join(here, f)) for f in ("run.py", "phase5_lib.py")}}
        self.env_info, self.git = environment(), git_state()

    def setting(self, s):
        if s not in self.models:
            d, _ = L.P3.split(s)
            self.bundles.setdefault(d, L.bundle(d))
            model, sha = L.P3.load_reference(s)
            assert sha == self.cfg["settings"][s]["reference_sha256"]
            land = json.load(open(os.path.join(L.LAND, f"{s}_landscape.json"), encoding="utf-8"))
            assert sha256_file(os.path.join(L.LAND, f"{s}_landscape.json")) == self.cfg["settings"][s]["landscape_sha256"]
            self.models[s] = model
            self.caches[s] = {tuple(p["actions"]): (p["val_rl_accuracy"], p["total_sparsity"]) for p in land["policies"]}
        return self.bundles[L.P3.split(s)[0]], self.models[s], self.caches[s]


def verified(ctx, rid):
    path = os.path.join(L.RUNS, f"{rid}.json")
    if not os.path.exists(path):
        return False
    try:
        r = json.load(open(path, encoding="utf-8"))
        ok = r["status"] == "complete" and r["config_sha256"] == ctx.hashes["config_sha256"] and \
            sha256_file(os.path.join(L.ROOT, r["curve_file"])) == r["curve_sha256"]
    except Exception as e:                                   # noqa: BLE001
        ok, r = False, {"error": repr(e)}
    if not ok:
        with open(os.path.join(L.RUNS, "corruption_log.txt"), "a", encoding="utf-8") as f:
            f.write(f"{datetime.datetime.now().isoformat(timespec='seconds')} {rid}: invalid record, repeated ({str(r)[:200]})\n")
    return ok


def run_one(ctx, s, cond, seed):
    rid = run_id(s, cond, seed)
    t0 = time.time()
    started = datetime.datetime.now().astimezone().isoformat(timespec="seconds")
    b, model, cache = ctx.setting(s)
    _, arch = L.P3.split(s)
    cs = ctx.cfg["settings"][s]
    kind = "R0" if cond == "C0" else "RC"
    env = L.Phase5Env(model, arch, cs["dense_val_rl_accuracy"], b.rl_loader(), sensitivities=[0.0] * 4, sparsity_coef=0.01,
                      cache=cache, reward_kind=kind, tau=ctx.cfg["tau"], dense_acc=cs["dense_val_rl_accuracy"])
    kw = dict(RL.PPO_KWARGS)
    if cond in ("C0", "C1"):
        kw["policy"] = RL.SoftPriorPolicy
        kw["policy_kwargs"] = {"prior": L.prior_matrix(cs["P2_coefficients"])}
    agent = PPO(env=env, seed=seed, **kw)
    log = RL.EpisodeLog()
    with b.training():                                      # TEST and VAL-SELECT raise in here
        agent.learn(total_timesteps=L.TOTAL_TIMESTEPS, callback=log)
    train_s = time.time() - t0
    assert agent.num_timesteps == 2048 and len(log.rows) == L.EPISODES and len(cache) == 1296
    actions, info = RL.deterministic_rollout(agent, env)
    os.makedirs(L.CURVES, exist_ok=True)
    os.makedirs(L.AGENTS, exist_ok=True)
    cf = os.path.join(L.CURVES, f"{rid}.csv.gz")
    buf = io.StringIO()
    pd.DataFrame(log.rows).to_csv(buf, index=False, float_format="%.17g")
    durable_write(gzip.compress(buf.getvalue().encode("utf-8"), mtime=0), cf)
    ap = os.path.join(L.AGENTS, f"{rid}.zip")
    agent.save(ap)
    rec = {"run_id": rid, "status": "complete", "setting": s, "condition": cond, "seed": seed, "reward": kind,
           "prior": "P2" if cond != "C2" else "none", "tau": ctx.cfg["tau"], "final_actions": actions,
           "final_val_rl_accuracy": info["final_accuracy"], "final_sparsity": info["final_sparsity"], "final_reward": info["reward"],
           "episodes": len(log.rows), "timesteps_trained": int(agent.num_timesteps), "threads": torch.get_num_threads(),
           "ppo": {k: (v.__name__ if isinstance(v, type) else v) for k, v in kw.items() if k not in ("verbose", "policy_kwargs")},
           "ppo_defaults": {"gae_lambda": agent.gae_lambda, "clip_range": float(agent.clip_range(1.0)), "vf_coef": agent.vf_coef,
                            "max_grad_norm": agent.max_grad_norm, "net_arch": "[64, 64] tanh", "optimizer": "Adam"},
           "test_set_accessed": False, "val_select_accessed": False,
           "reference_sha256": cs["reference_sha256"], "landscape_sha256": cs["landscape_sha256"],
           "curve_file": os.path.relpath(cf, L.ROOT).replace("\\", "/"), "curve_sha256": sha256_file(cf),
           "agent_file": os.path.relpath(ap, L.ROOT).replace("\\", "/"), "train_seconds": round(train_s, 1),
           "runtime_seconds": round(time.time() - t0, 1), "started_at": started,
           "finished_at": datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
           **ctx.hashes, "environment": ctx.env_info, "git": ctx.git}
    durable_write(json.dumps(rec, indent=1).encode("utf-8"), os.path.join(L.RUNS, f"{rid}.json"))
    return rec["runtime_seconds"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--worker", type=int, required=True)
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    os.makedirs(L.RUNS, exist_ok=True)
    ctx = Ctx()
    mine = [j for i, j in enumerate(jobs()) if i % a.workers == a.worker]
    print(f"worker {a.worker}: {len(mine)} jobs", flush=True)
    for s, c, x in mine:
        rid = run_id(s, c, x)
        if verified(ctx, rid):
            continue
        print(f"done {rid} {run_one(ctx, s, c, x):.0f}s", flush=True)
    print(f"worker {a.worker}: finished", flush=True)


if __name__ == "__main__":
    main()
