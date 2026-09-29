"""Phase 4 implementation checks (after prepare.py, before Stage A) -> results/phase4/phase4_implementation_checks.md.

Only the non-experimental seed 42 and existing Phase-3 records are used; timing runs report runtime only.
"""
import datetime
import json
import os
import random
import re
import time

import numpy as np
import pandas as pd
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CallbackList

import phase4_lib as L
import phase4_metrics as M
from src import data as D, rl as RL
from src.utils import sha256_file

OUT = os.path.join(L.RESULTS, "phase4_implementation_checks.md")
RES = []


def check(name, ok, detail=""):
    RES.append((name, bool(ok), str(detail)))
    print(f"[{'PASS' if ok else 'FAIL'}] {name}: {detail}", flush=True)


def short(setting, reward, coeffs, cache, algo="ppo", eta=0.0, timesteps=256, archive=False, cfg=None, bundles=None):
    d, a = L.P3.split(setting)
    model, _ = L.P3.load_reference(setting)
    b = bundles[d]
    env = L.RewardEnv(model, a, cfg["settings"][setting]["A_b_vrl"], b.rl_loader(), sensitivities=[0.0] * 4,
                      sparsity_coef=L.R0_LAMBDA, cache=cache, reward_kind=reward,
                      reward_params=cfg["reward_params"][setting][reward]["vrl"])
    kw = dict(RL.PPO_KWARGS)
    if coeffs is not None:
        kw["policy"] = RL.SoftPriorPolicy
        kw["policy_kwargs"] = {"prior": L.prior_matrix(coeffs)}
    log = RL.EpisodeLog()
    if algo == "ppo":
        agent, cb = PPO(env=env, seed=42, **kw), log
    else:
        agent = L.ElitePPO(env=env, seed=42, bc_eta=eta, **kw)
        cb = CallbackList([log, L.ArchiveCallback(env)]) if archive else log
    agent.learn(total_timesteps=timesteps, callback=cb)
    params = torch.cat([p.detach().flatten() for p in agent.policy.parameters()])
    return pd.DataFrame(log.rows), params, agent, env


def main():
    torch.set_num_threads(L.THREADS)
    cfg = json.load(open(L.CONFIG, encoding="utf-8"))
    check("thread count fixed at 1", torch.get_num_threads() == 1)
    check("pre-registration file = committed version (914b0418...)", sha256_file(L.PREREG).startswith("914b0418"),
          sha256_file(L.PREREG)[:16])
    check("reward tables unchanged since prepare.py",
          all(sha256_file(os.path.join(L.LAND, f"{s}_rewards.json")) == cfg["reward_tables_sha256"][s] for s in L.SETTINGS))
    p3 = L.P3.frozen_inputs()
    check("Phase-3 frozen inputs unchanged (content hash 7a323f87...)", p3["content_sha256"].startswith("7a323f87"))
    check("Phase-3 V_RL landscapes and reference checkpoints unchanged",
          all(sha256_file(L.P3.landscape_json(s)) == cfg["settings"][s]["phase3_landscape_sha256"]
              and L.P3.load_reference(s)[1] == cfg["settings"][s]["reference_sha256"] for s in L.SETTINGS))
    check("kappa calibration", not cfg["kappa_calibration"]["R1a_calibration_failed"]
          and not cfg["kappa_calibration"]["R1b_calibration_failed"], cfg["kappa"])

    # rewards reproduce the tables; R0 = Phase-3 reward
    rng = random.Random(42)
    bad, r0_same = [], True
    for s in L.SETTINGS:
        T = M.table(s)
        land = json.load(open(L.P3.landscape_json(s), encoding="utf-8"))["policies"]
        r0_same &= T["R0_reward_vrl"] == [p["reward"] for p in land]
        for i in rng.sample(range(1296), 25):
            for R in L.REWARDS:
                pv = cfg["reward_params"][s][R]["vrl"]
                ps = cfg["reward_params"][s][R]["vselect"]
                if (L.reward_parts(R, T["vrl_accuracy"][i], T["total_sparsity"][i], T["actions"][i], pv)["total"]
                        != T[f"{R}_reward_vrl"][i]
                        or L.reward_parts(R, T["vselect_accuracy"][i], T["total_sparsity"][i], T["actions"][i], ps)["total"]
                        != T[f"{R}_score_vselect"][i]):
                    bad.append((s, R, i))
    check("reward tables = reward formulas (25 random policies x 4 rewards x 2 splits x 6 settings)", not bad, bad[:3])
    check("R0 table = the Phase-3 reward (lambda_s 0.01) for all 1,296 policies in every setting", r0_same)

    bundles = {d: D.DataBundle(d, train_images=False) for d in L.P3.DATASETS}
    # R0 + Phase-3 P1 prior through RewardEnv == the Phase-3 PruningEnv pipeline
    s = "cifar10_resnet8"
    d, a = L.P3.split(s)
    model, _ = L.P3.load_reference(s)
    S = cfg["settings"][s]["sensitivity_normalized"]
    env3 = RL.PruningEnv(model, a, cfg["settings"][s]["A_b_vrl"], bundles[d].rl_loader(), sensitivities=[0.0] * 4,
                         sparsity_coef=0.01, cache=L.P3.landscape_cache(s))
    ag3 = PPO(env=env3, seed=42, policy=RL.SoftPriorPolicy, policy_kwargs={"prior": RL.prior_matrix(0.5, S)},
              **{k: v for k, v in RL.PPO_KWARGS.items() if k != "policy"})
    log3 = RL.EpisodeLog()
    ag3.learn(total_timesteps=256, callback=log3)
    r4, _, _, _ = short(s, "R0", cfg["priors"][s]["P1"]["coefficients"], L.P3.landscape_cache(s), cfg=cfg, bundles=bundles)
    check("R0 + P1 via RewardEnv == Phase-3 PruningEnv + Phase-3 prior (64 episodes, bit-identical)",
          pd.DataFrame(log3.rows).equals(r4) and L.prior_matrix(cfg["priors"][s]["P1"]["coefficients"]) == RL.prior_matrix(0.5, S))

    # cached == uncached under every reward
    for s, rewards in (("cifar10_lenet5", L.REWARDS), ("cifar100_resnet8", ["R1a", "R2"])):
        for R in rewards:
            x1, p1, _, _ = short(s, R, None, L.P3.landscape_cache(s), cfg=cfg, bundles=bundles)
            x2, p2, _, _ = short(s, R, None, None, cfg=cfg, bundles=bundles)
            check(f"{s} {R}: landscape cache == uncached environment (64 episodes)", x1.equals(x2) and torch.equal(p1, p2))

    # ElitePPO(eta=0) == SB3 PPO; BC active and exact elite states
    s = "cifar100_lenet5"
    coeffs = cfg["priors"][s]["P2"]["coefficients"]
    y1, q1, _, _ = short(s, "R0", coeffs, L.P3.landscape_cache(s), cfg=cfg, bundles=bundles)
    y2, q2, _, _ = short(s, "R0", coeffs, L.P3.landscape_cache(s), algo="elite", eta=0.0, archive=True, cfg=cfg, bundles=bundles)
    check("ElitePPO with eta = 0 (archive callback attached) == SB3 PPO (64 episodes, bit-identical)", y1.equals(y2) and torch.equal(q1, q2))
    y3, q3, ag, env = short(s, "R0", coeffs, L.P3.landscape_cache(s), algo="elite", eta=L.ETA, archive=True, cfg=cfg, bundles=bundles)
    same_obs = True
    for pol in ag.elite_policies:
        o, _ = env.reset()
        rec = []
        for act in pol:
            rec.append(o)
            o, *_ = env.step(act)
        same_obs &= np.array_equal(np.stack(rec), env.trajectory_obs(pol))
    check("ElitePPO eta = 0.1: behavioural cloning active, changes the policy, elite states == environment observations",
          ag.bc_updates > 0 and not torch.equal(q1, q3) and same_obs and len(ag.elite_policies) == L.ELITE_K,
          f"{ag.bc_updates} BC updates; elites {ag.elite_policies}")

    # priors
    ok = True
    for s in L.SETTINGS:
        S = np.array(cfg["settings"][s]["sensitivity_normalized"])
        pr = cfg["priors"][s]
        cen = S - S.mean()
        ok &= np.allclose(pr["P2"]["coefficients"], cen) and abs(sum(pr["P2"]["coefficients"])) < 1e-12
        ok &= np.allclose(pr["P5"]["coefficients"], -cen) and np.allclose(pr["P4"]["coefficients"], np.abs(cen).mean())
        ok &= pr["P1"]["coefficients"] == (0.5 * S).tolist()
        for seed in L.SEEDS:
            v = pr["P3"][str(seed)]
            ok &= sorted(v["coefficients"]) == sorted(cen.tolist()) and all(v["permutation"][i] != i for i in range(4))
        ok &= all(row[0] == 0 for row in L.prior_matrix(pr["P2"]["coefficients"]))
    check("priors: P1 = Phase-3; P2 centred (sum 0, zero net conservatism); P3 value-changing permutations; "
          "P4 = mean|S - mean S|; P5 = -P2; ratio 0 never shifted", ok)

    # metrics and archive rule on an existing Phase-3 run (R0 = Phase-3 reward)
    rec3 = json.load(open(os.path.join(L.P3.RUNS_DIR, "cifar10_simplecnn_P0_seed300.json"), encoding="utf-8"))
    fake = {"run_id": "check", "setting": rec3["setting"], "reward": "R0", "prior": "P0", "variant": "S0", "seed": 400,
            "final_actions": rec3["actions"], "curve_file": rec3["curve_file"]}
    m = M.run_metrics(fake, cfg)
    T = M.table(rec3["setting"])
    c = M.curve(fake)
    idx = [M.INDEX[tuple(json.loads(x))] for x in c["actions"]]
    top = sorted(set(idx), key=lambda i: (-T["R0_reward_vrl"][i], idx.index(i)))[:10]
    best = max(top, key=lambda i: (T["R0_score_vselect"][i], T["R0_reward_vrl"][i], -idx.index(i)))
    check("metrics on a Phase-3 run: curve rewards = R0 table; archive rule = independent re-implementation; "
          "final rank = Phase-3 recorded rank",
          m["archive_policy_index"] == best and m["final_rank_vrl"] == rec3["reward_rank"] and m["best_seen_rank"] <= m["final_rank_vrl"],
          f"final rank {m['final_rank_vrl']}, best-seen {m['best_seen_rank']}, archive rank {m['archive_rank_vrl']}")

    # guards, no test access in the runners
    b = bundles["cifar10"]
    raised = 0
    with b.training():
        for fn in (b.test_loader, b.select_loader):
            try:
                fn()
            except RuntimeError:
                raised += 1
    here = os.path.dirname(os.path.abspath(__file__))
    src = open(os.path.join(here, "run.py"), encoding="utf-8").read() + open(os.path.join(here, "select_stage.py"), encoding="utf-8").read()
    check("guards: TEST and V_SELECT raise inside training(); run.py and select_stage.py contain no test access",
          raised == 2 and not re.search(r"test_loader|test_tensors|freeze\(", src))

    # timing (runtime only): full budget S2 on the slowest architecture
    t0 = time.time()
    s = "cifar10_simplecnn"
    _, _, ag, _ = short(s, "R0", None, L.P3.landscape_cache(s), algo="elite", eta=L.ETA, archive=True, timesteps=L.TOTAL_TIMESTEPS,
                        cfg=cfg, bundles=bundles)
    secs = time.time() - t0
    check("full budget 2,048 timesteps with ElitePPO; runtime within budget R4 (<= 15 min at 1 thread)",
          ag.num_timesteps == 2048 and secs <= 900, f"{secs:.0f} s (outcome not inspected)")

    n = sum(ok for _, ok, _ in RES)
    lines = ["# Phase 4 implementation checks", "",
             f"Run {datetime.datetime.now().astimezone().isoformat(timespec='seconds')} after prepare.py and before any "
             f"experimental (seed 400-419) run; seed 42 and existing Phase-3 records only; 1 thread. "
             f"**{n}/{len(RES)} passed.**", "", "| Status | Check | Detail |", "|---|---|---|"]
    lines += [f"| {'PASS' if ok else 'FAIL'} | {nm} | {dt.replace('|', '/')[:300]} |" for nm, ok, dt in RES]
    with open(OUT, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"{n}/{len(RES)} passed")


if __name__ == "__main__":
    main()
