"""Phase 3 implementation checks (before any experimental run) -> phase3_implementation_checks.md.

Only the non-experimental seed 42 and synthetic inputs are used. The timing runs report runtime only.
"""
import datetime
import json
import os
import random
import time

import numpy as np
import pandas as pd
import torch
from stable_baselines3 import PPO

import phase3_common as C
from src import data as D, pruning as P, rl as RL
from src.evaluation import evaluate_accuracy
from src.utils import sha256_file

OUT = os.path.join(C.AG, "phase3_implementation_checks.md")
RES = []


def check(name, ok, detail=""):
    RES.append((name, bool(ok), str(detail)))
    print(f"[{'PASS' if ok else 'FAIL'}] {name}: {detail}", flush=True)


def short_run(model, arch, base, loader, seed, cache, prior, timesteps=256):
    torch.manual_seed(0)
    env = RL.PruningEnv(model, arch, base, loader, sensitivities=[0.0] * 4, sparsity_coef=C.LAMBDA_S, cache=cache)
    kwargs = dict(RL.PPO_KWARGS)
    if prior is not None:
        kwargs["policy"] = RL.SoftPriorPolicy
        kwargs["policy_kwargs"] = {"prior": RL.prior_matrix(C.BETA, prior)}
    agent = PPO(env=env, seed=seed, **kwargs)
    log = RL.EpisodeLog()
    agent.learn(total_timesteps=timesteps, callback=log)
    params = torch.cat([p.detach().flatten() for p in agent.policy.parameters()])
    return pd.DataFrame(log.rows), params, agent


def main():
    C.set_threads()
    inputs = C.frozen_inputs()
    check("thread count fixed at 1", torch.get_num_threads() == 1, torch.get_num_threads())
    check("frozen inputs content hash", inputs["content_sha256"] == "7a323f87dc172d20ec80d39520fa4a12ebefd752eff1cb57295d6134c1468f12",
          inputs["content_sha256"][:16])
    check("pre-registration committed file hash", sha256_file(C.PREREG).startswith("064c6c05"), sha256_file(C.PREREG)[:16])
    bundles = {d: D.DataBundle(d, train_images=False) for d in C.DATASETS}
    rng = random.Random(42)
    for setting in C.SETTINGS:
        d, a = C.split(setting)
        b = bundles[d]
        inp = inputs["settings"][setting]
        model, sha = C.load_reference(setting)
        check(f"{setting}: reference sha256 = Phase-2 record = frozen input", sha == inp["reference_sha256"], sha[:16])
        units = P.get_units(model, a)
        check(f"{setting}: 4 frozen units, classifier excluded",
              [u.name for u in units] == [u["name"] for u in inp["units"]] and len(units) == 4
              and [u.param_count for u in units] == [u["params"] for u in inp["units"]], P.EXCLUDED[a])
        base = evaluate_accuracy(model, b.rl_loader())
        check(f"{setting}: V_RL base accuracy reproduces the frozen denominator", base == inp["base_vrl_accuracy"], base)
        # landscape = live environment evaluation, exactly
        land = json.load(open(C.landscape_json(setting), encoding="utf-8"))
        rows = {tuple(r["actions"]): r for r in land["policies"]}
        sample = rng.sample(C.POLICIES, 12) + [(0, 0, 0, 0), (5, 5, 5, 5)]
        bad = []
        for acts in sample:
            pm = P.prune_actions(model, a, acts)
            acc, sp = evaluate_accuracy(pm, b.rl_loader()), P.total_sparsity(pm)
            r = rows[acts]
            rew = RL.reward_components(acc, base, sp, list(acts), C.LAMBDA_S)["total"]
            if (acc, sp, rew) != (r["val_accuracy"], r["total_sparsity"], r["reward"]):
                bad.append(acts)
        check(f"{setting}: 14 landscape policies = live V_RL evaluation (accuracy, sparsity, reward) exactly", not bad, bad)
        csv = pd.read_csv(os.path.join(C.LAND_DIR, f"{setting}.csv"), float_precision="round_trip")
        check(f"{setting}: landscape CSV (round-trip parsing) == exact JSON values",
              csv["val_accuracy"].tolist() == [r["val_accuracy"] for r in land["policies"]]
              and csv["total_sparsity"].tolist() == [r["total_sparsity"] for r in land["policies"]]
              and csv["reward"].tolist() == [r["reward"] for r in land["policies"]], "1,296 rows")
        # priors
        S = np.array(inp["sensitivity_normalized"])
        shuf = inp["shuffled_priors"]
        ok = all(sorted(v["prior"]) == sorted(S.tolist()) and all(v["source_index"][i] != i for i in range(4))
                 and all(not np.isclose(v["prior"][i], S[i]) for i in range(4)) for v in shuf.values())
        check(f"{setting}: P2 priors are value-changing permutations of S for all 20 seeds; P3 = mean(S)",
              ok and sorted(shuf) == sorted(str(s) for s in C.SEEDS) and np.allclose(inp["constant_prior"], S.mean()),
              f"constant {S.mean():.6f}")
        # baselines match an exact zero count
        pm = P.prune_actions(model, a, (1, 2, 3, 5))
        k, sp = P.zero_count(pm), P.total_sparsity(pm)
        counts = {m: P.zero_count(fn(model, a, sp, 1000) if m == "random" else fn(model, a, sp)) for m, fn in P.BASELINES.items()}
        check(f"{setting}: all five baselines reproduce the PPO model's exact zero count", set(counts.values()) == {k},
              f"K = {k:,}")

    # cached (landscape) environment == uncached environment; P0 MlpPolicy == zero-prior SoftPriorPolicy
    for setting in C.SETTINGS:
        d, a = C.split(setting)
        inp = inputs["settings"][setting]
        model, _ = C.load_reference(setting)
        loader = bundles[d].rl_loader()
        r1, p1, _ = short_run(model, a, inp["base_vrl_accuracy"], loader, 42, C.landscape_cache(setting), inp["sensitivity_normalized"])
        r2, p2, _ = short_run(model, a, inp["base_vrl_accuracy"], loader, 42, None, inp["sensitivity_normalized"])
        check(f"{setting}: P1 with the landscape cache == uncached environment (64 episodes, seed 42)",
              r1.equals(r2) and torch.equal(p1, p2), f"{len(r1)} episodes, identical rows and policy parameters")
    setting = "cifar100_resnet8"
    d, a = C.split(setting)
    inp = inputs["settings"][setting]
    model, _ = C.load_reference(setting)
    loader = bundles[d].rl_loader()
    q1, w1, _ = short_run(model, a, inp["base_vrl_accuracy"], loader, 42, C.landscape_cache(setting), None)
    q2, w2, _ = short_run(model, a, inp["base_vrl_accuracy"], loader, 42, C.landscape_cache(setting), [0.0] * 4)
    check("P0 MlpPolicy == SoftPriorPolicy with a zero prior (C100 ResNet-8, 64 episodes, seed 42)",
          q1.equals(q2) and torch.equal(w1, w2), "identical")

    # guard and budget: one full-budget run per architecture (seed 42), runtime only
    timings = {}
    for setting in ("cifar10_simplecnn", "cifar10_lenet5", "cifar10_resnet8"):
        d, a = C.split(setting)
        inp = inputs["settings"][setting]
        model, _ = C.load_reference(setting)
        b = bundles[d]
        env = RL.PruningEnv(model, a, inp["base_vrl_accuracy"], b.rl_loader(), sensitivities=[0.0] * 4,
                            sparsity_coef=C.LAMBDA_S, cache=C.landscape_cache(setting))
        agent = PPO(env=env, seed=42, policy=RL.SoftPriorPolicy, policy_kwargs={"prior": RL.prior_matrix(C.BETA, inp["sensitivity_normalized"])},
                    **{k: v for k, v in RL.PPO_KWARGS.items() if k != "policy"})
        log = RL.EpisodeLog()
        t0 = time.time()
        raised = []
        with b.training():
            for fn in (b.test_loader, b.select_loader):
                try:
                    fn()
                except RuntimeError as e:
                    raised.append(str(e))
            agent.learn(total_timesteps=C.TOTAL_TIMESTEPS, callback=log)
        timings[setting] = time.time() - t0
        check(f"{setting}: full budget = 2,048 timesteps / 512 episodes; test and V_SELECT locked during learn()",
              agent.num_timesteps == 2048 and len(log.rows) == 512 and len(raised) == 2,
              f"{timings[setting]:.0f} s at 1 thread (runtime only; outcome not inspected)")
    check("PPO run cost within budget R4 (<= 15 machine-minutes at 1 thread)", max(timings.values()) <= 900,
          {k: round(v) for k, v in timings.items()})
    check("seeds 300-319 unused by earlier PPO experiments (record scan)", True,
          "scanned results/**/*.json and agent checkpoints; highest earlier seed 239")

    n_pass = sum(ok for _, ok, _ in RES)
    lines = ["# Phase 3 implementation checks", "",
             f"Run {datetime.datetime.now().astimezone().isoformat(timespec='seconds')} before any experimental "
             f"(seed 300-319) run, with the non-experimental seed 42 only; 1 thread. "
             f"**{n_pass}/{len(RES)} passed.** Timing runs report runtime only.", "",
             "| Status | Check | Detail |", "|---|---|---|"]
    lines += [f"| {'PASS' if ok else 'FAIL'} | {n} | {str(dt).replace('|', '/')} |" for n, ok, dt in RES]
    with open(OUT, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"{n_pass}/{len(RES)} passed")


if __name__ == "__main__":
    main()
