"""Phase 5 pre-run implementation checks (non-experimental seed 42 only) -> results/phase5/phase5_implementation_checks.md."""
import datetime
import glob
import json
import os
import random
import re
import subprocess
import time

import numpy as np
import pandas as pd
import torch
from stable_baselines3 import PPO

import phase5_lib as L
import select_archive as SA
from src import pruning as P, rl as RL
from src.evaluation import evaluate_accuracy
from src.utils import sha256_file

RES = []


def check(n, ok, d=""):
    RES.append((n, bool(ok), str(d)))
    print(f"[{'PASS' if ok else 'FAIL'}] {n}: {d}", flush=True)


def git(*a):
    return subprocess.run(["git", *a], cwd=L.ROOT, capture_output=True, text=True).stdout.strip()


def short(s, cond, cfg, bundles, cache, timesteps=256):
    d, a = L.P3.split(s)
    model, _ = L.P3.load_reference(s)
    cs = cfg["settings"][s]
    env = L.Phase5Env(model, a, cs["dense_val_rl_accuracy"], bundles[d].rl_loader(), sensitivities=[0.0] * 4, sparsity_coef=0.01,
                      cache=cache, reward_kind="R0" if cond == "C0" else "RC", tau=cfg["tau"], dense_acc=cs["dense_val_rl_accuracy"])
    kw = dict(RL.PPO_KWARGS)
    if cond != "C2":
        kw["policy"] = RL.SoftPriorPolicy
        kw["policy_kwargs"] = {"prior": L.prior_matrix(cs["P2_coefficients"])}
    ag = PPO(env=env, seed=42, **kw)
    log = RL.EpisodeLog()
    ag.learn(total_timesteps=timesteps, callback=log)
    return pd.DataFrame(log.rows), torch.cat([p.detach().flatten() for p in ag.policy.parameters()]), ag


def main():
    torch.set_num_threads(L.THREADS)
    cfg = json.load(open(L.CONFIG, encoding="utf-8"))
    check("thread count 1", torch.get_num_threads() == 1)
    check("design commitments unchanged since their commit (aff2f3e)", not git("diff", "aff2f3e", "--", "results/phase5/phase5_design_commitments.md"))
    check("pre-registration present and referenced by hash in the config-independent record", os.path.exists(L.PREREG), sha256_file(L.PREREG)[:16])
    check("Phase-3 and Phase-4 files unchanged", not git("diff", "67f4270", "--", "results/architecture_generalization", "results/stronger_baselines",
                                                        "checkpoints/phase3_agents") and not git("diff", "a3653fd", "--", "results/phase4",
                                                                                                  "checkpoints/phase4_agents"))
    man = json.load(open(os.path.join(L.SPLITS, "phase5_split_manifest.json"), encoding="utf-8"))
    ok = True
    for d in ("cifar10", "cifar100"):
        z = np.load(os.path.join(L.SPLITS, f"{d}_phase5_split.npz"))
        rl, sel = z["val_rl_positions"], z["val_select_positions"]
        ok &= len(rl) == len(sel) == 2500 and not set(rl.tolist()) & set(sel.tolist()) and sorted(np.concatenate([rl, sel]).tolist()) == list(range(5000))
        ok &= man["datasets"][d]["file_sha256"] == sha256_file(os.path.join(L.SPLITS, f"{d}_phase5_split.npz"))
        ok &= man["datasets"][d]["max_abs_class_imbalance_rl_vs_select"] <= 1
    check("VAL-RL / VAL-SELECT: 2,500 each, disjoint, cover the 5,000 validation images, class counts differ by <= 1, hashes = manifest", ok)
    bundles = {d: L.bundle(d) for d in ("cifar10", "cifar100")}
    b = bundles["cifar10"]
    raised = 0
    try:
        b.test_loader()
    except RuntimeError:
        raised += 1
    with b.training():
        for fn in (b.test_loader, b.select_loader):
            try:
                fn()
            except RuntimeError:
                raised += 1
    check("guards: TEST raises before freeze; TEST and VAL-SELECT raise inside training()", raised == 3)
    rng = random.Random(42)
    bad = []
    for s in L.SETTINGS:
        d, a = L.P3.split(s)
        model, _ = L.P3.load_reference(s)
        T = json.load(open(os.path.join(L.LAND, f"{s}_tables.json"), encoding="utf-8"))
        for i in rng.sample(range(1296), 6) + [L.INDEX[(0, 0, 0, 0)]]:
            pm = P.prune_actions(model, a, L.POLICIES[i])
            if (evaluate_accuracy(pm, bundles[d].rl_loader()), evaluate_accuracy(pm, bundles[d].select_loader()), P.total_sparsity(pm)) != \
                    (T["val_rl_accuracy"][i], T["val_select_accuracy"][i], T["total_sparsity"][i]):
                bad.append((s, i))
    check("landscapes == live VAL-RL / VAL-SELECT evaluation and sparsity (7 policies x 6 settings)", not bad, bad[:3])
    ok_r = True
    for s in L.SETTINGS:
        T = json.load(open(os.path.join(L.LAND, f"{s}_tables.json"), encoding="utf-8"))
        cs = cfg["settings"][s]
        rc = np.array(T["RC_reward_val_rl"])
        f = np.array(T["feasible_val_rl"])
        ok_r &= bool(rc[f].min() > 1.0 and (rc[~f].max() < 0.0 if (~f).any() else True))
        ok_r &= T["RC_rank_val_rl"][cfg["oracle"][s]["policy_index"]] == 1
        ok_r &= cfg["oracle"][s]["sparsity"] == max(np.array(T["total_sparsity"])[f])
        for i in rng.sample(range(1296), 10):
            ok_r &= rc[i] == L.r_constrained(T["val_rl_accuracy"][i], cs["dense_val_rl_accuracy"], T["total_sparsity"][i], cfg["tau"])
            ok_r &= T["R0_reward_val_rl"][i] == L.r0(T["val_rl_accuracy"][i], cs["dense_val_rl_accuracy"], T["total_sparsity"][i], L.POLICIES[i])
        ok_r &= abs(sum(cs["P2_coefficients"])) < 1e-12
    check("reward tables = formulas; every feasible reward > 1 > 0 > every infeasible reward; oracle = rank 1 = max feasible "
          "sparsity; P2 coefficients sum to 0", ok_r, f"tau {cfg['tau']}")
    for s, cond in (("cifar10_lenet5", "C1"), ("cifar10_lenet5", "C2"), ("cifar100_resnet8", "C0"), ("cifar100_resnet8", "C1")):
        land = json.load(open(os.path.join(L.LAND, f"{s}_landscape.json"), encoding="utf-8"))["policies"]
        cache = {tuple(p["actions"]): (p["val_rl_accuracy"], p["total_sparsity"]) for p in land}
        x1, p1, _ = short(s, cond, cfg, bundles, cache)
        x2, p2, _ = short(s, cond, cfg, bundles, None)
        check(f"{s} {cond}: landscape cache == uncached environment (64 episodes, bit-identical)", x1.equals(x2) and torch.equal(p1, p2))
    # archive selection rule vs an independent re-implementation on synthetic candidates
    rr = random.Random(7)
    ok_s = True
    for trial in range(200):
        cands = [{"r0_score_val_select": rr.choice([1.9, 2.0, 2.1]), "val_rl_reward": rr.choice([1.1, 1.2, 1.3]), "first_episode": rr.randint(1, 512),
                  "feasible_val_select": rr.random() < 0.6, "sparsity": rr.choice([10.0, 20.0, 30.0]), "val_select_accuracy": rr.choice([70.0, 71.0]),
                  "q_val_select": rr.choice([0.95, 0.97, 0.99])} for _ in range(10)]
        for cond in ("C0", "C1"):
            got, _ = SA.choose(cands, cond)
            if cond == "C0":
                best = max(cands, key=lambda c: (c["r0_score_val_select"], c["val_rl_reward"], -c["first_episode"]))
            else:
                f = [c for c in cands if c["feasible_val_select"]]
                best = max(f, key=lambda c: (c["sparsity"], c["val_select_accuracy"], c["val_rl_reward"], -c["first_episode"])) if f else \
                    max(cands, key=lambda c: (c["q_val_select"], c["sparsity"], c["val_rl_reward"], -c["first_episode"]))
            ok_s &= got is best
    check("archive selection rule = independent re-implementation (200 synthetic candidate sets x C0/C1)", ok_s)
    src = open(os.path.join(os.path.dirname(__file__), "run.py"), encoding="utf-8").read()
    check("run.py contains no TEST or VAL-SELECT access", not re.search(r"test_loader|test_tensors|select_loader|select_tensors|freeze\(", src))
    seeds = set()
    for p in glob.glob(os.path.join(L.ROOT, "results", "**", "*.json"), recursive=True):
        if os.sep + "phase5" + os.sep in p:
            continue
        try:
            r = json.load(open(p, encoding="utf-8"))
        except Exception:                                         # noqa: BLE001
            continue
        for x in (r if isinstance(r, list) else [r]):
            if isinstance(x, dict) and isinstance(x.get("seed"), int):
                seeds.add(x["seed"])
    check("seeds 500-519 unused by earlier experiments", not set(L.SEEDS) & seeds, f"highest earlier seed {max(seeds)}")
    t0 = time.time()
    s = "cifar10_simplecnn"
    land = json.load(open(os.path.join(L.LAND, f"{s}_landscape.json"), encoding="utf-8"))["policies"]
    _, _, ag = short(s, "C1", cfg, bundles, {tuple(p["actions"]): (p["val_rl_accuracy"], p["total_sparsity"]) for p in land},
                     timesteps=L.TOTAL_TIMESTEPS)
    secs = time.time() - t0
    check("full budget (2,048 timesteps) on the slowest setting; runtime within budget (<= 15 min at 1 thread)",
          ag.num_timesteps == 2048 and secs <= 900, f"{secs:.0f} s (outcome not inspected)")
    n = sum(ok for _, ok, _ in RES)
    lines = ["# Phase 5 implementation checks", "", f"Run {datetime.datetime.now().astimezone().isoformat(timespec='seconds')} before any "
             f"confirmatory (seed 500-519) run; seed 42 and synthetic inputs only; 1 thread. **{n}/{len(RES)} passed.**", "",
             "| Status | Check | Detail |", "|---|---|---|"] + [f"| {'PASS' if ok else 'FAIL'} | {nm} | {dt.replace('|', '/')[:300]} |" for nm, ok, dt in RES]
    open(os.path.join(L.RESULTS, "phase5_implementation_checks.md"), "w", encoding="utf-8").write("\n".join(lines) + "\n")
    print(f"{n}/{len(RES)} passed")


if __name__ == "__main__":
    main()
