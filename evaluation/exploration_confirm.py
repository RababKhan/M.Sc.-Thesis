"""Confirmatory study: sensitivity-guided exploration (results/exploration_levelA_preregistration.md).

One module holds the runner, the trajectory metrics and the statistics, so the
implementation checks, the 120 runs and the final analysis share one code
path. Training reuses the unmodified soft-prior machinery
(evaluation/soft_prior.py: SoftPriorPolicy, train()).

No peeking: the runner prints only run identifiers, runtimes and errors; the
per-run result line printed by soft_prior.train is swallowed.

Subcommands
  check   the ten pre-registered implementation checks (seed 42 and synthetic data only)
  run     the 120 pre-registered runs, resumable: --worker K --workers 4
"""
import argparse
import contextlib
import datetime
import io
import json
import os
import re
import sys
import time

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import constrained_ppo as cp  # noqa: E402
import ppo_experiments as base  # noqa: E402
import soft_prior as sp  # noqa: E402

ROOT = base.ROOT
RESULTS = os.path.join(ROOT, "results")
PREREG = os.path.join(RESULTS, "exploration_levelA_preregistration.md")
LANDSCAPE = os.path.join(RESULTS, "policy_landscape_validation.csv")
RUNS_DIR = os.path.join(RESULTS, "exploration_levelA_runs")
CURVES_DIR = os.path.join(RESULTS, "exploration_levelA_training_curves_raw")
AGENTS_DIR = os.path.join(ROOT, "checkpoints", "exploration_levelA_agents")

SEEDS = list(range(100, 130))
CONDITIONS = ["E0", "E1", "E2", "E3"]
BETA = 0.50
N_EPISODES = 512
STEPS_PER_EPISODE = 4
TARGET_VAL, TARGET_SPARSITY = 76.0, 55.0
DESTRUCTIVE_F0_ACTION = 4          # features.0 action index >= 4  <=>  ratio >= 40%
T975 = {n: t for n, t in zip(range(1, 31), [
    12.706, 4.303, 3.182, 2.776, 2.571, 2.447, 2.365, 2.306, 2.262, 2.228, 2.201, 2.179, 2.160, 2.145, 2.131,
    2.120, 2.110, 2.101, 2.093, 2.086, 2.080, 2.074, 2.069, 2.064, 2.060, 2.056, 2.052, 2.048, 2.045, 2.042])}


# ------------------------------------------------------------------ design
def condition(name, seed):
    S = sp.sensitivities()["loss"]
    zero = [0.0] * 4
    if name == "E0":
        return {"condition": "E0", "definition": "none", "beta": 0.0, "prior_sensitivity": zero,
                "state_sensitivity": zero, "prior_mapping": None}
    if name == "E1":
        prior, mapping = S, None
    elif name == "E2":
        prior, mapping = sp.shuffled(S, seed)
    else:
        prior, mapping = np.full(4, S.mean()), None
    return {"condition": name, "definition": "loss", "beta": BETA, "prior_sensitivity": [float(x) for x in prior],
            "state_sensitivity": zero, "prior_mapping": mapping}


def run_id(name, seed):
    return f"explore_{name}_seed-{seed}"


# ------------------------------------------------------------------ trajectory metrics
def load_landscape():
    return pd.read_csv(LANDSCAPE).set_index("actions")


def trajectory(curve, landscape):
    """Checkpoint table for one run: one row per training episode (the pre-registered schedule)."""
    t = curve[["episode", "timestep", "actions"]].copy().sort_values("episode").reset_index(drop=True)
    assert len(t) == N_EPISODES and t["episode"].tolist() == list(range(1, N_EPISODES + 1)), "episode schedule"
    assert (t["timestep"] == STEPS_PER_EPISODE * t["episode"]).all(), "checkpoint timesteps"
    t["val_accuracy"] = t["actions"].map(landscape["val_accuracy"])
    t["sparsity"] = t["actions"].map(landscape["total_sparsity"])
    t["reward_rank"] = t["actions"].map(landscape["reward_rank_0.01"])
    t["rank_score"] = (1296 - t["reward_rank"]) / 1295
    t["utility"] = 1.5 * t["val_accuracy"] / 77.32 + 0.01 * t["sparsity"]
    acts = t["actions"].map(json.loads)
    for i, layer in enumerate(sp.LAYERS):
        t[f"ratio_{layer}"] = acts.map(lambda a: int(100 * sp.rl_env.ACTION_TO_PRUNE[a[i]]))
    t["destructive_f0"] = acts.map(lambda a: a[0] >= DESTRUCTIVE_F0_ACTION)
    assert t[["val_accuracy", "sparsity", "reward_rank"]].notna().all().all()
    return t


def auc(t, y):
    """Trapezoidal area of y over t, divided by the duration: on the scale of y."""
    t, y = np.asarray(t, float), np.asarray(y, float)
    return float(np.sum((y[1:] + y[:-1]) * np.diff(t)) / 2.0 / (t[-1] - t[0]))


def run_metrics(traj):
    hit = traj[(traj["val_accuracy"] >= TARGET_VAL) & (traj["sparsity"] >= TARGET_SPARSITY)]
    return {
        "val_accuracy_auc": auc(traj["timestep"], traj["val_accuracy"]),                 # primary
        "rank_score_auc": auc(traj["timestep"], traj["rank_score"]),                     # S1
        "utility_auc": auc(traj["timestep"], traj["utility"]),                           # S2
        "destructive_f0_frequency": float(traj["destructive_f0"].mean()),                # S3
        "worst_trajectory_val_accuracy": float(traj["val_accuracy"].min()),              # S4
        "destructive_f0_count": int(traj["destructive_f0"].sum()),
        "mean_f0_pruning": float(traj["ratio_features.0"].mean()),
        "max_f0_pruning": int(traj["ratio_features.0"].max()),
        "mean_val_accuracy_plain": float(traj["val_accuracy"].mean()),
        "target_reached": len(hit) > 0,
        "target_first_episode": int(hit["episode"].iloc[0]) if len(hit) else None,
        "target_first_timestep": int(hit["timestep"].iloc[0]) if len(hit) else None,
    }


# ------------------------------------------------------------------ statistics (seed = unit)
def _half_sums(d):
    m = len(d)
    signs = 1 - 2 * ((np.arange(2 ** m)[:, None] >> np.arange(m)) & 1)
    return signs @ d


def sign_flip_p(d):
    """Exact two-sided sign-flip p for the mean of paired differences, all 2^n patterns (meet-in-the-middle)."""
    d = np.asarray(d, float)
    n = len(d)
    obs = abs(d.sum())
    if obs == 0:
        return 1.0
    thr = obs - 1e-9 * max(1.0, obs)
    a, b = _half_sums(d[: n // 2]), np.sort(_half_sums(d[n // 2:]))
    upper = len(b) - np.searchsorted(b, thr - a, side="left")
    lower = np.searchsorted(b, -thr - a, side="right")
    return float((upper.sum() + lower.sum()) / 2.0 ** n)


def wilcoxon_p(d):
    """Exact two-sided Wilcoxon signed-rank p (zeros dropped, average ranks) by dynamic programming."""
    d = np.asarray([x for x in d if abs(x) > 1e-12], float)
    if len(d) == 0:
        return 1.0
    r2 = np.round(2 * pd.Series(np.abs(d)).rank().to_numpy()).astype(int)
    counts = np.zeros(r2.sum() + 1)
    counts[0] = 1
    for r in r2:
        counts[r:] = counts[r:] + counts[: len(counts) - r].copy() if r else counts
    counts /= counts.sum()
    centre = r2.sum() / 2
    w = r2[d > 0].sum()
    values = np.arange(len(counts))
    return float(counts[np.abs(values - centre) >= abs(w - centre) - 1e-9].sum())


def describe(d, rng):
    d = np.asarray(d, float)
    n = len(d)
    sd = d.std(ddof=1)
    half = T975[n - 1] * sd / np.sqrt(n)
    boots = d[rng.integers(0, n, (10000, n))].mean(axis=1)
    return {"n_seeds": n, "mean_diff": d.mean(), "median_diff": float(np.median(d)), "sd_diff": sd,
            "t_ci95_low": d.mean() - half, "t_ci95_high": d.mean() + half,
            "boot_ci95_low": float(np.percentile(boots, 2.5)), "boot_ci95_high": float(np.percentile(boots, 97.5)),
            "cohens_dz": d.mean() / sd if sd > 0 else float("nan"),
            "sign_flip_p": sign_flip_p(d), "wilcoxon_p": wilcoxon_p(d),
            "n_positive": int((d > 1e-12).sum()), "n_negative": int((d < -1e-12).sum())}


def paired(table, endpoint, a, b, rng, seeds=SEEDS):
    """E_a − E_b on one endpoint. `table` must have exactly one row per (condition, seed)."""
    assert not table.duplicated(["condition", "seed"]).any(), "more than one row per (condition, seed)"
    A = table[table.condition == a].set_index("seed")[endpoint]
    B = table[table.condition == b].set_index("seed")[endpoint]
    assert sorted(A.index) == sorted(seeds) and sorted(B.index) == sorted(seeds), "seed set is not the locked set"
    return describe((A.loc[seeds] - B.loc[seeds]).to_numpy(float), rng)


def holm(p):
    p = np.asarray(p, float)
    out, running = np.empty(len(p)), 0.0
    for rank, i in enumerate(np.argsort(p)):
        running = max(running, min(1.0, (len(p) - rank) * p[i]))
        out[i] = running
    return out


# ------------------------------------------------------------------ provenance
def context(threads):
    ctx = sp.context(threads)
    here = os.path.dirname(os.path.abspath(__file__))
    prov = ctx["provenance"]
    prov["soft_prior_preregistration_sha256"] = prov.pop("preregistration_sha256")
    prov["preregistration_sha256"] = base.sha256(PREREG)
    prov["landscape_sha256"] = base.sha256(LANDSCAPE)
    prov["script_sha256"]["exploration_confirm.py"] = base.sha256(os.path.join(here, "exploration_confirm.py"))
    return ctx


def quiet_train(ctx, c, seed, **kwargs):
    with contextlib.redirect_stdout(io.StringIO()):
        return sp.train(ctx, c, seed, **kwargs)


# ------------------------------------------------------------------ run
def cmd_run(args):
    ctx = context(args.threads)
    jobs = [(name, s) for s in SEEDS for name in CONDITIONS]
    mine = [j for i, j in enumerate(jobs) if i % args.workers == args.worker]
    print(f"worker {args.worker}/{args.workers}: {len(mine)} of {len(jobs)} runs", flush=True)
    for i, (name, s) in enumerate(mine, 1):
        rid = run_id(name, s)
        if os.path.exists(os.path.join(RUNS_DIR, f"{rid}.json")):
            print(f"[{args.worker}:{i}/{len(mine)}] {rid} already done", flush=True)
            continue
        started = time.time()
        quiet_train(ctx, condition(name, s), s, out=(RUNS_DIR, CURVES_DIR, AGENTS_DIR), rid=rid)
        print(f"[{args.worker}:{i}/{len(mine)}] {rid} done ({time.time() - started:.0f}s)", flush=True)
    print(f"WORKER {args.worker} DONE", flush=True)


# ------------------------------------------------------------------ checks
def cmd_check(args):
    ctx = context(args.threads)
    L = load_landscape()
    lines = ["# Confirmatory exploration study — implementation checks", "",
             f"Run {datetime.datetime.now().astimezone().isoformat(timespec='seconds')} with "
             f"`python evaluation/exploration_confirm.py check`, after the pre-registration commit and before any "
             f"fresh-seed run. Only seed 42 (non-experimental, used in earlier studies) and synthetic data are "
             f"used; no fresh-seed outcome exists or is inspected.", ""]
    ok_all = True

    def report(n, title, ok, detail):
        nonlocal ok_all
        ok_all &= bool(ok)
        lines.extend([f"## {n}. {title} — {'PASS' if ok else 'FAIL'}", "", detail, ""])
        print(f"check {n}: {'PASS' if ok else 'FAIL'} - {title}", flush=True)

    tmp = os.path.join(RESULTS, "_exploration_check")
    old = os.path.join(RESULTS, "soft_prior_training_curves_raw")
    curves = {}
    for name in CONDITIONS:
        cp._EVAL_CACHE.clear()
        quiet_train(ctx, condition(name, 42), 42, out=(os.path.join(tmp, name),) * 3, rid=f"check_{name}")
        curves[name] = pd.read_csv(os.path.join(tmp, name, f"check_{name}.csv"))
    cp._EVAL_CACHE.clear()
    quiet_train(ctx, condition("E0", 42), 42, policy="MlpPolicy", out=(os.path.join(tmp, "mlp"),) * 3, rid="check_mlp")
    mlp = pd.read_csv(os.path.join(tmp, "mlp", "check_mlp.csv"))

    # 1
    same_mlp = curves["E0"].equals(mlp)
    same_p0 = curves["E0"].equals(pd.read_csv(os.path.join(old, "softprior_P0_def-none_beta-0.00_seed-42.csv")))
    report(1, "beta = 0 reproduces E0 exactly (standard PPO)", same_mlp and same_p0,
           f"E0 (SoftPriorPolicy, beta 0) vs SB3 MlpPolicy, seed 42: all 512 training episodes identical = "
           f"**{same_mlp}**. E0 vs the recorded soft-prior P0 run (seed 42): identical = **{same_p0}**.")

    # 2
    S = sp.sensitivities()["loss"]
    same_p1 = curves["E1"].equals(pd.read_csv(os.path.join(old, "softprior_P1_def-loss_beta-0.50_seed-42.csv")))
    same_matrix = np.array_equal(np.array(sp.prior_matrix(BETA, condition("E1", 42)["prior_sensitivity"])),
                                 np.array(sp.prior_matrix(0.5, S)))
    report(2, "The correct prior reproduces the previously implemented P1 mechanism", same_p1 and same_matrix,
           f"E1 prior matrix equals soft-prior P1's (beta 0.5, loss S) = **{same_matrix}**. E1 on seed 42 reproduces "
           f"the recorded soft-prior P1 run, all 512 episodes = **{same_p1}**.")

    # 3
    prereg = open(PREREG, encoding="utf-8").read()
    ok3, n_rows = True, 0
    for s in SEEDS:
        c = condition("E2", s)
        m = re.search(rf"^\| {s} \| ([^|]+) \|", prereg, re.M)
        listed = [float(x) for x in m.group(1).split(",")]
        vec = np.array(c["prior_sensitivity"])
        ok3 &= np.allclose(np.round(vec, 6), listed, atol=1e-9)
        ok3 &= np.array_equal(np.sort(vec), np.sort(S))
        ok3 &= all(not np.isclose(vec[i], S[i]) for i in range(4))
        n_rows += 1
    report(3, "Shuffled prior uses exactly the same sensitivity-value multiset", ok3,
           f"For all {n_rows} locked seeds: the E2 vector is a permutation of S (sorted vectors equal), every layer's "
           f"value differs from its correct value, and it matches the vector listed in the pre-registration.")

    # 4
    e3 = np.array(condition("E3", 100)["prior_sensitivity"])
    ok4 = bool(np.all(e3 == S.mean()) and S.mean() == 0.26140902642330055
               and np.isclose(np.mean(sp.prior_matrix(BETA, e3)), np.mean(sp.prior_matrix(BETA, S)), atol=1e-15))
    report(4, "Constant prior uses exactly the mean sensitivity", ok4,
           f"E3 vector = {e3.tolist()}; mean(S) = {S.mean()!r}; mean penalty E3 = "
           f"{np.mean(sp.prior_matrix(BETA, e3)):.12f}, E1 = {np.mean(sp.prior_matrix(BETA, S)):.12f}.")

    # 5
    env = cp.MaskedCompressionEnv(ctx["model"], ctx["filtered"], ctx["baseline_val"], ctx["splits"].probe_loader(),
                                  ctx["max_param_count"], layer_sensitivities={n: 0.0 for n in sp.LAYERS})
    obs = np.stack([sp.policy_obs(ctx["filtered"], i, 0.0, {n: 0.0 for n in sp.LAYERS}, ctx["max_param_count"])
                    for i in range(4)])
    mins = {}
    for name in CONDITIONS:
        for s in (100, 129):
            c = condition(name, s)
            model = sp.PPO(env=env, seed=s, policy=sp.SoftPriorPolicy,
                           policy_kwargs={"prior": sp.prior_matrix(c["beta"], c["prior_sensitivity"])},
                           **{k: v for k, v in base.PPO_KWARGS.items() if k != "policy"})
            with torch.no_grad():
                dist = model.policy.get_distribution(torch.as_tensor(obs)).distribution
            mins[f"{name}/seed {s}"] = float(dist.probs.min())
            assert torch.isfinite(dist.logits).all()
    report(5, "All actions keep non-zero probability", all(v > 0 for v in mins.values()),
           "Minimum action probability over the four layer observations, untrained policies built with the "
           "experimental seeds' priors (no training, no outcome): "
           + ", ".join(f"{k}: {v:.4f}" for k, v in mins.items()) + ".")

    # 6
    raised = False
    with ctx["splits"].training():
        try:
            ctx["splits"].test_loader()
        except RuntimeError:
            raised = True
    report(6, "Test set is inaccessible during training", raised,
           f"Inside `Splits.training()` (wrapping every `agent.learn`) the test loader raises: **{raised}**.")

    # 7
    trajs = {n: trajectory(curves[n], L) for n in CONDITIONS}
    same_schedule = all(trajs[n]["timestep"].tolist() == [4 * k for k in range(1, 513)] for n in CONDITIONS)
    report(7, "Evaluation checkpoints are identical across conditions", same_schedule,
           "All four seed-42 check runs (E0–E3) have exactly 512 checkpoints at timesteps 4, 8, …, 2048; "
           "`trajectory()` asserts this for every run and fails otherwise.")

    # 8
    syn_t = np.arange(4, 2049, 4)
    ok8 = (np.isclose(auc(syn_t, np.full(512, 70.0)), 70.0) and np.isclose(auc(syn_t, syn_t), (4 + 2048) / 2)
           and np.isclose(auc([0, 1, 2], [0, 1, 0]), 0.5))
    one = run_metrics(trajs["E0"])
    two = run_metrics(trajectory(curves["E0"].sample(frac=1, random_state=1), L))
    ok8 &= one == two
    report(8, "Trajectory metrics are computed identically", ok8,
           "A single function (`trajectory` → `run_metrics`) computes every endpoint for every condition. Synthetic "
           "checks: constant y → AUC = y; linear y = t → AUC = midpoint; triangle → 0.5. Shuffling a curve's rows "
           "gives identical metrics (rows are ordered by episode first).")

    # 9
    again = {n: run_metrics(trajectory(curves[n], L)) for n in CONDITIONS}
    ok9 = all(again[n] == run_metrics(trajs[n]) for n in CONDITIONS)
    report(9, "AUC implementation is deterministic", ok9,
           "Recomputing all endpoints for the four check runs gives bit-identical values.")

    # 10
    rng = np.random.default_rng(5)
    ok10 = True
    for n in (6, 10, 14, 16):
        for _ in range(5):
            d = np.round(rng.normal(0, 1, n), 3)
            brute = np.abs((1 - 2 * ((np.arange(2 ** n)[:, None] >> np.arange(n)) & 1)) @ d)
            ok10 &= np.isclose(sign_flip_p(d), float((brute >= abs(d.sum()) - 1e-9).mean()))
            dd = d[np.abs(d) > 1e-12]
            r = pd.Series(np.abs(dd)).rank().to_numpy()
            pat = ((np.arange(2 ** len(dd))[:, None] >> np.arange(len(dd))) & 1)
            w = pat @ r
            ok10 &= np.isclose(wilcoxon_p(d), float((np.abs(w - r.sum() / 2) >= abs(r[dd > 0].sum() - r.sum() / 2) - 1e-9).mean()))
    fake = pd.DataFrame([{"condition": c_, "seed": s, "x": rng.normal()} for c_ in CONDITIONS for s in SEEDS])
    ok10 &= paired(fake, "x", "E1", "E0", rng)["n_seeds"] == 30
    episode_level = pd.concat([fake, fake])
    try:
        paired(episode_level, "x", "E1", "E0", rng)
        ok10 = False
    except AssertionError:
        pass
    d30 = rng.normal(0.2, 1, 30)
    mc = np.abs((rng.choice([-1, 1], (400000, 30)) * d30).sum(axis=1))
    ok10 &= abs(sign_flip_p(d30) - float((mc >= abs(d30.sum()) - 1e-9).mean())) < 0.005
    report(10, "Statistics use seeds, not episodes, as observations; exact tests are exact", ok10,
           "`paired()` requires exactly one row per (condition, seed) and exactly the 30 locked seeds, and raises on "
           "episode-level or duplicated input (verified). The meet-in-the-middle sign-flip p equals brute-force "
           "enumeration for n = 6, 10, 14, 16 (20 random cases), and matches a 400,000-draw Monte Carlo estimate for "
           "n = 30 within 0.005. The dynamic-programming Wilcoxon p equals brute-force enumeration on the same cases.")

    lines.insert(3, f"**Overall: {'ALL CHECKS PASS' if ok_all else 'AT LEAST ONE CHECK FAILED'}**\n")
    with open(os.path.join(RESULTS, "exploration_levelA_implementation_checks.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print("ALL PASS" if ok_all else "FAILED", flush=True)


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("check")
    c.add_argument("--threads", type=int, default=4)
    r = sub.add_parser("run")
    r.add_argument("--threads", type=int, default=1)
    r.add_argument("--worker", type=int, default=0)
    r.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()
    {"check": cmd_check, "run": cmd_run}[args.cmd](args)


if __name__ == "__main__":
    main()
