"""The fifteen pre-registered implementation checks for the archive-confirmatory experiment.

Uses only seed 42 (non-experimental) for training runs and synthetic inputs
for the selection rule. Writes results/archive_implementation_checks.md.
Invoked as `python evaluation/archive_confirm.py check`.
"""
import datetime
import glob
import json
import os
import re

import numpy as np
import pandas as pd
import torch

import archive_confirm as ac
import constrained_ppo as cp
import ppo_experiments as base
import soft_prior as sp


def run(args):
    ctx, data = ac.context(args.threads)
    S = np.array(ctx["S"])
    lines = ["# Archive-confirmatory experiment — implementation checks", "",
             f"Run {datetime.datetime.now().astimezone().isoformat(timespec='seconds')} with "
             f"`python evaluation/archive_confirm.py check`, after the pre-registration commit (`1bda844`) and "
             f"before any experimental run. Training checks use seed 42 only (non-experimental).", ""]
    ok_all = True

    def report(n, title, ok, detail):
        nonlocal ok_all
        ok_all &= bool(ok)
        lines.extend([f"## {n}. {title} — {'PASS' if ok else 'FAIL'}", "", detail, ""])
        print(f"check {n}: {'PASS' if ok else 'FAIL'} - {title}", flush=True)

    tmp = os.path.join(ac.RESULTS, "_archive_check")

    def paths(tag):
        d = os.path.join(tmp, tag)
        return {"dir": d, "archive": os.path.join(d, "archive.csv"), "curve": os.path.join(d, "training.csv"),
                "selected": os.path.join(d, "selected.json"), "record": os.path.join(d, "run.json"),
                "pred": os.path.join(d, "pred.csv"), "agent": os.path.join(d, "agent")}

    # instrument the test loader: record the phase and frozen flag of every call
    test_calls = []
    original_test = data.test_loader

    def watched_test_loader():
        test_calls.append((data.phase, data.frozen))
        return original_test()
    data.test_loader = watched_test_loader

    runs = {}
    for tag, cond, policy in (("A0_soft", "A0", "soft"), ("A0_mlp", "A0", "mlp"), ("A1_soft", "A1", "soft")):
        cp._EVAL_CACHE.clear()
        runs[tag] = ac.run_one(ctx, data, cond, 42, out=paths(tag), policy=policy)

    # 1
    same = pd.read_csv(paths("A0_soft")["curve"]).equals(pd.read_csv(paths("A0_mlp")["curve"]))
    report(1, "A0 reproduces ordinary PPO when beta = 0", same and runs["A0_soft"]["terminal_actions"] == runs["A0_mlp"]["terminal_actions"],
           f"A0 with SoftPriorPolicy (beta 0) vs SB3 MlpPolicy on V_RL, seed 42: all 512 training episodes identical = "
           f"**{same}**; terminal policies {runs['A0_soft']['terminal_actions']} vs {runs['A0_mlp']['terminal_actions']}.")

    # 2
    sens = json.load(open(ac.SENS_JSON, encoding="utf-8"))
    c1 = ac.condition("A1", 200, S)
    ok2 = (c1["prior_sensitivity"] == sens["normalized_sensitivity"] and c1["beta"] == 0.5
           and np.array_equal(np.array(sp.prior_matrix(0.5, c1["prior_sensitivity"])),
                              0.5 * S[:, None] * sp.A_NORM[None, :]))
    report(2, "A1 uses the correct sensitivity vector", ok2,
           f"A1 prior = {c1['prior_sensitivity']} = `archive_sensitivity_vector.json` (sha256 "
           f"`{ac.sha256(ac.SENS_JSON)[:16]}…`), beta 0.5; prior matrix = beta·S·a_norm exactly.")

    # 3
    prereg = open(ac.PREREG, encoding="utf-8").read()
    ok3 = True
    for s in ac.SEEDS:
        vec = np.array(ac.condition("A2", s, S)["prior_sensitivity"])
        listed = [float(x) for x in re.search(rf"^\| {s} \| ([^|]+) \|", prereg, re.M).group(1).split(",")]
        ok3 &= np.allclose(np.round(vec, 6), listed, atol=1e-9) and np.array_equal(np.sort(vec), np.sort(S))
        ok3 &= all(not np.isclose(vec[i], S[i]) for i in range(4))
    report(3, "A2 has identical sensitivity values but wrong assignment", ok3,
           "All 40 seeds: A2 vector is a permutation of S, every layer's value differs from its own, and it equals the "
           "vector listed in the pre-registration.")

    # 4
    c3 = np.array(ac.condition("A3", 200, S)["prior_sensitivity"])
    m1, m3 = np.mean(sp.prior_matrix(0.5, S)), np.mean(sp.prior_matrix(0.5, c3))
    report(4, "A3 has identical average prior strength", bool(np.all(c3 == S.mean()) and np.isclose(m1, m3, atol=1e-15)),
           f"A3 vector = mean(S) = {S.mean()!r} for every layer; mean penalty A1 {m1:.12f}, A3 {m3:.12f}.")

    # 5
    env = cp.MaskedCompressionEnv(ctx["model"], ctx["filtered"], ctx["baseline_v_rl"], data.rl_loader(),
                                  ctx["max_param_count"], layer_sensitivities={n: 0.0 for n in ac.LAYERS})
    obs = np.stack([sp.policy_obs(ctx["filtered"], i, 0.0, {n: 0.0 for n in ac.LAYERS}, ctx["max_param_count"])
                    for i in range(4)])
    mins = {}
    for name in ac.CONDITIONS:
        c = ac.condition(name, 200, S)
        m = sp.PPO(env=env, seed=200, policy=sp.SoftPriorPolicy,
                   policy_kwargs={"prior": sp.prior_matrix(c["beta"], c["prior_sensitivity"])},
                   **{k: v for k, v in base.PPO_KWARGS.items() if k != "policy"})
        with torch.no_grad():
            mins[name] = float(m.policy.get_distribution(torch.as_tensor(obs)).distribution.probs.min())
    report(5, "All actions remain available", all(v > 0 for v in mins.values()),
           "Minimum action probability of untrained policies (seed-200 priors; no training) over the four layer "
           "observations: " + ", ".join(f"{k} {v:.4f}" for k, v in mins.items()) + ". No masking anywhere.")

    # 6 + 7
    raised_sel = raised_test_train = False
    with data.training():
        try:
            ac.evaluate_select(ctx, data, [0, 0, 0, 0])
        except RuntimeError:
            raised_sel = True
        try:
            original_test()
        except RuntimeError:
            raised_test_train = True
    raised_test_unfrozen = False
    try:
        original_test()
    except RuntimeError:
        raised_test_unfrozen = True
    report(6, "V_SELECT is inaccessible during PPO training", raised_sel,
           f"Inside `ArchiveData.training()` (wrapping `agent.learn`), evaluating a policy on V_SELECT raises: **{raised_sel}**.")
    report(7, "TEST is inaccessible before archive selection", raised_test_train and raised_test_unfrozen,
           f"TEST raises during training (**{raised_test_train}**) and after training until a selection is frozen "
           f"(**{raised_test_unfrozen}**).")

    # 8
    ok8 = True
    for tag in ("A0_soft", "A1_soft"):
        curve = pd.read_csv(paths(tag)["curve"])
        arch = pd.read_csv(paths(tag)["archive"])
        ok8 &= set(curve["actions"]) == set(arch["actions"]) and arch["times_sampled"].sum() == len(curve) == 512
        ok8 &= not arch["actions"].duplicated().any()
    report(8, "The archive stores every unique completed policy", ok8,
           "For both seed-42 check runs: the archive's policy set equals the set of policies completed in the 512 "
           "training episodes, with no duplicates, and times_sampled sums to 512.")

    # 9
    unique_total = len(set(pd.read_csv(paths("A0_soft")["archive"])["actions"]) |
                       set(pd.read_csv(paths("A0_mlp")["archive"])["actions"]) |
                       set(pd.read_csv(paths("A1_soft")["archive"])["actions"]) |
                       {str(r["terminal_actions"]) for r in runs.values()})
    report(9, "Duplicate policies are not repeatedly evaluated", ac.SELECT_EVALUATIONS[0] == len(ac._SELECT_CACHE) == unique_total,
           f"V_SELECT evaluations performed = {ac.SELECT_EVALUATIONS[0]}; cached policies = {len(ac._SELECT_CACHE)}; "
           f"distinct policies across the three runs' archives and terminals = {unique_total}.")

    # 10 + 11
    syn = pd.DataFrame({"actions": [[0, 0, 5, 5], [1, 0, 5, 5], [0, 1, 5, 5], [2, 0, 5, 5], [0, 4, 4, 5], [0, 0, 5, 4]],
                        "sparsity": [57.9, 57.9, 58.2, 58.1, 56.7, 49.0],
                        "vselect_accuracy": [77.0, 77.0, 77.0, 76.0, 80.0, 81.0],
                        "vselect_loss": [0.60, 0.59, 0.50, 0.40, 0.3, 0.2]})
    a, band_hit, n_band = ac.select(syn)
    shuffled_pick = ac.select(syn.sample(frac=1, random_state=3))[0]
    lex = pd.DataFrame({"actions": [[1, 0, 5, 5], [0, 0, 5, 5]], "sparsity": [58.0, 58.0],
                        "vselect_accuracy": [77.0, 77.0], "vselect_loss": [0.5, 0.5]})
    none = pd.DataFrame({"actions": [[0, 4, 4, 5], [0, 5, 4, 5]], "sparsity": [56.7, 57.3],
                         "vselect_accuracy": [80.0, 70.0], "vselect_loss": [0.3, 0.9]})
    fb, fb_hit, _ = ac.select(none)
    edge = pd.DataFrame({"actions": [[0, 0, 0, 1], [0, 0, 0, 2], [0, 0, 0, 3]], "sparsity": [57.5, 58.5, 58.5000001],
                         "vselect_accuracy": [70.0, 71.0, 99.0], "vselect_loss": [1, 1, 1]})
    ok10 = (a["actions"] == [1, 0, 5, 5] and band_hit and n_band == 4 and shuffled_pick["actions"] == [1, 0, 5, 5]
            and ac.select(lex)[0]["actions"] == [0, 0, 5, 5] and fb["actions"] == [0, 5, 4, 5] and not fb_hit)
    report(10, "The selection rule is deterministic", ok10,
           "Synthetic archives: out-of-band high accuracy is ignored; equal accuracy is broken by distance to 58.0%, "
           "then lower loss ([1,0,5,5] over [0,0,5,5] and [0,1,5,5]); equal everything by the lexicographically smaller "
           "action vector; row order does not matter; with no band candidate the policy closest to 58.0% is chosen "
           "and flagged.")
    ok11 = ac.BAND == (57.5, 58.5) and ac.TARGET == 58.0 and ac.select(edge)[0]["actions"] == [0, 0, 0, 2]
    report(11, "The target sparsity interval is fixed", ok11,
           f"Constants BAND = {ac.BAND}, TARGET = {ac.TARGET} (as pre-registered). Boundaries are inclusive: 57.5 and "
           f"58.5 are in the band, 58.5000001 is not.")

    # 12 + 13
    ok12 = True
    for tag in ("A0_soft", "A0_mlp", "A1_soft"):
        p = paths(tag)
        ok12 &= os.path.getmtime(p["selected"]) <= os.path.getmtime(p["pred"])
        ok12 &= json.load(open(p["record"]))["frozen_policy_sha256"] == ac.sha256(p["selected"])
    report(12, "The final selected policy is frozen before test evaluation", ok12,
           "For each check run the selected-policy file exists before the prediction file, and its sha256 equals "
           "the hash recorded in the run record.")
    ok13 = len(test_calls) == 3 and all(ph == "selection" and fr for ph, fr in test_calls)
    report(13, "The test set is evaluated only after selection", ok13,
           f"Every test-loader request during the three check runs came in the selection phase with a frozen policy: "
           f"{test_calls}.")

    # 14 + 15
    used = set()
    for folder in ("runs", "refined_runs", "constrained_runs", "soft_prior_runs", "exploration_levelA_runs"):
        used |= {json.load(open(p))["seed"] for p in glob.glob(os.path.join(ac.RESULTS, folder, "*.json"))}
    report(14, "Seeds are fresh", not (set(ac.SEEDS) & used) and len(ac.SEEDS) == 40,
           f"Locked seeds 200–239 vs every earlier run record: overlap {sorted(set(ac.SEEDS) & used)}.")
    report(15, "Baseline hash unchanged", ac.sha256(os.path.join(ac.ROOT, "checkpoints", "cnn_baseline_FIXED.pth")) == ac.BASELINE_SHA,
           f"sha256 {ac.BASELINE_SHA}.")

    lines.insert(3, f"**Overall: {'ALL CHECKS PASS' if ok_all else 'AT LEAST ONE CHECK FAILED'}**\n")
    with open(os.path.join(ac.RESULTS, "archive_implementation_checks.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print("ALL PASS" if ok_all else "FAILED", flush=True)
