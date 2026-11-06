"""Verify that the consolidated src/ library reproduces the historical SimpleCNN results exactly.

Nothing historical is modified; the frozen modules evaluation/common.py and
evaluation/rl_env.py are imported read-only as the reference implementation.

  static        data tensors, splits, model, baseline accuracies, units, sensitivity
                (accuracy-drop and archive V_RL loss), unit pruning, 100 landscape
                policies, matched-sparsity baselines, statistics
  ppo NAME      one complete PPO run through src.rl.PruningEnv, compared episode by
                episode with the recorded training curve and final record
                (NAME: main42 | main1 | main2 | main3 | explore_E1_100)
  report        merge everything into results/reproducibility/library_verification.json

Run from the repository root with venv/Scripts/python.exe.
"""
import argparse
import copy
import glob
import json
import os
import sys
import time

import numpy as np
import pandas as pd
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
sys.path.insert(1, os.path.join(ROOT, "evaluation"))

from src import data as D, models as M, pruning as P, sensitivity as S, statistics as ST  # noqa: E402
from src import rl as RL  # noqa: E402
from src.evaluation import evaluate_accuracy, predictions, accuracy_from  # noqa: E402
from src.utils import sha256_file, sha256_json, environment, git_state  # noqa: E402

OUT_DIR = os.path.join(ROOT, "results", "reproducibility", "library_verification")
FIXED = os.path.join(ROOT, "checkpoints", "cnn_baseline_FIXED.pth")
FIXED_SHA = "ca28f8345c23365ee7b92686e780ae0a9befa76c9d6bb9b3bb0b9e42272e111c"
VAL_HASH_PREFIX = "9d3648af"


def check(results, name, ok, detail):
    results.append({"check": name, "pass": bool(ok), "detail": detail})
    print(f"[{'PASS' if ok else 'FAIL'}] {name}: {detail}", flush=True)


# ------------------------------------------------------------------ static checks
def cmd_static(args):
    import common
    import rl_env
    torch.set_num_threads(args.threads)
    res = []
    t0 = time.time()

    # --- data
    old = rl_env.Splits(os.path.join(ROOT, "data"))
    b = D.DataBundle("cifar10")
    check(res, "train/val indices identical to rl_env.Splits",
          b.train_indices == old.train_indices and b.val_indices == old.val_indices,
          f"val index sha256 {sha256_json(b.val_indices)[:16]}")
    check(res, "validation index hash is the historical 9d3648af...",
          sha256_json(b.val_indices).startswith(VAL_HASH_PREFIX), sha256_json(b.val_indices))
    check(res, "validation tensors bit-identical to the historical cache",
          torch.equal(b.val_x, old._val[0]) and torch.equal(b.val_y, old._val[1]), str(tuple(b.val_x.shape)))
    px, py = next(iter(b.probe_loader()))
    ox, oy = next(iter(old.probe_loader()))
    check(res, "legacy probe = first 1,000 validation images, identical batches",
          torch.equal(px, ox) and torch.equal(py, oy) and len(b.probe_loader().dataset) == 1000, "batch 0 equal")
    npz = np.load(D.SPLIT_FILES["cifar10"])
    check(res, "CIFAR-10 V_RL/V_SELECT = archive split (3,000/2,000, disjoint)",
          len(b.rl_positions) == 3000 and len(b.select_positions) == 2000
          and np.array_equal(b.rl_positions, npz["v_rl_positions"]), sha256_file(D.SPLIT_FILES["cifar10"])[:16])
    rl_again, sel_again, _, _ = D.stratified_split(b.val_y.numpy())
    check(res, "stratified_split regenerates the archive V_RL/V_SELECT split exactly",
          np.array_equal(rl_again, npz["v_rl_positions"]) and np.array_equal(sel_again, npz["v_select_positions"]),
          "seed 20260928")

    # --- guards
    raised = []
    try:
        b.test_loader()
    except RuntimeError as e:
        raised.append(str(e))
    with b.training():
        for fn in (b.test_loader, b.select_loader):
            try:
                fn()
            except RuntimeError as e:
                raised.append(str(e))
    check(res, "guards: test before freeze, test and V_SELECT inside training() all raise",
          len(raised) == 3, "; ".join(raised))

    # --- model and baseline accuracy
    check(res, "cnn_baseline_FIXED.pth sha256 unchanged", sha256_file(FIXED) == FIXED_SHA, FIXED_SHA[:16])
    model = M.load_checkpoint(FIXED, "simplecnn", 10)
    ref = common.load_baseline(FIXED)
    same_sd = all(torch.equal(model.state_dict()[k], ref.state_dict()[k]) for k in ref.state_dict())
    with torch.no_grad():
        same_out = torch.equal(model(b.val_x[:512]), ref(b.val_x[:512]))
    check(res, "src SimpleCNN loads FIXED; identical state dict and logits",
          same_sd and same_out and list(model.state_dict()) == list(ref.state_dict()), "512 validation images")
    val = evaluate_accuracy(model, b.val_loader())
    probe = evaluate_accuracy(model, b.probe_loader())
    b.freeze({"checkpoint": FIXED, "sha256": FIXED_SHA})
    preds, labels = predictions(model, b.test_loader())
    test = accuracy_from(preds, labels)
    old_test = rl_env.evaluate_model(ref, old.test_loader())
    check(res, "test tensors bit-identical to the historical cache",
          torch.equal(b._test[0], old._test[0]) and torch.equal(b._test[1], old._test[1]), "10,000 images")
    check(res, "baseline accuracies val 77.32 / probe 76.10 / test 77.03",
          val == 77.32 and probe == 76.1 and test == 77.03 == old_test, f"val {val}, probe {probe}, test {test}")
    params = M.count_parameters(model)
    check(res, "parameter count 620,362 (weights 619,872)",
          params["total"] == 620362 and params["weights"] == 619872, json.dumps(params))

    # --- units
    units = P.get_units(model, "simplecnn")
    counts = [u.param_count for u in units]
    check(res, "SimpleCNN units = historical FILTERED_LAYERS with 864/18,432/73,728/524,288 weights",
          [u.name for u in units] == common.FILTERED_LAYERS and counts == [864, 18432, 73728, 524288],
          f"{[u.name for u in units]} {counts}")

    # --- sensitivity
    _, filtered = rl_env.prunable_layers(ref)
    old_base, old_sens = rl_env.compute_layer_sensitivities(ref, filtered, old.probe_loader())
    new_base, new_sens = S.accuracy_drop(model, "simplecnn", b.probe_loader())
    check(res, "accuracy-drop sensitivity identical to rl_env.compute_layer_sensitivities",
          old_base == new_base and old_sens == new_sens, json.dumps(new_sens))
    arch_json = json.load(open(os.path.join(ROOT, "results", "archive_sensitivity_vector.json")))
    xr, yr = b.rl_tensors()
    # Loss sums depend on the float reduction order, which depends on the thread count;
    # the archive vector was computed at 4 threads, so the exact check runs at 4.
    torch.set_num_threads(1)
    ls1 = S.loss_sensitivity(model, "simplecnn", xr, yr)
    torch.set_num_threads(4)
    ls = S.loss_sensitivity(model, "simplecnn", xr, yr)
    torch.set_num_threads(args.threads)
    dev1 = max(abs(p - q) for p, q in zip(ls1["raw"], arch_json["raw_loss_sensitivity"]))
    check(res, "V_RL loss sensitivity reproduces archive_sensitivity_vector.json exactly (4 threads)",
          ls["raw"] == arch_json["raw_loss_sensitivity"] and ls["normalized"] == arch_json["normalized_sensitivity"],
          f"normalized {[round(v, 6) for v in ls['normalized']]}; at 1 thread the raw values differ by up to "
          f"{dev1:.1e} (float reduction order), so loss-based quantities are computed at a fixed 4 threads")

    # --- unit pruning vs historical prune_layerwise
    new_p = P.prune_actions(model, "simplecnn", [1, 1, 5, 5])
    old_p = common.prune_layerwise(ref, [0.1, 0.1, 0.6, 0.6])
    check(res, "unit pruning [1,1,5,5]: identical masks/weights, sparsity 58.19572427856073",
          all(torch.equal(new_p.state_dict()[k], old_p.state_dict()[k]) for k in old_p.state_dict())
          and P.total_sparsity(new_p) == 58.19572427856073 == common.calculate_sparsity(old_p),
          f"sparsity {P.total_sparsity(new_p)}")

    # --- landscape subset (validation + probe accuracy of 100 of the 1,296 policies)
    land = pd.read_csv(os.path.join(ROOT, "results", "policy_landscape_validation.csv"), float_precision="round_trip")
    idx = sorted(set(range(0, 1296, 13)) | set(land.index[land.actions.isin(
        ["[1, 1, 5, 5]", "[0, 0, 5, 5]", "[0, 5, 5, 5]"])].tolist()))
    bad = []
    vl, pl = b.val_loader(), b.probe_loader()
    # The landscape was computed at 1 thread (policy_landscape.py). A thread change can flip an argmax
    # on an exact logit tie: [3, 4, 2, 2] has one such validation image (3771), 73.24% at 1 thread,
    # 73.26% at 2 or 4 threads, in both the historical and the new code.
    torch.set_num_threads(1)
    for i in idx:
        row = land.iloc[i]
        acts = json.loads(row.actions)
        pm = P.prune_actions(model, "simplecnn", acts)
        got = (evaluate_accuracy(pm, vl), evaluate_accuracy(pm, pl), P.total_sparsity(pm))
        # accuracies exactly; sparsity to 1e-12 because the landscape CSV stores 16 significant digits
        if got[:2] != (row.val_accuracy, row.probe_accuracy) or abs(got[2] - row.total_sparsity) > 1e-12:
            bad.append((acts, got))
    torch.set_num_threads(args.threads)
    check(res, f"landscape: {len(idx)} policies reproduce recorded val/probe accuracy exactly and sparsity "
               "to the CSV's 16 significant digits (1 thread, as recorded)", not bad, f"mismatches: {bad[:3]}")

    # --- matched-sparsity baselines vs historical recorded numbers (each row by its own recorded definition)
    rec = pd.read_csv(os.path.join(ROOT, "results", "matched_sparsity_methods.csv"), float_precision="round_trip")
    sweep = rec[rec.Notes.str.startswith("all Conv2d/Linear")]          # notebook sweep: global over all 5 layers
    fixed = rec[rec.Notes.str.startswith("fixed action-space level")]    # uniform policies [a, a, a, a]
    rec = rec[rec.Notes.str.startswith("matched to PPO seed")]
    bad = []
    for _, r in sweep.iterrows():
        m = copy.deepcopy(model)
        torch.nn.utils.prune.global_unstructured([(mm, "weight") for _, mm in P.weight_layers(m)],
                                                 pruning_method=torch.nn.utils.prune.L1Unstructured,
                                                 amount=float(r.Method.split()[-1].rstrip("%")) / 100)
        acc = accuracy_from(*predictions(m, b.test_loader()))
        if acc != float(r["Accuracy (%)"]) or P.total_sparsity(m) != float(r["Sparsity (%)"]):
            bad.append((r.Method, acc, P.total_sparsity(m)))
    for _, r in fixed.iterrows():
        m = P.prune_actions(model, "simplecnn", json.loads(r.Actions))
        acc = accuracy_from(*predictions(m, b.test_loader()))
        if acc != float(r["Accuracy (%)"]) or P.total_sparsity(m) != float(r["Sparsity (%)"]):
            bad.append((r.Method, acc, P.total_sparsity(m)))
    check(res, f"reference baselines: {len(sweep)} notebook-sweep global (all 5 layers) and {len(fixed)} fixed "
               "uniform policies reproduce recorded test accuracy and sparsity exactly", not bad, f"mismatches: {bad}")
    bad = []
    for _, r in rec.iterrows():
        target = float(r["Sparsity (%)"])
        if r.Method.startswith("Global"):
            old_m, frac = common.prune_global_to_total(ref, target)
            new_m = P.global_fraction(model, "simplecnn", frac)
            new_exact = P.global_magnitude(model, "simplecnn", target)
        else:
            old_m, frac = common.prune_uniform_to_total(ref, target)
            new_m = P.uniform_fraction(model, "simplecnn", frac)
            new_exact = None
        same = all(torch.equal(new_m.state_dict()[k], old_m.state_dict()[k]) for k in old_m.state_dict())
        acc = accuracy_from(*predictions(new_m, b.test_loader()))
        if new_exact is not None:
            same = same and all(torch.equal(P.make_permanent(new_exact).state_dict()[k.replace("_orig", "")],
                                            (old_m.state_dict()[k] * old_m.state_dict()[k.replace("_orig", "_mask")]))
                                for k in old_m.state_dict() if k.endswith("_orig"))
        if not same or acc != float(r["Accuracy (%)"]):
            bad.append((r.Method, acc, float(r["Accuracy (%)"]), same))
    check(res, f"matched-sparsity: {len(rec)} recorded global-magnitude/uniform baselines reproduce exactly "
               "(masks + test accuracy; exact-count global = historical global)", not bad, f"mismatches: {bad}")

    # --- statistics vs historical implementations
    import exploration_confirm as ec
    import soft_prior_aggregate as spa
    rng = np.random.default_rng(1)
    vecs = [rng.normal(0.3, 1, n) for n in (5, 8, 11, 16, 20, 30)] + [np.round(rng.normal(0, 1, 12), 1)]
    auc = pd.read_csv(os.path.join(ROOT, "results", "exploration_levelA_auc.csv"), float_precision="round_trip")
    if {"condition", "seed", "val_accuracy_auc"} <= set(auc.columns):
        e1 = auc[auc.condition == "E1"].set_index("seed")["val_accuracy_auc"]
        e0 = auc[auc.condition == "E0"].set_index("seed")["val_accuracy_auc"]
        vecs.append((e1 - e0.loc[e1.index]).to_numpy(float))
    # exploration_confirm uses the same algorithms (exact equality); soft_prior_aggregate enumerates
    # all 2^n sign patterns (feasible only for n <= 20; float-level agreement)
    same = all(ST.sign_flip_p(v) == ec.sign_flip_p(v) and ST.wilcoxon_p(v) == ec.wilcoxon_p(v) for v in vecs)
    same &= all(abs(ST.wilcoxon_p(v) - spa.wilcoxon_p(v)) < 1e-12 for v in vecs if len(v) <= 20)
    brute = all(abs(ST.sign_flip_p(v) - ST.sign_flip_p_bruteforce(v)) < 1e-12 for v in vecs if len(v) <= 16)
    pv = rng.uniform(0, 0.2, 7)
    same_holm = np.array_equal(ST.holm(pv), ec.holm(pv))
    tq = max(abs(round(ST.t_quantile(0.975, df), 3) - ec.T975[df]) for df in range(1, 31))   # T975 keyed by df
    check(res, "statistics: sign-flip, Wilcoxon, Holm identical to historical; sign-flip = brute force; "
               "exact t-quantiles match the historical 3-decimal table",
          same and brute and same_holm and tq < 1e-9, f"{len(vecs)} vectors incl. recorded E1-E0 AUC; "
                                                      f"max |t - table| after rounding {tq}")
    b_counts = ST.mcnemar(preds == labels, preds == labels)
    check(res, "McNemar of a model against itself", b_counts == (0, 0, 0.0, 1.0), str(b_counts))

    out = {"static": res, "seconds": round(time.time() - t0, 1)}
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(os.path.join(OUT_DIR, "static.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1)
    print(f"static checks: {sum(r['pass'] for r in res)}/{len(res)} passed ({out['seconds']}s)")


# ------------------------------------------------------------------ PPO reproduction
RUNS = {
    "main42": dict(curve="results/training_curves/sens-correct_coef-0.04_seed-42.csv",
                   record="results/runs/sens-correct_coef-0.04_seed-42.json", seed=42, cache=False, prior=None),
    "main1": dict(curve="results/training_curves/sens-correct_coef-0.04_seed-1.csv",
                  record="results/runs/sens-correct_coef-0.04_seed-1.json", seed=1, cache=True, prior=None),
    "main2": dict(curve="results/training_curves/sens-correct_coef-0.04_seed-2.csv",
                  record="results/runs/sens-correct_coef-0.04_seed-2.json", seed=2, cache=True, prior=None),
    "main3": dict(curve="results/training_curves/sens-correct_coef-0.04_seed-3.csv",
                  record="results/runs/sens-correct_coef-0.04_seed-3.json", seed=3, cache=True, prior=None),
    "explore_E1_100": dict(curve="results/exploration_levelA_training_curves_raw/explore_E1_seed-100.csv",
                           record="results/exploration_levelA_runs/explore_E1_seed-100.json", seed=100, cache=True,
                           prior="record"),
}


def cmd_ppo(args):
    from stable_baselines3 import PPO
    torch.set_num_threads(args.threads)
    spec = RUNS[args.name]
    record = json.load(open(os.path.join(ROOT, spec["record"])))
    curve = pd.read_csv(os.path.join(ROOT, spec["curve"]), float_precision="round_trip")
    b = D.DataBundle("cifar10", train_images=False)
    model = M.load_checkpoint(FIXED, "simplecnn", 10)
    base_val = evaluate_accuracy(model, b.val_loader())
    assert base_val == 77.32
    with b.training():
        if record.get("sensitivity_vector") is not None:          # historical main runs: state sensitivity
            state = record["sensitivity_vector"]
        else:
            state = record["state_sensitivity"]
        coef = record["sparsity_coef"]
        env = RL.PruningEnv(model, "simplecnn", base_val, b.probe_loader(), sensitivities=state,
                            sparsity_coef=coef, cache={} if spec["cache"] else None)
        kwargs = dict(RL.PPO_KWARGS)
        if spec["prior"]:
            kwargs["policy"] = RL.SoftPriorPolicy
            kwargs["policy_kwargs"] = {"prior": RL.prior_matrix(record["beta"], record["prior_sensitivity"])}
        t0 = time.time()
        agent = PPO(env=env, seed=spec["seed"], **kwargs)
        log = RL.EpisodeLog()
        agent.learn(total_timesteps=RL.LEGACY_TOTAL_TIMESTEPS, callback=log)
        seconds = time.time() - t0
    actions, info = RL.deterministic_rollout(agent, env)
    pruned = P.prune_actions(model, "simplecnn", actions)
    b.freeze({"policy": actions})
    test = accuracy_from(*predictions(pruned, b.test_loader()))
    new = pd.DataFrame(log.rows)
    os.makedirs(OUT_DIR, exist_ok=True)
    new.to_csv(os.path.join(OUT_DIR, f"ppo_{args.name}_curve.csv"), index=False)
    # Recorded curves are parsed with float_precision="round_trip": pandas' default parser
    # misreads ~20% of 17-digit floats in the last bit, which would fake a mismatch.
    cols = ["episode", "timestep", "episode_reward", "accuracy_term", "sparsity_term", "diversity_term",
            "probe_accuracy", "sparsity", "actions"]
    identical = len(new) == len(curve) and all(new[c].tolist() == curve[c].tolist() for c in cols)
    first_diff = None
    if not identical:
        for i in range(min(len(new), len(curve))):
            if any(new[c].iloc[i] != curve[c].iloc[i] for c in cols):
                first_diff = int(i + 1)
                break
    out = {"name": args.name, "seed": spec["seed"], "cache": spec["cache"], "soft_prior": bool(spec["prior"]),
           "episodes": len(new), "curve_identical_all_columns": identical, "first_differing_episode": first_diff,
           "actions": actions, "recorded_actions": record["actions"],
           "sparsity": P.total_sparsity(pruned), "recorded_sparsity": record["total_sparsity"],
           "test_accuracy": test, "recorded_test_accuracy": record["test_accuracy"],
           "final_reward": info["reward"], "threads": args.threads, "train_seconds": round(seconds, 1)}
    out["pass"] = bool(identical and actions == record["actions"] and out["sparsity"] == record["total_sparsity"]
                       and test == record["test_accuracy"])
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(os.path.join(OUT_DIR, f"ppo_{args.name}.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1)
    print(json.dumps(out, indent=1))


def cmd_report(_args):
    static = json.load(open(os.path.join(OUT_DIR, "static.json")))
    ppo = [json.load(open(p)) for p in sorted(glob.glob(os.path.join(OUT_DIR, "ppo_*.json")))]
    here = os.path.dirname(os.path.abspath(__file__))
    lib = sorted(glob.glob(os.path.join(ROOT, "src", "**", "*.py"), recursive=True))
    out = {"static_checks": static["static"], "ppo_runs": ppo,
           "all_pass": all(r["pass"] for r in static["static"]) and all(r["pass"] for r in ppo) and len(ppo) == len(RUNS),
           "library_sha256": {os.path.relpath(p, ROOT).replace("\\", "/"): sha256_file(p) for p in lib},
           "script_sha256": sha256_file(os.path.join(here, "verify_library.py")),
           "environment": environment(), "git": git_state()}
    with open(os.path.join(ROOT, "results", "reproducibility", "library_verification.json"), "w",
              encoding="utf-8") as f:
        json.dump(out, f, indent=1)
    print("ALL PASS" if out["all_pass"] else "FAILURES PRESENT")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("static")
    s.add_argument("--threads", type=int, default=4)
    p = sub.add_parser("ppo")
    p.add_argument("name", choices=list(RUNS))
    p.add_argument("--threads", type=int, default=1)
    sub.add_parser("report")
    args = ap.parse_args()
    {"static": cmd_static, "ppo": cmd_ppo, "report": cmd_report}[args.cmd](args)


if __name__ == "__main__":
    main()
