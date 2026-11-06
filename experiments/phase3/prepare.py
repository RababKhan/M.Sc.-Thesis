"""Phase 3 frozen inputs (run once, before the pre-registration is committed and before any PPO run).

Per setting, on V_RL only, 1 thread:
  * reference checkpoint sha256 (asserted against the Phase-2 record)
  * reference V_RL accuracy (the reward denominator) and loss
  * refined loss sensitivity: mean over ratios {10,20,40,60}% of (L_r - L_0)/L_0 when one unit
    alone is L1-pruned; min-max normalised (the validated archive procedure, src.sensitivity)
  * destructive-action rule: the unit with the highest raw sensitivity (ties: earliest unit)
    pruned at >= 40%
  * P2 shuffled priors: per seed, numpy default_rng(seed) permutations until every unit receives
    another unit's, different, value (soft_prior.shuffled, unchanged algorithm)
  * P3 constant prior: every unit = mean of the normalised vector
Writes phase3_sensitivity_vectors.csv, phase3_destructive_action_rules.json and
phase3_frozen_inputs.json; refuses to overwrite.
"""
import datetime
import json
import os

import numpy as np
import pandas as pd

import phase3_common as C
from src import data as D, pruning as P, sensitivity as S
from src.evaluation import evaluate_accuracy, mean_loss_accuracy
from src.utils import environment, git_state, sha256_file, sha256_json


def shuffled(values, seed):
    rng = np.random.default_rng(seed)
    for _ in range(10000):
        perm = rng.permutation(len(values))
        if all(perm[i] != i and not np.isclose(values[perm[i]], values[i]) for i in range(len(values))):
            return values[perm], [int(p) for p in perm]
    raise RuntimeError("no value-changing permutation")


def main():
    for p in (C.INPUTS_JSON, C.SENS_CSV, C.DESTRUCTIVE_JSON):
        if os.path.exists(p):
            raise SystemExit(f"{p} exists; the Phase-3 inputs are frozen")
    C.set_threads()
    out, rows, rules = {}, [], {}
    bundles = {}
    for setting in C.SETTINGS:
        d, a = C.split(setting)
        b = bundles.setdefault(d, D.DataBundle(d, train_images=False))
        model, sha = C.load_reference(setting)
        units = P.get_units(model, a)
        prunable = sum(u.param_count for u in units)
        xr, yr = b.rl_tensors()
        base_acc = evaluate_accuracy(model, b.rl_loader())
        base_loss, base_acc_t = mean_loss_accuracy(model, xr, yr)
        assert base_acc == base_acc_t
        sens = S.loss_sensitivity(model, a, xr, yr)
        raw, norm = np.array(sens["raw"]), np.array(sens["normalized"])
        order = sorted(range(4), key=lambda i: (-raw[i], i))
        most = order[0]
        shuffles = {}
        for seed in C.SEEDS:
            vec, perm = shuffled(norm, seed)
            shuffles[str(seed)] = {"prior": vec.tolist(), "source_index": perm,
                                   "mapping": {units[i].name: units[perm[i]].name for i in range(4)}}
        rule = {"most_sensitive_unit": units[most].name, "unit_index": most,
                "destructive_if": f"action[{most}] >= {C.DESTRUCTIVE_ACTION} (ratio >= 40%)",
                "raw_sensitivity": raw.tolist(), "normalized_sensitivity": norm.tolist()}
        rules[setting] = rule
        out[setting] = {
            "dataset": d, "arch": a, "reference_checkpoint": os.path.relpath(C.reference_path(setting), C.ROOT).replace("\\", "/"),
            "reference_sha256": sha, "v_rl_split_file": os.path.relpath(D.SPLIT_FILES[d], C.ROOT).replace("\\", "/"),
            "v_rl_split_sha256": sha256_file(D.SPLIT_FILES[d]), "base_vrl_accuracy": base_acc, "base_vrl_loss": base_loss,
            "units": [{"name": u.name, "kind": u.kind, "params": u.param_count,
                       "pct_of_prunable": 100.0 * u.param_count / prunable,
                       "modules": [n for n, _ in u.modules]} for u in units],
            "excluded": P.EXCLUDED[a],
            "sensitivity_raw": raw.tolist(), "sensitivity_normalized": norm.tolist(),
            "sensitivity_ranking": [units[i].name for i in order], "sensitivity_detail": sens["detail"],
            "constant_prior": [float(norm.mean())] * 4, "shuffled_priors": shuffles, "destructive_rule": rule,
        }
        for i, u in enumerate(units):
            rows.append({"setting": setting, "dataset": d, "arch": a, "unit_index": i, "unit": u.name, "unit_type": u.kind,
                         "params": u.param_count, "pct_of_prunable": 100.0 * u.param_count / prunable,
                         "raw_sensitivity": raw[i], "normalized_sensitivity": norm[i],
                         "rank": order.index(i) + 1, "most_sensitive": i == most,
                         "base_vrl_accuracy": base_acc, "base_vrl_loss": base_loss, "threads": C.THREADS})
        print(f"{setting}: sha {sha[:12]}  V_RL acc {base_acc:.4f}  S_norm {np.round(norm, 5).tolist()}  "
              f"most sensitive {units[most].name}", flush=True)

    # SimpleCNN-C10 at 1 thread vs the archive vector (computed at 4 threads)
    arch = json.load(open(os.path.join(C.RESULTS, "archive_sensitivity_vector.json"), encoding="utf-8"))
    c10 = out["cifar10_simplecnn"]
    agreement = {"archive_raw": arch["raw_loss_sensitivity"], "archive_normalized": arch["normalized_sensitivity"],
                 "max_abs_diff_raw": max(abs(p - q) for p, q in zip(c10["sensitivity_raw"], arch["raw_loss_sensitivity"])),
                 "max_abs_diff_normalized": max(abs(p - q) for p, q in zip(c10["sensitivity_normalized"],
                                                                           arch["normalized_sensitivity"]))}
    meta = {"created": datetime.datetime.now().astimezone().isoformat(timespec="seconds"), "threads": C.THREADS,
            "procedure": "src.sensitivity.loss_sensitivity on V_RL, ratios 10/20/40/60%, min-max normalised",
            "seeds": C.SEEDS, "beta": C.BETA, "lambda_s": C.LAMBDA_S,
            "simplecnn_c10_vs_archive_vector": agreement, "environment": environment(), "git": git_state(),
            "script_sha256": sha256_file(os.path.abspath(__file__))}
    payload = {"meta": meta, "settings": out}
    payload["content_sha256"] = sha256_json({"settings": out})
    os.makedirs(C.AG, exist_ok=True)
    with open(C.INPUTS_JSON, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=1)
    pd.DataFrame(rows).to_csv(C.SENS_CSV, index=False, float_format="%.17g")
    with open(C.DESTRUCTIVE_JSON, "w", encoding="utf-8") as f:
        json.dump({"rule": "aggressive pruning (>= 40%, action index >= 4) of each setting's most sensitive unit "
                           "(highest raw V_RL loss sensitivity; ties: earliest unit)",
                   "fixed_at": meta["created"], "settings": rules,
                   "sensitivity_file_sha256": sha256_file(C.SENS_CSV)}, f, indent=1)
    print("SimpleCNN-C10 vs archive vector:", {k: v for k, v in agreement.items() if k.startswith("max")})
    for p in (C.INPUTS_JSON, C.SENS_CSV, C.DESTRUCTIVE_JSON):
        print(os.path.relpath(p, C.ROOT), sha256_file(p))


if __name__ == "__main__":
    main()
