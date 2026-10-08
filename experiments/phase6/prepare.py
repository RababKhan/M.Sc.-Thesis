"""Phase 6 frozen configuration and matched-mask table (after the pilot; before the pre-registration is committed).

  * E by the Part-A timing rule: 5 if the projected confirmatory wall time on 4 workers is <= 30 h, else 3
  * phase6_matched_masks.csv: for each of the 120 (setting, policy seed) pairs the PPO, LAMP and global masks at the
    identical zero count, with per-layer zeros, mask hashes and the (zero) mismatch. No training, no TEST access.
  * phase6_config.json, phase6_environment.json
Refuses to overwrite the config.
"""
import datetime
import json
import os

import pandas as pd
import torch

import phase6_lib as L
from src import pruning as P
from src.utils import environment, git_state, sha256_file, sha256_json

LIMIT_HOURS = 30.0


def projection(per, epochs):
    per_run = epochs * (per.train_s + per.val_s) + 3 * per.val_s            # + pre/post validation and two test passes
    return float((per_run * 20 * 4).sum() + per_run.sum()) / 3600.0 / 4.0    # 480 runs + 6 reproduction runs, 4 workers


def main():
    if os.path.exists(L.CONFIG):
        raise SystemExit("phase6_config.json exists; the configuration is frozen")
    torch.set_num_threads(L.THREADS)
    t = pd.read_csv(os.path.join(L.RESULTS, "phase6_timing_pilot.csv"))
    per = t.groupby("setting").agg(train_s=("train_seconds_per_epoch", "mean"), val_s=("validation_seconds_5000", "mean"),
                                   peak_mb=("peak_memory_mb", "max")).reindex(L.SETTINGS)
    proj = {e: projection(per, e) for e in (3, 5, 10)}
    epochs = 5 if proj[5] <= LIMIT_HOURS else 3
    fr = json.load(open(os.path.join(L.P5, "phase5_final_policies.json"), encoding="utf-8"))
    c1 = {(r["setting"], r["seed"]): r["selected_policy"] for r in fr["runs"] if r["condition"] == "C1"}
    assert len(c1) == 120
    rows, settings = [], {}
    for s in L.SETTINGS:
        d, arch = L.P3.split(s)
        ref, sha = L.P3.load_reference(s)
        prunable = sum(m.weight.numel() for _, m in P.prunable_tensors(P.get_units(ref, arch)))
        distinct = {}
        for seed in L.POLICY_SEEDS:
            pol = c1[(s, seed)]
            if tuple(pol) not in distinct:
                models = {m: L.build(ref, arch, m, pol) for m in L.METHODS}
                k = {m: P.zero_count(models[m]) for m in L.METHODS}
                assert len(set(k.values())) == 1
                distinct[tuple(pol)] = {"zero_count": k["ppo"], "sparsity": P.total_sparsity(models["ppo"]),
                                        **{f"{m}_layer_zeros": json.dumps(L.layer_zeros(models[m])) for m in L.METHODS},
                                        **{f"{m}_mask_sha256": L.mask_hash(models[m]) for m in L.METHODS},
                                        "classifier_zeros": max(L.layer_zeros(models[m])[P.EXCLUDED[arch][0]] for m in L.METHODS)}
            q = distinct[tuple(pol)]
            rows.append({"setting": s, "policy_seed": seed, "ft_seed": L.ft_seed(seed), "policy": str(pol),
                         "total_weights": P.weight_total(ref), "prunable_weights": prunable, **q,
                         "abs_zero_count_mismatch_lamp": 0, "abs_zero_count_mismatch_global": 0})
        settings[s] = {"reference_sha256": sha, "distinct_policies": len(distinct), "total_weights": P.weight_total(ref),
                       "prunable_weights": prunable}
    pd.DataFrame(rows).to_csv(os.path.join(L.RESULTS, "phase6_matched_masks.csv"), index=False, float_format="%.17g")
    cfg = {"created": datetime.datetime.now().astimezone().isoformat(timespec="seconds"), "epochs": epochs,
           "epoch_rule": {"limit_wall_hours": LIMIT_HOURS, "projected_wall_hours": proj},
           "protocol": L.PROTOCOL, "policy_seeds": L.POLICY_SEEDS, "ft_seed_offset": L.FT_SEED_OFFSET, "methods": L.ALL_METHODS,
           "threads": L.THREADS, "settings": settings,
           "inputs_sha256": {"phase5_final_policies": sha256_file(os.path.join(L.P5, "phase5_final_policies.json")),
                             "phase5_test_evaluations": sha256_file(os.path.join(L.P5, "phase5_test_evaluations.csv")),
                             "cifar10_split": sha256_file(os.path.join(L.P5, "splits", "cifar10_phase5_split.npz")),
                             "cifar100_split": sha256_file(os.path.join(L.P5, "splits", "cifar100_phase5_split.npz")),
                             "design_commitments": sha256_file(os.path.join(L.RESULTS, "phase6_design_commitments.md")),
                             "matched_masks": sha256_file(os.path.join(L.RESULTS, "phase6_matched_masks.csv")),
                             "timing_pilot": sha256_file(os.path.join(L.RESULTS, "phase6_timing_pilot.csv"))},
           "reproduction_runs": [f"{s}_seed{L.POLICY_SEEDS[0]}_ppo" for s in L.SETTINGS], "git": git_state()}
    cfg["content_sha256"] = sha256_json({k: cfg[k] for k in ("epochs", "protocol", "settings", "inputs_sha256")})
    json.dump(cfg, open(L.CONFIG, "w", encoding="utf-8"), indent=1)
    json.dump(environment(), open(os.path.join(L.RESULTS, "phase6_environment.json"), "w", encoding="utf-8"), indent=1)
    print(per.round(1).to_string())
    print("projected wall-hours:", {k: round(v, 1) for k, v in proj.items()}, "-> E =", epochs)
    print({s: v["distinct_policies"] for s, v in settings.items()})
    print("config sha256", sha256_file(L.CONFIG))


if __name__ == "__main__":
    main()
