"""Phase 4 frozen configuration (after the pre-registration commit, before any Phase-4 PPO run).

Deterministic rules on validation landscapes only (never test):
  * reward parameters per setting: A_b and the anchors A_60, rho_60 on V_RL and V_SELECT; R1b target
    rho*_s = max sparsity over V_RL-landscape policies with retention >= 95%; kappa for R1a and for R1b =
    the smallest value of {0.005, 0.01, 0.02, 0.04, 0.08} whose V_RL-landscape reward optimum lies within
    +-5 pp of the target in all six settings (else 0.08, flagged)
  * per policy and reward: V_RL reward, rank, utility; V_SELECT score and rank; V_RL frontier regret
  * priors P0-P5 per setting (and per seed for P3)
Writes results/phase4/phase4_config.json and results/phase4/landscapes/<setting>_rewards.{json,csv}; refuses
to overwrite.
"""
import datetime
import glob
import json
import os

import numpy as np
import pandas as pd

import phase4_lib as L
from src.utils import environment, git_state, sha256_file, sha256_json


def rank_min(values):
    return pd.Series(values).rank(ascending=False, method="min").astype(int).to_numpy()


def optimum(kind, prm, A, rho, acts):
    rew = np.array([L.reward_parts(kind, a, r, x, prm)["total"] for a, r, x in zip(A, rho, acts)])
    return int(np.argmax(rew)), rew           # np.argmax: first maximum = lowest policy index


def main():
    if os.path.exists(L.CONFIG):
        raise SystemExit("phase4_config.json exists; the Phase-4 configuration is frozen")
    p3 = L.P3.frozen_inputs()
    per = {}
    for s in L.SETTINGS:
        land = json.load(open(L.P3.landscape_json(s), encoding="utf-8"))
        rec = json.load(open(glob.glob(os.path.join(L.P3.RUNS_DIR, f"{s}_P0_seed300.json"))[0], encoding="utf-8"))
        assert rec["landscape_sha256"] == sha256_file(L.P3.landscape_json(s)), f"{s}: Phase-3 landscape changed"
        sel = json.load(open(os.path.join(L.LAND, f"{s}_vselect.json"), encoding="utf-8"))["policies"]
        pol = land["policies"]
        assert [tuple(p["actions"]) for p in pol] == L.P3.POLICIES == [tuple(p["actions"]) for p in sel]
        A = np.array([p["val_accuracy"] for p in pol])
        As = np.array([p["vselect_accuracy"] for p in sel])
        rho = np.array([p["total_sparsity"] for p in pol])
        acts = [p["actions"] for p in pol]
        i0, i60 = L.P3.POLICIES.index((0, 0, 0, 0)), L.P3.POLICIES.index(L.MAX_POLICY)
        base = p3["settings"][s]["base_vrl_accuracy"]
        assert A[i0] == base, f"{s}: dense V_RL accuracy {A[i0]} != frozen {base}"
        target_b = float(rho[A >= L.RETENTION_TOL * base].max())
        near50 = (np.abs(rho - L.TARGET_COMMON) <= 2.5)
        per[s] = {"A_b_vrl": base, "A_b_vselect": float(As[i0]), "A_60_vrl": float(A[i60]), "A_60_vselect": float(As[i60]),
                  "rho_60": float(rho[i60]), "rho_max": float(rho.max()), "target_R1b": target_b,
                  "R1a_appropriate": bool((near50 & (A >= L.RETENTION_TOL * base)).any()),
                  "R1a_best_retention_near_50": float((A[near50] / base).max()),
                  "sensitivity_normalized": p3["settings"][s]["sensitivity_normalized"],
                  "destructive_unit_index": p3["settings"][s]["destructive_rule"]["unit_index"],
                  "units": [u["name"] for u in p3["settings"][s]["units"]],
                  "reference_sha256": p3["settings"][s]["reference_sha256"],
                  "phase3_landscape_sha256": sha256_file(L.P3.landscape_json(s)),
                  "vselect_landscape_sha256": sha256_file(os.path.join(L.LAND, f"{s}_vselect.json")),
                  "_A": A, "_As": As, "_rho": rho, "_acts": acts}

    def params(s, kind, split, kappa=None):
        q = per[s]
        prm = {"A_b": q[f"A_b_{split}"], "A_60": q[f"A_60_{split}"], "rho_60": q["rho_60"]}
        if kind == "R1a":
            prm.update(kappa=kappa, target=L.TARGET_COMMON)
        if kind == "R1b":
            prm.update(kappa=kappa, target=q["target_R1b"])
        return prm

    kappa, calib = {}, {}
    for kind in ("R1a", "R1b"):
        chosen = None
        for k in L.KAPPA_GRID:
            dev = {}
            for s in L.SETTINGS:
                q = per[s]
                i, _ = optimum(kind, params(s, kind, "vrl", k), q["_A"], q["_rho"], q["_acts"])
                dev[s] = float(q["_rho"][i] - params(s, kind, "vrl", k)["target"])
            calib.setdefault(kind, []).append({"kappa": k, "optimum_minus_target": dev,
                                               "all_within_window": all(abs(v) <= L.TARGET_WINDOW for v in dev.values())})
            if chosen is None and all(abs(v) <= L.TARGET_WINDOW for v in dev.values()):
                chosen = k
        kappa[kind] = chosen if chosen is not None else L.KAPPA_GRID[-1]
        calib[kind + "_calibration_failed"] = chosen is None

    reward_params, oracle = {}, {}
    os.makedirs(L.LAND, exist_ok=True)
    for s in L.SETTINGS:
        q = per[s]
        A, As, rho, acts = q["_A"], q["_As"], q["_rho"], q["_acts"]
        front = np.array([A[rho >= r].max() for r in rho])
        table = {"policy_index": list(range(1296)), "actions": acts, "vrl_accuracy": A.tolist(), "vselect_accuracy": As.tolist(),
                 "total_sparsity": rho.tolist(), "frontier_regret_vrl": (front - A).tolist(),
                 "destructive": [a[q["destructive_unit_index"]] >= 4 for a in acts]}
        reward_params[s], oracle[s] = {}, {}
        for kind in L.REWARDS:
            pv = params(s, kind, "vrl", kappa.get(kind))
            ps = params(s, kind, "vselect", kappa.get(kind))
            rv = [L.reward_parts(kind, a, r, x, pv)["total"] for a, r, x in zip(A, rho, acts)]
            rs = [L.reward_parts(kind, a, r, x, ps)["total"] for a, r, x in zip(As, rho, acts)]
            table[f"{kind}_reward_vrl"] = rv
            table[f"{kind}_rank_vrl"] = rank_min(rv).tolist()
            table[f"{kind}_utility_vrl"] = [L.utility(kind, a, r, pv) for a, r in zip(A, rho)]
            table[f"{kind}_score_vselect"] = rs
            table[f"{kind}_rank_vselect"] = rank_min(rs).tolist()
            reward_params[s][kind] = {"vrl": pv, "vselect": ps}
            i = int(np.argmax(rv))
            oracle[s][kind] = {"optimum_actions": acts[i], "optimum_sparsity": float(rho[i]),
                               "optimum_vrl_retention": float(A[i] / q["A_b_vrl"]), "optimum_vselect_accuracy": float(As[i])}
        with open(os.path.join(L.LAND, f"{s}_rewards.json"), "w", encoding="utf-8") as f:
            json.dump(table, f)
        df = pd.DataFrame(table)
        df["actions"] = df["actions"].map(str)
        df.to_csv(os.path.join(L.LAND, f"{s}_rewards.csv"), index=False, float_format="%.17g")

    priors = {}
    for s in L.SETTINGS:
        S = per[s]["sensitivity_normalized"]
        priors[s] = {}
        for c in L.PRIORS:
            if c == "P3":
                priors[s][c] = {}
                for seed in L.SEEDS:
                    coef, perm = L.prior_coefficients(c, S, seed)
                    priors[s][c][str(seed)] = {"coefficients": coef, "permutation": perm}
            else:
                coef, _ = L.prior_coefficients(c, S, None)
                priors[s][c] = {"coefficients": coef}

    settings_out = {s: {k: v for k, v in q.items() if not k.startswith("_")} for s, q in per.items()}
    cfg = {"created": datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
           "seeds": L.SEEDS, "rewards": L.REWARDS, "priors": L.PRIORS, "search": L.SEARCH,
           "constants": {"ACC_COEF": L.ACC_COEF, "DIV_COEF": L.DIV_COEF, "R0_LAMBDA": L.R0_LAMBDA,
                         "TARGET_COMMON": L.TARGET_COMMON, "RETENTION_TOL": L.RETENTION_TOL, "KAPPA_GRID": L.KAPPA_GRID,
                         "TARGET_WINDOW": L.TARGET_WINDOW, "BETA_P1": L.BETA_P1, "BETA_NEW": L.BETA_NEW, "ETA": L.ETA,
                         "ELITE_K": L.ELITE_K, "ARCHIVE_K": L.ARCHIVE_K, "TOTAL_TIMESTEPS": L.TOTAL_TIMESTEPS,
                         "THREADS": L.THREADS, "TOPK": L.TOPK, "CENSOR": L.CENSOR},
           "kappa": kappa, "kappa_calibration": calib, "settings": settings_out, "reward_params": reward_params,
           "reward_oracle_vrl": oracle, "priors": priors,
           "reward_tables_sha256": {s: sha256_file(os.path.join(L.LAND, f"{s}_rewards.json")) for s in L.SETTINGS},
           "environment": environment(), "git": git_state(), "script_sha256": sha256_file(os.path.abspath(__file__)),
           "lib_sha256": sha256_file(os.path.join(os.path.dirname(os.path.abspath(__file__)), "phase4_lib.py"))}
    cfg["content_sha256"] = sha256_json({k: cfg[k] for k in ("kappa", "settings", "reward_params", "priors")})
    with open(L.CONFIG, "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=1)
    print("kappa:", kappa, "| calibration failed:", {k: calib[k + "_calibration_failed"] for k in ("R1a", "R1b")})
    for s in L.SETTINGS:
        q = settings_out[s]
        print(f"{s:20s} R1b target {q['target_R1b']:6.2f}  R1a appropriate {q['R1a_appropriate']}  "
              f"(best retention near 50%: {q['R1a_best_retention_near_50']:.3f})  A_60 {q['A_60_vrl']:.2f}")
    print("config sha256", sha256_file(L.CONFIG), "content", cfg["content_sha256"][:16])


if __name__ == "__main__":
    main()
