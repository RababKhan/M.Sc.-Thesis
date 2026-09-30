"""Phase 5 validation-only preparation (after the landscapes; before the full pre-registration and any PPO run).

  * dense accuracy on VAL-RL / VAL-SELECT (landscape policy [0,0,0,0], asserted = live evaluation)
  * sensitivity S on VAL-RL (src.sensitivity.loss_sensitivity, 1 thread); destructive unit; centred prior P2
  * tau diagnostic for {0.95, 0.97, 0.98, 0.99}; tau = the largest candidate with >= 20 feasible policies and
    >= 1 feasible policy of >= 10% sparsity in all six settings (phase5_design_commitments.md)
  * per-policy tables for the chosen tau: q on VAL-RL / VAL-SELECT, R_constrained and R0 on VAL-RL with ranks,
    R0 score on VAL-SELECT, oracle, destructive flag
Writes phase5_tau_diagnostic.csv, phase5_tau_selection.md, phase5_sensitivity_vectors.csv, landscapes/<s>_tables.json,
phase5_config.json, phase5_environment.json; refuses to overwrite the config.
"""
import datetime
import json
import os

import numpy as np
import pandas as pd
import torch

import phase5_lib as L
from src import pruning as P, sensitivity as S
from src.evaluation import evaluate_accuracy
from src.utils import environment, git_state, sha256_file, sha256_json


def rank_min(v):
    return pd.Series(v).rank(ascending=False, method="min").astype(int).to_numpy()


def main():
    if os.path.exists(L.CONFIG):
        raise SystemExit("phase5_config.json exists; the Phase-5 configuration is frozen")
    torch.set_num_threads(L.THREADS)
    per, diag, sens_rows = {}, [], []
    bundles = {}
    for s in L.SETTINGS:
        d, a = L.P3.split(s)
        b = bundles.setdefault(d, L.bundle(d))
        model, sha = L.P3.load_reference(s)
        land = json.load(open(os.path.join(L.LAND, f"{s}_landscape.json"), encoding="utf-8"))["policies"]
        assert [tuple(p["actions"]) for p in land] == L.POLICIES
        A = np.array([p["val_rl_accuracy"] for p in land])
        As = np.array([p["val_select_accuracy"] for p in land])
        rho = np.array([p["total_sparsity"] for p in land])
        i0 = L.INDEX[(0, 0, 0, 0)]
        Ad, Ads = float(A[i0]), float(As[i0])
        assert Ad == evaluate_accuracy(model, b.rl_loader()) and Ads == evaluate_accuracy(model, b.select_loader())
        xr, yr = b.rl_tensors()
        sv = S.loss_sensitivity(model, a, xr, yr)
        raw, norm = np.array(sv["raw"]), np.array(sv["normalized"])
        order = sorted(range(4), key=lambda i: (-raw[i], i))
        units = P.get_units(model, a)
        for i, u in enumerate(units):
            sens_rows.append({"setting": s, "unit_index": i, "unit": u.name, "params": u.param_count, "raw_sensitivity": raw[i],
                              "normalized_sensitivity": norm[i], "rank": order.index(i) + 1, "most_sensitive": i == order[0],
                              "dense_val_rl_accuracy": Ad, "threads": L.THREADS, "data": "VAL-RL (2,500)"})
        q = A / Ad
        for tau in L.TAUS:
            f = q >= tau
            fi = np.flatnonzero(f)
            best_acc = fi[np.lexsort((-rho[fi], -A[fi]))[0]]            # highest accuracy, ties: higher sparsity
            orc = fi[np.lexsort((fi, -A[fi], -rho[fi]))[0]]              # highest sparsity, ties: higher accuracy, lower index
            diag.append({"tau": tau, "setting": s, "n_feasible": int(f.sum()), "n_feasible_sparsity_ge_10": int((f & (rho >= 10)).sum()),
                         "max_feasible_sparsity": float(rho[f].max()), "median_feasible_sparsity": float(np.median(rho[f])),
                         "best_feasible_val_rl_accuracy": float(A[best_acc]), "sparsity_of_best_accuracy_policy": float(rho[best_acc]),
                         "oracle_policy": str(list(L.POLICIES[orc])), "oracle_sparsity": float(rho[orc]),
                         "oracle_val_rl_accuracy": float(A[orc]), "oracle_q": float(q[orc]),
                         "meets_rule": bool(f.sum() >= L.MIN_FEASIBLE and (f & (rho >= L.MIN_SPARSITY)).any())})
        per[s] = {"A": A, "As": As, "rho": rho, "Ad": Ad, "Ads": Ads, "S_raw": raw, "S_norm": norm, "most": order[0],
                  "units": [u.name for u in units], "sha": sha, "land": land}
    dg = pd.DataFrame(diag)
    ok = {t: bool(dg[dg.tau == t].meets_rule.all()) for t in L.TAUS}
    passing = [t for t in L.TAUS if ok[t]]
    if not passing:
        raise SystemExit("no tau candidate meets the rule in all six settings; stop and report")
    tau = max(passing)
    dg.to_csv(os.path.join(L.RESULTS, "phase5_tau_diagnostic.csv"), index=False, float_format="%.17g")
    pd.DataFrame(sens_rows).to_csv(os.path.join(L.RESULTS, "phase5_sensitivity_vectors.csv"), index=False, float_format="%.17g")

    cfg_settings, oracle = {}, {}
    for s in L.SETTINGS:
        p = per[s]
        A, As, rho = p["A"], p["As"], p["rho"]
        acts = [list(x) for x in L.POLICIES]
        rc = np.array([L.r_constrained(A[i], p["Ad"], rho[i], tau) for i in range(1296)])
        r0 = np.array([L.r0(A[i], p["Ad"], rho[i], acts[i]) for i in range(1296)])
        r0s = np.array([L.r0(As[i], p["Ads"], rho[i], acts[i]) for i in range(1296)])
        q, qs = A / p["Ad"], As / p["Ads"]
        rank_rc = rank_min(rc)
        orc = int(np.argmax(rc))
        assert rank_rc[orc] == 1 and q[orc] >= tau
        front = np.array([A[(rho >= r) & (q >= tau)].max() if ((rho >= r) & (q >= tau)).any() else np.nan for r in rho])
        tables = {"actions": acts, "val_rl_accuracy": A.tolist(), "val_select_accuracy": As.tolist(), "total_sparsity": rho.tolist(),
                  "unit_sparsity": [x["unit_sparsity"] for x in p["land"]], "q_val_rl": q.tolist(), "q_val_select": qs.tolist(),
                  "feasible_val_rl": (q >= tau).tolist(), "feasible_val_select": (qs >= tau).tolist(),
                  "RC_reward_val_rl": rc.tolist(), "RC_rank_val_rl": rank_rc.tolist(),
                  "R0_reward_val_rl": r0.tolist(), "R0_rank_val_rl": rank_min(r0).tolist(), "R0_score_val_select": r0s.tolist(),
                  "destructive": [x[p["most"]] >= 4 for x in acts]}
        with open(os.path.join(L.LAND, f"{s}_tables.json"), "w", encoding="utf-8") as f:
            json.dump(tables, f)
        cen = p["S_norm"] - p["S_norm"].mean()
        oracle[s] = {"policy_index": orc, "actions": acts[orc], "sparsity": float(rho[orc]), "val_rl_accuracy": float(A[orc]),
                     "q": float(q[orc]), "max_feasible_sparsity": float(rho[q >= tau].max()),
                     "feasible_sparsity_ge_20_exists": bool(((q >= tau) & (rho >= 20)).any()),
                     "n_feasible": int((q >= tau).sum())}
        cfg_settings[s] = {"reference_sha256": p["sha"], "dense_val_rl_accuracy": p["Ad"], "dense_val_select_accuracy": p["Ads"],
                           "sensitivity_raw": p["S_raw"].tolist(), "sensitivity_normalized": p["S_norm"].tolist(),
                           "most_sensitive_unit_index": p["most"], "units": p["units"],
                           "P2_coefficients": (L.BETA * cen).tolist(),
                           "landscape_sha256": sha256_file(os.path.join(L.LAND, f"{s}_landscape.json")),
                           "tables_sha256": sha256_file(os.path.join(L.LAND, f"{s}_tables.json"))}
    cfg = {"created": datetime.datetime.now().astimezone().isoformat(timespec="seconds"), "tau": tau,
           "tau_rule": {"candidates": L.TAUS, "min_feasible": L.MIN_FEASIBLE, "min_sparsity": L.MIN_SPARSITY, "passing": passing,
                        "meets_by_tau": ok},
           "constants": {"EPS": L.EPS, "KAPPA": L.KAPPA, "BETA": L.BETA, "ARCHIVE_K": L.ARCHIVE_K, "TOTAL_TIMESTEPS": L.TOTAL_TIMESTEPS,
                         "THREADS": L.THREADS, "CENSOR": L.CENSOR},
           "seeds": L.SEEDS, "conditions": {"C0": {"reward": "R0", "prior": "P2"}, "C1": {"reward": "RC", "prior": "P2"},
                                           "C2": {"reward": "RC", "prior": "none"}},
           "settings": cfg_settings, "oracle": oracle,
           "split_manifest_sha256": sha256_file(os.path.join(L.SPLITS, "phase5_split_manifest.json")),
           "design_commitments_sha256": sha256_file(L.COMMITMENTS), "git": git_state(),
           "script_sha256": sha256_file(os.path.abspath(__file__))}
    cfg["content_sha256"] = sha256_json({k: cfg[k] for k in ("tau", "settings", "oracle", "constants")})
    with open(L.CONFIG, "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=1)
    with open(os.path.join(L.RESULTS, "phase5_environment.json"), "w", encoding="utf-8") as f:
        json.dump(environment(), f, indent=1)
    lines = ["# Phase 5 tau selection", "",
             f"Rule (committed in `phase5_design_commitments.md` before the diagnostic): the largest tau in {L.TAUS} for which "
             f"every setting has >= {L.MIN_FEASIBLE} feasible policies (q = VAL-RL accuracy / dense VAL-RL accuracy >= tau) and "
             f">= 1 feasible policy with total sparsity >= {L.MIN_SPARSITY:.0f}%.", "",
             "| tau | meets the rule in all six settings |", "|---|---|"]
    lines += [f"| {t} | {'yes' if ok[t] else 'no'} |" for t in L.TAUS]
    lines += ["", f"**Chosen tau = {tau}.**", "", "Per setting at the chosen tau:", "",
              "| Setting | Feasible policies | Max feasible sparsity | Median feasible sparsity | Oracle policy | Oracle sparsity | Oracle q |",
              "|---|---|---|---|---|---|---|"]
    for _, r in dg[dg.tau == tau].iterrows():
        lines.append(f"| {r.setting} | {r.n_feasible} | {r.max_feasible_sparsity:.2f}% | {r.median_feasible_sparsity:.2f}% | "
                     f"{r.oracle_policy} | {r.oracle_sparsity:.2f}% | {r.oracle_q:.4f} |")
    open(os.path.join(L.RESULTS, "phase5_tau_selection.md"), "w", encoding="utf-8").write("\n".join(lines) + "\n")
    print(f"tau = {tau}; meets by tau: {ok}")
    print(dg.round(3).to_string(index=False))
    print("config sha256", sha256_file(L.CONFIG))


if __name__ == "__main__":
    main()
