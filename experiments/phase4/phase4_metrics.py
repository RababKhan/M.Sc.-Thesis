"""Per-run Phase-4 metrics, computed identically for stage selection and final analysis (validation only).

For a run under reward R the trajectory is the 512 sampled policies (one per episode). Using the frozen
per-reward tables (results/phase4/landscapes/<setting>_rewards.json):
  rank_k          reward rank (1 = best of 1,296) of the policy sampled in episode k, under R on V_RL
  rank_score_k    (1296 - rank_k) / 1295
  AUC(y)          trapezoidal area over timesteps t_k = 4k divided by (t_512 - t_1), as in Phase 3
Selection rules:
  final           the deterministic policy after training (S0, S2)
  archive         the top-10 unique sampled policies by V_RL reward (ties: earlier discovery); the one with
                  the highest V_SELECT score under R is chosen (ties: higher V_RL reward, then earlier) (S1, S2A)
"""
import gzip
import io
import json
import os

import numpy as np
import pandas as pd

import phase4_lib as L
from src import statistics as ST

INDEX = {p: i for i, p in enumerate(L.P3.POLICIES)}
_TABLES = {}


def table(setting):
    if setting not in _TABLES:
        _TABLES[setting] = json.load(open(os.path.join(L.LAND, f"{setting}_rewards.json"), encoding="utf-8"))
    return _TABLES[setting]


def auc(t, y):
    t, y = np.asarray(t, float), np.asarray(y, float)
    return float(np.sum((y[1:] + y[:-1]) * np.diff(t)) / 2.0 / (t[-1] - t[0]))


def curve(rec):
    raw = gzip.decompress(open(os.path.join(L.ROOT, rec["curve_file"]), "rb").read()).decode("utf-8")
    c = pd.read_csv(io.StringIO(raw), float_precision="round_trip")
    assert c["episode"].tolist() == list(range(1, 513)) and (c["timestep"] == 4 * c["episode"]).all()
    return c


def policy_row(setting, reward, idx):
    T = table(setting)
    return {"rank_vrl": T[f"{reward}_rank_vrl"][idx], "rank_vselect": T[f"{reward}_rank_vselect"][idx],
            "vrl_accuracy": T["vrl_accuracy"][idx], "vselect_accuracy": T["vselect_accuracy"][idx],
            "sparsity": T["total_sparsity"][idx], "reward_vrl": T[f"{reward}_reward_vrl"][idx],
            "score_vselect": T[f"{reward}_score_vselect"][idx], "regret_vrl": T["frontier_regret_vrl"][idx],
            "destructive": T["destructive"][idx], "actions": T["actions"][idx]}


def run_metrics(rec, cfg):
    s, R = rec["setting"], rec["reward"]
    T = table(s)
    c = curve(rec)
    idx = np.array([INDEX[tuple(json.loads(a))] for a in c["actions"]])
    rew = np.array(T[f"{R}_reward_vrl"])[idx]
    assert np.array_equal(rew, c["episode_reward"].to_numpy()), "curve reward != frozen reward table"
    rank = np.array(T[f"{R}_rank_vrl"])[idx]
    score = (1296 - rank) / 1295
    acc = np.array(T["vrl_accuracy"])[idx]
    util = np.array(T[f"{R}_utility_vrl"])[idx]
    destr = np.array(T["destructive"])[idx]
    t = c["timestep"].to_numpy()
    base = cfg["settings"][s]["A_b_vrl"]
    out = {"run_id": rec["run_id"], "setting": s, "reward": R, "prior": rec["prior"], "variant": rec["variant"],
           "seed": rec["seed"], "vrl_auc": auc(t, acc), "rank_auc": auc(t, score), "utility_auc": auc(t, util),
           "destructive_frequency": float(destr.mean()), "best_seen_rank": int(rank.min()),
           "median_rank_training": float(np.median(rank)), "unique_policies_sampled": int(len(set(idx.tolist())))}
    for k in (100, 50, 20, 10, 1):
        hit = np.flatnonzero(rank <= k)
        out[f"first_episode_top{k}"] = int(hit[0] + 1) if len(hit) else L.CENSOR
        out[f"reached_top{k}"] = bool(len(hit))
    fin = INDEX[tuple(rec["final_actions"])]
    # archive rule
    first = {}
    for k, i in enumerate(idx):
        first.setdefault(int(i), k)
    uniq = sorted(first, key=lambda i: (-T[f"{R}_reward_vrl"][i], first[i]))[: L.ARCHIVE_K]
    arc = sorted(uniq, key=lambda i: (-T[f"{R}_score_vselect"][i], -T[f"{R}_reward_vrl"][i], first[i]))[0]
    for name, i in (("final", fin), ("archive", arc)):
        row = policy_row(s, R, i)
        out.update({f"{name}_policy": str(row["actions"]), f"{name}_policy_index": int(i),
                    f"{name}_rank_vrl": row["rank_vrl"], f"{name}_rank_vselect": row["rank_vselect"],
                    f"{name}_vrl_accuracy": row["vrl_accuracy"], f"{name}_vselect_accuracy": row["vselect_accuracy"],
                    f"{name}_sparsity": row["sparsity"], f"{name}_vrl_retention": row["vrl_accuracy"] / base,
                    f"{name}_regret_vrl": row["regret_vrl"], f"{name}_destructive": row["destructive"]})
        for k in (1, 10, 20, 50, 100):
            out[f"{name}_top{k}"] = row["rank_vrl"] <= k
    out["_trajectory"] = pd.DataFrame({"episode": c["episode"], "timestep": t, "rank": rank, "rank_score": score,
                                       "vrl_accuracy": acc, "utility": util, "destructive": destr,
                                       "sparsity": np.array(T["total_sparsity"])[idx]})
    return out


def load_runs(cfg, selector=None):
    recs = []
    for f in sorted(os.listdir(L.RUNS)):
        if f.endswith(".json"):
            r = json.load(open(os.path.join(L.RUNS, f), encoding="utf-8"))
            if selector is None or selector(r):
                recs.append(r)
    return recs


def paired(values_a, values_b, rng, direction=+1):
    """values_*: dict seed -> value; returns the describe() dict plus direction."""
    assert sorted(values_a) == sorted(values_b) == L.SEEDS, "seed set is not the locked set"
    d = np.array([values_a[s] - values_b[s] for s in L.SEEDS], float)
    out = ST.describe(d, rng)
    out["direction_hypothesised"] = "positive" if direction > 0 else "negative"
    return out


def outcome(row, alpha=0.05):
    sign = 1 if row["direction_hypothesised"] == "positive" else -1
    m = row["mean_diff"] * sign
    lo, hi = (row["t_ci95_low"], row["t_ci95_high"]) if sign > 0 else (-row["t_ci95_high"], -row["t_ci95_low"])
    if m > 0 and row["holm_p"] < alpha and lo > 0:
        return "supported"
    if m < 0 and row["holm_p"] < alpha and hi < 0:
        return "opposite"
    return "inconclusive"


def cross_class(n_sup, n_opp):
    if n_sup >= 5 and n_opp == 0:
        c = "STRONG"
    elif n_sup >= 3 and n_opp == 0:
        c = "MODERATE"
    elif n_sup >= 1:
        c = "LIMITED"
    else:
        c = "NOT SUPPORTED"
    return c, bool(n_opp >= 3)
