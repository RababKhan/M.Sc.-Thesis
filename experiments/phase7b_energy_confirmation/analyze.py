"""Phase 7B energy confirmation: statistics, ENERGY-CONFIRMED decision, final table and figures (rules: protocol.json).

  python experiments/phase7b_energy_confirmation/analyze.py
Inputs: measurement_qualification.json, _raw/sessions/session*.json, _raw/secondary/secondary*.json (if present),
checkpoint_verification.json, and the frozen Phase 7B accuracy / memory / MAC results (read only).
Outputs: energy_raw.csv, energy_summary.csv, latency_summary.csv, final_comparison.csv, statistics.json, figures/.
"""
import csv
import datetime
import glob
import json
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(ROOT, "experiments", "phase7b"))
import final_analysis as FA  # noqa: E402  (exact Wilcoxon and figure style, read only)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

OUT = os.path.join(ROOT, "results", "phase7b_energy_confirmation")
RAW = os.path.join(OUT, "_raw")
FIG = os.path.join(OUT, "figures")
P7B = os.path.join(ROOT, "results", "phase7b_green_pruning")
DENSE, PPO = "0000", "2225"
NAME = {DENSE: "Dense", PPO: "Phase 7B PPO-Structured", "uniform_r0.208": "Uniform-Structured",
        "lamp_s56.2": "LAMP-Structured", "phase7a_ppo": "Old PPO-Structured (7A)"}
KEY = {DENSE: "dense", PPO: "new_ppo", "uniform_r0.208": "uniform", "lamp_s56.2": "lamp", "phase7a_ppo": "old_ppo"}


def jload(p):
    return json.load(open(p, encoding="utf-8"))


def write_csv(name, rows):
    if not rows:
        return
    with open(os.path.join(OUT, name), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows([{k: (json.dumps(v) if isinstance(v, (list, dict)) else v) for k, v in r.items()} for r in rows])


def cv(v):
    v = np.asarray(v, dtype=float)
    return float(v.std(ddof=1) / v.mean())


def describe(v):
    v = np.asarray(v, dtype=float)
    return {"median": float(np.median(v)), "mean": float(v.mean()), "sd": float(v.std(ddof=1)), "cv": cv(v),
            "min": float(v.min()), "max": float(v.max()), "n": int(len(v))}


def sign_test(wins, n):
    """Exact one-sided sign test: P(X >= wins), X ~ Binomial(n, 1/2)."""
    return float(sum(math.comb(n, k) for k in range(wins, n + 1)) / 2 ** n)


def runs(stage):
    return [jload(p) for p in sorted(glob.glob(os.path.join(RAW, stage, f"{stage[:-1] if stage == 'sessions' else stage}*.json")))
            if not p.endswith("run_log.json")]


def paired(sess, mid):
    """Per-session rows of model mid against the same session's Dense block."""
    rows = []
    for r in sess:
        b = {x["model"]: x for x in r["blocks"]}
        d, p = b[DENSE], b[mid]
        rows.append({"session": r["run"], "order": r["order"], "e_dense": d["e1000_j"], "e_model": p["e1000_j"],
                     "saving_pct": 100 * (1 - p["e1000_j"] / d["e1000_j"]), "lower": p["e1000_j"] < d["e1000_j"],
                     "lat_dense_ms": d["latency_median_ms"], "lat_model_ms": p["latency_median_ms"],
                     "speedup": d["latency_median_ms"] / p["latency_median_ms"],
                     "edp_norm": (p["j_per_inference"] * p["latency_wall_ms"]) / (d["j_per_inference"] * d["latency_wall_ms"])})
    return rows


def decide(rows, qual_cv, qualified):
    sav = np.array([r["saving_pct"] for r in rows])
    low = np.array([r["lower"] for r in rows])
    noise = 2 * max(qual_cv, cv([r["e_dense"] for r in rows])) * 100
    loo = [{"omitted": rows[i]["session"], "median_saving_pct": float(np.median(np.delete(sav, i))),
            "lower_count": int(np.delete(low, i).sum())} for i in range(len(rows))]
    c = {"1_qualified": bool(qualified),
         "2_lower_in_at_least_8_of_10": bool(len(rows) == 10 and low.sum() >= 8),
         "3_median_saving_above_noise": bool(np.median(sav) > noise),
         "4_leave_one_out": bool(len(rows) == 10 and all(x["median_saving_pct"] > noise and x["lower_count"] >= 7 for x in loo))}
    return c, noise, loo


def figures(rows, table):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update(FA.RC)
    os.makedirs(FIG, exist_ok=True)
    fig, ax = plt.subplots(figsize=(3.5, 2.6))
    xs = np.arange(1, len(rows) + 1)
    for x, r in zip(xs, rows):
        ax.plot([x, x], [r["e_dense"], r["e_model"]], color=FA.MUTED, lw=1.0, zorder=1)
    ax.scatter(xs, [r["e_dense"] for r in rows], s=22, marker="s", color=FA.COLOR["dense"], label="Dense", zorder=3)
    ax.scatter(xs, [r["e_model"] for r in rows], s=26, marker="o", color=FA.COLOR["new_ppo"], label="Phase 7B PPO", zorder=3)
    ax.set_xticks(xs)
    ax.set_xlabel("Session")
    ax.set_ylabel("Energy per 1,000 inferences (J)")
    ax.legend(frameon=False, loc="best")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig1_session_energy.png"))
    plt.close(fig)

    p = next(t for t in table if t["id"] == PPO)
    labels = ["Memory", "MACs", "Latency", "Energy", "EDP"]
    vals = [1 - p["memory_reduction"], 1 - p["mac_reduction"], 1 / p["speedup"], 1 - p["energy_saving_pct"] / 100,
            p["edp_norm"]]
    fig, ax = plt.subplots(figsize=(3.5, 2.4))
    ax.bar(labels, vals, width=0.55, color=FA.COLOR["new_ppo"], zorder=2)
    ax.axhline(1.0, color=FA.INK2, lw=0.9, ls="--", zorder=3)
    ax.text(4.45, 1.02, "Dense = 1.0", ha="right", va="bottom", fontsize=7, color=FA.INK2)
    for i, v in enumerate(vals):
        ax.text(i, v + 0.02, f"{v:.2f}", ha="center", va="bottom", fontsize=7, color=FA.INK)
    ax.set_ylim(0, 1.15)
    ax.set_ylabel("Phase 7B PPO / Dense")
    ax.grid(axis="x", visible=False)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig2_normalized_metrics.png"))
    plt.close(fig)


def main():
    proto = jload(os.path.join(OUT, "protocol.json"))
    ver = jload(os.path.join(OUT, "checkpoint_verification.json"))
    assert ver["all_match"], "checkpoint integrity failure"
    q = jload(os.path.join(OUT, "measurement_qualification.json"))
    qualified = q["decision"] == "qualified"
    sess = runs("sessions") if qualified else []
    sec = runs("secondary") if qualified else []
    S7 = pd.read_csv(os.path.join(P7B, "final_summary.csv")).set_index("id")

    raw = [{"stage": st, "run": r["run"], **{k: v for k, v in b.items()}}
           for st, rs in (("qualification", [jload(p) for p in sorted(glob.glob(os.path.join(RAW, "qualification", "qual*.json")))]),
                          ("sessions", sess), ("secondary", sec)) for r in rs for b in r["blocks"]]
    write_csv("energy_raw.csv", raw)

    stats = {"created": datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
             "protocol_sha256": FA.L.sha256_file(os.path.join(OUT, "protocol.json")),
             "checkpoints_verified": ver["all_match"],
             "qualification": {k: q[k] for k in ("e1000_j", "mean_j", "sd_j", "cv", "decision", "criterion")}}
    if not qualified or len(sess) < proto["sessions"]["count"]:
        stats["energy_confirmed"] = "NO"
        stats["reason"] = ("measurement setup not qualified" if not qualified
                           else f"only {len(sess)} of {proto['sessions']['count']} sessions complete")
        stats["claim"] = proto["claims"]["if_unstable"]
        json.dump(stats, open(os.path.join(OUT, "statistics.json"), "w", encoding="utf-8"), indent=1)
        print(json.dumps(stats, indent=1))
        return

    rows = paired(sess, PPO)
    crit, noise, loo = decide(rows, q["cv"], qualified)
    confirmed = all(crit.values())
    sav = [r["saving_pct"] for r in rows]
    ed, ep = [r["e_dense"] for r in rows], [r["e_model"] for r in rows]
    wins = int(sum(r["lower"] for r in rows))
    lower_med = float(np.median(ep)) < float(np.median(ed))
    claim = (proto["claims"]["if_yes"] if confirmed
             else proto["claims"]["if_no_energy_not_lower"] if not lower_med or wins < len(rows) / 2
             else "ENERGY-CONFIRMED = NO: the measured energy difference did not meet the pre-registered confirmation rule.")
    stats.update({
        "sessions": rows, "noise_threshold_pct": noise,
        "dense_e1000": describe(ed), "ppo_e1000": describe(ep), "saving_pct": describe(sav),
        "ppo_lower_sessions": {"count": wins, "of": len(rows)},
        "wilcoxon_exact_p_one_sided_ppo_lower": FA.wilcoxon(ep, ed, "less"),
        "wilcoxon_exact_p_two_sided": FA.wilcoxon(ep, ed, "two-sided"),
        "sign_test_exact_p_one_sided": sign_test(wins, len(rows)),
        "latency": {"dense_median_ms": describe([r["lat_dense_ms"] for r in rows]),
                    "ppo_median_ms": describe([r["lat_model_ms"] for r in rows]),
                    "speedup": describe([r["speedup"] for r in rows]),
                    "ppo_faster_sessions": int(sum(r["lat_model_ms"] < r["lat_dense_ms"] for r in rows))},
        "edp_norm": describe([r["edp_norm"] for r in rows]),
        "leave_one_out": loo, "criteria": crit, "energy_confirmed": "YES" if confirmed else "NO", "claim": claim,
        "not_allowed": proto["claims"]["never"]})

    table, secondary = [], {}
    for mid in (DENSE, PPO, "uniform_r0.208", "lamp_s56.2", "phase7a_ppo"):
        a = S7.loc[mid]
        base = {"id": mid, "method": NAME[mid], "accuracy_val_select": float(a.val_select_post),
                "delta_acc_pp": float(a.delta_val_select_pp), "memory_bytes": int(a.memory_bytes),
                "memory_reduction": float(a.memory_reduction), "macs": int(a.macs), "mac_reduction": float(a.mac_reduction)}
        if mid in (DENSE, PPO):
            src = rows
            e = [r["e_dense"] if mid == DENSE else r["e_model"] for r in src]
            lat = [r["lat_dense_ms"] if mid == DENSE else r["lat_model_ms"] for r in src]
            sp = [1.0] * len(src) if mid == DENSE else [r["speedup"] for r in src]
            sv = [0.0] * len(src) if mid == DENSE else sav
            edp = [1.0] * len(src) if mid == DENSE else [r["edp_norm"] for r in src]
            source = f"primary paired sessions (n = {len(src)})"
        elif sec:
            src = paired(sec, mid)
            secondary[KEY[mid]] = {"sessions": src, "saving_pct": describe([r["saving_pct"] for r in src]),
                                   "lower_sessions": int(sum(r["lower"] for r in src)),
                                   "speedup": describe([r["speedup"] for r in src]),
                                   "edp_norm": describe([r["edp_norm"] for r in src])}
            e, lat = [r["e_model"] for r in src], [r["lat_model_ms"] for r in src]
            sp, sv, edp = [r["speedup"] for r in src], [r["saving_pct"] for r in src], [r["edp_norm"] for r in src]
            source = f"secondary sessions, vs same-session Dense (n = {len(src)}; descriptive)"
        else:
            table.append({**base, "latency_median_ms": None, "speedup": None, "e1000_median_j": None,
                          "energy_saving_pct": None, "edp_norm": None, "source": "not measured"})
            continue
        table.append({**base, "latency_median_ms": float(np.median(lat)), "speedup": float(np.median(sp)),
                      "e1000_median_j": float(np.median(e)), "energy_saving_pct": float(np.median(sv)),
                      "edp_norm": float(np.median(edp)), "source": source})
    if sec:
        stats["secondary"] = secondary
    write_csv("final_comparison.csv", table)
    write_csv("energy_summary.csv", [{"id": t["id"], "method": t["method"], "source": t["source"],
                                      "e1000_median_j": t["e1000_median_j"], "j_per_inference_median": (t["e1000_median_j"] or 0) / 1000 or None,
                                      "energy_saving_median_pct": t["energy_saving_pct"]} for t in table] +
              [{"id": mid, "method": NAME[mid], "source": "primary paired sessions: distribution",
                **{f"e1000_{k}": v for k, v in describe(e).items()}} for mid, e in ((DENSE, ed), (PPO, ep))])
    write_csv("latency_summary.csv", [{"id": t["id"], "method": t["method"], "source": t["source"],
                                       "latency_median_ms": t["latency_median_ms"], "speedup_median": t["speedup"],
                                       "edp_norm_median": t["edp_norm"]} for t in table])
    try:
        figures(rows, table)
        stats["figures"] = ["figures/fig1_session_energy.png", "figures/fig2_normalized_metrics.png"]
    except ImportError as exc:
        stats["figures"] = f"not produced: {exc}"
    stats["final_comparison"] = table
    json.dump(stats, open(os.path.join(OUT, "statistics.json"), "w", encoding="utf-8"), indent=1, default=float)
    print(json.dumps({k: stats[k] for k in ("criteria", "energy_confirmed", "noise_threshold_pct", "saving_pct",
                                            "ppo_lower_sessions", "wilcoxon_exact_p_one_sided_ppo_lower", "edp_norm")},
                     indent=1, default=float))
    print(pd.DataFrame(table)[["method", "accuracy_val_select", "delta_acc_pp", "memory_reduction", "mac_reduction",
                               "latency_median_ms", "speedup", "e1000_median_j", "energy_saving_pct", "edp_norm"]].to_string())


if __name__ == "__main__":
    main()
