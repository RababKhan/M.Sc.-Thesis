"""Pre-registered stage selection (validation only; TEST stays locked). Appends one stage at a time to
results/phase4/phase4_stage_selection.json and refuses to redo a stage.

  python experiments/phase4/select_stage.py --stage A|B|C

Stage A (reward carried into B and C). For reward R, I(R) = population SD over the six settings of the
setting-mean final sparsity (20 seeds each).
  A1  I(R)/I(R0) <= 0.5 and the 95% seed-bootstrap upper bound of the ratio < 1 (10,000 resamples of the
      20 seeds, the same seeds in every setting and reward; numpy default_rng(0))
  A2  setting-mean final V_RL retention (A/A_b) >= 0.95 in at least 5 of 6 settings
  winner = among R1a, R1b, R2 meeting A1 and A2, the smallest I(R); none -> R0.
Stage B (prior carried into C). Primary endpoint: reward-rank AUC under the Stage-A reward.
  Family (12): P1-P0 and P2-P0 per setting, exact sign-flip, Holm; outcome rule as Phase 3.
  eligible = supported in >= 3 settings and opposite in none; winner = the eligible prior supported in more
  settings (tie: larger mean d_z over settings); none -> P0.
Stage C (search variant). For S0, S1, S2, S2A: per setting the median over seeds of the V_SELECT rank of the
final-selected policy; winner = the lowest mean over settings (ties: S0, S1, S2, S2A order).
"""
import argparse
import datetime
import json
import os

import numpy as np

import phase4_lib as L
import phase4_metrics as M
from src import statistics as ST
from src.utils import sha256_file


def load_sel():
    return json.load(open(L.SELECTION, encoding="utf-8")) if os.path.exists(L.SELECTION) else {}


def need(recs, configs):
    ids = {(r["setting"], r["reward"], r["prior"], r["variant"], r["seed"]) for r in recs if r["status"] == "complete"}
    want = {(s, *c, seed) for s in L.SETTINGS for c in configs for seed in L.SEEDS}
    missing = want - ids
    if missing:
        raise SystemExit(f"{len(missing)} required runs missing; selection refused")


def stage_a(cfg):
    recs = M.load_runs(cfg, lambda r: r["prior"] == "P0" and r["variant"] == "S0")
    need(recs, [(r, "P0", "S0") for r in L.REWARDS])
    met = [M.run_metrics(r, cfg) for r in recs]
    sp = {(m["reward"], m["setting"], m["seed"]): m["final_sparsity"] for m in met}
    ret = {(m["reward"], m["setting"], m["seed"]): m["final_vrl_retention"] for m in met}

    def I(reward, seeds):
        means = [np.mean([sp[(reward, s, x)] for x in seeds]) for s in L.SETTINGS]
        return float(np.std(means))

    rng = np.random.default_rng(0)
    boot = [rng.choice(L.SEEDS, size=len(L.SEEDS), replace=True) for _ in range(10000)]
    out = {"I": {r: I(r, L.SEEDS) for r in L.REWARDS}, "criteria": {}}
    base = out["I"]["R0"]
    for r in ("R1a", "R1b", "R2"):
        ratios = np.array([I(r, b) / I("R0", b) for b in boot])
        ratio = out["I"][r] / base
        a1 = bool(ratio <= 0.5 and np.percentile(ratios, 97.5) < 1.0)
        mean_ret = {s: float(np.mean([ret[(r, s, x)] for x in L.SEEDS])) for s in L.SETTINGS}
        n_ok = sum(v >= L.RETENTION_TOL for v in mean_ret.values())
        out["criteria"][r] = {"ratio": ratio, "ratio_boot_ci95": [float(np.percentile(ratios, 2.5)), float(np.percentile(ratios, 97.5))],
                              "A1": a1, "mean_vrl_retention": mean_ret, "settings_retention_ok": int(n_ok), "A2": bool(n_ok >= 5)}
    passing = [r for r in ("R1a", "R1b", "R2") if out["criteria"][r]["A1"] and out["criteria"][r]["A2"]]
    out["passing"] = passing
    out["winner"] = min(passing, key=lambda r: out["I"][r]) if passing else "R0"
    out["mean_final_sparsity"] = {r: {s: float(np.mean([sp[(r, s, x)] for x in L.SEEDS])) for s in L.SETTINGS} for r in L.REWARDS}
    return out


def stage_b(cfg, sel):
    R = sel["A"]["winner"]
    recs = M.load_runs(cfg, lambda r: r["reward"] == R and r["variant"] == "S0")
    need(recs, [(R, p, "S0") for p in L.PRIORS])
    met = {(m["prior"], m["setting"], m["seed"]): m for m in (M.run_metrics(r, cfg) for r in recs)}
    rng = np.random.default_rng(0)
    rows = []
    for p in ("P1", "P2"):
        for s in L.SETTINGS:
            a = {x: met[(p, s, x)]["rank_auc"] for x in L.SEEDS}
            b = {x: met[("P0", s, x)]["rank_auc"] for x in L.SEEDS}
            rows.append({"prior": p, "setting": s, **M.paired(a, b, rng)})
    for row, h in zip(rows, ST.holm([r["sign_flip_p"] for r in rows])):
        row["holm_p"] = float(h)
        row["outcome"] = M.outcome(row)
    summary = {}
    for p in ("P1", "P2"):
        rr = [r for r in rows if r["prior"] == p]
        summary[p] = {"supported": sum(r["outcome"] == "supported" for r in rr),
                      "opposite": sum(r["outcome"] == "opposite" for r in rr),
                      "mean_dz": float(np.mean([r["cohens_dz"] for r in rr]))}
    eligible = [p for p in ("P1", "P2") if summary[p]["supported"] >= 3 and summary[p]["opposite"] == 0]
    winner = max(eligible, key=lambda p: (summary[p]["supported"], summary[p]["mean_dz"])) if eligible else "P0"
    return {"reward": R, "tests": [{k: (float(v) if isinstance(v, (np.floating,)) else v) for k, v in r.items()} for r in rows],
            "summary": summary, "eligible": eligible, "winner": winner}


def stage_c(cfg, sel):
    R, Pr = sel["A"]["winner"], sel["B"]["winner"]
    recs = M.load_runs(cfg, lambda r: r["reward"] == R and r["prior"] == Pr)
    need(recs, [(R, Pr, "S0"), (R, Pr, "S2")])
    met = {(m["variant"], m["setting"], m["seed"]): m for m in (M.run_metrics(r, cfg) for r in recs)}
    rule = {"S0": ("S0", "final"), "S1": ("S0", "archive"), "S2": ("S2", "final"), "S2A": ("S2", "archive")}
    med = {c: {s: float(np.median([met[(v, s, x)][f"{w}_rank_vselect"] for x in L.SEEDS])) for s in L.SETTINGS}
           for c, (v, w) in rule.items()}
    score = {c: float(np.mean(list(med[c].values()))) for c in rule}
    winner = min(L.SEARCH, key=lambda c: (score[c], L.SEARCH.index(c)))
    return {"reward": R, "prior": Pr, "median_vselect_rank": med, "mean_over_settings": score, "winner": winner}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["A", "B", "C"])
    args = ap.parse_args()
    cfg = json.load(open(L.CONFIG, encoding="utf-8"))
    sel = load_sel()
    if args.stage in sel:
        raise SystemExit(f"stage {args.stage} already selected; frozen")
    if args.stage == "B" and "A" not in sel or args.stage == "C" and "B" not in sel:
        raise SystemExit("previous stage not selected")
    res = {"A": stage_a, "B": lambda c: stage_b(c, sel), "C": lambda c: stage_c(c, sel)}[args.stage](cfg)
    res.update({"selected_at": datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
                "test_set_accessed": False, "config_sha256": sha256_file(L.CONFIG),
                "script_sha256": sha256_file(os.path.abspath(__file__))})
    sel[args.stage] = res
    tmp = L.SELECTION + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(sel, f, indent=1)
    os.replace(tmp, L.SELECTION)
    print(f"stage {args.stage} winner: {res['winner']}")


if __name__ == "__main__":
    main()
