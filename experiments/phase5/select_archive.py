"""Phase 5 archive freeze and VAL-SELECT selection (after all 360 runs; before any TEST access).

  python experiments/phase5/select_archive.py --freeze-archive     # top-10 unique VAL-RL-reward policies per run -> hashed
  python experiments/phase5/select_archive.py --select              # live VAL-SELECT evaluation of the frozen candidates,
                                                                    # selection, final-policy list -> hashed
Archive (VAL-RL only): every unique policy sampled in the 512 training episodes with its VAL-RL reward and first
episode; candidates = top 10 by VAL-RL reward (ties: earlier discovery).
Selection (VAL-SELECT only):
  C1, C2: among candidates with q_SELECT >= tau, the highest sparsity (ties: higher VAL-SELECT accuracy, then higher
          VAL-RL reward, then earlier discovery); if none is feasible, the highest q_SELECT (same tie order).
  C0:     the highest R0 score on VAL-SELECT (ties: higher VAL-RL reward, then earlier discovery).
Candidates are evaluated live on VAL-SELECT (1 thread) and must equal the precomputed VAL-SELECT landscape.
"""
import argparse
import datetime
import gzip
import io
import json
import os

import pandas as pd
import torch

import phase5_lib as L
from src import pruning as P
from src.evaluation import evaluate_accuracy
from src.utils import sha256_file

ARCHIVE = os.path.join(L.RESULTS, "phase5_archive.json")


def choose(cands, condition):
    """The pre-registered VAL-SELECT selection rule over the frozen candidates (dicts with keys r0_score_val_select,
    val_rl_reward, first_episode, feasible_val_select, sparsity, val_select_accuracy, q_val_select)."""
    if condition == "C0":
        return sorted(cands, key=lambda c: (-c["r0_score_val_select"], -c["val_rl_reward"], c["first_episode"]))[0], \
            "C0: max R0 score on VAL-SELECT"
    feas = [c for c in cands if c["feasible_val_select"]]
    if feas:
        return sorted(feas, key=lambda c: (-c["sparsity"], -c["val_select_accuracy"], -c["val_rl_reward"], c["first_episode"]))[0], \
            "C1/C2: feasible on VAL-SELECT, max sparsity"
    return sorted(cands, key=lambda c: (-c["q_val_select"], -c["sparsity"], -c["val_rl_reward"], c["first_episode"]))[0], \
        "C1/C2 fallback: no feasible candidate, max q on VAL-SELECT"


def curve(rec):
    raw = gzip.decompress(open(os.path.join(L.ROOT, rec["curve_file"]), "rb").read()).decode("utf-8")
    return pd.read_csv(io.StringIO(raw), float_precision="round_trip")


def records():
    recs = [json.load(open(os.path.join(L.RUNS, f), encoding="utf-8")) for f in sorted(os.listdir(L.RUNS)) if f.endswith(".json")]
    want = {(s, c, x) for s in L.SETTINGS for c in L.CONDITIONS for x in L.SEEDS}
    got = {(r["setting"], r["condition"], r["seed"]) for r in recs if r["status"] == "complete"}
    if got != want:
        raise SystemExit(f"{len(want - got)} runs missing; archive step refused")
    return recs


def freeze_archive():
    if os.path.exists(ARCHIVE):
        raise SystemExit("archive already frozen")
    cfg = json.load(open(L.CONFIG, encoding="utf-8"))
    out = []
    for r in records():
        c = curve(r)
        first, rew = {}, {}
        for _, row in c.iterrows():
            k = tuple(json.loads(row["actions"]))
            if k not in first:
                first[k], rew[k] = int(row["episode"]), float(row["episode_reward"])
        top = sorted(first, key=lambda k: (-rew[k], first[k]))[: L.ARCHIVE_K]
        out.append({"run_id": r["run_id"], "setting": r["setting"], "condition": r["condition"], "seed": r["seed"],
                    "unique_policies_sampled": len(first),
                    "candidates": [{"actions": list(k), "val_rl_reward": rew[k], "first_episode": first[k]} for k in top]})
    json.dump({"frozen_at": datetime.datetime.now().astimezone().isoformat(timespec="seconds"), "tau": cfg["tau"],
               "config_sha256": sha256_file(L.CONFIG), "runs": out}, open(ARCHIVE, "w", encoding="utf-8"), indent=1)
    print(f"archive frozen: {len(out)} runs; sha256 {sha256_file(ARCHIVE)}")


def select():
    if os.path.exists(L.FREEZE):
        raise SystemExit("final-policy list already frozen")
    torch.set_num_threads(L.THREADS)
    cfg = json.load(open(L.CONFIG, encoding="utf-8"))
    arc = json.load(open(ARCHIVE, encoding="utf-8"))
    tau = cfg["tau"]
    bundles, models, live = {}, {}, {}
    tables = {s: json.load(open(os.path.join(L.LAND, f"{s}_tables.json"), encoding="utf-8")) for s in L.SETTINGS}
    rows, final = [], []
    for run in arc["runs"]:
        s = run["setting"]
        d, a = L.P3.split(s)
        b = bundles.setdefault(d, L.bundle(d))
        if s not in models:
            models[s] = L.P3.load_reference(s)[0]
        T = tables[s]
        ads = cfg["settings"][s]["dense_val_select_accuracy"]
        cands = []
        for rank_rl, cnd in enumerate(run["candidates"], 1):
            key = (s, tuple(cnd["actions"]))
            if key not in live:
                live[key] = evaluate_accuracy(P.prune_actions(models[s], a, cnd["actions"]), b.select_loader())
            i = L.INDEX[tuple(cnd["actions"])]
            assert live[key] == T["val_select_accuracy"][i], "live VAL-SELECT != landscape"
            acc = live[key]
            cands.append({**cnd, "archive_rank": rank_rl, "val_select_accuracy": acc, "q_val_select": acc / ads,
                          "sparsity": T["total_sparsity"][i], "r0_score_val_select": L.r0(acc, ads, T["total_sparsity"][i], cnd["actions"]),
                          "feasible_val_select": acc / ads >= tau})
        chosen, rule = choose(cands, run["condition"])
        for c in cands:
            rows.append({"run_id": run["run_id"], "setting": s, "condition": run["condition"], "seed": run["seed"],
                         **{k: (str(v) if k == "actions" else v) for k, v in c.items()}, "selected": c is chosen})
        final.append({"run_id": run["run_id"], "setting": s, "condition": run["condition"], "seed": run["seed"],
                      "selected_policy": chosen["actions"], "rule": rule, "fallback": "fallback" in rule})
    pd.DataFrame(rows).to_csv(os.path.join(L.RESULTS, "phase5_archive_selection.csv"), index=False, float_format="%.17g")
    json.dump({"frozen_at": datetime.datetime.now().astimezone().isoformat(timespec="seconds"), "archive_sha256": sha256_file(ARCHIVE),
               "config_sha256": sha256_file(L.CONFIG), "tau": tau, "runs": final}, open(L.FREEZE, "w", encoding="utf-8"), indent=1)
    print(f"final policies frozen: {len(final)}; fallbacks {sum(f['fallback'] for f in final)}; sha256 {sha256_file(L.FREEZE)}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--freeze-archive", action="store_true")
    ap.add_argument("--select", action="store_true")
    a = ap.parse_args()
    freeze_archive() if a.freeze_archive else select() if a.select else ap.error("choose a step")
