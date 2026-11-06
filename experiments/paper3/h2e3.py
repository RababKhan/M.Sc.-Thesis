"""Paper 3, pre-PPO stage: ResNet-8 proxy-threshold calibration and the final structured sensitivity prior.

Protocols: results/paper3/resnet8_threshold_calibration_protocol.md, paper3_sensitivity_definition.md.
  python experiments/paper3/h2e3.py select                 # freeze the 48 calibration policies (uses q1 only)
  python experiments/paper3/h2e3.py worker --cpu C         # 5-epoch recovery of the 48 (frozen Phase-6 protocol)
  python experiments/paper3/h2e3.py calibrate              # tau_proxy rule, latency-room rule, report
  python experiments/paper3/h2e3.py sensitivity            # final immediate structured sensitivity (only if calibration GO)
  python experiments/paper3/h2e3.py freeze                 # paper3_preppo_freeze.json + integrity checks
The test set is never read.
"""
import argparse
import hashlib
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import p3_lib as L  # noqa: E402

R = L.phase_module("h2_recovery")
SETTING, PAIR = "cifar10_resnet8", "resnet8_c10"
LAND = os.path.join(L.RESULTS, "h2_c10_resnet8_1epoch_landscape.csv")
SEL = os.path.join(L.RESULTS, "resnet8_threshold_calibration_policies.json")
D5 = os.path.join(L.RESULTS, "_h2e3_calibration_5epoch")
DEC = os.path.join(L.RESULTS, "resnet8_threshold_calibration_decision.json")
SENS = os.path.join(L.RESULTS, "paper3_resnet8_sensitivity.csv")
BINS = [(0.85, 0.90), (0.90, 0.92), (0.92, 0.94), (0.94, 0.96), (0.96, 0.98), (0.98, float("inf"))]
PER_BIN, SEED, MARGIN, MIN_PROXY, MIN_TRUE, MIN_SPEEDUP = 8, 20261025, 0.005, 20, 5, 1.10
SENS_RATIOS = (0.1, 0.2, 0.4, 0.6)
BETA = 1.0


def sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def key(p):
    return "".join(map(str, p))


def landscape():
    import pandas as pd
    return pd.read_csv(LAND, float_precision="round_trip")


def cmd_select(_a):
    import numpy as np
    if os.path.exists(SEL):
        raise SystemExit("calibration policies are frozen")
    t = landscape()
    rng = np.random.default_rng(SEED)
    counts = [int(((t.q1 >= lo) & (t.q1 < hi)).sum()) for lo, hi in BINS]
    alloc = [PER_BIN] * len(BINS)
    for i in range(len(BINS) - 1, -1, -1):              # shortfall moves to the nearest lower-q1 bin
        if counts[i] < alloc[i]:
            short = alloc[i] - counts[i]
            alloc[i] = counts[i]
            if i > 0:
                alloc[i - 1] += short
    chosen = []
    for (lo, hi), n in zip(BINS, alloc):
        pool = t[(t.q1 >= lo) & (t.q1 < hi)].sort_values("policy").policy.tolist()
        idx = sorted(rng.choice(len(pool), min(n, len(pool)), replace=False).tolist())
        chosen += [{"policy": json.loads(pool[i]), "bin": [lo, hi if hi != float("inf") else None]} for i in idx]
    out = {"seed": SEED, "bins": [[lo, hi if hi != float("inf") else None] for lo, hi in BINS], "available_per_bin": counts,
           "allocated_per_bin": alloc, "selection_used": "q1 of the frozen 1-epoch landscape only (no latency, no 5-epoch value)",
           "landscape_sha256": sha(LAND), "policies": chosen}
    json.dump(out, open(SEL, "w", encoding="utf-8"), indent=1)
    print(f"selected {len(chosen)}; available per bin {counts}; allocated {alloc}; sha256 {sha(SEL)[:16]}")


def selected():
    return [tuple(x["policy"]) for x in json.load(open(SEL, encoding="utf-8"))["policies"]]


def out_path(p):
    return os.path.join(D5, SETTING, f"{key(p)}.json")


def cmd_worker(a):
    R.pin(a.cpu)
    ctx = R.Ctx()
    ctx.code["h2e3.py"] = sha(os.path.join(HERE, "h2e3.py"))
    proto = sha(os.path.join(L.RESULTS, "resnet8_threshold_calibration_protocol.md"))
    R.EPOCHS = 5
    for p in selected():
        f = out_path(p)
        if os.path.exists(f):
            continue
        os.makedirs(os.path.join(D5, "_locks"), exist_ok=True)
        try:
            fd = os.open(os.path.join(D5, "_locks", f"{key(p)}.lock"), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.close(fd)
        except FileExistsError:
            continue
        rec, _ = R.recover_one(ctx, SETTING, p, f, work_tag="_cal5")
        rec.update({"stage": "resnet8_threshold_calibration", "calibration_protocol_sha256": proto})
        R.durable(json.dumps(rec, indent=1).encode("utf-8"), f)
        print(f"{R.now()} done {key(p)} {rec['seconds']:.0f}s", flush=True)
    print(f"{R.now()} finished", flush=True)


def cmd_calibrate(_a):
    import numpy as np
    import pandas as pd
    if os.path.exists(DEC):
        raise SystemExit("calibration decision is frozen")
    t = landscape().set_index("policy")
    a5_ident = json.load(open(os.path.join(L.RESULTS, "_h2e2_validation_5epoch", SETTING, "0000.json"), encoding="utf-8"))["post"]["val_rl"]
    rows = []
    for p in selected():
        r = json.load(open(out_path(p), encoding="utf-8"))
        assert r["epochs"] == 5 and r["recovery"]["seed"] == 600 and r["shape_unchanged"] and r["test_use"] == "none"
        k = str(list(p))
        rows.append({"policy": k, "q1": float(t.loc[k, "q1"]), "A1": float(t.loc[k, "A1_val_rl"]), "A5": r["post"]["val_rl"],
                     "q5": r["post"]["val_rl"] / a5_ident})
    c = pd.DataFrame(rows)
    c["delta_q5_minus_q1"] = c.q5 - c.q1
    c["true_feasible"] = c.q5 >= 0.98
    n_true = int(c.true_feasible.sum())
    land = t.q1.to_numpy()
    dec = {"a5_identity_val_rl": a5_ident, "n_calibration": len(c), "n_true_feasible": n_true, "rule": {
        "t": "largest q1 threshold with 100% recall of the calibration policies with q5 >= 0.98 and >= 20 proxy-feasible policies in the 1,296 landscape",
        "tau_proxy": "max(0, t - 0.005)", "stop_if_true_feasible_below": MIN_TRUE}}
    if n_true < MIN_TRUE:
        dec.update({"go_threshold": False, "reason": f"only {n_true} calibration policies with q5 >= 0.98 (< {MIN_TRUE})"})
    else:
        t_recall = float(c[c.true_feasible].q1.min())                    # any t <= this has 100% recall
        sorted_land = np.sort(land)[::-1]
        t_count = float(sorted_land[MIN_PROXY - 1])                      # largest t leaving >= 20 policies
        t_star = min(t_recall, t_count)
        tau = max(0.0, t_star - MARGIN)
        pred = c.q1 >= tau
        tp, fp = int((pred & c.true_feasible).sum()), int((pred & ~c.true_feasible).sum())
        fn, tn = int((~pred & c.true_feasible).sum()), int((~pred & ~c.true_feasible).sum())
        feas = t[t.q1 >= tau]
        order = np.lexsort((-feas.A1_val_rl.values, feas.latency_ms.values))
        front, best = 0, -1
        for i in order:
            if feas.A1_val_rl.values[i] > best:
                front, best = front + 1, feas.A1_val_rl.values[i]
        room = {"proxy_feasible": len(feas), "latency_min_ms": float(feas.latency_ms.min()), "latency_max_ms": float(feas.latency_ms.max()),
                "best_speedup": float(feas.speedup.max()), "param_reduction_range": [float(feas.param_reduction.min()), float(feas.param_reduction.max())],
                "mac_reduction_range": [float(feas.mac_reduction.min()), float(feas.mac_reduction.max())], "pareto_policies": front,
                "policies_with_speedup_ge_1.10": int((feas.speedup >= MIN_SPEEDUP).sum())}
        go = bool(len(feas) >= MIN_PROXY and feas.speedup.max() >= MIN_SPEEDUP)
        dec.update({"go_threshold": True, "t_recall": t_recall, "t_count": t_count, "t": t_star, "tau_proxy": tau,
                    "recall": tp / (tp + fn), "precision": tp / (tp + fp) if tp + fp else None, "tp": tp, "fp": fp, "fn": fn, "tn": tn,
                    "latency_room": room, "go_ppo": go,
                    "go_rule": f">= {MIN_PROXY} proxy-feasible policies and >= 1 with table speed-up >= {MIN_SPEEDUP}"})
    c.to_csv(os.path.join(L.RESULTS, "resnet8_threshold_calibration.csv"), index=False, float_format="%.6g")
    dec["delta_summary"] = {"mean": float(c.delta_q5_minus_q1.mean()), "median": float(c.delta_q5_minus_q1.median()),
                            "min": float(c.delta_q5_minus_q1.min()), "max": float(c.delta_q5_minus_q1.max()),
                            "share_positive": float((c.delta_q5_minus_q1 > 0).mean())}
    dec["spearman_q1_q5"] = float(c.q1.rank().corr(c.q5.rank()))
    dec["calibration_policies_sha256"] = sha(SEL)
    json.dump(dec, open(DEC, "w", encoding="utf-8"), indent=1)
    print(json.dumps({k: v for k, v in dec.items() if k not in ("rule",)}, indent=1, default=str))


def cmd_sensitivity(_a):
    import torch
    from src.data import eval_loader
    from src.evaluation import evaluate_accuracy
    from src.pruning import structured_units as SU
    from src.utils import sha256_state_dict
    dec = json.load(open(DEC, encoding="utf-8"))
    if not dec.get("go_ppo"):
        raise SystemExit("threshold calibration is not GO: no final sensitivity vector")
    if os.path.exists(SENS):
        raise SystemExit("sensitivity vector exists; it is frozen")
    torch.set_num_threads(1)
    ref, ref_sha = L.reference(SETTING)
    h0 = sha256_state_dict(ref.state_dict())
    x, y, _ = L.val_rl("cifar10")
    loader = eval_loader(x, y)
    a_dense = evaluate_accuracy(ref, loader)
    import pandas as pd
    rows = []
    for u, name in enumerate(SU.UNITS["resnet8"]):
        drops = {}
        for r in SENS_RATIOS:
            m, _ = SU.slim(ref, "resnet8", SU.single_unit_actions(u, r))   # immediate: no recovery
            drops[r] = (a_dense - evaluate_accuracy(m, loader)) / a_dense
        rows.append({"unit_index": u, "unit": name, **{f"S_at_{int(100 * r)}": v for r, v in drops.items()},
                     "S_raw": sum(drops.values()) / len(drops)})
    s = pd.DataFrame(rows)
    lo, hi = s.S_raw.min(), s.S_raw.max()
    s["S_normalized"] = (s.S_raw - lo) / (hi - lo) if hi > lo else 0.0
    s["prior_coefficient"] = BETA * (s.S_normalized - s.S_normalized.mean())
    s["a_dense_val_rl"] = a_dense
    s["reference_sha256"] = ref_sha
    assert sha256_state_dict(ref.state_dict()) == h0, "dense model changed"
    s.to_csv(SENS, index=False, float_format="%.9g")
    print(s.to_string(index=False))


def cmd_freeze(_a):
    import datetime
    import pandas as pd
    checks = []

    def chk(n, ok, d=""):
        checks.append({"check": n, "pass": bool(ok), "detail": str(d)})
        print(f"[{'PASS' if ok else 'FAIL'}] {n}: {str(d)[:150]}", flush=True)
    dec = json.load(open(DEC, encoding="utf-8"))
    sel = json.load(open(SEL, encoding="utf-8"))
    recs = [json.load(open(out_path(p), encoding="utf-8")) for p in selected()]
    when = datetime.datetime.fromisoformat
    commit_time = subprocess.run(["git", "log", "-1", "--format=%cI", "--", "results/paper3/resnet8_threshold_calibration_protocol.md"],
                                 cwd=L.ROOT, capture_output=True, text=True).stdout.strip()
    chk("48 calibration policies frozen from the landscape q1 only; landscape unchanged since", len(sel["policies"]) == 48
        and sel["landscape_sha256"] == sha(LAND), sel["allocated_per_bin"])
    chk("calibration protocol and policy list committed before the first 5-epoch calibration run",
        when(commit_time) < min(when(r["started_at"]) for r in recs), f"{commit_time} < {min(r['started_at'] for r in recs)}")
    chk("every calibration run: 5 epochs, seed 600, Phase-6 protocol, shapes unchanged, no test use, frozen reference",
        all(r["epochs"] == 5 and r["recovery"]["seed"] == 600 and r["shape_unchanged"] and r["test_use"] == "none"
            and r["reference_sha256"] == recs[0]["reference_sha256"] for r in recs) and len(recs) == 48, len(recs))
    chk("one data stream across all calibration runs", len({r["data_stream_sha256"] for r in recs}) == 1, "")
    lat = json.load(open(os.path.join(L.RESULTS, "h2_latency_freeze.json"), encoding="utf-8"))
    chk("frozen H2-A ResNet-8 latency table unchanged", sha(os.path.join(L.RESULTS, f"h2_latency_table_{PAIR}.csv")) == lat["tables"][PAIR]["sha256"], "")
    e2 = json.load(open(os.path.join(L.RESULTS, "h2_e2_freeze.json"), encoding="utf-8"))
    chk("frozen E2 ResNet-8 1-epoch landscape unchanged", sha(LAND) == e2["landscapes"][SETTING]["sha256"], "")
    s = pd.read_csv(SENS)
    sdef = subprocess.run(["git", "log", "-1", "--format=%cI", "--", "results/paper3/paper3_sensitivity_definition.md"],
                          cwd=L.ROOT, capture_output=True, text=True).stdout.strip()
    chk("sensitivity definition committed before the sensitivity vector was computed",
        when(sdef) < datetime.datetime.fromtimestamp(os.path.getmtime(SENS)).astimezone(), sdef)
    chk("four structured sensitivity values, normalised to [0, 1], centred coefficients sum to 0", len(s) == 4 and s.S_normalized.min() == 0
        and s.S_normalized.max() == 1 and abs(s.prior_coefficient.sum()) < 1e-12, s.S_raw.round(4).tolist())
    out = {"created": R.now(), "setting": SETTING, "tau_proxy": dec.get("tau_proxy"), "go_ppo": dec.get("go_ppo"),
           "files": {f: sha(os.path.join(L.RESULTS, f)) for f in (
               "h2_c10_resnet8_1epoch_landscape.csv", f"h2_latency_table_{PAIR}.csv", "resnet8_threshold_calibration_policies.json",
               "resnet8_threshold_calibration.csv", "resnet8_threshold_calibration_decision.json", "paper3_resnet8_sensitivity.csv",
               "paper3_sensitivity_definition.md", "resnet8_threshold_calibration_protocol.md", "paper3_proxy_protocol.md")},
           "prior_coefficients": s.prior_coefficient.tolist(), "checks": checks}
    json.dump(out, open(os.path.join(L.RESULTS, "paper3_preppo_freeze.json"), "w", encoding="utf-8"), indent=1)
    print(f"pre-PPO integrity {sum(c['pass'] for c in checks)}/{len(checks)}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    for n in ("select", "calibrate", "sensitivity", "freeze"):
        sub.add_parser(n)
    w = sub.add_parser("worker")
    w.add_argument("--cpu", type=int, required=True)
    a = ap.parse_args()
    sys.path.insert(0, L.ROOT)
    {"select": cmd_select, "worker": cmd_worker, "calibrate": cmd_calibrate, "sensitivity": cmd_sensitivity, "freeze": cmd_freeze}[a.cmd](a)
