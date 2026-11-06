"""Paper 3, E2 (amendment 2, results/paper3/paper3_proxy_protocol.md): 1-epoch proxy validation and landscapes.

  python experiments/paper3/h2e2.py worker --stage validation --cpu C   # 61 new 5-epoch + 187 1-epoch validation runs
  python experiments/paper3/h2e2.py validate                            # proxy statistics, GO/NO-GO, tau1
  python experiments/paper3/h2e2.py worker --stage landscape --cpu C    # 1-epoch landscapes of the GO architectures
  python experiments/paper3/h2e2.py merge                               # landscapes + integrity + freeze
Recovery = h2_recovery.recover_one (verified, unchanged) with E = 1 or E = 5. Test set never read.
"""
import argparse
import datetime
import hashlib
import itertools
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import p3_lib as L  # noqa: E402

R = L.phase_module("h2_recovery")
POLICIES = [tuple(p) for p in itertools.product(range(6), repeat=4)]
PROTOCOL = os.path.join(L.RESULTS, "paper3_proxy_protocol.md")
VPOL = json.load(open(os.path.join(L.RESULTS, "paper3_proxy_validation_policies.json"), encoding="utf-8"))
D5 = os.path.join(L.RESULTS, "_h2e2_validation_5epoch")
D1 = os.path.join(L.RESULTS, "_h2e2_1epoch_runs")
DECISION = os.path.join(L.RESULTS, "paper3_proxy_decision.json")
ARCH = {"cifar10_simplecnn": "simplecnn", "cifar10_resnet8": "resnet8"}
GO_RHO = 0.90


def sha(path):
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


def key(p):
    return "".join(map(str, p))


def path(base, setting, p):
    return os.path.join(base, setting, f"{key(p)}.json")


def validation_pairs():
    s, r = VPOL["cifar10_simplecnn"], VPOL["cifar10_resnet8"]
    return {"cifar10_simplecnn": [tuple(p) for p in s["pilot_5epoch_policies"] + s["random"]],
            "cifar10_resnet8": [tuple(p) for p in r["fixed"] + r["random"]]}


def jobs(stage):
    if stage == "validation":
        v = validation_pairs()
        five = [("cifar10_simplecnn", tuple(p), 5) for p in VPOL["cifar10_simplecnn"]["random"]] + \
               [("cifar10_resnet8", p, 5) for p in v["cifar10_resnet8"]]
        one = [(s, p, 1) for s in v for p in v[s]]
        return five + one
    dec = json.load(open(DECISION, encoding="utf-8"))
    return [(s, p, 1) for s in ARCH if dec["architectures"][s]["go"] for p in POLICIES]


def out_path(setting, p, epochs):
    return path(D5 if epochs == 5 else D1, setting, p)


def done(setting, p, epochs):
    f = out_path(setting, p, epochs)
    try:
        r = json.load(open(f, encoding="utf-8"))
        return r["status"] == "complete" and r["epochs"] == epochs
    except Exception:                                        # noqa: BLE001
        return False


def claim(setting, p, epochs):
    d = os.path.join(D5 if epochs == 5 else D1, "_locks")
    os.makedirs(d, exist_ok=True)
    try:
        fd = os.open(os.path.join(d, f"{setting}_{key(p)}.lock"), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        os.write(fd, str(os.getpid()).encode())
        os.close(fd)
        return True
    except FileExistsError:
        return False


def cmd_worker(a):
    R.pin(a.cpu)
    ctx = R.Ctx()
    ctx.code["h2e2.py"] = sha(os.path.join(HERE, "h2e2.py"))
    proto = sha(PROTOCOL)
    n = 0
    for setting, p, epochs in jobs(a.stage):
        if done(setting, p, epochs) or not claim(setting, p, epochs):
            continue
        R.EPOCHS = epochs                                    # recover_one reads the module constant at call time
        f = out_path(setting, p, epochs)
        rec, _ = R.recover_one(ctx, setting, p, f, work_tag=f"_e{epochs}")
        rec.update({"e2_stage": a.stage, "proxy_protocol_sha256": proto})
        R.durable(json.dumps(rec, indent=1).encode("utf-8"), f)
        n += 1
        print(f"{R.now()} done {a.stage} {setting} {key(p)} E={epochs} {rec['seconds']:.0f}s", flush=True)
    print(f"{R.now()} finished ({n} runs)", flush=True)


# ------------------------------------------------------------------ validation statistics
def load(base, setting, p):
    return json.load(open(path(base, setting, p), encoding="utf-8"))


def cmd_validate(_a):
    import numpy as np
    import pandas as pd
    if os.path.exists(DECISION):
        raise SystemExit("proxy decision exists; it is frozen")
    pilot_dir = R.RUNS
    rows, arch_out = [], {}
    for setting, pairs in validation_pairs().items():
        pilot = {tuple(p) for p in VPOL["cifar10_simplecnn"]["pilot_5epoch_policies"]} if setting == "cifar10_simplecnn" else set()
        for p in pairs:
            r5 = load(pilot_dir if p in pilot else D5, setting, p)
            r1 = load(D1, setting, p)
            assert r5["epochs"] == 5 and r1["epochs"] == 1 and r5["recovery"]["seed"] == r1["recovery"]["seed"] == 600
            rows.append({"setting": setting, "policy": str(list(p)), "source_5epoch": "pilot" if p in pilot else "validation",
                         "A1": r1["post"]["val_rl"], "A5": r5["post"]["val_rl"], "A_pre": r1["pre"]["val_rl"]})
    df = pd.DataFrame(rows)
    for setting in ARCH:
        d = df[df.setting == setting].copy()
        ident = d[d.policy == "[0, 0, 0, 0]"].iloc[0]
        d["q1"], d["q5"] = d.A1 / ident.A1, d.A5 / ident.A5
        df.loc[d.index, ["q1", "q5"]] = d[["q1", "q5"]].to_numpy()
        rho = float(d.A1.rank().corr(d.A5.rank()))
        r = float(np.corrcoef(d.A1, d.A5)[0, 1])
        top5, top1 = set(d.nlargest(10, "A5").policy), set(d.nlargest(10, "A1").policy)
        feas = d.q5 >= 0.98
        tau1, j = None, None
        if feas.sum() >= 3 and (~feas).sum() >= 3:
            best = (-2.0, -1.0)
            for t in sorted(d.q1.unique()):
                pred = d.q1 >= t
                jj = float(pred[feas].mean() + (~pred[~feas]).mean() - 1)
                if (jj, t) >= best:
                    best = (jj, t)
            j, tau1 = best
        conf = {}
        if tau1 is not None:
            pred = d.q1 >= tau1
            conf = {"tp": int((pred & feas).sum()), "fp": int((pred & ~feas).sum()), "fn": int((~pred & feas).sum()), "tn": int((~pred & ~feas).sum())}
        sub = {}
        for src in ("pilot", "validation"):
            s = d[d.source_5epoch == src]
            if len(s) >= 3:
                sub[src] = {"n": len(s), "spearman": float(s.A1.rank().corr(s.A5.rank()))}
        arch_out[setting] = {"n_pairs": len(d), "spearman_rho": rho, "pearson_r": r, "mae_q": float((d.q1 - d.q5).abs().mean()),
                             "bias_accuracy_A1_minus_A5": float((d.A1 - d.A5).mean()), "bias_q1_minus_q5": float((d.q1 - d.q5).mean()),
                             "share_A1_below_A5": float((d.A1 < d.A5).mean()), "top10_overlap": len(top5 & top1),
                             "q5_feasible_pairs": int(feas.sum()), "tau1": tau1, "youden_j": j, "confusion_at_tau1": conf,
                             "confusion_at_q1_0.98": {"tp": int(((d.q1 >= 0.98) & feas).sum()), "fp": int(((d.q1 >= 0.98) & ~feas).sum()),
                                                      "fn": int(((d.q1 < 0.98) & feas).sum()), "tn": int(((d.q1 < 0.98) & ~feas).sum())},
                             "spearman_by_source": sub, "A1_identity": float(ident.A1), "A5_identity": float(ident.A5),
                             "go": bool(rho >= GO_RHO)}
    df.to_csv(os.path.join(L.RESULTS, "paper3_proxy_validation.csv"), index=False, float_format="%.6g")
    dec = {"created": R.now(), "rule": f"GO per architecture iff Spearman rho(A1, A5) >= {GO_RHO}", "architectures": arch_out,
           "validation_policies_sha256": sha(os.path.join(L.RESULTS, "paper3_proxy_validation_policies.json")),
           "proxy_protocol_sha256": sha(PROTOCOL)}
    json.dump(dec, open(DECISION, "w", encoding="utf-8"), indent=1)
    f = lambda v, n=3: "—" if v is None else f"{v:.{n}f}"  # noqa: E731
    lines = ["# Paper 3 — 1-epoch proxy validation report", "",
             "Pre-registered in `paper3_proxy_protocol.md` (amendment 2). Validation pairs frozen before measurement "
             "(`paper3_proxy_validation_policies.json`). VAL-RL only; the test set is never read. "
             "A = VAL-RL accuracy after recovery; q = A / A(identity) at the same recovery length.", "",
             "| | SimpleCNN | ResNet-8 |", "|---|---|---|"]
    s, rr = arch_out["cifar10_simplecnn"], arch_out["cifar10_resnet8"]
    for label, k, n in (("Pairs", "n_pairs", 0), ("**Spearman ρ(A₁, A₅)** (primary)", "spearman_rho", 3), ("Pearson r", "pearson_r", 3),
                        ("MAE of q", "mae_q", 4), ("Bias A₁ − A₅ (pp)", "bias_accuracy_A1_minus_A5", 2), ("Bias q₁ − q₅", "bias_q1_minus_q5", 4),
                        ("Share of pairs with A₁ < A₅", "share_A1_below_A5", 3), ("Top-10 overlap (by A₅ vs A₁)", "top10_overlap", 0),
                        ("Pairs with q₅ ≥ 0.98", "q5_feasible_pairs", 0), ("τ₁ (Youden rule)", "tau1", 4), ("Youden J", "youden_j", 3)):
        lines.append(f"| {label} | {f(s[k], n) if isinstance(s[k], float) or s[k] is None else s[k]} | "
                     f"{f(rr[k], n) if isinstance(rr[k], float) or rr[k] is None else rr[k]} |")
    lines.append(f"| **Proxy decision (ρ ≥ {GO_RHO})** | **{'GO' if s['go'] else 'NO-GO'}** | **{'GO' if rr['go'] else 'NO-GO'}** |")
    for name, v in (("SimpleCNN", s), ("ResNet-8", rr)):
        lines += ["", f"**{name}.** Confusion of q₁ ≥ τ₁ against q₅ ≥ 0.98: {v['confusion_at_tau1'] or 'τ₁ not set (fallback: PI)'}. "
                      f"At q₁ ≥ 0.98: {v['confusion_at_q1_0.98']}. Spearman by 5-epoch source: {v['spearman_by_source']}. "
                      f"Identity: A₁ = {v['A1_identity']:.2f}, A₅ = {v['A5_identity']:.2f}."]
    open(os.path.join(L.RESULTS, "paper3_proxy_validation_report.md"), "w", encoding="utf-8").write("\n".join(lines) + "\n")
    print(json.dumps({k: {x: v[x] for x in ("spearman_rho", "pearson_r", "mae_q", "bias_accuracy_A1_minus_A5", "tau1", "go")} for k, v in arch_out.items()}, indent=1))


# ------------------------------------------------------------------ landscapes
def cmd_merge(_a):
    import pandas as pd
    dec = json.load(open(DECISION, encoding="utf-8"))
    lat = json.load(open(os.path.join(L.RESULTS, "h2_latency_freeze.json"), encoding="utf-8"))
    p5 = json.load(open(os.path.join(L.ROOT, "results", "phase5", "phase5_config.json"), encoding="utf-8"))
    amend = subprocess.run(["git", "log", "-1", "--format=%cI", "ec7a7f2"], cwd=L.ROOT, capture_output=True, text=True).stdout.strip()
    when = datetime.datetime.fromisoformat
    checks, frozen = [], {}

    def chk(name, ok, detail=""):
        checks.append({"check": name, "pass": bool(ok), "detail": str(detail)})
        print(f"[{'PASS' if ok else 'FAIL'}] {name}: {str(detail)[:150]}", flush=True)
    for setting, arch in ARCH.items():
        if not dec["architectures"][setting]["go"]:
            chk(f"{setting}: proxy NO-GO, no landscape (by rule)", True, dec["architectures"][setting]["spearman_rho"])
            continue
        pair = f"{arch}_c10"
        recs = [load(D1, setting, p) if os.path.exists(path(D1, setting, p)) else None for p in POLICIES]
        ok = [r for r in recs if r and r["status"] == "complete"]
        chk(f"{setting}: 1,296 complete 1-epoch records, one per policy", len(ok) == 1296 and len({tuple(r["policy"]) for r in ok}) == 1296, len(ok))
        if len(ok) != 1296:
            continue
        ref, ref_sha = L.reference(setting)
        lp = os.path.join(L.RESULTS, f"h2_latency_table_{pair}.csv")
        chk(f"{setting}: frozen H2-A latency table unchanged (sha256)", sha(lp) == lat["tables"][pair]["sha256"], lat["tables"][pair]["sha256"][:16])
        chk(f"{setting}: every record: exactly 1 epoch, seed 600, Phase-6 protocol, 1 thread, no test use, frozen reference, amendment-2 hash",
            all(r["epochs"] == 1 and r["recovery"]["epochs"] == 1 and r["recovery"]["seed"] == 600 and r["recovery"]["protocol"] == ok[0]["recovery"]["protocol"]
                and r["threads"] == 1 and r["test_use"] == "none" and r["reference_sha256"] == ref_sha
                and r["proxy_protocol_sha256"] == ok[0]["proxy_protocol_sha256"] for r in ok), ref_sha[:16])
        chk(f"{setting}: identical minibatch exposure (one data-stream hash)", len({r["data_stream_sha256"] for r in ok}) == 1, "")
        ident = next(r for r in ok if r["policy"] == [0, 0, 0, 0])
        chk(f"{setting}: identity before recovery = frozen reference (Phase-5 dense VAL-RL record)",
            ident["pre"]["val_rl"] == p5["settings"][setting]["dense_val_rl_accuracy"], ident["pre"]["val_rl"])
        t = pd.read_csv(lp)
        df = pd.DataFrame([{"policy": str(r["policy"]), "widths_rec": str(r["widths"]), "params_rec": r["params"], "macs_rec": r["macs"],
                            "pre_val_rl": r["pre"]["val_rl"], "A1_val_rl": r["post"]["val_rl"], "A1_val_select": r["post"]["val_select"],
                            "loss_epoch1": r["train_loss"][0], "final_state_sha256": r["final_state_sha256"], "seconds": r["seconds"]} for r in ok])
        m = t.merge(df, on="policy", validate="one_to_one")
        chk(f"{setting}: shapes unchanged in every run; widths, parameters and MACs equal the latency table",
            all(r["shape_unchanged"] for r in ok) and bool((m.widths == m.widths_rec).all() and (m.params == m.params_rec).all() and (m.macs == m.macs_rec).all()), "")
        m["q1"] = m.A1_val_rl / float(m[m.policy == "[0, 0, 0, 0]"].A1_val_rl.iloc[0])
        out = os.path.join(L.RESULTS, f"h2_c10_{arch}_1epoch_landscape.csv")
        m.drop(columns=["widths_rec", "params_rec", "macs_rec"]).to_csv(out, index=False, float_format="%.9g")
        first = min(when(r["started_at"]) for r in ok)
        chk(f"{setting}: amendment 2 (ec7a7f2) committed before the first 1-epoch run", when(amend) < first, f"{amend} < {first.isoformat()}")
        frozen[setting] = {"landscape": os.path.basename(out), "sha256": sha(out), "latency_table_sha256": lat["tables"][pair]["sha256"]}
    v = [json.load(open(os.path.join(D5, s, f), encoding="utf-8")) for s in ARCH for f in os.listdir(os.path.join(D5, s)) if f.endswith(".json")] \
        if os.path.isdir(D5) else []
    chk("5-epoch validation runs: every record 5 epochs, seed 600, shapes unchanged, no test use",
        bool(v) and all(r["epochs"] == 5 and r["recovery"]["seed"] == 600 and r["shape_unchanged"] and r["test_use"] == "none" for r in v), len(v))
    chk("pilot H2-C records unchanged since the stop commit (git diff empty)",
        subprocess.run(["git", "diff", "--quiet", "5f19f11", "--", "results/paper3/_h2_recovery_runs"], cwd=L.ROOT).returncode == 0, "")
    out = {"created": R.now(), "proxy_decision": dec, "landscapes": frozen, "checks": checks}
    json.dump(out, open(os.path.join(L.RESULTS, "h2_e2_freeze.json"), "w", encoding="utf-8"), indent=1, default=str)
    n = sum(c["pass"] for c in checks)
    lines = ["# Paper 3 — E2 integrity checks", "", f"**{n}/{len(checks)} passed.**", "", "| Status | Check | Detail |", "|---|---|---|"]
    lines += [f"| {'PASS' if c['pass'] else 'FAIL'} | {c['check']} | {c['detail'].replace('|', '/')[:200]} |" for c in checks]
    open(os.path.join(L.RESULTS, "h2_e2_integrity_checks.md"), "w", encoding="utf-8").write("\n".join(lines) + "\n")
    print(f"E2 integrity {n}/{len(checks)}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    w = sub.add_parser("worker")
    w.add_argument("--stage", choices=["validation", "landscape"], required=True)
    w.add_argument("--cpu", type=int, required=True)
    sub.add_parser("validate")
    sub.add_parser("merge")
    a = ap.parse_args()
    sys.path.insert(0, L.ROOT)
    {"worker": cmd_worker, "validate": cmd_validate, "merge": cmd_merge}[a.cmd](a)
