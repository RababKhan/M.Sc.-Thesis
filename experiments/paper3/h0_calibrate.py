"""Paper 3, H0: CPU latency calibration on CIFAR-10 SimpleCNN (protocol: results/paper3/h0_protocol.md).

  python experiments/paper3/h0_calibrate.py            # all sessions, merge, statistics, decision, report
A session process is started by the parent as
  python experiments/paper3/h0_calibrate.py session --session S --config K --seed X --out PART
and pins its CPU affinity and thread counts before torch is imported. No accuracy is computed; no data are read.
"""
import argparse
import json
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import p3_lib as L  # noqa: E402  (imports neither torch nor numpy)

PARTS = os.path.join(L.RESULTS, "_h0_parts")


# ------------------------------------------------------------------ child: one (session, configuration)
def session(a):
    cfg = L.CONFIGS[a.config]
    import psutil
    for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[var] = str(cfg["threads"])
    psutil.Process().cpu_affinity(cfg["cpus"])               # before numpy / torch create any thread
    import torch
    torch.set_num_interop_threads(cfg["interop"])
    torch.set_num_threads(cfg["threads"])
    from src.evaluation import latency as T
    from src.pruning import structured_units as SU
    ref, sha = L.reference("cifar10_simplecnn")
    models = {name: SU.slim(ref, "simplecnn", acts)[0] for name, acts in L.CALIBRATION.items()}
    x = torch.randn((cfg["batch"], 3, 32, 32), generator=torch.Generator().manual_seed(0)).contiguous()
    t0 = time.time()
    rows = T.run_session(models, x, cfg["warmup"], cfg["timed"], cfg["rounds"], a.seed)
    import pandas as pd
    df = pd.DataFrame(rows, columns=["round", "position", "model", "role", "pass", "ns"])
    df.insert(0, "config", a.config)
    df.insert(0, "session", a.session)
    df.to_csv(a.out + ".csv", index=False)
    meta = {"session": a.session, "config": a.config, "order_seed": a.seed, "batch": cfg["batch"],
            "requested": {"threads": cfg["threads"], "interop": cfg["interop"], "cpus": cfg["cpus"]},
            "effective": {"threads": torch.get_num_threads(), "interop": torch.get_num_interop_threads(),
                          "cpus": psutil.Process().cpu_affinity(), "mkldnn_enabled": torch.backends.mkldnn.enabled,
                          "omp_num_threads": os.environ["OMP_NUM_THREADS"]},
            "reference_sha256": sha, "input_shape": list(x.shape), "dtype": str(x.dtype),
            "timing_seconds": round(time.time() - t0, 2), "pid": os.getpid()}
    json.dump(meta, open(a.out + ".json", "w", encoding="utf-8"), indent=1)


# ------------------------------------------------------------------ parent
def cpu_load():
    import psutil
    return psutil.cpu_percent(interval=5.0)


def other_experiments():
    import psutil
    me = os.getpid()
    out = []
    for p in psutil.process_iter(["pid", "name", "cmdline"]):
        try:
            if p.info["pid"] != me and "python" in (p.info["name"] or "").lower() and \
                    any(k in " ".join(p.info["cmdline"] or []) for k in ("run.py", "landscape", "train", "finetune", "h0_calibrate")):
                out.append(" ".join(p.info["cmdline"]))
        except Exception:                                    # noqa: BLE001
            pass
    return out


def model_table():
    import torch
    from src import models as M
    from src.evaluation import efficiency as E
    from src.pruning import structured_units as SU
    torch.set_num_threads(4)
    ref, _ = L.reference("cifar10_simplecnn")
    rows = []
    for name, acts in L.CALIBRATION.items():
        m, _ = SU.slim(ref, "simplecnn", acts)
        macs, _ = E.conv_linear_macs(m)
        pt, _ = E.ptflops_macs(m)
        rows.append({"model": name, "actions": str(acts), "widths": str(SU.widths("simplecnn", acts)),
                     "params": M.count_parameters(m)["total"], "macs": macs, "flops": E.torch_flops(m), "ptflops_macs": pt})
    d = rows[0]
    for r in rows:
        r["param_reduction"] = 1 - r["params"] / d["params"]
        r["mac_reduction"] = 1 - r["macs"] / d["macs"]
    return rows


def run_sessions(log):
    import numpy as np
    os.makedirs(PARTS, exist_ok=True)
    rng = np.random.default_rng(L.SESSION_SEED)
    plan = []
    for s in range(1, L.SESSIONS + 1):
        for k in [list(L.CONFIGS)[i] for i in rng.permutation(len(L.CONFIGS))]:
            plan.append((s, k, int(rng.integers(2 ** 31 - 1))))
    for s, k, seed in plan:
        out = os.path.join(PARTS, f"s{s}_{k}")
        if os.path.exists(out + ".json"):
            continue
        for attempt in range(1, 4):
            before = cpu_load()
            for _ in range(6):
                if before <= L.MAX_CPU_LOAD:
                    break
                time.sleep(20)
                before = cpu_load()
            subprocess.run([sys.executable, os.path.abspath(__file__), "session", "--session", str(s), "--config", k,
                            "--seed", str(seed), "--out", out], check=True)
            after = cpu_load()
            ok = before <= L.MAX_CPU_LOAD and after <= L.MAX_CPU_LOAD
            log.append({"session": s, "config": k, "attempt": attempt, "load_before": before, "load_after": after, "ok": ok})
            print(f"session {s} config {k} attempt {attempt}: load {before:.1f}% / {after:.1f}% {'ok' if ok else 'REPEAT'}", flush=True)
            if ok or attempt == 3:
                break
            os.remove(out + ".json")
    return plan


def analyse(models, log, plan, env):
    import pandas as pd
    from src.evaluation import latency as T
    raw = pd.concat([pd.read_csv(os.path.join(PARTS, f"s{s}_{k}.csv")) for s, k, _ in plan], ignore_index=True)
    metas = [json.load(open(os.path.join(PARTS, f"s{s}_{k}.json"), encoding="utf-8")) for s, k, _ in plan]
    raw.to_csv(os.path.join(L.RESULTS, "h0_latency_raw.csv"), index=False)
    mt = pd.DataFrame(models).set_index("model")
    timed = raw[raw.role == "model"]
    per_session = timed.groupby(["config", "model", "session"]).ns.median().div(1e6).rename("median_ms").reset_index()
    rows, decision = [], {}
    for k, cfg in L.CONFIGS.items():
        ps = per_session[per_session.config == k].pivot(index="session", columns="model", values="median_ms")
        for m in L.CALIBRATION:
            d = T.describe(timed[(timed.config == k) & (timed.model == m)].ns, cfg["batch"])
            sess = ps[m]
            rows.append({"config": k, "batch": cfg["batch"], "threads": cfg["threads"], "model": m, "actions": mt.loc[m, "actions"],
                         "params": int(mt.loc[m, "params"]), "param_reduction": mt.loc[m, "param_reduction"],
                         "macs": int(mt.loc[m, "macs"]), "mac_reduction": mt.loc[m, "mac_reduction"], **d,
                         "session_median_ms_median": float(sess.median()), "session_median_ms_min": float(sess.min()),
                         "session_median_ms_max": float(sess.max()), "between_session_range_ms": float(sess.max() - sess.min()),
                         "between_session_cv": float(sess.std(ddof=1) / sess.mean()),
                         "speedup_vs_dense": float((ps["dense"] / sess).median()),
                         "latency_reduction_vs_dense": float(1 - (sess / ps["dense"]).median())})
        delta = ps["dense"] - ps[L.SIGNAL_MODEL]
        signal = float(delta.median())
        noise = float(max(ps["dense"].max() - ps["dense"].min(), ps[L.SIGNAL_MODEL].max() - ps[L.SIGNAL_MODEL].min()))
        noise = max(noise, 1e-6)
        rel = signal / float(ps["dense"].median())
        snr = abs(signal) / noise
        usable = bool(snr >= L.MIN_SNR and rel >= L.MIN_RELATIVE_REDUCTION and (delta > 0).all())
        decision[k] = {"signal_ms": signal, "noise_ms": noise, "snr": snr, "relative_reduction": rel,
                       "delta_positive_all_sessions": bool((delta > 0).all()), "per_session_delta_ms": delta.round(6).tolist(),
                       "dense_session_medians_ms": ps["dense"].round(6).tolist(),
                       "aggressive_session_medians_ms": ps[L.SIGNAL_MODEL].round(6).tolist(), "usable": usable}
    summary = pd.DataFrame(rows)
    summary.to_csv(os.path.join(L.RESULTS, "h0_latency_summary.csv"), index=False, float_format="%.6g")
    # drift of the dense anchor blocks within each session
    anc = raw[raw.role == "anchor"].groupby(["config", "session", "round"]).ns.median()
    drift = anc.groupby(["config", "session"]).apply(lambda v: float((v / v.median() - 1).abs().max())).reset_index(name="max_anchor_drift")
    pruned = [m for m in L.CALIBRATION if m != "dense"]
    params_down = all(mt.loc[m, "params"] < mt.loc["dense", "params"] for m in pruned)
    macs_down = all(mt.loc[m, "macs"] < mt.loc["dense", "macs"] for m in pruned)
    primary = next((k for k in L.SELECTION_ORDER if decision[k]["usable"]), None)
    go = bool(params_down and macs_down and primary is not None)
    checks = [
        ("every pruned calibration model has fewer physical parameters than dense", params_down, ""),
        ("every pruned calibration model has fewer MACs than dense", macs_down, ""),
        ("FlopCounter FLOPs = 2 x analytic MACs for every calibration model", bool((mt.flops == 2 * mt.macs).all()), ""),
        ("every session process ran with the requested threads, inter-op threads and CPU affinity, oneDNN enabled",
         all(x["effective"]["threads"] == x["requested"]["threads"] and x["effective"]["interop"] == x["requested"]["interop"]
             and sorted(x["effective"]["cpus"]) == sorted(x["requested"]["cpus"]) and x["effective"]["mkldnn_enabled"] for x in metas),
         f"{len(metas)} session processes"),
        ("every session used the frozen reference (sha256) and the same input shape",
         len({x["reference_sha256"] for x in metas}) == 1 and all(x["input_shape"][0] == L.CONFIGS[x["config"]]["batch"] for x in metas), ""),
        ("CPU load <= 15% before and after every kept session process", all(r["ok"] for r in log if r["attempt"] == max(
            q["attempt"] for q in log if q["session"] == r["session"] and q["config"] == r["config"])), f"{sum(not r['ok'] for r in log)} repeats"),
        ("repeated dense timing stable: between-session CV of the dense median <= 5% in every configuration",
         all(summary[(summary.config == k) & (summary.model == "dense")].between_session_cv.iloc[0] <= 0.05 for k in L.CONFIGS),
         {k: round(float(summary[(summary.config == k) & (summary.model == "dense")].between_session_cv.iloc[0]), 4) for k in L.CONFIGS}),
        ("dense anchor drift within every session <= 10%", bool((drift.max_anchor_drift <= 0.10).all()),
         {k: round(float(drift[drift.config == k].max_anchor_drift.max()), 4) for k in L.CONFIGS}),
        ("latency signal larger than noise (SNR >= 5) in the selected configuration", primary is not None and decision[primary]["snr"] >= L.MIN_SNR,
         {k: round(v["snr"], 2) for k, v in decision.items()}),
    ]
    out = {"selection_order": L.SELECTION_ORDER, "rule": {"min_snr": L.MIN_SNR, "min_relative_reduction": L.MIN_RELATIVE_REDUCTION,
                                                         "signal_model": L.SIGNAL_MODEL}, "configs": decision,
           "params_reduced": params_down, "macs_reduced": macs_down, "primary_config": primary, "h0": "GO" if go else "NO-GO",
           "checks": [{"check": c, "pass": bool(ok), "detail": str(d)} for c, ok, d in checks], "load_log": log,
           "anchor_drift": drift.to_dict("records")}
    json.dump(out, open(os.path.join(L.RESULTS, "h0_decision.json"), "w", encoding="utf-8"), indent=1, default=str)
    report(summary, mt, decision, primary, go, checks, drift, env, metas)
    print(f"H0 {out['h0']}; primary configuration {primary}; SNR {[round(v['snr'], 1) for v in decision.values()]}")


def report(summary, mt, decision, primary, go, checks, drift, env, metas):
    f = lambda v, n=3: f"{v:.{n}f}"  # noqa: E731
    lines = ["# Paper 3 — H0 CPU latency calibration report", "",
             f"Protocol: `h0_protocol.md` (committed before this measurement). Machine: {env['cpu']}, {env['os']}, "
             f"Python {env['python']}, torch {env['torch']}; oneDNN available {env['mkldnn_available']}, MKL {env['mkl_available']}; "
             f"power plan: {env['power_plan']}; battery/AC: {env['battery']}.", "",
             f"**H0 decision: {'GO' if go else 'NO-GO'}.** Primary latency configuration: **{primary or 'none'}**.", "",
             "## Calibration models (CIFAR-10 SimpleCNN, physically compacted)", "",
             "| Model | Actions | Widths (conv1, conv2, conv3, fc1) | Parameters | Param. reduction | MACs | MAC reduction |", "|---|---|---|---|---|---|---|"]
    for m, r in mt.iterrows():
        lines.append(f"| {m} | {r.actions} | {r.widths} | {r.params:,} | {100 * r.param_reduction:.1f}% | {r.macs:,} | {100 * r.mac_reduction:.1f}% |")
    for k, cfg in L.CONFIGS.items():
        d = decision[k]
        lines += ["", f"## Configuration {k}: batch {cfg['batch']}, {cfg['threads']} intra-op thread(s), 1 inter-op, CPUs {cfg['cpus']}", "",
                  "| Model | Median (ms) | p95 (ms) | IQR (ms) | Throughput (img/s) | Speed-up vs dense | Session medians min–max (ms) | Between-session CV |",
                  "|---|---|---|---|---|---|---|---|"]
        for _, r in summary[summary.config == k].iterrows():
            lines.append(f"| {r.model} | {f(r.median_ms, 4)} | {f(r.p95_ms, 4)} | {f(r.iqr_ms, 4)} | {r.throughput_img_s:,.0f} | "
                         f"{f(r.speedup_vs_dense)} | {f(r.session_median_ms_min, 4)}–{f(r.session_median_ms_max, 4)} | {100 * r.between_session_cv:.2f}% |")
        lines += ["", f"Signal (median over sessions of dense − aggressive): **{f(d['signal_ms'], 4)} ms** "
                      f"({100 * d['relative_reduction']:.1f}% of dense); noise (largest between-session range of the two): "
                      f"{f(d['noise_ms'], 4)} ms; **SNR = {d['snr']:.1f}**; dense − aggressive positive in every session: "
                      f"{d['delta_positive_all_sessions']}; usable: **{d['usable']}**."]
    lines += ["", "## Decision (rule of `h0_protocol.md`, applied mechanically)", "",
              f"Order A → C → B; usable = SNR ≥ {L.MIN_SNR:g}, relative reduction ≥ {100 * L.MIN_RELATIVE_REDUCTION:.0f}% and positive in every "
              f"session. Usable: {', '.join(k for k in L.SELECTION_ORDER if decision[k]['usable']) or 'none'}. "
              f"Selected: **{primary or 'none'}**. Parameters reduced: {all(mt.params.iloc[1:] < mt.params.iloc[0])}; "
              f"MACs reduced: {all(mt.macs.iloc[1:] < mt.macs.iloc[0])}. **H0 {'GO' if go else 'NO-GO'}.**", "",
              "## Hardware checks", "", "| Status | Check | Detail |", "|---|---|---|"]
    lines += [f"| {'PASS' if ok else 'FAIL'} | {c} | {str(d).replace('|', '/')} |" for c, ok, d in checks]
    lines += ["", f"Sessions: {L.SESSIONS} per configuration, each in a fresh process ({len(metas)} processes); orders drawn from seed "
                  f"{L.SESSION_SEED}. Raw passes: `h0_latency_raw.csv`; summary: `h0_latency_summary.csv`; decision: `h0_decision.json`; "
                  "environment: `h0_environment.json`. All sessions ran back-to-back on one day, so the between-session spread "
                  "does not include day-to-day variation; the final Paper-3 confirmation session is on a different occasion."]
    open(os.path.join(L.RESULTS, "h0_calibration_report.md"), "w", encoding="utf-8").write("\n".join(lines) + "\n")


def main():
    from src.evaluation import latency as T
    others = other_experiments()
    if others:
        raise SystemExit(f"another experiment process is running: {others}")
    env = T.machine_info()
    if env["battery"] is not None and not env["battery"]["power_plugged"]:
        raise SystemExit("on battery power: H0 requires AC power")
    os.makedirs(L.RESULTS, exist_ok=True)
    env.update({"protocol": "results/paper3/h0_protocol.md", "configs": L.CONFIGS, "sessions": L.SESSIONS, "session_seed": L.SESSION_SEED,
                "calibration": L.CALIBRATION, "load_before_start_pct": cpu_load(), "other_experiment_processes": others,
                "started": time.strftime("%Y-%m-%dT%H:%M:%S%z")})
    from src.utils import git_state
    env["git"] = git_state()
    json.dump(env, open(os.path.join(L.RESULTS, "h0_environment.json"), "w", encoding="utf-8"), indent=1)
    models = model_table()
    log = []
    plan = run_sessions(log)
    analyse(models, log, plan, env)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd")
    s = sub.add_parser("session")
    s.add_argument("--session", type=int, required=True)
    s.add_argument("--config", required=True)
    s.add_argument("--seed", type=int, required=True)
    s.add_argument("--out", required=True)
    a = ap.parse_args()
    if a.cmd == "session":
        sys.path.insert(0, L.ROOT)
        session(a)
    else:
        main()
