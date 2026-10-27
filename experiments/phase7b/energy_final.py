"""Phase 7B, Stage B: hardware validation of the recovered final models (design section 11, Phase-7A protocol).

  python experiments/phase7b/energy_final.py pilot [--n 100000]     # dense-only stability pilot (CV <= 3%)
  python experiments/phase7b/energy_final.py sessions --n 100000    # 5 fresh-process sessions
Child: pinned to logical CPU 2, one intra-op and one inter-op thread, set BEFORE importing torch. Every model is
built and its recovered weights loaded and hash-checked before the idle reading and the first block. Block = 1,000
warm-up inferences, then N contiguous batch-1 inferences between two RAPL package counter reads. Each session runs
the non-dense models once and Dense three times, all in one random order (seeded per session).
"""
import argparse
import datetime
import json
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import p7b_lib as L  # noqa: E402

OUT = os.path.join(L.RESULTS, "_energy_final")
WARMUP, IDLE_S, PAUSE_S, SESSIONS, SESSION_SEED, DENSE_BLOCKS = 1000, 30, 5, 5, 20261031, 3
MAX_CPU_LOAD, PILOT_MAX_CV = 15.0, 0.03
DENSE, OLD = "0000", "phase7a_ppo"


def now():
    return datetime.datetime.now().astimezone().isoformat(timespec="seconds")


def model_ids():
    jobs = json.load(open(os.path.join(L.RESULTS, "final_jobs.json"), encoding="utf-8"))
    return [j["id"] for j in jobs] + [OLD]


def load_model(mid, ref, P7):
    """Recovered, physically compacted model; the loaded state must hash to the recorded recovery output."""
    import torch
    from src.pruning import structured as SP
    from src.utils import sha256_state_dict
    if mid == OLD:
        sel = json.load(open(os.path.join(L.RESULTS, "selected_policies.json"), encoding="utf-8"))["methods"]["old_ppo"]
        assert L.sha256_file(os.path.join(L.ROOT, sel["checkpoint"])) == sel["checkpoint_sha256"]
        return P7.load_recovered(L.ARCH, "ppo")
    rec = json.load(open(os.path.join(L.RESULTS, "_recovery_runs", "final", f"{mid}_E{L.FINAL_EPOCHS}.json"), encoding="utf-8"))
    m, _ = SP.simplecnn_structured(ref, rec["ratios"])
    m.load_state_dict(torch.load(os.path.join(L.ROOT, rec["checkpoint"]), map_location="cpu", weights_only=True))
    assert sha256_state_dict(m.state_dict()) == rec["final_state_sha256"], f"{mid}: checkpoint differs from recovery"
    return m.eval()


def child(a):
    import psutil
    os.environ["OMP_NUM_THREADS"] = os.environ["MKL_NUM_THREADS"] = "1"
    psutil.Process().cpu_affinity([L.CPU])
    import gc
    import numpy as np
    import torch
    torch.set_num_interop_threads(1)
    torch.set_num_threads(1)
    P7 = L.energy_meter()
    meter = P7.EnergyMeter()
    ref, ref_sha = L.p3().reference(L.SETTING)
    ids = a.models.split(",")
    models = {m: load_model(m, ref, P7) for m in ids}
    order = [m for m in ids if m != DENSE] + [DENSE] * a.dense_blocks
    if a.seed is not None:
        order = [order[i] for i in np.random.default_rng(a.seed).permutation(len(order))]
    x = torch.randn((1, 3, 32, 32), generator=torch.Generator().manual_seed(0)).contiguous()
    lat = np.empty(a.n, dtype=np.int64)
    out = {"session": a.session, "order_seed": a.seed, "order": order, "n": a.n, "warmup": WARMUP,
           "reference_sha256": ref_sha, "started_at": now(), "blocks": []}
    time.sleep(2)
    e0, t0 = meter.read(), time.perf_counter()
    time.sleep(IDLE_S)
    e1, t1 = meter.read(), time.perf_counter()
    out["idle"] = {"seconds": t1 - t0, "pkg_j": P7.EnergyMeter.joules(e0, e1), "pkg_w": P7.EnergyMeter.joules(e0, e1) / (t1 - t0)}
    seen = {}
    with torch.inference_mode():
        for pos, name in enumerate(order):
            m = models[name]
            for _ in range(WARMUP):
                m(x)
            gc.collect()
            gc.disable()
            ea, wa = meter.read(), time.perf_counter_ns()
            for i in range(a.n):
                s = time.perf_counter_ns()
                m(x)
                lat[i] = time.perf_counter_ns() - s
            wb, eb = time.perf_counter_ns(), meter.read()
            gc.enable()
            dur = (wb - wa) * 1e-9
            pkg = P7.EnergyMeter.joules(ea, eb)
            seen[name] = seen.get(name, -1) + 1
            out["blocks"].append({
                "model": name, "appearance": seen[name], "position": pos, "at": now(), "seconds": dur, "pkg_j": pkg,
                "pp0_j": P7.EnergyMeter.joules(ea, eb, "RAPL_Package0_PP0"),
                "pkg_counter_ms": eb["RAPL_Package0_PKG_Time"] - ea["RAPL_Package0_PKG_Time"],
                "e1000_j": pkg / a.n * 1000, "pkg_w": pkg / dur,
                "net_e1000_j": (pkg - out["idle"]["pkg_w"] * dur) / a.n * 1000,
                "latency_median_ms": float(np.median(lat)) * 1e-6, "latency_mean_ms": float(lat.mean()) * 1e-6,
                "latency_p95_ms": float(np.percentile(lat, 95)) * 1e-6,
                "latency_iqr_ms": float(np.subtract(*np.percentile(lat, [75, 25]))) * 1e-6})
            print(f"  {name:14s} {out['blocks'][-1]['e1000_j']:.3f} J/1000 {out['blocks'][-1]['latency_median_ms']:.4f} ms",
                  flush=True)
            time.sleep(PAUSE_S)
    out["finished_at"] = now()
    out["effective"] = {"threads": torch.get_num_threads(), "interop": torch.get_num_interop_threads(),
                        "cpu_affinity": psutil.Process().cpu_affinity(), "torch": torch.__version__}
    json.dump(out, open(a.out, "w", encoding="utf-8"), indent=1)


def gate(log):
    import psutil
    P7 = L.energy_meter()
    ac, load, scheme = P7.on_ac_power(), psutil.cpu_percent(interval=5.0), P7.power_scheme()
    log.append({"at": now(), "ac_power": ac, "cpu_load_pct": load, "power_scheme": scheme})
    if not ac:
        raise SystemExit("STOP: not on AC power")
    if load > MAX_CPU_LOAD:
        raise SystemExit(f"STOP: CPU load {load:.1f}% > {MAX_CPU_LOAD}%")
    return scheme


def spawn(session, seed, n, models, dense_blocks, out):
    cmd = [sys.executable, os.path.abspath(__file__), "child", "--session", str(session), "--n", str(n),
           "--models", ",".join(models), "--dense-blocks", str(dense_blocks), "--out", out]
    if seed is not None:
        cmd += ["--seed", str(seed)]
    subprocess.run(cmd, check=True)
    return json.load(open(out, encoding="utf-8"))


def cmd_pilot(a):
    import numpy as np
    os.makedirs(OUT, exist_ok=True)
    log, vals = [], []
    for rep in range(2):
        gate(log)
        r = spawn(f"pilot{rep}", None, a.n, [DENSE], 3, os.path.join(OUT, f"pilot_n{a.n}_rep{rep}.json"))
        vals += [b["e1000_j"] for b in r["blocks"]]
    cv = float(np.std(vals, ddof=1) / np.mean(vals))
    res = {"created": now(), "n": a.n, "e1000_j": vals, "cv": cv, "rule_max_cv": PILOT_MAX_CV,
           "decision": "stable" if cv <= PILOT_MAX_CV else "unstable", "gates": log}
    json.dump(res, open(os.path.join(OUT, f"pilot_n{a.n}.json"), "w", encoding="utf-8"), indent=1)
    print(f"dense E_1000 {np.mean(vals):.3f} J, CV {100 * cv:.2f}% -> {res['decision']}")


def cmd_sessions(a):
    import numpy as np
    os.makedirs(OUT, exist_ok=True)
    ids = model_ids()
    seeds = [int(s) for s in np.random.default_rng(SESSION_SEED).integers(0, 2 ** 31 - 1, size=SESSIONS)]
    log, schemes = [], set()
    for s in range(1, SESSIONS + 1):
        out = os.path.join(OUT, f"session{s}.json")
        if os.path.exists(out):
            continue
        schemes.add(gate(log))
        print(f"{now()} session {s}", flush=True)
        spawn(s, seeds[s - 1], a.n, ids, DENSE_BLOCKS, out + ".tmp.json")
        os.replace(out + ".tmp.json", out)
        log.append({"at": now(), "finished": out})
    json.dump({"gates": log, "power_schemes": sorted(schemes)},
              open(os.path.join(OUT, f"gates_{datetime.datetime.now():%Y%m%d_%H%M%S}.json"), "w", encoding="utf-8"), indent=1)
    if len(schemes) > 1:
        print("WARNING: the power scheme changed between sessions:", schemes)


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    pp = sub.add_parser("pilot")
    pp.add_argument("--n", type=int, default=100000)
    ps = sub.add_parser("sessions")
    ps.add_argument("--n", type=int, required=True)
    c = sub.add_parser("child")
    for k, t in (("--session", str), ("--n", int), ("--models", str), ("--dense-blocks", int), ("--out", str)):
        c.add_argument(k, type=t, required=True)
    c.add_argument("--seed", type=int, default=None)
    a = p.parse_args()
    {"pilot": cmd_pilot, "sessions": cmd_sessions, "child": child}[a.cmd](a)


if __name__ == "__main__":
    main()
