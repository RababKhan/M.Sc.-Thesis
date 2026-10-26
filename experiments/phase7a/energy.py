"""Phase 7A, steps 8-10: energy backend check, stability pilot, five energy sessions (pre-registration "energy").

  python experiments/phase7a/energy.py backend           # step 8: verify the RAPL counter and log the environment
  python experiments/phase7a/energy.py pilot [--n N]     # step 9: dense-only stability pilot
  python experiments/phase7a/energy.py sessions --n N    # step 10: 5 sessions x 2 architectures
A session child (started by the parent, one fresh process per architecture and session):
  python experiments/phase7a/energy.py child --arch A --session S --seed X --n N --models m1,m2,.. --blocks B --out P
The child pins itself to logical CPU 2 and sets 1 thread BEFORE importing torch; models are built and loaded before
any measurement; each measured block is N contiguous batch-1 inferences bracketed by two counter reads.
"""
import argparse
import datetime
import json
import os
import platform
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import p7_lib as L  # noqa: E402  (imports neither torch nor numpy)

PARTS = os.path.join(L.RESULTS, "energy_parts")
PILOT = os.path.join(L.RESULTS, "energy_pilot")
WARMUP, IDLE_S, PAUSE_S, SESSIONS, SESSION_SEED = 1000, 30, 5, 5, 20261026
MAX_CPU_LOAD, PILOT_MAX_CV = 15.0, 0.03
PY = sys.executable


def now():
    return datetime.datetime.now().astimezone().isoformat(timespec="seconds")


# ------------------------------------------------------------------ child
def cmd_child(a):
    import psutil
    os.environ["OMP_NUM_THREADS"] = os.environ["MKL_NUM_THREADS"] = str(L.THREADS)
    psutil.Process().cpu_affinity([L.CPU])
    import gc
    import numpy as np
    import torch
    torch.set_num_interop_threads(1)
    torch.set_num_threads(L.THREADS)
    meter = L.EnergyMeter()
    models = {m: L.load_recovered(a.arch, m) for m in a.models.split(",")}
    x = torch.randn((1, 3, 32, 32), generator=torch.Generator().manual_seed(0)).contiguous()
    order = list(models)
    if a.seed is not None:
        order = [order[i] for i in np.random.default_rng(a.seed).permutation(len(order))]
    lat = np.empty(a.n, dtype=np.int64)
    out = {"arch": a.arch, "session": a.session, "order_seed": a.seed, "order": order, "n": a.n, "blocks_per_model": a.blocks,
           "warmup": WARMUP, "started_at": now(), "blocks": []}
    time.sleep(2)
    e0, t0 = meter.read(), time.perf_counter()
    time.sleep(IDLE_S)
    e1, t1 = meter.read(), time.perf_counter()
    out["idle"] = {"seconds": t1 - t0, "pkg_j": L.EnergyMeter.joules(e0, e1), "pp0_j": L.EnergyMeter.joules(e0, e1, "RAPL_Package0_PP0"),
                   "pkg_w": L.EnergyMeter.joules(e0, e1) / (t1 - t0)}
    with torch.inference_mode():
        for name in order:
            m = models[name]
            for _ in range(WARMUP):
                m(x)
            for blk in range(a.blocks):
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
                pkg = L.EnergyMeter.joules(ea, eb)
                out["blocks"].append({
                    "model": name, "block": blk, "at": now(), "seconds": dur,
                    "pkg_j": pkg, "pp0_j": L.EnergyMeter.joules(ea, eb, "RAPL_Package0_PP0"),
                    "pkg_counter_ms": eb["RAPL_Package0_PKG_Time"] - ea["RAPL_Package0_PKG_Time"],
                    "e1000_j": pkg / a.n * 1000, "pkg_w": pkg / dur,
                    "net_e1000_j": (pkg - out["idle"]["pkg_w"] * dur) / a.n * 1000,
                    "latency_median_ms": float(np.median(lat)) * 1e-6, "latency_mean_ms": float(lat.mean()) * 1e-6,
                    "latency_p95_ms": float(np.percentile(lat, 95)) * 1e-6,
                    "latency_iqr_ms": float(np.subtract(*np.percentile(lat, [75, 25]))) * 1e-6})
                print(f"  {name:8s} block {blk}: {out['blocks'][-1]['e1000_j']:.3f} J/1000 "
                      f"{out['blocks'][-1]['latency_median_ms']:.4f} ms", flush=True)
            time.sleep(PAUSE_S)
    out["finished_at"] = now()
    out["effective"] = {"threads": torch.get_num_threads(), "interop": torch.get_num_interop_threads(),
                        "cpu_affinity": psutil.Process().cpu_affinity(), "torch": torch.__version__,
                        "mkldnn_enabled": torch.backends.mkldnn.enabled}
    json.dump(out, open(a.out, "w", encoding="utf-8"), indent=1)


# ------------------------------------------------------------------ parent helpers
def cpu_load():
    import psutil
    return psutil.cpu_percent(interval=5.0)


def gate(log):
    ac = L.on_ac_power()
    load = cpu_load()
    scheme = L.power_scheme()
    log.append({"at": now(), "ac_power": ac, "cpu_load_pct": load, "power_scheme": scheme})
    if not ac:
        raise SystemExit("STOP: the laptop is not on AC power; energy is measured on AC only (pre-registration).")
    if load > MAX_CPU_LOAD:
        raise SystemExit(f"STOP: system CPU load {load:.1f}% > {MAX_CPU_LOAD}% (another workload is running).")
    return scheme


def child(arch, session, seed, n, models, blocks, out):
    cmd = [PY, os.path.abspath(__file__), "child", "--arch", arch, "--session", str(session), "--n", str(n),
           "--models", ",".join(models), "--blocks", str(blocks), "--out", out]
    if seed is not None:
        cmd += ["--seed", str(seed)]
    subprocess.run(cmd, check=True, env=os.environ.copy())
    return json.load(open(out, encoding="utf-8"))


def environment():
    import psutil
    import torch
    cfg = torch.__config__.show()
    onednn = next((ln.strip() for ln in cfg.splitlines() if "oneDNN" in ln or "MKLDNN" in ln.upper()), "unknown")
    cpu = subprocess.run(["powershell", "-NoProfile", "-Command", "(Get-CimInstance Win32_Processor).Name"],
                         capture_output=True, text=True).stdout.strip()
    return {"created": now(), "cpu": cpu, "logical_cpus": psutil.cpu_count(), "os": platform.platform(),
            "python": sys.version.split()[0], "torch": torch.__version__, "onednn": onednn,
            "threads": {"intra_op": L.THREADS, "inter_op": 1}, "cpu_affinity": [L.CPU],
            "power_scheme": L.power_scheme(), "ac_power": L.on_ac_power(),
            "energy_backend": {"interface": "Windows Energy Meter Interface (EMI) performance counters via PDH",
                               "counter": "\\Energy Meter(RAPL_Package0_PKG)\\Energy (primary), RAPL_Package0_PP0 (secondary)",
                               "source": "Intel RAPL hardware energy counters", "units": "picowatt-hours",
                               "dram_domain": "present but reads 0 on this machine (not used)",
                               "estimation": "none (no TDP model, no software estimate)"}}


# ------------------------------------------------------------------ step 8
def cmd_backend(_a):
    meter = L.EnergyMeter()
    a = meter.read()
    t0 = time.perf_counter()
    time.sleep(5)
    b = meter.read()
    dt = time.perf_counter() - t0
    env = environment()
    env["backend_check"] = {"pkg_idle_w": L.EnergyMeter.joules(a, b) / dt,
                            "pp0_idle_w": L.EnergyMeter.joules(a, b, "RAPL_Package0_PP0") / dt,
                            "counter_ms": b["RAPL_Package0_PKG_Time"] - a["RAPL_Package0_PKG_Time"], "wall_s": dt}
    json.dump(env, open(os.path.join(L.RESULTS, "environment.json"), "w", encoding="utf-8"), indent=1)
    print(json.dumps(env, indent=1))


# ------------------------------------------------------------------ step 9
def cmd_pilot(a):
    import numpy as np
    os.makedirs(PILOT, exist_ok=True)
    log, res = [], {}
    for arch in L.SETTINGS:
        vals = []
        for rep in range(2):
            gate(log)
            r = child(arch, f"pilot{rep}", None, a.n, ["dense"], 3, os.path.join(PILOT, f"{arch}_n{a.n}_rep{rep}.json"))
            vals += [blk["e1000_j"] for blk in r["blocks"]]
        cv = float(np.std(vals, ddof=1) / np.mean(vals))
        res[arch] = {"e1000_j": vals, "cv": cv, "stable": cv <= PILOT_MAX_CV}
        print(f"{arch}: dense E_1000 {np.mean(vals):.3f} J, CV {100 * cv:.2f}%")
    out = {"created": now(), "n": a.n, "rule_max_cv": PILOT_MAX_CV, "results": res, "gates": log,
           "decision": "stable" if all(v["stable"] for v in res.values()) else "unstable"}
    json.dump(out, open(os.path.join(PILOT, f"pilot_n{a.n}.json"), "w", encoding="utf-8"), indent=1)
    print("decision:", out["decision"])


# ------------------------------------------------------------------ step 10
def cmd_sessions(a):
    import numpy as np
    os.makedirs(PARTS, exist_ok=True)
    rng = np.random.default_rng(SESSION_SEED)
    log, schemes = [], set()
    for s in range(1, SESSIONS + 1):
        for arch in ([*L.SETTINGS] if s % 2 else [*L.SETTINGS][::-1]):
            out = os.path.join(PARTS, f"{arch}_session{s}.json")
            if os.path.exists(out):
                continue
            schemes.add(gate(log))
            seed = int(rng.integers(0, 2 ** 31 - 1)) + s
            print(f"{now()} session {s} {arch}", flush=True)
            child(arch, s, seed, a.n, L.METHODS, 1, out)
            log.append({"at": now(), "finished": out})
    json.dump({"gates": log, "power_schemes": sorted(schemes)},
              open(os.path.join(PARTS, f"gates_{datetime.datetime.now():%Y%m%d_%H%M%S}.json"), "w", encoding="utf-8"), indent=1)
    if len(schemes) > 1:
        print("WARNING: the power scheme changed between sessions:", schemes)


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("backend")
    pp = sub.add_parser("pilot")
    pp.add_argument("--n", type=int, default=100000)
    ps = sub.add_parser("sessions")
    ps.add_argument("--n", type=int, required=True)
    c = sub.add_parser("child")
    for k, t in (("--arch", str), ("--session", str), ("--n", int), ("--models", str), ("--blocks", int), ("--out", str)):
        c.add_argument(k, type=t, required=True)
    c.add_argument("--seed", type=int, default=None)
    a = p.parse_args()
    {"backend": cmd_backend, "pilot": cmd_pilot, "sessions": cmd_sessions, "child": cmd_child}[a.cmd](a)


if __name__ == "__main__":
    main()
