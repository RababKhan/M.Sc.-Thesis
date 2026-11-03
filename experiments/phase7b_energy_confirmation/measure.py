"""Phase 7B energy confirmation: direct RAPL measurement of the frozen Phase 7B SimpleCNN models (protocol.json).

  python experiments/phase7b_energy_confirmation/measure.py verify        # checkpoint hashes -> checkpoint_verification.json
  python experiments/phase7b_energy_confirmation/measure.py environment   # -> environment.json
  python experiments/phase7b_energy_confirmation/measure.py qualify       # Dense-only measurement qualification
  python experiments/phase7b_energy_confirmation/measure.py sessions      # 10 paired Dense / PPO sessions (if qualified)
  python experiments/phase7b_energy_confirmation/measure.py secondary     # 5 sessions with all 5 models (if qualified)
Every run is a fresh child process pinned to logical CPU 2 with one intra-op and one inter-op thread. The child loads
and hash-checks the frozen recovered models, soaks the package thermally, then measures blocks of WARMUP + N batch-1
inferences between two RAPL package counter reads. Nothing here trains, tunes or changes a model.
"""
import argparse
import ctypes
import datetime
import json
import os
import subprocess
import sys
import time
from ctypes import wintypes

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(ROOT, "experiments", "phase7b"))
import p7b_lib as L  # noqa: E402

OUT = os.path.join(ROOT, "results", "phase7b_energy_confirmation")
RAW = os.path.join(OUT, "_raw")
DENSE, PPO = "0000", "2225"
SECONDARY = [DENSE, PPO, "uniform_r0.208", "lamp_s56.2", "phase7a_ppo"]
CPU, WARMUP, N, SOAK_S, PAUSE_S = L.CPU, 2000, 200000, 180, 5
MAX_LOAD, MAX_CV = 15.0, 0.03
QUAL_RUNS, QUAL_BLOCKS, SESSIONS, SESSION_SEED, SECONDARY_SESSIONS = 3, 2, 10, 20261103, 5
MAX_VOIDS, GATE_WAIT_S = 3, 1800


def now():
    return datetime.datetime.now().astimezone().isoformat(timespec="seconds")


def jdump(obj, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    json.dump(obj, open(path + ".tmp", "w", encoding="utf-8"), indent=1)
    os.replace(path + ".tmp", path)


# ------------------------------------------------------------------ PDH: clock level and thermal zone (outside the window)
class _Fmt(ctypes.Structure):
    _fields_ = [("CStatus", wintypes.DWORD), ("pad", wintypes.DWORD), ("value", ctypes.c_double)]


class Pdh:
    """% Processor Performance is a rate counter: the value after a collect is the average since the previous collect,
    so collecting at block start and block end gives the in-block average without touching the measured loop."""
    PATHS = {"perf_pct_cpu2": "\\Processor Information(0,2)\\% Processor Performance",
             "perf_pct_total": "\\Processor Information(_Total)\\% Processor Performance",
             "nominal_mhz": "\\Processor Information(0,2)\\Processor Frequency",
             "thermal_zone_k": "\\Thermal Zone Information(\\_TZ.THRM)\\Temperature"}

    def __init__(self):
        self.pdh = ctypes.WinDLL("pdh.dll")
        self.q = ctypes.c_void_p()
        assert self.pdh.PdhOpenQueryW(None, None, ctypes.byref(self.q)) == 0
        self.h = {}
        for k, p in self.PATHS.items():
            h = ctypes.c_void_p()
            if self.pdh.PdhAddEnglishCounterW(self.q, p, None, ctypes.byref(h)) == 0:
                self.h[k] = h
        self.pdh.PdhCollectQueryData(self.q)

    def sample(self):
        self.pdh.PdhCollectQueryData(self.q)
        out = {}
        for k, h in self.h.items():
            v, t = _Fmt(), wintypes.DWORD()
            st = self.pdh.PdhGetFormattedCounterValue(h, 0x00000200, ctypes.byref(t), ctypes.byref(v))
            out[k] = float(v.value) if st == 0 else None
        if out.get("thermal_zone_k") is not None:
            out["thermal_zone_c"] = out.pop("thermal_zone_k") - 273.15
        return out


# ------------------------------------------------------------------ child: one fresh measurement process
def child(a):
    import psutil
    os.environ["OMP_NUM_THREADS"] = os.environ["MKL_NUM_THREADS"] = "1"
    psutil.Process().cpu_affinity([CPU])
    import gc
    import numpy as np
    import torch
    torch.set_num_interop_threads(1)
    torch.set_num_threads(1)
    P7 = L.energy_meter()
    FB = L.load_module("energy_final", os.path.join(ROOT, "experiments", "phase7b"))   # frozen loader, read only
    meter, pdh = P7.EnergyMeter(), Pdh()
    ref, ref_sha = L.p3().reference(L.SETTING)
    order = a.order.split(",")
    models = {m: FB.load_model(m, ref, P7).eval() for m in dict.fromkeys(order)}
    x = torch.zeros((1, 3, 32, 32), dtype=torch.float32)
    x.copy_(torch.randn((1, 3, 32, 32), generator=torch.Generator().manual_seed(0)))
    lat = np.empty(a.n, dtype=np.int64)
    out = {"run": a.run, "stage": a.stage, "order": order, "n": a.n, "warmup": WARMUP, "soak_s": SOAK_S,
           "reference_sha256": ref_sha, "started_at": now(), "blocks": []}
    with torch.inference_mode():
        m = models[order[0]]                                       # thermal soak with the first block's model
        s0, e0, t0, k = pdh.sample(), meter.read(), time.perf_counter(), 0
        while time.perf_counter() - t0 < SOAK_S:
            for _ in range(1000):
                m(x)
            k += 1000
        t1, e1, s1 = time.perf_counter(), meter.read(), pdh.sample()
        out["soak"] = {"model": order[0], "seconds": t1 - t0, "inferences": k, "pkg_j": P7.EnergyMeter.joules(e0, e1),
                       "pkg_w": P7.EnergyMeter.joules(e0, e1) / (t1 - t0), "pdh_start": s0, "pdh_end": s1}
        seen = {}
        for pos, name in enumerate(order):
            m = models[name]
            for _ in range(WARMUP):
                m(x)
            gc.collect()
            gc.disable()
            pre = pdh.sample()
            psutil.cpu_percent(None)
            ea, wa = meter.read(), time.perf_counter_ns()
            for i in range(a.n):
                s = time.perf_counter_ns()
                m(x)
                lat[i] = time.perf_counter_ns() - s
            wb, eb = time.perf_counter_ns(), meter.read()
            load = psutil.cpu_percent(None)
            post = pdh.sample()
            gc.enable()
            dur = (wb - wa) * 1e-9
            pkg = P7.EnergyMeter.joules(ea, eb)
            seen[name] = seen.get(name, -1) + 1
            b = {"model": name, "appearance": seen[name], "position": pos, "at": now(), "n": a.n, "seconds": dur,
                 "pkg_j": pkg, "pp0_j": P7.EnergyMeter.joules(ea, eb, "RAPL_Package0_PP0"),
                 "pkg_counter_ms": eb["RAPL_Package0_PKG_Time"] - ea["RAPL_Package0_PKG_Time"],
                 "e1000_j": pkg / a.n * 1000, "j_per_inference": pkg / a.n, "pkg_w": pkg / dur,
                 "latency_median_ms": float(np.median(lat)) * 1e-6, "latency_mean_ms": float(lat.mean()) * 1e-6,
                 "latency_wall_ms": dur / a.n * 1e3, "latency_p95_ms": float(np.percentile(lat, 95)) * 1e-6,
                 "latency_iqr_ms": float(np.subtract(*np.percentile(lat, [75, 25]))) * 1e-6,
                 "system_load_pct": load, "perf_pct_cpu2": post.get("perf_pct_cpu2"),
                 "perf_pct_total": post.get("perf_pct_total"), "nominal_mhz": post.get("nominal_mhz"),
                 "thermal_zone_c_start": pre.get("thermal_zone_c"), "thermal_zone_c_end": post.get("thermal_zone_c"),
                 "ac_power_end": P7.on_ac_power()}
            out["blocks"].append(b)
            print(f"  {a.run} {name:15s} {b['e1000_j']:.3f} J/1000 {b['latency_median_ms']:.4f} ms "
                  f"load {load:.1f}% perf {b['perf_pct_cpu2']}", flush=True)
            time.sleep(PAUSE_S)
    out["finished_at"] = now()
    out["effective"] = {"threads": torch.get_num_threads(), "interop": torch.get_num_interop_threads(),
                        "cpu_affinity": psutil.Process().cpu_affinity(), "torch": torch.__version__,
                        "power_scheme": P7.power_scheme()}
    jdump(out, a.out)


# ------------------------------------------------------------------ parent: gates, void handling, stages
def gate():
    """Pre-run gate; waits (measuring nothing) up to GATE_WAIT_S for AC power and a quiet system, else STOP."""
    import psutil
    P7 = L.energy_meter()
    waited, t0 = [], time.time()
    while True:
        ac, load, scheme = P7.on_ac_power(), psutil.cpu_percent(interval=5.0), P7.power_scheme()
        rec = {"at": now(), "ac_power": ac, "cpu_load_pct": load, "power_scheme": scheme}
        if ac and load <= MAX_LOAD and "Balanced" in scheme:
            return rec, waited
        waited.append(rec)
        if time.time() - t0 > GATE_WAIT_S:
            raise SystemExit(f"STOP: gate not satisfied for {GATE_WAIT_S} s: {rec}")
        time.sleep(55)


def valid(run):
    bad = [b["model"] + f"@{b['position']}" for b in run["blocks"] if b["system_load_pct"] > MAX_LOAD or not b["ac_power_end"]]
    return not bad, bad


def execute(stage, name, order, log):
    """Run one fresh child; a run with any block over the load limit or off AC is void and repeated (never judged on
    energy values); more than MAX_VOIDS voids in a stage -> STOP."""
    final = os.path.join(RAW, stage, f"{name}.json")
    if os.path.exists(final):
        return json.load(open(final, encoding="utf-8"))
    while True:
        g, waited = gate()
        tmp = os.path.join(RAW, stage, f"{name}.running.json")
        print(f"{now()} {stage} {name} order={order}", flush=True)
        subprocess.run([sys.executable, os.path.abspath(__file__), "child", "--stage", stage, "--run", name,
                        "--order", ",".join(order), "--n", str(N), "--out", tmp], check=True)
        run = json.load(open(tmp, encoding="utf-8"))
        run["gate"], run["gate_waits"] = g, waited
        ok, bad = valid(run)
        if ok:
            jdump(run, final)
            os.remove(tmp)
            log.append({"at": now(), "run": name, "status": "valid"})
            return run
        voids = len([e for e in log if e["status"] == "void"]) + 1
        vpath = os.path.join(RAW, stage, "void", f"{name}_void{voids}.json")
        jdump({**run, "void_reason": bad}, vpath)
        os.remove(tmp)
        log.append({"at": now(), "run": name, "status": "void", "blocks": bad, "file": os.path.relpath(vpath, OUT)})
        print(f"{now()} VOID {name}: {bad}", flush=True)
        if voids > MAX_VOIDS:
            raise SystemExit(f"STOP: more than {MAX_VOIDS} void runs in stage {stage}")


def stage_log(stage):
    p = os.path.join(RAW, stage, "run_log.json")
    return p, (json.load(open(p, encoding="utf-8")) if os.path.exists(p) else [])


def cmd_qualify(a):
    import numpy as np
    p, log = stage_log("qualification")
    try:
        runs = [execute("qualification", f"qual{r + 1}", [DENSE] * QUAL_BLOCKS, log) for r in range(QUAL_RUNS)]
    finally:
        jdump(log, p)
    rows = [{"run": r["run"], **{k: b[k] for k in ("model", "position", "at", "n", "seconds", "pkg_j", "e1000_j",
                                                    "j_per_inference", "pkg_w", "latency_median_ms", "latency_mean_ms",
                                                    "system_load_pct", "perf_pct_cpu2", "perf_pct_total",
                                                    "thermal_zone_c_start", "thermal_zone_c_end", "ac_power_end")},
             "warmup": r["warmup"], "soak_s": r["soak_s"], "cpu_affinity": r["effective"]["cpu_affinity"],
             "power_scheme": r["effective"]["power_scheme"]} for r in runs for b in r["blocks"]]
    v = np.array([r["e1000_j"] for r in rows])
    cv = float(v.std(ddof=1) / v.mean())
    res = {"created": now(), "criterion": f"CV of the {len(v)} Dense E_1000 blocks <= {MAX_CV:.0%} (decided before any data)",
           "n": N, "warmup": WARMUP, "soak_s": SOAK_S, "runs": QUAL_RUNS, "blocks_per_run": QUAL_BLOCKS,
           "e1000_j": v.tolist(), "mean_j": float(v.mean()), "sd_j": float(v.std(ddof=1)), "cv": cv,
           "decision": "qualified" if cv <= MAX_CV else "not qualified", "run_log": log, "blocks": rows}
    jdump(res, os.path.join(OUT, "measurement_qualification.json"))
    import csv
    with open(os.path.join(OUT, "measurement_qualification.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows([{k: json.dumps(x) if isinstance(x, list) else x for k, x in r.items()} for r in rows])
    print(f"qualification: mean {v.mean():.3f} J/1000, CV {100 * cv:.2f}% -> {res['decision']}")


def qualified():
    q = os.path.join(OUT, "measurement_qualification.json")
    if not os.path.exists(q) or json.load(open(q, encoding="utf-8"))["decision"] != "qualified":
        raise SystemExit("STOP: the measurement setup is not qualified; no model comparison is run")


def cmd_sessions(a):
    import numpy as np
    qualified()
    first = np.random.default_rng(SESSION_SEED).permutation([0] * (SESSIONS // 2) + [1] * (SESSIONS - SESSIONS // 2))
    p, log = stage_log("sessions")
    try:
        for s in range(SESSIONS):
            execute("sessions", f"session{s + 1:02d}", [DENSE, PPO] if first[s] == 0 else [PPO, DENSE], log)
    finally:
        jdump(log, p)


def cmd_secondary(a):
    import numpy as np
    qualified()
    assert all(os.path.exists(os.path.join(RAW, "sessions", f"session{s + 1:02d}.json")) for s in range(SESSIONS)), \
        "the primary sessions come first"
    rng = np.random.default_rng(SESSION_SEED + 1)
    orders = [[SECONDARY[i] for i in rng.permutation(len(SECONDARY))] for _ in range(SECONDARY_SESSIONS)]
    p, log = stage_log("secondary")
    try:
        for s in range(SECONDARY_SESSIONS):
            execute("secondary", f"secondary{s + 1:02d}", orders[s], log)
    finally:
        jdump(log, p)


# ------------------------------------------------------------------ freeze checks and environment
def cmd_verify(a):
    import torch
    from src.utils import sha256_state_dict
    P7 = L.energy_meter()
    FB = L.load_module("energy_final", os.path.join(ROOT, "experiments", "phase7b"))
    ref, ref_sha = L.p3().reference(L.SETTING)
    sel = json.load(open(os.path.join(L.RESULTS, "selected_policies.json"), encoding="utf-8"))
    out = {"created": now(), "reference_sha256": ref_sha, "models": {}}
    for mid in SECONDARY:
        if mid == "phase7a_ppo":
            o = sel["methods"]["old_ppo"]
            path, want_file = o["checkpoint"], o["checkpoint_sha256"]
            m = FB.load_model(mid, ref, P7)
            import pandas as pd
            a7 = pd.read_csv(os.path.join(ROOT, "results", "phase7a_sustainability", "accuracy_results.csv"))
            want_state = a7[(a7.setting == L.SETTING) & (a7.method == "ppo")].final_state_sha256.iloc[0]
            src = "results/phase7b_green_pruning/selected_policies.json + results/phase7a_sustainability/accuracy_results.csv"
        else:
            rec = json.load(open(os.path.join(L.RESULTS, "_recovery_runs", "final", f"{mid}_E{L.FINAL_EPOCHS}.json"), encoding="utf-8"))
            path, want_file, want_state = rec["checkpoint"], rec["checkpoint_sha256"], rec["final_state_sha256"]
            m = FB.load_model(mid, ref, P7)
            src = f"results/phase7b_green_pruning/_recovery_runs/final/{mid}_E{L.FINAL_EPOCHS}.json"
        got_file, got_state = L.sha256_file(os.path.join(ROOT, path)), sha256_state_dict(m.state_dict())
        out["models"][mid] = {"role": {DENSE: "dense", PPO: "phase7b_ppo [2,2,2,5]"}.get(mid, "secondary comparator"),
                              "checkpoint": path.replace("\\", "/"), "freeze_record": src,
                              "file_sha256": got_file, "file_sha256_recorded": want_file, "file_match": got_file == want_file,
                              "state_sha256": got_state, "state_sha256_recorded": want_state,
                              "state_match": got_state == want_state,
                              "params": sum(p.numel() for p in m.parameters()), "training_mode": m.training}
    out["all_match"] = all(v["file_match"] and v["state_match"] for v in out["models"].values())
    jdump(out, os.path.join(OUT, "checkpoint_verification.json"))
    print(json.dumps({k: (v["file_match"], v["state_match"]) for k, v in out["models"].items()}), "all", out["all_match"])
    if not out["all_match"]:
        raise SystemExit("STOP: checkpoint integrity failure")


def cmd_environment(a):
    import platform
    import psutil
    import torch
    P7 = L.energy_meter()
    cpu = subprocess.run(["powershell", "-NoProfile", "-Command",
                          "Get-CimInstance Win32_Processor | Select-Object Name,NumberOfCores,NumberOfLogicalProcessors,"
                          "MaxClockSpeed | ConvertTo-Json"], capture_output=True, text=True).stdout
    osv = subprocess.run(["powershell", "-NoProfile", "-Command",
                          "Get-CimInstance Win32_OperatingSystem | Select-Object Caption,Version,BuildNumber | ConvertTo-Json"],
                         capture_output=True, text=True).stdout
    git = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    env = {"created": now(), "cpu": json.loads(cpu), "os": json.loads(osv), "python": sys.version,
           "platform": platform.platform(), "torch": torch.__version__,
           "mkldnn_available": torch.backends.mkldnn.is_available(), "torch_config": torch.__config__.show(),
           "parallel_info": torch.__config__.parallel_info(), "ram_gb": psutil.virtual_memory().total / 2 ** 30,
           "ac_power": P7.on_ac_power(), "power_scheme": P7.power_scheme(), "battery": str(psutil.sensors_battery()),
           "energy_backend": "RAPL package counter via Windows Energy Meter Interface (PDH \\Energy Meter(RAPL_Package0_PKG))",
           "pdh_counters": Pdh.PATHS, "pdh_sample": Pdh().sample(), "git_head": git,
           "measurement": {"cpu_affinity": [CPU], "intra_op_threads": 1, "inter_op_threads": 1, "batch": 1,
                           "dtype": "float32", "input": "preallocated (1,3,32,32), torch.manual_seed(0) normal",
                           "mode": "eval() + torch.inference_mode()", "warmup_per_block": WARMUP, "n_per_block": N,
                           "soak_s_per_run": SOAK_S, "pause_s": PAUSE_S, "model_loading": "excluded (before the soak)"}}
    jdump(env, os.path.join(OUT, "environment.json"))
    print(json.dumps({k: env[k] for k in ("cpu", "os", "torch", "ac_power", "power_scheme", "pdh_sample")}, default=str))


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    for c in ("verify", "environment", "qualify", "sessions", "secondary"):
        sub.add_parser(c)
    c = sub.add_parser("child")
    for k, t in (("--stage", str), ("--run", str), ("--order", str), ("--n", int), ("--out", str)):
        c.add_argument(k, type=t, required=True)
    a = p.parse_args()
    {"verify": cmd_verify, "environment": cmd_environment, "qualify": cmd_qualify, "sessions": cmd_sessions,
     "secondary": cmd_secondary, "child": child}[a.cmd](a)


if __name__ == "__main__":
    main()
