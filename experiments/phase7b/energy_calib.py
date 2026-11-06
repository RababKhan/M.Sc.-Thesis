"""Phase 7B, Stage A: RAPL energy of the 200-policy calibration set (design section 7).

  python experiments/phase7b/energy_calib.py pilot [--n N]      # dense-only stability pilot
  python experiments/phase7b/energy_calib.py sessions --n N     # 3 fresh-process sessions over the 200 policies
  python experiments/phase7b/energy_calib.py dataset            # -> energy_surrogate_dataset.csv
Child: pinned to logical CPU 2 and 1 thread before torch import; each policy is slimmed from the frozen dense
reference before its block (never timed); 200 warm-up then N measured batch-1 inferences between two counter reads;
a dense anchor block after every 25 policies. Weights do not change dense CPU work, so unrecovered shapes are used.
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

OUT = os.path.join(L.RESULTS, "_energy_calibration")
WARMUP, ANCHOR_EVERY, DRIFT_LIMIT, SESSIONS, SEED = 200, 25, 0.10, 3, 20261030
PILOT_MAX_CV, MAX_CPU_LOAD = 0.04, 15.0


def now():
    return datetime.datetime.now().astimezone().isoformat(timespec="seconds")


def child(a):
    import psutil
    os.environ["OMP_NUM_THREADS"] = os.environ["MKL_NUM_THREADS"] = "1"
    psutil.Process().cpu_affinity([L.CPU])
    import gc
    import numpy as np
    import torch
    torch.set_num_interop_threads(1)
    torch.set_num_threads(1)
    from src.pruning import structured_units as SU
    E7 = L.energy_meter()
    meter = E7.EnergyMeter()
    ref, _ = L.p3().reference(L.SETTING)
    ref.eval()
    x = torch.randn((1, 3, 32, 32), generator=torch.Generator().manual_seed(0)).contiguous()
    pols = [tuple(p) for p in json.loads(a.policies)]
    if a.seed is not None:
        pols = [pols[i] for i in np.random.default_rng(a.seed).permutation(len(pols))]
    lat = np.empty(a.n, dtype=np.int64)
    blocks = []

    def block(model, label, kind):
        with torch.inference_mode():
            for _ in range(WARMUP):
                model(x)
            gc.collect()
            gc.disable()
            ea, wa = meter.read(), time.perf_counter_ns()
            for i in range(a.n):
                s = time.perf_counter_ns()
                model(x)
                lat[i] = time.perf_counter_ns() - s
            wb, eb = time.perf_counter_ns(), meter.read()
            gc.enable()
        j = E7.EnergyMeter.joules(ea, eb)
        blocks.append({"policy": label, "kind": kind, "at": now(), "seconds": (wb - wa) * 1e-9, "pkg_j": j,
                       "pp0_j": E7.EnergyMeter.joules(ea, eb, "RAPL_Package0_PP0"), "e1000_j": j / a.n * 1000,
                       "latency_median_ms": float(np.median(lat)) * 1e-6})
    for k in range(a.reps):
        block(ref, "0000", "anchor" if pols else "pilot")
    for i, p in enumerate(pols):
        if i and i % ANCHOR_EVERY == 0:
            block(ref, "0000", "anchor")
        m = SU.slim(ref, L.ARCH, list(p))[0].eval()
        block(m, L.key(p), "policy")
    if pols:
        block(ref, "0000", "anchor")
    json.dump({"session": a.session, "order_seed": a.seed, "n": a.n, "warmup": WARMUP, "blocks": blocks,
               "threads": torch.get_num_threads(), "cpu_affinity": psutil.Process().cpu_affinity(),
               "finished_at": now()}, open(a.out, "w", encoding="utf-8"), indent=1)


def gate():
    import psutil
    E7 = L.energy_meter()
    ac, load = E7.on_ac_power(), psutil.cpu_percent(interval=5.0)
    if not ac:
        raise SystemExit("STOP: not on AC power")
    if load > MAX_CPU_LOAD:
        raise SystemExit(f"STOP: CPU load {load:.1f}% > {MAX_CPU_LOAD}%")
    return {"at": now(), "ac_power": ac, "cpu_load_pct": load, "power_scheme": E7.power_scheme()}


def spawn(args):
    subprocess.run([sys.executable, os.path.abspath(__file__), "child", *args], check=True)


def cmd_pilot(a):
    import numpy as np
    os.makedirs(OUT, exist_ok=True)
    vals, gates = [], []
    for rep in range(2):
        gates.append(gate())
        out = os.path.join(OUT, f"pilot_n{a.n}_rep{rep}.json")
        spawn(["--session", f"pilot{rep}", "--n", str(a.n), "--policies", "[]", "--reps", "3", "--out", out])
        vals += [b["e1000_j"] for b in json.load(open(out, encoding="utf-8"))["blocks"]]
    cv = float(np.std(vals, ddof=1) / np.mean(vals))
    res = {"created": now(), "n": a.n, "e1000_j": vals, "cv": cv, "rule_max_cv": PILOT_MAX_CV,
           "decision": "stable" if cv <= PILOT_MAX_CV else "unstable", "gates": gates}
    json.dump(res, open(os.path.join(OUT, f"pilot_n{a.n}.json"), "w", encoding="utf-8"), indent=1)
    print(f"dense E_1000 {np.mean(vals):.3f} J, CV {100 * cv:.2f}% -> {res['decision']}")


def anchor_drift(path):
    import numpy as np
    a = [b["e1000_j"] for b in json.load(open(path, encoding="utf-8"))["blocks"] if b["kind"] == "anchor"]
    return float(np.max(np.abs(np.array(a) / np.median(a) - 1)))


def cmd_sessions(a):
    import numpy as np
    pols = json.load(open(os.path.join(L.RESULTS, "stage_a_samples.json"), encoding="utf-8"))["energy_calibration"]["policies"]
    rng = np.random.default_rng(SEED)
    seeds = [int(s) for s in rng.integers(0, 2 ** 31 - 1, size=SESSIONS + 3)]
    log = []
    s, attempt = 1, 0
    while s <= SESSIONS:
        out = os.path.join(OUT, f"session{s}.json")
        if os.path.exists(out):
            s += 1
            continue
        log.append(gate())
        tmp = out + ".tmp.json"
        print(f"{now()} calibration session {s}", flush=True)
        spawn(["--session", str(s), "--n", str(a.n), "--policies", json.dumps(pols), "--seed", str(seeds[s - 1 + attempt]),
               "--reps", "1", "--out", tmp])
        d = anchor_drift(tmp)
        log.append({"at": now(), "session": s, "anchor_drift": d})
        if d > DRIFT_LIMIT:
            print(f"  anchor drift {100 * d:.1f}% > 10%: session repeated", flush=True)
            os.replace(tmp, out.replace(".json", f"_rejected{attempt}.json"))
            attempt += 1
            if attempt > 3:
                raise SystemExit("STOP: three sessions rejected for drift")
            continue
        os.replace(tmp, out)
        s += 1
    json.dump(log, open(os.path.join(OUT, f"sessions_log_{datetime.datetime.now():%Y%m%d_%H%M%S}.json"), "w",
                        encoding="utf-8"), indent=1)


def cmd_dataset(_a):
    import csv
    import numpy as np
    t = L.table()
    per = {}
    for s in range(1, SESSIONS + 1):
        for b in json.load(open(os.path.join(OUT, f"session{s}.json"), encoding="utf-8"))["blocks"]:
            if b["kind"] == "policy":
                per.setdefault(b["policy"], []).append(b)
    rows = []
    for k, bl in sorted(per.items()):
        p = tuple(int(c) for c in k)
        assert len(bl) == SESSIONS
        e = [b["e1000_j"] for b in bl]
        rows.append({"policy": json.dumps(list(p)), "w1": t[p]["widths"][0], "w2": t[p]["widths"][1], "w3": t[p]["widths"][2],
                     "w4": t[p]["widths"][3], "macs": t[p]["macs"], "params": t[p]["params"],
                     "memory_bytes": t[p]["memory_bytes"], "latency_table_ms": t[p]["latency_ms"],
                     "latency_measured_ms": float(np.median([b["latency_median_ms"] for b in bl])),
                     "e1000_j": float(np.median(e)), "e1000_s1": e[0], "e1000_s2": e[1], "e1000_s3": e[2],
                     "e1000_cv": float(np.std(e, ddof=1) / np.mean(e))})
    path = os.path.join(L.RESULTS, "energy_surrogate_dataset.csv")
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"{len(rows)} policies -> {path}; median per-policy CV {100 * np.median([r['e1000_cv'] for r in rows]):.2f}%")


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    pp = sub.add_parser("pilot")
    pp.add_argument("--n", type=int, default=20000)
    ps = sub.add_parser("sessions")
    ps.add_argument("--n", type=int, required=True)
    sub.add_parser("dataset")
    c = sub.add_parser("child")
    for k, t in (("--session", str), ("--n", int), ("--policies", str), ("--reps", int), ("--out", str)):
        c.add_argument(k, type=t, required=True)
    c.add_argument("--seed", type=int, default=None)
    a = p.parse_args()
    {"pilot": cmd_pilot, "sessions": cmd_sessions, "dataset": cmd_dataset, "child": child}[a.cmd](a)


if __name__ == "__main__":
    main()
