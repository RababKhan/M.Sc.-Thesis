"""Paper 3, H2-A: exhaustive structured latency tables (pre-registration §4; H0 configuration A, unchanged).

  python experiments/paper3/h2_latency.py            # 4 tables x 5 sessions, merge, integrity, freeze
A session process (started by the parent) times all 1,296 shapes of one (architecture, classes) pair:
  python experiments/paper3/h2_latency.py session --pair P --session S --seed X --out PART
No accuracy, no data: shapes are built from the frozen dense reference before each timed block.
"""
import argparse
import hashlib
import itertools
import json
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import p3_lib as L  # noqa: E402  (imports neither torch nor numpy)

PAIRS = {"simplecnn_c10": "cifar10_simplecnn", "simplecnn_c100": "cifar100_simplecnn",
         "resnet8_c10": "cifar10_resnet8", "resnet8_c100": "cifar100_resnet8"}
POLICIES = [tuple(p) for p in itertools.product(range(6), repeat=4)]
CFG = L.CONFIGS["A"]
SESSIONS, ANCHOR_EVERY, DRIFT_LIMIT, SEED = 5, 25, 0.10, 20261024
PARTS = os.path.join(L.RESULTS, "_h2_latency_parts")


def session(a):
    import psutil
    for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[var] = str(CFG["threads"])
    psutil.Process().cpu_affinity(CFG["cpus"])
    import gc
    import numpy as np
    import torch
    torch.set_num_interop_threads(CFG["interop"])
    torch.set_num_threads(CFG["threads"])
    from src.pruning import structured_units as SU
    setting = PAIRS[a.pair]
    arch = setting.split("_")[1]
    ref, sha = L.reference(setting)
    ref.eval()
    x = torch.randn((CFG["batch"], 3, 32, 32), generator=torch.Generator().manual_seed(0)).contiguous()
    order = np.random.default_rng(a.seed).permutation(len(POLICIES))
    shape_ns = np.zeros((len(POLICIES), CFG["timed"]), dtype=np.int64)
    anchors = []

    def timed(model, n):
        out = np.empty(n, dtype=np.int64)
        for i in range(n):
            t0 = time.perf_counter_ns()
            model(x)
            out[i] = time.perf_counter_ns() - t0
        return out
    t_start = time.time()
    with torch.inference_mode():
        for _ in range(CFG["warmup"]):
            ref(x)
        for pos, idx in enumerate(order):
            if pos % ANCHOR_EVERY == 0:
                gc.collect()
                gc.disable()
                anchors.append(timed(ref, CFG["timed"]))
                gc.enable()
            m = SU.slim(ref, arch, POLICIES[idx])[0].eval()          # built before timing, never timed
            gc.collect()
            gc.disable()
            for _ in range(CFG["warmup"]):
                m(x)
            shape_ns[idx] = timed(m, CFG["timed"])
            gc.enable()
        gc.disable()
        anchors.append(timed(ref, CFG["timed"]))
        gc.enable()
    np.savez_compressed(a.out + ".npz", shape_ns=shape_ns.astype(np.int32), anchors=np.stack(anchors).astype(np.int32),
                        order=order.astype(np.int32))
    assert shape_ns.max() < 2 ** 31
    meta = {"pair": a.pair, "setting": setting, "session": a.session, "order_seed": a.seed, "reference_sha256": sha,
            "requested": {"threads": CFG["threads"], "interop": CFG["interop"], "cpus": CFG["cpus"]},
            "effective": {"threads": torch.get_num_threads(), "interop": torch.get_num_interop_threads(),
                          "cpus": psutil.Process().cpu_affinity(), "mkldnn_enabled": torch.backends.mkldnn.enabled},
            "torch": torch.__version__, "python": sys.version.split()[0], "batch": CFG["batch"], "warmup": CFG["warmup"],
            "timed": CFG["timed"], "anchor_every": ANCHOR_EVERY, "seconds": round(time.time() - t_start, 1)}
    json.dump(meta, open(a.out + ".json", "w", encoding="utf-8"), indent=1)


# ------------------------------------------------------------------ parent
def cpu_load():
    import psutil
    return psutil.cpu_percent(interval=5.0)


def anchor_drift(path):
    import numpy as np
    z = np.load(path + ".npz")
    med = np.median(z["anchors"], axis=1)
    return float(np.abs(med / np.median(med) - 1).max())


def run_sessions(log):
    import numpy as np
    os.makedirs(PARTS, exist_ok=True)
    rng = np.random.default_rng(SEED)
    plan = [(p, s, int(rng.integers(2 ** 31 - 1))) for s in range(1, SESSIONS + 1) for p in [list(PAIRS)[i] for i in rng.permutation(len(PAIRS))]]
    for pair, s, seed in plan:
        out = os.path.join(PARTS, f"{pair}_s{s}")
        if os.path.exists(out + ".json"):
            continue
        for attempt in range(1, 4):
            before = cpu_load()
            for _ in range(6):
                if before <= L.MAX_CPU_LOAD:
                    break
                time.sleep(20)
                before = cpu_load()
            subprocess.run([sys.executable, os.path.abspath(__file__), "session", "--pair", pair, "--session", str(s),
                            "--seed", str(seed), "--out", out], check=True)
            after = cpu_load()
            drift = anchor_drift(out)
            ok = before <= L.MAX_CPU_LOAD and after <= L.MAX_CPU_LOAD and drift <= DRIFT_LIMIT
            log.append({"pair": pair, "session": s, "attempt": attempt, "load_before": before, "load_after": after,
                        "anchor_drift": drift, "ok": ok})
            json.dump(log, open(os.path.join(PARTS, "session_log.json"), "w", encoding="utf-8"), indent=1)
            print(f"{time.strftime('%H:%M')} {pair} session {s} attempt {attempt}: load {before:.1f}/{after:.1f}% drift {100 * drift:.1f}% "
                  f"{'ok' if ok else 'REPEAT'}", flush=True)
            if ok or attempt == 3:
                break
            for ext in (".json", ".npz"):
                os.rename(out + ext, out + f".failed{attempt}{ext}")
    return plan


def sha(path):
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


def build_tables(plan, log):
    import numpy as np
    import pandas as pd
    import torch
    from src import models as M
    from src.evaluation import efficiency as E
    from src.pruning import structured_units as SU
    torch.set_num_threads(4)
    h0env = json.load(open(os.path.join(L.RESULTS, "h0_environment.json"), encoding="utf-8"))
    checks, frozen = [], {}

    def chk(name, ok, detail=""):
        checks.append({"check": name, "pass": bool(ok), "detail": str(detail)})
        print(f"[{'PASS' if ok else 'FAIL'}] {name}: {str(detail)[:150]}", flush=True)
    for pair, setting in PAIRS.items():
        arch, classes = setting.split("_")[1], (10 if setting.startswith("cifar10_") else 100)
        sess = sorted(s for p, s, _ in plan if p == pair)
        parts = [os.path.join(PARTS, f"{pair}_s{s}") for s in sess]
        data = [np.load(p + ".npz") for p in parts]
        metas = [json.load(open(p + ".json", encoding="utf-8")) for p in parts]
        ns = np.stack([d["shape_ns"].astype(np.int64) for d in data])                 # (sessions, 1296, 300)
        anc = [d["anchors"].astype(np.int64) for d in data]
        sess_med = np.median(ns, axis=2) / 1e6                                        # ms
        anchor_med = np.array([np.median(a) for a in anc]) / 1e6
        # anchor-normalised (secondary): each shape divided by the mean of its two surrounding anchors' medians
        norm = np.zeros_like(sess_med)
        for j, (d, a) in enumerate(zip(data, anc)):
            am = np.median(a, axis=1) / 1e6
            for pos, idx in enumerate(d["order"]):
                k = pos // ANCHOR_EVERY
                norm[j, idx] = sess_med[j, idx] / ((am[k] + am[k + 1]) / 2) * np.median(am)
        raw_rows = [{"pair": pair, "session": s, "policy": str(list(POLICIES[i])), "median_ms": sess_med[j, i]}
                    for j, s in enumerate(sess) for i in range(len(POLICIES))]
        pd.DataFrame(raw_rows).to_csv(os.path.join(L.RESULTS, f"h2_latency_raw_{pair}.csv"), index=False, float_format="%.6f")
        ref, ref_sha = L.reference(setting)
        dense_params, dense_macs = M.count_parameters(ref)["total"], E.conv_linear_macs(ref)[0]
        lat = np.median(sess_med, axis=0)
        pooled = ns.transpose(1, 0, 2).reshape(len(POLICIES), -1) / 1e6
        q1, q3, p95 = (np.percentile(pooled, q, axis=1) for q in (25, 75, 95))
        ident = POLICIES.index((0, 0, 0, 0))
        rows, shapes, bad_counts = [], set(), []
        for i, p in enumerate(POLICIES):
            m = SU.slim(ref, arch, p)[0]
            shapes.add(SU.shape_key(m))
            params, macs = M.count_parameters(m)["total"], E.conv_linear_macs(m)[0]
            if (macs, params) != SU.analytic(arch, p, classes):
                bad_counts.append(p)
            rows.append({"policy": str(list(p)), "a1": p[0], "a2": p[1], "a3": p[2], "a4": p[3], "widths": str(SU.widths(arch, p)),
                         "params": params, "param_reduction": 1 - params / dense_params, "macs": macs, "mac_reduction": 1 - macs / dense_macs,
                         "flops": E.torch_flops(m), "latency_ms": lat[i], "p95_ms": p95[i], "iqr_ms": q3[i] - q1[i],
                         "session_min_ms": sess_med[:, i].min(), "session_max_ms": sess_med[:, i].max(),
                         "latency_anchor_normalised_ms": float(np.median(norm[:, i])), "speedup": lat[ident] / lat[i],
                         "latency_gain": 1 - lat[i] / lat[ident], "throughput_img_s": 1000.0 / lat[i]})
        table = pd.DataFrame(rows)
        path = os.path.join(L.RESULTS, f"h2_latency_table_{pair}.csv")
        table.to_csv(path, index=False, float_format="%.9g")
        frozen[pair] = {"table": os.path.relpath(path, L.ROOT).replace("\\", "/"), "sha256": sha(path), "reference_sha256": ref_sha,
                        "sessions": sess, "parts_sha256": {os.path.basename(p) + ".npz": sha(p + ".npz") for p in parts}}
        chk(f"{pair}: exactly 1,296 policies, no duplicate action tuple", len(table) == 1296 and table.policy.nunique() == 1296)
        chk(f"{pair}: 1,296 distinct physical shapes", len(shapes) == 1296, len(shapes))
        chk(f"{pair}: parameters and MACs equal the H1 analytic counts for all 1,296; FLOPs = 2 x MACs",
            not bad_counts and bool((table.flops == 2 * table.macs).all()), bad_counts[:3])
        ratio = lat[ident] / np.median(anchor_med)
        chk(f"{pair}: identity-policy latency matches the dense anchor latency of the same sessions (within 5%)", abs(ratio - 1) <= 0.05,
            f"identity {lat[ident]:.4f} ms / anchors {np.median(anchor_med):.4f} ms = {ratio:.4f}")
        chk(f"{pair}: {len(sess)} sessions; every session timed 1,296 x {CFG['timed']} valid passes (all > 0)",
            len(sess) == SESSIONS and ns.shape == (SESSIONS, 1296, CFG["timed"]) and bool((ns > 0).all()), ns.shape)
        chk(f"{pair}: session metadata = H0 configuration A and H0 environment (CPU affinity [2], 1+1 threads, oneDNN, torch/python)",
            all(mt["effective"]["cpus"] == CFG["cpus"] and mt["effective"]["threads"] == 1 and mt["effective"]["interop"] == 1
                and mt["effective"]["mkldnn_enabled"] and mt["torch"] == h0env["torch"] and mt["python"] == h0env["python"]
                and mt["reference_sha256"] == ref_sha for mt in metas), "")
        drifts = [anchor_drift(p) for p in parts]
        chk(f"{pair}: dense anchor drift within every kept session <= {100 * DRIFT_LIMIT:.0f}%", max(drifts) <= DRIFT_LIMIT,
            [round(d, 4) for d in drifts])
    if "simplecnn_c10" in frozen:
        t = pd.read_csv(os.path.join(L.RESULTS, "h2_latency_table_simplecnn_c10.csv"))
        h0 = pd.read_csv(os.path.join(L.RESULTS, "h0_latency_summary.csv"))
        h0d = float(h0[(h0.config == "A") & (h0.model == "dense")].median_ms.iloc[0])
        ident = float(t[t.policy == "[0, 0, 0, 0]"].latency_ms.iloc[0])
        chk("simplecnn_c10: identity latency agrees with the H0 configuration-A dense median (within 10%)", abs(ident / h0d - 1) <= 0.10,
            f"H2 {ident:.4f} ms vs H0 {h0d:.4f} ms")
    chk("CPU load <= 15% before and after every kept session", all(r["ok"] for r in log if r["attempt"] == max(
        q["attempt"] for q in log if q["pair"] == r["pair"] and q["session"] == r["session"])) if log else True,
        f"{sum(not r['ok'] for r in log)} repeats")
    out = {"created": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "protocol": "results/paper3/paper3_preregistration.md §4", "config": CFG,
           "sessions": SESSIONS, "anchor_every": ANCHOR_EVERY, "seed": SEED, "tables": frozen, "checks": checks, "session_log": log}
    json.dump(out, open(os.path.join(L.RESULTS, "h2_latency_freeze.json"), "w", encoding="utf-8"), indent=1, default=str)
    print(f"latency integrity {sum(c['pass'] for c in checks)}/{len(checks)}")


def main():
    from src.evaluation import latency as T
    env = T.machine_info()
    if env["battery"] is not None and not env["battery"]["power_plugged"]:
        raise SystemExit("on battery power: latency tables require AC power")
    log_path = os.path.join(PARTS, "session_log.json")
    log = json.load(open(log_path, encoding="utf-8")) if os.path.exists(log_path) else []
    json.dump(env, open(os.path.join(L.RESULTS, "h2_latency_environment.json"), "w", encoding="utf-8"), indent=1)
    plan = run_sessions(log)
    build_tables(plan, log)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd")
    s = sub.add_parser("session")
    s.add_argument("--pair", required=True)
    s.add_argument("--session", type=int, required=True)
    s.add_argument("--seed", type=int, required=True)
    s.add_argument("--out", required=True)
    a = ap.parse_args()
    session(a) if a.cmd == "session" else main()
