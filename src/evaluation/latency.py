"""Paper 3: CPU latency sessions (protocol: results/paper3/h0_protocol.md).

Extends the Phase-2 protocol (results/cpu_efficiency_protocol.md, src/evaluation/efficiency.py, both unchanged)
with what a latency *reward* needs:
  * one fresh process per (session, configuration); the thread counts and the CPU affinity are fixed in the
    environment before torch is imported (see `pin_process`);
  * per-pass timing with time.perf_counter_ns under torch.inference_mode(), garbage collection disabled;
  * interleaved rounds: in every round a dense anchor block is timed first, then every model once, in an order
    drawn from a recorded seed; the anchor blocks track frequency / thermal drift within a session;
  * the latency of a model in a session is the median of its timed passes; between-session spread is the noise.
Timing never includes model construction, checkpoint loading or data handling: models and the single
preallocated float32 NCHW input exist before the first warm-up pass.
"""
import ctypes
import gc
import os
import platform
import subprocess
import sys
import time

import numpy as np


# ------------------------------------------------------------------ process setup (call before importing torch)
def pin_process(cpus, threads):
    """Fix CPU affinity and thread counts for this process. Must run before `import torch`."""
    import psutil
    for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[var] = str(threads)
    psutil.Process().cpu_affinity(list(cpus))
    return psutil.Process().cpu_affinity()


def cpu_sets():
    """[(logical index, core index, efficiency class)] from GetSystemCpuSetInformation (Windows); [] elsewhere."""
    if sys.platform != "win32":
        return []
    k32 = ctypes.WinDLL("kernel32", use_last_error=True)
    size = ctypes.c_ulong(0)
    k32.GetSystemCpuSetInformation(None, 0, ctypes.byref(size), None, 0)
    buf = ctypes.create_string_buffer(size.value)
    if not k32.GetSystemCpuSetInformation(buf, size, ctypes.byref(size), None, 0):
        return []
    rows, off = [], 0
    while off < size.value:
        n = int.from_bytes(buf.raw[off:off + 4], "little")
        body = buf.raw[off + 8: off + n]
        rows.append((body[6], body[7], body[10]))          # LogicalProcessorIndex, CoreIndex, EfficiencyClass
        off += n
    return sorted(rows)


def _cmd(args):
    try:
        return subprocess.run(args, capture_output=True, text=True, timeout=30).stdout.strip()
    except Exception as e:                                 # noqa: BLE001
        return f"unavailable ({e!r})"


def machine_info():
    """CPU, OS, Python/torch build, threading back-ends, power plan and AC state."""
    import psutil
    import torch
    from src.utils import cpu_name
    bat = psutil.sensors_battery()
    return {
        "cpu": cpu_name(), "physical_cores": psutil.cpu_count(logical=False), "logical_cpus": psutil.cpu_count(),
        "cpu_sets": [{"logical": lp, "core": c, "efficiency_class": e} for lp, c, e in cpu_sets()],
        "os": platform.platform(), "python": sys.version.split()[0], "torch": torch.__version__,
        "torch_config": torch.__config__.show(), "torch_parallel_info": torch.__config__.parallel_info(),
        "mkldnn_available": torch.backends.mkldnn.is_available(), "mkldnn_enabled": torch.backends.mkldnn.enabled,
        "mkl_available": torch.backends.mkl.is_available(), "openmp_available": torch.backends.openmp.is_available(),
        "power_plan": _cmd(["powercfg", "/getactivescheme"]) if sys.platform == "win32" else None,
        "battery": None if bat is None else {"percent": bat.percent, "power_plugged": bat.power_plugged},
    }


# ------------------------------------------------------------------ timing
def time_block(model, x, n):
    """n timed forward passes (ns each); caller holds inference_mode and has disabled gc."""
    out = np.empty(n, dtype=np.int64)
    for i in range(n):
        t0 = time.perf_counter_ns()
        model(x)
        out[i] = time.perf_counter_ns() - t0
    return out


def run_session(models, x, warmup, timed, rounds, order_seed, anchor="dense"):
    """Interleaved timing of `models` (name -> eval model) on input x.

    Each model gets `warmup` untimed passes, then `timed` passes split evenly over `rounds` rounds. Every round
    starts with an anchor block of the same size on models[anchor] (recorded as role "anchor"), followed by every
    model (anchor included, role "model") in an order drawn from numpy default_rng(order_seed).
    Returns rows (round, position, model, role, pass, ns).
    """
    import torch
    assert timed % rounds == 0
    per = timed // rounds
    rng = np.random.default_rng(order_seed)
    names = list(models)
    rows = []
    with torch.inference_mode():
        for n in names:
            models[n].eval()
            for _ in range(warmup):
                models[n](x)
        gc.collect()
        gc.disable()
        try:
            for r in range(rounds):
                order = [names[i] for i in rng.permutation(len(names))]
                plan = [(anchor, "anchor")] + [(n, "model") for n in order]
                for pos, (n, role) in enumerate(plan):
                    ns = time_block(models[n], x, per)
                    rows += [(r, pos, n, role, i, int(v)) for i, v in enumerate(ns)]
        finally:
            gc.enable()
    return rows


def describe(ns_values, batch):
    v = np.asarray(ns_values, float) / 1e6                  # ms
    q1, med, q3, p95 = np.percentile(v, [25, 50, 75, 95])
    return {"n": int(len(v)), "median_ms": float(med), "mean_ms": float(v.mean()), "p95_ms": float(p95),
            "iqr_ms": float(q3 - q1), "min_ms": float(v.min()), "throughput_img_s": float(batch * 1000.0 / med)}
