"""CPU efficiency measurements (protocol: results/cpu_efficiency_protocol.md).

Counters
  conv_linear_macs   analytic, from forward hooks: Conv2d out_elements * (C_in/groups) * k_h * k_w,
                     Linear in * out, per image. The quantity usually reported as "MACs".
  flops              torch.utils.flop_counter.FlopCounterMode (pinned by the torch version):
                     2 FLOPs per multiply-accumulate of convolutions and matrix products.
                     Must equal 2 * conv_linear_macs (asserted).
  ptflops_macs       ptflops 0.7.5, pytorch backend: conv/linear MACs plus BatchNorm, ReLU,
                     pooling and other element-wise work, per image.
  nominal_sparse_macs  theoretical only: each Conv2d/Linear layer's MACs scaled by its fraction
                     of nonzero weights. Dense CPU kernels do not realise this; never reported
                     as a speed-up.
Storage
  raw bytes: torch.save of the state dict after pruning masks are folded in; gzip bytes: the
  same file compressed with gzip level 9 (deterministic, mtime 0).
Latency / peak RAM: see latency_samples() and peak_inference_ram() below.
"""
import gzip
import io
import json
import os
import subprocess
import sys
import time

import numpy as np
import torch
import torch.nn as nn

from src.pruning import make_permanent
import copy


def conv_linear_macs(model, input_shape=(1, 3, 32, 32)):
    model = model.eval()
    per_layer = {}
    hooks = []

    def hook(name):
        def fn(mod, inp, out):
            if isinstance(mod, nn.Conv2d):
                k = mod.in_channels // mod.groups * mod.kernel_size[0] * mod.kernel_size[1]
                per_layer[name] = out.numel() // out.shape[0] * k
            else:
                per_layer[name] = mod.in_features * mod.out_features
        return fn

    for n, m in model.named_modules():
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            hooks.append(m.register_forward_hook(hook(n)))
    with torch.inference_mode():
        model(torch.zeros(input_shape))
    for h in hooks:
        h.remove()
    return sum(per_layer.values()), per_layer


def torch_flops(model, input_shape=(1, 3, 32, 32)):
    from torch.utils.flop_counter import FlopCounterMode
    counter = FlopCounterMode(display=False)
    with torch.no_grad(), counter:
        model.eval()(torch.zeros(input_shape))
    return counter.get_total_flops() // input_shape[0]


def ptflops_macs(model, input_res=(3, 32, 32)):
    from ptflops import get_model_complexity_info
    with open(os.devnull, "w") as sink:
        macs, params = get_model_complexity_info(copy.deepcopy(model).eval(), input_res, as_strings=False,
                                                 print_per_layer_stat=False, backend="pytorch", ost=sink)
    return int(macs), int(params)


def nominal_sparse_macs(model, input_shape=(1, 3, 32, 32)):
    _, per_layer = conv_linear_macs(model, input_shape)
    named = dict(model.named_modules())
    total = 0.0
    for n, macs in per_layer.items():
        w = named[n].weight
        total += macs * float((w != 0).sum()) / w.numel()
    return total


def storage_bytes(model):
    """(raw, gzip-9) bytes of the saved state dict with masks folded in."""
    m = make_permanent(copy.deepcopy(model))
    buf = io.BytesIO()
    torch.save(m.state_dict(), buf)
    raw = buf.getvalue()
    gz = io.BytesIO()
    with gzip.GzipFile(fileobj=gz, mode="wb", compresslevel=9, mtime=0) as f:
        f.write(raw)
    return len(raw), len(gz.getvalue())


def nonzero_parameters(model):
    return int(sum(int((p != 0).sum()) for p in make_permanent(copy.deepcopy(model)).parameters()))


# ------------------------------------------------------------------ latency
def cpu_load(interval=1.0):
    import psutil
    return psutil.cpu_percent(interval=interval)


def latency_samples(models, batch, threads=4, warmup=100, iterations=500, rounds=5, seed=0):
    """Interleaved CPU latency (ms per forward pass) for several models at one batch size.

    eval() + torch.inference_mode(); fixed thread count; fixed random input;
    `warmup` untimed passes per model, then `iterations` timed passes per model
    split over `rounds` rounds in which the model order is rotated.
    """
    torch.set_num_threads(threads)
    x = torch.randn((batch, 3, 32, 32), generator=torch.Generator().manual_seed(seed))
    names = list(models)
    samples = {n: [] for n in names}
    with torch.inference_mode():
        for n in names:
            models[n].eval()
            for _ in range(warmup):
                models[n](x)
        per_round = iterations // rounds
        for r in range(rounds):
            order = names[r % len(names):] + names[:r % len(names)]
            for n in order:
                for _ in range(per_round):
                    t0 = time.perf_counter()
                    models[n](x)
                    samples[n].append(1000.0 * (time.perf_counter() - t0))
    return samples


def summarize_latency(values, rng_seed=0, boots=5000):
    v = np.asarray(values, float)
    rng = np.random.default_rng(rng_seed)
    med = np.median(v[rng.integers(0, len(v), (boots, len(v)))], axis=1)
    return {"median_ms": float(np.median(v)), "mean_ms": float(v.mean()), "sd_ms": float(v.std(ddof=1)),
            "median_ci95_low": float(np.percentile(med, 2.5)), "median_ci95_high": float(np.percentile(med, 97.5)),
            "n": int(len(v))}


# ------------------------------------------------------------------ peak RAM (fresh process)
_PEAK_SCRIPT = r"""
import json, sys, torch, psutil
sys.path.insert(0, sys.argv[1])
from src.models import load_checkpoint
arch, ncls, path, batch, passes, threads = sys.argv[2], int(sys.argv[3]), sys.argv[4], int(sys.argv[5]), int(sys.argv[6]), int(sys.argv[7])
torch.set_num_threads(threads)
model = load_checkpoint(path, arch, ncls)
x = torch.randn(batch, 3, 32, 32)
proc = psutil.Process()
before = proc.memory_info()
with torch.inference_mode():
    for _ in range(passes):
        model(x)
after = proc.memory_info()
peak = getattr(after, "peak_wset", None)
print(json.dumps({"rss_after_load": before.rss, "rss_after": after.rss,
                  "peak_before_inference": getattr(before, "peak_wset", None), "peak": peak}))
"""


def peak_inference_ram(root, arch, num_classes, checkpoint, batch, passes=20, threads=4):
    """Process-level memory of inference in a fresh interpreter (Windows peak working set).

    Reported: peak working set of the whole process (Python + torch + model + activations),
    and the increase of the peak caused by the inference passes alone.
    """
    out = subprocess.run([sys.executable, "-c", _PEAK_SCRIPT, root, arch, str(num_classes), checkpoint,
                          str(batch), str(passes), str(threads)], capture_output=True, text=True, check=True)
    r = json.loads(out.stdout.strip().splitlines()[-1])
    r["inference_peak_increase"] = (r["peak"] - r["peak_before_inference"]) if r["peak"] else None
    return r
