"""Seeding, hashing and provenance shared by every Phase-2+ runner."""
import hashlib
import json
import os
import platform
import random
import subprocess
import sys

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ------------------------------------------------------------------ seeding
def set_seed(seed, threads=None):
    """Seed Python, NumPy and the global torch generator; optionally fix the thread count.

    Training code in this package draws its randomness from explicit
    torch.Generator objects wherever it can, so the global generator is only
    used for weight initialisation and by third-party code (SB3).
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if threads is not None:
        torch.set_num_threads(threads)


# ------------------------------------------------------------------ hashing
def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def sha256_json(obj):
    """Hash of a JSON-serialisable object in its compact default encoding.

    json.dumps(list_of_ints) is the encoding the historical val_indices hash
    (9d3648af...) was computed with, so the two are directly comparable.
    """
    return hashlib.sha256(json.dumps(obj).encode()).hexdigest()


def sha256_array(a):
    a = np.ascontiguousarray(a)
    h = hashlib.sha256()
    h.update(str(a.dtype).encode() + str(a.shape).encode())
    h.update(a.tobytes())
    return h.hexdigest()


def sha256_state_dict(state_dict):
    """Content hash of a state dict, independent of the pickle container."""
    h = hashlib.sha256()
    for k in sorted(state_dict):
        t = state_dict[k].detach().cpu().contiguous()
        h.update(k.encode() + str(t.dtype).encode() + str(tuple(t.shape)).encode())
        h.update(t.numpy().tobytes())
    return h.hexdigest()


# ------------------------------------------------------------------ provenance
def git_state():
    def run(*args):
        return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    return {"commit": run("rev-parse", "HEAD"), "dirty_paths": run("status", "--porcelain").splitlines()}


def cpu_name():
    if sys.platform == "win32":
        import winreg
        key = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DESCRIPTION\System\CentralProcessor\0")
        return winreg.QueryValueEx(key, "ProcessorNameString")[0].strip()
    return platform.processor()


def environment():
    import torchvision
    out = {"python": platform.python_version(), "torch": torch.__version__, "torchvision": torchvision.__version__,
           "numpy": np.__version__, "cpu": cpu_name(), "logical_cpus": os.cpu_count(),
           "torch_threads": torch.get_num_threads(), "cuda_available": torch.cuda.is_available(),
           "os": platform.platform()}
    from importlib.metadata import PackageNotFoundError, version
    for name in ("stable-baselines3", "sb3-contrib", "gymnasium", "ptflops", "pandas", "psutil"):
        try:
            out[name] = version(name)
        except PackageNotFoundError:
            pass
    return out


def atomic_torch_save(obj, path):
    """Write-then-rename, so a power cut never leaves a half-written checkpoint under the final name."""
    tmp = path + ".tmp"
    torch.save(obj, tmp)
    os.replace(tmp, path)


def atomic_json_dump(obj, path):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=1, default=str)
    os.replace(tmp, path)
