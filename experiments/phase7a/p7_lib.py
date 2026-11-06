"""Phase 7A library: sustainable-computing evaluation of the adaptive pruning policy (results/phase7a_sustainability).

Deployment realization, not the thesis models themselves: the PPO policies of Phases 5-6 were unstructured masks,
which do not reduce dense CPU work. Here each method's per-unit pruning RATIO is realized as structured channel
removal with the validated Paper-3 slimming code (src/pruning/structured_units.py, unchanged), recovered with the
frozen Phase-6 protocol, and measured for CPU energy (Intel RAPL package counter via the Windows Energy Meter
Interface) and latency. Earlier phases and Paper 3 are read only.
"""
import ctypes
import hashlib
import json
import os
import sys
from ctypes import wintypes

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
HERE = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
RESULTS = os.path.join(ROOT, "results", "phase7a_sustainability")
CKPT = os.path.join(ROOT, "checkpoints", "phase7a")
PAPER3 = os.path.join(ROOT, "experiments", "paper3")
PREREG = os.path.join(RESULTS, "preregistration.json")

SETTINGS = {"simplecnn": "cifar10_simplecnn", "resnet8": "cifar10_resnet8"}
METHODS = ["dense", "ppo", "uniform", "lamp"]
LABELS = {"dense": "Dense", "ppo": "PPO-Structured", "uniform": "Uniform-Structured", "lamp": "LAMP-Structured"}
# thesis pruning units (results/architecture_generalization/cpu_prunable_units.csv) -> weight tensors
THESIS_UNITS = {"simplecnn": [["features.0"], ["features.3"], ["features.6"], ["classifier.1"]],
                "resnet8": [["conv1"], ["layer1.conv1", "layer1.conv2"],
                            ["layer2.conv1", "layer2.conv2", "layer2.shortcut.0"],
                            ["layer3.conv1", "layer3.conv2", "layer3.shortcut.0"]]}
ACTION_RATIO = [0.0, 0.1, 0.2, 0.3, 0.4, 0.6]
UNIFORM_GRID = [i / 1000 for i in range(0, 901)]        # candidate common ratios for the MAC-matched uniform method
RECOVERY_SEED, RECOVERY_EPOCHS = 600, 5                  # frozen Phase-6 / Paper-3 recovery protocol
CPU, THREADS = 2, 1                                      # Paper-3 configuration A: logical CPU 2 (P-core), 1 thread
ACC_TOLERANCE_PP = 0.5


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def p3():
    """Paper-3 library (paths, frozen dense references)."""
    sys.path.insert(0, PAPER3)
    import p3_lib
    return p3_lib


def load_module(name, folder):
    import importlib.util
    path = os.path.join(folder, f"{name}.py")
    spec = importlib.util.spec_from_file_location(f"p7_{os.path.basename(folder)}_{name}", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ------------------------------------------------------------------ structured realization from per-unit ratios
def widths(arch, ratios):
    from src.pruning import structured_units as SU
    return [SU.keep_count(c, r) for c, r in zip(SU.BASE_WIDTHS[arch], ratios)]


def analytic(arch, w, num_classes=10):
    """(conv/linear MACs per image, total parameters) from the slimmed widths; same closed form as
    structured_units.analytic, written for arbitrary widths."""
    if arch == "simplecnn":
        a, b, c, h = w
        macs = 32 * 32 * a * 3 * 9 + 16 * 16 * b * a * 9 + 8 * 8 * c * b * 9 + c * 16 * h + h * num_classes
        params = (a * 27 + a) + (b * a * 9 + b) + (c * b * 9 + c) + (c * 16 * h + h) + (h * num_classes + num_classes)
        return macs, params
    g, i1, i2, i3 = w
    macs = (32 * 32 * g * 3 * 9 + 32 * 32 * i1 * g * 9 + 32 * 32 * g * i1 * 9 + 16 * 16 * i2 * g * 9 + 16 * 16 * 32 * i2 * 9
            + 16 * 16 * 32 * g + 8 * 8 * i3 * 32 * 9 + 8 * 8 * 64 * i3 * 9 + 8 * 8 * 64 * 32 + 64 * num_classes)
    conv = g * 27 + i1 * g * 9 + g * i1 * 9 + i2 * g * 9 + 32 * i2 * 9 + 32 * g + i3 * 32 * 9 + 64 * i3 * 9 + 64 * 32
    bn = 2 * (g + i1 + g + i2 + 32 + 32 + i3 + 64 + 64)
    return macs, conv + bn + 64 * num_classes + num_classes


def slim(ref, arch, ratios):
    """Physically compacted copy of the dense reference for per-unit ratios (validated Paper-3 code paths)."""
    from src.pruning import structured as SP
    from src.pruning import structured_units as SU
    if arch == "simplecnn":
        m, kept = SP.simplecnn_structured(ref, list(ratios))
        return m.eval(), dict(zip(SU.UNITS[arch], kept.values()))
    return SU.resnet8_slim(ref, list(ratios))


def policies():
    return json.load(open(os.path.join(RESULTS, "selected_policies.json"), encoding="utf-8"))


def ckpt_path(arch, method):
    return os.path.join(CKPT, f"{SETTINGS[arch]}_{method}.pth")


def load_recovered(arch, method):
    """A recovered, physically compacted model: skeleton from the frozen reference, then the frozen state dict."""
    import torch
    rec = policies()["settings"][SETTINGS[arch]]["methods"][method]
    ref, _ = p3().reference(SETTINGS[arch])
    m, _ = slim(ref, arch, rec["structured_ratios"])
    path = ckpt_path(arch, method)
    state = torch.load(path, map_location="cpu", weights_only=True)
    m.load_state_dict(state)
    return m.eval()


# ------------------------------------------------------------------ energy: Windows Energy Meter Interface (RAPL) via PDH
class _Raw(ctypes.Structure):
    _fields_ = [("CStatus", wintypes.DWORD), ("TimeStamp", wintypes.FILETIME), ("FirstValue", ctypes.c_longlong),
                ("SecondValue", ctypes.c_longlong), ("MultiCount", wintypes.DWORD)]


class EnergyMeter:
    """Cumulative RAPL energy counters. Energy is reported in picowatt-hours (verified against the counter's own
    Power value in mW), Time in milliseconds; the hardware value refreshes every ~1 ms."""
    DOMAINS = ("RAPL_Package0_PKG", "RAPL_Package0_PP0")
    PWH_TO_J = 3600e-12

    def __init__(self):
        self.pdh = ctypes.WinDLL("pdh.dll")
        self.q = ctypes.c_void_p()
        assert self.pdh.PdhOpenQueryW(None, None, ctypes.byref(self.q)) == 0
        self.h = {}
        for d in self.DOMAINS:
            for c in ("Energy", "Time"):
                h = ctypes.c_void_p()
                st = self.pdh.PdhAddEnglishCounterW(self.q, f"\\Energy Meter({d})\\{c}", None, ctypes.byref(h))
                if st != 0:
                    raise RuntimeError(f"Energy Meter counter {d}/{c} unavailable (PDH status {st & 0xffffffff:#x})")
                self.h[(d, c)] = h

    def read(self):
        assert self.pdh.PdhCollectQueryData(self.q) == 0
        out = {}
        for k, h in self.h.items():
            raw, t = _Raw(), wintypes.DWORD()
            assert self.pdh.PdhGetRawCounterValue(h, ctypes.byref(t), ctypes.byref(raw)) == 0
            out[f"{k[0]}_{k[1]}"] = int(raw.FirstValue)
        return out

    @classmethod
    def joules(cls, a, b, domain="RAPL_Package0_PKG"):
        return (b[f"{domain}_Energy"] - a[f"{domain}_Energy"]) * cls.PWH_TO_J


def on_ac_power():
    """True only if Windows reports the AC adapter online (root/wmi BatteryStatus.PowerOnline)."""
    import subprocess
    out = subprocess.run(["powershell", "-NoProfile", "-Command",
                          "(Get-CimInstance -Namespace root/wmi -ClassName BatteryStatus).PowerOnline"],
                         capture_output=True, text=True, timeout=60).stdout.strip().splitlines()
    return bool(out) and all(s.strip() == "True" for s in out)


def power_scheme():
    import subprocess
    return subprocess.run(["powercfg", "/getactivescheme"], capture_output=True, text=True, timeout=60).stdout.strip()
