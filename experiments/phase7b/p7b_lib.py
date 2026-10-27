"""Phase 7B library (results/phase7b_green_pruning/phase7b_design.md).

Structured space, counts, memory, the Paper-3 latency table and the Phase-7A energy backend. Earlier phases are
imported read only; nothing here writes outside results/phase7b_green_pruning or checkpoints/phase7b.
"""
import hashlib
import itertools
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
HERE = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
RESULTS = os.path.join(ROOT, "results", "phase7b_green_pruning")
CKPT = os.path.join(ROOT, "checkpoints", "phase7b")
PAPER3 = os.path.join(ROOT, "experiments", "paper3")
PHASE7A = os.path.join(ROOT, "experiments", "phase7a")
DESIGN = os.path.join(RESULTS, "phase7b_design.md")
FINAL_PREREG = os.path.join(RESULTS, "phase7b_final_preregistration.json")

ARCH, SETTING = "simplecnn", "cifar10_simplecnn"
POLICIES = [tuple(p) for p in itertools.product(range(6), repeat=4)]
RATIO = [0.0, 0.1, 0.2, 0.3, 0.4, 0.6]
RECOVERY_SEED, FINAL_EPOCHS = 600, 5
CPU = 2
ACC_TOL_PP = 0.5
BYTES_PER_PARAM = 4                 # float32 everywhere (checked on the built models)


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_module(name, folder):
    import importlib.util
    path = os.path.join(folder, f"{name}.py")
    spec = importlib.util.spec_from_file_location(f"p7b_{os.path.basename(folder)}_{name}", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def p3():
    if PAPER3 not in sys.path:
        sys.path.insert(0, PAPER3)
    import p3_lib
    return p3_lib


def key(policy):
    return "".join(str(int(a)) for a in policy)


def counts(policy, arch=ARCH):
    """(MACs, params, widths) from the verified Paper-3 closed form."""
    from src.pruning import structured_units as SU
    macs, params = SU.analytic(arch, list(policy), 10)
    return macs, params, SU.widths(arch, list(policy))


def counts_ratios(ratios, arch=ARCH):
    """(MACs, params, widths) for arbitrary per-unit ratios (Uniform, LAMP): keep max(1, round((1 - r) C))."""
    P7 = energy_meter()
    w = P7.widths(arch, list(ratios))
    macs, params = P7.analytic(arch, w)
    return macs, params, w


def table():
    """{policy: dict(macs, params, memory_bytes, widths, mac_reduction, param_reduction, latency_ms)} for all 1,296
    policies; latency is the measured Paper-3 batch-1 table (configuration A)."""
    import csv
    lat = {}
    with open(os.path.join(ROOT, "results", "paper3", f"h2_latency_table_{ARCH}_c10.csv"), encoding="utf-8") as f:
        for r in csv.DictReader(f):
            lat[tuple(json.loads(r["policy"]))] = float(r["latency_ms"])
    m0, p0, _ = counts((0, 0, 0, 0))
    out = {}
    for p in POLICIES:
        macs, params, w = counts(p)
        out[p] = {"macs": macs, "params": params, "memory_bytes": params * BYTES_PER_PARAM, "widths": w,
                  "mac_reduction": 1 - macs / m0, "param_reduction": 1 - params / p0, "latency_ms": lat[p]}
    return out


def energy_meter():
    """The Phase-7A Windows Energy Meter (RAPL) reader, imported read only."""
    return load_module("p7_lib", PHASE7A)


def slim_reference(policy):
    ref, sha = p3().reference(SETTING)
    from src.pruning import structured_units as SU
    m, kept = SU.slim(ref.eval(), ARCH, list(policy))
    return m.eval(), kept, sha
