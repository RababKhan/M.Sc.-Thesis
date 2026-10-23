"""Paper 3 library: paths, frozen references, calibration policies (results/paper3/h0_protocol.md).

Earlier phases are read only. phase3_common is the only module taken from an earlier phase; its folder is appended
to sys.path (never placed first), so a Paper-3 script name can never resolve to an earlier phase's script.
"""
import importlib.util
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
HERE = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
if os.path.join(ROOT, "experiments", "phase3") not in sys.path:
    sys.path.append(os.path.join(ROOT, "experiments", "phase3"))

RESULTS = os.path.join(ROOT, "results", "paper3")
P5_SPLITS = os.path.join(ROOT, "results", "phase5", "splits")

# H0 calibration policies on CIFAR-10 SimpleCNN, action vectors over (conv1, conv2, conv3, fc1),
# actions -> {0, 10, 20, 30, 40, 60}% of the unit's channels. Fixed before any timing.
CALIBRATION = {
    "dense": [0, 0, 0, 0],
    "light": [1, 1, 1, 1],
    "moderate": [3, 3, 3, 3],
    "aggressive": [5, 5, 5, 5],
    "fc_heavy": [0, 0, 0, 5],
    "conv_heavy": [0, 5, 5, 0],
}
SIGNAL_MODEL = "aggressive"            # the "substantially slimmed" model of the signal-to-noise rule

# H0 timing configurations: (batch, intra-op threads, inter-op threads, logical CPUs pinned)
# The i7-1260P is hybrid: logical 0-7 = 4 P-cores x 2 hyper-threads, 8-15 = 8 E-cores. Pinning keeps every timed
# configuration on P-cores; the 1-thread configuration avoids core 0 (most interrupt handling).
CONFIGS = {
    "A": {"batch": 1, "threads": 1, "interop": 1, "cpus": [2], "warmup": 50, "timed": 300, "rounds": 5},
    "B": {"batch": 1, "threads": 4, "interop": 1, "cpus": [0, 2, 4, 6], "warmup": 50, "timed": 300, "rounds": 5},
    "C": {"batch": 128, "threads": 4, "interop": 1, "cpus": [0, 2, 4, 6], "warmup": 10, "timed": 50, "rounds": 5},
}
SELECTION_ORDER = ["A", "C", "B"]
SESSIONS = 5
SESSION_SEED = 20261023
MIN_SNR = 5.0
MIN_RELATIVE_REDUCTION = 0.10
MAX_CPU_LOAD = 15.0                    # % over 5 s, before and after every session process


def phase_module(name, folder=HERE):
    """Load a script by file path (never by bare name: earlier phases reuse script names)."""
    path = os.path.join(folder, f"{name}.py")
    spec = importlib.util.spec_from_file_location(f"p3_{os.path.basename(folder)}_{name}", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    assert os.path.samefile(mod.__file__, path)
    return mod


def reference(setting):
    """Frozen dense reference (sha256 asserted against the Phase-2 record) and its hash."""
    import phase3_common as P3
    return P3.load_reference(setting)


def val_rl(dataset):
    """VAL-RL images and labels of the Phase-5 split (validation data only; the test set is never touched)."""
    import numpy as np
    from src import data as D
    b = D.DataBundle(dataset, train_images=False, rl_split=False)
    z = np.load(os.path.join(P5_SPLITS, f"{dataset}_phase5_split.npz"))
    pos = z["val_rl_positions"]
    return b.val_x[pos], b.val_y[pos], b
