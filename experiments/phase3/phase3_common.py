"""Phase 3 shared definitions: settings, frozen inputs and their verification (results/phase3_preregistration.md)."""
import itertools
import json
import os
import sys

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src import data as D, models as M, pruning as P  # noqa: E402
from src.utils import sha256_file  # noqa: E402

RESULTS = os.path.join(ROOT, "results")
AG = os.path.join(RESULTS, "architecture_generalization")
SB = os.path.join(RESULTS, "stronger_baselines")
LAND_DIR = os.path.join(AG, "phase3_landscapes")
RUNS_DIR = os.path.join(AG, "phase3_runs")
PRED_DIR = os.path.join(AG, "phase3_predictions")
AGENTS_DIR = os.path.join(ROOT, "checkpoints", "phase3_agents")
PREREG = os.path.join(RESULTS, "phase3_preregistration.md")
INPUTS_JSON = os.path.join(AG, "phase3_frozen_inputs.json")
SENS_CSV = os.path.join(AG, "phase3_sensitivity_vectors.csv")
DESTRUCTIVE_JSON = os.path.join(AG, "phase3_destructive_action_rules.json")

DATASETS = ["cifar10", "cifar100"]
ARCHS = ["simplecnn", "lenet5", "resnet8"]
SETTINGS = [f"{d}_{a}" for d in DATASETS for a in ARCHS]
CONDITIONS = ["P0", "P1", "P2", "P3"]
SEEDS = list(range(300, 320))
BETA = 0.50
LAMBDA_S = 0.01
TOTAL_TIMESTEPS = 2000          # -> 2,048 trained = 512 episodes of 4 steps (identical everywhere)
EPISODES = 512
THREADS = 1                     # every Phase-3 evaluation and PPO run
DESTRUCTIVE_ACTION = 4          # action index >= 4  <=>  ratio >= 40%
RANDOM_SEEDS = list(range(1000, 1010))
GRID = [10.0, 20.0, 30.0, 40.0, 50.0, 60.0]
POLICIES = [tuple(p) for p in itertools.product(range(6), repeat=4)]
REFERENCE_SHA = {
    "cifar10_simplecnn": "ca28f8345c23365ee7b92686e780ae0a9befa76c9d6bb9b3bb0b9e42272e111c",
}


def split(setting):
    d, a = setting.split("_")
    return d, a


def reference_path(setting):
    d, a = split(setting)
    if setting == "cifar10_simplecnn":
        return os.path.join(ROOT, "checkpoints", "cnn_baseline_FIXED.pth")
    return os.path.join(ROOT, "checkpoints", "phase2_dense", d, f"{a}_seed0.pth")


def phase2_reference_sha(setting):
    """sha256 recorded by Phase 2 for the seed-0 reference (dense run record)."""
    if setting in REFERENCE_SHA:
        return REFERENCE_SHA[setting]
    d, a = split(setting)
    rec = json.load(open(os.path.join(AG, "dense_runs", f"{d}_{a}_seed0.json"), encoding="utf-8"))
    assert rec["reference_model"] and rec["seed"] == 0
    return rec["checkpoint_sha256"]


def load_reference(setting):
    """The frozen Phase-2 reference model, after asserting its sha256."""
    d, a = split(setting)
    path = reference_path(setting)
    got = sha256_file(path)
    want = phase2_reference_sha(setting)
    if got != want:
        raise RuntimeError(f"{setting}: reference checkpoint sha256 {got} != Phase-2 record {want}")
    return M.load_checkpoint(path, a, D.DATASETS[d]["num_classes"]), got


def frozen_inputs():
    return json.load(open(INPUTS_JSON, encoding="utf-8"))


def landscape_json(setting):
    return os.path.join(LAND_DIR, f"{setting}.json")


def landscape_cache(setting):
    """{actions tuple: (V_RL accuracy, total sparsity)} with exact floats (JSON round-trip)."""
    rows = json.load(open(landscape_json(setting), encoding="utf-8"))["policies"]
    return {tuple(r["actions"]): (r["val_accuracy"], r["total_sparsity"]) for r in rows}


def run_id(setting, condition, seed):
    return f"{setting}_{condition}_seed{seed}"


def jobs():
    """All 480 runs in the fixed execution order (setting, seed, condition)."""
    return [(s, c, seed) for s in SETTINGS for seed in SEEDS for c in CONDITIONS]


def set_threads():
    torch.set_num_threads(THREADS)
    assert torch.get_num_threads() == THREADS


def policy_key(actions):
    return "".join(str(int(a)) for a in actions)


def pareto(sparsity, accuracy):
    """Non-dominated in (sparsity up, accuracy up); identical to evaluation/policy_landscape.py."""
    order = np.lexsort((-accuracy, -sparsity))
    efficient = np.zeros(len(sparsity), bool)
    best = -np.inf
    for i in order:
        if accuracy[i] > best + 1e-12:
            efficient[i] = True
            best = accuracy[i]
    return efficient
