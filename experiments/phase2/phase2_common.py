"""Shared constants and provenance for the Phase-2 runners (see results/cpu_dense_training_protocol.md)."""
import glob
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.data import SPLIT_FILES  # noqa: E402
from src.utils import environment, git_state, sha256_file  # noqa: E402

RESULTS = os.path.join(ROOT, "results")
AG = os.path.join(RESULTS, "architecture_generalization")
DENSE_RUNS = os.path.join(AG, "dense_runs")
DENSE_CKPT = os.path.join(ROOT, "checkpoints", "phase2_dense")
WORK = os.path.join(DENSE_CKPT, "_work")
BENCH_CSV = os.path.join(AG, "cpu_epoch_benchmark.csv")
TRAIN_PROTOCOL_MD = os.path.join(RESULTS, "cpu_dense_training_protocol.md")
BUDGET_MD = os.path.join(RESULTS, "cpu_runtime_budget.md")
MASTER_MD = os.path.join(RESULTS, "rocksolid_cpu_master_protocol.md")
FIXED = os.path.join(ROOT, "checkpoints", "cnn_baseline_FIXED.pth")
FIXED_SHA = "ca28f8345c23365ee7b92686e780ae0a9befa76c9d6bb9b3bb0b9e42272e111c"

EPOCHS = 30
SEEDS = [0, 1, 2]
BUDGET_HOURS = 2.0
STABLE_LR, FALLBACK_LR = 1e-3, 3e-4


def protocol(lr=STABLE_LR):
    return {"optimizer": "Adam", "lr": lr, "betas": [0.9, 0.999], "eps": 1e-8, "weight_decay": 0.0,
            "schedule": "CosineAnnealingLR(T_max=epochs, eta_min=0), stepped per epoch", "epochs": EPOCHS,
            "batch_size": 128, "augmentation": "random crop 32 (zero pad 4) + horizontal flip",
            "selection": "max full-validation accuracy, ties -> earliest epoch", "early_stopping": None}


def code_hashes():
    files = sorted(glob.glob(os.path.join(ROOT, "src", "**", "*.py"), recursive=True))
    files += sorted(glob.glob(os.path.join(ROOT, "experiments", "phase2", "*.py")))
    return {os.path.relpath(p, ROOT).replace("\\", "/"): sha256_file(p) for p in files}


def provenance(dataset):
    split = SPLIT_FILES[dataset]
    return {"git": git_state(), "environment": environment(),
            "protocol_sha256": {os.path.basename(p): sha256_file(p) for p in (TRAIN_PROTOCOL_MD, BUDGET_MD, MASTER_MD)},
            "split_file": os.path.relpath(split, ROOT).replace("\\", "/"),
            "split_sha256": sha256_file(split) if os.path.exists(split) else None,
            "code_sha256": code_hashes()}
