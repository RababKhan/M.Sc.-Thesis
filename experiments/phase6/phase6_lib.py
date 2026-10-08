"""Phase 6 library: matched masks and resumable fixed-mask fine-tuning (results/phase6/phase6_preregistration.md).

Earlier phases are imported read only. A pruned model is the frozen dense reference with torch.nn.utils.prune
masks: the effective weight is weight_orig * weight_mask, so a masked weight is exactly 0 in every forward
pass, receives a zero gradient and (weight decay 0) is never updated; the mask buffers are not parameters.
After every epoch the loop still verifies that the mask buffers are unchanged and that the zero count equals
the initial one.
"""
import copy
import hashlib
import os
import sys
import time

import numpy as np
import psutil
import torch
import torch.nn as nn

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
for p in (ROOT, os.path.join(ROOT, "experiments", "phase3")):
    if p not in sys.path:
        sys.path.insert(0, p)

import phase3_common as P3  # noqa: E402  (frozen references and settings)
from src import data as D, pruning as P  # noqa: E402
from src.evaluation import accuracy_from, evaluate_accuracy, predictions  # noqa: E402
from src.utils import atomic_torch_save  # noqa: E402

RESULTS = os.path.join(ROOT, "results", "phase6")
RUNS = os.path.join(RESULTS, "runs")
REPRO = os.path.join(RESULTS, "reproduction_runs")
PRED = os.path.join(RESULTS, "predictions")
WORK = os.path.join(ROOT, "checkpoints", "phase6_finetuned", "_work")
CKPT = os.path.join(ROOT, "checkpoints", "phase6_finetuned")
PREREG = os.path.join(RESULTS, "phase6_preregistration.md")
CONFIG = os.path.join(RESULTS, "phase6_config.json")
P5 = os.path.join(ROOT, "results", "phase5")

SETTINGS = P3.SETTINGS
POLICY_SEEDS = list(range(500, 520))
FT_SEED_OFFSET = 100                     # fine-tuning seed = policy seed + 100  (600-619)
METHODS = ["ppo", "lamp", "global"]
ALL_METHODS = METHODS + ["dense"]
THREADS = 1
DATA_GENERATOR_OFFSET = 10 ** 6
PROTOCOL = {"optimizer": "Adam", "lr": 1e-4, "betas": [0.9, 0.999], "eps": 1e-8, "weight_decay": 0.0,
            "schedule": "CosineAnnealingLR(T_max=epochs, eta_min=0), stepped per epoch", "batch_size": 128,
            "augmentation": "random crop 32 (zero pad 4) + horizontal flip", "batchnorm": "train mode during fine-tuning "
            "(running statistics updated identically for every method); eval mode for every evaluation; no separate recalibration",
            "selection": "none: the model after the final epoch"}


def ft_seed(policy_seed):
    return policy_seed + FT_SEED_OFFSET


def bundle(dataset):
    """DataBundle with the 45,000 training images and the Phase-5 VAL-RL / VAL-SELECT split."""
    b = D.DataBundle(dataset, train_images=True, rl_split=False)
    z = np.load(os.path.join(P5, "splits", f"{dataset}_phase5_split.npz"))
    b.rl_positions, b.select_positions = z["val_rl_positions"], z["val_select_positions"]
    assert len(b.rl_positions) == len(b.select_positions) == 2500
    return b


# ------------------------------------------------------------------ matched masks
def build(reference, arch, method, policy):
    """The fixed-mask model of `method` at exactly the zero count of the PPO `policy` (dense: no mask)."""
    if method == "dense":
        return copy.deepcopy(reference)
    ppo = P.prune_actions(reference, arch, policy)
    if method == "ppo":
        return ppo
    out = P.BASELINES[method](reference, arch, P.total_sparsity(ppo))
    assert P.zero_count(out) == P.zero_count(ppo), f"{method}: zero count mismatch"
    return out


def mask_tensors(model):
    """[(layer name, mask)] over every Conv2d/Linear weight tensor (all-ones for unmasked tensors)."""
    return [(n, m.weight_mask if hasattr(m, "weight_mask") else torch.ones_like(m.weight)) for n, m in P.weight_layers(model)]


def mask_hash(model):
    h = hashlib.sha256()
    for n, mk in mask_tensors(model):
        h.update(n.encode())
        h.update(np.packbits(mk.detach().cpu().numpy().astype(bool).ravel()).tobytes())
    return h.hexdigest()


def layer_zeros(model):
    return {n: int((m.weight == 0).sum()) for n, m in P.weight_layers(model)}


def mask_violations(model):
    """Number of masked positions whose effective weight is not exactly zero (must be 0)."""
    return int(sum(((m.weight != 0) & (mk == 0)).sum() for (n, m), (_, mk) in zip(P.weight_layers(model), mask_tensors(model))))


# ------------------------------------------------------------------ fine-tuning
def make_optimizer(model, epochs):
    opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=PROTOCOL["lr"], betas=tuple(PROTOCOL["betas"]),
                           eps=PROTOCOL["eps"], weight_decay=PROTOCOL["weight_decay"])
    return opt, torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=0.0)


def evaluate_val(model, b):
    return evaluate_accuracy(model, b.rl_loader()), evaluate_accuracy(model, b.select_loader())


def finetune(model, b, seed, epochs, work_path, log=print, validate=True):
    """Fixed-mask fine-tuning for `epochs` epochs (resumable per epoch). Returns (history, resumed_from)."""
    torch.manual_seed(seed)
    gen = torch.Generator().manual_seed(seed + DATA_GENERATOR_OFFSET)
    opt, sched = make_optimizer(model, epochs)
    loss_fn = nn.CrossEntropyLoss()
    history, start = [], 0
    mask0, zeros0 = mask_hash(model), P.zero_count(model)
    if os.path.exists(work_path):
        st = torch.load(work_path, map_location="cpu", weights_only=False)
        assert st["mask_hash"] == mask0 and st["epochs"] == epochs and st["seed"] == seed, "work state belongs to another run"
        model.load_state_dict(st["model"])
        opt.load_state_dict(st["optimizer"])
        sched.load_state_dict(st["scheduler"])
        gen.set_state(st["generator"])
        history, start = st["history"], st["epoch"]
        log(f"  resumed after epoch {start}")
    for epoch in range(start + 1, epochs + 1):
        t0 = time.time()
        lr = opt.param_groups[0]["lr"]
        model.train()
        loss_sum, correct, seen = 0.0, 0, 0
        with b.training():                                   # TEST and VAL-SELECT raise inside
            for xb, yb in b.train_batches(gen, PROTOCOL["batch_size"]):
                opt.zero_grad(set_to_none=True)
                out = model(xb)
                loss = loss_fn(out, yb)
                if not torch.isfinite(loss):
                    raise FloatingPointError(f"non-finite loss in epoch {epoch}")
                loss.backward()
                opt.step()
                loss_sum += loss.item() * len(yb)
                correct += (out.argmax(1) == yb).sum().item()
                seen += len(yb)
        sched.step()
        train_s = time.time() - t0
        model.eval()
        assert mask_hash(model) == mask0, "mask buffers changed"
        assert mask_violations(model) == 0, "a masked weight is non-zero"
        zeros = P.zero_count(model)
        assert zeros >= zeros0, "zero count decreased"
        vrl, vsel = evaluate_val(model, b) if validate else (None, None)
        history.append({"epoch": epoch, "lr": lr, "train_loss": loss_sum / seen, "train_accuracy": 100.0 * correct / seen,
                        "val_rl_accuracy": vrl, "val_select_accuracy": vsel, "zero_count": zeros,
                        "train_seconds": round(train_s, 2), "epoch_seconds": round(time.time() - t0, 2)})
        atomic_torch_save({"epoch": epoch, "epochs": epochs, "seed": seed, "mask_hash": mask0, "model": model.state_dict(),
                           "optimizer": opt.state_dict(), "scheduler": sched.state_dict(), "generator": gen.get_state(),
                           "history": history}, work_path)
        log(f"  epoch {epoch}/{epochs}  loss {history[-1]['train_loss']:.4f}  ({history[-1]['epoch_seconds']:.0f}s)")
    # identical for every run that consumed the same data stream (same seed, same number of epochs)
    history[-1]["data_stream_sha256"] = hashlib.sha256(gen.get_state().numpy().tobytes()).hexdigest()
    return history, start


def test_predictions(model, b, record):
    """TEST predictions of a fixed model (the bundle is frozen for this model first)."""
    model.eval()
    b.freeze(record)
    preds, labels = predictions(model, b.test_loader())
    return preds.numpy().astype(np.int16), labels.numpy().astype(np.int16), accuracy_from(preds, labels)


def peak_memory_mb():
    mi = psutil.Process().memory_info()
    return round(getattr(mi, "peak_wset", mi.rss) / 2 ** 20, 1)
