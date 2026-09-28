"""Dense training under the pre-registered Phase-2 protocol (results/cpu_dense_training_protocol.md).

Resumable at epoch granularity: after every epoch the complete training state
(model, optimiser, scheduler, data generator, history, best-so-far) is written
atomically, so a power cut costs at most one epoch and the resumed run is
bit-identical to an uninterrupted one at the same thread count.

Randomness
  torch.manual_seed(seed)          weight initialisation (nothing else uses the
                                   global generator during training)
  torch.Generator(seed + 10**6)    epoch permutations and augmentation offsets
Checkpoint selection
  after each epoch the full 5,000-image validation accuracy is computed; the
  selected checkpoint is the epoch with the highest validation accuracy
  (ties: earliest epoch). No early stopping. The test set is locked throughout
  (DataBundle.training()) and is read once, after the selected checkpoint is
  written and hashed.
"""
import math
import os
import time

import torch
import torch.nn as nn

from src.data import eval_loader
from src.evaluation import evaluate_accuracy, mean_loss_accuracy, predictions, accuracy_from
from src.models import build_model, count_parameters
from src.utils import atomic_torch_save, sha256_file, sha256_state_dict

DATA_GENERATOR_OFFSET = 10 ** 6


def make_optimizer(model, protocol):
    return torch.optim.Adam(model.parameters(), lr=protocol["lr"], betas=tuple(protocol["betas"]),
                            weight_decay=protocol["weight_decay"])


def make_scheduler(optimizer, protocol):
    return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=protocol["epochs"], eta_min=0.0)


def train_one_epoch(model, bundle, optimizer, generator, batch_size):
    model.train()
    loss_fn = nn.CrossEntropyLoss()
    loss_sum, correct, seen, nonfinite = 0.0, 0, 0, False
    for xb, yb in bundle.train_batches(generator, batch_size):
        optimizer.zero_grad(set_to_none=True)
        out = model(xb)
        loss = loss_fn(out, yb)
        if not torch.isfinite(loss):
            nonfinite = True
            break
        loss.backward()
        optimizer.step()
        loss_sum += loss.item() * len(yb)
        correct += (out.argmax(1) == yb).sum().item()
        seen += len(yb)
    return {"train_loss": loss_sum / max(seen, 1), "train_accuracy": 100.0 * correct / max(seen, 1),
            "nonfinite": nonfinite, "batches_seen": seen}


def run_dense_training(arch, bundle, seed, protocol, work_path, final_path, threads, log=print, evaluate_test=True):
    """Train (or resume) one dense run; returns the run record. Writes final_path when complete.

    evaluate_test=False is for implementation checks only: the test set is not read.
    """
    torch.set_num_threads(threads)
    epochs = protocol["epochs"]
    model = build_model(arch, bundle.num_classes, seed=seed)
    init_hash = sha256_state_dict(model.state_dict())
    optimizer = make_optimizer(model, protocol)
    scheduler = make_scheduler(optimizer, protocol)
    generator = torch.Generator().manual_seed(seed + DATA_GENERATOR_OFFSET)
    history, best, start_epoch = [], None, 0

    if os.path.exists(work_path):
        state = torch.load(work_path, map_location="cpu", weights_only=False)
        if state["threads"] != threads:
            raise RuntimeError(f"resume with {threads} threads, run started with {state['threads']}")
        assert state["init_hash"] == init_hash, "initialisation differs from the interrupted run"
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        generator.set_state(state["generator"])
        history, best, start_epoch = state["history"], state["best"], state["epoch"]
        log(f"  resumed after epoch {start_epoch}")

    vx, vy = bundle.val_tensors()
    with bundle.training():
        for epoch in range(start_epoch + 1, epochs + 1):
            t0 = time.time()
            lr = optimizer.param_groups[0]["lr"]
            stats = train_one_epoch(model, bundle, optimizer, generator, protocol["batch_size"])
            if stats["nonfinite"]:
                raise FloatingPointError(f"non-finite training loss in epoch {epoch}")
            scheduler.step()
            train_seconds = time.time() - t0
            val_loss, _ = mean_loss_accuracy(model, vx, vy)
            val_acc = evaluate_accuracy(model, eval_loader(vx, vy))
            row = {"epoch": epoch, "lr": lr, **{k: v for k, v in stats.items() if k != "nonfinite"},
                   "val_loss": val_loss, "val_accuracy": val_acc, "train_seconds": round(train_seconds, 2),
                   "epoch_seconds": round(time.time() - t0, 2)}
            history.append(row)
            if best is None or val_acc > best["val_accuracy"]:
                best = {"epoch": epoch, "val_accuracy": val_acc, "val_loss": val_loss,
                        "state_dict": {k: v.clone() for k, v in model.state_dict().items()}}
            atomic_torch_save({"epoch": epoch, "threads": threads, "init_hash": init_hash,
                               "model": model.state_dict(), "optimizer": optimizer.state_dict(),
                               "scheduler": scheduler.state_dict(), "generator": generator.get_state(),
                               "history": history, "best": best}, work_path)
            log(f"  epoch {epoch:>3}/{epochs}  loss {stats['train_loss']:.4f}  train {stats['train_accuracy']:.2f}%  "
                f"val {val_acc:.2f}%  ({row['epoch_seconds']:.0f}s)")

    # ---- freeze the selected checkpoint, then (and only then) read the test set once
    selected = {"state_dict": best["state_dict"], "arch": arch, "dataset": bundle.dataset,
                "num_classes": bundle.num_classes, "seed": seed, "selected_epoch": best["epoch"],
                "val_accuracy": best["val_accuracy"], "protocol": protocol}
    atomic_torch_save(selected, final_path)
    ckpt_sha = sha256_file(final_path)
    bundle.freeze({"checkpoint": final_path, "sha256": ckpt_sha})
    model.load_state_dict(best["state_dict"])
    model.eval()
    if evaluate_test:
        preds, labels = predictions(model, bundle.test_loader())
        test_acc = accuracy_from(preds, labels)
    else:
        preds, test_acc = torch.zeros(0, dtype=torch.long), None
    last = history[-1]
    return {
        "arch": arch, "dataset": bundle.dataset, "num_classes": bundle.num_classes, "seed": seed,
        "selected_epoch": best["epoch"], "val_accuracy": best["val_accuracy"], "val_loss": best["val_loss"],
        "test_accuracy": test_acc, "last_epoch_val_accuracy": last["val_accuracy"],
        "final_train_loss": last["train_loss"], "final_train_accuracy": last["train_accuracy"],
        "checkpoint": final_path, "checkpoint_sha256": ckpt_sha,
        "state_dict_sha256": sha256_state_dict(best["state_dict"]), "init_state_sha256": init_hash,
        "parameters": count_parameters(model), "epochs": epochs,
        "train_seconds_total": round(sum(h["train_seconds"] for h in history), 1),
        "wall_seconds_total": round(sum(h["epoch_seconds"] for h in history), 1),
        "threads": threads, "history": history,
        "test_predictions": preds.to(torch.uint8).tolist() if bundle.num_classes <= 255 else preds.tolist(),
        "loss_is_finite": all(math.isfinite(h["train_loss"]) for h in history),
    }
