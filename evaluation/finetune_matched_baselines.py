"""Matched global-magnitude baseline for the fine-tuning arm.

The fine-tuning notebook compares PPO + fine-tune only against uniform
per-layer baselines. This adds the missing like-for-like comparison: global
magnitude pruning over the same four layers, calibrated to the exact total
sparsity of the PPO + fine-tune policy, then fine-tuned with the notebook's
exact protocol:

  * Adam, lr 1e-4, 3 epochs, on the 45,000-image augmented train split
  * epoch selected on the 5,000-image validation split (starting from the
    pruned model's own validation accuracy, as the notebook does)
  * pruning masks stay attached during fine-tuning, so pruned weights stay zero
  * test set measured once, on the selected model

Single run, like every row of the fine-tuning table; fine-tuning randomness
(shuffle order, augmentation) makes repeated runs differ by a few tenths of a
point, so no significance test is attempted.

Output: results/finetune_matched_global_magnitude.csv
Run from the repository root:  python evaluation/finetune_matched_baselines.py
"""
import copy
import datetime
import os
import sys
import time

import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common  # noqa: E402
import ppo_experiments  # noqa: E402  (provenance helpers only)
import rl_env  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CKPT = os.path.join(ROOT, "checkpoints", "cnn_baseline_FIXED.pth")
HOLDOUT = os.path.join(ROOT, "results", "final_results_fair_finetune_holdout.csv")
SEED = 42


def finetune_model(model, train_loader, val_loader, epochs=3, lr=0.0001):
    """Final fine tuned.ipynb, finetune_model: best epoch on VALIDATION accuracy."""
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    best_val_acc = rl_env.evaluate_model(model, val_loader)
    best_state = copy.deepcopy(model.state_dict())
    history = [best_val_acc]
    for _ in range(epochs):
        model.train()
        for images, labels in train_loader:
            optimizer.zero_grad()
            loss = criterion(model(images), labels)
            loss.backward()
            optimizer.step()
        val_acc = rl_env.evaluate_model(model, val_loader)
        history.append(val_acc)
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = copy.deepcopy(model.state_dict())
    model.load_state_dict(best_state)
    return model, best_val_acc, history


def main():
    holdout = pd.read_csv(HOLDOUT)
    ppo_row = holdout[holdout["Method"] == "PPO_FineTune"].iloc[0]
    target = float(ppo_row["Sparsity (%)"])

    torch.manual_seed(SEED)
    splits = rl_env.Splits(os.path.join(ROOT, "data"))
    transform_train = transforms.Compose([
        transforms.RandomHorizontalFlip(), transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(), rl_env._NORMALIZE])
    train_aug = torchvision.datasets.CIFAR10(root=os.path.join(ROOT, "data"), train=True,
                                             download=False, transform=transform_train)
    train_loader = DataLoader(Subset(train_aug, splits.train_indices), batch_size=128,
                              shuffle=True, num_workers=0)

    model = common.load_baseline(CKPT)
    gm, fraction = common.prune_global_to_total(model, target)
    sparsity = common.calculate_sparsity(gm)
    assert abs(sparsity - target) < 0.01, f"not matched: {sparsity} vs {target}"
    per_layer = common.per_layer_sparsity(gm)

    started = time.time()
    with splits.training():
        pre_val = rl_env.evaluate_model(gm, splits.val_loader())
        tuned, val_acc, history = finetune_model(gm, train_loader, splits.val_loader())
    assert abs(common.calculate_sparsity(tuned) - sparsity) < 1e-9, "fine-tuning changed sparsity"
    test_acc = rl_env.evaluate_model(tuned, splits.test_loader())  # once, after selection

    env = ppo_experiments.environment()
    row = {
        "Method": f"FT Global magnitude @ {sparsity:.2f}%",
        "matched_to": f"PPO_FineTune {ppo_row['Actions']}",
        "Val Accuracy (%)": val_acc, "Test Accuracy (%)": test_acc, "Sparsity (%)": sparsity,
        "pre_finetune_val_accuracy": pre_val,
        "val_accuracy_by_epoch": str(history),
        "fraction_of_four_layers": fraction,
        **{f"sparsity_{k}": v for k, v in per_layer.items()},
        "ppo_finetune_test_accuracy": float(ppo_row["Test Accuracy (%)"]),
        "ppo_finetune_val_accuracy": float(ppo_row["Val Accuracy (%)"]),
        "protocol": "Adam lr 1e-4, 3 epochs, full train split, best epoch on validation, test once",
        "runtime_seconds": round(time.time() - started, 1),
        "baseline_sha256": ppo_experiments.sha256(CKPT),
        "git_commit": ppo_experiments.git_state()["commit"],
        "cpu": env["cpu"], "python": env["python"], "torch": env["torch"],
        "run_at": datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
    }
    pd.DataFrame([row]).to_csv(os.path.join(ROOT, "results", "finetune_matched_global_magnitude.csv"),
                               index=False)
    print(f"GM @ {sparsity:.2f}% + FT: val {val_acc:.2f}%  test {test_acc:.2f}%  "
          f"(PPO+FT {ppo_row['Actions']}: test {ppo_row['Test Accuracy (%)']:.2f}%)")
    print("per-layer:", {k: round(v, 1) for k, v in per_layer.items()})


if __name__ == "__main__":
    main()
