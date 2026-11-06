"""Accuracy, predictions and loss with the historical arithmetic.

evaluate_accuracy is notebook cell 11 / rl_env.evaluate_model: integer counts,
100 * correct / total, torch.max over the logits, evaluation in batches of 128.
"""
import torch
import torch.nn as nn


def evaluate_accuracy(model, loader):
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for images, labels in loader:
            _, predicted = torch.max(model(images), 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return 100 * correct / total


def predictions(model, loader):
    """Per-sample predicted labels and ground truth, in loader order."""
    model.eval()
    preds, labels = [], []
    with torch.no_grad():
        for images, targets in loader:
            _, predicted = torch.max(model(images), 1)
            preds.append(predicted)
            labels.append(targets)
    return torch.cat(preds), torch.cat(labels)


def accuracy_from(preds, labels):
    return 100.0 * int((preds == labels).sum()) / len(labels)


def mean_loss_accuracy(model, x, y, batch_size=128):
    """Mean cross-entropy and accuracy on tensors (archive_confirm.mean_loss_acc arithmetic)."""
    model.eval()
    loss_fn = nn.CrossEntropyLoss(reduction="sum")
    loss, correct = 0.0, 0
    with torch.no_grad():
        for i in range(0, len(x), batch_size):
            out = model(x[i:i + batch_size])
            loss += loss_fn(out, y[i:i + batch_size]).item()
            correct += (out.argmax(1) == y[i:i + batch_size]).sum().item()
    return loss / len(x), 100.0 * correct / len(x)
