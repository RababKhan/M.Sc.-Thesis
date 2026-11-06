"""PPO pruning environment, mirroring notebooks/Main code.ipynb cell by cell.

Two things are parameterised so the ablation and sweep can change exactly one
variable at a time. Both default to the notebook's values:

  * sparsity_coef   lambda_s in  R = 1.5*(acc/base) + lambda_s*S + 0.05*D
                    (S is sparsity as a PERCENTAGE, 0-100; never a fraction)
  * sensitivities   the per-layer value placed in the state's sensitivity slot

Everything else -- state vector, action space, pruning, episode structure,
reward arithmetic -- is copied from the notebook. ppo_experiments.py asserts
that this module reproduces the notebook's recorded PPO runs exactly before it
reports anything new.
"""
import copy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
import torchvision
import torchvision.transforms as transforms
import gymnasium as gym
from gymnasium import spaces
from torch.utils.data import DataLoader, TensorDataset

import common

ACTION_TO_PRUNE = common.ACTION_TO_PRUNE
FILTERED_LAYERS = common.FILTERED_LAYERS

ACCURACY_COEF = 1.5          # notebook cell 38
DEFAULT_SPARSITY_COEF = 0.04  # notebook cell 38
DIVERSITY_COEF = 0.05        # notebook cell 38
STATE_DIM = 7

SPLIT_SEED = 42
VAL_SIZE = 5000
RL_PROBE_SIZE = 1000
BATCH_SIZE = 128

_NORMALIZE = transforms.Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.2470, 0.2435, 0.2616))
_TRANSFORM_EVAL = transforms.Compose([transforms.ToTensor(), _NORMALIZE])


# ------------------------------------------------------------------- data
class Splits:
    """The notebook's split, with every evaluation view cached as tensors.

    Caching is exact: the evaluation transform is deterministic, and the loaders
    keep the notebook's batch size and order, so every batch is bit-identical.

    The test tensors are only handed out while no training is in progress.
    Calling test_loader() inside a `training()` block raises, so the test set
    cannot reach the reward, the sensitivity probe or policy selection.
    """

    def __init__(self, root):
        train_plain = torchvision.datasets.CIFAR10(root=root, train=True, download=False,
                                                   transform=_TRANSFORM_EVAL)
        generator = torch.Generator().manual_seed(SPLIT_SEED)
        permuted = torch.randperm(len(train_plain), generator=generator).tolist()
        self.train_indices = permuted[:-VAL_SIZE]
        self.val_indices = permuted[-VAL_SIZE:]
        self.probe_indices = self.val_indices[:RL_PROBE_SIZE]

        assert len(self.train_indices) == 45000 and len(self.val_indices) == VAL_SIZE
        assert not set(self.train_indices) & set(self.val_indices), "train/val overlap"
        assert set(self.probe_indices) <= set(self.val_indices), "probe is not inside validation"

        self._val = _cache(train_plain, self.val_indices)
        self._probe = (self._val[0][:RL_PROBE_SIZE], self._val[1][:RL_PROBE_SIZE])
        self._root = root
        self._test = None
        self._training = False

    def probe_loader(self):
        return _loader(*self._probe)

    def val_loader(self):
        return _loader(*self._val)

    def test_loader(self):
        if self._training:
            raise RuntimeError("test set requested while training is in progress")
        if self._test is None:
            test = torchvision.datasets.CIFAR10(root=self._root, train=False, download=False,
                                                transform=_TRANSFORM_EVAL)
            self._test = _cache(test, range(len(test)))
        return _loader(*self._test)

    def training(self):
        splits = self

        class _Block:
            def __enter__(self):
                splits._training = True

            def __exit__(self, *exc):
                splits._training = False

        return _Block()


def _cache(dataset, indices):
    images, labels = [], []
    for i in indices:
        image, label = dataset[i]
        images.append(image)
        labels.append(label)
    return torch.stack(images), torch.tensor(labels)


def _loader(images, labels):
    return DataLoader(TensorDataset(images, labels), batch_size=BATCH_SIZE, shuffle=False,
                      num_workers=0)


# ------------------------------------------------------------ notebook helpers
def evaluate_model(model, loader, device="cpu"):
    """Notebook cell 11: integer counts, 100 * correct / total."""
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for images, labels in loader:
            outputs = model(images.to(device))
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted.cpu() == labels).sum().item()
    return 100 * correct / total


def apply_layer_pruning_by_name_inplace(model, layer_name, amount):
    """Notebook cell 22."""
    for name, module in model.named_modules():
        if name == layer_name and isinstance(module, (nn.Conv2d, nn.Linear)):
            prune.l1_unstructured(module, name="weight", amount=amount)
            return model
    raise ValueError(f"Layer {layer_name} not found or not prunable")


calculate_sparsity = common.calculate_sparsity  # notebook cell 23, same arithmetic


def prunable_layers(model):
    """Notebook cells 17-19: all Conv2d/Linear layers, and the filtered four."""
    all_layers = [(n, m) for n, m in model.named_modules() if isinstance(m, (nn.Conv2d, nn.Linear))]
    filtered = [(n, m) for n, m in all_layers if n != "classifier.3"]
    assert [n for n, _ in filtered] == FILTERED_LAYERS, "prunable layer set differs from the notebook"
    return all_layers, filtered


def compute_layer_sensitivities(model, layers, loader, probe_amount=0.1):
    """Notebook cell 28: accuracy drop when one layer alone is pruned by 10%."""
    base = evaluate_model(model, loader)
    out = {}
    for name, _ in layers:
        pruned = apply_layer_pruning_by_name_inplace(copy.deepcopy(model), name, probe_amount)
        out[name] = base - evaluate_model(pruned, loader)
    return base, out


def extract_layer_state(layer, layer_index, total_layers, cumulative_pruning,
                        layer_name, layer_sensitivities):
    """Notebook cell 33."""
    weights = layer.weight.detach().cpu()
    sensitivity = 0.0
    if layer_name is not None and layer_sensitivities is not None:
        sensitivity = layer_sensitivities.get(layer_name, 0.0)
    return {
        "layer_index": layer_index,
        "total_layers": total_layers,
        "param_count": weights.numel(),
        "mean_abs_weight": weights.abs().mean().item(),
        "std_weight": weights.std().item(),
        "cumulative_pruning": cumulative_pruning,
        "sensitivity": sensitivity,
    }


def state_to_vector(state, max_param_count):
    """Notebook cell 35."""
    return np.array([
        state["layer_index"] / max(1, state["total_layers"] - 1),
        np.log1p(state["param_count"]),
        state["param_count"] / max_param_count,
        state["mean_abs_weight"],
        state["std_weight"],
        state["cumulative_pruning"],
        state["sensitivity"],
    ], dtype=np.float32)


def reward_components(final_accuracy, baseline_accuracy, sparsity, actions,
                      sparsity_coef=DEFAULT_SPARSITY_COEF):
    """Notebook cell 38, split into its three terms.

    `total` is accumulated in the notebook's order, so it matches the notebook's
    reward bit for bit when sparsity_coef is 0.04.
    """
    accuracy_term = ACCURACY_COEF * (final_accuracy / baseline_accuracy)
    sparsity_term = sparsity_coef * sparsity
    diversity_term = DIVERSITY_COEF * len(set(actions))
    total = accuracy_term + sparsity_term
    total += diversity_term
    return {"accuracy_term": accuracy_term, "sparsity_term": sparsity_term,
            "diversity_term": diversity_term, "total": total}


# -------------------------------------------------------------- environment
class CompressionEnvGym(gym.Env):
    """Notebook cell 40, with the sparsity coefficient as a parameter."""

    def __init__(self, base_model, prunable_layers, baseline_accuracy, eval_loader,
                 max_param_count, layer_sensitivities=None,
                 sparsity_coef=DEFAULT_SPARSITY_COEF):
        super().__init__()
        self.base_model = base_model
        self.prunable_layers_info = [(name, type(layer).__name__) for name, layer in prunable_layers]
        self.total_layers = len(prunable_layers)
        self.baseline_accuracy = baseline_accuracy
        self.eval_loader = eval_loader
        self.max_param_count = max_param_count
        self.layer_sensitivities = layer_sensitivities if layer_sensitivities is not None else {}
        self.sparsity_coef = sparsity_coef

        self.action_space = spaces.Discrete(len(ACTION_TO_PRUNE))
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(STATE_DIM,),
                                            dtype=np.float32)
        self.current_layer_index = 0
        self.current_model = None
        self.cumulative_pruning = 0.0
        self.actions_taken = []

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_layer_index = 0
        self.current_model = copy.deepcopy(self.base_model)
        self.cumulative_pruning = 0.0
        self.actions_taken = []
        return self._get_state(), {}

    def _get_state(self):
        if self.current_layer_index >= self.total_layers:
            return np.zeros((STATE_DIM,), dtype=np.float32)
        layer_name = self.prunable_layers_info[self.current_layer_index][0]
        for name, module in self.current_model.named_modules():
            if name == layer_name:
                state = extract_layer_state(module, self.current_layer_index, self.total_layers,
                                            self.cumulative_pruning, name, self.layer_sensitivities)
                return state_to_vector(state, self.max_param_count)
        raise ValueError(f"Layer {layer_name} not found in current model")

    def step(self, action):
        action = int(action)
        prune_amount = ACTION_TO_PRUNE[action]
        layer_name = self.prunable_layers_info[self.current_layer_index][0]

        self.current_model = apply_layer_pruning_by_name_inplace(self.current_model, layer_name,
                                                                 prune_amount)
        self.actions_taken.append(action)
        self.cumulative_pruning += prune_amount
        self.current_layer_index += 1

        terminated = self.current_layer_index >= self.total_layers
        if terminated:
            final_accuracy = evaluate_model(self.current_model, self.eval_loader)
            final_sparsity = calculate_sparsity(self.current_model)
            parts = reward_components(final_accuracy, self.baseline_accuracy, final_sparsity,
                                      self.actions_taken, self.sparsity_coef)
            reward = parts["total"]
            observation = np.zeros((STATE_DIM,), dtype=np.float32)
            info = {"final_accuracy": final_accuracy, "final_sparsity": final_sparsity,
                    "reward": reward, "actions": list(self.actions_taken), **parts}
        else:
            reward = 0.0
            observation = self._get_state()
            info = {"current_layer": layer_name, "prune_amount": prune_amount}

        return observation, reward, terminated, False, info
