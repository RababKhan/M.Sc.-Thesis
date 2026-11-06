"""Inspect what a trained PPO pruning policy responds to.

Observations are rebuilt without running the environment: the state for layer
i depends only on layer i's own (still unpruned) weights, its index, its
sensitivity value and the cumulative pruning of the layers before it, so it
can be computed exactly from the baseline model and the policy's own actions.
rollout_observations() asserts that the agent, fed these observations, picks
the recorded actions -- a check that the reconstruction is exact.

All analyses here are local and descriptive: they show how the trained
network's output changes when one input changes, not why it was learned.
"""
import numpy as np
import torch

import rl_env

FEATURES = ["layer_index_norm", "log1p_param_count", "param_count_ratio",
            "mean_abs_weight", "std_weight", "cumulative_pruning", "sensitivity"]
SENSITIVITY_DIM = 6


def layer_observation(model_layers, i, cumulative, sensitivities, max_param_count):
    name, module = model_layers[i]
    state = rl_env.extract_layer_state(module, i, len(model_layers), cumulative, name, sensitivities)
    return rl_env.state_to_vector(state, max_param_count)


def rollout_observations(agent, model_layers, sensitivities, max_param_count, expected_actions=None):
    """The deterministic rollout's observations, rebuilt exactly as the env builds them."""
    observations, actions, cumulative = [], [], 0.0
    for i in range(len(model_layers)):
        obs = layer_observation(model_layers, i, cumulative, sensitivities, max_param_count)
        action, _ = agent.predict(obs, deterministic=True)
        observations.append(obs)
        actions.append(int(action))
        cumulative += rl_env.ACTION_TO_PRUNE[int(action)]
    if expected_actions is not None:
        assert actions == list(expected_actions), f"rebuilt rollout {actions} != recorded {expected_actions}"
    return np.stack(observations), actions


def action_probs(agent, observations):
    with torch.no_grad():
        obs = torch.as_tensor(np.asarray(observations, dtype=np.float32))
        return agent.policy.get_distribution(obs).distribution.probs.numpy()


def _tvd(p, q):
    return 0.5 * np.abs(p - q).sum(axis=-1)


def sensitivity_perturbation(agent, observations, sensitivity_vector, extra_values=()):
    """Replace ONLY the sensitivity input of each layer's observation.

    Candidate values: 0, every other layer's sensitivity value, and any
    extra_values (e.g. the ends of the normalised range).
    """
    base = action_probs(agent, observations)
    rows = []
    for i in range(len(observations)):
        candidates = {"zero": 0.0}
        for j, v in enumerate(sensitivity_vector):
            if j != i:
                candidates[f"layer{j}_value"] = float(v)
        for k, v in enumerate(extra_values):
            candidates[f"extra{k}_{v}"] = float(v)
        for label, value in candidates.items():
            perturbed = observations[i].copy()
            perturbed[SENSITIVITY_DIM] = value
            p = action_probs(agent, perturbed[None])[0]
            rows.append({"layer_position": i, "perturbation": label,
                         "original_value": float(observations[i][SENSITIVITY_DIM]),
                         "new_value": value,
                         "unchanged_input": bool(np.isclose(value, observations[i][SENSITIVITY_DIM])),
                         "tvd": float(_tvd(base[i], p)),
                         "argmax_changed": bool(p.argmax() != base[i].argmax())})
    return rows


def permutation_importance(agent, observations):
    """For each input dimension, swap in the value another layer has for it.

    Returns mean total-variation distance and argmax-change rate per dimension
    over all (layer, other layer) pairs, so the sensitivity input can be
    compared with the other six inputs on the same footing.
    """
    base = action_probs(agent, observations)
    n = len(observations)
    out = []
    for d, feature in enumerate(FEATURES):
        tvds, changes, informative = [], [], 0
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                perturbed = observations[i].copy()
                perturbed[d] = observations[j][d]
                if not np.isclose(perturbed[d], observations[i][d]):
                    informative += 1
                p = action_probs(agent, perturbed[None])[0]
                tvds.append(_tvd(base[i], p))
                changes.append(p.argmax() != base[i].argmax())
        out.append({"feature": feature, "dim": d, "mean_tvd": float(np.mean(tvds)),
                    "argmax_change_rate": float(np.mean(changes)),
                    "pairs_with_different_value": informative, "pairs": len(tvds)})
    return out


def input_scale(agent, observations):
    """First-layer view of the policy network: how much each input contributes.

    contribution[d] = mean over hidden units and layers of |W1[:, d] * x_d|;
    saturation = share of first-layer tanh units with |pre-activation| > 2
    (tanh' < 0.071 there, so gradients through them are small).
    """
    first = agent.policy.mlp_extractor.policy_net[0]
    w = first.weight.detach().numpy()
    b = first.bias.detach().numpy()
    x = np.asarray(observations, dtype=np.float32)
    contributions = np.abs(w[None, :, :] * x[:, None, :]).mean(axis=(0, 1))
    pre = x @ w.T + b
    total = contributions.sum()
    return {
        "contribution": {f: float(c) for f, c in zip(FEATURES, contributions)},
        "contribution_share": {f: float(c / total) for f, c in zip(FEATURES, contributions)},
        "column_norm": {f: float(np.linalg.norm(w[:, d])) for d, f in enumerate(FEATURES)},
        "saturated_share": float((np.abs(pre) > 2).mean()),
    }
