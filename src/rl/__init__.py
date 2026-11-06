"""Architecture-independent PPO pruning environment (historical formulation, generalised to units).

With arch="simplecnn" and the historical inputs this computes exactly what
rl_env.CompressionEnvGym (and, with a cache, constrained_ppo.MaskedCompressionEnv)
computes: same state vector, action space, pruning, reward arithmetic and use of
the global random stream. experiments/phase2/verify_library.py checks that
complete PPO runs reproduce the recorded ones bit for bit.

State (7 features, for the current unit u of n units)
  u / (n - 1), log1p(params_u), params_u / max_u params, mean|w_u|, sd(w_u),
  cumulative pruning (sum of chosen ratios so far), sensitivity_u
Actions    Discrete(6): ratios {0, 10, 20, 30, 40, 60}% applied to every tensor of the unit
Reward     R = 1.5 * acc / base + lambda_s * S(%) + 0.05 * (number of distinct actions),
           paid at the end of the episode; 0 at intermediate steps
"""
import copy

import gymnasium as gym
import numpy as np
import torch
from gymnasium import spaces
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.policies import ActorCriticPolicy

from src.evaluation import evaluate_accuracy
from src.pruning import ACTION_TO_PRUNE, apply_unit_ratio, get_units, total_sparsity

ACCURACY_COEF = 1.5
DIVERSITY_COEF = 0.05
DEFAULT_SPARSITY_COEF = 0.04
STATE_DIM = 7
N_ACTIONS = len(ACTION_TO_PRUNE)
EPISODES = 512
# Historical PPO settings (notebook cells 50/63); every other SB3 default is unchanged.
PPO_KWARGS = dict(policy="MlpPolicy", learning_rate=0.0003, n_steps=64, batch_size=32,
                  n_epochs=10, gamma=0.99, ent_coef=0.01, verbose=0)
LEGACY_TOTAL_TIMESTEPS = 2000      # historical SimpleCNN runs: 2,000 requested -> 2,048 = 512 episodes


def total_timesteps(n_units, episodes=EPISODES):
    return episodes * n_units


def reward_components(final_accuracy, baseline_accuracy, sparsity, actions, sparsity_coef=DEFAULT_SPARSITY_COEF):
    accuracy_term = ACCURACY_COEF * (final_accuracy / baseline_accuracy)
    sparsity_term = sparsity_coef * sparsity
    diversity_term = DIVERSITY_COEF * len(set(actions))
    total = accuracy_term + sparsity_term
    total += diversity_term
    return {"accuracy_term": accuracy_term, "sparsity_term": sparsity_term,
            "diversity_term": diversity_term, "total": total}


def unit_state(unit, index, n_units, cumulative, sensitivity, max_param_count):
    w = unit.weights()
    n = w.numel()
    return np.array([index / max(1, n_units - 1), np.log1p(n), n / max_param_count,
                     w.abs().mean().item(), w.std().item(), cumulative, sensitivity], dtype=np.float32)


class PruningEnv(gym.Env):
    """One episode = one pruning decision per unit, in unit order.

    eval_loader    V_RL (or the legacy probe) loader; the reward's accuracy source
    sensitivities  one value per unit for the state's sensitivity slot (default 0)
    cache          optional dict shared across episodes/runs on the same model and
                   loader: memoises the terminal (accuracy, sparsity) per action tuple.
                   On a cache hit a DataLoader iterator is still created, so the
                   global torch random stream advances exactly as without the cache.
    layer_masks    optional per-unit boolean action masks (MaskablePPO)
    """

    def __init__(self, base_model, arch, baseline_accuracy, eval_loader, sensitivities=None,
                 sparsity_coef=DEFAULT_SPARSITY_COEF, cache=None, layer_masks=None):
        super().__init__()
        self.base_model, self.arch = base_model, arch
        units = get_units(base_model, arch)
        self.unit_names = [u.name for u in units]
        self.n_units = len(units)
        self.max_param_count = max(u.param_count for u in units)
        self.baseline_accuracy = baseline_accuracy
        self.eval_loader = eval_loader
        self.sensitivities = list(sensitivities) if sensitivities is not None else [0.0] * self.n_units
        assert len(self.sensitivities) == self.n_units
        self.sparsity_coef = sparsity_coef
        self.cache = cache
        self.layer_masks = None if layer_masks is None else [np.array(m, bool) for m in layer_masks]
        self.action_space = spaces.Discrete(N_ACTIONS)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(STATE_DIM,), dtype=np.float32)
        self.current_model, self.units = None, None
        self.index, self.cumulative, self.actions_taken = 0, 0.0, []

    def action_masks(self):
        if self.layer_masks is None or self.index >= self.n_units:
            return np.ones(N_ACTIONS, dtype=bool)
        return self.layer_masks[self.index].copy()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_model = copy.deepcopy(self.base_model)
        self.units = get_units(self.current_model, self.arch)
        self.index, self.cumulative, self.actions_taken = 0, 0.0, []
        return self._state(), {}

    def _state(self):
        if self.index >= self.n_units:
            return np.zeros((STATE_DIM,), dtype=np.float32)
        return unit_state(self.units[self.index], self.index, self.n_units, self.cumulative,
                          self.sensitivities[self.index], self.max_param_count)

    def _terminal_eval(self):
        key = tuple(self.actions_taken)
        if self.cache is None:
            return evaluate_accuracy(self.current_model, self.eval_loader), total_sparsity(self.current_model)
        if key not in self.cache:
            self.cache[key] = (evaluate_accuracy(self.current_model, self.eval_loader),
                               total_sparsity(self.current_model))
        else:
            iter(self.eval_loader)       # consume the global random stream as a real evaluation would
        return self.cache[key]

    def step(self, action):
        action = int(action)
        ratio = ACTION_TO_PRUNE[action]
        name = self.unit_names[self.index]
        apply_unit_ratio(self.units[self.index], ratio)
        self.actions_taken.append(action)
        self.cumulative += ratio
        self.index += 1
        terminated = self.index >= self.n_units
        if terminated:
            acc, sparsity = self._terminal_eval()
            parts = reward_components(acc, self.baseline_accuracy, sparsity, self.actions_taken, self.sparsity_coef)
            reward = parts["total"]
            obs = np.zeros((STATE_DIM,), dtype=np.float32)
            info = {"final_accuracy": acc, "final_sparsity": sparsity, "reward": reward,
                    "actions": list(self.actions_taken), **parts}
        else:
            reward, obs = 0.0, self._state()
            info = {"current_layer": name, "prune_amount": ratio}
        return obs, reward, terminated, False, info


# ------------------------------------------------------------------ soft sensitivity prior
def prior_matrix(beta, sensitivity):
    """B[u, a] = beta * S_u * ratio(a) / 0.6: the amount subtracted from the logits of unit u."""
    a_norm = np.array([ACTION_TO_PRUNE[a] / 0.6 for a in range(N_ACTIONS)])
    return (beta * np.asarray(sensitivity, float)[:, None] * a_norm[None, :]).tolist()


class SoftPriorPolicy(ActorCriticPolicy):
    """MlpPolicy whose logits for unit u are shifted by -prior[u] (soft_prior.SoftPriorPolicy, n units).

    The unit is read from the observation's first feature, u / (n - 1). A zero prior
    computes exactly what MlpPolicy computes.
    """

    def __init__(self, *args, prior=None, **kwargs):
        if prior is None:
            raise ValueError("prior matrix required (use MlpPolicy for no prior)")
        self._prior_list = prior
        super().__init__(*args, **kwargs)
        self.register_buffer("prior", torch.tensor(self._prior_list, dtype=torch.float32), persistent=False)
        self._obs = None

    def _get_constructor_parameters(self):
        data = super()._get_constructor_parameters()
        data["prior"] = self._prior_list
        return data

    def unit_index(self, obs):
        n = len(self._prior_list)
        return torch.round(obs[..., 0] * (n - 1)).long().clamp(0, n - 1)

    def _get_action_dist_from_latent(self, latent_pi):
        logits = self.action_net(latent_pi) - self.prior[self.unit_index(self._obs)]
        return self.action_dist.proba_distribution(action_logits=logits)

    def forward(self, obs, deterministic=False):
        self._obs = obs
        return super().forward(obs, deterministic)

    def evaluate_actions(self, obs, actions):
        self._obs = obs
        return super().evaluate_actions(obs, actions)

    def get_distribution(self, obs):
        self._obs = obs
        return super().get_distribution(obs)


# ------------------------------------------------------------------ training helpers
class EpisodeLog(BaseCallback):
    """Records every finished training episode. Read-only: touches no random stream."""

    def __init__(self):
        super().__init__()
        self.rows = []

    def _on_step(self):
        for done, info in zip(self.locals["dones"], self.locals["infos"]):
            if done and "final_accuracy" in info:
                self.rows.append({"episode": len(self.rows) + 1, "timestep": self.num_timesteps,
                                  "episode_reward": info["reward"], "accuracy_term": info["accuracy_term"],
                                  "sparsity_term": info["sparsity_term"], "diversity_term": info["diversity_term"],
                                  "probe_accuracy": info["final_accuracy"], "sparsity": info["final_sparsity"],
                                  "actions": str(info["actions"])})
        return True


def deterministic_rollout(agent, env, masked=False):
    obs, _ = env.reset()
    actions, done, info = [], False, {}
    while not done:
        if masked:
            a, _ = agent.predict(obs, deterministic=True, action_masks=env.action_masks())
        else:
            a, _ = agent.predict(obs, deterministic=True)
        actions.append(int(a))
        obs, _, done, _, info = env.step(int(a))
    return actions, info
