"""Phase 4 library (results/phase4/phase4_preregistration.md): reward variants, priors, reward-aware
environment, elite-archive PPO. Phase-3 inputs are read only (frozen).

Rewards (A = pruned-model accuracy on the evaluation split, A_b = reference accuracy on the same split,
rho = total sparsity in %, D = number of distinct actions; every term is computed in this order so that
R0 is bit-identical to the Phase-3 reward):
  R0   1.5*(A/A_b) + 0.01*rho + 0.05*D                                   Phase-3 reward (lambda_s 0.01)
  R1a  1.5*(A/A_b) - kappa_a*|rho - 50| + 0.05*D                          common target 50%
  R1b  1.5*(A/A_b) - kappa_b*|rho - rho*_s| + 0.05*D                      rho*_s = max landscape sparsity
                                                                           with V_RL retention >= 95%
  R2   1.5*(A - A_60)/(A_b - A_60) + 1.5*rho/rho_60 + 0.05*D              anchor-normalised: A_60, rho_60 of
                                                                           the maximal policy [5,5,5,5]
Priors (subtracted from the PPO logits of unit u: c_u * ratio(a)/60; all actions always available):
  P0 none; P1 0.5*S_u (Phase 3); P2 1.0*(S_u - mean S) centred; P3 P2 permuted per seed;
  P4 1.0*mean|S - mean S| for every unit (magnitude-matched pure conservatism); P5 -P2 (sign-reversed)
"""
import copy
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.utils import explained_variance

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
for p in (ROOT, os.path.join(ROOT, "experiments", "phase3")):
    if p not in sys.path:
        sys.path.insert(0, p)

import phase3_common as P3  # noqa: E402  (frozen Phase-3 inputs and landscapes, read only)
from src import pruning as P, rl as RL  # noqa: E402

RESULTS = os.path.join(ROOT, "results", "phase4")
LAND = os.path.join(RESULTS, "landscapes")
RUNS = os.path.join(RESULTS, "runs")
CURVES = os.path.join(RUNS, "curves")
AGENTS = os.path.join(ROOT, "checkpoints", "phase4_agents")
PRED = os.path.join(RESULTS, "predictions")
PREREG = os.path.join(RESULTS, "phase4_preregistration.md")
CONFIG = os.path.join(RESULTS, "phase4_config.json")
SELECTION = os.path.join(RESULTS, "phase4_stage_selection.json")

SETTINGS = P3.SETTINGS
SEEDS = list(range(400, 420))
REWARDS = ["R0", "R1a", "R1b", "R2"]
PRIORS = ["P0", "P1", "P2", "P3", "P4", "P5"]
SEARCH = ["S0", "S1", "S2", "S2A"]
THREADS = 1
TOTAL_TIMESTEPS = 2000                 # -> 2,048 trained = 512 episodes (identical in every run)
EPISODES = 512
ACC_COEF, DIV_COEF = 1.5, 0.05
R0_LAMBDA = 0.01
TARGET_COMMON = 50.0
RETENTION_TOL = 0.95
KAPPA_GRID = [0.005, 0.01, 0.02, 0.04, 0.08]
TARGET_WINDOW = 5.0
BETA_P1, BETA_NEW = 0.5, 1.0
ETA, ELITE_K, ARCHIVE_K = 0.1, 5, 10
TOPK = [1, 10, 20, 50, 100]
CENSOR = 513                           # "episodes to first top-k" when never reached
A_NORM = np.array([P.ACTION_TO_PRUNE[a] / 0.6 for a in range(6)])
MAX_POLICY = (5, 5, 5, 5)


# ------------------------------------------------------------------ rewards
def reward_parts(kind, A, rho, actions, prm):
    """prm: {'A_b', 'kappa', 'target', 'A_60', 'rho_60'} for the evaluation split (V_RL or V_SELECT)."""
    D = len(set(int(a) for a in actions))
    if kind == "R0":
        acc = ACC_COEF * (A / prm["A_b"])
        sp = R0_LAMBDA * rho
    elif kind in ("R1a", "R1b"):
        acc = ACC_COEF * (A / prm["A_b"])
        sp = -prm["kappa"] * abs(rho - prm["target"])
    elif kind == "R2":
        acc = ACC_COEF * ((A - prm["A_60"]) / (prm["A_b"] - prm["A_60"]))
        sp = ACC_COEF * (rho / prm["rho_60"])
    else:
        raise ValueError(kind)
    div = DIV_COEF * D
    total = acc + sp
    total += div
    return {"accuracy_term": acc, "sparsity_term": sp, "diversity_term": div, "total": total}


def utility(kind, A, rho, prm):
    """The reward without its diversity term (exploration 'utility' endpoint)."""
    parts = reward_parts(kind, A, rho, (0,), prm)
    return parts["accuracy_term"] + parts["sparsity_term"]


# ------------------------------------------------------------------ priors
def value_changing_permutation(values, seed):
    """soft_prior.shuffled / Phase-3 algorithm: every unit receives another unit's, different, value."""
    rng = np.random.default_rng(seed)
    for _ in range(10000):
        perm = rng.permutation(len(values))
        if all(perm[i] != i and not np.isclose(values[perm[i]], values[i]) for i in range(len(values))):
            return [int(p) for p in perm]
    raise RuntimeError("no value-changing permutation")


def prior_coefficients(condition, S, seed):
    """c_u for each unit (None = no prior)."""
    S = np.asarray(S, float)
    cen = S - S.mean()
    if condition == "P0":
        return None, None
    if condition == "P1":
        return (BETA_P1 * S).tolist(), None
    if condition == "P2":
        return (BETA_NEW * cen).tolist(), None
    if condition == "P3":
        perm = value_changing_permutation(cen, seed)
        return (BETA_NEW * cen[perm]).tolist(), perm
    if condition == "P4":
        return [BETA_NEW * float(np.abs(cen).mean())] * len(S), None
    if condition == "P5":
        return (-BETA_NEW * cen).tolist(), None
    raise ValueError(condition)


def prior_matrix(coeffs):
    """B[u, a] = c_u * ratio(a)/60, subtracted from the logits. P1 equals RL.prior_matrix(0.5, S) exactly."""
    return (np.asarray(coeffs, float)[:, None] * A_NORM[None, :]).tolist()


# ------------------------------------------------------------------ environment
class RewardEnv(RL.PruningEnv):
    """src.rl.PruningEnv with the terminal reward replaced by a Phase-4 reward variant.

    Everything else (state, actions, pruning, the landscape cache and its random-stream behaviour) is the
    verified environment. The parent's R0 computation is overwritten, not used.
    """

    def __init__(self, *args, reward_kind="R0", reward_params=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.reward_kind, self.reward_params = reward_kind, reward_params

    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)
        if terminated:
            parts = reward_parts(self.reward_kind, info["final_accuracy"], info["final_sparsity"],
                                 info["actions"], self.reward_params)
            reward = parts["total"]
            info.update(parts)
            info["reward"] = reward
        return obs, reward, terminated, truncated, info

    def trajectory_obs(self, actions):
        """The four observations the environment emits along `actions` (no evaluation, no random draw)."""
        model = copy.deepcopy(self.base_model)
        units = P.get_units(model, self.arch)
        obs, cum = [], 0.0
        for i, a in enumerate(actions):
            obs.append(RL.unit_state(units[i], i, self.n_units, cum, self.sensitivities[i], self.max_param_count))
            ratio = P.ACTION_TO_PRUNE[int(a)]
            P.apply_unit_ratio(units[i], ratio)
            cum += ratio
        return np.stack(obs)


# ------------------------------------------------------------------ elite archive + behavioural cloning
class ArchiveCallback(BaseCallback):
    """Keeps every unique sampled policy with its (V_RL) reward; before each PPO update it hands the top-K
    unique policies (ties: earlier discovery) to the model as behavioural-cloning targets. Read-only on the
    environment; draws no random numbers."""

    def __init__(self, env, k=ELITE_K):
        super().__init__()
        self.env, self.k = env, k
        self.seen = {}                 # actions -> (reward, first episode)
        self.episode = 0
        self._obs_cache = {}

    def _on_step(self):
        for done, info in zip(self.locals["dones"], self.locals["infos"]):
            if done and "final_accuracy" in info:
                self.episode += 1
                key = tuple(info["actions"])
                if key not in self.seen:
                    self.seen[key] = (info["reward"], self.episode)
        return True

    def _on_rollout_end(self):
        if len(self.seen) < self.k:
            self.model.elite_obs = self.model.elite_actions = None
            return
        elite = sorted(self.seen.items(), key=lambda kv: (-kv[1][0], kv[1][1]))[: self.k]
        obs, acts = [], []
        for key, _ in elite:
            if key not in self._obs_cache:
                self._obs_cache[key] = self.env.trajectory_obs(key)
            obs.append(self._obs_cache[key])
            acts.extend(key)
        self.model.elite_obs = torch.as_tensor(np.concatenate(obs), dtype=torch.float32)
        self.model.elite_actions = torch.as_tensor(acts, dtype=torch.long)
        self.model.elite_policies = [list(k) for k, _ in elite]


class ElitePPO(PPO):
    """SB3 2.8.0 PPO whose update adds eta * L_elite to every minibatch loss:
    L_elite = -mean log pi(a_e | s_e) over the (state, action) pairs of the current elite policies.
    With eta = 0 or no elites, train() is SB3's PPO.train() line for line (verified bit-identical)."""

    def __init__(self, *args, bc_eta=ETA, **kwargs):
        super().__init__(*args, **kwargs)
        self.bc_eta = bc_eta
        self.elite_obs = self.elite_actions = None
        self.elite_policies = None
        self.bc_updates = 0

    def train(self) -> None:
        th = torch
        self.policy.set_training_mode(True)
        self._update_learning_rate(self.policy.optimizer)
        clip_range = self.clip_range(self._current_progress_remaining)
        if self.clip_range_vf is not None:
            clip_range_vf = self.clip_range_vf(self._current_progress_remaining)
        entropy_losses, pg_losses, value_losses, clip_fractions = [], [], [], []
        continue_training = True
        use_bc = self.bc_eta > 0 and self.elite_obs is not None
        for epoch in range(self.n_epochs):
            approx_kl_divs = []
            for rollout_data in self.rollout_buffer.get(self.batch_size):
                actions = rollout_data.actions
                if isinstance(self.action_space, spaces.Discrete):
                    actions = rollout_data.actions.long().flatten()
                values, log_prob, entropy = self.policy.evaluate_actions(rollout_data.observations, actions)
                values = values.flatten()
                advantages = rollout_data.advantages
                if self.normalize_advantage and len(advantages) > 1:
                    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
                ratio = th.exp(log_prob - rollout_data.old_log_prob)
                policy_loss_1 = advantages * ratio
                policy_loss_2 = advantages * th.clamp(ratio, 1 - clip_range, 1 + clip_range)
                policy_loss = -th.min(policy_loss_1, policy_loss_2).mean()
                pg_losses.append(policy_loss.item())
                clip_fraction = th.mean((th.abs(ratio - 1) > clip_range).float()).item()
                clip_fractions.append(clip_fraction)
                if self.clip_range_vf is None:
                    values_pred = values
                else:
                    values_pred = rollout_data.old_values + th.clamp(values - rollout_data.old_values,
                                                                     -clip_range_vf, clip_range_vf)
                value_loss = F.mse_loss(rollout_data.returns, values_pred)
                value_losses.append(value_loss.item())
                if entropy is None:
                    entropy_loss = -th.mean(-log_prob)
                else:
                    entropy_loss = -th.mean(entropy)
                entropy_losses.append(entropy_loss.item())
                loss = policy_loss + self.ent_coef * entropy_loss + self.vf_coef * value_loss
                if use_bc:
                    _, elite_log_prob, _ = self.policy.evaluate_actions(self.elite_obs, self.elite_actions)
                    loss = loss + self.bc_eta * (-elite_log_prob.mean())
                with th.no_grad():
                    log_ratio = log_prob - rollout_data.old_log_prob
                    approx_kl_div = th.mean((th.exp(log_ratio) - 1) - log_ratio).cpu().numpy()
                    approx_kl_divs.append(approx_kl_div)
                if self.target_kl is not None and approx_kl_div > 1.5 * self.target_kl:
                    continue_training = False
                    break
                self.policy.optimizer.zero_grad()
                loss.backward()
                th.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.policy.optimizer.step()
            self._n_updates += 1
            if not continue_training:
                break
        if use_bc:
            self.bc_updates += 1
        explained_var = explained_variance(self.rollout_buffer.values.flatten(), self.rollout_buffer.returns.flatten())
        self.logger.record("train/entropy_loss", np.mean(entropy_losses))
        self.logger.record("train/policy_gradient_loss", np.mean(pg_losses))
        self.logger.record("train/value_loss", np.mean(value_losses))
        self.logger.record("train/approx_kl", np.mean(approx_kl_divs))
        self.logger.record("train/clip_fraction", np.mean(clip_fractions))
        self.logger.record("train/loss", loss.item())
        self.logger.record("train/explained_variance", explained_var)
        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/clip_range", clip_range)
