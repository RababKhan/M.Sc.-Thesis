"""Phase 5 library (results/phase5/phase5_design_commitments.md, phase5_preregistration.md).

Phase-3/4 modules are imported read only. VAL-RL / VAL-SELECT come from results/phase5/splits.
"""
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
for p in (ROOT, os.path.join(ROOT, "experiments", "phase3")):
    if p not in sys.path:
        sys.path.insert(0, p)

import phase3_common as P3  # noqa: E402  (frozen references, settings, policy list)
from src import data as D, rl as RL  # noqa: E402

RESULTS = os.path.join(ROOT, "results", "phase5")
SPLITS = os.path.join(RESULTS, "splits")
LAND = os.path.join(RESULTS, "landscapes")
RUNS = os.path.join(RESULTS, "runs")
CURVES = os.path.join(RUNS, "curves")
AGENTS = os.path.join(ROOT, "checkpoints", "phase5_agents")
PRED = os.path.join(RESULTS, "predictions")
COMMITMENTS = os.path.join(RESULTS, "phase5_design_commitments.md")
PREREG = os.path.join(RESULTS, "phase5_preregistration.md")
CONFIG = os.path.join(RESULTS, "phase5_config.json")
FREEZE = os.path.join(RESULTS, "phase5_final_policies.json")

SETTINGS = P3.SETTINGS
POLICIES = P3.POLICIES
INDEX = {p: i for i, p in enumerate(POLICIES)}
SEEDS = list(range(500, 520))
CONDITIONS = ["C0", "C1", "C2"]
THREADS = 1
TOTAL_TIMESTEPS = 2000             # -> 2,048 = 512 episodes, as in Phases 3-4
EPISODES = 512
TAUS = [0.95, 0.97, 0.98, 0.99]
MIN_FEASIBLE, MIN_SPARSITY = 20, 10.0
EPS, KAPPA = 0.01, 10.0
BETA = 1.0                          # centred prior P2 (Phase 4)
ARCHIVE_K = 10
CENSOR = 513
A_NORM = np.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.6]) / 0.6


def bundle(dataset):
    """DataBundle whose V_RL / V_SELECT are the Phase-5 VAL-RL / VAL-SELECT (guards unchanged)."""
    b = D.DataBundle(dataset, train_images=False, rl_split=False)
    z = np.load(os.path.join(SPLITS, f"{dataset}_phase5_split.npz"))
    b.rl_positions, b.select_positions = z["val_rl_positions"], z["val_select_positions"]
    assert len(b.rl_positions) == len(b.select_positions) == 2500
    assert np.array_equal(b.val_y.numpy()[b.rl_positions], z["val_rl_labels"])
    return b


# ------------------------------------------------------------------ rewards
def r_constrained(A, A_d, rho_pct, tau):
    q = A / A_d
    if q >= tau:
        return 1.0 + rho_pct / 100.0 + EPS * q
    return -KAPPA * (tau - q)


def r0(A, A_d, rho_pct, actions):
    """Phase-3/4 reward (lambda_s 0.01), term order as in src.rl.reward_components."""
    acc = 1.5 * (A / A_d)
    sp = 0.01 * rho_pct
    div = 0.05 * len(set(int(a) for a in actions))
    total = acc + sp
    total += div
    return total


def reward(kind, A, A_d, rho_pct, actions, tau):
    return r0(A, A_d, rho_pct, actions) if kind == "R0" else r_constrained(A, A_d, rho_pct, tau)


class Phase5Env(RL.PruningEnv):
    """Verified src.rl.PruningEnv with the terminal reward replaced (R0 or R_constrained on VAL-RL)."""

    def __init__(self, *args, reward_kind="RC", tau=None, dense_acc=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.reward_kind, self.tau, self.dense_acc = reward_kind, tau, dense_acc

    def step(self, action):
        obs, rew, terminated, truncated, info = super().step(action)
        if terminated:
            rew = reward(self.reward_kind, info["final_accuracy"], self.dense_acc, info["final_sparsity"],
                         info["actions"], self.tau)
            info["reward"] = rew
            info["q"] = info["final_accuracy"] / self.dense_acc
            info["accuracy_term"] = info["sparsity_term"] = info["diversity_term"] = float("nan")
        return obs, rew, terminated, truncated, info


def prior_matrix(coeffs):
    return (np.asarray(coeffs, float)[:, None] * A_NORM[None, :]).tolist()
