"""Phase 7B: sensitivity-aware energy-constrained structured PPO (design section 9).

  python experiments/phase7b/ppo.py smoke                      # implementation check on a synthetic landscape
  python experiments/phase7b/ppo.py dev                        # Stage A: alpha in {0.5,1,2,4} x seeds 900-904
  python experiments/phase7b/ppo.py confirm                    # Stage B: seeds 700-719 under the frozen pre-registration
Each episode chooses one structured action per unit (conv1, conv2, conv3, fc1); the action vector is a physically
compacted policy. The terminal reward uses the cached E-epoch proxy accuracy (VAL-RL) and the validated energy
surrogate; the feasible archive and the selection rule follow the design exactly.
"""
import argparse
import csv
import datetime
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import p7b_lib as L  # noqa: E402

import gymnasium as gym  # noqa: E402
import numpy as np  # noqa: E402
from gymnasium import spaces  # noqa: E402

KAPPA, EPS, BETA, TIE_BAND = 10.0, 0.01, 1.0, 0.01
EPISODES, N_UNITS, STATE_DIM = 512, 4, 8
DEV_SEEDS, ALPHAS, CONFIRM_SEEDS = list(range(900, 905)), [0.5, 1.0, 2.0, 4.0], list(range(700, 720))
A_NORM = np.array(L.RATIO) / 0.6
RUNS = os.path.join(L.RESULTS, "_ppo_runs")


# ------------------------------------------------------------------ inputs
class Inputs:
    """Everything the environment reads: proxy cache, tau, surrogate predictions, table, sensitivity."""

    def __init__(self, proxy_q, tau, energy, tab, sens_centred):
        self.q, self.tau, self.e, self.t = proxy_q, tau, energy, tab
        self.e_dense = energy[(0, 0, 0, 0)]
        self.s_max = max(1 - v / self.e_dense for v in energy.values())
        self.sens = np.asarray(sens_centred, float)
        # unit shares of dense parameters and MACs (the layer each unit owns: conv1, conv2, conv3, fc1)
        par = np.array([3 * 9 * 32 + 32, 32 * 9 * 64 + 64, 64 * 9 * 128 + 128, 2048 * 256 + 256], float)
        mac = np.array([32 * 32 * 32 * 27, 16 * 16 * 64 * 32 * 9, 8 * 8 * 128 * 64 * 9, 2048 * 256], float)
        self.param_share = par / tab[(0, 0, 0, 0)]["params"]
        self.mac_share = mac / tab[(0, 0, 0, 0)]["macs"]
        self.channels = np.array([32, 64, 128, 256], float)

    @classmethod
    def load(cls):
        prox = json.load(open(os.path.join(L.RESULTS, "accuracy_proxy_metrics.json"), encoding="utf-8"))
        e = prox["accepted_epochs"]
        assert e in (2, 3)
        tau = prox["proxies"][str(e)]["tau"]
        land = {tuple(json.loads(r["policy"])): float(r["q"]) for r in
                csv.DictReader(open(os.path.join(L.RESULTS, "proxy_landscape.csv"), encoding="utf-8"))}
        assert len(land) == 1296
        en = {tuple(json.loads(r["policy"])): float(r["e1000_pred_j"]) for r in
              csv.DictReader(open(os.path.join(L.RESULTS, "energy_surrogate_predictions.csv"), encoding="utf-8"))}
        sens = json.load(open(os.path.join(L.RESULTS, "structured_sensitivity.json"), encoding="utf-8"))["centred"]
        return cls(land, tau, en, L.table(), sens)

    def reward(self, policy, alpha):
        q = self.q[policy]
        if q >= self.tau:
            return 1.0 + alpha * (1 - self.e[policy] / self.e_dense) / self.s_max + EPS * q
        return -KAPPA * (self.tau - q)


class StructuredEnv(gym.Env):
    """One structured decision per unit; reward at the end of the episode (constraint first)."""

    def __init__(self, inp, alpha):
        super().__init__()
        self.inp, self.alpha = inp, alpha
        self.action_space = spaces.Discrete(len(L.RATIO))
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(STATE_DIM,), dtype=np.float32)
        self.actions = []

    def _state(self):
        u = len(self.actions)
        if u >= N_UNITS:
            return np.zeros(STATE_DIM, dtype=np.float32)
        ratios = [L.RATIO[a] for a in self.actions]
        partial = tuple(self.actions + [0] * (N_UNITS - u))
        i = self.inp
        return np.array([u / (N_UNITS - 1), np.log1p(i.channels[u]), i.param_share[u], i.mac_share[u], i.sens[u],
                         ratios[-1] if ratios else 0.0, float(np.mean(ratios)) if ratios else 0.0,
                         i.e[partial] / i.e_dense], dtype=np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.actions = []
        return self._state(), {}

    def step(self, action):
        self.actions.append(int(action))
        if len(self.actions) < N_UNITS:
            return self._state(), 0.0, False, False, {}
        p = tuple(self.actions)
        r = self.inp.reward(p, self.alpha)
        q = self.inp.q[p]
        t = self.inp.t[p]
        info = {"actions": list(p), "q": q, "feasible": q >= self.inp.tau, "e_pred": self.inp.e[p], "reward": r,
                "macs": t["macs"], "memory_bytes": t["memory_bytes"], "latency_ms": t["latency_ms"]}
        return np.zeros(STATE_DIM, dtype=np.float32), r, True, False, info


def prior_matrix(sens_centred):
    return (BETA * np.asarray(sens_centred, float)[:, None] * A_NORM[None, :]).tolist()


def select(archive, inp):
    """Lowest predicted energy; within 1% count as tied -> higher proxy q, lower MACs, lower memory, smaller vector."""
    if not archive:
        return None
    e_min = min(a["e_pred"] for a in archive)
    tied = [a for a in archive if a["e_pred"] <= e_min * (1 + TIE_BAND)]
    return min(tied, key=lambda a: (-a["q"], a["macs"], a["memory_bytes"], a["actions"]))


def _matplotlib_guard():
    """SB3's logger imports matplotlib.figure only for its Figure output type, which is never used here. If Windows
    Smart App Control blocks a matplotlib extension, a placeholder module stands in; no computation touches it."""
    try:
        import matplotlib.figure  # noqa: F401
    except ImportError:
        import types
        for k in [k for k in sys.modules if k == "matplotlib" or k.startswith("matplotlib.")]:
            del sys.modules[k]
        mpl, fig = types.ModuleType("matplotlib"), types.ModuleType("matplotlib.figure")
        fig.Figure = type("Figure", (), {})
        mpl.figure = fig
        sys.modules.update({"matplotlib": mpl, "matplotlib.figure": fig})


_matplotlib_guard()                         # at import, before anything (here or in an importer) loads SB3


def run(inp, alpha, seed, tag):
    import torch
    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import BaseCallback
    from src import rl as RL
    torch.set_num_threads(1)

    class Log(BaseCallback):
        def __init__(self):
            super().__init__()
            self.rows = []

        def _on_step(self):
            for done, info in zip(self.locals["dones"], self.locals["infos"]):
                if done and "actions" in info:
                    keep = ("actions", "q", "feasible", "e_pred", "reward", "macs", "memory_bytes", "latency_ms")
                    self.rows.append({**{k: info[k] for k in keep}, "episode": len(self.rows) + 1})
            return True
    t0 = time.time()
    env = StructuredEnv(inp, alpha)
    kw = dict(RL.PPO_KWARGS)
    kw["policy"] = RL.SoftPriorPolicy
    kw["policy_kwargs"] = {"prior": prior_matrix(inp.sens)}
    agent = PPO(env=env, seed=seed, **kw)
    log = Log()
    agent.learn(total_timesteps=EPISODES * N_UNITS, callback=log)
    assert len(log.rows) == EPISODES, len(log.rows)
    obs, _ = env.reset()
    for _ in range(N_UNITS):
        a, _ = agent.predict(obs, deterministic=True)
        obs, _, done, _, info = env.step(int(a))
    archive, seen = [], set()
    for r in log.rows:
        k = tuple(r["actions"])
        if r["feasible"] and k not in seen:
            seen.add(k)
            archive.append({**r, "first_episode": r["episode"]})
    sel = select(archive, inp)
    rec = {"tag": tag, "alpha": alpha, "seed": seed, "episodes": EPISODES, "distinct_visited": len({tuple(r["actions"]) for r in log.rows}),
           "feasible_episodes": sum(r["feasible"] for r in log.rows), "archive_size": len(archive),
           "final_deterministic_actions": info["actions"], "final_deterministic_feasible": info["feasible"],
           "selected": sel, "mean_reward_last_64": float(np.mean([r["reward"] for r in log.rows[-64:]])),
           "seconds": round(time.time() - t0, 1)}
    return rec, log.rows, archive


def write_rows(path, rows):
    if not rows:
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows([{k: (json.dumps(v) if isinstance(v, list) else v) for k, v in r.items()} for r in rows])


def cmd_smoke(_a):
    """Synthetic landscape (q decreasing with pruning), real table and surrogate-like energy: checks shapes and speed."""
    tab = L.table()
    q = {p: 1.0 - 0.02 * sum(p) for p in L.POLICIES}
    e = {p: 11.0 * (0.3 + 0.7 * tab[p]["macs"] / tab[(0, 0, 0, 0)]["macs"]) for p in L.POLICIES}
    inp = Inputs(q, 0.95, e, tab, [0.338, 0.44, -0.219, -0.56])
    rec, rows, archive = run(inp, 1.0, 1, "smoke")
    print(json.dumps({k: v for k, v in rec.items() if k != "selected"}, indent=1), "\nselected", rec["selected"])


def cmd_dev(_a):
    inp = Inputs.load()
    out = []
    for alpha in ALPHAS:
        for s in DEV_SEEDS:
            rec, rows, archive = run(inp, alpha, s, "dev")
            write_rows(os.path.join(RUNS, "dev", f"alpha{alpha}_seed{s}_episodes.csv"), rows)
            out.append(rec)
            sel = rec["selected"]
            print(f"alpha {alpha} seed {s}: archive {rec['archive_size']} selected "
                  f"{sel['actions'] if sel else None} e_pred {sel['e_pred'] if sel else None}", flush=True)
    summ = {}
    for alpha in ALPHAS:
        es = [r["selected"]["e_pred"] if r["selected"] else float("inf") for r in out if r["alpha"] == alpha]
        summ[alpha] = {"median_selected_e_pred": float(np.median(es)), "empty_archives": int(sum(np.isinf(es)))}
    best = min(ALPHAS, key=lambda a: (summ[a]["median_selected_e_pred"], a))
    res = {"created": datetime.datetime.now().astimezone().isoformat(timespec="seconds"), "stage": "A (development)",
           "rule": "alpha with the lowest median predicted energy of the seeds' selected policies; ties -> smaller alpha",
           "summary": summ, "chosen_alpha": best, "runs": out}
    json.dump(res, open(os.path.join(L.RESULTS, "ppo_dev_tuning.json"), "w", encoding="utf-8"), indent=1, default=float)
    print("chosen alpha", best, summ)


def cmd_confirm(_a):
    pre = json.load(open(L.FINAL_PREREG, encoding="utf-8"))
    alpha = pre["reward"]["alpha"]
    assert pre["seeds"]["confirmatory"] == CONFIRM_SEEDS
    inp = Inputs.load()
    assert abs(inp.tau - pre["proxy"]["tau"]) < 1e-12
    out, all_arch = [], []
    for s in CONFIRM_SEEDS:
        rec, rows, archive = run(inp, alpha, s, "confirm")
        write_rows(os.path.join(RUNS, "confirm", f"seed{s}_episodes.csv"), rows)
        out.append(rec)
        all_arch += [{"seed": s, **a} for a in archive]
        sel = rec["selected"]
        print(f"seed {s}: archive {rec['archive_size']} selected {sel['actions'] if sel else None}", flush=True)
    write_rows(os.path.join(L.RESULTS, "archive_all_policies.csv"), all_arch)
    write_rows(os.path.join(L.RESULTS, "ppo_all_seeds.csv"),
               [{**{k: v for k, v in r.items() if k != "selected"},
                 "selected_actions": r["selected"]["actions"] if r["selected"] else None,
                 "selected_q": r["selected"]["q"] if r["selected"] else None,
                 "selected_e_pred": r["selected"]["e_pred"] if r["selected"] else None} for r in out])
    json.dump({"created": datetime.datetime.now().astimezone().isoformat(timespec="seconds"), "alpha": alpha,
               "preregistration_sha256": L.sha256_file(L.FINAL_PREREG), "runs": out},
              open(os.path.join(L.RESULTS, "ppo_confirmatory_runs.json"), "w", encoding="utf-8"), indent=1, default=float)


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    for c in ("smoke", "dev", "confirm"):
        sub.add_parser(c)
    a = p.parse_args()
    {"smoke": cmd_smoke, "dev": cmd_dev, "confirm": cmd_confirm}[a.cmd](a)


if __name__ == "__main__":
    main()
