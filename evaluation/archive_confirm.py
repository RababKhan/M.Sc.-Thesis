"""Archive-confirmatory experiment (results/archive_confirmatory_preregistration.md).

Sensitivity-guided PPO + a best-policy archive, selected on a held-out part of
the validation split. Everything the protocol requires to be separated is
separated in code:

  V_RL      3,000 validation images: sensitivity, PPO reward, training
  V_SELECT  2,000 validation images: post-training evaluation of archived policies only
  TEST      10,000 images: only after the selected policy is written to disk and hashed

ArchiveData enforces the phases: V_SELECT and TEST raise during training; TEST
also raises until a selection has been frozen for the run.

Subcommands
  prepare   stratified V_RL / V_SELECT split and V_RL-only loss sensitivity (before the pre-registration)
  check     the fifteen pre-registered implementation checks
  run       the 160 pre-registered runs, resumable: --worker K --workers 4
"""
import argparse
import contextlib
import copy
import datetime
import json
import os
import sys
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from stable_baselines3 import PPO
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common  # noqa: E402
import constrained_ppo as cp  # noqa: E402
import ppo_experiments as base  # noqa: E402
import rl_env  # noqa: E402
import soft_prior as sp  # noqa: E402

ROOT = base.ROOT
RESULTS = os.path.join(ROOT, "results")
PREREG = os.path.join(RESULTS, "archive_confirmatory_preregistration.md")
SPLIT_NPZ = os.path.join(RESULTS, "archive_validation_split.npz")
SENS_JSON = os.path.join(RESULTS, "archive_sensitivity_vector.json")
RUNS_ROOT = os.path.join(RESULTS, "archive_runs")
PRED_ROOT = os.path.join(RESULTS, "archive_predictions")
AGENTS_DIR = os.path.join(ROOT, "checkpoints", "archive_agents")
BASELINE_SHA = "ca28f8345c23365ee7b92686e780ae0a9befa76c9d6bb9b3bb0b9e42272e111c"

SEEDS = list(range(200, 240))
CONDITIONS = ["A0", "A1", "A2", "A3"]
BETA = 0.50
LAMBDA_S = 0.01
SPLIT_SEED = 20260928
N_RL, N_SELECT = 3000, 2000
PROBE_RATIOS = [0.1, 0.2, 0.4, 0.6]            # the refined multi-level algorithm
BAND = (57.5, 58.5)
TARGET = 58.0
LAYERS = rl_env.FILTERED_LAYERS


def sha256(path):
    return base.sha256(path)


# ------------------------------------------------------------------ data phases
class ArchiveData:
    def __init__(self, splits, split):
        vx, vy = splits._val
        self._rl = (vx[split["v_rl_positions"]], vy[split["v_rl_positions"]])
        self._select = (vx[split["v_select_positions"]], vy[split["v_select_positions"]])
        self._splits = splits
        self.phase = "idle"             # idle | training | selection
        self.frozen = False

    @staticmethod
    def _loader(x, y):
        return DataLoader(TensorDataset(x, y), batch_size=128, shuffle=False, num_workers=0)

    def rl_loader(self):
        return self._loader(*self._rl)

    def select_loader(self):
        if self.phase == "training":
            raise RuntimeError("V_SELECT requested during PPO training")
        return self._loader(*self._select)

    def test_loader(self):
        if self.phase == "training":
            raise RuntimeError("TEST requested during PPO training")
        if not self.frozen:
            raise RuntimeError("TEST requested before the selected policy was frozen")
        return self._splits.test_loader()

    @contextlib.contextmanager
    def training(self):
        self.phase, self.frozen = "training", False
        with self._splits.training():
            yield
        self.phase = "selection"


# ------------------------------------------------------------------ prepare
def stratified_split(labels):
    rng = np.random.default_rng(SPLIT_SEED)
    classes = np.arange(10)
    counts = np.array([(labels == c).sum() for c in classes])
    exact = counts * N_RL / len(labels)
    n_rl = np.floor(exact).astype(int)
    order = sorted(classes, key=lambda c: (-(exact[c] - n_rl[c]), c))
    for c in order[: N_RL - n_rl.sum()]:
        n_rl[c] += 1
    rl, sel = [], []
    for c in classes:
        pos = rng.permutation(np.flatnonzero(labels == c))
        rl.extend(pos[: n_rl[c]])
        sel.extend(pos[n_rl[c]:])
    return np.sort(np.array(rl)), np.sort(np.array(sel)), counts, n_rl


def mean_loss_acc(model, x, y):
    model.eval()
    loss_fn = nn.CrossEntropyLoss(reduction="sum")
    loss, correct = 0.0, 0
    with torch.no_grad():
        for i in range(0, len(x), 128):
            out = model(x[i:i + 128])
            loss += loss_fn(out, y[i:i + 128]).item()
            correct += (out.argmax(1) == y[i:i + 128]).sum().item()
    return loss / len(x), 100.0 * correct / len(x)


def cmd_prepare(_args):
    for path in (SPLIT_NPZ, SENS_JSON):
        if os.path.exists(path):
            raise SystemExit(f"{path} already exists; the split and sensitivity are fixed.")
    assert sha256(os.path.join(ROOT, "checkpoints", "cnn_baseline_FIXED.pth")) == BASELINE_SHA
    splits = rl_env.Splits(os.path.join(ROOT, "data"))
    vx, vy = splits._val
    labels = vy.numpy()
    rl, sel, counts, n_rl = stratified_split(labels)
    assert len(rl) == N_RL and len(sel) == N_SELECT and not set(rl) & set(sel)
    val_idx = np.array(splits.val_indices)
    np.savez(SPLIT_NPZ, v_rl_positions=rl, v_select_positions=sel,
             v_rl_cifar_train_indices=val_idx[rl], v_select_cifar_train_indices=val_idx[sel],
             v_rl_labels=labels[rl], v_select_labels=labels[sel], split_seed=SPLIT_SEED)
    print("class counts  val:", counts.tolist())
    print("              V_RL:", np.bincount(labels[rl], minlength=10).tolist())
    print("          V_SELECT:", np.bincount(labels[sel], minlength=10).tolist())

    # V_RL-only refined loss sensitivity: the exact multi-level algorithm
    model = common.load_baseline(os.path.join(ROOT, "checkpoints", "cnn_baseline_FIXED.pth"))
    x, y = vx[rl], vy[rl]
    loss0, acc0 = mean_loss_acc(model, x, y)
    raw, detail = [], {}
    for layer in LAYERS:
        rel = []
        for r in PROBE_RATIOS:
            pruned = rl_env.apply_layer_pruning_by_name_inplace(copy.deepcopy(model), layer, r)
            loss_r, acc_r = mean_loss_acc(pruned, x, y)
            rel.append((loss_r - loss0) / loss0)
            detail[f"{layer}@{int(100 * r)}"] = {"loss": loss_r, "accuracy": acc_r}
        raw.append(float(np.mean(rel)))
    lo, hi = min(raw), max(raw)
    norm = [0.0] * 4 if hi - lo <= 0 else [(v - lo) / (hi - lo) for v in raw]
    out = {"layer_order": LAYERS, "raw_loss_sensitivity": raw, "normalized_sensitivity": norm,
           "definition": "mean over probe ratios 10/20/40/60% of (loss_r - loss_0)/loss_0 on V_RL; min-max to [0,1]",
           "n_images": int(len(x)), "baseline_v_rl_loss": loss0, "baseline_v_rl_accuracy": acc0,
           "probe_detail": detail, "split_sha256": sha256(SPLIT_NPZ),
           "computed_at": datetime.datetime.now().astimezone().isoformat(timespec="seconds")}
    with open(SENS_JSON, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1)
    sel_loss, sel_acc = mean_loss_acc(model, vx[sel], vy[sel])
    print("V_RL baseline accuracy", acc0, "| V_SELECT baseline accuracy", sel_acc)
    print("raw", raw, "\nnormalized", norm)
    print("split sha256", sha256(SPLIT_NPZ), "\nsensitivity sha256", sha256(SENS_JSON))


# ------------------------------------------------------------------ design
def load_fixed():
    split = dict(np.load(SPLIT_NPZ))
    sens = json.load(open(SENS_JSON, encoding="utf-8"))
    assert sens["layer_order"] == LAYERS
    return split, sens


def condition(name, seed, S):
    S = np.asarray(S, float)
    if name == "A0":
        prior, beta, mapping = np.zeros(4), 0.0, None
    elif name == "A1":
        prior, beta, mapping = S, BETA, None
    elif name == "A2":
        (prior, mapping), beta = sp.shuffled(S, seed), BETA
    else:
        prior, beta, mapping = np.full(4, S.mean()), BETA, None
    return {"condition": name, "beta": beta, "prior_sensitivity": [float(v) for v in prior],
            "state_sensitivity": [0.0] * 4, "prior_mapping": mapping}


# ------------------------------------------------------------------ selection
def select(scores):
    """Pre-registered rule. `scores`: DataFrame of unique archived policies with
    actions (list), sparsity, vselect_accuracy, vselect_loss."""
    s = scores.copy()
    s["dist"] = (s["sparsity"] - TARGET).abs()
    s["lex"] = s["actions"].map(lambda a: tuple(a))
    band = s[(s["sparsity"] >= BAND[0]) & (s["sparsity"] <= BAND[1])]
    if len(band):
        best = band.sort_values(["vselect_accuracy", "dist", "vselect_loss", "lex"],
                                ascending=[False, True, True, True]).iloc[0]
        return best, True, len(band)
    best = s.sort_values(["dist", "vselect_accuracy", "vselect_loss", "lex"],
                         ascending=[True, False, True, True]).iloc[0]
    return best, False, 0


_SELECT_CACHE = {}
SELECT_EVALUATIONS = [0]


def evaluate_select(ctx, data, actions):
    data.select_loader()                          # phase guard first: raises during training
    key = tuple(actions)
    if key not in _SELECT_CACHE:
        pruned = common.prune_layerwise(ctx["model"], [rl_env.ACTION_TO_PRUNE[a] for a in actions])
        loss, acc = mean_loss_acc(pruned, *data._select)
        _SELECT_CACHE[key] = (acc, loss, common.calculate_sparsity(pruned))
        SELECT_EVALUATIONS[0] += 1
    return _SELECT_CACHE[key]


# ------------------------------------------------------------------ one run
def run_paths(name, seed):
    d = os.path.join(RUNS_ROOT, name)
    return {"dir": d, "archive": os.path.join(d, f"seed_{seed}_archive.csv"),
            "curve": os.path.join(d, f"seed_{seed}_training.csv"),
            "selected": os.path.join(d, f"seed_{seed}_selected.json"),
            "record": os.path.join(d, f"seed_{seed}_run.json"),
            "pred": os.path.join(PRED_ROOT, name, f"seed_{seed}.csv"),
            "agent": os.path.join(AGENTS_DIR, f"{name}_seed-{seed}")}


def run_one(ctx, data, name, seed, out=None, policy="soft"):
    paths = out or run_paths(name, seed)
    if os.path.exists(paths["record"]):
        return None
    for key in ("dir",):
        os.makedirs(paths[key], exist_ok=True)
    os.makedirs(os.path.dirname(paths["pred"]), exist_ok=True)
    c = condition(name, seed, ctx["S"])
    started = time.time()
    started_at = datetime.datetime.now().astimezone().isoformat(timespec="seconds")

    # ---- training: V_RL only
    env = cp.MaskedCompressionEnv(ctx["model"], ctx["filtered"], ctx["baseline_v_rl"], data.rl_loader(),
                                  ctx["max_param_count"], layer_sensitivities={n: 0.0 for n in LAYERS},
                                  sparsity_coef=LAMBDA_S)
    kwargs = {k: v for k, v in base.PPO_KWARGS.items() if k != "policy"}
    if policy == "soft":
        agent = PPO(env=env, seed=seed, policy=sp.SoftPriorPolicy,
                    policy_kwargs={"prior": sp.prior_matrix(c["beta"], c["prior_sensitivity"])}, **kwargs)
    else:
        agent = PPO(env=env, seed=seed, policy="MlpPolicy", **kwargs)
    log = base.EpisodeLog()
    with data.training():
        agent.learn(total_timesteps=base.TOTAL_TIMESTEPS, callback=log)
    episodes = pd.DataFrame(log.rows)
    episodes.to_csv(paths["curve"], index=False)

    obs, _ = env.reset()
    terminal, done = [], False
    while not done:
        a, _ = agent.predict(obs, deterministic=True)
        terminal.append(int(a))
        obs, _, done, _, _ = env.step(int(a))
    agent.save(paths["agent"])

    # ---- archive: every unique completed policy, frozen after training
    ep = episodes.copy()
    ep["actions_list"] = ep["actions"].map(json.loads)
    arch = (ep.groupby("actions", sort=False)
            .agg(episode_first_seen=("episode", "min"), timestep_first_seen=("timestep", "min"),
                 times_sampled=("episode", "size"), ppo_reward=("episode_reward", "first"),
                 sparsity=("sparsity", "first")).reset_index())
    arch["actions_list"] = arch["actions"].map(json.loads)
    arch["prune_percent"] = arch["actions_list"].map(lambda a: [int(100 * rl_env.ACTION_TO_PRUNE[x]) for x in a])

    # ---- selection: V_SELECT only, after training
    sc = [evaluate_select(ctx, data, a) for a in arch["actions_list"]]
    arch["vselect_accuracy"] = [s[0] for s in sc]
    arch["vselect_loss"] = [s[1] for s in sc]
    assert np.allclose(arch["sparsity"], [s[2] for s in sc]), "archive sparsity mismatch"
    arch[["actions", "prune_percent", "sparsity", "episode_first_seen", "timestep_first_seen", "times_sampled",
          "ppo_reward", "vselect_accuracy", "vselect_loss"]].to_csv(paths["archive"], index=False)
    best, in_band, n_band = select(arch.rename(columns={"actions_list": "actions_l"}).assign(
        actions=arch["actions_list"]))
    t_acc, t_loss, t_sp = evaluate_select(ctx, data, terminal)

    # ---- freeze before test
    frozen = {"condition": name, "seed": seed, "selected_actions": list(map(int, best["actions"])),
              "selected_sparsity": float(best["sparsity"]), "selected_vselect_accuracy": float(best["vselect_accuracy"]),
              "selected_vselect_loss": float(best["vselect_loss"]), "target_band_reached": bool(in_band),
              "n_band_candidates": int(n_band), "terminal_actions": terminal,
              "frozen_at": datetime.datetime.now().astimezone().isoformat(timespec="seconds")}
    with open(paths["selected"], "w", encoding="utf-8") as f:
        json.dump(frozen, f, indent=1)
    frozen_sha = sha256(paths["selected"])
    data.frozen = True

    # ---- test: once per frozen policy, fresh baseline copy
    loader = data.test_loader()
    sel_model = common.prune_layerwise(ctx["model"], [rl_env.ACTION_TO_PRUNE[a] for a in frozen["selected_actions"]])
    sel_pred, labels = common.predictions(sel_model, loader)
    term_model = common.prune_layerwise(ctx["model"], [rl_env.ACTION_TO_PRUNE[a] for a in terminal])
    term_pred, _ = common.predictions(term_model, loader)
    data.frozen = False
    pd.DataFrame({"true_label": labels.numpy(), "selected_pred": sel_pred.numpy(),
                  "terminal_pred": term_pred.numpy()}).to_csv(paths["pred"], index=False)

    record = {**frozen, **c, "frozen_policy_sha256": frozen_sha,
              "selected_test_accuracy": common.accuracy_from(sel_pred, labels),
              "terminal_test_accuracy": common.accuracy_from(term_pred, labels),
              "terminal_sparsity": float(t_sp), "terminal_vselect_accuracy": float(t_acc),
              "selected_first_episode": int(best["episode_first_seen"]),
              "selected_first_timestep": int(best["timestep_first_seen"]),
              "selected_times_sampled": int(best["times_sampled"]),
              "n_unique_policies": int(len(arch)), "episodes": int(len(episodes)),
              "timesteps_trained": int(agent.num_timesteps), "policy_class": policy,
              "runtime_seconds": round(time.time() - started, 1), "started_at": started_at,
              **ctx["provenance"]}
    with open(paths["record"], "w", encoding="utf-8") as f:
        json.dump(record, f, indent=1)
    return record


def context(threads):
    """Minimal context. Unlike ppo_experiments.build_context, it evaluates nothing on the old
    1,000-image probe (which overlaps V_SELECT) and nothing on the test set."""
    torch.set_num_threads(threads)
    ckpt = os.path.join(ROOT, "checkpoints", "cnn_baseline_FIXED.pth")
    assert sha256(ckpt) == BASELINE_SHA, "baseline checkpoint hash changed"
    split, sens = load_fixed()
    model = common.load_baseline(ckpt)
    all_layers, filtered = rl_env.prunable_layers(model)
    splits = rl_env.Splits(os.path.join(ROOT, "data"))
    data = ArchiveData(splits, split)
    _, acc = mean_loss_acc(model, *data._rl)
    assert acc == sens["baseline_v_rl_accuracy"], "V_RL baseline accuracy does not reproduce"
    here = os.path.dirname(os.path.abspath(__file__))
    prov = {
        "baseline_checkpoint": "checkpoints/cnn_baseline_FIXED.pth", "baseline_sha256": BASELINE_SHA,
        "preregistration_sha256": sha256(PREREG), "split_sha256": sha256(SPLIT_NPZ),
        "sensitivity_sha256": sha256(SENS_JSON), "reward_baseline_v_rl_accuracy": acc,
        "git": base.git_state(),
        "script_sha256": {n: sha256(os.path.join(here, n)) for n in
                          ("archive_confirm.py", "soft_prior.py", "constrained_ppo.py", "rl_env.py", "common.py",
                           "ppo_experiments.py")},
        "environment": {**base.environment(), "torch_threads": torch.get_num_threads()},
        "test_set_use": "only after the selected and terminal policies are frozen to disk",
    }
    ctx = {"model": model, "filtered": filtered, "splits": splits,
           "max_param_count": max(m.weight.numel() for _, m in all_layers),
           "S": sens["normalized_sensitivity"], "baseline_v_rl": acc, "provenance": prov}
    return ctx, data


def cmd_run(args):
    ctx, data = context(args.threads)
    jobs = [(n, s) for s in SEEDS for n in CONDITIONS]
    mine = [j for i, j in enumerate(jobs) if i % args.workers == args.worker]
    print(f"worker {args.worker}/{args.workers}: {len(mine)} of {len(jobs)} runs", flush=True)
    for i, (n, s) in enumerate(mine, 1):
        if os.path.exists(run_paths(n, s)["record"]):
            print(f"[{args.worker}:{i}/{len(mine)}] {n} seed {s} already done", flush=True)
            continue
        t0 = time.time()
        run_one(ctx, data, n, s)
        print(f"[{args.worker}:{i}/{len(mine)}] {n} seed {s} done ({time.time() - t0:.0f}s)", flush=True)
    print(f"WORKER {args.worker} DONE", flush=True)


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("prepare")
    c = sub.add_parser("check")
    c.add_argument("--threads", type=int, default=4)
    r = sub.add_parser("run")
    r.add_argument("--threads", type=int, default=1)
    r.add_argument("--worker", type=int, default=0)
    r.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()
    if args.cmd == "check":
        import archive_checks
        archive_checks.run(args)
    else:
        {"prepare": cmd_prepare, "run": cmd_run}[args.cmd](args)


if __name__ == "__main__":
    main()
