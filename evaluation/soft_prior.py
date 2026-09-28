"""Soft Sensitivity-Prior PPO (see results/soft_prior_preregistration.md).

SoftPriorPolicy is SB3's ActorCriticPolicy (MlpPolicy) with one change:
the categorical logits for the current layer l are shifted by

    adjusted_logit(l, a) = raw_logit(l, a) - beta * S_l * a_norm(a),   a_norm = ratio / 60

inside _get_action_dist_from_latent, i.e. before the distribution exists. It
therefore shapes rollout sampling, the log-probabilities and entropy in the PPO
update, and the deterministic final policy. Nothing is masked, clipped,
remapped, or added to the reward. The layer is read from the observation's
layer-index input (index / 3). With a zero prior matrix the policy computes
exactly what MlpPolicy computes (x - 0.0 == x), which `check` verifies on
logits and on complete training runs.

Subcommands
  check   the seven pre-registered implementation checks -> results/soft_prior_implementation_checks.md
  run     all pre-registered runs (resumable), --worker K --workers N --seeds 20|10
"""
import argparse
import datetime
import json
import os
import sys
import time

import numpy as np
import pandas as pd
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.policies import ActorCriticPolicy

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common  # noqa: E402
import constrained_ppo as cp  # noqa: E402  (verified memoised env, ALL_VALID; not modified)
import ppo_experiments as base  # noqa: E402
import rl_env  # noqa: E402

ROOT = base.ROOT
RESULTS = os.path.join(ROOT, "results")
RUNS_DIR = os.path.join(RESULTS, "soft_prior_runs")
CURVES_DIR = os.path.join(RESULTS, "soft_prior_training_curves_raw")
AGENTS_DIR = os.path.join(ROOT, "checkpoints", "soft_prior_experiments")
VALUES_CSV = os.path.join(RESULTS, "refined_sensitivity_values.csv")
PREREG = os.path.join(RESULTS, "soft_prior_preregistration.md")

LAYERS = rl_env.FILTERED_LAYERS
A_NORM = np.array([rl_env.ACTION_TO_PRUNE[a] / 0.6 for a in range(len(rl_env.ACTION_TO_PRUNE))])
LAMBDA_S = 0.01
SEEDS_20 = [42] + list(range(1, 20))
SEEDS_10 = SEEDS_20[:10]


# ------------------------------------------------------------------ policy
def prior_matrix(beta, sensitivity):
    """B[l, a] = beta * S_l * a_norm(a); the amount subtracted from the logits."""
    return (beta * np.asarray(sensitivity, float)[:, None] * A_NORM[None, :]).tolist()


class SoftPriorPolicy(ActorCriticPolicy):
    def __init__(self, *args, prior=None, **kwargs):
        self._prior_list = prior if prior is not None else [[0.0] * len(A_NORM) for _ in LAYERS]
        super().__init__(*args, **kwargs)
        self.register_buffer("prior", torch.tensor(self._prior_list, dtype=torch.float32), persistent=False)
        self._obs = None

    def _get_constructor_parameters(self):
        data = super()._get_constructor_parameters()
        data["prior"] = self._prior_list
        return data

    def layer_index(self, obs):
        return torch.round(obs[..., 0] * (len(LAYERS) - 1)).long().clamp(0, len(LAYERS) - 1)

    def adjust(self, logits, obs):
        return logits - self.prior[self.layer_index(obs)]

    def _get_action_dist_from_latent(self, latent_pi):
        logits = self.action_net(latent_pi)
        logits = self.adjust(logits, self._obs)
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


# ------------------------------------------------------------------ design
def sensitivities():
    v = pd.read_csv(VALUES_CSV).set_index("layer").loc[LAYERS]
    return {"loss": v["loss_normalized"].to_numpy(float), "accuracy": v["accuracy_normalized"].to_numpy(float)}


def shuffled(values, seed):
    """Seed-deterministic permutation in which every layer gets another layer's (different) value."""
    rng = np.random.default_rng(seed)
    for _ in range(10000):
        perm = rng.permutation(len(values))
        if all(perm[i] != i and not np.isclose(values[perm[i]], values[i]) for i in range(len(values))):
            return values[perm], {LAYERS[i]: LAYERS[int(perm[i])] for i in range(len(values))}
    raise RuntimeError("no value-changing permutation")


def conditions(seed):
    S = sensitivities()
    zero = np.zeros(len(LAYERS))
    out = []

    def add(cond, d, beta, prior_s, state_s, mapping=None):
        out.append({"condition": cond, "definition": d, "beta": beta,
                    "prior_sensitivity": [float(x) for x in prior_s],
                    "state_sensitivity": [float(x) for x in state_s], "prior_mapping": mapping})

    add("P0", "none", 0.0, zero, zero)
    for d, betas, conds in (("loss", (0.5,), ("P1", "P2", "P3", "P4", "P5")),
                            ("loss", (1.0,), ("P1", "P2", "P3")),
                            ("accuracy", (0.5,), ("P1", "P2", "P3"))):
        for beta in betas:
            sh, mapping = shuffled(S[d], seed)
            for c in conds:
                if c == "P1":
                    add(c, d, beta, S[d], zero)
                elif c == "P2":
                    add(c, d, beta, sh, zero, mapping)
                elif c == "P3":
                    add(c, d, beta, np.full(len(LAYERS), S[d].mean()), zero)
                elif c == "P4":
                    add(c, d, 0.0, zero, S[d])
                elif c == "P5":
                    add(c, d, beta, S[d], S[d])
    return out


def run_id(c, seed):
    return f"softprior_{c['condition']}_def-{c['definition']}_beta-{c['beta']:.2f}_seed-{seed}"


# ------------------------------------------------------------------ training
def train(ctx, c, seed, policy=SoftPriorPolicy, out=(RUNS_DIR, CURVES_DIR, AGENTS_DIR), rid=None, coef=LAMBDA_S,
          state=None):
    runs_dir, curves_dir, agents_dir = out
    rid = rid or run_id(c, seed)
    path = os.path.join(runs_dir, f"{rid}.json")
    if os.path.exists(path):
        print(f"  {rid}: already done, skipping", flush=True)
        return None
    started = time.time()
    started_at = datetime.datetime.now().astimezone().isoformat(timespec="seconds")
    state = state if state is not None else dict(zip(LAYERS, c["state_sensitivity"]))
    env = cp.MaskedCompressionEnv(ctx["model"], ctx["filtered"], ctx["baseline_val"], ctx["splits"].probe_loader(),
                                  ctx["max_param_count"], layer_sensitivities=state, sparsity_coef=coef)
    kwargs = dict(base.PPO_KWARGS)
    if policy is SoftPriorPolicy:
        kwargs["policy"] = SoftPriorPolicy
        kwargs["policy_kwargs"] = {"prior": prior_matrix(c["beta"], c["prior_sensitivity"])}
    agent = PPO(env=env, seed=seed, **kwargs)
    log = base.EpisodeLog()
    with ctx["splits"].training():              # the test set raises if requested in here
        agent.learn(total_timesteps=base.TOTAL_TIMESTEPS, callback=log)
    train_seconds = time.time() - started

    obs, _ = env.reset()
    actions, done = [], False
    while not done:
        a, _ = agent.predict(obs, deterministic=True)
        actions.append(int(a))
        obs, _, done, _, info = env.step(int(a))

    fractions = [rl_env.ACTION_TO_PRUNE[a] for a in actions]
    pruned = common.prune_layerwise(ctx["model"], fractions)            # fresh baseline copy
    sparsity = common.calculate_sparsity(pruned)
    val_accuracy = rl_env.evaluate_model(pruned, ctx["splits"].val_loader())
    preds, labels = common.predictions(pruned, ctx["splits"].test_loader())   # once, frozen policy
    test_accuracy = common.accuracy_from(preds, labels)
    reward = rl_env.reward_components(info["final_accuracy"], ctx["baseline_val"], sparsity, actions, coef)
    episodes = pd.DataFrame(log.rows)
    record = {
        "run_id": rid, "experiment": "Soft Sensitivity-Prior PPO", **c, "sparsity_coef": coef, "seed": seed,
        "policy_class": policy.__name__ if isinstance(policy, type) else str(policy),
        "prior_matrix": prior_matrix(c["beta"], c["prior_sensitivity"]),
        "actions": actions, "prune_percent": [round(100 * f, 1) for f in fractions],
        "total_sparsity": sparsity, "per_layer_sparsity": common.per_layer_sparsity(pruned),
        "probe_accuracy": info["final_accuracy"], "val_accuracy": val_accuracy, "test_accuracy": test_accuracy,
        "accuracy_retention": 100.0 * test_accuracy / ctx["baseline_test"], "training_reward": reward,
        "episodes": len(episodes), "timesteps_trained": int(agent.num_timesteps),
        "ppo": {k: (v.__name__ if isinstance(v, type) else v) for k, v in kwargs.items()
                if k not in ("verbose", "policy_kwargs")},
        "ppo_defaults": {"gae_lambda": agent.gae_lambda, "clip_range": 0.2, "vf_coef": agent.vf_coef,
                         "max_grad_norm": agent.max_grad_norm, "net_arch": "[64, 64] tanh"},
        "test_predictions": "".join(str(int(p)) for p in preds.tolist()),
        "runtime_seconds": round(time.time() - started, 1), "train_seconds": round(train_seconds, 1),
        "started_at": started_at, **ctx["provenance"],
    }
    for d in (runs_dir, curves_dir, agents_dir):
        os.makedirs(d, exist_ok=True)
    episodes.to_csv(os.path.join(curves_dir, f"{rid}.csv"), index=False)
    agent.save(os.path.join(agents_dir, rid))
    with open(path, "w", encoding="utf-8") as f:
        json.dump(record, f, indent=1)
    print(f"  {rid}: {actions} sparsity {sparsity:.2f}% val {val_accuracy:.2f}% test {test_accuracy:.2f}% "
          f"({record['runtime_seconds']:.0f}s)", flush=True)
    return record


def context(threads):
    torch.set_num_threads(threads)
    ctx = base.build_context()
    here = os.path.dirname(os.path.abspath(__file__))
    for name in ("soft_prior.py", "constrained_ppo.py"):
        ctx["provenance"]["script_sha256"][name] = base.sha256(os.path.join(here, name))
    ctx["provenance"]["environment"]["torch_threads"] = torch.get_num_threads()
    ctx["provenance"]["preregistration_sha256"] = base.sha256(PREREG)
    ctx["provenance"]["sensitivity_values_sha256"] = base.sha256(VALUES_CSV)
    return ctx


# ------------------------------------------------------------------ checks
def cmd_check(args):
    ctx = context(args.threads)
    lines = ["# Soft Sensitivity-Prior PPO — implementation checks", "",
             f"Run {datetime.datetime.now().astimezone().isoformat(timespec='seconds')} with "
             f"`python evaluation/soft_prior.py check`, before any experimental run.", ""]
    ok_all = True

    def report(n, title, ok, detail):
        nonlocal ok_all
        ok_all &= bool(ok)
        lines.extend([f"## {n}. {title} — {'PASS' if ok else 'FAIL'}", "", detail, ""])
        print(f"check {n}: {'PASS' if ok else 'FAIL'} - {title}", flush=True)

    S = sensitivities()
    filtered = ctx["filtered"]
    obs = np.stack([policy_obs(filtered, i, 0.0, {n: 0.0 for n in LAYERS}, ctx["max_param_count"])
                    for i in range(len(LAYERS))])
    env = cp.MaskedCompressionEnv(ctx["model"], filtered, ctx["baseline_val"], ctx["splits"].probe_loader(),
                                  ctx["max_param_count"], layer_sensitivities={n: 0.0 for n in LAYERS})

    def build(prior, seed=7):
        torch.manual_seed(seed)
        m = PPO(env=env, seed=seed, policy=SoftPriorPolicy, policy_kwargs={"prior": prior},
                **{k: v for k, v in base.PPO_KWARGS.items() if k != "policy"})
        return m.policy

    def probs(policy, o):
        with torch.no_grad():
            return policy.get_distribution(torch.as_tensor(o)).distribution.probs.numpy()

    # (0) the penalty itself
    grid_ok, zero_a, zero_s = True, True, True
    for beta in (0.5, 1.0):
        for s in np.linspace(0, 1, 21):
            row = np.array(prior_matrix(beta, [s])[0])
            grid_ok &= bool(np.all(np.diff(row) >= 0))
            zero_a &= row[0] == 0.0
            zero_s &= (s > 0) or np.all(row == 0)
        for a in range(6):
            col = [prior_matrix(beta, [s])[0][a] for s in np.linspace(0, 1, 21)]
            grid_ok &= bool(np.all(np.diff(col) >= 0))
    report(0, "Penalty algebra: beta*S*a_norm is 0 at a=0, 0 at S=0, non-decreasing in S and in a",
           grid_ok and zero_a and zero_s,
           "Checked on S in {0, 0.05, ..., 1} x all six actions for beta 0.5 and 1.0. "
           "a_norm = " + str([round(x, 4) for x in A_NORM]) + ".")

    # (1) beta = 0 reproduces standard PPO: whole training runs, episode by episode
    tmp = os.path.join(RESULTS, "_soft_prior_check")
    c0 = {"condition": "P0", "definition": "none", "beta": 0.0, "prior_sensitivity": [0.0] * 4,
          "state_sensitivity": [0.0] * 4, "prior_mapping": None}
    cp._EVAL_CACHE.clear()
    std = train(ctx, c0, 42, policy="MlpPolicy", out=(os.path.join(tmp, "std"),) * 3, rid="check_std")
    cp._EVAL_CACHE.clear()
    sp0 = train(ctx, c0, 42, policy=SoftPriorPolicy, out=(os.path.join(tmp, "soft"),) * 3, rid="check_soft")
    same_run = pd.read_csv(os.path.join(tmp, "std", "check_std.csv")).equals(
        pd.read_csv(os.path.join(tmp, "soft", "check_soft.csv")))
    cp._EVAL_CACHE.clear()
    orig = train(ctx, c0, 42, policy=SoftPriorPolicy, out=(os.path.join(tmp, "orig"),) * 3, rid="check_orig",
                 coef=0.04, state=dict(ctx["measured_sensitivity"]))
    same_orig = pd.read_csv(os.path.join(tmp, "orig", "check_orig.csv")).equals(
        pd.read_csv(os.path.join(base.CURVES_DIR, "sens-correct_coef-0.04_seed-42.csv")))
    report(1, "beta = 0 reproduces standard PPO", same_run and same_orig and std["actions"] == sp0["actions"],
           f"(a) P0 configuration (zero state, lambda_s 0.01, seed 42): SoftPriorPolicy with beta 0 vs SB3 "
           f"MlpPolicy — all 512 training episodes identical: **{same_run}**; final policies "
           f"{sp0['actions']} vs {std['actions']}.\n\n(b) SoftPriorPolicy with beta 0 on the ORIGINAL "
           f"configuration (original sensitivity state, lambda_s 0.04, seed 42) reproduces the recorded original "
           f"run `results/training_curves/sens-correct_coef-0.04_seed-42.csv`, all 512 episodes: **{same_orig}**; "
           f"final policy {orig['actions']} (recorded [1, 1, 5, 5]).")

    # (2) sensitivity = 0 reproduces the original logits exactly
    std_pol = build(prior_matrix(0.0, [0] * 4))
    zero_pol = build(prior_matrix(0.5, [0.0] * 4))
    zero_pol.load_state_dict(std_pol.state_dict(), strict=False)
    rng = np.random.default_rng(0)
    rand_obs = np.concatenate([obs, obs + rng.normal(0, 0.01, obs.shape).astype(np.float32)])
    rand_obs[:, 0] = np.tile([0, 1 / 3, 2 / 3, 1], 2).astype(np.float32)
    with torch.no_grad():
        d_std = std_pol.get_distribution(torch.as_tensor(rand_obs)).distribution.logits
        d_zero = zero_pol.get_distribution(torch.as_tensor(rand_obs)).distribution.logits
    report(2, "S = 0 (with beta 0.5) produces exactly the original logits",
           bool(torch.equal(d_std, d_zero)),
           "Two policies with identical weights, one with prior S = [0,0,0,0] and beta 0.5, one without: "
           f"distribution logits bit-identical on the four layer observations and four perturbed ones: "
           f"**{bool(torch.equal(d_std, d_zero))}**.")

    # (3) + (4) all actions possible; the correct prior changes probabilities without removing any
    detail, ok3, ok4 = [], True, True
    base_p = probs(std_pol, obs)
    for d in ("loss", "accuracy"):
        for beta in (0.5, 1.0):
            for label, vec in (("correct", S[d]), ("shuffled seed 42", shuffled(S[d], 42)[0]),
                               ("constant", np.full(4, S[d].mean()))):
                pol = build(prior_matrix(beta, vec))
                pol.load_state_dict(std_pol.state_dict(), strict=False)
                p = probs(pol, obs)
                ok3 &= bool((p > 0).all())
                if label == "correct":
                    changed = np.abs(p - base_p).max(axis=1)
                    ok4 &= bool(changed[0] > 0 and (p > 0).all())
                detail.append(f"| {d} | {beta} | {label} | {p.min():.4f} | "
                              f"{np.abs(p - base_p).max(axis=1).round(4).tolist()} |")
    table = ("| Definition | beta | Prior | Min probability over layers x actions | Max prob. change per layer "
             "(f.0, f.3, f.6, c.1) |\n|---|---|---|---|---|\n" + "\n".join(detail))
    report(3, "All six actions remain possible for every layer under every prior", ok3,
           "Initial policy (seed 7), layer observations at the start of an episode. The prior subtracts a finite "
           "amount from finite logits, so every softmax probability stays > 0.\n\n" + table)
    report(4, "The correct prior changes probabilities but never removes an action", ok4,
           "Same table: the correct prior changes features.0's distribution (largest penalty) and leaves every "
           "probability > 0. On the other layers the change is tiny because S is ≤ 0.051 there.")

    # (5) shuffled prior: same multiset, every layer's value changes
    ok5, rows5 = True, []
    for d in ("loss", "accuracy"):
        for seed in SEEDS_20:
            sh, mapping = shuffled(S[d], seed)
            same = np.array_equal(np.sort(sh), np.sort(S[d]))
            moved = all(not np.isclose(sh[i], S[d][i]) for i in range(4))
            ok5 &= same and moved
            if seed in (42, 1):
                rows5.append(f"{d}, seed {seed}: {mapping}")
    report(5, "Shuffled prior uses exactly the same sensitivity multiset, with every layer's value changed", ok5,
           "All 20 seeds x both definitions checked. Examples: " + "; ".join(rows5) + ".")

    # (6) constant prior: equal average penalty
    ok6, rows6 = True, []
    for d in ("loss", "accuracy"):
        for beta in (0.5, 1.0):
            m1 = np.mean(prior_matrix(beta, S[d]))
            m3 = np.mean(prior_matrix(beta, np.full(4, S[d].mean())))
            ok6 &= bool(np.isclose(m1, m3, rtol=0, atol=1e-12))
            rows6.append(f"{d}, beta {beta}: P1 {m1:.6f}, P3 {m3:.6f}")
    report(6, "Constant-prior control has the same average penalty as the correct prior", ok6,
           "Mean of beta*S*a_norm over layers and actions: " + "; ".join(rows6) + ".")

    # (7) no test-set call during training
    raised = False
    with ctx["splits"].training():
        try:
            ctx["splits"].test_loader()
        except RuntimeError:
            raised = True
    report(7, "No test-set call can occur during PPO training", raised,
           "Every run trains inside `Splits.training()`, where requesting the test loader raises "
           f"`RuntimeError` (demonstrated: raised = **{raised}**). The test set is used once, after "
           "`agent.learn()` returns, on a fresh baseline copy pruned by the frozen policy. The three check runs in "
           "item 1 trained under the same guard without error.")

    lines.insert(3, f"**Overall: {'ALL CHECKS PASS' if ok_all else 'AT LEAST ONE CHECK FAILED'}**\n")
    with open(os.path.join(RESULTS, "soft_prior_implementation_checks.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print("ALL PASS" if ok_all else "FAILED", flush=True)


def policy_obs(filtered, i, cumulative, sens, max_param_count):
    name, module = filtered[i]
    s = rl_env.extract_layer_state(module, i, len(filtered), cumulative, name, sens)
    return rl_env.state_to_vector(s, max_param_count)


def cmd_run(args):
    seeds = SEEDS_20 if args.seeds == 20 else SEEDS_10
    ctx = context(args.threads)
    jobs = [(c, s) for s in seeds for c in conditions(s)]
    mine = [j for i, j in enumerate(jobs) if i % args.workers == args.worker]
    print(f"worker {args.worker}/{args.workers}: {len(mine)} of {len(jobs)} runs, seeds {seeds}", flush=True)
    for i, (c, s) in enumerate(mine, 1):
        print(f"[{args.worker}:{i}/{len(mine)}] {run_id(c, s)}", flush=True)
        train(ctx, c, s)
    print(f"WORKER {args.worker} DONE", flush=True)


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("check")
    c.add_argument("--threads", type=int, default=1)
    r = sub.add_parser("run")
    r.add_argument("--threads", type=int, default=1)
    r.add_argument("--worker", type=int, default=0)
    r.add_argument("--workers", type=int, default=1)
    r.add_argument("--seeds", type=int, choices=[10, 20], default=20)
    args = parser.parse_args()
    {"check": cmd_check, "run": cmd_run}[args.cmd](args)


if __name__ == "__main__":
    main()
