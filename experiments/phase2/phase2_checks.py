"""Phase-2 implementation checks -> results/architecture_generalization/cpu_phase2_checks.md (+ .json).

No experiment is run and the test set is never read here (the guard checks only
confirm that it is locked). Checks that need CIFAR-100 or the trained dense
checkpoints report SKIPPED until those exist; the final Phase-2 run of this
script must have no SKIPPED or FAIL.

  python experiments/phase2/phase2_checks.py
"""
import copy
import datetime
import glob
import json
import os
import tempfile

import numpy as np
import torch

import phase2_common as C
from src import data as D, models as M, pruning as P, rl as RL, sensitivity as S
from src.evaluation import efficiency as E, evaluate_accuracy
from src.evaluation.training import run_dense_training
from src.pruning import structured as SP
from src.utils import sha256_file, sha256_json, sha256_state_dict

ARCHS = ["simplecnn", "lenet5", "resnet8", "smallvgg"]
RESULTS = []


def check(section, name, ok, detail=""):
    status = "SKIPPED" if ok is None else ("PASS" if ok else "FAIL")
    RESULTS.append({"section": section, "check": name, "status": status, "detail": str(detail)})
    print(f"[{status}] {section} | {name}: {detail}", flush=True)


def expected_params(arch, c):
    if arch == "simplecnn":
        return 896 + 18496 + 73856 + 524544 + 257 * c
    if arch == "lenet5":
        return 456 + 2416 + 48120 + 10164 + 85 * c
    if arch == "resnet8":
        return 77392 + 65 * c
    if arch == "smallvgg":
        conv = sum(ci * co * 9 + co + 2 * co for ci, co in [(3, 32), (32, 32), (32, 64), (64, 64), (64, 128), (128, 128)])
        return conv + 2048 * 256 + 256 + 257 * c


def have_c100():
    return os.path.exists(os.path.join(D.DATA_ROOT, "cifar-100-python")) and os.path.exists(D.SPLIT_FILES["cifar100"])


def reference(arch, dataset):
    """Trained reference if available (FIXED or the seed-0 Phase-2 checkpoint), else the seed-0 initial model."""
    c = D.DATASETS[dataset]["num_classes"]
    if arch == "simplecnn" and dataset == "cifar10":
        return M.load_checkpoint(C.FIXED, arch, c), "cnn_baseline_FIXED.pth"
    path = os.path.join(C.DENSE_CKPT, dataset, f"{arch}_seed0.pth")
    if os.path.exists(path):
        return M.load_checkpoint(path, arch, c), os.path.relpath(path, C.ROOT)
    return M.build_model(arch, c, seed=0).eval(), "untrained seed-0 initialisation"


def main():
    torch.set_num_threads(4)
    b10 = D.DataBundle("cifar10")
    bundles = {"cifar10": b10}
    if have_c100():
        bundles["cifar100"] = D.DataBundle("cifar100")

    # ---------------------------------------------------------------- 0 library verification
    lv = os.path.join(C.RESULTS, "reproducibility", "library_verification.json")
    if os.path.exists(lv):
        v = json.load(open(lv))
        n_s = sum(r["pass"] for r in v["static_checks"])
        n_p = sum(r["pass"] for r in v["ppo_runs"])
        check("0 reproduction", "src/ reproduces the historical SimpleCNN results", v["all_pass"],
              f"{n_s}/{len(v['static_checks'])} static checks, {n_p}/{len(v['ppo_runs'])} complete PPO runs "
              f"bit-identical (library_verification.json)")
    else:
        check("0 reproduction", "src/ reproduces the historical SimpleCNN results", None, "not run yet")

    # ---------------------------------------------------------------- 1 architectures
    for arch in ARCHS:
        for c in (10, 100):
            m1, m2, m3 = M.build_model(arch, c, seed=0), M.build_model(arch, c, seed=0), M.build_model(arch, c, seed=1)
            h1, h2, h3 = (sha256_state_dict(m.state_dict()) for m in (m1, m2, m3))
            x = torch.randn(4, 3, 32, 32, generator=torch.Generator().manual_seed(0))
            with torch.no_grad():
                y1, y2 = m1.eval()(x), m1.eval()(x)
            check("1 architectures", f"{arch} C={c}: instantiates, output (4,{c}), deterministic init and eval",
                  y1.shape == (4, c) and torch.equal(y1, y2) and h1 == h2 and h1 != h3,
                  f"init hash seed0 {h1[:12]} (repeat equal), seed1 differs")

    # ---------------------------------------------------------------- 2 forward passes on real data
    for ds, b in bundles.items():
        for arch in ARCHS:
            m = M.build_model(arch, b.num_classes, seed=0).eval()
            with torch.no_grad():
                out = m(b.val_x[:128])
            check("2 forward", f"{ds} {arch}: forward on 128 validation images",
                  out.shape == (128, b.num_classes) and bool(torch.isfinite(out).all()), tuple(out.shape))
    if "cifar100" not in bundles:
        check("2 forward", "cifar100 forward passes", None, "CIFAR-100 not prepared yet")

    # ---------------------------------------------------------------- 3 parameter counting
    for arch in ARCHS:
        for c in (10, 100):
            m = M.build_model(arch, c, seed=0)
            pc = M.count_parameters(m)
            ok = pc["total"] == expected_params(arch, c) == sum(p.numel() for p in m.parameters()) \
                and pc["weights"] == P.weight_total(m) and pc["trainable"] == pc["total"]
            check("3 parameters", f"{arch} C={c}: count = analytic formula", ok,
                  f"total {pc['total']:,}, weights {pc['weights']:,}, BN {pc['batchnorm']:,}")

    # ---------------------------------------------------------------- 4 units
    for arch in ARCHS:
        m = M.build_model(arch, 10, seed=0)
        units = P.get_units(m, arch)
        ok = len(units) == 4 and all(u.param_count == sum(mm.weight.numel() for _, mm in u.modules) for u in units)
        for i, u in enumerate(units):
            pm = copy.deepcopy(m)
            P.apply_unit_ratio(P.get_units(pm, arch)[i], 0.3)
            for (n, a), (_, bb) in zip(P.weight_layers(m), P.weight_layers(pm)):
                inside = n in [mn for mn, _ in u.modules]
                zeros = int((bb.weight == 0).sum()) - int((a.weight == 0).sum())
                ok &= (zeros == round(0.3 * a.weight.numel())) if inside else torch.equal(a.weight, bb.weight)
        check("4 units", f"{arch}: 4 units; pruning unit i changes exactly its tensors by round(r*n)", ok,
              ", ".join(f"{u.name}={u.param_count:,}" for u in units) + f"; excluded {P.EXCLUDED[arch]}")

    # ---------------------------------------------------------------- 5 zero-pruning policy
    for arch in ARCHS[:3]:
        m, src = reference(arch, "cifar10")
        z = P.prune_actions(m, arch, [0, 0, 0, 0])
        with torch.no_grad():
            same = torch.equal(m(b10.val_x[:512]), z(b10.val_x[:512]))
        acc_dense = evaluate_accuracy(m, b10.rl_loader())
        env = RL.PruningEnv(m, arch, acc_dense, b10.rl_loader())
        env.reset()
        for _ in range(4):
            _, _, done, _, info = env.step(0)
        check("5 zero policy", f"{arch} ({src}): [0,0,0,0] preserves logits and V_RL accuracy; sparsity unchanged",
              same and info["final_accuracy"] == acc_dense and info["final_sparsity"] == P.total_sparsity(m),
              f"V_RL acc {acc_dense:.2f} == episode acc {info['final_accuracy']:.2f}")

    # ---------------------------------------------------------------- 6 checkpoint reload
    with tempfile.TemporaryDirectory() as tmp:
        for arch in ARCHS:
            m = M.build_model(arch, 100, seed=3).eval()
            p = os.path.join(tmp, f"{arch}.pth")
            torch.save({"state_dict": m.state_dict(), "arch": arch}, p)
            r = M.load_checkpoint(p, arch, 100)
            with torch.no_grad():
                ok = torch.equal(m(b10.val_x[:64]), r(b10.val_x[:64]))
            check("6 checkpoints", f"{arch}: save -> load_checkpoint gives identical logits", ok and not r.training)
    check("6 checkpoints", "cnn_baseline_FIXED.pth unchanged and reloads", sha256_file(C.FIXED) == C.FIXED_SHA,
          C.FIXED_SHA[:16])
    recs = sorted(glob.glob(os.path.join(C.DENSE_RUNS, "*.json")))
    if recs:
        bad = []
        for rp in recs:
            r = json.load(open(rp))
            path = os.path.join(C.ROOT, r["checkpoint"])
            m = M.load_checkpoint(path, r["arch"], r["num_classes"])
            val = evaluate_accuracy(m, bundles[r["dataset"]].val_loader()) if r["dataset"] in bundles else None
            if sha256_file(path) != r["checkpoint_sha256"] or sha256_state_dict(m.state_dict()) != r["state_dict_sha256"] \
                    or val != r["val_accuracy"]:
                bad.append(r["run_id"])
        check("6 checkpoints", f"{len(recs)} Phase-2 dense checkpoints: hash matches record, reload reproduces "
                               "recorded validation accuracy", not bad, f"mismatches {bad}")
    else:
        check("6 checkpoints", "Phase-2 dense checkpoints reload", None, "no dense runs yet")

    # ---------------------------------------------------------------- 7 splits
    tr1, va1 = D.train_val_indices()
    tr2, va2 = D.train_val_indices()
    check("7 splits", "train/val permutation deterministic; CIFAR-10 val hash 9d3648af...",
          tr1 == tr2 and va1 == va2 and sha256_json(va1).startswith("9d3648af"), sha256_json(va1)[:16])
    npz = np.load(D.SPLIT_FILES["cifar10"])
    rl, sel, _, _ = D.stratified_split(b10.val_y.numpy())
    check("7 splits", "CIFAR-10 V_RL/V_SELECT regenerated identically from labels",
          np.array_equal(rl, npz["v_rl_positions"]) and np.array_equal(sel, npz["v_select_positions"]),
          sha256_file(D.SPLIT_FILES["cifar10"])[:16])
    if "cifar100" in bundles:
        b = bundles["cifar100"]
        z = np.load(D.SPLIT_FILES["cifar100"])
        rl, sel, _, n_c = D.stratified_split(b.val_y.numpy())
        man = json.load(open(os.path.join(C.RESULTS, "reproducibility", "cifar100_split_manifest.json")))
        ok = (np.array_equal(rl, z["v_rl_positions"]) and np.array_equal(sel, z["v_select_positions"])
              and z["val_indices"].tolist() == b.val_indices and sha256_file(D.SPLIT_FILES["cifar100"]) == man["file_sha256"]
              and len(rl) == 3000 and len(sel) == 2000 and not set(rl.tolist()) & set(sel.tolist()))
        check("7 splits", "CIFAR-100 split regenerated identically; file hash = manifest; 3,000/2,000 disjoint", ok,
              f"sha256 {man['file_sha256'][:16]}; V_RL per class {int(n_c.min())}-{int(n_c.max())}")
    else:
        check("7 splits", "CIFAR-100 split", None, "not prepared yet")

    # ---------------------------------------------------------------- 8 guards
    g = D.DataBundle("cifar10", train_images=False)
    msgs = []
    for fn in (g.test_loader,):
        try:
            fn()
        except RuntimeError as e:
            msgs.append(str(e))
    with g.training():
        for fn in (g.test_loader, g.select_loader, lambda: g.freeze("x")):
            try:
                fn()
            except RuntimeError as e:
                msgs.append(str(e))
        rl_ok = len(g.rl_loader().dataset) == 3000
    sel_ok = len(g.select_loader().dataset) == 2000
    try:                                         # after freeze the test tensors unlock (not read further here)
        g.freeze({"checkpoint": "guard-test"})
        unlocked = g.frozen and g.phase == "selection"
    except RuntimeError:
        unlocked = False
    with g.training():
        relock = not g.frozen
    check("8 guards", "test raises before freeze and inside training(); V_SELECT and freeze raise inside training(); "
                      "V_RL available in training; freeze unlocks; re-entering training re-locks",
          len(msgs) == 4 and rl_ok and sel_ok and unlocked and relock, " | ".join(msgs))

    # ---------------------------------------------------------------- 9 one-shot baselines
    for arch in ARCHS[:3]:
        m, src = reference(arch, "cifar10")
        units = P.get_units(m, arch)
        tensors = P.prunable_tensors(units)
        n = np.array([t.weight.numel() for _, t in tensors])
        for target in (30.0, 50.0, 58.1957):
            k = P.target_count(m, target)
            out = {name: fn(m, arch, target, 1000) if name == "random" else fn(m, arch, target)
                   for name, fn in P.BASELINES.items()}
            ok = all(P.zero_count(pm) == k for pm in out.values())
            ok &= all(all(int((mm.weight == 0).sum()) == 0 for nn_, mm in P.weight_layers(pm) if nn_ in P.EXCLUDED[arch])
                      for pm in out.values())
            share = n * k / n.sum()
            uz = np.array([int((t.weight == 0).sum()) for _, t in P.prunable_tensors(P.get_units(out["uniform"], arch))])
            ok &= bool(np.all(np.abs(uz - share) < 1.0))
            # global: every removed |w| <= every kept |w|
            gw = [(t.weight_orig.detach().abs().flatten(), t.weight_mask.flatten())
                  for _, t in P.prunable_tensors(P.get_units(out["global"], arch))]
            removed = torch.cat([w[mk == 0] for w, mk in gw])
            kept = torch.cat([w[mk == 1] for w, mk in gw])
            ok &= bool(removed.max() <= kept.min())
            rz = np.array([int((t.weight == 0).sum()) for _, t in P.prunable_tensors(P.get_units(out["random"], arch))])
            r_same = P.random_uniform(m, arch, target, 1000)
            r_diff = P.random_uniform(m, arch, target, 1001)
            ok &= bool(np.array_equal(rz, uz)) and sha256_state_dict(r_same.state_dict()) == \
                sha256_state_dict(out["random"].state_dict()) != sha256_state_dict(r_diff.state_dict())
            lz = [t for _, t in P.prunable_tensors(P.get_units(out["lamp"], arch))]
            ok &= all(int((t.weight_mask != 0).sum()) >= 1 for t in lz)
            ez = np.array([int((t.weight != 0).sum()) for _, t in P.prunable_tensors(P.get_units(out["erk"], arch))])
            d = P.erk_densities([tuple(t.weight.shape) for _, t in tensors], int(n.sum() - k))
            ok &= bool(np.all(d <= 1 + 1e-12)) and int(ez.sum()) == int(n.sum() - k) and bool(np.all(np.abs(ez - d * n) <= 1.0))
            check("9 baselines", f"{arch} @ {target}%: uniform/global/random/LAMP/ERK hit K={k:,} zeros exactly; "
                                 "classifier dense; per-method invariants", ok, src)
    # LAMP formula against brute force
    w = torch.tensor([[0.5, -0.1, 0.3], [0.0, -0.8, 0.2]])
    brute = torch.zeros(6, dtype=torch.float64)
    flat = w.flatten().double()
    for i in range(6):
        brute[i] = flat[i] ** 2 / sum(flat[j] ** 2 for j in range(6) if flat[j] ** 2 >= flat[i] ** 2)
    check("9 baselines", "LAMP score = w_u^2 / sum_{|w_v| >= |w_u|} w_v^2 (brute force, 6 weights)",
          torch.allclose(P.lamp_scores(w).flatten(), brute), P.lamp_scores(w).flatten().tolist())
    shapes = [(32, 3, 3, 3), (64, 32, 3, 3), (256, 2048)]
    d = P.erk_densities(shapes, 100000)
    raw = np.array([sum(s) / np.prod(s) for s in shapes])
    free = d < 1
    check("9 baselines", "ERK densities proportional to (sum of dims)/(prod of dims) below 1; total kept exact",
          np.allclose((d / raw)[free], (d / raw)[free][0]) and abs((d * [np.prod(s) for s in shapes]).sum() - 100000) < 1e-6,
          np.round(d, 4).tolist())

    # ---------------------------------------------------------------- 10 structured (prepared)
    for arch, ratios in (("simplecnn", (0.25, 0.5, 0.5, 0.5)), ("resnet8", (0.25, 0.5, 0.5))):
        m, src = reference(arch, "cifar10")
        x = b10.val_x[:256]
        same, _ = SP.structured(m, arch, [0.0] * len(ratios))
        slim, kept = SP.structured(m, arch, ratios)
        masked = SP.masked_equivalent(m, arch, kept)
        with torch.no_grad():
            a, bb, c0, c1 = m(x), same(x), slim(x), masked(x)
        agree = float((c0.argmax(1) == c1.argmax(1)).float().mean())
        dm, _ = E.conv_linear_macs(m)
        sm, _ = E.conv_linear_macs(slim)
        ok = torch.equal(a, bb) and float((c0 - c1).abs().max()) < 1e-4 and agree == 1.0 and sm < dm \
            and M.count_parameters(slim)["total"] < M.count_parameters(m)["total"]
        check("10 structured", f"{arch}: ratio 0 exact; slimmed == masked-equivalent (max |dlogit| "
                               f"{float((c0 - c1).abs().max()):.1e}); MACs {dm:,} -> {sm:,}", ok, src)

    # ---------------------------------------------------------------- 11 training determinism and resume
    with tempfile.TemporaryDirectory() as tmp:
        sub = D.DataBundle("cifar10")
        sub.train_x, sub.train_y = sub.train_x[:1280], sub.train_y[:1280]
        proto = dict(C.protocol(), epochs=2)
        ra = run_dense_training("lenet5", sub, 7, proto, os.path.join(tmp, "a.state"), os.path.join(tmp, "a.pth"), 4,
                                log=lambda s: None, evaluate_test=False)

        class Stop(Exception):
            pass

        def stop_after_first(s):
            if "epoch   1/" in s:
                raise Stop()
        try:
            run_dense_training("lenet5", sub, 7, proto, os.path.join(tmp, "b.state"), os.path.join(tmp, "b.pth"), 4,
                               log=stop_after_first, evaluate_test=False)
        except Stop:
            pass
        try:
            run_dense_training("lenet5", sub, 7, proto, os.path.join(tmp, "b.state"), os.path.join(tmp, "b.pth"), 2,
                               log=lambda s: None, evaluate_test=False)
            thread_guard = False
        except RuntimeError:
            thread_guard = True
        rb = run_dense_training("lenet5", sub, 7, proto, os.path.join(tmp, "b.state"), os.path.join(tmp, "b.pth"), 4,
                                log=lambda s: None, evaluate_test=False)
        hist_eq = [{k: v for k, v in h.items() if "seconds" not in k} for h in ra["history"]] == \
                  [{k: v for k, v in h.items() if "seconds" not in k} for h in rb["history"]]
        check("11 training", "2-epoch run == interrupted-after-epoch-1 + resumed run (weights, history); "
                             "resume with another thread count refused; test not read",
              ra["state_dict_sha256"] == rb["state_dict_sha256"] and hist_eq and thread_guard
              and ra["test_accuracy"] is None, ra["state_dict_sha256"][:16])
        rc = run_dense_training("lenet5", sub, 7, proto, os.path.join(tmp, "c.state"), os.path.join(tmp, "c.pth"), 4,
                                log=lambda s: None, evaluate_test=False)
        check("11 training", "same seed + threads -> bit-identical training", rc["state_dict_sha256"] == ra["state_dict_sha256"],
              "LeNet-5, 1,280 images, 2 epochs")

    # ---------------------------------------------------------------- 12 PPO environment on every architecture
    from stable_baselines3 import PPO
    for arch in ARCHS[:3]:
        m, src = reference(arch, "cifar10")
        loader = b10.rl_loader()
        acc = evaluate_accuracy(m, loader)
        seqs = [[1, 2, 3, 4], [5, 0, 5, 0], [1, 2, 3, 4]]

        def episodes(cache):
            torch.manual_seed(123)
            env = RL.PruningEnv(m, arch, acc, loader, cache=cache)
            out = []
            for s in seqs:
                obs, _ = env.reset()
                for a in s:
                    obs, r, done, _, info = env.step(a)
                out.append((r, info["final_accuracy"], info["final_sparsity"]))
            return out, torch.get_rng_state(), env
        plain, st_plain, env = episodes(None)
        cached, st_cached, _ = episodes({})
        u = P.get_units(m, arch)[2]
        env.reset()
        for a in (0, 0):
            obs, *_ = env.step(a)
        manual = torch.cat([mm.weight.detach().flatten() for _, mm in u.modules])
        ok = plain == cached and torch.equal(st_plain, st_cached) and obs.shape == (7,) \
            and abs(obs[3] - manual.abs().mean().item()) < 1e-7 and abs(obs[2] - u.param_count / env.max_param_count) < 1e-7
        with b10.training():
            agent = PPO(env=RL.PruningEnv(m, arch, acc, loader, cache={}), seed=300, **RL.PPO_KWARGS)
            agent.learn(total_timesteps=64)
        check("12 PPO env", f"{arch} ({src}): memoised == uncached (rewards and global RNG state); unit features over "
                            "concatenated weights; SB3 PPO trains one rollout", ok and agent.num_timesteps == 64,
              f"V_RL acc {acc:.2f}")
    env = RL.PruningEnv(*reference("resnet8", "cifar10")[:1], "resnet8", 50.0, b10.rl_loader())
    zero_prior = [[0.0] * 6 for _ in range(4)]
    a1 = PPO(env=env, seed=5, **RL.PPO_KWARGS)
    a2 = PPO(env=env, seed=5, **dict(RL.PPO_KWARGS, policy=RL.SoftPriorPolicy), policy_kwargs={"prior": zero_prior})
    obs = torch.tensor(np.stack([env.reset()[0]] * 3))
    obs[:, 0] = torch.tensor([0.0, 1 / 3, 1.0])
    with torch.no_grad():
        p1 = a1.policy.get_distribution(obs).distribution.probs
        p2 = a2.policy.get_distribution(obs).distribution.probs
    check("12 PPO env", "SoftPriorPolicy with a zero prior == MlpPolicy (4-unit ResNet-8 env)", torch.equal(p1, p2))

    # ---------------------------------------------------------------- 13 efficiency counters and sensitivity
    for arch in ARCHS:
        m = M.build_model(arch, 10, seed=0)
        macs, _ = E.conv_linear_macs(m)
        pt, _ = E.ptflops_macs(m)
        raw, gz = E.storage_bytes(m)
        check("13 efficiency", f"{arch}: FlopCounterMode = 2 x MACs; ptflops >= MACs; storage raw > gzip > 0",
              E.torch_flops(m) == 2 * macs and pt >= macs and raw > gz > 0, f"MACs {macs:,}, ptflops {pt:,}")
    fixed = M.load_checkpoint(C.FIXED, "simplecnn")
    pm = P.prune_actions(fixed, "simplecnn", [1, 1, 5, 5])
    check("13 efficiency", "nominal sparse MACs of the [1,1,5,5] SimpleCNN < dense MACs (theoretical only)",
          E.nominal_sparse_macs(pm) < E.conv_linear_macs(fixed)[0],
          f"{E.nominal_sparse_macs(pm):,.0f} vs {E.conv_linear_macs(fixed)[0]:,}")
    for arch in ARCHS[:3]:
        m, _ = reference(arch, "cifar10")
        x, y = b10.rl_tensors()
        s = S.loss_sensitivity(m, arch, x[:256], y[:256], ratios=(0.4,))
        a0, ad = S.accuracy_drop(m, arch, D.eval_loader(x[:256], y[:256]))
        check("13 sensitivity", f"{arch}: loss sensitivity finite and min-max normalised; accuracy drop runs",
              all(np.isfinite(s["raw"])) and min(s["normalized"]) == 0.0 and max(s["normalized"]) in (0.0, 1.0)
              and len(ad) == 4, [round(v, 4) for v in s["normalized"]])

    # ---------------------------------------------------------------- write
    n = {k: sum(r["status"] == k for r in RESULTS) for k in ("PASS", "FAIL", "SKIPPED")}
    stamp = datetime.datetime.now().astimezone().isoformat(timespec="seconds")
    with open(os.path.join(C.AG, "cpu_phase2_checks.json"), "w", encoding="utf-8") as f:
        json.dump({"generated": stamp, "counts": n, "checks": RESULTS, "code_sha256": C.code_hashes()}, f, indent=1)
    lines = ["# Phase-2 implementation checks", "",
             f"Generated {stamp} by `experiments/phase2/phase2_checks.py` (4 threads). "
             f"**{n['PASS']} passed, {n['FAIL']} failed, {n['SKIPPED']} skipped.** The test set is not read by "
             "these checks. Machine-readable copy: `cpu_phase2_checks.json`.", ""]
    section = None
    for r in RESULTS:
        if r["section"] != section:
            section = r["section"]
            lines += ["", f"## {section}", "", "| Status | Check | Detail |", "|---|---|---|"]
        lines.append(f"| {r['status']} | {r['check']} | {r['detail'].replace('|', '/')} |")
    lines += ["", "## Baseline formulas", "",
              "K = round(s · W / 100) zeros for target total sparsity s over all W Conv2d/Linear weights; all zeros "
              "come from the prunable tensors P (n_t weights, N = Σ n_t); the output classifier stays dense; "
              "integer per-tensor counts use largest-remainder rounding.", "",
              "- **Uniform:** each tensor loses ≈ n_t · K / N of its smallest-|w| weights.",
              "- **Global magnitude:** the K smallest |w| over the union of P.",
              "- **Random:** uniform per-tensor counts; the removed weights inside each tensor are drawn uniformly "
              "at random (torch.Generator(seed)); 10 seeds from 1000.",
              "- **LAMP** (Lee et al., 2021): with |w_(1)| ≤ … ≤ |w_(n)| inside a tensor, "
              "score(u) = w_(u)² / Σ_{v ≥ u} w_(v)²; remove the K smallest scores over P.",
              "- **ERK** (Evci et al., 2020): density d_t = min(1, ε · Σdims(t) / Πdims(t)), ε solved so that "
              "Σ d_t n_t = N − K (saturated tensors fixed dense, ε re-solved); smallest-|w| removal inside each tensor.",
              ""]
    with open(os.path.join(C.AG, "cpu_phase2_checks.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(n)


if __name__ == "__main__":
    main()
