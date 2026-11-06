"""Phase 6 self-tests of the confirmatory code path, run before the pre-registration is frozen (non-confirmatory).

  python experiments/phase6/selftest.py pipeline               # stages 1-3 of run.py end to end, TEST replaced by a stub
  python experiments/phase6/selftest.py analysis [--keep DIR]  # aggregate.py + make_figures.py on synthetic records

pipeline: seed 42 and the uniform policy [2,2,2,2] (never a Phase-5 run), 1 epoch, temporary folders. The official test
          set is made unreadable for the whole process; VAL-RL images stand in for it. Checks that a fine-tuned model
          saved by stage 1, frozen by stage 2 and reloaded by stage 3 is the same model (identical accuracy), that a
          tampered checkpoint is refused, and that the frozen list cannot be overwritten.
analysis: 480 synthetic run records with random accuracies and one planted effect. Checks that every integrity
          check, family, criterion, table and figure is produced. No real result is read or written.
Both write only to results/phase6/_pilot_parts/selftest_*.json; the temporary folders are deleted.
"""
import argparse
import datetime
import hashlib
import importlib.util
import json
import os
import shutil
import tempfile

import numpy as np
import pandas as pd
import torch

import phase6_lib as L
from src import data as D
from src.evaluation import accuracy_from, predictions
from src.utils import sha256_file

HERE = os.path.dirname(os.path.abspath(__file__))
PARTS = os.path.join(L.RESULTS, "_pilot_parts")
PILOT_POLICY, PILOT_SEED = [2, 2, 2, 2], 42


def phase6_module(name):
    """Load a Phase-6 script by its file path, never by name: earlier phases have scripts with the same names."""
    path = os.path.join(HERE, f"{name}.py")
    spec = importlib.util.spec_from_file_location(f"phase6_{name}", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    assert os.path.samefile(mod.__file__, path)
    return mod


def redirect(tmp):
    for name in ("RUNS", "REPRO", "PRED", "TESTS", "WORK", "CKPT"):
        setattr(L, name, os.path.join(tmp, name.lower()))
    L.FINAL, L.PREREG = os.path.join(tmp, "final_models.json"), os.path.join(tmp, "preregistration.md")
    open(L.PREREG, "w", encoding="utf-8").write("self-test placeholder\n")


def report(name, res):
    for r in res:
        print(f"[{'PASS' if r['pass'] else 'FAIL'}] {r['check']}: {r['detail']}", flush=True)
    os.makedirs(PARTS, exist_ok=True)
    json.dump(res, open(os.path.join(PARTS, f"selftest_{name}.json"), "w", encoding="utf-8"), indent=1)
    print(f"{sum(r['pass'] for r in res)}/{len(res)} passed")


def cmd_pipeline(_a):
    R = phase6_module("run")
    jobs = [("cifar10_lenet5", PILOT_SEED, m) for m in L.ALL_METHODS] + [("cifar10_resnet8", PILOT_SEED, "lamp")]
    tmp = tempfile.mkdtemp(prefix="_selftest_", dir=L.RESULTS)
    res = []

    def chk(n, ok, d=""):
        res.append({"check": n, "pass": bool(ok), "detail": str(d)})
    try:
        redirect(tmp)
        L.CONFIG = os.path.join(tmp, "config.json")
        json.dump({"epochs": 1, "settings": {s: {"reference_sha256": L.P3.load_reference(s)[1]} for s in {j[0] for j in jobs}}},
                  open(L.CONFIG, "w", encoding="utf-8"))

        def no_test(self):
            raise RuntimeError("the official test set is unreadable in the self-test")
        D.DataBundle.test_tensors = no_test

        def stub(model, b, record):                           # VAL-RL stands in for TEST
            model.eval()
            b.freeze(record)
            p, y = predictions(model, b.rl_loader())
            return p.numpy().astype(np.int16), y.numpy().astype(np.int16), accuracy_from(p, y)
        L.test_predictions = stub
        R.jobs = lambda: jobs
        ctx = R.Ctx()
        ctx.policy = {(s, PILOT_SEED): PILOT_POLICY for s, _, _ in jobs}
        ctx.phase5_test = lambda s, policy, method: stub(L.build(ctx.setting(s)[1], ctx.setting(s)[2], method, policy), ctx.setting(s)[0], {})[2]
        for j in jobs:
            R.finetune_one(ctx, *j, L.RUNS)
        rec = {j: json.load(open(os.path.join(L.RUNS, f"{R.run_id(*j)}.json"), encoding="utf-8")) for j in jobs}
        chk("stage 1: every record complete, validation only, masks and zero counts intact",
            all(r["status"] == "complete" and "test" not in r["pre"] and "test" not in r["post"] and r["mask_hash_pre"] == r["mask_hash_post"]
                and r["mask_violations_post"] == 0 and r["zero_count_pre"] == r["zero_count_post"] and r["layer_zeros_pre"] == r["layer_zeros_post"]
                and not os.path.exists(os.path.join(L.WORK, f"{r['run_id']}.state.pt")) for r in rec.values()), f"{len(rec)} runs")
        chk("stage 1: the three pruned LeNet-5 runs share zero count, fine-tuning seed and data stream; the dense control shares seed and stream",
            len({rec[j]["zero_count_pre"] for j in jobs[:3]}) == 1 and len({rec[j]["ft_seed"] for j in jobs[:4]}) == 1
            and len({rec[j]["history"][-1]["data_stream_sha256"] for j in jobs[:4]}) == 1)
        R.finetune_one(ctx, *jobs[0], L.REPRO, False)
        rep = json.load(open(os.path.join(L.REPRO, f"{R.run_id(*jobs[0])}.json"), encoding="utf-8"))
        chk("reproduction run: bit-identical final weights and validation accuracy", rep["final_state_sha256"] == rec[jobs[0]]["final_state_sha256"]
            and rep["post"] == rec[jobs[0]]["post"])
        try:
            R.test_one(ctx, {}, "", *jobs[0])
            early = False
        except KeyError:
            early = True                                      # no frozen entry: stage 3 cannot evaluate a model that is not in the list
        R.freeze(ctx)
        try:
            R.freeze(ctx)
            twice = False
        except SystemExit:
            twice = True
        chk("stage 2: the list is written once and cannot be overwritten; a model outside the list cannot be evaluated", twice and early)
        final_sha = sha256_file(L.FINAL)
        final = {m["run_id"]: m for m in json.load(open(L.FINAL, encoding="utf-8"))["models"]}
        for j in jobs:
            R.test_one(ctx, final, final_sha, *j)
        tst = {j: json.load(open(os.path.join(L.TESTS, f"{R.run_id(*j)}.json"), encoding="utf-8")) for j in jobs}
        chk("stage 3: the checkpoint reloaded from disk is the fine-tuned model (stub accuracy == stage-1 VAL-RL accuracy, incl. ResNet-8 BatchNorm)",
            all(tst[j]["post_test"] == rec[j]["post"]["val_rl"] and tst[j]["pre_test"] == rec[j]["pre"]["val_rl"] for j in jobs),
            {R.run_id(*j): (tst[j]["pre_test"], tst[j]["post_test"]) for j in (jobs[0], jobs[-1])})
        chk("stage 3: prediction files written (one per fine-tuned model, one per distinct pre-fine-tuning model)",
            all(os.path.exists(os.path.join(L.ROOT, tst[j][k])) for j in jobs for k in ("pre_predictions_file", "post_predictions_file")))
        ck = os.path.join(L.ROOT, final[R.run_id(*jobs[1])]["checkpoint_file"])
        raw = bytearray(open(ck, "rb").read())
        raw[-1] ^= 1
        open(ck, "wb").write(bytes(raw))
        try:
            R.test_one(ctx, final, final_sha, *jobs[1])
            refused = False
        except AssertionError:
            refused = True
        chk("stage 3: a checkpoint file that differs from the frozen hash is refused", refused)
        chk("the official test set was never read", all(b._test is None for b in ctx.bundles.values()))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    report("pipeline", res)


def cmd_analysis(a):
    A, MF, R = phase6_module("aggregate"), phase6_module("make_figures"), phase6_module("run")
    cfg = json.load(open(L.CONFIG, encoding="utf-8"))
    masks = pd.read_csv(os.path.join(L.RESULTS, "phase6_matched_masks.csv"), float_precision="round_trip").set_index(["setting", "policy_seed"])
    tmp = tempfile.mkdtemp(prefix="_selftest_", dir=L.RESULTS)
    res = []

    def chk(n, ok, d=""):
        res.append({"check": n, "pass": bool(ok), "detail": str(d)})
    try:
        redirect(tmp)
        A.OUT, A.FIG, MF.FIG = tmp, os.path.join(tmp, "figures"), os.path.join(tmp, "figures")
        shutil.copy(os.path.join(L.RESULTS, "phase6_matched_masks.csv"), tmp)
        for d in (L.RUNS, L.REPRO, L.TESTS, L.CKPT):
            os.makedirs(d)
        rng = np.random.default_rng(1)
        hashes = {"preregistration_sha256": sha256_file(L.PREREG), "config_sha256": sha256_file(L.CONFIG)}
        t = datetime.datetime.now().astimezone()
        stamp = lambda k: (t + datetime.timedelta(seconds=k)).isoformat(timespec="seconds")  # noqa: E731
        # synthetic truth: PPO recovers 0.4 pp more than LAMP and global in the first three settings, nothing differs in the others
        gain = {"ppo": 0.8, "lamp": 0.4, "global": 0.4, "dense": 0.1}
        models, n = [], 0
        for si, s in enumerate(L.SETTINGS):
            dense = 60.0 + 5 * si
            pre_of = {}
            for seed in L.POLICY_SEEDS:
                row, common = masks.loc[(s, seed)], rng.normal(0, 0.2)
                for m in L.ALL_METHODS:
                    rid = R.run_id(s, seed, m)
                    pre = dense if m == "dense" else pre_of.setdefault((row.policy, m), dense - 1.0 - (0.3 if m == "ppo" else 0.0) + rng.normal(0, 0.1))
                    post = pre + (gain["lamp"] if (si >= 3 and m == "ppo") else gain[m]) + common + rng.normal(0, 0.1)
                    zc = 0 if m == "dense" else int(row.zero_count)
                    lz = {} if m == "dense" else json.loads(row[f"{m}_layer_zeros"])
                    mh = "dense" if m == "dense" else row[f"{m}_mask_sha256"]
                    raw = rid.encode()
                    path = os.path.join(L.CKPT, f"{rid}.pth")
                    open(path, "wb").write(raw)
                    hist = [{"epoch": e, "val_rl_accuracy": pre + (post - pre) * e / cfg["epochs"], "val_select_accuracy": pre + (post - pre) * e / cfg["epochs"] + 0.3,
                             "train_loss": 1.0 / e} for e in range(1, cfg["epochs"] + 1)]
                    hist[-1]["data_stream_sha256"] = hashlib.sha256(str(seed).encode()).hexdigest()
                    r = {"run_id": rid, "status": "complete", "stage": "finetune", "setting": s, "arch": s.split("_")[1], "method": m,
                         "policy_seed": seed, "ft_seed": L.ft_seed(seed), "policy": json.loads(row.policy), "epochs": cfg["epochs"],
                         "protocol": L.PROTOCOL, "threads": 1, "zero_count_pre": zc, "zero_count_post": zc, "sparsity_pre": 0.0 if m == "dense" else float(row.sparsity),
                         "sparsity_post": 0.0 if m == "dense" else float(row.sparsity), "layer_zeros_pre": lz, "layer_zeros_post": lz, "mask_hash_pre": mh,
                         "mask_hash_post": mh, "mask_violations_post": 0, "pre": {"val_rl": pre, "val_select": pre + 0.3},
                         "post": {"val_rl": hist[-1]["val_rl_accuracy"], "val_select": hist[-1]["val_select_accuracy"]}, "history": hist,
                         "final_state_sha256": hashlib.sha256(raw + b"state").hexdigest(), "checkpoint_sha256": hashlib.sha256(raw).hexdigest(),
                         "checkpoint_file": os.path.relpath(path, L.ROOT).replace("\\", "/"), "checkpoint_bytes_raw": 1000, "checkpoint_bytes_gzip": 800,
                         "train_seconds": 100.0, "finetune_seconds": 110.0 + si, "runtime_seconds": 120.0, "peak_memory_mb_process": 700.0,
                         "started_at": stamp(n), "finished_at": stamp(n + 1), "test_use": "none: TEST is not read in stage 1", **hashes}
                    json.dump(r, open(os.path.join(L.RUNS, f"{rid}.json"), "w", encoding="utf-8"))
                    models.append({"run_id": rid, "checkpoint_file": r["checkpoint_file"], "checkpoint_sha256": r["checkpoint_sha256"],
                                   "final_state_sha256": r["final_state_sha256"], "mask_hash": mh, "zero_count": zc})
                    if seed == L.POLICY_SEEDS[0] and m == "ppo":
                        json.dump(r, open(os.path.join(L.REPRO, f"{rid}.json"), "w", encoding="utf-8"))
                    n += 1
        json.dump({"created": stamp(n + 2), "n_models": len(models), "models": models}, open(L.FINAL, "w", encoding="utf-8"))
        fsha = sha256_file(L.FINAL)
        for k, mdl in enumerate(models):
            r = json.load(open(os.path.join(L.RUNS, f"{mdl['run_id']}.json"), encoding="utf-8"))
            json.dump({"run_id": mdl["run_id"], "status": "complete", "stage": "test", "pre_test": r["pre"]["val_rl"], "phase5_recorded_test": r["pre"]["val_rl"],
                       "post_test": r["post"]["val_rl"], "final_state_sha256": mdl["final_state_sha256"], "final_models_sha256": fsha,
                       "started_at": stamp(n + 3 + k), "finished_at": stamp(n + 4 + k)}, open(os.path.join(L.TESTS, f"{mdl['run_id']}.json"), "w", encoding="utf-8"))
        A.main()
        MF.main()
        st = pd.read_csv(os.path.join(tmp, "phase6_statistics.csv"))
        sizes = st.groupby("family", sort=False).size().to_dict()
        chk("integrity file written and every check passed on the synthetic records",
            "FAIL" not in open(os.path.join(tmp, "phase6_integrity_checks.md"), encoding="utf-8").read())
        chk("families and sizes as pre-registered",
            sizes == {A.H1: 18, A.DENSE: 6, A.SPECIFIC: 18, A.fam_of("lamp"): 12, A.ref_of("lamp"): 6, A.fam_of("global"): 12, A.ref_of("global"): 6,
                      A.H4: 6, A.H4P: 6, A.H6: 6}, sizes)
        chk("Holm-adjusted p >= raw p in every family; outcome labels valid", bool((st.holm_p >= st.sign_flip_p - 1e-15).all())
            and set(st.outcome) <= {"supported", "opposite", "inconclusive"})
        dg = st[(st.family == A.fam_of("lamp")) & (st.comparison == "dG ppo - lamp")].set_index("setting").outcome
        chk("the planted effect is found where it was planted and nowhere else (dG PPO - LAMP: 3 supported, 3 inconclusive)",
            list(dg[L.SETTINGS[:3]]) == ["supported"] * 3 and list(dg[L.SETTINGS[3:]]) == ["inconclusive"] * 3, dg.to_dict())
        cl = json.load(open(os.path.join(tmp, "phase6_classification.json"), encoding="utf-8"))["criteria"]
        chk("criteria A-F each MET / PARTIALLY MET / NOT MET", sorted(cl) == list("ABCDEF")
            and all(v["status"] in ("MET", "PARTIALLY MET", "NOT MET") for v in cl.values()), {k: v["status"] for k, v in cl.items()})
        need = ["phase6_all_runs.csv", "phase6_training_curves.csv", "phase6_pre_post_accuracy.csv", "phase6_accuracy_recovery.csv",
                "phase6_ppo_vs_lamp.csv", "phase6_ppo_vs_global.csv", "phase6_statistics.csv", "phase6_cross_setting_summary.csv",
                "phase6_cluster_sensitivity.csv", "phase6_classification.json", "phase6_integrity_checks.md"]
        figs = [f"phase6_{f}.png" for f in ("fig1_recovery", "fig2_ppo_vs_lamp", "fig3_recovery_curves", "fig4_accuracy_sparsity", "fig5_cost")]
        missing = [f for f in need if not os.path.exists(os.path.join(tmp, f))] + [f for f in figs if not os.path.exists(os.path.join(tmp, "figures", f))]
        chk("every required table, figure and figure source file is produced", not missing
            and len([f for f in os.listdir(os.path.join(tmp, "figures")) if f.endswith(".csv")]) == 5, missing)
        if a.keep:
            shutil.copytree(tmp, a.keep, dirs_exist_ok=True, ignore=shutil.ignore_patterns("ckpt", "runs", "tests", "repro"))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    report("analysis", res)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("pipeline")
    sub.add_parser("analysis").add_argument("--keep")
    a = ap.parse_args()
    torch.set_num_threads(L.THREADS)
    {"pipeline": cmd_pipeline, "analysis": cmd_analysis}[a.cmd](a)
