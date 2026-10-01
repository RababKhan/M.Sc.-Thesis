"""Phase 6 implementation and timing pilot (non-confirmatory: seed 42, uniform policy [2,2,2,2]; TEST never read).

  python experiments/phase6/pilot.py timing --worker K        # K = 0..3 run concurrently: 1 epoch per setting
  python experiments/phase6/pilot.py checks                    # mask, determinism, resume, serialization, Phase-5 match
  python experiments/phase6/pilot.py summarize                 # -> phase6_timing_pilot.csv + projection
Worker K fine-tunes method ALL_METHODS[K] (ppo-style mask, LAMP, global, dense), as in the confirmatory layout.
"""
import argparse
import copy
import glob
import gzip
import io
import json
import os
import shutil
import tempfile
import time

import pandas as pd
import torch

import phase6_lib as L
from src import pruning as P
from src.evaluation import evaluate_accuracy
from src.utils import sha256_state_dict

PILOT_POLICY = [2, 2, 2, 2]
PILOT_SEED = 42
PARTS = os.path.join(L.RESULTS, "_pilot_parts")


def cmd_timing(a):
    torch.set_num_threads(L.THREADS)
    os.makedirs(PARTS, exist_ok=True)
    method = L.ALL_METHODS[a.worker]
    rows, bundles = [], {}
    tmp = tempfile.mkdtemp()
    for s in L.SETTINGS:
        d, arch = L.P3.split(s)
        b = bundles.setdefault(d, L.bundle(d))
        ref, _ = L.P3.load_reference(s)
        model = L.build(ref, arch, method, PILOT_POLICY)
        z0 = P.zero_count(model)
        t0 = time.time()
        vr0, vs0 = L.evaluate_val(model.eval(), b)
        t_val = time.time() - t0
        hist, _ = L.finetune(model, b, PILOT_SEED, 1, os.path.join(tmp, f"{s}.state"), log=lambda m: None)
        rows.append({"setting": s, "method": method, "worker": a.worker, "threads": L.THREADS, "concurrent_workers": 4,
                     "train_seconds_per_epoch": hist[0]["train_seconds"], "validation_seconds_5000": round(t_val, 2),
                     "epoch_seconds_total": hist[0]["epoch_seconds"], "zero_count_before": z0, "zero_count_after": hist[0]["zero_count"],
                     "mask_violations": L.mask_violations(model), "peak_memory_mb": L.peak_memory_mb(),
                     "train_loss_epoch1": hist[0]["train_loss"]})
        print(rows[-1], flush=True)
    shutil.rmtree(tmp, ignore_errors=True)
    pd.DataFrame(rows).to_csv(os.path.join(PARTS, f"timing_{a.worker}.csv"), index=False)


def folded_state(model):
    return P.make_permanent(copy.deepcopy(model)).state_dict()


def cmd_checks(_a):
    torch.set_num_threads(L.THREADS)
    res = []

    def chk(n, ok, d=""):
        res.append({"check": n, "pass": bool(ok), "detail": str(d)})
        print(f"[{'PASS' if ok else 'FAIL'}] {n}: {d}", flush=True)
    s = "cifar10_lenet5"
    d, arch = L.P3.split(s)
    b = L.bundle(d)
    ref, _ = L.P3.load_reference(s)
    tmp = tempfile.mkdtemp()

    def run(method, epochs, name, stop_after=None):
        m = L.build(ref, arch, method, PILOT_POLICY)
        if stop_after:
            class Stop(Exception):
                pass

            def log(msg):
                if f"epoch {stop_after}/" in msg:
                    raise Stop()
            try:
                L.finetune(m, b, PILOT_SEED, epochs, os.path.join(tmp, name), log=log, validate=False)
            except Stop:
                pass
            m = L.build(ref, arch, method, PILOT_POLICY)
        h, resumed = L.finetune(m, b, PILOT_SEED, epochs, os.path.join(tmp, name), log=lambda x: None, validate=False)
        return m, h, resumed

    for method in L.ALL_METHODS:
        m0 = L.build(ref, arch, method, PILOT_POLICY)
        z0, h0 = P.zero_count(m0), L.mask_hash(m0)
        m1, hist, _ = run(method, 2, f"a_{method}")
        changed = sha256_state_dict(folded_state(m1)) != sha256_state_dict(folded_state(m0))
        chk(f"{method}: fixed mask after 2 epochs (mask hash equal, 0 violations, zero count {z0:,} unchanged); weights trained",
            L.mask_hash(m1) == h0 and L.mask_violations(m1) == 0 and P.zero_count(m1) == z0 and changed,
            f"train loss {hist[0]['train_loss']:.4f} -> {hist[1]['train_loss']:.4f}")
    k = {m: P.zero_count(L.build(ref, arch, m, PILOT_POLICY)) for m in L.METHODS}
    chk("PPO-style, LAMP and global masks have the identical zero count; the classifier is dense in all three",
        len(set(k.values())) == 1 and all(L.layer_zeros(L.build(ref, arch, m, PILOT_POLICY))[P.EXCLUDED[arch][0]] == 0 for m in L.METHODS), k)
    a1, _, _ = run("lamp", 2, "det1")
    a2, _, _ = run("lamp", 2, "det2")
    chk("deterministic: two 2-epoch runs with the same seed give bit-identical weights",
        sha256_state_dict(a1.state_dict()) == sha256_state_dict(a2.state_dict()))
    a3, h3, resumed = run("lamp", 2, "res", stop_after=1)
    chk("resume: interrupted after epoch 1 and resumed == uninterrupted run (bit-identical weights)",
        resumed == 1 and sha256_state_dict(a3.state_dict()) == sha256_state_dict(a1.state_dict()))
    g1, _, _ = run("global", 2, "pair_g")
    chk("pairing: different methods with the same seed see the same data stream (identical generator state after 2 epochs)",
        torch.equal(torch.load(os.path.join(tmp, "pair_g"), weights_only=False)["generator"],
                    torch.load(os.path.join(tmp, "det1"), weights_only=False)["generator"]))
    sd = folded_state(a1)
    buf = io.BytesIO()
    torch.save(sd, buf)
    fresh = L.P3.M.build_model(arch, 10)
    fresh.load_state_dict(torch.load(io.BytesIO(buf.getvalue()), weights_only=True))
    x = b.val_x[:256]
    with torch.no_grad():
        same = torch.equal(fresh.eval()(x), a1.eval()(x))
    chk("serialization: folded checkpoint reloads into a plain model with identical logits and zero count",
        same and P.zero_count(fresh) == P.zero_count(a1),
        f"raw {len(buf.getvalue()):,} bytes, gzip {len(gzip.compress(buf.getvalue(), 9, mtime=0)):,} bytes")
    # the Phase-5 pruned models are reproduced (validation only; TEST is not read in the pilot)
    ev = pd.read_csv(os.path.join(L.P5, "phase5_test_evaluations.csv"), float_precision="round_trip")
    fr = json.load(open(os.path.join(L.P5, "phase5_final_policies.json"), encoding="utf-8"))
    bad, n = [], 0
    for st in ("cifar10_lenet5", "cifar100_lenet5"):
        dd, ar = L.P3.split(st)
        bb = b if dd == d else L.bundle(dd)
        rf, _ = L.P3.load_reference(st)
        pols = sorted({tuple(r["selected_policy"]) for r in fr["runs"] if r["setting"] == st and r["condition"] == "C1"})[:3]
        for pol in pols:
            for method in L.METHODS:
                kind, meth = ("policy", None) if method == "ppo" else ("baseline", method)
                q = ev[(ev.setting == st) & (ev.policy == str(list(pol))) & (ev.kind == kind)]
                q = q[q.method.isna()] if meth is None else q[q.method == meth]
                mdl = L.build(rf, ar, method, list(pol))
                n += 1
                if evaluate_accuracy(mdl, bb.rl_loader()) != q.val_rl_accuracy.iloc[0] or P.zero_count(mdl) != q.zero_count.iloc[0]:
                    bad.append((st, pol, method))
    chk(f"Phase-5 pruned models reproduced: VAL-RL accuracy and zero count equal the Phase-5 records ({n} models)", not bad, bad[:3])
    shutil.rmtree(tmp, ignore_errors=True)
    os.makedirs(PARTS, exist_ok=True)
    json.dump(res, open(os.path.join(PARTS, "checks.json"), "w", encoding="utf-8"), indent=1)
    print(f"{sum(r['pass'] for r in res)}/{len(res)} passed")


def cmd_summarize(_a):
    t = pd.concat([pd.read_csv(p) for p in sorted(glob.glob(os.path.join(PARTS, "timing_*.csv")))])
    t.to_csv(os.path.join(L.RESULTS, "phase6_timing_pilot.csv"), index=False)
    per = t.groupby("setting").agg(train_s=("train_seconds_per_epoch", "mean"), val_s=("validation_seconds_5000", "mean"),
                                   peak_mb=("peak_memory_mb", "max")).reindex(L.SETTINGS)
    print(per.round(1).to_string())
    for epochs in (3, 5, 10):
        per_run = epochs * (per.train_s + per.val_s) + 3 * per.val_s        # + pre/post validation and two test passes
        total_process_h = float((per_run * 20 * 4).sum() + (per_run * 1).sum()) / 3600.0
        print(f"E = {epochs}: 480 runs + 6 reproduction runs = {total_process_h:.1f} process-hours = {total_process_h / 4:.1f} wall-hours on 4 workers")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("timing")
    t.add_argument("--worker", type=int, required=True)
    sub.add_parser("checks")
    sub.add_parser("summarize")
    a = ap.parse_args()
    {"timing": cmd_timing, "checks": cmd_checks, "summarize": cmd_summarize}[a.cmd](a)
