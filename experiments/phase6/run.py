"""Phase 6 confirmatory runs in three stages (results/phase6/phase6_preregistration.md). 1 thread per worker.

  python experiments/phase6/run.py finetune --worker K --workers 4   # stage 1: 480 fixed-mask fine-tuning runs (TEST never read)
  python experiments/phase6/run.py reproduce --worker K --workers 4  # stage 1: the 6 pre-registered repeat runs (TEST never read)
  python experiments/phase6/run.py freeze                             # stage 2: phase6_final_models.json (refuses to overwrite)
  python experiments/phase6/run.py test --worker K --workers 4        # stage 3: TEST for the frozen pre- and post-fine-tuning models

Job order (setting, policy seed, method): job i goes to worker i mod 4, i.e. worker K handles method K of every pair.
Stage 1 is resumable per epoch and records validation accuracy only. Stage 3 needs the frozen list of stage 2 and
evaluates exactly the checkpoint files named there. Progress lines show run ids and runtimes only.
"""
import argparse
import copy
import datetime
import gzip
import hashlib
import io
import json
import os
import time

import numpy as np
import pandas as pd
import torch

import phase6_lib as L
from src import pruning as P
from src.utils import environment, git_state, sha256_file, sha256_json, sha256_state_dict


def now():
    return datetime.datetime.now().astimezone().isoformat(timespec="seconds")


def durable(data, path):
    tmp = f"{path}.{os.getpid()}.tmp"
    with open(tmp, "wb") as f:
        f.write(data)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def run_id(s, seed, method):
    return f"{s}_seed{seed}_{method}"


def jobs():
    return [(s, x, m) for s in L.SETTINGS for x in L.POLICY_SEEDS for m in L.ALL_METHODS]


def reproduction_jobs():
    return [(s, L.POLICY_SEEDS[0], "ppo") for s in L.SETTINGS]


class Ctx:
    def __init__(self):
        torch.set_num_threads(L.THREADS)
        assert torch.get_num_threads() == L.THREADS
        self.cfg = json.load(open(L.CONFIG, encoding="utf-8"))
        fr = json.load(open(os.path.join(L.P5, "phase5_final_policies.json"), encoding="utf-8"))
        self.policy = {(r["setting"], r["seed"]): r["selected_policy"] for r in fr["runs"] if r["condition"] == "C1"}
        self.p5 = pd.read_csv(os.path.join(L.P5, "phase5_test_evaluations.csv"), float_precision="round_trip")
        self.bundles, self.refs, self.pre_cache = {}, {}, {}
        here = os.path.dirname(os.path.abspath(__file__))
        self.hashes = {"preregistration_sha256": sha256_file(L.PREREG), "config_sha256": sha256_file(L.CONFIG),
                       "code_sha256": {f: sha256_file(os.path.join(here, f)) for f in ("run.py", "phase6_lib.py")}}
        self.env, self.git = environment(), git_state()

    def setting(self, s):
        d, arch = L.P3.split(s)
        if s not in self.refs:
            self.bundles.setdefault(d, L.bundle(d))
            ref, sha = L.P3.load_reference(s)
            assert sha == self.cfg["settings"][s]["reference_sha256"]
            self.refs[s] = ref
        return self.bundles[d], self.refs[s], arch

    def phase5_test(self, s, policy, method):
        q = self.p5[self.p5.setting == s]
        if method == "dense":
            return float(q[q.kind == "dense"].test_accuracy.iloc[0])
        q = q[q.policy == str(list(policy))]
        q = q[(q.kind == "policy")] if method == "ppo" else q[(q.kind == "baseline") & (q.method == method)]
        return float(q.test_accuracy.iloc[0])


def verified(ctx, path, **must):
    """A complete record of this configuration (an unreadable record is logged and repeated)."""
    if not os.path.exists(path):
        return False
    try:
        r = json.load(open(path, encoding="utf-8"))
        return r["status"] == "complete" and r["config_sha256"] == ctx.hashes["config_sha256"] and all(r[k] == v for k, v in must.items())
    except Exception as e:                                   # noqa: BLE001
        with open(os.path.join(os.path.dirname(path), "corruption_log.txt"), "a", encoding="utf-8") as f:
            f.write(f"{now()} {os.path.basename(path)}: invalid record, repeated ({e!r})\n")
        return False


# ------------------------------------------------------------------ stage 1: fine-tuning (validation only)
def finetune_one(ctx, s, seed, method, out_dir, save_artifacts=True):
    rid = run_id(s, seed, method)
    t0, started = time.time(), now()
    b, ref, arch = ctx.setting(s)
    policy = ctx.policy[(s, seed)]
    epochs = ctx.cfg["epochs"]
    fts = L.ft_seed(seed)
    model = L.build(ref, arch, method, policy)
    mask0, zeros0, layers0, sparsity0 = L.mask_hash(model), P.zero_count(model), L.layer_zeros(model), P.total_sparsity(model)
    pre_rl, pre_sel = L.evaluate_val(model.eval(), b)
    os.makedirs(L.WORK, exist_ok=True)
    work = os.path.join(L.WORK, f"{rid}{'' if save_artifacts else '_repro'}.state.pt")
    history, resumed = L.finetune(model, b, fts, epochs, work, log=lambda m: None)
    plain = L.folded(model)                                  # the result: the model after the final epoch, masks folded in
    sd = plain.state_dict()
    buf = io.BytesIO()
    torch.save(sd, buf)
    raw = buf.getvalue()
    rec = {"run_id": rid, "status": "complete", "stage": "finetune", "setting": s, "arch": arch, "method": method, "policy_seed": seed,
           "ft_seed": fts, "policy": policy, "epochs": epochs, "protocol": L.PROTOCOL, "threads": torch.get_num_threads(),
           "total_weights": P.weight_total(plain), "zero_count_pre": zeros0, "zero_count_post": P.zero_count(plain),
           "sparsity_pre": sparsity0, "sparsity_post": P.total_sparsity(plain), "layer_zeros_pre": layers0,
           "layer_zeros_post": L.layer_zeros(plain), "mask_hash_pre": mask0, "mask_hash_post": L.mask_hash(model),
           "mask_violations_post": L.mask_violations(model),
           "pre": {"val_rl": pre_rl, "val_select": pre_sel},
           "post": {"val_rl": history[-1]["val_rl_accuracy"], "val_select": history[-1]["val_select_accuracy"]},
           "history": history, "resumed_from_epoch": resumed, "final_state_sha256": sha256_state_dict(sd),
           "checkpoint_sha256": hashlib.sha256(raw).hexdigest(), "checkpoint_bytes_raw": len(raw),
           "checkpoint_bytes_gzip": len(gzip.compress(raw, 9, mtime=0)),
           "train_seconds": round(sum(h["train_seconds"] for h in history), 1),
           "finetune_seconds": round(sum(h["epoch_seconds"] for h in history), 1), "runtime_seconds": round(time.time() - t0, 1),
           "peak_memory_mb_process": L.peak_memory_mb(), "started_at": started, "finished_at": now(),
           "test_use": "none: TEST is not read in stage 1", **ctx.hashes, "environment": ctx.env, "git": ctx.git}
    os.makedirs(out_dir, exist_ok=True)
    if save_artifacts:
        ck = os.path.join(L.CKPT, s)
        os.makedirs(ck, exist_ok=True)
        durable(raw, os.path.join(ck, f"{rid}.pth"))
        rec["checkpoint_file"] = os.path.relpath(os.path.join(ck, f"{rid}.pth"), L.ROOT).replace("\\", "/")
    durable(json.dumps(rec, indent=1).encode("utf-8"), os.path.join(out_dir, f"{rid}.json"))
    os.remove(work)
    return rec["runtime_seconds"]


# ------------------------------------------------------------------ stage 2: freeze the list of final models
def freeze(ctx):
    if os.path.exists(L.FINAL):
        raise SystemExit("phase6_final_models.json exists; the list of final models is frozen")
    models = []
    for s, seed, method in jobs():
        rid = run_id(s, seed, method)
        path = os.path.join(L.RUNS, f"{rid}.json")
        if not verified(ctx, path, stage="finetune"):
            raise SystemExit(f"{rid}: no complete stage-1 record; nothing frozen")
        r = json.load(open(path, encoding="utf-8"))
        if sha256_file(os.path.join(L.ROOT, r["checkpoint_file"])) != r["checkpoint_sha256"]:
            raise SystemExit(f"{rid}: checkpoint file does not match its record; nothing frozen")
        models.append({"run_id": rid, "checkpoint_file": r["checkpoint_file"], "checkpoint_sha256": r["checkpoint_sha256"],
                       "final_state_sha256": r["final_state_sha256"], "mask_hash": r["mask_hash_post"], "zero_count": r["zero_count_post"]})
    out = {"created": now(), "n_models": len(models), "rule": "the model after the final epoch of every run; no selection",
           "models": models, "content_sha256": sha256_json(models), **ctx.hashes, "git": ctx.git}
    durable(json.dumps(out, indent=1).encode("utf-8"), L.FINAL)
    print(f"frozen {len(models)} final models; sha256 {sha256_file(L.FINAL)}", flush=True)


# ------------------------------------------------------------------ stage 3: TEST for the frozen models
def test_one(ctx, final, final_sha, s, seed, method):
    rid = run_id(s, seed, method)
    t0, started = time.time(), now()
    b, ref, arch = ctx.setting(s)
    policy = ctx.policy[(s, seed)]
    r = json.load(open(os.path.join(L.RUNS, f"{rid}.json"), encoding="utf-8"))
    fz = final[rid]
    folder = os.path.join(L.PRED, s)
    os.makedirs(folder, exist_ok=True)

    def save(arr, name):
        bb = io.BytesIO()
        np.save(bb, arr)
        durable(bb.getvalue(), os.path.join(folder, name))
        return os.path.relpath(os.path.join(folder, name), L.ROOT).replace("\\", "/")
    # the pre-fine-tuning model is deterministic given (setting, method, policy): evaluated once per worker, never per seed
    key = (s, method, tuple(policy) if method != "dense" else ())
    if key not in ctx.pre_cache:
        pre = L.build(ref, arch, method, policy).eval()
        assert L.mask_hash(pre) == r["mask_hash_pre"], f"{rid}: pre-fine-tuning mask differs from the stage-1 record"
        pred, _, acc = L.test_predictions(pre, b, {"run": rid, "stage": "pre", "mask_hash": r["mask_hash_pre"]})
        name = "pre_dense.npy" if method == "dense" else f"pre_{method}_{''.join(map(str, policy))}.npy"
        ctx.pre_cache[key] = (acc, save(pred, name))
    pre_test, pre_file = ctx.pre_cache[key]
    want = ctx.phase5_test(s, policy, method)
    assert pre_test == want, f"{rid}: pre-fine-tuning test accuracy {pre_test} != Phase-5 record {want}"
    raw = open(os.path.join(L.ROOT, fz["checkpoint_file"]), "rb").read()
    assert hashlib.sha256(raw).hexdigest() == fz["checkpoint_sha256"], f"{rid}: checkpoint file is not the frozen one"
    plain = copy.deepcopy(ref)
    plain.load_state_dict(torch.load(io.BytesIO(raw), map_location="cpu", weights_only=True))
    assert sha256_state_dict(plain.state_dict()) == fz["final_state_sha256"] and P.zero_count(plain) == fz["zero_count"]
    post_pred, _, post_test = L.test_predictions(plain, b, {"run": rid, "stage": "post", "final_state_sha256": fz["final_state_sha256"]})
    rec = {"run_id": rid, "status": "complete", "stage": "test", "setting": s, "method": method, "policy_seed": seed, "policy": policy,
           "pre_test": pre_test, "phase5_recorded_test": want, "post_test": post_test, "pre_predictions_file": pre_file,
           "post_predictions_file": save(post_pred, f"{rid}_post.npy"), "final_state_sha256": fz["final_state_sha256"],
           "checkpoint_sha256": fz["checkpoint_sha256"], "final_models_sha256": final_sha, "threads": torch.get_num_threads(),
           "test_use": "the pre-fine-tuning model and the frozen final-epoch model only", "runtime_seconds": round(time.time() - t0, 1),
           "started_at": started, "finished_at": now(), **ctx.hashes, "git": ctx.git}
    os.makedirs(L.TESTS, exist_ok=True)
    durable(json.dumps(rec, indent=1).encode("utf-8"), os.path.join(L.TESTS, f"{rid}.json"))
    return rec["runtime_seconds"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["finetune", "reproduce", "freeze", "test"])
    ap.add_argument("--worker", type=int, default=0)
    ap.add_argument("--workers", type=int, default=1)
    a = ap.parse_args()
    ctx = Ctx()
    if a.stage == "freeze":
        return freeze(ctx)
    mine = [j for i, j in enumerate(jobs()) if i % a.workers == a.worker]
    if a.stage == "test":
        if not os.path.exists(L.FINAL):
            raise SystemExit("TEST is read only after phase6_final_models.json is frozen (run.py freeze)")
        final_sha = sha256_file(L.FINAL)
        final = {m["run_id"]: m for m in json.load(open(L.FINAL, encoding="utf-8"))["models"]}
        print(f"test worker {a.worker}: {len(mine)} jobs", flush=True)
        for s, seed, method in mine:
            if not verified(ctx, os.path.join(L.TESTS, f"{run_id(s, seed, method)}.json"), stage="test", final_models_sha256=final_sha):
                print(f"done {run_id(s, seed, method)} {test_one(ctx, final, final_sha, s, seed, method):.0f}s", flush=True)
    else:
        if os.path.exists(L.FINAL) and a.stage == "finetune":
            raise SystemExit("the list of final models is frozen; no fine-tuning run may be added or repeated")
        repeat = [j for i, j in enumerate(reproduction_jobs()) if i % a.workers == a.worker]
        todo, out, save = (repeat, L.REPRO, False) if a.stage == "reproduce" else (mine, L.RUNS, True)
        print(f"{a.stage} worker {a.worker}: {len(todo)} jobs", flush=True)
        for s, seed, method in todo:
            if not verified(ctx, os.path.join(out, f"{run_id(s, seed, method)}.json"), stage="finetune"):
                print(f"done {run_id(s, seed, method)} {finetune_one(ctx, s, seed, method, out, save):.0f}s", flush=True)
    print("finished", flush=True)


if __name__ == "__main__":
    main()
