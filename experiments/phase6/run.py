"""Phase 6 confirmatory fine-tuning runs. Resumable per epoch; 1 thread per worker.

  python experiments/phase6/run.py --worker K --workers 4             # 480 runs: 6 settings x 20 policy seeds x {ppo, lamp, global, dense}
  python experiments/phase6/run.py --reproduce                         # the 6 pre-registered reproduction runs
Job order (setting, policy seed, method): job i goes to worker i mod 4, i.e. worker K fine-tunes method K of every pair.
Per run: pre-fine-tuning accuracy on VAL-RL / VAL-SELECT / TEST (TEST must equal the Phase-5 record), E epochs of
fixed-mask fine-tuning, final-epoch accuracy on VAL-RL / VAL-SELECT / TEST. TEST is read only for these two fixed
models. Progress lines show run ids and runtimes only.
"""
import argparse
import datetime
import gzip
import io
import json
import os
import time

import numpy as np
import pandas as pd
import torch

import phase6_lib as L
from src import pruning as P
from src.utils import environment, git_state, sha256_file, sha256_state_dict


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
        self.bundles, self.refs = {}, {}
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


def verified(ctx, path):
    if not os.path.exists(path):
        return False
    try:
        r = json.load(open(path, encoding="utf-8"))
        return r["status"] == "complete" and r["config_sha256"] == ctx.hashes["config_sha256"]
    except Exception as e:                                   # noqa: BLE001
        with open(os.path.join(os.path.dirname(path), "corruption_log.txt"), "a", encoding="utf-8") as f:
            f.write(f"{datetime.datetime.now().isoformat(timespec='seconds')} {os.path.basename(path)}: invalid record, repeated ({e!r})\n")
        return False


def run_one(ctx, s, seed, method, out_dir, save_artifacts=True):
    rid = run_id(s, seed, method)
    t0 = time.time()
    started = datetime.datetime.now().astimezone().isoformat(timespec="seconds")
    b, ref, arch = ctx.setting(s)
    policy = ctx.policy[(s, seed)]
    epochs = ctx.cfg["epochs"]
    fts = L.ft_seed(seed)
    model = L.build(ref, arch, method, policy)
    mask0, zeros0, layers0 = L.mask_hash(model), P.zero_count(model), L.layer_zeros(model)
    sparsity0 = P.total_sparsity(model)
    model.eval()
    pre_rl, pre_sel = L.evaluate_val(model, b)
    pre_pred, labels, pre_test = L.test_predictions(model, b, {"run": rid, "stage": "pre"})
    want = ctx.phase5_test(s, policy, method)
    assert pre_test == want, f"{rid}: pre-fine-tuning test accuracy {pre_test} != Phase-5 record {want}"
    os.makedirs(L.WORK, exist_ok=True)
    work = os.path.join(L.WORK, f"{rid}{'' if save_artifacts else '_repro'}.state.pt")
    history, resumed = L.finetune(model, b, fts, epochs, work, log=lambda m: None)
    model.eval()
    post_rl, post_sel = L.evaluate_val(model, b)
    post_pred, _, post_test = L.test_predictions(model, b, {"run": rid, "stage": "post"})
    folded = P.make_permanent(__import__("copy").deepcopy(model))
    sd = folded.state_dict()
    buf = io.BytesIO()
    torch.save(sd, buf)
    raw = buf.getvalue()
    rec = {"run_id": rid, "status": "complete", "setting": s, "arch": arch, "method": method, "policy_seed": seed, "ft_seed": fts,
           "policy": policy, "epochs": epochs, "protocol": L.PROTOCOL, "threads": torch.get_num_threads(),
           "total_weights": P.weight_total(model), "zero_count_pre": zeros0, "zero_count_post": P.zero_count(folded),
           "sparsity_pre": sparsity0, "sparsity_post": P.total_sparsity(folded), "layer_zeros_pre": layers0,
           "layer_zeros_post": L.layer_zeros(folded), "mask_hash_pre": mask0, "mask_hash_post": L.mask_hash(model),
           "mask_violations_post": L.mask_violations(model),
           "pre": {"val_rl": pre_rl, "val_select": pre_sel, "test": pre_test}, "phase5_recorded_test": want,
           "post": {"val_rl": post_rl, "val_select": post_sel, "test": post_test}, "history": history, "resumed_from_epoch": resumed,
           "final_state_sha256": sha256_state_dict(sd), "checkpoint_bytes_raw": len(raw),
           "checkpoint_bytes_gzip": len(gzip.compress(raw, 9, mtime=0)),
           "train_seconds": round(sum(h["train_seconds"] for h in history), 1),
           "finetune_seconds": round(sum(h["epoch_seconds"] for h in history), 1), "runtime_seconds": round(time.time() - t0, 1),
           "peak_memory_mb_process": L.peak_memory_mb(), "started_at": started,
           "finished_at": datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
           "test_use": "pre-fine-tuning model and final-epoch model only", **ctx.hashes, "environment": ctx.env, "git": ctx.git}
    os.makedirs(out_dir, exist_ok=True)
    if save_artifacts:
        for name, arr in (("pre", pre_pred), ("post", post_pred)):
            folder = os.path.join(L.PRED, s)
            os.makedirs(folder, exist_ok=True)
            bb = io.BytesIO()
            np.save(bb, arr)
            durable(bb.getvalue(), os.path.join(folder, f"{rid}_{name}.npy"))
        ck = os.path.join(L.CKPT, s)
        os.makedirs(ck, exist_ok=True)
        durable(raw, os.path.join(ck, f"{rid}.pth"))
        rec["checkpoint_file"] = os.path.relpath(os.path.join(ck, f"{rid}.pth"), L.ROOT).replace("\\", "/")
    durable(json.dumps(rec, indent=1).encode("utf-8"), os.path.join(out_dir, f"{rid}.json"))
    os.remove(work)
    return rec["runtime_seconds"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--worker", type=int)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--reproduce", action="store_true")
    a = ap.parse_args()
    ctx = Ctx()
    if a.reproduce:
        todo, out, save = reproduction_jobs(), L.REPRO, False
    else:
        todo, out, save = [j for i, j in enumerate(jobs()) if i % a.workers == a.worker], L.RUNS, True
    print(f"{'reproduction' if a.reproduce else f'worker {a.worker}'}: {len(todo)} jobs", flush=True)
    for s, seed, method in todo:
        path = os.path.join(out, f"{run_id(s, seed, method)}.json")
        if verified(ctx, path):
            continue
        print(f"done {run_id(s, seed, method)} {run_one(ctx, s, seed, method, out, save):.0f}s", flush=True)
    print("finished", flush=True)


if __name__ == "__main__":
    main()
