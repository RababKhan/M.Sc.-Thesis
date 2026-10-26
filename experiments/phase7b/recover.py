"""Phase 7B: structured recovery jobs (proxy validation, proxy landscape, final models).

  python experiments/phase7b/recover.py worker --queue proxy_validation --cpu C
  python experiments/phase7b/recover.py worker --queue landscape --cpu C        # after the proxy is accepted
  python experiments/phase7b/recover.py status --queue Q
A job = (policy, E): frozen dense reference -> structured_units.slim -> phase6_lib.finetune (cosine over E epochs,
seed 600, 1 thread, validate=False) -> VAL-RL and VAL-SELECT accuracy. The test set is never read. Workers claim
jobs through atomic lock files; per-epoch work states make every job resumable. Final-model jobs (queue "final")
also save the recovered state dict.
"""
import argparse
import datetime
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import p7b_lib as L  # noqa: E402

RUNS = os.path.join(L.RESULTS, "_recovery_runs")
WORK = os.path.join(L.CKPT, "_work")


def now():
    return datetime.datetime.now().astimezone().isoformat(timespec="seconds")


def jobs(queue):
    if queue == "proxy_validation":
        pol = json.load(open(os.path.join(L.RESULTS, "stage_a_samples.json"), encoding="utf-8"))["proxy_validation"]["policies"]
        return [(tuple(p), e) for e in (5, 2, 3) for p in pol]
    if queue == "landscape":
        e = json.load(open(os.path.join(L.RESULTS, "accuracy_proxy_metrics.json"), encoding="utf-8"))["accepted_epochs"]
        assert e in (2, 3), "no accepted proxy: the landscape must not be computed"
        return [(p, e) for p in L.POLICIES]
    if queue == "final":
        return [(tuple(p), L.FINAL_EPOCHS) for p in json.load(open(os.path.join(L.RESULTS, "final_jobs.json"), encoding="utf-8"))]
    raise ValueError(queue)


def record_path(queue, policy, e):
    return os.path.join(RUNS, queue, f"{L.key(policy)}_E{e}.json")


def claim(queue, policy, e):
    d = os.path.join(RUNS, queue, "_locks")
    os.makedirs(d, exist_ok=True)
    try:
        fd = os.open(os.path.join(d, f"{L.key(policy)}_E{e}.lock"), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        os.write(fd, f"{os.getpid()} {now()}".encode())
        os.close(fd)
        return True
    except FileExistsError:
        return False


def recover(ctx, queue, policy, e):
    import psutil
    import torch
    from src import models as M
    from src.evaluation import efficiency as E
    from src.pruning import structured_units as SU
    from src.utils import sha256_state_dict
    t0, started = time.time(), now()
    b, ref, ref_sha = ctx.get(L.SETTING)
    m, _ = SU.slim(ref, L.ARCH, list(policy))
    shape0 = SU.shape_key(m)
    pre_rl, pre_sel = ctx.P6.evaluate_val(m.eval(), b)
    os.makedirs(WORK, exist_ok=True)
    work = os.path.join(WORK, f"{queue}_{L.key(policy)}_E{e}.state.pt")
    hist, resumed = ctx.P6.finetune(m, b, L.RECOVERY_SEED, e, work, log=lambda s: None, validate=False)
    assert SU.shape_key(m) == shape0, "topology changed during recovery"
    post_rl, post_sel = ctx.P6.evaluate_val(m.eval(), b)
    sd = m.state_dict()
    macs, params, widths = L.counts(policy)
    assert macs == E.conv_linear_macs(m)[0] and params == M.count_parameters(m)["total"]
    rec = {"queue": queue, "policy": list(policy), "epochs": e, "status": "complete", "widths": widths, "macs": macs,
           "params": params, "pre": {"val_rl": pre_rl, "val_select": pre_sel}, "post": {"val_rl": post_rl, "val_select": post_sel},
           "train_loss": [h["train_loss"] for h in hist], "resumed_from_epoch": resumed,
           "data_stream_sha256": hist[-1]["data_stream_sha256"], "final_state_sha256": sha256_state_dict(sd),
           "recovery": {"protocol": ctx.P6.PROTOCOL, "epochs": e, "seed": L.RECOVERY_SEED},
           "reference_sha256": ref_sha, "threads": torch.get_num_threads(), "cpu_affinity": psutil.Process().cpu_affinity(),
           "seconds": round(time.time() - t0, 1), "started_at": started, "finished_at": now(), "test_use": "none"}
    if queue == "final":
        os.makedirs(L.CKPT, exist_ok=True)
        ck = os.path.join(L.CKPT, f"{L.SETTING}_{L.key(policy)}_E{e}.pth")
        torch.save(sd, ck)
        rec.update({"checkpoint": os.path.relpath(ck, L.ROOT), "checkpoint_sha256": L.sha256_file(ck),
                    "checkpoint_bytes": os.path.getsize(ck)})
    path = record_path(queue, policy, e)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + f".{os.getpid()}.tmp"
    json.dump(rec, open(tmp, "w", encoding="utf-8"), indent=1)
    os.replace(tmp, path)
    if os.path.exists(work):
        os.remove(work)
    return rec


def cmd_worker(a):
    import psutil
    os.environ["OMP_NUM_THREADS"] = os.environ["MKL_NUM_THREADS"] = "1"
    psutil.Process().cpu_affinity([a.cpu])
    ctx = L.load_module("h2_recovery", L.PAPER3).Ctx()          # frozen Phase-6 protocol and data bundle (read only)
    for policy, e in jobs(a.queue):
        if os.path.exists(record_path(a.queue, policy, e)) or not claim(a.queue, policy, e):
            continue
        r = recover(ctx, a.queue, policy, e)
        print(f"{now()} {a.queue} {L.key(policy)} E{e} VAL-RL {r['post']['val_rl']:.2f} ({r['seconds']:.0f}s)", flush=True)


def cmd_status(a):
    js = jobs(a.queue)
    done = sum(os.path.exists(record_path(a.queue, p, e)) for p, e in js)
    print(f"{a.queue}: {done}/{len(js)} complete")


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    w = sub.add_parser("worker")
    w.add_argument("--queue", required=True)
    w.add_argument("--cpu", type=int, required=True)
    s = sub.add_parser("status")
    s.add_argument("--queue", required=True)
    a = p.parse_args()
    {"worker": cmd_worker, "status": cmd_status}[a.cmd](a)


if __name__ == "__main__":
    main()
