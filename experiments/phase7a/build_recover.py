"""Phase 7A, steps 4-7: build the physically compact models, check them, recover them (5 epochs), freeze them.

  python experiments/phase7a/build_recover.py check              # step 5: structural integrity (no training)
  python experiments/phase7a/build_recover.py worker --cpu C     # step 6: recovery worker pinned to logical CPU C
  python experiments/phase7a/build_recover.py freeze             # step 7: accuracy table + checkpoint hashes
Recovery = Paper-3 recover_one with per-unit ratios instead of action indices: frozen dense reference -> slim ->
phase6_lib.finetune (5 epochs, seed 600, 1 thread, validate=False) -> VAL-RL / VAL-SELECT. The test set is never read.
"""
import argparse
import datetime
import io
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import p7_lib as L  # noqa: E402

RUNS = os.path.join(L.RESULTS, "recovery_runs")
LOCKS = os.path.join(RUNS, "_locks")
WORK = os.path.join(L.CKPT, "_work")
JOBS = [(arch, m) for arch in L.SETTINGS for m in L.METHODS]


def now():
    return datetime.datetime.now().astimezone().isoformat(timespec="seconds")


def record_path(arch, method):
    return os.path.join(RUNS, f"{L.SETTINGS[arch]}_{method}.json")


# ------------------------------------------------------------------ step 5
def cmd_check(_a):
    import torch
    from src import models as M
    from src.evaluation import efficiency as E
    from src.pruning import structured_units as SU
    torch.set_num_threads(1)
    sel = L.policies()["settings"]
    x = torch.randn(16, 3, 32, 32, generator=torch.Generator().manual_seed(7))
    rows, ok = [], True
    for arch, setting in L.SETTINGS.items():
        ref, _ = L.p3().reference(setting)
        ref.eval()
        for method in L.METHODS:
            spec = sel[setting]["methods"][method]
            m, kept = L.slim(ref, arch, spec["structured_ratios"])
            state = m.state_dict()
            no_mask = (not any(k.endswith(("_orig", "_mask")) for k in state)
                       and not any(len(mod._forward_pre_hooks) for mod in m.modules()))
            params = M.count_parameters(m)["total"]
            macs = E.conv_linear_macs(m)[0]
            got_w = [len(kept[u]) for u in SU.UNITS[arch]]
            with torch.inference_mode():
                y = m(x)
                diff = float((y - SU.masked_equivalent(ref, arch, kept).eval()(x)).abs().max())
                buf = io.BytesIO()
                torch.save(state, buf)
                buf.seek(0)
                fresh, _ = L.slim(L.p3().reference(setting)[0], arch, spec["structured_ratios"])
                fresh.load_state_dict(torch.load(buf, weights_only=True))
                reload_identical = bool(torch.equal(fresh.eval()(x), y))
            checks = {"no_masks": no_mask, "widths": got_w == spec["widths"], "params": params == spec["params"],
                      "macs": macs == spec["macs"], "masked_equivalent_max_abs_diff": diff, "masked_equivalent_ok": diff <= 1e-4,
                      "reload_identical": reload_identical}
            passed = all(v for k, v in checks.items() if k != "masked_equivalent_max_abs_diff")
            ok &= passed
            rows.append({"setting": setting, "method": method, "widths": got_w, "params": params, "macs": macs,
                         "state_dict_bytes": len(buf.getvalue()), **checks, "passed": passed})
            print(f"{setting:18s} {method:8s} widths {got_w} params {params} macs {macs} diff {diff:.2e} -> "
                  f"{'PASS' if passed else 'FAIL'}")
    out = {"created": now(), "all_passed": bool(ok), "checks": rows}
    json.dump(out, open(os.path.join(L.RESULTS, "integrity_checks.json"), "w", encoding="utf-8"), indent=1)
    print("ALL PASSED" if ok else "FAILURES")
    if not ok:
        sys.exit(1)


# ------------------------------------------------------------------ step 6
def pin(cpu):
    import psutil
    os.environ["OMP_NUM_THREADS"] = os.environ["MKL_NUM_THREADS"] = "1"
    psutil.Process().cpu_affinity([cpu])


def recover(ctx, arch, method):
    import psutil
    import torch
    from src import models as M
    from src.evaluation import efficiency as E
    from src.pruning import structured_units as SU
    from src.utils import sha256_state_dict
    setting = L.SETTINGS[arch]
    spec = L.policies()["settings"][setting]["methods"][method]
    t0, started = time.time(), now()
    b, ref, ref_sha = ctx.get(setting)
    m, _ = L.slim(ref, arch, spec["structured_ratios"])
    shape0 = SU.shape_key(m)
    pre_rl, pre_sel = ctx.P6.evaluate_val(m.eval(), b)
    os.makedirs(WORK, exist_ok=True)
    work = os.path.join(WORK, f"{setting}_{method}.state.pt")
    hist, resumed = ctx.P6.finetune(m, b, L.RECOVERY_SEED, L.RECOVERY_EPOCHS, work, log=lambda s: None, validate=False)
    assert SU.shape_key(m) == shape0, "topology changed during recovery"
    post_rl, post_sel = ctx.P6.evaluate_val(m.eval(), b)
    sd = m.state_dict()
    os.makedirs(L.CKPT, exist_ok=True)
    ck = L.ckpt_path(arch, method)
    torch.save(sd, ck)
    rec = {"setting": setting, "arch": arch, "method": method, "status": "complete",
           "structured_ratios": spec["structured_ratios"], "widths": spec["widths"],
           "params": M.count_parameters(m)["total"], "macs": E.conv_linear_macs(m)[0],
           "pre": {"val_rl": pre_rl, "val_select": pre_sel}, "post": {"val_rl": post_rl, "val_select": post_sel},
           "train_loss": [h["train_loss"] for h in hist], "epochs": len(hist), "resumed_from_epoch": resumed,
           "data_stream_sha256": hist[-1]["data_stream_sha256"], "final_state_sha256": sha256_state_dict(sd),
           "checkpoint": os.path.relpath(ck, L.ROOT), "checkpoint_sha256": L.sha256_file(ck),
           "checkpoint_bytes": os.path.getsize(ck),
           "recovery": {"protocol": ctx.P6.PROTOCOL, "epochs": L.RECOVERY_EPOCHS, "seed": L.RECOVERY_SEED,
                        "data_generator_seed": L.RECOVERY_SEED + 10 ** 6},
           "reference_sha256": ref_sha, "threads": torch.get_num_threads(), "cpu_affinity": psutil.Process().cpu_affinity(),
           "seconds": round(time.time() - t0, 1), "started_at": started, "finished_at": now(), "test_use": "none",
           "preregistration_sha256": L.sha256_file(L.PREREG), "git": ctx.git}
    os.makedirs(RUNS, exist_ok=True)
    tmp = record_path(arch, method) + ".tmp"
    json.dump(rec, open(tmp, "w", encoding="utf-8"), indent=1)
    os.replace(tmp, record_path(arch, method))
    if os.path.exists(work):
        os.remove(work)
    return rec


def claim(arch, method):
    os.makedirs(LOCKS, exist_ok=True)
    try:
        fd = os.open(os.path.join(LOCKS, f"{L.SETTINGS[arch]}_{method}.lock"), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        os.write(fd, f"{os.getpid()} {now()}".encode())
        os.close(fd)
        return True
    except FileExistsError:
        return False


def cmd_worker(a):
    pin(a.cpu)
    rec_mod = L.load_module("h2_recovery", L.PAPER3)      # Paper-3 context: frozen Phase-6 protocol, data bundle
    ctx = rec_mod.Ctx()
    for arch, method in JOBS:
        if os.path.exists(record_path(arch, method)) or not claim(arch, method):
            continue
        r = recover(ctx, arch, method)
        print(f"{now()} done {r['setting']} {method} VAL-RL {r['pre']['val_rl']:.2f} -> {r['post']['val_rl']:.2f} "
              f"({r['seconds']:.0f} s)", flush=True)


# ------------------------------------------------------------------ step 7
def cmd_freeze(_a):
    import csv
    rows = []
    for arch, method in JOBS:
        r = json.load(open(record_path(arch, method), encoding="utf-8"))
        assert r["status"] == "complete" and r["epochs"] == L.RECOVERY_EPOCHS and r["test_use"] == "none"
        assert L.sha256_file(os.path.join(L.ROOT, r["checkpoint"])) == r["checkpoint_sha256"]
        rows.append(r)
    streams = {}
    for r in rows:
        streams.setdefault(r["setting"], set()).add(r["data_stream_sha256"])
    same_stream = all(len(v) == 1 for v in streams.values())
    table = []
    for r in rows:
        dense = next(x for x in rows if x["setting"] == r["setting"] and x["method"] == "dense")
        d = r["post"]["val_rl"] - dense["post"]["val_rl"]
        table.append({"setting": r["setting"], "method": r["method"], "widths": json.dumps(r["widths"]),
                      "params": r["params"], "macs": r["macs"],
                      "val_rl_pre": r["pre"]["val_rl"], "val_rl_post": r["post"]["val_rl"],
                      "val_select_pre": r["pre"]["val_select"], "val_select_post": r["post"]["val_select"],
                      "delta_val_rl_pp": round(d, 4), "q_val_rl": r["post"]["val_rl"] / dense["post"]["val_rl"],
                      "accuracy_preserving": abs(d) <= L.ACC_TOLERANCE_PP + 1e-9,
                      "checkpoint_sha256": r["checkpoint_sha256"], "final_state_sha256": r["final_state_sha256"],
                      "data_stream_sha256": r["data_stream_sha256"], "recovery_seconds": r["seconds"]})
    path = os.path.join(L.RESULTS, "accuracy_results.csv")
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(table[0]))
        w.writeheader()
        w.writerows(table)
    freeze = {"frozen_at": now(), "same_data_stream_within_setting": same_stream,
              "models": {f"{r['setting']}_{r['method']}": {"checkpoint": r["checkpoint"], "sha256": r["checkpoint_sha256"]}
                         for r in rows}, "accuracy_results_sha256": L.sha256_file(path)}
    json.dump(freeze, open(os.path.join(L.RESULTS, "recovered_models_freeze.json"), "w", encoding="utf-8"), indent=1)
    for t in table:
        print(f"{t['setting']:18s} {t['method']:8s} VAL-RL {t['val_rl_pre']:6.2f} -> {t['val_rl_post']:6.2f} "
              f"delta {t['delta_val_rl_pp']:+.2f} pp preserving={t['accuracy_preserving']}")
    print("same data stream within each setting:", same_stream)


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("check")
    w = sub.add_parser("worker")
    w.add_argument("--cpu", type=int, required=True)
    sub.add_parser("freeze")
    a = p.parse_args()
    {"check": cmd_check, "worker": cmd_worker, "freeze": cmd_freeze}[a.cmd](a)


if __name__ == "__main__":
    main()
