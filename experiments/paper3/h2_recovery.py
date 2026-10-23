"""Paper 3, H2-B / H2-C: fixed 5-epoch recovery of structured policies (pre-registration §1, §5).

  python experiments/paper3/h2_recovery.py sanity                      # H2-B: representative policies + checks
  python experiments/paper3/h2_recovery.py worker --cpu C              # H2-C: one worker, pinned to logical CPU C
  python experiments/paper3/h2_recovery.py repeat --cpu C              # H2-B: determinism rerun on another core
Every policy: frozen dense reference -> structured_units.slim -> phase6_lib.finetune (frozen Phase-6 protocol,
5 epochs, seed 600, 1 thread, validate=False) -> VAL-RL and VAL-SELECT accuracy. The test set is never read.
Workers claim jobs through atomic lock files, so any number of workers on any cores can share the queue;
which core runs a policy does not change its result (checked in the sanity run).
"""
import argparse
import datetime
import hashlib
import io
import itertools
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import p3_lib as L  # noqa: E402

SETTINGS = ["cifar10_simplecnn", "cifar10_resnet8", "cifar100_simplecnn", "cifar100_resnet8"]
POLICIES = [tuple(p) for p in itertools.product(range(6), repeat=4)]
SEED, EPOCHS = 600, 5
RUNS = os.path.join(L.RESULTS, "_h2_recovery_runs")
LOCKS = os.path.join(RUNS, "_locks")
WORK = os.path.join(L.ROOT, "checkpoints", "paper3_work")
SANITY = {"simplecnn": {"identity": [0, 0, 0, 0], "uniform10": [1, 1, 1, 1], "mixed": [2, 3, 3, 1], "fc_heavy": [0, 0, 0, 5],
                        "conv_heavy": [0, 5, 5, 0], "aggressive": [5, 5, 5, 5]},
          "resnet8": {"identity": [0, 0, 0, 0], "uniform10": [1, 1, 1, 1], "mixed": [2, 3, 3, 1], "stem_heavy": [5, 0, 0, 0],
                      "conv_heavy": [0, 5, 5, 5], "aggressive": [5, 5, 5, 5]}}


def now():
    return datetime.datetime.now().astimezone().isoformat(timespec="seconds")


def pin(cpu):
    import psutil
    os.environ["OMP_NUM_THREADS"] = os.environ["MKL_NUM_THREADS"] = "1"
    psutil.Process().cpu_affinity([cpu])


def durable(data, path):
    tmp = f"{path}.{os.getpid()}.tmp"
    with open(tmp, "wb") as f:
        f.write(data)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def key(policy):
    return "".join(map(str, policy))


class Ctx:
    def __init__(self):
        import torch
        torch.set_num_interop_threads(1)
        torch.set_num_threads(1)
        self.P6 = L.phase_module("phase6_lib", os.path.join(L.ROOT, "experiments", "phase6"))
        self.bundle_ds, self.bundle, self.refs = None, None, {}
        from src.utils import git_state
        self.git = git_state()
        self.code = {f: hashlib.sha256(open(os.path.join(HERE, f), "rb").read()).hexdigest()
                     for f in ("h2_recovery.py", "p3_lib.py")}
        self.code["phase6_lib.py"] = hashlib.sha256(open(os.path.join(L.ROOT, "experiments", "phase6", "phase6_lib.py"), "rb").read()).hexdigest()
        self.code["structured_units.py"] = hashlib.sha256(open(os.path.join(L.ROOT, "src", "pruning", "structured_units.py"), "rb").read()).hexdigest()
        self.prereg = hashlib.sha256(open(os.path.join(L.RESULTS, "paper3_preregistration.md"), "rb").read()).hexdigest()

    def get(self, setting):
        ds = setting.split("_")[0]
        if self.bundle_ds != ds:                     # one dataset in memory per worker
            self.bundle, self.bundle_ds = None, None
            self.bundle, self.bundle_ds = self.P6.bundle(ds), ds
        if setting not in self.refs:
            self.refs[setting] = L.reference(setting)
        return self.bundle, *self.refs[setting]


def recover_one(ctx, setting, policy, out_path, work_tag=""):
    import psutil
    import torch
    from src import models as M
    from src.evaluation import efficiency as E
    from src.pruning import structured_units as SU
    from src.utils import sha256_state_dict
    t0, started = time.time(), now()
    b, ref, ref_sha = ctx.get(setting)
    arch = setting.split("_")[1]
    m, _ = SU.slim(ref, arch, policy)
    shape0 = SU.shape_key(m)
    pre_rl, pre_sel = ctx.P6.evaluate_val(m.eval(), b)
    os.makedirs(WORK, exist_ok=True)
    work = os.path.join(WORK, f"{setting}_{key(policy)}{work_tag}.state.pt")
    hist, resumed = ctx.P6.finetune(m, b, SEED, EPOCHS, work, log=lambda s: None, validate=False)
    assert SU.shape_key(m) == shape0, "topology changed during recovery"
    post_rl, post_sel = ctx.P6.evaluate_val(m.eval(), b)
    sd = m.state_dict()
    rec = {"setting": setting, "arch": arch, "policy": list(policy), "status": "complete", "widths": SU.widths(arch, policy),
           "params": M.count_parameters(m)["total"], "macs": E.conv_linear_macs(m)[0], "shape_unchanged": True,
           "pre": {"val_rl": pre_rl, "val_select": pre_sel}, "post": {"val_rl": post_rl, "val_select": post_sel},
           "train_loss": [h["train_loss"] for h in hist], "train_accuracy": [h["train_accuracy"] for h in hist],
           "lr": [h["lr"] for h in hist], "epochs": len(hist), "resumed_from_epoch": resumed,
           "data_stream_sha256": hist[-1]["data_stream_sha256"], "final_state_sha256": sha256_state_dict(sd),
           "recovery": {"protocol": ctx.P6.PROTOCOL, "epochs": EPOCHS, "seed": SEED, "data_generator_seed": SEED + 10 ** 6},
           "reference_sha256": ref_sha, "threads": torch.get_num_threads(), "cpu_affinity": psutil.Process().cpu_affinity(),
           "seconds": round(time.time() - t0, 1), "train_seconds": round(sum(h["train_seconds"] for h in hist), 1),
           "peak_memory_mb": ctx.P6.peak_memory_mb(), "started_at": started, "finished_at": now(),
           "test_use": "none", "preregistration_sha256": ctx.prereg, "code_sha256": ctx.code, "git": ctx.git}
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    durable(json.dumps(rec, indent=1).encode("utf-8"), out_path)
    if os.path.exists(work):
        os.remove(work)
    return rec, m


# ------------------------------------------------------------------ H2-B sanity
def cmd_sanity(a):
    pin(a.cpu)
    import torch
    ctx = Ctx()
    from src.pruning import structured_units as SU
    out = os.path.join(L.RESULTS, "_h2_sanity")
    checks, rows = [], []

    def chk(name, ok, detail=""):
        checks.append({"check": name, "pass": bool(ok), "detail": str(detail)})
        print(f"[{'PASS' if ok else 'FAIL'}] {name}: {str(detail)[:160]}", flush=True)
    for arch, pols in SANITY.items():
        if a.arch and arch != a.arch:
            continue
        setting = f"cifar10_{arch}"
        for name, p in pols.items():
            path = os.path.join(out, f"{setting}_{key(p)}.json")
            rec, m = recover_one(ctx, setting, p, path)
            b, ref, _ = ctx.get(setting)
            sk = SU.skeleton(arch, p, 10)
            buf = io.BytesIO()
            torch.save(m.state_dict(), buf)
            sk.load_state_dict(torch.load(io.BytesIO(buf.getvalue()), weights_only=True), strict=True)
            x = b.rl_tensors()[0][:256]
            with torch.no_grad():
                reload_ok = torch.equal(sk.eval()(x), m.eval()(x))
            rows.append({"setting": setting, "name": name, **{k: rec[k] for k in ("policy", "widths", "params", "macs", "seconds", "peak_memory_mb",
                                                                                   "epochs", "shape_unchanged")},
                         "pre_val_rl": rec["pre"]["val_rl"], "post_val_rl": rec["post"]["val_rl"], "loss_epoch1": rec["train_loss"][0],
                         "loss_epoch5": rec["train_loss"][-1], "reload_ok": reload_ok, "final_state_sha256": rec["final_state_sha256"]})
            print(f"  {setting} {name} {p}: VAL-RL {rec['pre']['val_rl']:.2f} -> {rec['post']['val_rl']:.2f}, {rec['seconds']:.0f}s", flush=True)
    for arch in [x for x in SANITY if not a.arch or x == a.arch]:
        r = [x for x in rows if x["setting"] == f"cifar10_{arch}"]
        chk(f"{arch}: every recovery ran 5 epochs with shapes unchanged (no topology change, no regrowth)",
            all(x["epochs"] == EPOCHS and x["shape_unchanged"] for x in r), len(r))
        chk(f"{arch}: training loss falls from epoch 1 to epoch 5 for every policy", all(x["loss_epoch5"] < x["loss_epoch1"] for x in r),
            [(x["name"], round(x["loss_epoch1"], 3), round(x["loss_epoch5"], 3)) for x in r])
        chk(f"{arch}: VAL-RL accuracy after recovery >= before for every pruned policy", all(x["post_val_rl"] >= x["pre_val_rl"] for x in r if x["name"] != "identity"),
            [(x["name"], x["pre_val_rl"], x["post_val_rl"]) for x in r])
        chk(f"{arch}: recovered state dict reloads into the policy skeleton (strict) with identical logits", all(x["reload_ok"] for x in r), "")
    json.dump({"rows": rows, "checks": checks, "cpu": a.cpu}, open(os.path.join(out, f"sanity_{a.arch or 'all'}_cpu{a.cpu}.json"), "w", encoding="utf-8"),
              indent=1, default=str)


def cmd_repeat(a):
    """Determinism: rerun two sanity policies on another core; the final state hash must be bit-identical."""
    pin(a.cpu)
    ctx = Ctx()
    out = os.path.join(L.RESULTS, "_h2_sanity")
    res = []
    for setting, p in (("cifar10_simplecnn", SANITY["simplecnn"]["mixed"]), ("cifar10_resnet8", SANITY["resnet8"]["mixed"])):
        rec, _ = recover_one(ctx, setting, p, os.path.join(out, f"repeat_cpu{a.cpu}_{setting}_{key(p)}.json"), work_tag=f"_rep{a.cpu}")
        first = json.load(open(os.path.join(out, f"{setting}_{key(p)}.json"), encoding="utf-8"))
        res.append({"setting": setting, "policy": p, "cpu": a.cpu, "identical_weights": rec["final_state_sha256"] == first["final_state_sha256"],
                    "identical_accuracy": rec["post"] == first["post"], "seconds": rec["seconds"], "first_cpu": first["cpu_affinity"]})
        print(res[-1], flush=True)
    json.dump(res, open(os.path.join(out, f"repeat_cpu{a.cpu}.json"), "w", encoding="utf-8"), indent=1)


# ------------------------------------------------------------------ H2-C workers
def jobs():
    return [(s, p) for s in SETTINGS for p in POLICIES]


def run_path(setting, policy):
    return os.path.join(RUNS, setting, f"{key(policy)}.json")


def done(setting, policy):
    p = run_path(setting, policy)
    if not os.path.exists(p):
        return False
    try:
        r = json.load(open(p, encoding="utf-8"))
        return r["status"] == "complete" and r["epochs"] == EPOCHS
    except Exception:                                        # noqa: BLE001
        with open(os.path.join(RUNS, "corruption_log.txt"), "a", encoding="utf-8") as f:
            f.write(f"{now()} {p}: unreadable record, repeated\n")
        return False


def claim(setting, policy):
    os.makedirs(LOCKS, exist_ok=True)
    try:
        fd = os.open(os.path.join(LOCKS, f"{setting}_{key(policy)}.lock"), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        os.write(fd, str(os.getpid()).encode())
        os.close(fd)
        return True
    except FileExistsError:
        return False


def cmd_worker(a):
    pin(a.cpu)
    ctx = Ctx()
    n = 0
    for setting, policy in jobs():
        if done(setting, policy) or not claim(setting, policy):
            continue
        rec, _ = recover_one(ctx, setting, policy, run_path(setting, policy))
        n += 1
        print(f"{now()} done {setting} {key(policy)} {rec['seconds']:.0f}s", flush=True)
    print(f"{now()} finished ({n} runs)", flush=True)


def cmd_regenerate(a):
    """After the landscape: regenerate 2 policies per setting (drawn with seed 0) and compare with the records."""
    import numpy as np
    pin(a.cpu)
    ctx = Ctx()
    rng = np.random.default_rng(0)
    out, tmp = [], os.path.join(L.RESULTS, "_h2_regenerate_runs")
    for setting in SETTINGS:
        for i in rng.choice(len(POLICIES), 2, replace=False):
            p = POLICIES[int(i)]
            first = json.load(open(run_path(setting, p), encoding="utf-8"))
            rec, _ = recover_one(ctx, setting, p, os.path.join(tmp, f"{setting}_{key(p)}.json"), work_tag="_regen")
            out.append({"setting": setting, "policy": list(p), "cpu": a.cpu, "first_cpu": first["cpu_affinity"],
                        "identical": rec["final_state_sha256"] == first["final_state_sha256"] and rec["post"] == first["post"]})
            print(out[-1], flush=True)
    json.dump(out, open(os.path.join(L.RESULTS, "_h2_regenerate.json"), "w", encoding="utf-8"), indent=1)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("sanity", "repeat", "worker", "regenerate"):
        s = sub.add_parser(name)
        s.add_argument("--cpu", type=int, default=2)
        s.add_argument("--arch", default=None)
    a = ap.parse_args()
    sys.path.insert(0, L.ROOT)
    {"sanity": cmd_sanity, "repeat": cmd_repeat, "worker": cmd_worker, "regenerate": cmd_regenerate}[a.cmd](a)
