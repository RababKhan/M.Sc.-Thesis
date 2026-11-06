"""Paper 3, H1: structured pruning and compaction validation (SimpleCNN, ResNet-8). No timing; no test data.

  python experiments/paper3/h1_validate.py   -> results/paper3/h1_checks.json, h1_validation_report.md

Validation images only (Phase-5 VAL-RL). Structural checks run over all 6^4 = 1,296 action vectors (shapes,
parameter and MAC counts; no accuracy, no latency). Logit-level checks use representative policies.
VAL-RL accuracy of a few representative policies is reported as a sanity check only; it decides nothing.
"""
import io
import itertools
import json
import os
import sys

import numpy as np
import torch
import torch.nn as nn

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import p3_lib as L  # noqa: E402
from src import models as M  # noqa: E402
from src.evaluation import efficiency as E, evaluate_accuracy, mean_loss_accuracy  # noqa: E402
from src.pruning import ACTION_TO_PRUNE, structured as SP, structured_units as SU  # noqa: E402
from src.utils import sha256_state_dict  # noqa: E402

RES = []
POLICIES = [tuple(p) for p in itertools.product(range(6), repeat=4)]
TOL = 1e-4


def chk(section, name, ok, detail=""):
    RES.append({"section": section, "check": name, "pass": bool(ok), "detail": str(detail)})
    print(f"[{'PASS' if ok else 'FAIL'}] {section} | {name}: {str(detail)[:160]}", flush=True)


def representative(arch):
    rng = np.random.default_rng(0)
    reps = [[0, 0, 0, 0], [1, 1, 1, 1], [3, 3, 3, 3], [5, 5, 5, 5]]
    reps += [[5 if i == u else 0 for i in range(4)] for u in range(4)]
    reps += list(L.CALIBRATION.values()) if arch == "simplecnn" else [[5, 0, 0, 0], [5, 5, 0, 0], [3, 0, 5, 2], [0, 4, 4, 4]]
    reps += [list(map(int, rng.integers(0, 6, 4))) for _ in range(20)]
    out = []
    for p in reps:
        if p not in out:
            out.append(p)
    return out


def no_masks(model):
    return not any(hasattr(m, "weight_mask") or hasattr(m, "weight_orig") or nn.utils.prune.is_pruned(m) for m in model.modules())


def reload_equal(model, arch, acts, classes, x):
    buf = io.BytesIO()
    torch.save(model.state_dict(), buf)
    sk = SU.skeleton(arch, acts, classes)
    sk.load_state_dict(torch.load(io.BytesIO(buf.getvalue()), weights_only=True), strict=True)
    with torch.no_grad():
        return torch.equal(sk.eval()(x), model(x))


def structural(arch, classes):
    """All 1,296 policies: unique shapes; physical parameters and hook MACs equal the analytic counts; monotone."""
    base = M.build_model(arch, classes, seed=0).eval()
    shapes, bad, table = set(), [], {}
    for p in POLICIES:
        m, _ = SU.slim(base, arch, p)
        shapes.add(SU.shape_key(m))
        macs, _ = E.conv_linear_macs(m)
        params = M.count_parameters(m)["total"]
        table[p] = (params, macs)
        if (macs, params) != SU.analytic(arch, p, classes) or not no_masks(m):
            bad.append(p)
    dense = table[(0, 0, 0, 0)]
    mono = all(table[q][0] <= table[p][0] and table[q][1] < table[p][1]
               for p in POLICIES for u in range(4) if p[u] < 5 for q in [tuple(p[i] + (i == u) for i in range(4))])
    smaller = all(table[p][0] < dense[0] and table[p][1] < dense[1] for p in POLICIES if any(p))
    return shapes, bad, mono, smaller, table


def logits(model, x):
    with torch.no_grad():
        return model.eval()(x)


def validate_simplecnn():
    sec = "SimpleCNN"
    w = {u: [SU.keep_count(c, ACTION_TO_PRUNE[a]) for a in range(6)] for u, c in zip(SU.UNITS["simplecnn"], SU.BASE_WIDTHS["simplecnn"])}
    chk(sec, "every action gives a valid width (1 <= w <= C) and the 6 widths of each unit are distinct",
        all(1 <= v <= c for (u, ws), c in zip(w.items(), SU.BASE_WIDTHS["simplecnn"]) for v in ws) and all(len(set(ws)) == 6 for ws in w.values()), w)
    shapes, bad, mono, smaller, table = structural("simplecnn", 10)
    chk(sec, "1,296 action vectors -> 1,296 distinct physical shapes", len(shapes) == 1296, len(shapes))
    chk(sec, "all 1,296: physical parameters and hook MACs equal the analytic counts; no pruning mask left", not bad, bad[:3])
    chk(sec, "parameters and MACs physically decrease for every non-zero policy; raising any action never adds parameters and always removes MACs",
        mono and smaller, f"dense {table[(0, 0, 0, 0)]}, [5,5,5,5] {table[(5, 5, 5, 5)]}")
    for ds in ("cifar10", "cifar100"):
        ref, _ = L.reference(f"{ds}_simplecnn")
        classes = 10 if ds == "cifar10" else 100
        x, y, _ = L.val_rl(ds)
        xs = x[:512]
        d = logits(ref, xs)
        ident, k0 = SU.slim(ref, "simplecnn", [0, 0, 0, 0])
        chk(sec, f"{ds}: [0,0,0,0] reproduces the dense model exactly (logits and state dict)",
            torch.equal(logits(ident, xs), d) and all(torch.equal(ref.state_dict()[k], v) for k, v in ident.state_dict().items()), "")
        worst, agree, ok_shape, ok_mask, ok_reload, ok_flat, ok_cls = 0.0, 1.0, True, True, True, True, True
        for p in representative("simplecnn"):
            m, kept = SU.slim(ref, "simplecnn", p)
            out = logits(m, xs)
            eq = logits(SU.masked_equivalent(ref, "simplecnn", kept), xs)
            worst = max(worst, float((out - eq).abs().max()))
            agree = min(agree, float((out.argmax(1) == eq.argmax(1)).float().mean()))
            ok_shape &= tuple(out.shape) == (512, classes)
            ok_mask &= no_masks(m)
            ok_reload &= reload_equal(m, "simplecnn", p, classes, xs)
            cols = [c * 16 + s for c in kept["conv3"] for s in range(16)]
            ok_flat &= torch.equal(m.classifier[1].weight, ref.classifier[1].weight[kept["fc1"]][:, cols])
            ok_cls &= m.classifier[3].out_features == classes and torch.equal(m.classifier[3].weight, ref.classifier[3].weight[:, kept["fc1"]]) \
                and torch.equal(m.classifier[3].bias, ref.classifier[3].bias)
        n = len(representative("simplecnn"))
        chk(sec, f"{ds}: slimmed logits == masked-equivalent logits ({n} policies, 512 VAL-RL images)", worst < TOL and agree == 1.0,
            f"max |dlogit| {worst:.2e}, argmax agreement {agree:.3f}")
        chk(sec, f"{ds}: output dimension ({classes}) correct; no mask after compaction; state dict reloads into the policy skeleton (strict) with identical logits",
            ok_shape and ok_mask and ok_reload, "")
        chk(sec, f"{ds}: flatten mapping: classifier.1 keeps exactly columns c*16..c*16+15 of every kept conv3 channel", ok_flat, "")
        chk(sec, f"{ds}: output classifier never pruned (all {classes} outputs and the bias kept; only its inputs follow fc1)", ok_cls, "")
    return table


def bn_ok(bn, dense_bn, idx):
    return bn.num_features == len(idx) and all(torch.equal(getattr(bn, a), getattr(dense_bn, a)[idx])
                                               for a in ("weight", "bias", "running_mean", "running_var")) \
        and torch.equal(bn.num_batches_tracked, dense_bn.num_batches_tracked) and bn.eps == dense_bn.eps


def block_shapes(block, x):
    """(residual branch shape, shortcut shape, output shape) of one BasicBlock, computed explicitly."""
    import torch.nn.functional as F
    with torch.no_grad():
        out = block.bn2(block.conv2(F.relu(block.bn1(block.conv1(x)))))
        sc = block.shortcut(x)
        return tuple(out.shape), tuple(sc.shape), block(x)


def validate_resnet8():
    sec = "ResNet-8"
    w = {u: [SU.keep_count(c, ACTION_TO_PRUNE[a]) for a in range(6)] for u, c in zip(SU.UNITS["resnet8"], SU.BASE_WIDTHS["resnet8"])}
    chk(sec, "every action gives a valid width and the 6 widths of each unit are distinct",
        all(1 <= v <= c for (u, ws), c in zip(w.items(), SU.BASE_WIDTHS["resnet8"]) for v in ws) and all(len(set(ws)) == 6 for ws in w.values()), w)
    shapes, bad, mono, smaller, table = structural("resnet8", 10)
    chk(sec, "1,296 action vectors -> 1,296 distinct physical shapes", len(shapes) == 1296, len(shapes))
    chk(sec, "all 1,296: physical parameters and hook MACs equal the analytic counts; no pruning mask left", not bad, bad[:3])
    chk(sec, "parameters and MACs physically decrease for every non-zero policy; monotone in every unit", mono and smaller,
        f"dense {table[(0, 0, 0, 0)]}, [5,5,5,5] {table[(5, 5, 5, 5)]}")
    for ds in ("cifar10", "cifar100"):
        ref, _ = L.reference(f"{ds}_resnet8")
        classes = 10 if ds == "cifar10" else 100
        x, y, _ = L.val_rl(ds)
        xs, xr = x[:512], torch.randn((64, 3, 32, 32), generator=torch.Generator().manual_seed(1))
        d = logits(ref, xs)
        ident, _ = SU.slim(ref, "resnet8", [0, 0, 0, 0])
        chk(sec, f"{ds}: [0,0,0,0] reproduces the dense model exactly (logits and state dict)",
            torch.equal(logits(ident, xs), d) and all(torch.equal(ref.state_dict()[k], v) for k, v in ident.state_dict().items()), "")
        same_p2 = True
        for p in [q for q in representative("resnet8") if q[0] == 0]:
            a, _ = SU.slim(ref, "resnet8", p)
            b, _ = SP.resnet8_structured(ref, [ACTION_TO_PRUNE[v] for v in p[1:]])
            same_p2 &= torch.equal(logits(a, xs), logits(b, xs))
        chk(sec, f"{ds}: with the stem action 0, the 3 block-internal units equal the verified Phase-2 resnet8_structured exactly", same_p2, "")
        worst, agree, ok = 0.0, 1.0, {"conv": True, "bn": True, "group": True, "residual": True, "fc": True, "forward": True, "mask": True, "reload": True}
        for p in representative("resnet8"):
            m, k = SU.slim(ref, "resnet8", p)
            g, i1, i2, i3 = k["stem"], k["stage1"], k["stage2"], k["stage3"]
            l1, l2, l3 = m.layer1, m.layer2, m.layer3
            ok["conv"] &= (m.conv1.out_channels == m.bn1.num_features == l1.conv1.in_channels == l1.conv2.out_channels == l1.bn2.num_features
                           == l2.conv1.in_channels == l2.shortcut[0].in_channels == len(g)) \
                and l1.conv1.out_channels == l1.bn1.num_features == l1.conv2.in_channels == len(i1) \
                and l2.conv1.out_channels == l2.bn1.num_features == l2.conv2.in_channels == len(i2) \
                and l3.conv1.out_channels == l3.bn1.num_features == l3.conv2.in_channels == len(i3) \
                and l2.conv2.out_channels == l2.shortcut[0].out_channels == l2.bn2.num_features == l3.conv1.in_channels == l3.shortcut[0].in_channels == 32 \
                and l3.conv2.out_channels == l3.shortcut[0].out_channels == l3.bn2.num_features == m.fc.in_features == 64
            ok["bn"] &= bn_ok(m.bn1, ref.bn1, g) and bn_ok(l1.bn1, ref.layer1.bn1, i1) and bn_ok(l1.bn2, ref.layer1.bn2, g) \
                and bn_ok(l2.bn1, ref.layer2.bn1, i2) and bn_ok(l3.bn1, ref.layer3.bn1, i3) \
                and bn_ok(l2.bn2, ref.layer2.bn2, list(range(32))) and bn_ok(l2.shortcut[1], ref.layer2.shortcut[1], list(range(32))) \
                and bn_ok(l3.bn2, ref.layer3.bn2, list(range(64))) and bn_ok(l3.shortcut[1], ref.layer3.shortcut[1], list(range(64)))
            ok["group"] &= torch.equal(m.conv1.weight, ref.conv1.weight[g]) and torch.equal(l1.conv1.weight, ref.layer1.conv1.weight[i1][:, g]) \
                and torch.equal(l1.conv2.weight, ref.layer1.conv2.weight[g][:, i1]) and torch.equal(l2.conv1.weight, ref.layer2.conv1.weight[i2][:, g]) \
                and torch.equal(l2.shortcut[0].weight, ref.layer2.shortcut[0].weight[:, g]) and len(l1.shortcut) == 0
            ok["fc"] &= torch.equal(m.fc.weight, ref.fc.weight) and torch.equal(m.fc.bias, ref.fc.bias) and m.fc.out_features == classes
            h = torch.relu(m.bn1(m.conv1(xr)))
            for blk in (l1, l2, l3):
                rb, sb, h = block_shapes(blk, h)
                ok["residual"] &= rb == sb                                   # identical shapes: no broadcasting
            out_r, out = logits(m, xr), logits(m, xs)
            ok["forward"] &= tuple(out_r.shape) == (64, classes) and tuple(out.shape) == (512, classes) and bool(torch.isfinite(out).all())
            eq = logits(SU.masked_equivalent(ref, "resnet8", k), xs)
            worst = max(worst, float((out - eq).abs().max()))
            agree = min(agree, float((out.argmax(1) == eq.argmax(1)).float().mean()))
            ok["mask"] &= no_masks(m)
            ok["reload"] &= reload_equal(m, "resnet8", p, classes, xs)
        n = len(representative("resnet8"))
        chk(sec, f"{ds}: Conv2d input/output channels consistent along the coupled stem group, the three block-internal units and the unpruned widths ({n} policies)", ok["conv"], "")
        chk(sec, f"{ds}: BatchNorm num_features, weight, bias, running_mean, running_var, eps, num_batches_tracked = dense values at the kept channels", ok["bn"], "")
        chk(sec, f"{ds}: coupled stem group: the same channel set is cut in stem conv/bn, layer1.conv1 inputs, layer1.conv2/bn2 outputs, layer2.conv1 and layer2.shortcut inputs; layer1 shortcut stays identity", ok["group"], "")
        chk(sec, f"{ds}: residual branch and shortcut have identical shapes in every block (no broadcasting)", ok["residual"], "")
        chk(sec, f"{ds}: classifier input 64 and the classifier unchanged ({classes} outputs)", ok["fc"], "")
        chk(sec, f"{ds}: random and real-data forward passes have the right output shape and finite logits", ok["forward"], "")
        chk(sec, f"{ds}: slimmed logits == masked-equivalent logits ({n} policies, 512 VAL-RL images)", worst < TOL and agree == 1.0,
            f"max |dlogit| {worst:.2e}, argmax agreement {agree:.3f}")
        chk(sec, f"{ds}: no mask after compaction; state dict reloads into the policy skeleton (strict) with identical logits", ok["mask"] and ok["reload"], "")
    return table


def counters():
    sec = "MAC/FLOP"
    rows = []
    for arch in ("simplecnn", "resnet8"):
        ref, _ = L.reference(f"cifar10_{arch}")
        for p in representative(arch)[:10]:
            m, _ = SU.slim(ref, arch, p)
            macs, _ = E.conv_linear_macs(m)
            pt, _ = E.ptflops_macs(m)
            rows.append({"arch": arch, "policy": p, "analytic": SU.analytic(arch, p, 10)[0], "hooks": macs, "flops": E.torch_flops(m), "ptflops": pt})
    chk(sec, "analytic MACs = forward-hook MACs; FlopCounter FLOPs = 2 x MACs (representative policies, both architectures)",
        all(r["analytic"] == r["hooks"] and r["flops"] == 2 * r["hooks"] for r in rows), len(rows))
    ratio = [r["ptflops"] / r["hooks"] for r in rows]
    chk(sec, "ptflops (adds BatchNorm, ReLU, pooling) >= conv/linear MACs and within 10% of them", all(1.0 <= q <= 1.10 for q in ratio),
        f"ptflops / MACs {min(ratio):.4f}-{max(ratio):.4f}")
    return rows


def sensitivity():
    sec = "structured sensitivity"
    out = {}
    for arch in ("simplecnn", "resnet8"):
        ref, _ = L.reference(f"cifar10_{arch}")
        x, y, _ = L.val_rl("cifar10")
        xs, ys = x[:512], y[:512]
        h0 = sha256_state_dict(ref.state_dict())
        s1 = SU.structured_loss_sensitivity(ref, arch, xs, ys)
        s2 = SU.structured_loss_sensitivity(ref, arch, xs, ys)
        chk(sec, f"{arch}: the dense model is unchanged after probing (state-dict hash)", sha256_state_dict(ref.state_dict()) == h0, "")
        chk(sec, f"{arch}: deterministic (two calls identical)", s1 == s2, "")
        n = s1["normalized"]
        chk(sec, f"{arch}: normalised values in [0, 1] with min 0 and max 1", min(n) == 0.0 and max(n) == 1.0 and all(0 <= v <= 1 for v in n), [round(v, 3) for v in n])
        cons = True
        for u, name in enumerate(SU.UNITS[arch]):
            m, _ = SU.slim(ref, arch, SU.single_unit_actions(u, 0.6))
            cons &= mean_loss_accuracy(m, xs, ys)[0] == s1["detail"][f"{name}@60"]["loss"]
        chk(sec, f"{arch}: each probe equals the compacted single-unit policy of the action space (60% probes)", cons, "")
        zero, _ = SU.slim(ref, arch, SU.single_unit_actions(0, 0.0))
        chk(sec, f"{arch}: a 0% probe is the dense model (zero loss change)", mean_loss_accuracy(zero, xs, ys)[0] == s1["baseline_loss"], "")
        out[arch] = {"units": s1["units"], "raw": s1["raw"], "normalized": n, "n_images": s1["n_images"], "status": "unit test only, not a final vector"}
    return out


def sanity_accuracy():
    """VAL-RL accuracy of a few representative structured policies (sanity only; decides nothing)."""
    rows = []
    for arch in ("simplecnn", "resnet8"):
        ref, _ = L.reference(f"cifar10_{arch}")
        x, y, _ = L.val_rl("cifar10")
        from src.data import eval_loader
        loader = eval_loader(x, y)
        dense = evaluate_accuracy(ref, loader)
        pol = dict(L.CALIBRATION) if arch == "simplecnn" else {"dense": [0, 0, 0, 0], "light": [1, 1, 1, 1], "moderate": [3, 3, 3, 3],
                                                                  "aggressive": [5, 5, 5, 5], "stem_only_60": [5, 0, 0, 0]}
        for name, p in pol.items():
            m, _ = SU.slim(ref, arch, p)
            a = evaluate_accuracy(m, loader)
            rows.append({"arch": arch, "policy": name, "actions": p, "val_rl_accuracy": a, "retention": a / dense,
                         "params": M.count_parameters(m)["total"], "macs": E.conv_linear_macs(m)[0]})
    return rows


def main():
    torch.set_num_threads(4)
    validate_simplecnn()
    validate_resnet8()
    cnt = counters()
    sens = sensitivity()
    acc = sanity_accuracy()
    n_ok = sum(r["pass"] for r in RES)
    json.dump({"checks": RES, "counters": cnt, "sensitivity_unit_test": sens, "sanity_val_rl": acc, "passed": n_ok, "total": len(RES)},
              open(os.path.join(L.RESULTS, "h1_checks.json"), "w", encoding="utf-8"), indent=1, default=str)
    lines = ["# Paper 3 — H1 structured pruning validation", "", f"**{n_ok}/{len(RES)} checks passed.** Validation images only "
             "(Phase-5 VAL-RL); the test set is never read. Structural checks cover all 1,296 action vectors per architecture.", "",
             "| Status | Section | Check | Detail |", "|---|---|---|---|"]
    lines += [f"| {'PASS' if r['pass'] else 'FAIL'} | {r['section']} | {r['check']} | {r['detail'].replace('|', '/')[:200]} |" for r in RES]
    lines += ["", "## MAC/FLOP counters (representative policies)", "", "| Arch | Policy | Analytic MACs | Hook MACs | FlopCounter FLOPs | ptflops MACs |",
              "|---|---|---|---|---|---|"]
    lines += [f"| {r['arch']} | {r['policy']} | {r['analytic']:,} | {r['hooks']:,} | {r['flops']:,} | {r['ptflops']:,} |" for r in cnt]
    lines += ["", "## Structured sensitivity (unit test on 512 VAL-RL images; not a final vector)", ""]
    lines += [f"- {a}: units {v['units']}, raw {[round(q, 4) for q in v['raw']]}, normalised {[round(q, 3) for q in v['normalized']]}" for a, v in sens.items()]
    lines += ["", "## Sanity: VAL-RL accuracy of representative policies (CIFAR-10; 2,500 images; decides nothing)", "",
              "| Arch | Policy | Actions | Params | MACs | VAL-RL acc. (%) | Retention |", "|---|---|---|---|---|---|---|"]
    lines += [f"| {r['arch']} | {r['policy']} | {r['actions']} | {r['params']:,} | {r['macs']:,} | {r['val_rl_accuracy']:.2f} | {r['retention']:.4f} |" for r in acc]
    open(os.path.join(L.RESULTS, "h1_validation_report.md"), "w", encoding="utf-8").write("\n".join(lines) + "\n")
    print(f"{n_ok}/{len(RES)} passed")


if __name__ == "__main__":
    main()
