"""Phase 7A, step 3a: derive the compared allocations from frozen inputs only (no accuracy, latency or energy read).

  python experiments/phase7a/select_policies.py   -> results/phase7a_sustainability/selected_policies.json
PPO       most frequent Phase-5 C1 selected policy over the 20 policy seeds (ties: smallest action vector)
LAMP      LAMP per-unit sparsity at PPO's zero count (Phase-6 matched masks), unit ratio = zeros / weights
Uniform   one common ratio r for all four units, r on a 0.001 grid minimizing |MACs - MACs_PPO| (ties: smallest r)
Dense     all ratios 0
Every allocation is realized with the same rule: structured unit k keeps max(1, round((1 - r_k) C_k)) channels.
"""
import collections
import datetime
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import p7_lib as L  # noqa: E402

P5_POLICIES = os.path.join(L.ROOT, "results", "phase5", "phase5_final_policies.json")
P6_MASKS = os.path.join(L.ROOT, "results", "phase6", "phase6_matched_masks.csv")


def tensor_sizes(ref):
    return {n: m.weight.numel() for n, m in ref.named_modules() if hasattr(m, "weight") and m.weight is not None
            and m.weight.dim() > 1}


def describe(arch, ratios):
    w = L.widths(arch, ratios)
    macs, params = L.analytic(arch, w)
    m0, p0 = L.analytic(arch, L.widths(arch, [0, 0, 0, 0]))
    return {"structured_ratios": [round(float(r), 6) for r in ratios], "widths": w, "macs": macs, "params": params,
            "mac_reduction": 1 - macs / m0, "param_reduction": 1 - params / p0}


def main():
    import pandas as pd
    p5 = json.load(open(P5_POLICIES, encoding="utf-8"))
    masks = pd.read_csv(P6_MASKS)
    out = {"created": datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
           "sources": {"phase5_final_policies.json": L.sha256_file(P5_POLICIES),
                       "phase6_matched_masks.csv": L.sha256_file(P6_MASKS)},
           "realization_rule": "structured unit k keeps max(1, round((1 - r_k) * C_k)) channels with the largest L1 norm "
                               "(src/pruning/structured_units.py); thesis unit k -> structured unit k by index",
           "settings": {}}
    for arch, setting in L.SETTINGS.items():
        runs = [r for r in p5["runs"] if r["setting"] == setting and r["condition"] == "C1"]
        assert len(runs) == 20 and not any(r["fallback"] for r in runs)
        counts = collections.Counter(tuple(r["selected_policy"]) for r in runs)
        top = max(counts.values())
        ppo = min(p for p, c in counts.items() if c == top)
        ppo_ratios = [L.ACTION_RATIO[a] for a in ppo]
        ppo_d = describe(arch, ppo_ratios)

        rows = masks[(masks.setting == setting) & (masks.policy == json.dumps(list(ppo)))]
        assert len(rows) == top, (len(rows), top)
        lamp_sets = {r.lamp_layer_zeros for r in rows.itertuples()}
        assert len(lamp_sets) == 1, "LAMP allocation differs between seeds with the same PPO policy"
        zeros = json.loads(lamp_sets.pop())
        ref, ref_sha = L.p3().reference(setting)
        sizes = tensor_sizes(ref)
        lamp_ratios = [sum(zeros[t] for t in unit) / sum(sizes[t] for t in unit) for unit in L.THESIS_UNITS[arch]]
        ppo_zero_check = [sum(round(L.ACTION_RATIO[a] * sizes[t]) for t in unit) for a, unit in zip(ppo, L.THESIS_UNITS[arch])]

        best = None
        for r in L.UNIFORM_GRID:
            macs = L.analytic(arch, L.widths(arch, [r] * 4))[0]
            gap = abs(macs - ppo_d["macs"])
            if best is None or gap < best[0]:
                best = (gap, r)
        uni = describe(arch, [best[1]] * 4)

        out["settings"][setting] = {
            "arch": arch, "reference_sha256": ref_sha,
            "ppo_source": {"condition": "C1 (Phase 5; the policy fine-tuned in Phase 6)", "seeds": "500-519",
                           "policy_counts": {json.dumps(list(p)): c for p, c in counts.most_common()},
                           "selected": list(ppo), "frequency": f"{top}/20",
                           "seeds_with_policy": sorted(r["seed"] for r in runs if tuple(r["selected_policy"]) == ppo)},
            "methods": {
                "dense": {**describe(arch, [0.0] * 4), "origin": "identity policy"},
                "ppo": {**ppo_d, "actions": list(ppo), "thesis_unit_ratios": ppo_ratios,
                        "origin": "most frequent Phase-5 C1 policy; action ratios realized as structured channel removal",
                        "rounding": "channels = max(1, round((1 - r) C)); widths above"},
                "uniform": {**uni, "matching": "MAC reduction", "mac_gap_to_ppo": best[0],
                            "mac_gap_relative": best[0] / ppo_d["macs"],
                            "origin": "common ratio on a 0.001 grid minimizing |MACs - MACs_PPO|, ties -> smallest ratio"},
                "lamp": {**describe(arch, lamp_ratios), "thesis_unit_ratios": [round(x, 6) for x in lamp_ratios],
                         "thesis_unit_zero_counts": [sum(zeros[t] for t in unit) for unit in L.THESIS_UNITS[arch]],
                         "ppo_thesis_unit_zero_counts": ppo_zero_check,
                         "origin": "Phase-6 LAMP mask at exactly PPO's zero count; per-unit sparsity realized as "
                                   "structured channel removal"}},
        }
        if arch == "resnet8":
            out["settings"][setting]["mapping_note"] = (
                "Approximate realization: thesis units stage2/stage3 pruned all weights of a stage (conv1, conv2, "
                "shortcut), whereas the validated structured units remove block-internal channels only (stage output "
                "widths are not pruned); thesis unit 'stem' (conv1 weights) maps to the coupled stage-1 residual width.")
    os.makedirs(L.RESULTS, exist_ok=True)
    path = os.path.join(L.RESULTS, "selected_policies.json")
    json.dump(out, open(path, "w", encoding="utf-8"), indent=1)
    for s, v in out["settings"].items():
        print(s, "PPO", v["ppo_source"]["selected"], v["ppo_source"]["frequency"])
        for k, m in v["methods"].items():
            print(f"  {k:8s} ratios {m['structured_ratios']} widths {m['widths']} MAC-red {m['mac_reduction']:.3f} "
                  f"param-red {m['param_reduction']:.3f}")


if __name__ == "__main__":
    main()
