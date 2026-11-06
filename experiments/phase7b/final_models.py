"""Phase 7B, Stage B: the compared models (design section 11), derived from frozen inputs and the confirmatory runs.

  python experiments/phase7b/final_models.py   -> selected_policies.json, final_jobs.json
Primary new PPO  the most frequent confirmatory selected policy (ties: lower predicted energy, then smaller vector)
Per-seed         every distinct confirmatory selection
Enumeration ref  the per-seed selection rule applied to all proxy-feasible policies of the 1,296-policy cache
Uniform          one common ratio on a 0.001 grid (0.000-0.900) minimizing |MACs - MACs_primary| (ties: smaller ratio)
LAMP             LAMP scores on the dense reference (src.pruning.lamp) at global sparsity s on a 0.1% grid
                 (0.0-90.0%); unit ratio = zeros / weights of the unit; the s whose structured realization has the
                 parameter count closest to the primary's (ties: smaller s)
Old PPO          the Phase-7A model [0,2,5,5], frozen checkpoint reused read only (not recovered again)
Every allocation is realized with the same rule: unit k keeps max(1, round((1 - r_k) C_k)) channels (largest L1).
No accuracy beyond the E-epoch proxy cache, and no energy measurement, is read here.
"""
import collections
import datetime
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import p7b_lib as L  # noqa: E402
import ppo as P  # noqa: E402

UNIFORM_GRID = [i / 1000 for i in range(0, 901)]
LAMP_GRID = [i / 10 for i in range(0, 901)]                      # global sparsity in percent
UNIT_TENSORS = ["features.0", "features.3", "features.6", "classifier.1"]


def describe(ratios):
    macs, params, w = L.counts_ratios(ratios)
    m0, p0, _ = L.counts((0, 0, 0, 0))
    return {"ratios": [round(float(r), 6) for r in ratios], "widths": w, "macs": macs, "params": params,
            "memory_bytes": params * L.BYTES_PER_PARAM, "mac_reduction": 1 - macs / m0, "param_reduction": 1 - params / p0}


def policy_entry(p, inp, extra=None):
    t = inp.t[p]
    return {**describe([L.RATIO[a] for a in p]), "actions": list(p), "id": L.key(p), "q_proxy": inp.q[p],
            "proxy_feasible": inp.q[p] >= inp.tau, "e_pred_j": inp.e[p], "saving_pred": 1 - inp.e[p] / inp.e_dense,
            "latency_table_ms": t["latency_ms"], **(extra or {})}


def uniform_match(macs_target):
    best = None
    for r in UNIFORM_GRID:
        gap = abs(L.counts_ratios([r] * 4)[0] - macs_target)
        if best is None or gap < best[0]:
            best = (gap, r)
    return best


def lamp_match(params_target):
    import torch
    from src import pruning as PR
    ref, sha = L.p3().reference(L.SETTING)
    before = {k: v.clone() for k, v in ref.state_dict().items()}
    best, sizes = None, None
    for s in LAMP_GRID:
        m = PR.lamp(ref, L.ARCH, s)
        named = dict(m.named_modules())
        zeros = [int((named[t].weight_mask == 0).sum()) for t in UNIT_TENSORS]
        sizes = [int(named[t].weight_mask.numel()) for t in UNIT_TENSORS]
        ratios = [z / n for z, n in zip(zeros, sizes)]
        gap = abs(L.counts_ratios(ratios)[1] - params_target)
        if best is None or gap < best[0]:
            best = (gap, s, ratios, zeros)
    assert all(torch.equal(before[k], v) for k, v in ref.state_dict().items()), "dense reference modified"
    return best, sizes, sha


def main():
    pre = json.load(open(L.FINAL_PREREG, encoding="utf-8"))
    conf = json.load(open(os.path.join(L.RESULTS, "ppo_confirmatory_runs.json"), encoding="utf-8"))
    assert conf["preregistration_sha256"] == L.sha256_file(L.FINAL_PREREG)
    assert [r["seed"] for r in conf["runs"]] == pre["seeds"]["confirmatory"]
    inp = P.Inputs.load()

    sel = {r["seed"]: (tuple(r["selected"]["actions"]) if r["selected"] else None) for r in conf["runs"]}
    counts = collections.Counter(p for p in sel.values() if p is not None)
    assert counts, "no confirmatory seed has a selected policy"
    top = max(counts.values())
    primary = min((p for p, c in counts.items() if c == top), key=lambda p: (inp.e[p], p))
    feasible = [{"actions": list(p), "q": inp.q[p], "e_pred": inp.e[p], "macs": inp.t[p]["macs"],
                 "memory_bytes": inp.t[p]["memory_bytes"]} for p in L.POLICIES if inp.q[p] >= inp.tau]
    enum = tuple(P.select(feasible, inp)["actions"])

    prim = policy_entry(primary, inp)
    gap_u, r_u = uniform_match(prim["macs"])
    (gap_l, s_l, r_l, zeros_l), sizes, ref_sha = lamp_match(prim["params"])
    old = json.load(open(os.path.join(L.ROOT, "results", "phase7a_sustainability", "selected_policies.json"),
                         encoding="utf-8"))["settings"][L.SETTING]["methods"]["ppo"]
    old_ck = os.path.join(L.ROOT, "checkpoints", "phase7a", f"{L.SETTING}_ppo.pth")

    out = {"created": datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
           "preregistration_sha256": L.sha256_file(L.FINAL_PREREG), "reference_sha256": ref_sha,
           "realization_rule": "unit k keeps max(1, round((1 - r_k) C_k)) channels with the largest L1 norm "
                               "(src/pruning/structured.simplecnn_structured, the Paper-3 backend)",
           "per_seed": {str(s): (None if p is None else policy_entry(p, inp)) for s, p in sel.items()},
           "policy_counts": {json.dumps(list(p)): c for p, c in counts.most_common()},
           "failed_seeds": [s for s, p in sel.items() if p is None],
           "methods": {
               "dense": {**policy_entry((0, 0, 0, 0), inp), "origin": "identity policy"},
               "new_ppo": {**prim, "frequency": f"{top}/20",
                           "seeds": [s for s, p in sel.items() if p == primary],
                           "origin": "most frequent confirmatory selected policy (ties: lower predicted energy)"},
               "uniform": {**describe([r_u] * 4), "id": f"uniform_r{r_u:.3f}", "common_ratio": r_u, "mac_gap": gap_u,
                           "mac_gap_relative": gap_u / prim["macs"],
                           "origin": "common ratio, 0.001 grid, minimizing |MACs - MACs_new_ppo| (ties: smaller ratio)"},
               "lamp": {**describe(r_l), "id": f"lamp_s{s_l:.1f}", "global_sparsity_pct": s_l, "unit_zero_counts": zeros_l,
                        "unit_weight_counts": sizes, "param_gap": gap_l, "param_gap_relative": gap_l / prim["params"],
                        "origin": "LAMP on the dense reference at the 0.1%-grid global sparsity whose structured "
                                  "realization has the parameter count closest to new_ppo's (ties: smaller sparsity)"},
               "old_ppo": {**describe(old["structured_ratios"]), "actions": old["actions"], "id": "phase7a_ppo",
                           "checkpoint": os.path.relpath(old_ck, L.ROOT), "checkpoint_sha256": L.sha256_file(old_ck),
                           "origin": "Phase-7A PPO-Structured model, frozen recovered checkpoint reused read only"}},
           "enumeration_reference": policy_entry(enum, inp, {"origin": "selection rule over all proxy-feasible policies"})}

    jobs = collections.OrderedDict()

    def add(job_id, ratios, policy, role):
        j = jobs.setdefault(job_id, {"id": job_id, "ratios": [float(r) for r in ratios],
                                     "policy": None if policy is None else list(policy), "roles": []})
        j["roles"].append(role)

    add("0000", [0.0] * 4, (0, 0, 0, 0), "dense")
    add(L.key(primary), [L.RATIO[a] for a in primary], primary, "new_ppo")
    add(out["methods"]["uniform"]["id"], [r_u] * 4, None, "uniform")
    add(out["methods"]["lamp"]["id"], r_l, None, "lamp")
    for s, p in sel.items():
        if p is not None:
            add(L.key(p), [L.RATIO[a] for a in p], p, f"seed{s}")
    add(L.key(enum), [L.RATIO[a] for a in enum], enum, "enumeration_reference")
    json.dump(list(jobs.values()), open(os.path.join(L.RESULTS, "final_jobs.json"), "w", encoding="utf-8"), indent=1)
    out["final_jobs"] = len(jobs)
    json.dump(out, open(os.path.join(L.RESULTS, "selected_policies.json"), "w", encoding="utf-8"), indent=1)
    print(f"primary {list(primary)} ({top}/20); enumeration {list(enum)}; uniform r={r_u:.3f} (MAC gap {gap_u}); "
          f"LAMP s={s_l:.1f}% ratios {[round(x, 3) for x in r_l]} (param gap {gap_l}); {len(jobs)} recovery jobs")
    for j in jobs.values():
        print(" ", j["id"], j["roles"])


if __name__ == "__main__":
    main()
