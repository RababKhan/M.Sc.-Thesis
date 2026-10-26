"""Phase 7B, Stage A: the fixed proxy-validation sample and the energy-calibration set (design sections 4 and 7).

  python experiments/phase7b/samples.py   -> results/phase7b_green_pruning/stage_a_samples.json
Only structural quantities (MACs, widths) are used; no accuracy, latency or energy value is read.
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import p7b_lib as L  # noqa: E402

PROXY_SEED, CALIB_SEED = 20261027, 20261028
PROXY_BINS = [(0.00, 0.05, 10), (0.05, 0.15, 10), (0.15, 0.25, 8), (0.25, 0.40, 8), (0.40, 0.60, 6), (0.60, 1.01, 6)]
CALIB_BINS = [(0.00, 0.05), (0.05, 0.10), (0.10, 0.20), (0.20, 0.30), (0.30, 0.40), (0.40, 0.50), (0.50, 0.65), (0.65, 1.01)]
NAMED = {"dense": (0, 0, 0, 0), "light": (1, 1, 1, 1), "moderate": (3, 3, 3, 3), "aggressive": (5, 5, 5, 5),
         "fc_heavy": (0, 0, 0, 5), "conv_heavy": (0, 5, 5, 0)}


def main():
    import numpy as np
    t = {p: L.counts(p) for p in L.POLICIES}
    m0 = t[(0, 0, 0, 0)][0]
    red = {p: 1 - v[0] / m0 for p, v in t.items()}

    rng = np.random.default_rng(PROXY_SEED)
    proxy = [(0, 0, 0, 0)]
    carry = 0                                                   # a bin with too few policies passes its shortfall on
    for lo, hi, n in PROXY_BINS:
        pool = sorted(p for p in L.POLICIES if lo <= red[p] < hi and p not in proxy)
        k = min(n + carry, len(pool))
        carry = n + carry - k
        proxy += [pool[i] for i in sorted(rng.choice(len(pool), size=k, replace=False))]
    assert len(set(proxy)) == 49

    rng = np.random.default_rng(CALIB_SEED)
    calib = list(dict.fromkeys(NAMED.values()))                 # dense + 5 distinct named policies
    for lo, hi in CALIB_BINS:
        pool = sorted(p for p in L.POLICIES if lo <= red[p] < hi and p not in calib)
        calib += [pool[i] for i in sorted(rng.choice(len(pool), size=min(20, len(pool)), replace=False))]
    base = np.array(L.counts((0, 0, 0, 0))[2], float)
    wnorm = {p: np.array(t[p][2]) / base for p in L.POLICIES}
    pairs = []
    order = [L.POLICIES[i] for i in rng.permutation(len(L.POLICIES))]
    for a in order:
        if len(calib) >= 200:
            break
        if a in calib:
            continue
        cand = [p for p in L.POLICIES if p != a and p not in calib and abs(t[p][0] - t[a][0]) <= 0.02 * t[a][0]]
        if not cand:
            continue
        b = max(cand, key=lambda p: (float(np.abs(wnorm[p] - wnorm[a]).sum()), [-x for x in p]))
        if len(calib) + 2 > 200:
            break
        calib += [a, b]
        pairs.append([list(a), list(b)])
    while len(calib) < 200:                                     # odd remainder: one more stratified random policy
        pool = sorted(p for p in L.POLICIES if p not in calib)
        calib.append(pool[int(rng.integers(len(pool)))])
    assert len(set(calib)) == 200
    out = {"proxy_validation": {"seed": PROXY_SEED, "bins": PROXY_BINS, "policies": [list(p) for p in proxy],
                                "mac_reduction": [red[p] for p in proxy]},
           "energy_calibration": {"seed": CALIB_SEED, "bins": CALIB_BINS, "named": {k: list(v) for k, v in NAMED.items()},
                                  "similar_mac_pairs": pairs, "policies": [list(p) for p in calib]}}
    json.dump(out, open(os.path.join(L.RESULTS, "stage_a_samples.json"), "w", encoding="utf-8"), indent=1)
    print("proxy sample", len(proxy), "| calibration", len(calib), "with", len(pairs), "similar-MAC pairs")
    for lo, hi in CALIB_BINS:
        print(f"  calib bin [{lo:.2f},{hi:.2f}): {sum(lo <= red[tuple(p)] < hi for p in calib)}")


if __name__ == "__main__":
    main()
