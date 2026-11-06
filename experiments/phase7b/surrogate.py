"""Phase 7B, Stage A: energy surrogate selection and validation (design section 7), applied exactly as written.

  python experiments/phase7b/surrogate.py   -> energy_surrogate_metrics.json, energy_surrogate_predictions.csv
Candidates in fixed order; the first that passes 5-fold cross-validation (seed 20261029) is accepted:
  1  E ~ a + b MACs     2  E ~ a + b L     3  E ~ a + b MACs + c params + d L
  4  ridge (lambda 1e-3) on a degree-2 polynomial of the normalized widths + MACs + params + L
L = measured Paper-3 batch-1 latency table. Criteria on out-of-fold predictions: Spearman >= 0.95, MdAPE <= 5%,
MdAPE <= 5% in the lowest-energy quartile.
"""
import csv
import datetime
import itertools
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import p7b_lib as L  # noqa: E402

import numpy as np  # noqa: E402

CV_SEED, FOLDS, RHO_MIN, MDAPE_MAX, RIDGE = 20261029, 5, 0.95, 0.05, 1e-3
BASE = np.array([32, 64, 128, 256], float)


def features(kind, w, macs, params, lat):
    w = np.asarray(w, float) / BASE
    m, p = macs / 1e7, params / 1e6
    if kind == 1:
        return [1.0, m]
    if kind == 2:
        return [1.0, lat]
    if kind == 3:
        return [1.0, m, p, lat]
    base = list(w) + [m, p, lat]
    quad = [a * b for a, b in itertools.combinations_with_replacement(base, 2)]
    return [1.0] + base + quad


def fit(X, y, kind):
    if kind < 4:
        return np.linalg.lstsq(X, y, rcond=None)[0]
    reg = RIDGE * np.eye(X.shape[1])
    reg[0, 0] = 0.0
    return np.linalg.solve(X.T @ X + reg, X.T @ y)


def spearman(a, b):
    import pandas as pd
    return float(pd.Series(a).rank().corr(pd.Series(b).rank()))


def main():
    rows = list(csv.DictReader(open(os.path.join(L.RESULTS, "energy_surrogate_dataset.csv"), encoding="utf-8")))
    y = np.array([float(r["e1000_j"]) for r in rows])
    raw = [([int(r[f"w{i}"]) for i in range(1, 5)], float(r["macs"]), float(r["params"]), float(r["latency_table_ms"]))
           for r in rows]
    folds = np.random.default_rng(CV_SEED).permutation(len(rows)) % FOLDS
    low = y <= np.percentile(y, 25)
    out = {"created": datetime.datetime.now().astimezone().isoformat(timespec="seconds"), "n": len(rows),
           "criteria": {"spearman_min": RHO_MIN, "mdape_max": MDAPE_MAX, "low_quartile_mdape_max": MDAPE_MAX},
           "candidates": {}}
    accepted = None
    for kind in (1, 2, 3, 4):
        X = np.array([features(kind, *r) for r in raw])
        pred = np.empty(len(y))
        for f in range(FOLDS):
            tr, te = folds != f, folds == f
            pred[te] = X[te] @ fit(X[tr], y[tr], kind)
        ape = np.abs(pred - y) / y
        res = {"spearman": spearman(pred, y), "mdape": float(np.median(ape)), "low_quartile_mdape": float(np.median(ape[low])),
               "max_ape": float(ape.max()), "p90_ape": float(np.percentile(ape, 90))}
        res["passed"] = res["spearman"] >= RHO_MIN and res["mdape"] <= MDAPE_MAX and res["low_quartile_mdape"] <= MDAPE_MAX
        out["candidates"][kind] = res
        print(kind, res)
        if res["passed"] and accepted is None:
            accepted = kind
    out["accepted_model"] = accepted
    out["decision"] = f"candidate {accepted} accepted" if accepted else "STOP: no surrogate passes"
    if accepted:
        X = np.array([features(accepted, *r) for r in raw])
        coef = fit(X, y, accepted)
        out["coefficients"] = coef.tolist()
        t = L.table()
        preds = []
        for p in L.POLICIES:
            v = t[p]
            e = float(np.array(features(accepted, v["widths"], v["macs"], v["params"], v["latency_ms"])) @ coef)
            preds.append({"policy": json.dumps(list(p)), "e1000_pred_j": e, "macs": v["macs"], "params": v["params"],
                          "memory_bytes": v["memory_bytes"], "latency_table_ms": v["latency_ms"]})
        dense = next(r["e1000_pred_j"] for r in preds if r["policy"] == "[0, 0, 0, 0]")
        for r in preds:
            r["saving_pred"] = 1 - r["e1000_pred_j"] / dense
        with open(os.path.join(L.RESULTS, "energy_surrogate_predictions.csv"), "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(preds[0]))
            w.writeheader()
            w.writerows(preds)
        out["dense_pred_e1000_j"] = dense
        out["max_pred_saving"] = max(r["saving_pred"] for r in preds)
    json.dump(out, open(os.path.join(L.RESULTS, "energy_surrogate_metrics.json"), "w", encoding="utf-8"), indent=1)
    print(out["decision"])


if __name__ == "__main__":
    main()
