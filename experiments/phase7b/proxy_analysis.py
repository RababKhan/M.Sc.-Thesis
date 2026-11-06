"""Phase 7B, Stage A: accuracy-proxy acceptance (design section 4), applied exactly as written.

  python experiments/phase7b/proxy_analysis.py   -> accuracy_proxy_validation.csv, accuracy_proxy_metrics.json
"""
import csv
import datetime
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import p7b_lib as L  # noqa: E402

RHO_MIN, PRECISION_MIN, MIN_FEASIBLE, MARGIN = 0.90, 0.50, 5, 0.005


def spearman(a, b):
    import pandas as pd
    return float(pd.Series(a).rank().corr(pd.Series(b).rank()))


def main():
    samp = json.load(open(os.path.join(L.RESULTS, "stage_a_samples.json"), encoding="utf-8"))["proxy_validation"]["policies"]
    rec = {}
    for p in samp:
        for e in (2, 3, 5):
            r = json.load(open(os.path.join(L.RESULTS, "_recovery_runs", "proxy_validation", f"{L.key(p)}_E{e}.json"),
                               encoding="utf-8"))
            assert r["status"] == "complete" and r["epochs"] == e and r["test_use"] == "none"
            rec[(tuple(p), e)] = r
    dense = {e: rec[((0, 0, 0, 0), e)]["post"]["val_rl"] for e in (2, 3, 5)}
    streams = {e: {rec[(tuple(p), e)]["data_stream_sha256"] for p in samp} for e in (2, 3, 5)}
    rows = []
    for p in samp:
        p = tuple(p)
        a = {e: rec[(p, e)]["post"]["val_rl"] for e in (2, 3, 5)}
        rows.append({"policy": json.dumps(list(p)), "mac_reduction": 1 - rec[(p, 5)]["macs"] / rec[((0, 0, 0, 0), 5)]["macs"],
                     "A2": a[2], "A3": a[3], "A5": a[5], "q2": a[2] / dense[2], "q3": a[3] / dense[3], "q5": a[5] / dense[5],
                     "A5_minus_dense_pp": a[5] - dense[5], "feasible5": a[5] >= dense[5] - L.ACC_TOL_PP - 1e-9,
                     "A5_val_select": rec[(p, 5)]["post"]["val_select"]})
    feas = [r for r in rows if r["feasible5"]]
    metrics = {"created": datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
               "dense_val_rl": dense, "n_policies": len(rows), "n_feasible5": len(feas),
               "same_data_stream_within_budget": {e: len(v) == 1 for e, v in streams.items()},
               "rule": {"rho_min": RHO_MIN, "recall": 1.0, "precision_min": PRECISION_MIN, "min_feasible": MIN_FEASIBLE,
                        "tau": "min q_E over 5-epoch-feasible sample policies - 0.005"}, "proxies": {}}
    accepted = None
    for e in (2, 3):
        rho = spearman([r[f"A{e}"] for r in rows], [r["A5"] for r in rows])
        if not feas:
            metrics["proxies"][e] = {"rho": rho, "accepted": False, "reason": "no 5-epoch-feasible sample policy"}
            continue
        tau = min(r[f"q{e}"] for r in feas) - MARGIN
        sel = [r for r in rows if r[f"q{e}"] >= tau]
        tp = sum(r["feasible5"] for r in sel)
        rec_ = tp / len(feas)
        prec = tp / len(sel)
        ok = rho >= RHO_MIN and rec_ == 1.0 and prec >= PRECISION_MIN and len(feas) >= MIN_FEASIBLE
        metrics["proxies"][e] = {"rho": rho, "tau": tau, "recall": rec_, "precision": prec, "selected": len(sel),
                                 "true_positive": tp, "accepted": ok}
        if ok and accepted is None:
            accepted = e
    metrics["accepted_epochs"] = accepted
    metrics["decision"] = f"{accepted}-epoch proxy accepted" if accepted else "STOP: no proxy passes"
    with open(os.path.join(L.RESULTS, "accuracy_proxy_validation.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    json.dump(metrics, open(os.path.join(L.RESULTS, "accuracy_proxy_metrics.json"), "w", encoding="utf-8"), indent=1)
    print(json.dumps(metrics, indent=1))


if __name__ == "__main__":
    main()
