"""Phase 7B, Stage A: assemble the E-epoch proxy landscape (design section 8) from the 1,296 recovery records.

  python experiments/phase7b/landscape.py   -> results/phase7b_green_pruning/proxy_landscape.csv
q = A_E / A_E,dense on VAL-RL; feasible = q >= tau_E. Only possible after the proxy was accepted.
"""
import csv
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import p7b_lib as L  # noqa: E402


def main():
    prox = json.load(open(os.path.join(L.RESULTS, "accuracy_proxy_metrics.json"), encoding="utf-8"))
    e = prox["accepted_epochs"]
    assert e in (2, 3), "no accepted proxy"
    tau = prox["proxies"][str(e)]["tau"]
    rec = {}
    for p in L.POLICIES:
        r = json.load(open(os.path.join(L.RESULTS, "_recovery_runs", "landscape", f"{L.key(p)}_E{e}.json"), encoding="utf-8"))
        assert r["status"] == "complete" and r["epochs"] == e and r["test_use"] == "none"
        rec[p] = r
    dense = rec[(0, 0, 0, 0)]["post"]["val_rl"]
    streams = {r["data_stream_sha256"] for r in rec.values()}
    assert len(streams) == 1, "data stream differs between policies"
    t = L.table()
    rows = [{"policy": json.dumps(list(p)), "A_E_val_rl": r["post"]["val_rl"], "q": r["post"]["val_rl"] / dense,
             "feasible": r["post"]["val_rl"] / dense >= tau, "macs": t[p]["macs"], "params": t[p]["params"],
             "memory_bytes": t[p]["memory_bytes"], "latency_table_ms": t[p]["latency_ms"], "seconds": r["seconds"]}
            for p, r in rec.items()]
    with open(os.path.join(L.RESULTS, "proxy_landscape.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"E = {e}, tau = {tau:.4f}, dense A_E = {dense:.2f}, feasible {sum(r['feasible'] for r in rows)}/1296")


if __name__ == "__main__":
    main()
