"""Phase 7B: structured sensitivity of the dense reference (design section 9).

  python experiments/phase7b/sensitivity.py   -> results/phase7b_green_pruning/structured_sensitivity.json
Paper-3 structured_loss_sensitivity (unit-tested in H1) on all 2,500 VAL-RL images with ratios {0.1, 0.2, 0.4, 0.6},
min-max normalised; centred S~ = S - mean(S). Uses no recovery, no VAL-SELECT and no test data.
"""
import datetime
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import p7b_lib as L  # noqa: E402


def main():
    import numpy as np
    import torch
    torch.set_num_threads(1)
    from src.pruning import structured_units as SU
    x, y, _ = L.p3().val_rl("cifar10")
    ref, sha = L.p3().reference(L.SETTING)
    before = {k: v.clone() for k, v in ref.state_dict().items()}
    s = SU.structured_loss_sensitivity(ref.eval(), L.ARCH, x, y)
    assert all(torch.equal(before[k], v) for k, v in ref.state_dict().items()), "dense model modified"
    norm = np.array(s["normalized"], float)
    out = {"created": datetime.datetime.now().astimezone().isoformat(timespec="seconds"), "setting": L.SETTING,
           "reference_sha256": sha, "split": "VAL-RL (2,500 images)", "ratios": s["ratios"], "units": s["units"],
           "raw": s["raw"], "normalized": norm.tolist(), "centred": (norm - norm.mean()).tolist(),
           "baseline_loss": s["baseline_loss"], "baseline_accuracy": s["baseline_accuracy"], "detail": s["detail"]}
    json.dump(out, open(os.path.join(L.RESULTS, "structured_sensitivity.json"), "w", encoding="utf-8"), indent=1)
    print("normalized", [round(v, 3) for v in out["normalized"]], "centred", [round(v, 3) for v in out["centred"]])


if __name__ == "__main__":
    main()
