"""Phase 5 validation split: the frozen 5,000-image validation set of each dataset -> VAL-RL 2,500 / VAL-SELECT 2,500.

Deterministic and stratified by class: src.data.stratified_split (largest-remainder per-class allocation, then
a per-class permutation) with numpy default_rng(20260930) and n_first = 2,500. Training set (45,000), official
test set (10,000) and dense checkpoints are unchanged. Writes results/phase5/splits/<dataset>_phase5_split.npz and
a manifest with hashes and class counts; refuses to overwrite.
"""
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
from src import data as D  # noqa: E402
from src.utils import sha256_array, sha256_file, sha256_json  # noqa: E402

OUT = os.path.join(ROOT, "results", "phase5", "splits")
SEED = 20260930
N_RL = 2500


def main():
    os.makedirs(OUT, exist_ok=True)
    manifest = {"rule": "src.data.stratified_split(val_labels, n_first=2500, seed=20260930) on the frozen 5,000-image "
                        "validation split (positions in its fixed order); first = VAL-RL, second = VAL-SELECT",
                "seed": SEED, "datasets": {}}
    for d in ("cifar10", "cifar100"):
        path = os.path.join(OUT, f"{d}_phase5_split.npz")
        if os.path.exists(path):
            raise SystemExit(f"{path} exists; the Phase-5 split is frozen")
        b = D.DataBundle(d, train_images=False, rl_split=False)
        labels = b.val_y.numpy()
        rl, sel, counts, n_c = D.stratified_split(labels, n_first=N_RL, seed=SEED)
        rl2, sel2, _, _ = D.stratified_split(labels, n_first=N_RL, seed=SEED)
        assert np.array_equal(rl, rl2) and np.array_equal(sel, sel2), "not deterministic"
        assert len(rl) == len(sel) == N_RL and not set(rl.tolist()) & set(sel.tolist())
        assert sorted(np.concatenate([rl, sel]).tolist()) == list(range(5000))
        val_idx = np.array(b.val_indices)
        np.savez(path, val_rl_positions=rl, val_select_positions=sel, val_rl_cifar_train_indices=val_idx[rl],
                 val_select_cifar_train_indices=val_idx[sel], val_rl_labels=labels[rl], val_select_labels=labels[sel],
                 seed=SEED)
        c_rl = np.bincount(labels[rl], minlength=len(counts))
        c_sel = np.bincount(labels[sel], minlength=len(counts))
        manifest["datasets"][d] = {
            "file": os.path.relpath(path, ROOT).replace("\\", "/"), "file_sha256": sha256_file(path),
            "val_indices_sha256": sha256_json(b.val_indices), "val_rl_positions_sha256": sha256_array(rl),
            "val_select_positions_sha256": sha256_array(sel), "sizes": [int(len(rl)), int(len(sel))],
            "class_counts_validation": {"min": int(counts.min()), "max": int(counts.max())},
            "class_counts_val_rl": {"min": int(c_rl.min()), "max": int(c_rl.max())},
            "class_counts_val_select": {"min": int(c_sel.min()), "max": int(c_sel.max())},
            "max_abs_class_imbalance_rl_vs_select": int(np.abs(c_rl - c_sel).max()),
            "overlap_with_old_V_RL": None}
        old = np.load(D.SPLIT_FILES[d])
        manifest["datasets"][d]["overlap_with_old_V_RL"] = {
            "val_rl": int(len(set(rl.tolist()) & set(old["v_rl_positions"].tolist()))),
            "val_select": int(len(set(sel.tolist()) & set(old["v_rl_positions"].tolist())))}
        print(d, json.dumps(manifest["datasets"][d]))
    with open(os.path.join(OUT, "phase5_split_manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=1)


if __name__ == "__main__":
    main()
