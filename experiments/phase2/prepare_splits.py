"""CIFAR-100 split (and a CIFAR-10 manifest of the existing split). Writes once; refuses to overwrite.

CIFAR-100
  train / validation   torch.randperm(50,000, seed 42): last 5,000 validation (the CIFAR-10 rule;
                       with the same n and seed the index lists coincide with CIFAR-10's)
  V_RL / V_SELECT      3,000 / 2,000 of validation, stratified by the 100 fine labels,
                       numpy default_rng(20260928), the archive algorithm
  normalisation        per-channel mean and pixel std of the 50,000 training images, checked
                       against the constants in src/data (4 decimals)
Only training-set files are read; the test set is not opened (its archive md5 is verified).

Outputs
  results/reproducibility/cifar100_split_indices.npz
  results/reproducibility/cifar100_split_manifest.json
  results/reproducibility/cifar10_split_manifest.json
"""
import hashlib
import json
import os

import numpy as np

import phase2_common as C
from src import data as D
from src.utils import sha256_array, sha256_file, sha256_json

OUT = os.path.join(C.RESULTS, "reproducibility")
TGZ_MD5 = {"cifar10": ("cifar-10-python.tar.gz", "c58f30108f718f92721af3b95e74349a"),
           "cifar100": ("cifar-100-python.tar.gz", "eb9058c3a382ffc7106e4002c42a8d85")}


def md5(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def class_summary(labels, n_classes):
    c = np.bincount(labels, minlength=n_classes)
    return {"min": int(c.min()), "max": int(c.max()), "mean": float(c.mean()), "per_class": c.tolist()}


def main():
    os.makedirs(OUT, exist_ok=True)
    npz_path = D.SPLIT_FILES["cifar100"]
    if os.path.exists(npz_path):
        raise SystemExit(f"{npz_path} exists; the split is fixed")
    name, want = TGZ_MD5["cifar100"]
    got = md5(os.path.join(D.DATA_ROOT, name))
    assert got == want, f"CIFAR-100 archive md5 {got} != {want}"

    images, labels = D.load_arrays("cifar100", True)
    assert images.shape == (50000, 32, 32, 3) and labels.max() == 99
    mean, std = np.zeros(3), np.zeros(3)
    for ch in range(3):                                    # per channel keeps memory at ~0.4 GB
        v = images[..., ch].astype(np.float64) / 255.0
        mean[ch], std[ch] = v.mean(), v.std()
    assert np.allclose(np.round(mean, 4), D.DATASETS["cifar100"]["mean"]) and \
        np.allclose(np.round(std, 4), D.DATASETS["cifar100"]["std"]), f"normalisation {mean} {std}"

    tr, va = D.train_val_indices(len(labels))
    assert len(tr) == 45000 and len(va) == 5000 and not set(tr) & set(va)
    tr2, va2 = D.train_val_indices(len(labels))
    assert tr == tr2 and va == va2, "train/val split not deterministic"
    val_labels = labels[np.array(va)]
    rl, sel, counts, n_rl = D.stratified_split(val_labels)
    rl2, sel2, _, _ = D.stratified_split(val_labels)
    assert np.array_equal(rl, rl2) and np.array_equal(sel, sel2), "stratified split not deterministic"
    assert len(rl) == D.N_RL and len(sel) == D.N_SELECT and not set(rl) & set(sel)
    assert sorted(np.concatenate([rl, sel]).tolist()) == list(range(5000))
    va_arr = np.array(va)
    np.savez(npz_path, train_indices=np.array(tr), val_indices=va_arr,
             v_rl_positions=rl, v_select_positions=sel,
             v_rl_cifar_train_indices=va_arr[rl], v_select_cifar_train_indices=va_arr[sel],
             val_labels=val_labels, v_rl_labels=val_labels[rl], v_select_labels=val_labels[sel],
             split_seed=D.SPLIT_SEED, stratified_seed=D.STRATIFIED_SEED)
    manifest = {
        "dataset": "CIFAR-100 (fine labels)", "archive": name, "archive_md5": got,
        "file": os.path.relpath(npz_path, C.ROOT).replace("\\", "/"), "file_sha256": sha256_file(npz_path),
        "train_size": len(tr), "val_size": len(va), "V_RL_size": int(len(rl)), "V_SELECT_size": int(len(sel)),
        "test_size": 10000, "split_rule": "torch.randperm(50000, generator seed 42); last 5,000 = validation",
        "stratified_rule": "largest-remainder per-class allocation of 3,000; per-class permutation; "
                           "numpy default_rng(20260928)",
        "train_indices_sha256": sha256_json(tr), "val_indices_sha256": sha256_json(va),
        "v_rl_positions_sha256": sha256_array(rl), "v_select_positions_sha256": sha256_array(sel),
        "identical_index_lists_to_cifar10": True,
        "class_counts": {"train": class_summary(labels[np.array(tr)], 100),
                         "validation": class_summary(val_labels, 100),
                         "V_RL": class_summary(val_labels[rl], 100),
                         "V_SELECT": class_summary(val_labels[sel], 100)},
        "normalisation_computed": {"mean": mean.tolist(), "std": std.tolist(),
                                   "used": {k: D.DATASETS["cifar100"][k] for k in ("mean", "std")}},
    }
    c10_tr, c10_va = D.train_val_indices(50000)
    manifest["identical_index_lists_to_cifar10"] = (c10_tr == tr and c10_va == va)
    with open(os.path.join(OUT, "cifar100_split_manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=1)

    # CIFAR-10: manifest of the existing (historical) split; nothing is regenerated.
    c10 = np.load(D.SPLIT_FILES["cifar10"])
    l10 = D.load_arrays("cifar10", True)[1]
    v10 = l10[np.array(c10_va)]
    m10 = {"dataset": "CIFAR-10", "archive_md5": md5(os.path.join(D.DATA_ROOT, TGZ_MD5["cifar10"][0])),
           "train_val_rule": "rl_env.Splits: torch.randperm(50000, seed 42); last 5,000 = validation",
           "train_indices_sha256": sha256_json(c10_tr), "val_indices_sha256": sha256_json(c10_va),
           "v_rl_v_select_file": os.path.relpath(D.SPLIT_FILES["cifar10"], C.ROOT).replace("\\", "/"),
           "v_rl_v_select_file_sha256": sha256_file(D.SPLIT_FILES["cifar10"]),
           "class_counts": {"validation": class_summary(v10, 10),
                            "V_RL": class_summary(v10[c10["v_rl_positions"]], 10),
                            "V_SELECT": class_summary(v10[c10["v_select_positions"]], 10)}}
    with open(os.path.join(OUT, "cifar10_split_manifest.json"), "w", encoding="utf-8") as f:
        json.dump(m10, f, indent=1)
    print(json.dumps({k: v for k, v in manifest.items() if k != "class_counts"}, indent=1))
    print("class counts (min/max):", {k: (v["min"], v["max"]) for k, v in manifest["class_counts"].items()})


if __name__ == "__main__":
    main()
