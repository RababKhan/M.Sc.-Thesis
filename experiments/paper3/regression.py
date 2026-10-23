"""Paper 3 regression: rerun the existing Phase-2 implementation checks unchanged, without touching frozen files.

  python experiments/paper3/regression.py   -> results/paper3/regression/cpu_phase2_checks.{md,json}

experiments/phase2/phase2_checks.py is loaded by file path and run as is. Its only two writes
(results/architecture_generalization/cpu_phase2_checks.md / .json, frozen Phase-2 evidence) are redirected
into results/paper3/regression/ by replacing `open` in the module's namespace; every read is untouched.
Afterwards the frozen Phase-2 files are verified unchanged (git diff empty).
"""
import builtins
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import p3_lib as L  # noqa: E402

PHASE2 = os.path.join(L.ROOT, "experiments", "phase2")
OUT = os.path.join(L.RESULTS, "regression")


def main():
    os.makedirs(OUT, exist_ok=True)
    if PHASE2 not in sys.path:
        sys.path.append(PHASE2)
    mod = L.phase_module("phase2_checks", PHASE2)
    frozen = os.path.join(mod.C.AG, "")

    def redirecting_open(file, mode="r", *a, **k):
        path = os.path.abspath(str(file))
        if any(c in mode for c in "wax") and path.startswith(os.path.abspath(frozen)):
            path = os.path.join(OUT, os.path.basename(path))
            print(f"(write redirected to {os.path.relpath(path, L.ROOT)})")
        return builtins.open(path, mode, *a, **k)
    mod.open = redirecting_open
    mod.main()
    diff = subprocess.run(["git", "status", "--porcelain", "--", "results/architecture_generalization", "results/reproducibility"],
                          cwd=L.ROOT, capture_output=True, text=True).stdout.strip()
    print("frozen Phase-2 files unchanged:", diff == "", diff)
    if diff:
        raise SystemExit("a frozen file changed")


if __name__ == "__main__":
    main()
