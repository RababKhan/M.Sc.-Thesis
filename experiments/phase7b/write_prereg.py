"""Phase 7B: write the frozen Stage-B pre-registration (design section 10) after Stage A is complete.

  python experiments/phase7b/write_prereg.py   -> results/phase7b_green_pruning/phase7b_final_preregistration.json
Refuses to overwrite an existing file. Everything Stage B uses is recorded here, with SHA-256 of its inputs and code.
"""
import datetime
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import p7b_lib as L  # noqa: E402
import ppo as P  # noqa: E402
import final_models as F  # noqa: E402
import energy_final as EF  # noqa: E402

INPUTS = ["phase7b_design.md", "stage_a_samples.json", "accuracy_proxy_metrics.json", "energy_surrogate_metrics.json",
          "energy_surrogate_predictions.csv", "proxy_landscape.csv", "structured_sensitivity.json", "ppo_dev_tuning.json"]
CODE = ["p7b_lib.py", "recover.py", "ppo.py", "final_models.py", "energy_final.py", "final_analysis.py"]


def main():
    assert not os.path.exists(L.FINAL_PREREG), "the final pre-registration already exists and is frozen"
    from src import rl as RL
    prox = json.load(open(os.path.join(L.RESULTS, "accuracy_proxy_metrics.json"), encoding="utf-8"))
    sur = json.load(open(os.path.join(L.RESULTS, "energy_surrogate_metrics.json"), encoding="utf-8"))
    dev = json.load(open(os.path.join(L.RESULTS, "ppo_dev_tuning.json"), encoding="utf-8"))
    e = prox["accepted_epochs"]
    assert e in (2, 3) and sur["accepted_model"] in (1, 2, 3, 4)
    inp = P.Inputs.load()
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=L.ROOT, capture_output=True, text=True).stdout.strip()
    pre = {
        "title": "Phase 7B final (Stage-B) pre-registration: accuracy-preserving sustainable structured pruning with "
                 "sensitivity-aware PPO (SimpleCNN, CIFAR-10)",
        "created": datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
        "git_head_at_writing": commit,
        "frozen_rule": "Nothing below changes after this file is committed. Stage-B results are reported as they are.",
        "inputs_sha256": {f: L.sha256_file(os.path.join(L.RESULTS, f)) for f in INPUTS},
        "code_sha256": {f: L.sha256_file(os.path.join(HERE, f)) for f in CODE},
        "setting": {"architecture": L.ARCH, "dataset": "CIFAR-10", "reference": "Paper-3 frozen dense reference",
                    "test_set": "never read"},
        "structured_space": {"units": ["conv1", "conv2", "conv3", "fc1"], "ratios": L.RATIO,
                             "rule": "k = max(1, round((1 - r) C)) channels with the largest L1 norm (Paper-3 backend)"},
        "proxy": {"epochs": e, "tau": prox["proxies"][str(e)]["tau"], "split": "VAL-RL", "q": "A_E / A_E,dense",
                  "validation": prox["proxies"][str(e)], "landscape": "all 1,296 policies recovered once at E epochs"},
        "surrogate": {"accepted_model": sur["accepted_model"], "coefficients": sur["coefficients"],
                      "features": "E_1000 = a + b * L_table (measured Paper-3 batch-1 latency, ms)" if sur["accepted_model"] == 2
                      else "see energy_surrogate_metrics.json", "validation": sur["candidates"][str(sur["accepted_model"])],
                      "dense_pred_e1000_j": sur["dense_pred_e1000_j"], "s_max": inp.s_max},
        "state": ["u/(n-1)", "log1p(C_u)", "unit share of dense params", "unit share of dense MACs", "centred sensitivity",
                  "previous ratio", "mean ratio so far", "E_hat(partial policy)/E_hat_dense"],
        "prior": {"beta": P.BETA, "form": "logit(a|u) -= beta * S~_u * r(a)/0.6 (src.rl.SoftPriorPolicy)"},
        "reward": {"alpha": dev["chosen_alpha"], "kappa": P.KAPPA, "epsilon": P.EPS,
                   "feasible": "R = 1 + alpha * s_hat / s_max + epsilon * q", "infeasible": "R = -kappa * (tau - q)",
                   "alpha_tuning": {"rule": dev["rule"], "summary": dev["summary"], "dev_seeds": P.DEV_SEEDS}},
        "ppo": {"library": "stable-baselines3 PPO", "kwargs": {k: (v if isinstance(v, (int, float, str)) else str(v))
                                                               for k, v in RL.PPO_KWARGS.items()},
                "episodes": P.EPISODES, "steps": P.EPISODES * 4, "threads": 1},
        "seeds": {"development": P.DEV_SEEDS, "confirmatory": P.CONFIRM_SEEDS, "report": "all 20 seeds"},
        "archive_and_selection": {
            "archive": "every distinct proxy-feasible policy visited during training",
            "selection": f"lowest predicted energy; within {100 * P.TIE_BAND:.0f}% tied -> higher q, lower MACs, lower "
                         "parameter memory, smaller action vector",
            "empty_archive": "no selected policy; counted as a failure (GREEN-SUCCESS = NO)",
            "primary_new_ppo": "most frequent selected policy over the 20 confirmatory seeds; ties -> lower predicted "
                               "energy, then smaller action vector",
            "enumeration_reference": "the same selection rule over all proxy-feasible policies (reported, not a method)"},
        "baselines": {
            "dense": "identity policy",
            "uniform": f"common ratio on a 0.001 grid {F.UNIFORM_GRID[0]}-{F.UNIFORM_GRID[-1]} minimizing |MACs - MACs_new_ppo| "
                       "(ties -> smaller ratio)",
            "lamp": f"src.pruning.lamp on the dense reference at global sparsity on a 0.1% grid {F.LAMP_GRID[0]}-"
                    f"{F.LAMP_GRID[-1]}%; unit ratio = zeros/weights of the unit; the sparsity whose structured "
                    "realization has the parameter count closest to new_ppo's (ties -> smaller sparsity)",
            "old_ppo": "Phase-7A PPO-Structured [0,2,5,5], frozen checkpoint reused read only (Phase-7B dense recovery "
                       "verified bit-identical to Phase 7A's, same protocol and data stream)"},
        "recovery": {"protocol": "Phase-6 finetune: Adam lr 1e-4, cosine over 5 epochs, batch 128, crop+flip, seed 600",
                     "epochs": L.FINAL_EPOCHS, "applies_to": "every final model, identically"},
        "accuracy": {"criterion_split": "VAL-SELECT (held out from the search)",
                     "preserved": f"VAL-SELECT(model) >= VAL-SELECT(dense, 5-epoch recovered) - {L.ACC_TOL_PP} pp",
                     "also_reported": "VAL-RL"},
        "memory": {"primary": "parameter bytes = sum(numel * element_size) over parameters",
                   "secondary": "serialized checkpoint bytes"},
        "computation": "conv/linear MACs per image (closed form, checked against hooks)",
        "energy_protocol": {"backend": "RAPL package counter via Windows Energy Meter Interface (no TDP, no estimate)",
                            "pilot": {"n": 100000, "max_cv": EF.PILOT_MAX_CV, "unstable": "N doubles once; still "
                                      "unstable -> STOP"},
                            "sessions": EF.SESSIONS, "session_seed": EF.SESSION_SEED, "n": 100000, "warmup": EF.WARMUP,
                            "idle_s": EF.IDLE_S, "pause_s": EF.PAUSE_S, "dense_blocks_per_session": EF.DENSE_BLOCKS,
                            "order": "non-dense models once and Dense 3 times, one random order per session",
                            "conditions": "AC, Balanced plan, batch 1, 1 intra-op and 1 inter-op thread, CPU 2, fresh "
                                          f"process per session, system load <= {EF.MAX_CPU_LOAD}%",
                            "dense_per_session": "median of its 3 blocks"},
        "statistics": {"saving": "1 - E / E_dense within the session",
                       "noise_threshold": "2 x between-session CV of E_dense",
                       "energy_reduced": "lower than same-session dense in 5/5 sessions AND median saving > noise threshold",
                       "latency_reduced": "median-over-sessions latency below dense AND lower in 5/5 sessions",
                       "tests": "exact one-sided Wilcoxon vs dense (min p = 0.031); paired comparisons descriptive",
                       "edp": "normalized (E x latency) / (E_dense x latency_dense), per session, median"},
        "green_success": {"rule": "accuracy preserved (VAL-SELECT) AND memory reduction > 0 AND MAC reduction > 0 AND "
                                  "latency reduced AND energy reduced", "override": "never"},
        "cases": {"GREEN-A": "new_ppo GREEN-SUCCESS and neither Uniform nor LAMP has a better trade-off",
                  "GREEN-B": "new_ppo GREEN-SUCCESS but Uniform or LAMP has a better trade-off",
                  "GREEN-C": "memory, MACs and energy improve but accuracy fails",
                  "GREEN-D": "accuracy preserved but energy not reduced",
                  "otherwise": "reported as none of GREEN-A-D with the failed criteria"},
        "comparison_rule": {"baseline_better": "comparable VAL-SELECT accuracy (|d| <= 0.5 pp) and baseline E_1000 lower in "
                                               "5/5 sessions, or baseline accuracy higher by > 0.5 pp with median E_1000 "
                                               "not higher (Phase-7A rule)",
                            "new_ppo_better": "the same with the roles exchanged", "otherwise": "no clear difference"},
        "green_frequency": "GREEN-SUCCESS of each confirmatory seed's selected policy (5-epoch recovered and measured); "
                           "seeds without a selection count as NO",
        "pareto": ["accuracy-energy", "accuracy-MACs", "accuracy-memory", "energy-MACs"],
        "stop_conditions": ["integrity failure", "non-RAPL backend", "unstable energy pilot",
                            "a step would need a frozen rule changed", "noise comparable to model differences"],
        "next_architecture": "ResNet-8 only after the SimpleCNN results are frozen; no CIFAR-100",
    }
    json.dump(pre, open(L.FINAL_PREREG, "w", encoding="utf-8"), indent=1, default=float)
    print(f"written {L.FINAL_PREREG}: alpha {pre['reward']['alpha']}, tau {pre['proxy']['tau']:.5f}, E {e}")


if __name__ == "__main__":
    main()
