"""Phase 7A, step 3b: write the pre-registration (before any model is built, recovered or measured).

  python experiments/phase7a/write_prereg.py   -> results/phase7a_sustainability/preregistration.json
"""
import datetime
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import p7_lib as L  # noqa: E402


def main():
    sel_path = os.path.join(L.RESULTS, "selected_policies.json")
    sel = json.load(open(sel_path, encoding="utf-8"))
    pol = {s: {k: {f: v[f] for f in ("structured_ratios", "widths", "macs", "params", "mac_reduction", "param_reduction")}
               for k, v in d["methods"].items()} for s, d in sel["settings"].items()}
    pre = {
        "title": "Phase 7A - Sustainable Computing Evaluation of Adaptive Pruning",
        "created": datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
        "status": "written before any Phase-7A model was built, recovered or measured",
        "questions": {
            "primary": "Can the layer-wise adaptive pruning policy reduce measured CPU inference energy while preserving "
                       "predictive accuracy?",
            "secondary": "Does the adaptive policy provide a better energy-accuracy trade-off than uniform structured "
                         "pruning and a strong non-RL structured baseline (LAMP-derived)?"},
        "positioning": "Deployment realization: the thesis PPO policies were unstructured masks, which do not reduce dense "
                       "CPU work. Their per-unit pruning ratios are realized here as physical structured channel removal. "
                       "No claim is made that the masked thesis models themselves reduce latency or energy.",
        "scope": {"dataset": "CIFAR-10", "architectures": ["SimpleCNN", "ResNet-8"], "gpu": False,
                  "phase1_6_results_modified": False},
        "methods": {
            "dense": "frozen Phase-2 dense reference, recovered with the same protocol (identity policy)",
            "ppo": "most frequent Phase-5 C1 selected policy over policy seeds 500-519 (ties: smallest action vector)",
            "uniform": "one common ratio for all four units on a 0.001 grid minimizing |MACs - MACs_PPO| (MAC matching, "
                       "fixed before results; ties: smallest ratio). Exact matching is limited by integer channel counts; "
                       "the residual gap is recorded and reported",
            "lamp": "Phase-6 LAMP mask at exactly PPO's zero count; thesis-unit sparsity = zeros / weights, realized with "
                    "the same structured rule"},
        "selected_policies_file": {"path": "results/phase7a_sustainability/selected_policies.json",
                                   "sha256": L.sha256_file(sel_path)},
        "selected_policies": pol,
        "structured_rule": {
            "code": "src/pruning/structured_units.py and src/pruning/structured.py (validated in Paper 3, unchanged)",
            "keep": "max(1, round((1 - r) * C)) channels with the largest L1 norm, scored on the dense model",
            "simplecnn_units": ["conv1 (features.0 filters)", "conv2 (features.3)", "conv3 (features.6)",
                                "fc1 hidden units (classifier.1)"],
            "resnet8_units": ["stem: coupled stage-1 residual width", "stage1/2/3: block-internal channels"],
            "resnet8_mapping": sel["settings"]["cifar10_resnet8"]["mapping_note"],
            "integrity_checks_before_measurement": [
                "slimmed model contains no pruning masks or reparameterizations",
                "parameter counts equal the analytic values", "conv/linear MACs equal the analytic values",
                "slimmed vs masked-equivalent logits agree within 1e-4 (Paper-3 tolerance) on a fixed batch",
                "state dict saved, reloaded into a fresh skeleton, identical logits"]},
        "recovery": {"protocol": "frozen Phase-6 fine-tuning protocol as used in Paper 3 (Adam lr 1e-4, betas 0.9/0.999, "
                                 "eps 1e-8, wd 0, cosine to 0 per epoch, batch 128, random crop pad 4 + horizontal flip, "
                                 "BN train mode in training, eval mode for evaluation)",
                     "epochs": L.RECOVERY_EPOCHS, "seed": L.RECOVERY_SEED, "data_generator_seed": L.RECOVERY_SEED + 10 ** 6,
                     "same_for_all_methods": True, "threads": 1, "selection": "model after the final epoch; no selection"},
        "accuracy": {
            "primary_metric": "VAL-RL accuracy (2,500 images, Phase-5 split) after recovery",
            "secondary_metric": "VAL-SELECT accuracy (2,500 images)",
            "reference": "recovered dense model (identity policy, same protocol)",
            "preservation_rule": "|A_method - A_dense_recovered| <= 0.5 percentage points on VAL-RL; otherwise the model "
                                 "is classified as not accuracy-preserving (threshold fixed, never changed)",
            "retained_accuracy": "q = A_method / A_dense_recovered",
            "test_set": "not used (no frozen test protocol exists for these new structured models)",
            "known_bias": "the PPO policies were trained with VAL-RL rewards and selected on VAL-SELECT, so PPO's "
                          "validation accuracy may be optimistically biased relative to the baselines"},
        "energy": {
            "backend": "Intel RAPL package energy (\\Energy Meter(RAPL_Package0_PKG)\\Energy) via the Windows Energy Meter "
                       "Interface, read with PDH raw values; units picowatt-hours (1 pWh = 3.6e-9 J); ~1 ms refresh. "
                       "Secondary domain: RAPL_Package0_PP0 (cores). DRAM is not exposed on this machine. "
                       "No TDP or software estimate is used",
            "primary_metric": "E_1000 = package joules / N * 1000 for one contiguous block of N inferences",
            "secondary_metrics": ["joules per inference", "PP0 core energy", "net package energy above the session's "
                                  "idle package power (30 s idle measured at session start)"],
            "energy_saving": "100 * (1 - E_method / E_dense) with the dense anchor of the same session",
            "workload": {"warmup_inferences": 1000, "measured_inferences_N": 100000, "batch": 1,
                         "input": "one preallocated float32 tensor 1x3x32x32 (seeded)", "mode": "eval + inference_mode",
                         "excluded": "model construction, checkpoint loading and warm-up"},
            "sessions": 5, "session_design": "one fresh process per (architecture, session); the 4 models of the "
                        "architecture (dense anchor included) in a seeded random order per session",
            "conditions": {"ac_power": "required (the script refuses to measure on battery)",
                           "power_plan": "logged; must be identical in every session",
                           "threads": "1 intra-op, 1 inter-op", "cpu_affinity": f"logical CPU {L.CPU} (P-core), set before "
                           "torch import", "cpu_load_gate": "system CPU load <= 15% over 5 s before each session"},
            "pilot": {"purpose": "stability of readings only; dense models only, so no pruned-model outcome is seen",
                      "design": "2 fresh processes per architecture, each 1,000 warm-up then 3 consecutive dense blocks of N",
                      "rule": "if the coefficient of variation of dense E_1000 over the 6 blocks is <= 3% for both "
                              "architectures, use N = 100,000; otherwise double N once and repeat the pilot; if still > 3%, "
                              "stop and report that the energy readings are not stable enough"}},
        "latency_and_compute": {"latency": "median per-inference time (perf_counter_ns) over the same N inferences",
                                "speedup": "dense median / method median within a session",
                                "params_macs": "analytic, verified against the built model",
                                "edp": "E_per_inference * median latency, normalized to the same-session dense anchor; "
                                       "not a carbon metric", "storage": "size of the saved state dict"},
        "statistics": {
            "summary": "per method: median, mean, SD and IQR over 5 sessions of E_1000, latency, paired saving vs dense",
            "paired_test_vs_dense": "exact one-sided Wilcoxon signed-rank on the 5 paired E_1000 values (H1: method < "
                                    "dense); with n = 5 the smallest attainable p is 0.031, so p-values are secondary",
            "paired_tests_between_methods": "exact two-sided Wilcoxon signed-rank on the 5 session pairs (smallest "
                                            "attainable p 0.0625); reported descriptively",
            "noise_threshold": "2 x the between-session coefficient of variation of the dense anchor's E_1000",
            "consistent_energy_reduction": "lower E_1000 than the same-session dense anchor in 5 of 5 sessions AND median "
                                           "paired saving > noise threshold",
            "stop_rule": "if a method's median saving lies within the noise threshold, its energy result is reported as "
                         "not distinguishable from session noise"},
        "primary_success_criterion": "Case A for an architecture: PPO-Structured satisfies the 0.5 pp preservation rule "
                                     "AND shows a consistent energy reduction versus dense",
        "interpretation_cases": {
            "A": "PPO preserves accuracy and consistently reduces energy: 'The structured deployment realization of the "
                 "adaptive pruning policy reduces measured CPU inference energy while preserving predictive performance.'",
            "B": "consistent energy reduction but > 0.5 pp loss: 'The policy reduces operational energy, but the observed "
                 "saving comes with a measurable accuracy trade-off.'",
            "C": "accuracy preserved but no consistent energy reduction: 'Accuracy-preserving structured pruning did not "
                 "translate into measurable energy savings on this CPU.'",
            "D": "a baseline is more energy-efficient than PPO at comparable accuracy: stated transparently"},
        "method_comparison_rule": {
            "comparable_accuracy": "|A_PPO - A_baseline| <= 0.5 pp on VAL-RL",
            "baseline_better": "comparable accuracy and baseline E_1000 lower than PPO's in 5 of 5 sessions, or baseline "
                               "accuracy higher by > 0.5 pp with median E_1000 not higher than PPO's",
            "ppo_better": "the same conditions with the roles exchanged",
            "otherwise": "no clear difference in the energy-accuracy trade-off"},
        "projection": "deployment-scale savings for 1e6, 1e7, 1e8 inferences = (E_dense - E_method) per inference x N, "
                      "in kWh; labelled a projection; no CO2e conversion",
        "claims_policy": "no 'green', 'environmentally friendly', 'carbon-neutral' or carbon-footprint wording; allowed: "
                         "energy-efficient inference, reduced operational inference energy",
        "known_before_registration": "Paper-3 batch-1 latency tables already contain the action-grid shapes of the PPO "
                                     "policies; no Phase-7A energy, accuracy or latency value had been measured",
    }
    json.dump(pre, open(L.PREREG, "w", encoding="utf-8"), indent=1)
    print("wrote", L.PREREG, L.sha256_file(L.PREREG))


if __name__ == "__main__":
    main()
