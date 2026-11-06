"""Figure map shared by the assembly and DOCX tools: figure number -> neutral thesis file (figures/final/)."""
import os
import re
import shutil

HERE = os.path.dirname(os.path.abspath(__file__))
W = os.path.dirname(HERE)
FIN = os.path.join(W, "figures", "final")
# figure number -> (shortlist id, source path relative to W, neutral name, status note)
FIG = {
    "3.1": ("F1", "figures/F1_framework.png", "fig_3_1_overall_framework.png", "original thesis figure (Milestone 1)"),
    "5.1": ("F3", "figures/final/fig_5_1_phase3_accuracy_vs_sparsity.png", "fig_5_1_phase3_accuracy_vs_sparsity.png", "300-dpi re-render with the unchanged Phase-3 plotting function (was 130 dpi)"),
    "5.2": ("F4", "figures/final/fig_5_2_phase3_allocation_heatmap.png", "fig_5_2_phase3_allocation_heatmap.png", "300-dpi re-render with the unchanged Phase-3 plotting function (was 130 dpi)"),
    "5.3": ("F2", "figures/ch5_fig5_3_sensitivity_tolerance.png", "fig_5_3_sensitivity_profile.png", "thesis re-render (colour key), 86/86 values verified"),
    "5.4": ("F6", "figures/ch5_fig5_4_topk_reach_rate.png", "fig_5_4_archive_topk_reach.png", "thesis re-render (legend placement), values verified"),
    "5.5": ("F7", "figures/ch5_fig5_5_resnet_reward_alignment.png", "fig_5_5_resnet8_reward_alignment.png", "thesis re-render (mean/median labels), values verified"),
    "5.6": ("F5", "../phase5/phase5_fig_D_sensitivity_exploration.png", "fig_5_6_sensitivity_exploration.png", "original phase figure"),
    "5.7": ("O4", "../phase5/phase5_fig_E_feasible_sparsity_gap.png", "fig_5_7_feasible_sparsity_gap.png", "original phase figure"),
    "5.8": ("F8", "../phase6/figures/phase6_fig2_ppo_vs_lamp.png", "fig_5_8_ppo_vs_lamp_recovery.png", "original phase figure"),
    "6.1": ("F9", "../paper3/paper_figures_revised/fig1_latency_vs_reduction.png", "fig_6_1_latency_vs_param_mac_reduction.png", "original Paper-3 figure (single-column width)"),
    "6.2": ("O6", "../paper3/paper_figures_revised/fig3_resnet8_latency_room.png", "fig_6_2_resnet8_latency_room.png", "original Paper-3 figure (single-column width)"),
    "6.3": ("F10", "../phase7a_sustainability/figures/fig_A_accuracy_vs_energy.png", "fig_6_3_phase7a_accuracy_vs_energy.png", "original phase figure"),
    "6.4": ("F11", "../phase7b_green_pruning/figures/fig_accuracy_planes.png", "fig_6_4_phase7b_accuracy_planes.png", "original phase figure; copied to a neutral name (source folder name contains 'green')"),
}
DATA = {"3.1": "conceptual (no data)", "5.1": "architecture_generalization/phase3_fig_A_accuracy_sparsity.csv",
        "5.2": "architecture_generalization/phase3_fig_B_policy_heatmap.csv", "5.3": "architecture_generalization/phase3_fig_D_sensitivity_tolerance.csv",
        "5.4": "phase4/phase4_fig_D_topk_reach_rate.csv", "5.5": "phase5/phase5_fig_B_resnet_reward_alignment.csv",
        "5.6": "phase5/phase5_fig_D_sensitivity_exploration.csv", "5.7": "phase5/phase5_fig_E_feasible_sparsity_gap.csv",
        "5.8": "phase6/figures (phase6_fig2 source CSV)", "6.1": "paper3/h2_latency_table_{simplecnn,resnet8}_c10.csv",
        "6.2": "paper3/h2_c10_resnet8_1epoch_landscape.csv; resnet8_threshold_calibration.csv", "6.3": "phase7a_sustainability/energy_summary.csv, accuracy_results.csv",
        "6.4": "phase7b_green_pruning/final_summary.csv"}


def copy_figures():
    os.makedirs(FIN, exist_ok=True)
    for n, (_, src, dst, _) in FIG.items():
        s = os.path.normpath(os.path.join(W, src))
        d = os.path.join(FIN, dst)
        if os.path.abspath(s) != os.path.abspath(d):
            shutil.copyfile(s, d)


def fix_figs(t):
    return re.sub(r"!\[Figure ([\d.]+)\]\([^)]+\)", lambda m: f"![Figure {m.group(1)}](figures/final/{FIG[m.group(1)][2]})", t)
