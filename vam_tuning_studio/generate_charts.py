"""
Publication-Quality Visualization Generator for VAM Single-Model Optimization
=============================================================================
Generates high-resolution performance plots for a single model run:
  - Process Window (PW) vs. Voxel Error Rate (VER%)
  - Optimization Method & Filter Comparison Bar Plots
  - Parameter Sweet-Spot Heatmaps & Scatter Distributions
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="darkgrid")
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.titlesize": 15,
    "figure.dpi": 200,
})


def generate_single_model_charts(trials_csv_path: str, output_dir: str, stl_name: str = "Model"):
    """
    Generates summary charts for a single model optimization run.
    """
    os.makedirs(output_dir, exist_ok=True)
    if not os.path.exists(trials_csv_path):
        return

    try:
        df = pd.read_csv(trials_csv_path)
    except Exception:
        return

    if len(df) == 0:
        return

    # 1. Process Window vs Voxel Error Rate (Pareto Scatter)
    plt.figure(figsize=(10, 6))
    sns.scatterplot(
        data=df,
        x="ver_pct",
        y="window",
        hue="method",
        style="filter",
        size="n_angles",
        sizes=(60, 220),
        alpha=0.9,
        palette="bright"
    )
    plt.axhline(0, color="red", linestyle="--", linewidth=1.5, label="Clean Gelation Threshold (PW = 0)")
    plt.title(f"[{stl_name}] Sweet-Spot Pareto: Process Window vs. Voxel Error Rate (VER%)")
    plt.xlabel("Voxel Error Rate VER (%) [Lower is Better]")
    plt.ylabel("Process Window (PW = In_Min - Out_Max) [Higher is Better]")
    plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "process_window_vs_ver.png"), bbox_inches="tight")
    plt.close()

    # 2. Algorithm Comparison (Process Window by Method & Filter)
    if "method" in df.columns and "filter" in df.columns:
        plt.figure(figsize=(10, 5))
        sns.barplot(
            data=df,
            x="filter",
            y="window",
            hue="method",
            palette="muted",
            errorbar=None
        )
        plt.axhline(0, color="red", linestyle="--", linewidth=1.2)
        plt.title(f"[{stl_name}] Process Window by Reconstruction Filter & Method")
        plt.xlabel("Reconstruction Filter")
        plt.ylabel("Process Window (PW)")
        plt.legend(title="Algorithm")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "filter_method_comparison.png"), bbox_inches="tight")
        plt.close()

    # 3. Parameter Impact Heatmap (if multiple dh and dl are present)
    if len(df["d_h"].unique()) > 1 and len(df["d_l"].unique()) > 1:
        try:
            pivot_pw = df.pivot_table(index="d_h", columns="d_l", values="window", aggfunc="mean")
            plt.figure(figsize=(8, 6))
            sns.heatmap(pivot_pw, cmap="coolwarm", annot=True, fmt=".2f", cbar_kws={'label': 'Process Window (PW)'})
            plt.title(f"[{stl_name}] Process Window vs. Dose Thresholds (d_h, d_l)")
            plt.xlabel("Void Dose Ceiling (d_l)")
            plt.ylabel("Gel Dose Floor (d_h)")
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, "heatmap_dh_vs_dl.png"), bbox_inches="tight")
            plt.close()
        except Exception:
            pass
