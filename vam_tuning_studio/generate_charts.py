#!/usr/bin/env python3
"""
Publication-Quality Visualization & Heatmap Generator for OpenCAL / VAMToolbox
Generates 2D parameter interaction heatmaps, radar profiles, Pareto frontiers, and sensitivity rankings.
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
    "axes.titlesize": 14,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.titlesize": 16,
    "figure.dpi": 200,
})

def generate_all_charts(df_log_path: str, output_dir: str):
    os.makedirs(output_dir, exist_ok=True)
    if not os.path.exists(df_log_path):
        print(f"[ChartGen] Error: {df_log_path} not found.")
        return

    df = pd.read_csv(df_log_path)
    print(f"[ChartGen] Generating publication charts for {len(df)} trials into '{output_dir}' ...")

    # 1. 2D Heatmaps: dh vs dl for each geometry
    for geom in df["geometry"].unique():
        sub = df[(df["geometry"] == geom) & (df["phase"].str.contains("sweep_dh_dl", na=False))]
        if len(sub) >= 4:
            pivot_ver = sub.pivot_table(index="d_h", columns="d_l", values="ver", aggfunc="mean")
            pivot_pw = sub.pivot_table(index="d_h", columns="d_l", values="pw", aggfunc="mean")

            fig, axes = plt.subplots(1, 2, figsize=(16, 6))
            sns.heatmap(pivot_ver * 100, ax=axes[0], cmap="viridis_r", annot=True, fmt=".1f", cbar_kws={'label': 'Volumetric Error Rate (%)'})
            axes[0].set_title(f"[{geom}] VER (%) vs (d_h, d_l)\n(Lower is Better)")
            axes[0].set_xlabel("Void Ceiling (d_l)")
            axes[0].set_ylabel("Gel Floor (d_h)")

            sns.heatmap(pivot_pw, ax=axes[1], cmap="coolwarm", annot=True, fmt=".2f", cbar_kws={'label': 'Process Window (PW)'})
            axes[1].set_title(f"[{geom}] Process Window vs (d_h, d_l)\n(Higher/Positive is Better)")
            axes[1].set_xlabel("Void Ceiling (d_l)")
            axes[1].set_ylabel("Gel Floor (d_h)")

            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f"heatmap_{geom}_dh_vs_dl.png"), bbox_inches="tight")
            plt.close()

    # 2. 2D Heatmaps: n_angles vs filter for each geometry
    for geom in df["geometry"].unique():
        sub = df[(df["geometry"] == geom) & (df["phase"].str.contains("sweep_angles_filter", na=False))]
        if len(sub) >= 4:
            pivot_pw = sub.pivot_table(index="filter", columns="n_angles", values="pw", aggfunc="mean")
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.heatmap(pivot_pw, ax=ax, cmap="magma", annot=True, fmt=".2f", cbar_kws={'label': 'Process Window (PW)'})
            ax.set_title(f"[{geom}] Process Window: Filter vs Angular Sampling\n(Higher is Better)")
            ax.set_xlabel("Number of Angles")
            ax.set_ylabel("Reconstruction Filter")
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f"heatmap_{geom}_angles_vs_filter.png"), bbox_inches="tight")
            plt.close()

    # 3. 2D Heatmaps: eps vs p-norm for BCLP
    for geom in df["geometry"].unique():
        sub = df[(df["geometry"] == geom) & (df["method"] == "BCLP") & (df["phase"].str.contains("sweep_eps_p", na=False))]
        if len(sub) >= 4:
            pivot_pw = sub.pivot_table(index="eps", columns="p_norm", values="pw", aggfunc="mean")
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.heatmap(pivot_pw, ax=ax, cmap="crest", annot=True, fmt=".2f", cbar_kws={'label': 'Process Window (PW)'})
            ax.set_title(f"[{geom}] BCLP Tuning: Relaxation (eps) vs Norm Power (p)\n(Higher is Better)")
            ax.set_xlabel("Lp Norm Power (p)")
            ax.set_ylabel("Relaxation Band (eps)")
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f"heatmap_{geom}_bclp_eps_vs_p.png"), bbox_inches="tight")
            plt.close()

    # 4. Pareto Frontier: VER vs PW across Methods and Geometries
    plt.figure(figsize=(12, 7))
    sns.scatterplot(
        data=df,
        x="ver",
        y="pw",
        hue="geometry",
        style="method",
        size="n_angles",
        sizes=(40, 200),
        alpha=0.85,
        palette="tab10"
    )
    plt.axhline(0, color="red", linestyle="--", linewidth=1.5, label="Printability Boundary (PW=0)")
    plt.title("VAM Multi-Geometry Pareto Frontier: Volumetric Error vs Process Window", fontsize=15, fontweight="bold")
    plt.xlabel("Volumetric Error Rate (VER) [Lower is Better]")
    plt.ylabel("Process Window (PW = Min Gel - Max Void) [Higher is Better]")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "pareto_frontier_ver_vs_pw.png"), bbox_inches="tight")
    plt.close()

    # 5. Method Comparison Boxplots across Geometries
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    sns.boxplot(data=df, x="geometry", y="pw", hue="method", ax=axes[0], palette="Set2")
    axes[0].axhline(0, color="red", linestyle="--", alpha=0.7)
    axes[0].set_title("Process Window Distribution by Optimizer Method")
    axes[0].set_ylabel("Process Window (PW)")

    sns.boxplot(data=df, x="geometry", y="ver", hue="method", ax=axes[1], palette="Set2")
    axes[1].set_title("Volumetric Error Rate (VER) by Optimizer Method")
    axes[1].set_ylabel("Volumetric Error Rate (VER)")

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "optimizer_method_comparison_boxplots.png"), bbox_inches="tight")
    plt.close()

    # 6. Parameter Sensitivity & Importance Analysis (Correlation with PW and VER)
    numeric_cols = ["d_h", "d_l", "n_angles", "n_iter", "eps", "p_norm", "inhibition", "contrast", "cv", "ipdr"]
    valid_cols = [c for c in numeric_cols if c in df.columns and df[c].nunique() > 1]
    
    corr_pw = {}
    corr_ver = {}
    for col in valid_cols:
        corr_pw[col] = df[col].corr(df["pw"])
        corr_ver[col] = df[col].corr(df["ver"])
    
    corr_df = pd.DataFrame({"PW_Correlation": corr_pw, "VER_Correlation": corr_ver}).dropna()
    
    if not corr_df.empty:
        fig, ax = plt.subplots(figsize=(12, 6))
        corr_df.plot(kind="barh", ax=ax, color=["#2ecc71", "#e74c3c"], width=0.7)
        ax.set_title("Quantitative Parameter Sensitivity & Metric Correlation\n(Positive PW & Negative VER Correlation is Desired)", fontweight="bold")
        ax.set_xlabel("Pearson Correlation Coefficient")
        ax.axvline(0, color="black", linewidth=0.8)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "parameter_importance_ranking.png"), bbox_inches="tight")
        plt.close()

    # 7. Radar Profile for Best Configuration per Geometry
    categories = ["1 - VER", "PW_norm", "1 - CV", "Contrast_norm", "1 - IPDR"]
    best_rows = {}
    for geom in df["geometry"].unique():
        sub = df[df["geometry"] == geom]
        best_row = sub.sort_values(by=["pw", "ver"], ascending=[False, True]).iloc[0]
        best_rows[geom] = best_row

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    angles_rad = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
    angles_rad += angles_rad[:1]

    colors = ["#e74c3c", "#3498db", "#2ecc71", "#f39c12"]
    for idx, (geom, row) in enumerate(best_rows.items()):
        v_ver = max(0.0, min(1.0, 1.0 - float(row["ver"])))
        v_pw = max(0.0, min(1.0, (float(row["pw"]) + 0.5) / 1.5))
        v_cv = max(0.0, min(1.0, 1.0 - min(1.0, float(row.get("cv", 0.3)))))
        v_cr = max(0.0, min(1.0, float(row.get("contrast", 1.0)) / 4.0))
        v_ipdr = max(0.0, min(1.0, 1.0 - float(row.get("ipdr", 0.5))))
        
        values = [v_ver, v_pw, v_cv, v_cr, v_ipdr]
        values += values[:1]
        
        ax.plot(angles_rad, values, color=colors[idx % len(colors)], linewidth=2.5, label=f"{geom} (PW={row['pw']:.2f}, VER={row['ver']*100:.1f}%)")
        ax.fill(angles_rad, values, color=colors[idx % len(colors)], alpha=0.15)

    ax.set_xticks(angles_rad[:-1])
    ax.set_xticklabels(categories, fontsize=11, fontweight="bold")
    ax.set_ylim(0, 1.0)
    ax.set_title("Optimal Reconstruction Quality Profile by Geometry", fontsize=15, fontweight="bold", pad=25)
    plt.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1))
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "radar_geometry_profiles.png"), bbox_inches="tight")
    plt.close()

    print(f"[ChartGen] Successfully generated all visualization artifacts in '{output_dir}'.")
