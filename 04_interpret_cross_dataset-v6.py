# =============================================================================
# SCRIPT 04 — FULL CROSS-DATASET INTERPRETATION
# =============================================================================
# Purpose:
#   Reads ALL outputs from Script 03 (one or more reference models),
#   merges them into a unified cross-dataset analysis, and generates
#   paper-ready visualizations, tables, and an interpretive report.
#
# Prerequisites:
#   Run Script 03 at least once (ideally once per reference model):
#     REFERENCE_MODEL_DIR = "model_outputs/CIC17"    → python 03_...py
#     REFERENCE_MODEL_DIR = "model_outputs/CIC18_TUE" → python 03_...py
#     ...
#
# Usage:
#   python 04_interpret_cross_dataset.py
#
# Output:
#   cross_dataset_interpretation/
#   ├── full_mcc_heatmap.png            (N×N train→test grid)
#   ├── full_f1_heatmap.png
#   ├── full_metrics_table.csv          (all results merged)
#   ├── generalization_gap_all.png      (gap per reference model)
#   ├── best_worst_pairs.png
#   ├── per_reference_comparison.png    (within vs cross per ref)
#   ├── shap_global_stability.png       (feature stability across ALL)
#   ├── shap_consensus_top10.png        (features agreed by all refs)
#   ├── academic_summary_table.csv      (paper-ready table)
#   └── interpretation_report.txt       (full narrative report)
# =============================================================================

import os, sys, warnings, glob
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

warnings.filterwarnings("ignore")
plt.style.use("seaborn-v0_8-whitegrid")


# =============================================================================
# CONFIGURATION
# =============================================================================

# Root folder where Script 03 saves results (auto-discovers from_* subfolders)
CROSS_RESULTS_ROOT = "cross_dataset_results"

# Output folder for this script
OUTPUT_DIR         = "cross_dataset_interpretation"

os.makedirs(OUTPUT_DIR, exist_ok=True)


# =============================================================================
# PHASE 1: DISCOVER & MERGE ALL SCRIPT 03 OUTPUTS
# =============================================================================

def discover_cross_results():
    """Find all from_<ref>/ folders produced by Script 03."""
    if not os.path.isdir(CROSS_RESULTS_ROOT):
        print(f"  ✗ '{CROSS_RESULTS_ROOT}/' not found. Run Script 03 first.")
        sys.exit(1)

    folders = sorted([
        d for d in os.listdir(CROSS_RESULTS_ROOT)
        if d.startswith("from_") and os.path.isdir(os.path.join(CROSS_RESULTS_ROOT, d))
    ])

    results = {}
    for folder in folders:
        ref_name = folder.replace("from_", "")
        csv_path = os.path.join(CROSS_RESULTS_ROOT, folder, "cross_dataset_metrics.csv")
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
            df["Reference"] = ref_name
            results[ref_name] = df
            print(f"  ✔ Loaded {ref_name}: {len(df)} targets")
        else:
            print(f"  ⚠ {folder}/cross_dataset_metrics.csv not found — skipped")

    return results


def merge_all_results(results_dict):
    """Merge all per-reference CSVs into one unified DataFrame."""
    if not results_dict:
        print("  ✗ No results found.")
        sys.exit(1)

    df_all = pd.concat(results_dict.values(), ignore_index=True)

    # Clean up: ensure numeric types
    for col in ["Accuracy", "F1", "Precision", "Recall", "MCC", "AUC-ROC"]:
        if col in df_all.columns:
            df_all[col] = pd.to_numeric(df_all[col], errors="coerce")

    df_all.to_csv(os.path.join(OUTPUT_DIR, "full_metrics_table.csv"), index=False)
    print(f"\n  Merged: {len(df_all)} experiment rows "
          f"({len(results_dict)} references × targets)")
    return df_all


# =============================================================================
# PHASE 2: N×N HEATMAPS  (the core cross-dataset visualization)
# =============================================================================

def generate_heatmaps(df_all):
    """Generate Train×Test heatmaps for MCC and F1."""
    print("\n" + "="*60)
    print("  PHASE 2 — CROSS-DATASET HEATMAPS")
    print("="*60)

    for metric, vmin, vmax, fmt, cmap, label in [
        ("MCC",  -10, 100, ".1f", "RdYlGn", "MCC (%)"),
        ("F1",   0,   1,   ".3f", "RdYlGn", "F1-Score"),
    ]:
        pivot = df_all.pivot_table(
            index="Reference", columns="Target",
            values=metric, aggfunc="mean"
        )

        if pivot.empty:
            print(f"  ⚠ No data for {metric} heatmap")
            continue

        # Sort to put diagonal (within) visually clear
        all_names = sorted(set(pivot.index) | set(pivot.columns))
        pivot = pivot.reindex(index=all_names, columns=all_names)

        fig, ax = plt.subplots(figsize=(max(8, len(all_names)*2),
                                        max(6, len(all_names)*1.5)))
        mask = pivot.isna()

        sns.heatmap(pivot, annot=True, fmt=fmt, cmap=cmap,
                    vmin=vmin, vmax=vmax, mask=mask,
                    linewidths=1.5, linecolor="white",
                    cbar_kws={"label": label},
                    ax=ax)

        # Highlight diagonal
        for i in range(len(all_names)):
            if all_names[i] in pivot.index and all_names[i] in pivot.columns:
                ax.add_patch(plt.Rectangle((i, i), 1, 1,
                             fill=False, edgecolor="black", lw=2.5))

        ax.set_title(f"Cross-Dataset {label}\n"
                     f"(rows = trained on, columns = tested on, "
                     f"diagonal = within-dataset)",
                     fontsize=12, fontweight="bold")
        ax.set_ylabel("Trained On (Reference)")
        ax.set_xlabel("Tested On (Target)")
        plt.tight_layout()

        fname = f"full_{metric.lower()}_heatmap.png"
        plt.savefig(os.path.join(OUTPUT_DIR, fname), dpi=200, bbox_inches="tight")
        plt.close()
        print(f"  ✔ {fname}")

    return pivot


# =============================================================================
# PHASE 3: GENERALIZATION GAP ANALYSIS
# =============================================================================

def analyze_generalization_gap(df_all):
    """Compare within vs cross performance for each reference model."""
    print("\n" + "="*60)
    print("  PHASE 3 — GENERALIZATION GAP")
    print("="*60)

    gap_rows = []
    for ref in df_all["Reference"].unique():
        sub = df_all[df_all["Reference"] == ref]
        within = sub[sub["Type"] == "Within"]
        cross  = sub[sub["Type"] == "Cross"]

        if len(within) == 0 or len(cross) == 0:
            continue

        w_f1  = within["F1"].mean()
        w_mcc = within["MCC"].mean()
        c_f1  = cross["F1"].mean()
        c_mcc = cross["MCC"].mean()

        gap_rows.append({
            "Reference": ref,
            "Within_F1": round(w_f1, 4), "Cross_F1": round(c_f1, 4),
            "Gap_F1": round(w_f1 - c_f1, 4),
            "Within_MCC": round(w_mcc, 2), "Cross_MCC": round(c_mcc, 2),
            "Gap_MCC": round(w_mcc - c_mcc, 2),
            "N_cross_targets": len(cross),
        })

    if not gap_rows:
        print("  ⚠ No within+cross pairs found")
        return pd.DataFrame()

    df_gap = pd.DataFrame(gap_rows).sort_values("Gap_MCC", ascending=False)
    df_gap.to_csv(os.path.join(OUTPUT_DIR, "generalization_gap.csv"), index=False)

    print(f"\n{df_gap.to_string(index=False)}")

    # ── Visualization ──
    n = len(df_gap)
    fig, axes = plt.subplots(1, 2, figsize=(14, max(5, n*0.8)))

    x = np.arange(n)
    w = 0.35

    # F1 gap
    axes[0].barh(x - w/2, df_gap["Within_F1"], w, label="Within",
                 color="forestgreen", edgecolor="black", alpha=0.85)
    axes[0].barh(x + w/2, df_gap["Cross_F1"], w, label="Cross (avg)",
                 color="steelblue", edgecolor="black", alpha=0.85)
    for i, (_, row) in enumerate(df_gap.iterrows()):
        axes[0].text(max(row["Within_F1"], row["Cross_F1"]) + 0.01, i,
                     f"↓{row['Gap_F1']:.3f}", va="center",
                     fontsize=9, color="red", fontweight="bold")
    axes[0].set_yticks(x); axes[0].set_yticklabels(df_gap["Reference"])
    axes[0].set_xlabel("F1-Score"); axes[0].set_xlim(0, 1.15)
    axes[0].set_title("F1-Score Gap", fontweight="bold")
    axes[0].legend(fontsize=9)

    # MCC gap
    axes[1].barh(x - w/2, df_gap["Within_MCC"], w, label="Within",
                 color="forestgreen", edgecolor="black", alpha=0.85)
    axes[1].barh(x + w/2, df_gap["Cross_MCC"], w, label="Cross (avg)",
                 color="steelblue", edgecolor="black", alpha=0.85)
    for i, (_, row) in enumerate(df_gap.iterrows()):
        axes[1].text(max(row["Within_MCC"], row["Cross_MCC"]) + 1, i,
                     f"↓{row['Gap_MCC']:.1f}%", va="center",
                     fontsize=9, color="red", fontweight="bold")
    axes[1].set_yticks(x); axes[1].set_yticklabels(df_gap["Reference"])
    axes[1].set_xlabel("MCC (%)"); axes[1].set_xlim(-10, 115)
    axes[1].set_title("MCC Gap", fontweight="bold")
    axes[1].legend(fontsize=9)

    plt.suptitle("Generalization Gap per Reference Model",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "generalization_gap_all.png"), dpi=150)
    plt.close()
    print("  ✔ generalization_gap_all.png")

    return df_gap


# =============================================================================
# PHASE 4: BEST / WORST PAIRS
# =============================================================================

def analyze_best_worst(df_all):
    """Identify best and worst cross-dataset pairs."""
    print("\n" + "="*60)
    print("  PHASE 4 — BEST & WORST CROSS-DATASET PAIRS")
    print("="*60)

    cross = df_all[df_all["Type"] == "Cross"].dropna(subset=["MCC"]).copy()
    if cross.empty:
        print("  ⚠ No cross-dataset results")
        return

    cross["Pair"] = cross["Reference"] + " → " + cross["Target"]
    cross_sorted = cross.sort_values("MCC", ascending=False)

    n_show = min(10, len(cross_sorted))
    top    = cross_sorted.head(n_show)
    bottom = cross_sorted.tail(n_show)

    print(f"\n  Top {n_show} cross-dataset pairs:")
    for _, row in top.iterrows():
        print(f"    {row['Pair']:35s} MCC={row['MCC']:6.2f}%  F1={row['F1']:.4f}")

    print(f"\n  Bottom {n_show} cross-dataset pairs:")
    for _, row in bottom.iterrows():
        print(f"    {row['Pair']:35s} MCC={row['MCC']:6.2f}%  F1={row['F1']:.4f}")

    # ── Visualization ──
    fig, axes = plt.subplots(2, 1, figsize=(12, max(8, n_show*0.8)))

    # Top
    colors_top = ["#2E8B57" if m > 50 else "#DAA520" if m > 25 else "#CD5C5C"
                  for m in top["MCC"]]
    axes[0].barh(top["Pair"][::-1], top["MCC"][::-1],
                 color=colors_top[::-1], edgecolor="black", alpha=0.85)
    axes[0].set_xlabel("MCC (%)"); axes[0].set_xlim(0, 105)
    axes[0].set_title(f"Top {n_show} Cross-Dataset Pairs", fontweight="bold")
    for i, (_, row) in enumerate(top[::-1].iterrows()):
        axes[0].text(row["MCC"] + 1, i, f"{row['MCC']:.1f}%",
                     va="center", fontsize=9)

    # Bottom
    colors_bot = ["#CD5C5C" if m < 25 else "#DAA520" for m in bottom["MCC"]]
    axes[1].barh(bottom["Pair"][::-1], bottom["MCC"][::-1],
                 color=colors_bot[::-1], edgecolor="black", alpha=0.85)
    axes[1].set_xlabel("MCC (%)"); axes[1].set_xlim(-10, 105)
    axes[1].set_title(f"Bottom {n_show} Cross-Dataset Pairs", fontweight="bold")
    for i, (_, row) in enumerate(bottom[::-1].iterrows()):
        axes[1].text(max(row["MCC"] + 1, 1), i, f"{row['MCC']:.1f}%",
                     va="center", fontsize=9)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "best_worst_pairs.png"), dpi=150)
    plt.close()
    print("  ✔ best_worst_pairs.png")


# =============================================================================
# PHASE 5: SHAP GLOBAL STABILITY  (merge from all references)
# =============================================================================

def analyze_shap_global_stability():
    """
    Merge SHAP stability reports from all Script 03 runs.
    Identifies features that are consistently important regardless of
    which dataset the model was trained on AND tested on.
    """
    print("\n" + "="*60)
    print("  PHASE 5 — SHAP GLOBAL FEATURE STABILITY")
    print("="*60)

    # Collect all shap_stability_report.csv
    all_rankings = {}
    pattern = os.path.join(CROSS_RESULTS_ROOT, "from_*", "shap_stability",
                           "shap_stability_report.csv")
    files = sorted(glob.glob(pattern))

    if not files:
        print("  ⚠ No SHAP stability reports found. "
              "Run Script 03 with RUN_SHAP=True.")
        return

    for f in files:
        ref_name = os.path.basename(os.path.dirname(os.path.dirname(f))).replace("from_", "")
        df = pd.read_csv(f, index_col=0)
        # Each column (except Avg Rank, Std) is a target dataset's SHAP ranking
        for col in df.columns:
            if col in ["Avg Rank", "Std"]:
                continue
            key = f"{ref_name}→{col}"
            all_rankings[key] = df[col]
        print(f"  ✔ {ref_name}: {len([c for c in df.columns if c not in ['Avg Rank','Std']])} targets")

    if len(all_rankings) < 2:
        print("  ⚠ Need ≥2 experiment combinations for global analysis")
        return

    # Build global ranking matrix
    df_global = pd.DataFrame(all_rankings)
    df_global["Global Avg Rank"] = df_global.mean(axis=1)
    df_global["Global Std"]      = df_global.drop(columns=["Global Avg Rank"]).std(axis=1)
    df_global = df_global.sort_values("Global Avg Rank")

    # Save
    df_global.to_csv(os.path.join(OUTPUT_DIR, "shap_global_stability.csv"))

    # ── Consensus top-10 ──
    top10 = df_global.head(10)
    print(f"\n  ── Global Top 10 Most Important Features ──")
    print(f"  (averaged across ALL train→test combinations)")
    for i, (feat, row) in enumerate(top10.iterrows(), 1):
        stability = "★★★" if row["Global Std"] < 3 else \
                    ("★★" if row["Global Std"] < 5 else "★")
        print(f"    {i:2}. {feat:35s} avg_rank={row['Global Avg Rank']:5.1f}  "
              f"std={row['Global Std']:4.1f}  {stability}")

    # ── Visualization: consensus bar ──
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = ["#2E8B57" if s < 3 else "#DAA520" if s < 5 else "#CD5C5C"
              for s in top10["Global Std"]]
    bars = ax.barh(top10.index[::-1],
                   top10["Global Avg Rank"].max() - top10["Global Avg Rank"][::-1] + 1,
                   color=colors[::-1], edgecolor="black", alpha=0.85)
    ax.set_xlabel("Relative Importance (higher = more important)")
    ax.set_title("SHAP Consensus Top-10 Features\n"
                 "(across all train→test combinations)",
                 fontsize=12, fontweight="bold")

    # Add stability indicator
    for i, (feat, row) in enumerate(top10[::-1].iterrows()):
        stability = "★★★" if row["Global Std"] < 3 else \
                    ("★★" if row["Global Std"] < 5 else "★")
        ax.text(0.5, i, f" std={row['Global Std']:.1f} {stability}",
                va="center", fontsize=8, color="white", fontweight="bold")

    from matplotlib.patches import Patch
    legend = [Patch(color="#2E8B57", label="Highly stable (std<3)"),
              Patch(color="#DAA520", label="Moderate (std<5)"),
              Patch(color="#CD5C5C", label="Unstable (std≥5)")]
    ax.legend(handles=legend, fontsize=8, loc="lower right")

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "shap_consensus_top10.png"), dpi=150)
    plt.close()
    print("  ✔ shap_consensus_top10.png")

    # ── Heatmap of all rankings ──
    top15_feats = df_global.head(15).index
    rank_cols = [c for c in df_global.columns if c not in ["Global Avg Rank", "Global Std"]]
    df_heat = df_global.loc[top15_feats, rank_cols]

    if len(rank_cols) >= 2:
        fig, ax = plt.subplots(figsize=(max(10, len(rank_cols)*1.5), 8))
        sns.heatmap(df_heat, annot=True, fmt=".0f", cmap="YlOrRd_r",
                    linewidths=0.5, linecolor="white", ax=ax)
        ax.set_title("SHAP Feature Ranking — All Train→Test Combinations\n"
                     "(lower number = more important)",
                     fontsize=12, fontweight="bold")
        ax.set_xlabel("Train→Test Experiment")
        ax.set_ylabel("Feature")
        plt.xticks(rotation=45, ha="right", fontsize=8)
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, "shap_global_stability.png"),
                    dpi=150, bbox_inches="tight")
        plt.close()
        print("  ✔ shap_global_stability.png")

    # ── Pairwise Spearman across ALL combinations ──
    try:
        from scipy.stats import spearmanr
        from itertools import combinations

        corr_matrix = pd.DataFrame(index=rank_cols, columns=rank_cols, dtype=float)
        for a, b in combinations(rank_cols, 2):
            common = df_global[[a, b]].dropna()
            if len(common) >= 5:
                rho, _ = spearmanr(common[a], common[b])
                corr_matrix.loc[a, b] = rho
                corr_matrix.loc[b, a] = rho
        np.fill_diagonal(corr_matrix.values, 1.0)

        avg_rho = corr_matrix.values[np.triu_indices_from(corr_matrix.values, k=1)]
        avg_rho = avg_rho[~np.isnan(avg_rho)]

        if len(avg_rho) > 0:
            print(f"\n  Spearman ρ across all pairs:")
            print(f"    Mean  : {np.mean(avg_rho):.4f}")
            print(f"    Median: {np.median(avg_rho):.4f}")
            print(f"    Min   : {np.min(avg_rho):.4f}")
            print(f"    Max   : {np.max(avg_rho):.4f}")

            strength = "Strong" if np.mean(avg_rho) > 0.7 else \
                       ("Moderate" if np.mean(avg_rho) > 0.4 else "Weak")
            print(f"    → {strength} global feature consistency")

    except ImportError:
        print("  ⚠ scipy not installed — Spearman skipped")

    return df_global


# =============================================================================
# PHASE 6: ACADEMIC SUMMARY TABLE
# =============================================================================

def generate_academic_table(df_all, df_gap):
    """Generate a paper-ready summary table."""
    print("\n" + "="*60)
    print("  PHASE 6 — ACADEMIC SUMMARY TABLE")
    print("="*60)

    # Table 1: Per-reference within vs cross
    if len(df_gap) > 0:
        table1 = df_gap[[
            "Reference", "Within_F1", "Cross_F1", "Gap_F1",
            "Within_MCC", "Cross_MCC", "Gap_MCC", "N_cross_targets"
        ]].copy()
        table1.columns = [
            "Train Dataset", "Within F1", "Cross F1 (avg)", "F1 Gap",
            "Within MCC%", "Cross MCC% (avg)", "MCC Gap%", "N Cross Targets"
        ]
        table1.to_csv(os.path.join(OUTPUT_DIR, "academic_table_gap.csv"), index=False)
        print(f"\n  Table 1 — Generalization Gap:")
        print(f"  {table1.to_string(index=False)}")

    # Table 2: Full N×N MCC matrix
    pivot_mcc = df_all.pivot_table(
        index="Reference", columns="Target", values="MCC", aggfunc="mean"
    ).round(2)
    pivot_mcc.to_csv(os.path.join(OUTPUT_DIR, "academic_table_mcc_matrix.csv"))
    print(f"\n  Table 2 — MCC Matrix (%):")
    print(f"  {pivot_mcc.to_string()}")

    # Table 3: Full N×N F1 matrix
    pivot_f1 = df_all.pivot_table(
        index="Reference", columns="Target", values="F1", aggfunc="mean"
    ).round(4)
    pivot_f1.to_csv(os.path.join(OUTPUT_DIR, "academic_table_f1_matrix.csv"))
    print(f"\n  Table 3 — F1 Matrix:")
    print(f"  {pivot_f1.to_string()}")

    # Combined summary CSV
    summary_rows = []
    for _, row in df_all.iterrows():
        summary_rows.append({
            "Train": row["Reference"],
            "Test": row["Target"],
            "Type": row["Type"],
            "F1": row["F1"],
            "MCC%": row["MCC"],
            "Accuracy": row["Accuracy"],
            "Precision": row["Precision"],
            "Recall": row["Recall"],
            "AUC-ROC": row.get("AUC-ROC"),
        })
    df_academic = pd.DataFrame(summary_rows)
    df_academic.to_csv(os.path.join(OUTPUT_DIR, "academic_summary_table.csv"), index=False)
    print(f"\n  ✔ All academic tables saved")


# =============================================================================
# PHASE 7: NARRATIVE INTERPRETATION REPORT
# =============================================================================

def generate_interpretation_report(df_all, df_gap, shap_global):
    """Generate a structured text report with findings and interpretations."""
    print("\n" + "="*60)
    print("  PHASE 7 — INTERPRETATION REPORT")
    print("="*60)

    lines = []
    lines.append("="*70)
    lines.append("  CROSS-DATASET GENERALIZATION — FULL INTERPRETATION REPORT")
    lines.append("="*70)

    refs = sorted(df_all["Reference"].unique())
    targets = sorted(df_all["Target"].unique())
    within = df_all[df_all["Type"] == "Within"]
    cross  = df_all[df_all["Type"] == "Cross"].dropna(subset=["MCC"])

    lines.append(f"\n  Experiment scope:")
    lines.append(f"    Reference models : {len(refs)} → {refs}")
    lines.append(f"    Target datasets  : {len(targets)} → {targets}")
    lines.append(f"    Total experiments: {len(df_all)}")
    lines.append(f"    Within-dataset   : {len(within)}")
    lines.append(f"    Cross-dataset    : {len(cross)}")

    # ── Finding 1: Within-dataset performance ──
    lines.append(f"\n{'─'*70}")
    lines.append(f"  FINDING 1: WITHIN-DATASET PERFORMANCE")
    lines.append(f"{'─'*70}")
    if len(within) > 0:
        avg_w_f1  = within["F1"].mean()
        avg_w_mcc = within["MCC"].mean()
        lines.append(f"  Average within-dataset: F1={avg_w_f1:.4f}, MCC={avg_w_mcc:.2f}%")
        for _, row in within.iterrows():
            lines.append(f"    {row['Reference']:20s} F1={row['F1']:.4f}  MCC={row['MCC']:.2f}%")

        if avg_w_f1 > 0.99:
            lines.append(f"\n  Interpretation: Near-perfect within-dataset performance indicates")
            lines.append(f"  that the binary classification task is well-separated in each dataset's")
            lines.append(f"  own feature space. However, this does NOT guarantee generalization.")
        elif avg_w_f1 > 0.95:
            lines.append(f"\n  Interpretation: High but imperfect within-dataset performance suggests")
            lines.append(f"  some genuine classification difficulty, which is actually a positive")
            lines.append(f"  indicator for generalization potential.")

    # ── Finding 2: Cross-dataset collapse ──
    lines.append(f"\n{'─'*70}")
    lines.append(f"  FINDING 2: CROSS-DATASET GENERALIZATION")
    lines.append(f"{'─'*70}")
    if len(cross) > 0:
        avg_c_f1  = cross["F1"].mean()
        avg_c_mcc = cross["MCC"].mean()
        lines.append(f"  Average cross-dataset: F1={avg_c_f1:.4f}, MCC={avg_c_mcc:.2f}%")

        if len(within) > 0:
            gap_f1  = within["F1"].mean() - avg_c_f1
            gap_mcc = within["MCC"].mean() - avg_c_mcc
            lines.append(f"  Generalization gap: F1={gap_f1:+.4f}, MCC={gap_mcc:+.2f}%")

            if gap_mcc > 50:
                lines.append(f"\n  Interpretation: SEVERE generalization failure.")
                lines.append(f"  The {gap_mcc:.0f}-point MCC drop indicates that models learn")
                lines.append(f"  dataset-specific artifacts rather than universal intrusion patterns.")
                lines.append(f"  This is consistent with Cantone et al. (2024) who reported a")
                lines.append(f"  65-point gap (94.63% within vs 29.35% cross).")
            elif gap_mcc > 20:
                lines.append(f"\n  Interpretation: MODERATE generalization gap.")
                lines.append(f"  Some transfer learning occurs but significant performance loss")
                lines.append(f"  when deployed on unseen network environments.")
            else:
                lines.append(f"\n  Interpretation: MILD generalization gap.")
                lines.append(f"  Models show reasonable transfer ability, possibly due to")
                lines.append(f"  overlapping network characteristics or attack patterns.")

    # ── Finding 3: Best/worst pairs ──
    lines.append(f"\n{'─'*70}")
    lines.append(f"  FINDING 3: BEST & WORST CROSS-DATASET PAIRS")
    lines.append(f"{'─'*70}")
    if len(cross) > 0:
        best_row  = cross.loc[cross["MCC"].idxmax()]
        worst_row = cross.loc[cross["MCC"].idxmin()]
        lines.append(f"  Best  : {best_row['Reference']} → {best_row['Target']}  "
                     f"MCC={best_row['MCC']:.2f}%")
        lines.append(f"  Worst : {worst_row['Reference']} → {worst_row['Target']}  "
                     f"MCC={worst_row['MCC']:.2f}%")

        if best_row["MCC"] > 50:
            lines.append(f"\n  The best pair achieves above-chance generalization,")
            lines.append(f"  suggesting partial overlap in attack signatures between")
            lines.append(f"  {best_row['Reference']} and {best_row['Target']}.")

        if worst_row["MCC"] < 10:
            lines.append(f"\n  The worst pair performs at near-random levels,")
            lines.append(f"  indicating fundamental distributional mismatch.")

    # ── Finding 4: Which reference generalizes best? ──
    lines.append(f"\n{'─'*70}")
    lines.append(f"  FINDING 4: BEST REFERENCE MODEL FOR GENERALIZATION")
    lines.append(f"{'─'*70}")
    if len(df_gap) > 0:
        best_gen = df_gap.loc[df_gap["Gap_MCC"].idxmin()]
        worst_gen = df_gap.loc[df_gap["Gap_MCC"].idxmax()]
        lines.append(f"  Smallest gap : {best_gen['Reference']}  "
                     f"(gap={best_gen['Gap_MCC']:.2f}%)")
        lines.append(f"  Largest gap  : {worst_gen['Reference']}  "
                     f"(gap={worst_gen['Gap_MCC']:.2f}%)")
        lines.append(f"\n  The model trained on {best_gen['Reference']} generalizes best,")
        lines.append(f"  possibly due to greater data diversity or more representative")
        lines.append(f"  attack patterns in the training set.")

    # ── Finding 5: SHAP stability ──
    lines.append(f"\n{'─'*70}")
    lines.append(f"  FINDING 5: FEATURE IMPORTANCE STABILITY (SHAP)")
    lines.append(f"{'─'*70}")
    if shap_global is not None and len(shap_global) > 0:
        top5 = shap_global.head(5)
        lines.append(f"  Top 5 globally stable features:")
        for i, (feat, row) in enumerate(top5.iterrows(), 1):
            lines.append(f"    {i}. {feat:35s} avg_rank={row['Global Avg Rank']:.1f}  "
                         f"std={row['Global Std']:.1f}")

        avg_std = shap_global["Global Std"].head(10).mean()
        if avg_std < 3:
            lines.append(f"\n  Interpretation: Top features are HIGHLY STABLE across datasets.")
            lines.append(f"  This suggests the model captures genuinely discriminative traffic")
            lines.append(f"  characteristics rather than dataset-specific artifacts.")
        elif avg_std < 5:
            lines.append(f"\n  Interpretation: MODERATE feature stability.")
            lines.append(f"  Some features are consistent, others shift between datasets.")
        else:
            lines.append(f"\n  Interpretation: LOW feature stability.")
            lines.append(f"  Different datasets rely on different features for classification.")
            lines.append(f"  This directly explains the poor cross-dataset generalization:")
            lines.append(f"  the model learns different decision rules per dataset.")
    else:
        lines.append(f"  No SHAP data available. Run Script 03 with RUN_SHAP=True.")

    # ── Methodological notes ──
    lines.append(f"\n{'─'*70}")
    lines.append(f"  METHODOLOGICAL NOTES")
    lines.append(f"{'─'*70}")
    lines.append(f"  • All models trained with leakage-free pipeline")
    lines.append(f"    (split → feature select → scale → SMOTE, train partition only)")
    lines.append(f"  • Cross-dataset: reference model's scaler applied to target data")
    lines.append(f"  • Missing features in target filled with 0 (logged per experiment)")
    lines.append(f"  • MCC used as primary metric (robust to class imbalance)")
    lines.append(f"  • SHAP stability measured via rank std across experiments")

    lines.append(f"\n{'='*70}")

    report_text = "\n".join(lines)
    print(report_text)

    with open(os.path.join(OUTPUT_DIR, "interpretation_report.txt"), "w",
              encoding="utf-8") as f:
        f.write(report_text)
    print(f"\n  ✔ interpretation_report.txt saved")


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("\n" + "="*70)
    print("  SCRIPT 04 — CROSS-DATASET INTERPRETATION")
    print(f"  Source : {CROSS_RESULTS_ROOT}/")
    print(f"  Output : {OUTPUT_DIR}/")
    print("="*70)

    # Phase 1: Discover & merge
    print("\n" + "="*60)
    print("  PHASE 1 — DISCOVERING RESULTS")
    print("="*60)
    results_dict = discover_cross_results()
    df_all = merge_all_results(results_dict)

    # Phase 2: Heatmaps
    generate_heatmaps(df_all)

    # Phase 3: Gap analysis
    df_gap = analyze_generalization_gap(df_all)

    # Phase 4: Best/worst
    analyze_best_worst(df_all)

    # Phase 5: SHAP global stability
    shap_global = analyze_shap_global_stability()

    # Phase 6: Academic tables
    generate_academic_table(df_all, df_gap)

    # Phase 7: Narrative report
    generate_interpretation_report(df_all, df_gap, shap_global)

    # Done
    n_refs    = df_all["Reference"].nunique()
    n_targets = df_all["Target"].nunique()
    n_within  = len(df_all[df_all["Type"] == "Within"])
    n_cross   = len(df_all[df_all["Type"] == "Cross"])

    print(f"""
  ╔═══════════════════════════════════════════════════════════╗
  ║  SCRIPT 04 COMPLETE ✔
  ╠═══════════════════════════════════════════════════════════╣
  ║  References : {n_refs}
  ║  Targets    : {n_targets}
  ║  Within     : {n_within} experiments
  ║  Cross      : {n_cross} experiments
  ║
  ║  Output → ./{OUTPUT_DIR}/
  ║   ├── full_mcc_heatmap.png          (N×N grid)
  ║   ├── full_f1_heatmap.png
  ║   ├── full_metrics_table.csv
  ║   ├── generalization_gap_all.png
  ║   ├── generalization_gap.csv
  ║   ├── best_worst_pairs.png
  ║   ├── academic_table_gap.csv        (paper Table 1)
  ║   ├── academic_table_mcc_matrix.csv (paper Table 2)
  ║   ├── academic_table_f1_matrix.csv  (paper Table 3)
  ║   ├── academic_summary_table.csv
  ║   ├── shap_global_stability.csv
  ║   ├── shap_global_stability.png
  ║   ├── shap_consensus_top10.png
  ║   └── interpretation_report.txt
  ╚═══════════════════════════════════════════════════════════╝
    """)


if __name__ == "__main__":
    main()
