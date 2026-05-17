# =============================================================================
# SCRIPT 01 — TRAIN, EVALUATE & SAVE MODELS  (batch or single)
# =============================================================================
# Prerequisites:
#   Run 00_prepare_datasets.py first.
#
# Modes:
#   A) BATCH — process ALL subfolders in PREPARED_ROOT automatically:
#        SINGLE_DATASET = None
#
#   B) SINGLE — process only one specific subfolder:
#        SINGLE_DATASET = "CIC18_TUESDAY"
#
# Usage:
#   python 01_train_save_models.py
#
# Output:
#   model_outputs/
#   ├── CIC17/
#   │   ├── best_model.pkl, <Model>.pkl, scaler.pkl, feature_names.pkl
#   │   ├── X_test.pkl, y_test.pkl, X_train_sample.pkl, train_config.pkl
#   │   ├── label_encoder.pkl
#   │   ├── feature_importance.png, cross_validation.png, cm_*.png
#   │   ├── roc_curves.png, model_comparison.png/.csv, training_time.png
#   │   └── ...
#   ├── CIC18_TUESDAY/
#   │   └── ...
#   └── training_summary.csv     (aggregated comparison across datasets)
#
# Then:
#   python 02_load_explain_models.py
# =============================================================================

# ┌─────────────────────────────────────────────────────┐
# │  INTEL CPU ENVIRONMENT TUNING                       │
# └─────────────────────────────────────────────────────┘
import os, sys, psutil

N_PHYSICAL = psutil.cpu_count(logical=False) or 4

os.environ["MKL_NUM_THREADS"]        = str(N_PHYSICAL)
os.environ["OMP_NUM_THREADS"]        = str(N_PHYSICAL)
os.environ["OPENBLAS_NUM_THREADS"]   = str(N_PHYSICAL)
os.environ["NUMEXPR_NUM_THREADS"]    = str(N_PHYSICAL)
os.environ["VECLIB_MAXIMUM_THREADS"] = str(N_PHYSICAL)
os.environ["OMP_NESTED"]    = "FALSE"
os.environ["OMP_DYNAMIC"]   = "TRUE"
os.environ["MKL_DYNAMIC"]   = "TRUE"
os.environ["OMP_PROC_BIND"] = "close"
os.environ["OMP_PLACES"]    = "cores"
os.environ["KMP_AFFINITY"]  = "granularity=fine,compact,1,0"
os.environ["XGB_TREE_METHOD"] = "hist"

INTEL_PATCHED = False
try:
    from sklearnex import patch_sklearn
    patch_sklearn()
    INTEL_PATCHED = True
    print("[INTEL] ✔ scikit-learn-intelex (oneDAL) ENABLED")
except ImportError:
    print("[INTEL] ⚠ scikit-learn-intelex not installed.")

# =============================================================================
import time, warnings, shutil, gc
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")           # non-interactive backend for batch mode
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from collections import Counter
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    classification_report, confusion_matrix,
    accuracy_score, f1_score, precision_score,
    recall_score, roc_auc_score, roc_curve
)
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
import xgboost as xgb
import lightgbm as lgb
from imblearn.over_sampling import SMOTE

warnings.filterwarnings("ignore")
plt.style.use("seaborn-v0_8-whitegrid")


# =============================================================================
# CONFIGURATION  ← EDIT THESE
# =============================================================================

PREPARED_ROOT    = "prepared_data"      # root folder from Script 00
MODEL_ROOT       = "model_outputs-v5"      # root folder for trained models

# Set to None to process ALL subfolders, or a name to process just one:
SINGLE_DATASET   = None
# SINGLE_DATASET = "CIC18_TUESDAY"

TOP_FEATURES     = 20
RANDOM_STATE     = 42

os.makedirs(MODEL_ROOT, exist_ok=True)


# =============================================================================
# AUTO-DISCOVER DATASETS
# =============================================================================

def discover_datasets():
    """Find all valid prepared-data subfolders (must contain X_train_raw.pkl)."""
    if SINGLE_DATASET:
        candidates = [SINGLE_DATASET]
    else:
        candidates = sorted([
            d for d in os.listdir(PREPARED_ROOT)
            if os.path.isdir(os.path.join(PREPARED_ROOT, d))
        ])

    valid = []
    for name in candidates:
        prep_dir = os.path.join(PREPARED_ROOT, name)
        marker   = os.path.join(prep_dir, "X_train_raw.pkl")
        if os.path.exists(marker):
            valid.append(name)
        else:
            print(f"  ⚠ Skipping '{name}' — X_train_raw.pkl not found")

    return valid


# =============================================================================
# LOAD PREPARED PARTITIONS
# =============================================================================

def load_prepared_data(prep_dir, out_dir):
    print(f"\n  Loading: {prep_dir}")

    X_train = joblib.load(os.path.join(prep_dir, "X_train_raw.pkl"))
    X_test  = joblib.load(os.path.join(prep_dir, "X_test_raw.pkl"))
    y_train = joblib.load(os.path.join(prep_dir, "y_train.pkl"))
    y_test  = joblib.load(os.path.join(prep_dir, "y_test.pkl"))

    print(f"    X_train : {X_train.shape}   {dict(Counter(y_train))}")
    print(f"    X_test  : {X_test.shape}   {dict(Counter(y_test))}")

    try:
        report = joblib.load(os.path.join(prep_dir, "prep_report.pkl"))
        ds_name     = report.get("dataset_name", os.path.basename(prep_dir))
        binary_mode = report.get("binary_mode", True)
    except Exception:
        ds_name     = os.path.basename(prep_dir)
        binary_mode = True

    print(f"    Dataset : {ds_name} | Mode : {'Binary' if binary_mode else 'Multi-class'}")

    le_src = os.path.join(prep_dir, "label_encoder.pkl")
    if os.path.exists(le_src):
        shutil.copy2(le_src, os.path.join(out_dir, "label_encoder.pkl"))

    return X_train, X_test, y_train, y_test, ds_name, binary_mode


# =============================================================================
# FEATURE SELECTION  (TRAIN ONLY)
# =============================================================================

def select_features(X_train, y_train, out_dir, top_n=TOP_FEATURES):
    print(f"\n  Feature selection (top {top_n}, train only)...")
    rf = RandomForestClassifier(n_estimators=50, random_state=RANDOM_STATE, n_jobs=-1)
    rf.fit(X_train, y_train)
    importance = pd.Series(rf.feature_importances_, index=X_train.columns)
    top = importance.nlargest(top_n).index.tolist()

    plt.figure(figsize=(10, 6))
    importance.nlargest(top_n).sort_values().plot(
        kind="barh", color="steelblue", edgecolor="black")
    plt.title("Feature Importances (train only)", fontsize=13, fontweight="bold")
    plt.xlabel("Importance Score"); plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "feature_importance.png"), dpi=150)
    plt.close()
    print(f"    Selected {top_n} features.")
    return top


# =============================================================================
# SCALE + SMOTE  (TRAIN ONLY)
# =============================================================================

def scale_and_resample(X_train_sel, X_test_sel, y_train, features, out_dir):
    print(f"  Scaling (train only)...")
    scaler = StandardScaler()
    X_tr = pd.DataFrame(scaler.fit_transform(X_train_sel), columns=features)
    X_te = pd.DataFrame(scaler.transform(X_test_sel), columns=features)
    joblib.dump(scaler, os.path.join(out_dir, "scaler.pkl"))

    print(f"  SMOTE (train only)...  Before: {dict(Counter(y_train))}")
    smote = SMOTE(random_state=RANDOM_STATE)
    X_res, y_res = smote.fit_resample(X_tr, y_train)
    print(f"    After: {dict(Counter(y_res))}")

    return (pd.DataFrame(X_res, columns=features).reset_index(drop=True),
            X_te.reset_index(drop=True),
            pd.Series(y_res).reset_index(drop=True))


# =============================================================================
# MODELS
# =============================================================================

def get_models():
    return {
        "Decision Tree": DecisionTreeClassifier(max_depth=10, random_state=RANDOM_STATE),
        "Random Forest": RandomForestClassifier(n_estimators=100, max_depth=15,
                                                random_state=RANDOM_STATE, n_jobs=-1),
        "Gradient Boosting": GradientBoostingClassifier(n_estimators=100, learning_rate=0.1,
                                                       max_depth=5, random_state=RANDOM_STATE),
        "XGBoost": xgb.XGBClassifier(n_estimators=100, learning_rate=0.1, max_depth=6,
                                     random_state=RANDOM_STATE, eval_metric="logloss",
                                     n_jobs=-1, tree_method="hist"),
        "LightGBM": lgb.LGBMClassifier(n_estimators=100, learning_rate=0.1, max_depth=6,
                                       random_state=RANDOM_STATE, n_jobs=-1, verbose=-1),
        "Logistic Regression": LogisticRegression(max_iter=1000,
                                                  random_state=RANDOM_STATE, n_jobs=-1),
        "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=5, n_jobs=-1),
        "Naive Bayes": GaussianNB(),
    }


# =============================================================================
# TRAINING
# =============================================================================

def train_all(models, X_train, y_train, out_dir):
    print(f"\n  Training {len(models)} models...")
    trained, train_times = {}, {}
    for name, model in models.items():
        t0 = time.time()
        model.fit(X_train, y_train)
        elapsed = time.time() - t0
        train_times[name] = round(elapsed, 2)
        trained[name] = model
        joblib.dump(model, os.path.join(out_dir, f"{name.replace(' ', '_')}.pkl"))
        print(f"    {name:25s} {elapsed:7.2f}s ✔")
    return trained, train_times


def cross_validate(models, X_train, y_train, out_dir):
    print(f"\n  Cross-validation (5-fold)...")
    cv_results = {}
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    skip = {"Gradient Boosting", "K-Nearest Neighbors"}
    for name, model in models.items():
        if name in skip:
            continue
        scores = cross_val_score(model, X_train, y_train, cv=skf,
                                 scoring="f1_weighted", n_jobs=-1)
        cv_results[name] = scores
        print(f"    {name:25s} F1={scores.mean():.4f} ± {scores.std():.4f}")
    if cv_results:
        names = list(cv_results.keys())
        means = [cv_results[n].mean() for n in names]
        stds  = [cv_results[n].std()  for n in names]
        plt.figure(figsize=(10, 5))
        bars = plt.bar(names, means, yerr=stds, capsize=5,
                       color="steelblue", edgecolor="black", alpha=0.85)
        plt.title("5-Fold CV — Weighted F1", fontweight="bold")
        plt.ylabel("F1"); plt.xticks(rotation=30, ha="right"); plt.ylim(0, 1.05)
        for bar, val in zip(bars, means):
            plt.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.01,
                     f"{val:.3f}", ha="center", fontsize=9)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "cross_validation.png"), dpi=150)
        plt.close()
    return cv_results


# =============================================================================
# EVALUATION
# =============================================================================

def evaluate_all(trained, X_test, y_test, train_times, binary_mode, out_dir):
    print(f"\n  Evaluating...")
    all_metrics = {}
    for name, model in trained.items():
        y_pred = model.predict(X_test)
        acc  = accuracy_score(y_test, y_pred)
        f1   = f1_score(y_test, y_pred, average="weighted")
        prec = precision_score(y_test, y_pred, average="weighted", zero_division=0)
        rec  = recall_score(y_test, y_pred, average="weighted", zero_division=0)
        try:
            y_prob = model.predict_proba(X_test)
            auc = roc_auc_score(y_test, y_prob, multi_class="ovr", average="weighted")
        except Exception:
            auc = None
        all_metrics[name] = {
            "Accuracy": round(acc,4), "F1": round(f1,4),
            "Precision": round(prec,4), "Recall": round(rec,4),
            "AUC-ROC": round(auc,4) if auc else "N/A",
            "Train(s)": train_times.get(name, "N/A"),
        }
        print(f"    {name:25s} F1={f1:.4f}  Acc={acc:.4f}", end="")
        if auc: print(f"  AUC={auc:.4f}")
        else: print()

        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(8, 6) if not binary_mode else (6, 5))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                    linewidths=0.5, linecolor="gray")
        plt.title(f"{name} — Confusion Matrix", fontweight="bold")
        plt.ylabel("Actual"); plt.xlabel("Predicted"); plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"cm_{name.replace(' ', '_')}.png"), dpi=150)
        plt.close()

    # Save classification reports to text file
    with open(os.path.join(out_dir, "classification_reports.txt"), "w") as f:
        for name, model in trained.items():
            y_pred = model.predict(X_test)
            f.write(f"{'='*60}\n{name}\n{'='*60}\n")
            f.write(classification_report(y_test, y_pred))
            f.write("\n\n")

    return all_metrics


def plot_roc(trained, X_test, y_test, binary_mode, out_dir):
    if not binary_mode:
        return
    plt.figure(figsize=(10, 7))
    for name, model in trained.items():
        try:
            y_prob = model.predict_proba(X_test)[:, 1]
            fpr, tpr, _ = roc_curve(y_test, y_prob)
            auc = roc_auc_score(y_test, y_prob)
            plt.plot(fpr, tpr, lw=2, label=f"{name} (AUC={auc:.3f})")
        except Exception:
            pass
    plt.plot([0,1],[0,1], "k--", lw=1.5, label="Random")
    plt.xlabel("FPR"); plt.ylabel("TPR")
    plt.title("ROC Curves", fontweight="bold")
    plt.legend(loc="lower right"); plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "roc_curves.png"), dpi=150)
    plt.close()


def compare_and_save_best(all_metrics, train_times, trained, out_dir):
    df_res = pd.DataFrame(all_metrics).T
    df_res.index.name = "Model"
    df_res.to_csv(os.path.join(out_dir, "model_comparison.csv"))

    metrics = ["Accuracy", "F1", "Precision", "Recall"]
    df_plot = df_res[metrics].apply(pd.to_numeric, errors="coerce")
    df_plot.plot(kind="bar", figsize=(13,6), colormap="tab10",
                 edgecolor="black", alpha=0.85)
    plt.title("Model Performance", fontweight="bold")
    plt.ylabel("Score"); plt.xticks(rotation=30, ha="right"); plt.ylim(0, 1.1)
    plt.legend(loc="lower right"); plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "model_comparison.png"), dpi=150)
    plt.close()

    times = {k:v for k,v in train_times.items() if isinstance(v,(int,float))}
    plt.figure(figsize=(10,5))
    plt.bar(times.keys(), times.values(), color="darkorange",
            edgecolor="black", alpha=0.85)
    plt.title("Training Time", fontweight="bold")
    plt.ylabel("Time (s)"); plt.xticks(rotation=30, ha="right"); plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "training_time.png"), dpi=150)
    plt.close()

    best = df_plot["F1"].idxmax()
    path = os.path.join(out_dir, "best_model.pkl")
    joblib.dump(trained[best], path)
    print(f"  🏆 Best: {best} (F1={df_plot.loc[best,'F1']:.4f})")
    return best, df_res


# =============================================================================
# PROCESS ONE DATASET  (the full pipeline for a single prepared folder)
# =============================================================================

def process_one_dataset(ds_name):
    """
    End-to-end: load prepared data → feature select → scale → SMOTE
    → train → CV → evaluate → save.

    Returns a summary dict for the cross-dataset summary table.
    """
    prep_dir = os.path.join(PREPARED_ROOT, ds_name)
    out_dir  = os.path.join(MODEL_ROOT, ds_name)
    os.makedirs(out_dir, exist_ok=True)

    print("\n" + "="*70)
    print(f"  PROCESSING: {ds_name}")
    print(f"  Input  : {prep_dir}")
    print(f"  Output : {out_dir}")
    print("="*70)

    t0 = time.time()

    # ── Load ──
    X_train_raw, X_test_raw, y_train, y_test, full_name, binary_mode = \
        load_prepared_data(prep_dir, out_dir)

    # ── Feature selection (train only) ──
    features = select_features(X_train_raw, y_train, out_dir, top_n=TOP_FEATURES)
    X_train_sel = X_train_raw[features]
    X_test_sel  = X_test_raw[features]

    # ── Scale + SMOTE (train only) ──
    X_train, X_test_scaled, y_train_final = \
        scale_and_resample(X_train_sel, X_test_sel, y_train, features, out_dir)
    y_test = y_test.reset_index(drop=True)

    # ── Save artifacts for Script 02 ──
    joblib.dump(features,      os.path.join(out_dir, "feature_names.pkl"))
    joblib.dump(X_test_scaled, os.path.join(out_dir, "X_test.pkl"))
    joblib.dump(y_test,        os.path.join(out_dir, "y_test.pkl"))
    joblib.dump(
        X_train.sample(n=min(5000, len(X_train)), random_state=RANDOM_STATE),
        os.path.join(out_dir, "X_train_sample.pkl"))
    joblib.dump({
        "BINARY_MODE": binary_mode, "TOP_FEATURES": TOP_FEATURES,
        "RANDOM_STATE": RANDOM_STATE, "PREPARED_DIR": prep_dir,
        "DATASET_NAME": full_name,
    }, os.path.join(out_dir, "train_config.pkl"))

    # ── Train ──
    models = get_models()
    trained, train_times = train_all(models, X_train, y_train_final, out_dir)

    # ── Cross-validate ──
    cross_validate(trained, X_train, y_train_final, out_dir)

    # ── Evaluate ──
    all_metrics = evaluate_all(trained, X_test_scaled, y_test,
                               train_times, binary_mode, out_dir)
    plot_roc(trained, X_test_scaled, y_test, binary_mode, out_dir)
    best_name, df_res = compare_and_save_best(all_metrics, train_times, trained, out_dir)

    elapsed = time.time() - t0

    # ── Free memory ──
    del X_train, X_test_scaled, X_train_raw, X_test_raw, trained
    gc.collect()

    print(f"\n  ✔ {ds_name} complete in {elapsed:.1f}s")

    return {
        "Dataset": ds_name,
        "Train": len(y_train_final),
        "Test": len(y_test),
        "Features": TOP_FEATURES,
        "Best Model": best_name,
        "Best F1": df_res.loc[best_name, "F1"],
        "Best Acc": df_res.loc[best_name, "Accuracy"],
        "Time(s)": round(elapsed, 1),
    }


# =============================================================================
# MAIN
# =============================================================================

def main():
    datasets = discover_datasets()

    if not datasets:
        print(f"\n  ✗ No prepared datasets found in '{PREPARED_ROOT}/'")
        print(f"    Run 00_prepare_datasets.py first.")
        sys.exit(1)

    print("\n" + "="*70)
    print(f"  SCRIPT 01 — TRAIN MODELS (BATCH)")
    print(f"  Datasets : {len(datasets)} → {datasets}")
    print(f"  Output   : ./{MODEL_ROOT}/<dataset>/")
    print("="*70)

    t_start = time.time()
    summaries = []

    for i, ds_name in enumerate(datasets, 1):
        print(f"\n{'━'*70}")
        print(f"  [{i}/{len(datasets)}] {ds_name}")
        print(f"{'━'*70}")
        try:
            summary = process_one_dataset(ds_name)
            summaries.append(summary)
        except Exception as e:
            print(f"\n  ✗ FAILED: {ds_name} — {e}")
            import traceback; traceback.print_exc()
            summaries.append({
                "Dataset": ds_name, "Train": "-", "Test": "-",
                "Features": "-", "Best Model": "FAILED",
                "Best F1": "-", "Best Acc": "-", "Time(s)": "-",
            })

    # ── Cross-dataset summary ──
    elapsed = time.time() - t_start
    df_summary = pd.DataFrame(summaries)
    summary_path = os.path.join(MODEL_ROOT, "training_summary.csv")
    df_summary.to_csv(summary_path, index=False)

    print("\n" + "="*70)
    print("  TRAINING SUMMARY — ALL DATASETS")
    print("="*70)
    print(f"\n{df_summary.to_string(index=False)}")

    # ── Summary comparison plot ──
    valid = df_summary[df_summary["Best F1"] != "-"].copy()
    if len(valid) > 1:
        valid["Best F1"] = pd.to_numeric(valid["Best F1"])
        valid["Best Acc"] = pd.to_numeric(valid["Best Acc"])

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # F1 comparison
        axes[0].bar(valid["Dataset"], valid["Best F1"],
                    color="steelblue", edgecolor="black", alpha=0.85)
        for i, (_, row) in enumerate(valid.iterrows()):
            axes[0].text(i, row["Best F1"] + 0.005,
                         f"{row['Best F1']:.4f}\n({row['Best Model']})",
                         ha="center", fontsize=8)
        axes[0].set_title("Best F1 per Dataset", fontweight="bold")
        axes[0].set_ylabel("F1 Score"); axes[0].set_ylim(0, 1.1)
        axes[0].tick_params(axis="x", rotation=30)

        # Time comparison
        valid_time = valid[valid["Time(s)"] != "-"].copy()
        valid_time["Time(s)"] = pd.to_numeric(valid_time["Time(s)"])
        axes[1].bar(valid_time["Dataset"], valid_time["Time(s)"],
                    color="darkorange", edgecolor="black", alpha=0.85)
        axes[1].set_title("Total Training Time", fontweight="bold")
        axes[1].set_ylabel("Time (s)")
        axes[1].tick_params(axis="x", rotation=30)

        plt.suptitle("Cross-Dataset Training Summary", fontsize=14, fontweight="bold")
        plt.tight_layout()
        plt.savefig(os.path.join(MODEL_ROOT, "training_summary.png"), dpi=150)
        plt.close()
        print(f"\n  ✔ training_summary.png saved")

    print(f"""
  ╔═══════════════════════════════════════════════════════════╗
  ║  SCRIPT 01 COMPLETE ✔  ({elapsed:.1f}s total)
  ╠═══════════════════════════════════════════════════════════╣
  ║  Processed : {len([s for s in summaries if s['Best Model'] != 'FAILED'])}/{len(datasets)} datasets
  ║  Output    : ./{MODEL_ROOT}/
  ║  Summary   : {summary_path}
  ║
  ║  Next — for EACH dataset, run Script 02:
  ║""")
    for ds in datasets:
        out = os.path.join(MODEL_ROOT, ds)
        if os.path.exists(os.path.join(out, "best_model.pkl")):
            print(f"  ║    MODEL_DIR = \"{out}\"")
    print(f"""  ║
  ║    python 02_load_explain_models.py
  ╚═══════════════════════════════════════════════════════════╝
    """)


if __name__ == "__main__":
    main()
