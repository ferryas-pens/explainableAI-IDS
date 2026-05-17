# =============================================================================
# SCRIPT 2 — LOAD MODEL & EXPLAIN WITH SHAP + LIME
# =============================================================================
# Usage:
#   python 02_load_explain_models.py
#
# Prerequisites:
#   Run 01_train_save_models.py first to generate model artifacts.
#
# Two modes:
#   MODE A (default) — Explain the SAME test set saved by Script 1.
#       Set NEW_DATA_PATH = None
#
#   MODE B — Explain a NEW dataset using the saved model + scaler.
#       Set NEW_DATA_PATH = "path/to/new_dataset.csv"
#       The new data will be preprocessed, scaled with the SAME scaler
#       from training, and explained — enabling cross-dataset analysis.
#
# Outputs saved to XAI_DIR/:
#   ├── shap_summary_beeswarm.png, shap_bar.png, shap_violin.png
#   ├── shap_heatmap.png, shap_dep_<feature>.png
#   ├── shap_attack_vs_normal.png  (binary only)
#   ├── shap_waterfall_inst0~2.png, shap_contrib_inst0~2.png
#   ├── shap_decision_plot.png
#   ├── lime_inst0~4.png, lime_aggregate_importance.png
#   ├── lime_summary.csv
#   └── shap_vs_lime_comparison.png
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

INTEL_PATCHED = False
try:
    from sklearnex import patch_sklearn
    patch_sklearn()
    INTEL_PATCHED = True
    print("[INTEL] ✔ scikit-learn-intelex (oneDAL) ENABLED")
except ImportError:
    print("[INTEL] ⚠ scikit-learn-intelex not installed.")

# =============================================================================
import time
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from collections import Counter
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedShuffleSplit

import shap
import lime
import lime.lime_tabular

warnings.filterwarnings("ignore")
try:
    shap.initjs()
except Exception:
    pass
plt.style.use("seaborn-v0_8-whitegrid")


# =============================================================================
# CONFIGURATION  ← EDIT THESE
# =============================================================================

# ── Paths ──
MODEL_DIR      = "model_outputs-v5\CIC17"        # must match OUTPUT_DIR from Script 1
XAI_DIR        = "xai_outputs"

# ── Mode selection ──
# Set to None to explain the SAME test set from Script 1 (Mode A).
# Set to a CSV path to explain a NEW dataset (Mode B).
NEW_DATA_PATH  = None
# NEW_DATA_PATH = r"H:\Dataset-IDS\CIC-IDS-2018\some_other_day.csv"

# ── Which model to explain ──
# "best" loads best_model.pkl. Or specify a model name like "Random_Forest".
MODEL_TO_EXPLAIN = "best"

# ── XAI parameters ──
SHAP_MAX_ROWS  = 500
LIME_SAMPLES   = 5
LIME_AGG_N     = 100
RANDOM_STATE   = 42

# Stratified-subsampling policy
HARD_FLOOR     = 50
SOFT_FLOOR     = 200

os.makedirs(XAI_DIR, exist_ok=True)


# =============================================================================
# STRATIFIED SUBSAMPLING HELPER
# =============================================================================

def stratified_subsample_indices(
    y_stratify, n_target,
    hard_floor=HARD_FLOOR, soft_floor=SOFT_FLOOR,
    random_state=RANDOM_STATE, label="subsample",
):
    y = pd.Series(np.asarray(y_stratify)).reset_index(drop=True)
    counts = y.value_counts()
    print(f"\n  [stratified_subsample :: {label}]")
    print(f"    Input  : {len(y):,} rows, {len(counts)} classes")

    rare = counts[counts < hard_floor]
    valid_classes = counts[counts >= hard_floor].index.tolist()
    if len(rare) > 0:
        print(f"    HARD floor ({hard_floor}) drops {len(rare)} class(es):")
        for c, n in rare.items():
            print(f"      • class {c!r}: {n} samples → DROPPED")
    valid_mask = y.isin(valid_classes).values
    pos_valid  = np.where(valid_mask)[0]
    n_valid    = len(pos_valid)

    if n_valid <= n_target:
        print(f"    {n_valid:,} ≤ target {n_target:,} — no subsampling.")
        return np.sort(pos_valid), {"method": "hard_floor_only"}

    counts_valid = counts.loc[valid_classes]
    scale = n_target / counts_valid.sum()
    targets, soft_used = {}, []
    for c, n in counts_valid.items():
        proportional = int(round(int(n) * scale))
        floor_cap    = min(soft_floor, int(n))
        target       = max(proportional, floor_cap)
        if target > proportional:
            soft_used.append({"class": c, "available": int(n),
                              "proportional": proportional, "kept": target})
        targets[c] = target

    if not soft_used:
        y_valid = y.iloc[pos_valid]
        sss = StratifiedShuffleSplit(n_splits=1, train_size=n_target,
                                     random_state=random_state)
        rel_idx, _ = next(sss.split(np.zeros(n_valid), y_valid))
        keep_pos = np.sort(pos_valid[rel_idx])
        print(f"    Method  : StratifiedShuffleSplit")
    else:
        rng = np.random.RandomState(random_state)
        parts = []
        for c, target in targets.items():
            class_pos = np.where(y.values == c)[0]
            if target >= len(class_pos):
                parts.append(class_pos)
            else:
                parts.append(rng.choice(class_pos, size=target, replace=False))
        keep_pos = np.sort(np.concatenate(parts))
        print(f"    Method  : per-class stratified (soft floor × {len(soft_used)})")

    print(f"    Output : {len(keep_pos):,} rows")
    return keep_pos, {"method": "done"}


# =============================================================================
# LOADING ARTIFACTS FROM SCRIPT 1
# =============================================================================

def load_model_artifacts():
    print("\n" + "="*60)
    print("  LOADING SAVED ARTIFACTS FROM SCRIPT 1")
    print("="*60)

    # Load model
    if MODEL_TO_EXPLAIN == "best":
        model_path = os.path.join(MODEL_DIR, "best_model.pkl")
    else:
        model_path = os.path.join(MODEL_DIR, f"{MODEL_TO_EXPLAIN}.pkl")

    model  = joblib.load(model_path)
    scaler = joblib.load(os.path.join(MODEL_DIR, "scaler.pkl"))
    features = joblib.load(os.path.join(MODEL_DIR, "feature_names.pkl"))

    print(f"  ✔ Model          : {type(model).__name__}  ← {model_path}")
    print(f"  ✔ Scaler         : StandardScaler")
    print(f"  ✔ Features ({len(features)}): {features[:5]}...")

    # Class names
    try:
        le = joblib.load(os.path.join(MODEL_DIR, "label_encoder.pkl"))
        class_names = list(le.classes_)
    except Exception:
        class_names = ["Normal", "Attack"]
    print(f"  ✔ Class names    : {class_names}")

    # Config from training
    try:
        config = joblib.load(os.path.join(MODEL_DIR, "train_config.pkl"))
        binary_mode = config.get("BINARY_MODE", True)
        print(f"  ✔ Training mode  : {'Binary' if binary_mode else 'Multi-class'}")
    except Exception:
        binary_mode = len(class_names) <= 2
        print(f"  ⚠ train_config.pkl not found — inferred mode: "
              f"{'Binary' if binary_mode else 'Multi-class'}")

    return model, scaler, features, class_names, binary_mode


def load_test_data(features, scaler):
    """
    Mode A: Load saved test set from Script 1.
    Mode B: Load new CSV, preprocess, apply saved scaler + features.
    """
    if NEW_DATA_PATH is None:
        # ── MODE A: same test set ──
        print("\n  MODE A — Loading SAVED test set from Script 1")
        X_test = joblib.load(os.path.join(MODEL_DIR, "X_test.pkl"))
        y_test = joblib.load(os.path.join(MODEL_DIR, "y_test.pkl"))
        print(f"  ✔ X_test : {X_test.shape}")
        print(f"  ✔ y_test : {Counter(y_test)}")

        try:
            X_train_bg = joblib.load(os.path.join(MODEL_DIR, "X_train_sample.pkl"))
            print(f"  ✔ X_train_sample : {X_train_bg.shape} (SHAP background)")
        except Exception:
            X_train_bg = X_test.iloc[:500]
            print(f"  ⚠ X_train_sample.pkl not found — using test subset as BG")

        return X_test, y_test, X_train_bg

    else:
        # ── MODE B: new dataset ──
        print(f"\n  MODE B — Loading NEW dataset: {NEW_DATA_PATH}")
        df = pd.read_csv(NEW_DATA_PATH, low_memory=False)
        df.columns = df.columns.str.strip()
        print(f"  Raw shape : {df.shape}")

        # Preprocess
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        df.drop_duplicates(inplace=True)
        df.dropna(thresh=len(df)*0.5, axis=1, inplace=True)

        # Separate target
        TARGET_COL = "Label"
        if TARGET_COL not in df.columns:
            print(f"  ⚠ '{TARGET_COL}' column not found — creating dummy labels (all 0)")
            y_new = pd.Series(np.zeros(len(df), dtype=np.int32))
        else:
            y_raw = df[TARGET_COL].astype(str).str.strip()
            # Try to load label encoder from training to apply consistent encoding
            try:
                le = joblib.load(os.path.join(MODEL_DIR, "label_encoder.pkl"))
                n_classes = len(le.classes_)
                if n_classes <= 2:
                    # Binary mode
                    y_new = y_raw.apply(
                        lambda x: 0 if x.lower() in ["benign", "normal"] else 1
                    ).astype(np.int32)
                else:
                    # Multi-class — map known labels, assign -1 to unknown
                    known = {c: i for i, c in enumerate(le.classes_)}
                    y_new = y_raw.map(known).fillna(-1).astype(np.int32)
                    n_unknown = (y_new == -1).sum()
                    if n_unknown > 0:
                        print(f"  ⚠ {n_unknown} samples have labels not seen in training")
            except Exception:
                y_new = y_raw.apply(
                    lambda x: 0 if x.lower() in ["benign", "normal"] else 1
                ).astype(np.int32)

            df.drop(columns=[TARGET_COL], inplace=True)

        # Select features + fill missing
        X = df.select_dtypes(include=np.number)
        missing = [f for f in features if f not in X.columns]
        if missing:
            print(f"  ⚠ {len(missing)} feature(s) missing in new data — filling with 0:")
            print(f"    {missing}")
            for f in missing:
                X[f] = 0.0
        X = X[features]
        X = X.fillna(X.median())

        # Scale using SAVED scaler from training
        X_scaled = pd.DataFrame(
            scaler.transform(X),
            columns=features
        ).reset_index(drop=True)
        y_new = y_new.reset_index(drop=True)

        print(f"  ✔ New data scaled : {X_scaled.shape}")
        print(f"  ✔ Class dist      : {Counter(y_new)}")

        # Background data for SHAP
        try:
            X_train_bg = joblib.load(os.path.join(MODEL_DIR, "X_train_sample.pkl"))
        except Exception:
            X_train_bg = X_scaled.iloc[:500]

        return X_scaled, y_new, X_train_bg


# =============================================================================
# SHAP EXPLANATIONS
# =============================================================================

def get_shap_explainer(model, X_bg):
    model_type = type(model).__name__
    tree_models = {"RandomForestClassifier", "DecisionTreeClassifier",
                   "GradientBoostingClassifier", "XGBClassifier", "LGBMClassifier"}
    if model_type in tree_models:
        print(f"  Using TreeExplainer for {model_type}")
        return shap.TreeExplainer(model)
    else:
        print(f"  Using KernelExplainer for {model_type} (slower)")
        bg = shap.sample(X_bg, 100, random_state=RANDOM_STATE)
        return shap.KernelExplainer(model.predict_proba, bg)


def compute_shap_values(explainer, X_sample, n_classes):
    sv = explainer.shap_values(X_sample)
    if isinstance(sv, list):
        if len(sv) == 2:
            vals = np.array(sv[1])
        else:
            sv_stack = np.stack([np.array(s) for s in sv], axis=-1)
            vals = np.abs(sv_stack).mean(axis=-1)
    else:
        vals = np.array(sv)
    if vals.ndim == 3: vals = vals[:, :, 0]
    if vals.ndim == 1: vals = vals.reshape(1, -1)

    ev = explainer.expected_value
    if isinstance(ev, (list, np.ndarray)):
        ev_arr = np.array(ev).flatten()
        if len(ev_arr) == 2:     ev = float(ev_arr[1])
        elif len(ev_arr) > 2:    ev = float(ev_arr.mean())
        else:                    ev = float(ev_arr[0])
    else:
        ev = float(ev)
    return vals, ev


def run_shap_explanations(model, X_test, y_test, features, class_names, binary_mode):
    model_name = type(model).__name__
    n_classes  = len(class_names)

    print("\n" + "="*60)
    print(f"  SHAP GLOBAL EXPLANATIONS  ({n_classes}-class)")
    print("="*60)

    # Stratified SHAP sample
    if len(X_test) <= SHAP_MAX_ROWS:
        X_df = X_test.reset_index(drop=True)
        y_s  = y_test.reset_index(drop=True)
    else:
        keep, _ = stratified_subsample_indices(y_test, SHAP_MAX_ROWS, label="SHAP sample")
        X_df = X_test.iloc[keep].reset_index(drop=True)
        y_s  = y_test.iloc[keep].reset_index(drop=True)

    X_np = X_df.values.astype(np.float64)
    feat = list(X_df.columns)
    print(f"  Sample : {X_np.shape} | Features : {len(feat)}")

    explainer = get_shap_explainer(model, X_df)
    sv, ev = compute_shap_values(explainer, X_np, n_classes)
    print(f"  SHAP   : {sv.shape} | Base value : {ev:.4f}")

    # Dimension guard
    n_sv = sv.shape[1]
    if len(feat) != n_sv:
        print(f"  ⚠ Trimming features {len(feat)} → {n_sv}")
        feat = feat[:n_sv]; X_np = X_np[:, :n_sv]

    mean_abs = np.abs(sv).mean(axis=0).flatten()

    # ── Beeswarm ──
    plt.figure(figsize=(10, 7))
    shap.summary_plot(sv, X_np, feature_names=feat, plot_type="dot", show=False)
    plt.title(f"SHAP Summary — {model_name}", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(XAI_DIR, "shap_summary_beeswarm.png"), dpi=150, bbox_inches="tight")
    plt.show(); print("  ✔ Beeswarm")

    # ── Bar ──
    plt.figure(figsize=(10, 6))
    shap.summary_plot(sv, X_np, feature_names=feat, plot_type="bar", show=False)
    plt.title(f"SHAP Feature Importance — {model_name}", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(XAI_DIR, "shap_bar.png"), dpi=150, bbox_inches="tight")
    plt.show(); print("  ✔ Bar")

    # ── Violin ──
    plt.figure(figsize=(10, 7))
    shap.summary_plot(sv, X_np, feature_names=feat, plot_type="violin", show=False)
    plt.title(f"SHAP Violin — {model_name}", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(XAI_DIR, "shap_violin.png"), dpi=150, bbox_inches="tight")
    plt.show(); print("  ✔ Violin")

    # ── Heatmap ──
    n_top = min(15, len(mean_abs))
    top_idx = np.argsort(mean_abs)[-n_top:][::-1].tolist()
    plt.figure(figsize=(14, 6))
    sns.heatmap(sv[:100][:, top_idx].T, xticklabels=False,
                yticklabels=[feat[i] for i in top_idx],
                cmap="RdBu_r", center=0, linewidths=0.3)
    plt.title(f"SHAP Heatmap — {model_name}", fontsize=12, fontweight="bold")
    plt.xlabel("Samples"); plt.ylabel("Features"); plt.tight_layout()
    plt.savefig(os.path.join(XAI_DIR, "shap_heatmap.png"), dpi=150, bbox_inches="tight")
    plt.show(); print("  ✔ Heatmap")

    # ── Dependence (top 3) ──
    dep_idx = np.argsort(mean_abs)[-3:][::-1].tolist()
    for idx in dep_idx:
        plt.figure(figsize=(8, 5))
        shap.dependence_plot(idx, sv, X_np, feature_names=feat, show=False)
        safe_name = feat[idx].replace(' ','_').replace('/','_')
        plt.title(f"SHAP Dependence — {feat[idx]}", fontsize=12, fontweight="bold")
        plt.tight_layout()
        plt.savefig(os.path.join(XAI_DIR, f"shap_dep_{safe_name}.png"), dpi=150)
        plt.show(); print(f"  ✔ Dependence: {feat[idx]}")

    # ── Attack vs Normal (binary only) ──
    if binary_mode:
        y_arr = y_s.values[:len(sv)]
        idx_a = np.where(y_arr == 1)[0]
        idx_n = np.where(y_arr == 0)[0]
        m_a = np.abs(sv[idx_a]).mean(axis=0).flatten() if len(idx_a) else np.zeros(len(feat))
        m_n = np.abs(sv[idx_n]).mean(axis=0).flatten() if len(idx_n) else np.zeros(len(feat))
        cmp = pd.DataFrame({"Feature": feat, "Attack": m_a, "Normal": m_n}) \
                .sort_values("Attack", ascending=False).head(15)
        x_ = np.arange(len(cmp)); w_ = 0.35
        plt.figure(figsize=(12, 6))
        plt.bar(x_-w_/2, cmp["Attack"], w_, label="Attack", color="crimson",
                edgecolor="black", alpha=0.85)
        plt.bar(x_+w_/2, cmp["Normal"], w_, label="Normal", color="steelblue",
                edgecolor="black", alpha=0.85)
        plt.xticks(x_, cmp["Feature"], rotation=45, ha="right")
        plt.ylabel("Mean |SHAP|")
        plt.title(f"Attack vs Normal — {model_name}", fontsize=13, fontweight="bold")
        plt.legend(); plt.tight_layout()
        plt.savefig(os.path.join(XAI_DIR, "shap_attack_vs_normal.png"), dpi=150)
        plt.show(); print("  ✔ Attack vs Normal")

    # ── Local explanations (waterfall + contribution) ──
    print("\n  SHAP LOCAL EXPLANATIONS")
    for i in range(min(3, len(sv))):
        s_sv   = np.array(sv[i]).flatten()
        s_data = np.array(X_np[i]).flatten()
        n_a    = min(len(feat), len(s_sv), len(s_data))
        _f, _s, _d = feat[:n_a], s_sv[:n_a], s_data[:n_a]
        true_cls = int(y_s.iloc[i])
        label = class_names[true_cls] if true_cls < len(class_names) else f"Class_{true_cls}"

        try:
            exp = shap.Explanation(values=_s, base_values=ev, data=_d, feature_names=_f)
            plt.figure(figsize=(10, 5))
            shap.waterfall_plot(exp, show=False)
            plt.title(f"Waterfall #{i} | True: {label}", fontsize=11, fontweight="bold")
            plt.tight_layout()
            plt.savefig(os.path.join(XAI_DIR, f"shap_waterfall_inst{i}.png"),
                        dpi=150, bbox_inches="tight")
            plt.show(); print(f"  ✔ Waterfall #{i}")
        except Exception as e:
            print(f"  ⚠ Waterfall #{i} skipped: {e}")

        cdf = pd.DataFrame({"Feature": _f, "SHAP": _s, "Value": _d}) \
                .sort_values("SHAP", key=abs, ascending=False).head(15)
        colors = ["crimson" if v > 0 else "steelblue" for v in cdf["SHAP"]]
        fig, ax = plt.subplots(figsize=(10, 6))
        bars = ax.barh(cdf["Feature"][::-1], cdf["SHAP"][::-1],
                       color=colors[::-1], edgecolor="black", alpha=0.85)
        ax.axvline(0, color="black", lw=0.8, ls="--")
        ax.set_xlabel("SHAP Value")
        ax.set_title(f"Contribution #{i} | True: {label} | "
                     f"base={ev:.3f} → pred={ev+_s.sum():.3f}",
                     fontsize=10, fontweight="bold")
        plt.tight_layout()
        plt.savefig(os.path.join(XAI_DIR, f"shap_contrib_inst{i}.png"),
                    dpi=150, bbox_inches="tight")
        plt.show(); print(f"  ✔ Contribution #{i}")

    # Decision plot
    try:
        plt.figure(figsize=(10, 8))
        shap.decision_plot(ev, sv[:10], feature_names=feat, show=False)
        plt.title(f"SHAP Decision Plot — {model_name}", fontsize=12, fontweight="bold")
        plt.tight_layout()
        plt.savefig(os.path.join(XAI_DIR, "shap_decision_plot.png"), dpi=150)
        plt.show(); print("  ✔ Decision plot")
    except Exception as e:
        print(f"  ⚠ Decision plot skipped: {e}")

    return sv, ev, feat


# =============================================================================
# LIME EXPLANATIONS
# =============================================================================

def run_lime_explanations(model, X_train_bg, X_test, y_test, features, class_names):
    model_name = type(model).__name__

    print("\n" + "="*60)
    print(f"  LIME EXPLANATIONS  ({len(class_names)} classes)")
    print("="*60)

    explainer = lime.lime_tabular.LimeTabularExplainer(
        training_data=X_train_bg.values,
        feature_names=features,
        class_names=class_names,
        mode="classification",
        discretize_continuous=True,
        random_state=RANDOM_STATE
    )

    lime_records = []
    for i in range(min(LIME_SAMPLES, len(X_test))):
        instance = X_test.iloc[i].values.reshape(1, -1)
        true_idx = int(y_test.iloc[i])
        pred_idx = int(model.predict(instance)[0])
        pred_prob = model.predict_proba(instance)[0]
        true_str = class_names[true_idx] if true_idx < len(class_names) else f"Class_{true_idx}"
        pred_str = class_names[pred_idx] if pred_idx < len(class_names) else f"Class_{pred_idx}"
        conf = float(np.max(pred_prob)) * 100

        exp = explainer.explain_instance(
            instance.flatten(), model.predict_proba,
            num_features=10, top_labels=1)
        feats = exp.as_list(label=exp.top_labels[0])

        print(f"\n  ── Instance #{i} | True: {true_str} | Pred: {pred_str} ({conf:.1f}%) ──")
        for ft, wt in feats:
            print(f"    {ft:45s} {wt:+.4f}")

        lime_records.append({
            "Instance": i, "True": true_str, "Predicted": pred_str,
            "Confidence": round(conf, 2),
            "Top Feature": feats[0][0] if feats else "N/A",
            "Top Weight": feats[0][1] if feats else 0,
        })

        fig, ax = plt.subplots(figsize=(10, 6))
        colors = ["crimson" if w > 0 else "steelblue" for _, w in feats]
        labels_ = [f[0] for f in feats]; weights_ = [f[1] for f in feats]
        ax.barh(labels_[::-1], weights_[::-1], color=colors[::-1],
                edgecolor="black", alpha=0.85)
        ax.axvline(0, color="black", lw=0.8, ls="--")
        ax.set_xlabel(f"LIME Weight (red → {pred_str} | blue → other)")
        ax.set_title(f"LIME — {model_name} #{i}\nTrue: {true_str} | Pred: "
                     f"{pred_str} | Conf: {conf:.1f}%", fontsize=11, fontweight="bold")
        plt.tight_layout()
        plt.savefig(os.path.join(XAI_DIR, f"lime_inst{i}.png"), dpi=150, bbox_inches="tight")
        plt.show(); print(f"  ✔ lime_inst{i}.png")

    df_lime = pd.DataFrame(lime_records)
    df_lime.to_csv(os.path.join(XAI_DIR, "lime_summary.csv"), index=False)
    print(f"\n{df_lime.to_string(index=False)}")

    # ── Aggregate LIME (stratified) ──
    print(f"\n  [LIME] Aggregating over {LIME_AGG_N} samples (stratified)...")
    imp_sum = {f: 0.0 for f in features}
    if len(X_test) <= LIME_AGG_N:
        sample = X_test.copy()
    else:
        keep, _ = stratified_subsample_indices(y_test, LIME_AGG_N, label="LIME aggregate")
        sample = X_test.iloc[keep]

    for idx, (_, row) in enumerate(sample.iterrows()):
        try:
            exp = explainer.explain_instance(
                row.values, model.predict_proba,
                num_features=len(features), top_labels=1)
            feats = exp.as_list(label=exp.top_labels[0])
            for ft, wt in feats:
                for fn in features:
                    if fn in ft:
                        imp_sum[fn] += abs(wt); break
        except Exception:
            pass
        if (idx + 1) % 20 == 0:
            print(f"    {idx + 1}/{len(sample)} ...")

    lime_imp = pd.Series(imp_sum).sort_values(ascending=False).head(20)
    plt.figure(figsize=(10, 6))
    lime_imp.sort_values().plot(kind="barh", color="darkcyan",
                                edgecolor="black", alpha=0.85)
    plt.title(f"LIME Aggregated Importance — {model_name}", fontsize=12, fontweight="bold")
    plt.xlabel("Sum |LIME Weights|"); plt.tight_layout()
    plt.savefig(os.path.join(XAI_DIR, "lime_aggregate_importance.png"), dpi=150)
    plt.show(); print("  ✔ Aggregate importance")

    return df_lime, lime_imp


# =============================================================================
# SHAP vs LIME COMPARISON + REPORT
# =============================================================================

def shap_vs_lime_comparison(sv, feature_names, lime_imp, model_name):
    print("\n" + "="*60)
    print("  SHAP vs LIME COMPARISON")
    print("="*60)
    n = min(len(feature_names), sv.shape[1])
    shap_imp = pd.Series(np.abs(sv[:, :n]).mean(axis=0).flatten(),
                         index=feature_names[:n])
    shap_n = shap_imp / shap_imp.max()
    lime_n = lime_imp / lime_imp.max() if lime_imp.max() > 0 else lime_imp
    top15  = shap_n.nlargest(15).index
    df_c = pd.DataFrame({
        "SHAP": shap_n[top15],
        "LIME": lime_n.reindex(top15).fillna(0)
    }).sort_values("SHAP", ascending=True)
    fig, ax = plt.subplots(figsize=(10, 7))
    x = np.arange(len(df_c)); w = 0.35
    ax.barh(x-w/2, df_c["SHAP"], w, label="SHAP",
            color="steelblue", edgecolor="black", alpha=0.85)
    ax.barh(x+w/2, df_c["LIME"], w, label="LIME",
            color="darkorange", edgecolor="black", alpha=0.85)
    ax.set_yticks(x); ax.set_yticklabels(df_c.index)
    ax.set_xlabel("Normalized Importance"); ax.legend()
    ax.set_title(f"SHAP vs LIME — {model_name}", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(XAI_DIR, "shap_vs_lime_comparison.png"),
                dpi=150, bbox_inches="tight")
    plt.show(); print("  ✔ Saved.")


def print_xai_report(sv, feature_names, lime_df, lime_imp, model_name, class_names):
    print("\n" + "="*60)
    print("  XAI SUMMARY REPORT")
    print("="*60)
    n = min(len(feature_names), sv.shape[1])
    shap_imp = pd.Series(np.abs(sv[:, :n]).mean(axis=0).flatten(),
                         index=feature_names[:n]).sort_values(ascending=False)
    print(f"\n  Model         : {model_name}")
    print(f"  Classes       : {class_names}")
    print(f"  SHAP Samples  : {len(sv)}")
    print(f"  LIME Instances: {len(lime_df)}")

    print(f"\n  ── Top 10 Features by SHAP ──")
    for rank, (ft, val) in enumerate(shap_imp.head(10).items(), 1):
        bar = "█" * int(val * 50)
        print(f"  {rank:2}. {ft:35s} {val:.4f}  {bar}")

    print(f"\n  ── Top 10 Features by LIME ──")
    lime_sorted = lime_imp.sort_values(ascending=False)
    for rank, (ft, val) in enumerate(lime_sorted.head(10).items(), 1):
        bar = "█" * int((val / lime_sorted.max()) * 50) if lime_sorted.max() > 0 else ""
        print(f"  {rank:2}. {ft:35s} {val:.4f}  {bar}")

    print(f"\n  ── LIME Instance Results ──")
    print(lime_df.to_string(index=False))
    correct = (lime_df["True"] == lime_df["Predicted"]).sum()
    print(f"\n  Correct: {correct}/{len(lime_df)} ({correct/len(lime_df)*100:.1f}%)")

    # Top-k overlap
    for k in [5, 10]:
        s_topk = set(shap_imp.head(k).index)
        l_topk = set(lime_sorted.head(k).index)
        overlap = s_topk & l_topk
        print(f"  Top-{k} overlap: {len(overlap)}/{k} → "
              f"{', '.join(sorted(overlap)) if overlap else 'none'}")

    # Spearman
    try:
        from scipy.stats import spearmanr
        common = sorted(set(shap_imp.index) & set(lime_imp.index))
        if len(common) >= 5:
            rho, pval = spearmanr(
                shap_imp[common].rank(ascending=False),
                lime_imp.reindex(common).fillna(0).rank(ascending=False))
            strength = "Strong" if abs(rho) > 0.7 else ("Moderate" if abs(rho) > 0.4 else "Weak")
            print(f"\n  Spearman rho : {rho:.4f} (p={pval:.4e}) — {strength} consistency")
    except ImportError:
        print("\n  ⚠ scipy not installed — Spearman skipped")

    print("\n" + "="*60)
    print(f"  All XAI outputs → ./{XAI_DIR}/")
    print("="*60)


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("\n" + "="*70)
    print("  SCRIPT 2 — LOAD MODEL & EXPLAIN (SHAP + LIME)")
    print(f"  Mode: {'NEW dataset' if NEW_DATA_PATH else 'SAVED test set'}")
    print("="*70)
    t0 = time.time()

    # ── Load artifacts from Script 1 ──
    model, scaler, features, class_names, binary_mode = load_model_artifacts()

    # ── Load data (Mode A or Mode B) ──
    X_test, y_test, X_train_bg = load_test_data(features, scaler)

    # ── SHAP ──
    sv, ev, feat = run_shap_explanations(
        model, X_test, y_test, features, class_names, binary_mode)

    # ── LIME ──
    lime_df, lime_imp = run_lime_explanations(
        model, X_train_bg, X_test, y_test, features, class_names)

    # ── Compare ──
    shap_vs_lime_comparison(sv, feat, lime_imp, type(model).__name__)

    # ── Report ──
    print_xai_report(sv, feat, lime_df, lime_imp, type(model).__name__, class_names)

    elapsed = time.time() - t0
    mode_str = f"NEW: {NEW_DATA_PATH}" if NEW_DATA_PATH else "SAVED test set"
    print(f"""
  ╔═══════════════════════════════════════════════════════════╗
  ║  SCRIPT 2 COMPLETE ✔  ({elapsed:.1f}s)
  ╠═══════════════════════════════════════════════════════════╣
  ║  Data source : {mode_str[:50]}
  ║  Model       : {type(model).__name__}
  ║  XAI outputs : ./{XAI_DIR}/
  ║
  ║  To explain a DIFFERENT dataset, set:
  ║    NEW_DATA_PATH = "path/to/other_dataset.csv"
  ║  and re-run this script.
  ╚═══════════════════════════════════════════════════════════╝
    """)


if __name__ == "__main__":
    main()
