# =============================================================================
# SCRIPT 00 — PREPARE DATASETS  (Load → Clean → Sample → Split → Save)
# =============================================================================
# Purpose:
#   Heavy I/O script that runs ONCE per dataset. Reads multi-GB CSVs,
#   preprocesses, takes a stratified ~10% sample, performs train/test split,
#   and saves lightweight .pkl partitions for downstream scripts.
#
# Usage:
#   python 00_prepare_datasets.py
#
# Then for each prepared dataset:
#   python 01_train_save_models.py --data prepared_data/CIC18_TUESDAY
#   python 02_load_explain_models.py
#
# Output structure:
#   prepared_data/
#   ├── CIC17/
#   │   ├── X_train_raw.pkl      (pre-selected features, unscaled)
#   │   ├── X_test_raw.pkl
#   │   ├── y_train.pkl
#   │   ├── y_test.pkl
#   │   ├── train.csv             (features + label, human-readable)
#   │   ├── test.csv
#   │   ├── label_encoder.pkl
#   │   ├── all_numeric_cols.pkl  (full feature list before selection)
#   │   ├── prep_report.pkl       (metadata: distributions, sampling log)
#   │   └── prep_report.json
#   ├── CIC18_TUESDAY/
#   │   └── ...
#   └── ...
# =============================================================================

import os
import sys
import time
import warnings
import re
import numpy as np
import pandas as pd
import joblib
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from collections import Counter, OrderedDict
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings("ignore")
plt.style.use("seaborn-v0_8-whitegrid")


# =============================================================================
# CONFIGURATION  ← EDIT THESE FOR YOUR ENVIRONMENT
# =============================================================================

# ── Dataset registry ──
# Each entry: "SHORT_NAME" → path (single CSV or folder of CSVs)
DATASETS = OrderedDict([
    # CIC-IDS-2017  (1 merged CSV, ~2.8M rows)
 # CIC-IDS-2017: path may be one CSV or a folder containing CSV files.
 ("CIC17", r"C:\Users\hduser\Downloads\cicids2017"),

 # CIC-IDS-2018: each day/file is prepared separately.
  ("CIC18_TUESDAY",
   r"E:\cicids2018\all\Tuesday-20-02-2018_TrafficForML_CICFlowMeter.csv"),
 ("CIC18_WEDNESDAY-1",
  r"E:\cicids2018\all\Wednesday-14-02-2018_TrafficForML_CICFlowMeter.csv"),
 ("CIC18_WEDNESDAY-2",
  r"E:\cicids2018\all\Wednesday-21-02-2018_TrafficForML_CICFlowMeter.csv"),
 ("CIC18_WEDNESDAY-3",
  r"E:\cicids2018\all\Wednesday-28-02-2018_TrafficForML_CICFlowMeter.csv"),
 ("CIC18_THURSDAY-1",
  r"E:\cicids2018\all\Thursday-15-02-2018_TrafficForML_CICFlowMeter.csv"),
 ("CIC18_THURSDAY-2",
  r"E:\cicids2018\all\Thursday-22-02-2018_TrafficForML_CICFlowMeter.csv"),
 ("CIC18_THURSDAY-3",
  r"E:\cicids2018\all\Thursday-01-03-2018_TrafficForML_CICFlowMeter.csv"),
 ("CIC18_FRIDAY-1",
  r"E:\cicids2018\all\Friday-16-02-2018_TrafficForML_CICFlowMeter.csv"),
 ("CIC18_FRIDAY-2",
  r"E:\cicids2018\all\Friday-23-02-2018_TrafficForML_CICFlowMeter.csv"),
 ("CIC18_FRIDAY-3",
  r"E:\cicids2018\all\Friday-02-03-2018_TrafficForML_CICFlowMeter-del56776.csv"),

    # Uncomment and add more as needed:
    # ("CIC19", r"path/to/CIC-IDS-2019.csv"),
    # ("LycoS17", r"path/to/LycoS-IDS2017.csv"),
    # ("LycoS18", r"path/to/LycoS-Unicas-IDS2018.csv"),
])

# ── Global settings ──
TARGET_COL     = "Label"
OUTPUT_ROOT    = "prepared_data"       # base folder for all prepared datasets
BINARY_MODE    = True                  # True = Normal/Attack | False = Multi-class
TEST_SIZE      = 0.2                   # 20% held out for testing
RANDOM_STATE   = 42

# ── Sampling settings ──
SAMPLE_FRAC    = 0.10                  # keep ~10% of each dataset (stratified)
MAX_ROWS       = 300_000              # hard cap after sampling (safety net)
HARD_FLOOR     = 50                    # drop classes with fewer than this
SOFT_FLOOR     = 200                   # protect minority classes above this

# ── Columns to exclude (metadata, not features) ──
EXCLUDE_COLS = [
    "Flow ID", "Source IP", "Src IP", "Source Port", "Src Port",
    "Destination IP", "Dst IP", "Destination Port", "Dst Port",
    "Protocol", "Timestamp", "Fwd Header Length.1",
]

# ── Reference feature schema alignment ──
# Purpose:
#   Different CICFlowMeter/CIC-IDS releases often use slightly different
#   feature names, for example "Tot Fwd Pkts" vs "Total Fwd Packets".
#   These settings make Script 00 normalize feature names and then force every
#   prepared dataset to use the same columns/order as the reference dataset.
ENFORCE_REFERENCE_SCHEMA       = True
REFERENCE_DATASET_NAME         = "CIC17"   # first dataset used as feature reference
REFERENCE_SCHEMA_FILE          = os.path.join(OUTPUT_ROOT, "_reference_numeric_cols.pkl")
MISSING_REFERENCE_FILL_VALUE   = 0.0       # value inserted when a reference feature is absent
DROP_NON_REFERENCE_FEATURES    = True      # drop features that are not in the reference schema

# Common aliases across CIC-IDS-2017 / CSE-CIC-IDS-2018 / CICFlowMeter variants.
# Keys may use any spacing/case; they are internally compacted before matching.
COLUMN_RENAME_ALIASES_RAW = {
    # Target / label
    "label": TARGET_COL,
    "class": TARGET_COL,
    "classification": TARGET_COL,
    "attack label": TARGET_COL,

    # Packet/byte counters
    "Tot Fwd Pkts": "Total Fwd Packets",
    "Tot Bwd Pkts": "Total Backward Packets",
    "Total Bwd Packets": "Total Backward Packets",
    "TotLen Fwd Pkts": "Total Length of Fwd Packets",
    "TotLen Bwd Pkts": "Total Length of Bwd Packets",
    "Fwd Pkt Len Max": "Fwd Packet Length Max",
    "Fwd Pkt Len Min": "Fwd Packet Length Min",
    "Fwd Pkt Len Mean": "Fwd Packet Length Mean",
    "Fwd Pkt Len Std": "Fwd Packet Length Std",
    "Bwd Pkt Len Max": "Bwd Packet Length Max",
    "Bwd Pkt Len Min": "Bwd Packet Length Min",
    "Bwd Pkt Len Mean": "Bwd Packet Length Mean",
    "Bwd Pkt Len Std": "Bwd Packet Length Std",
    "Flow Byts/s": "Flow Bytes/s",
    "Flow Pkts/s": "Flow Packets/s",

    # Inter-arrival time
    "Fwd IAT Tot": "Fwd IAT Total",
    "Bwd IAT Tot": "Bwd IAT Total",

    # Packet length statistics
    "Pkt Len Min": "Min Packet Length",
    "Pkt Len Max": "Max Packet Length",
    "Pkt Len Mean": "Packet Length Mean",
    "Pkt Len Std": "Packet Length Std",
    "Pkt Len Var": "Packet Length Variance",

    # Flags
    "FIN Flag Cnt": "FIN Flag Count",
    "SYN Flag Cnt": "SYN Flag Count",
    "RST Flag Cnt": "RST Flag Count",
    "PSH Flag Cnt": "PSH Flag Count",
    "ACK Flag Cnt": "ACK Flag Count",
    "URG Flag Cnt": "URG Flag Count",
    "CWE Flag Cnt": "CWE Flag Count",
    "ECE Flag Cnt": "ECE Flag Count",

    # Segment / bulk / subflow
    "Fwd Header Len": "Fwd Header Length",
    "Bwd Header Len": "Bwd Header Length",
    "Fwd Pkts/s": "Fwd Packets/s",
    "Bwd Pkts/s": "Bwd Packets/s",
    "Pkt Size Avg": "Average Packet Size",
    "Fwd Seg Size Avg": "Avg Fwd Segment Size",
    "Bwd Seg Size Avg": "Avg Bwd Segment Size",
    "Fwd Byts/b Avg": "Fwd Avg Bytes/Bulk",
    "Fwd Pkts/b Avg": "Fwd Avg Packets/Bulk",
    "Fwd Blk Rate Avg": "Fwd Avg Bulk Rate",
    "Bwd Byts/b Avg": "Bwd Avg Bytes/Bulk",
    "Bwd Pkts/b Avg": "Bwd Avg Packets/Bulk",
    "Bwd Blk Rate Avg": "Bwd Avg Bulk Rate",
    "Subflow Fwd Pkts": "Subflow Fwd Packets",
    "Subflow Fwd Byts": "Subflow Fwd Bytes",
    "Subflow Bwd Pkts": "Subflow Bwd Packets",
    "Subflow Bwd Byts": "Subflow Bwd Bytes",

    # Window / active data / min segment
    "Init Fwd Win Byts": "Init_Win_bytes_forward",
    "Init Bwd Win Byts": "Init_Win_bytes_backward",
    "Fwd Act Data Pkts": "act_data_pkt_fwd",
    "Fwd Seg Size Min": "min_seg_size_forward",
}

os.makedirs(OUTPUT_ROOT, exist_ok=True)

# Runtime cache for the reference schema.
REFERENCE_FEATURES = None



# =============================================================================
# COLUMN NAME NORMALIZATION + REFERENCE FEATURE ALIGNMENT
# =============================================================================

def _clean_col_name(col):
    """Trim BOM/whitespace and collapse repeated spaces without changing meaning."""
    return " ".join(str(col).replace("\ufeff", "").strip().split())


def _compact_col_name(col):
    """Case/spacing/punctuation-insensitive key for matching column aliases."""
    return re.sub(r"[^a-z0-9]+", "", str(col).lower())


# Known CIC/CSE-CIC label values used for value-based target recovery.
# This makes the script robust when a folder contains mixed/dirty headers where
# the real label column is not exactly named "Label" after concat.
KNOWN_LABEL_VALUES = {
    "benign", "normal",
    "bot", "ddos", "dos hulk", "dos goldeneye", "dos slowhttptest", "dos slowloris",
    "heartbleed", "infiltration", "infilteration", "portscan",
    "ftp-patator", "ssh-patator", "web attack brute force", "web attack xss",
    "web attack sql injection",
    "brute force -web", "brute force -xss", "sql injection",
    "dos attacks-hulk", "dos attacks-goldeneye", "dos attacks-slowhttptest",
    "dos attacks-slowloris", "ddos attacks-loic-http", "ddos attack-hoic",
    "ddos attack-loic-udp", "ftp-bruteforce", "ssh-bruteforce",
}

LABEL_NAME_KEYS = {
    _compact_col_name(TARGET_COL),
    _compact_col_name(" Label"),
    _compact_col_name("label"),
    _compact_col_name("class"),
    _compact_col_name("classification"),
    _compact_col_name("attack label"),
    _compact_col_name("attacklabel"),
}

MISSING_LABEL_STRINGS = {"", "nan", "none", "null", "na", "n/a", "?"}


def _as_clean_label_text(series):
    """Return stripped text labels while preserving missing values as <NA>."""
    out = series.astype("string").str.replace("\ufeff", "", regex=False).str.strip()
    out = out.mask(out.str.lower().isin(MISSING_LABEL_STRINGS))
    return out


def _score_label_values(series, sample_size=20000):
    """
    Heuristic score for detecting a CIC label column by its values.

    A real label column normally has many repeated string labels, relatively few
    unique values, and most non-empty values are known CIC/CSE-CIC class names.
    """
    if len(series) == 0:
        return 0.0, 0, 0

    s = series
    if len(s) > sample_size:
        s = s.sample(sample_size, random_state=RANDOM_STATE)

    txt = _as_clean_label_text(s).dropna()
    n_valid = len(txt)
    if n_valid == 0:
        return 0.0, 0, 0

    lower = txt.str.lower()
    unique_count = int(lower.nunique(dropna=True))
    known_ratio = float(lower.isin(KNOWN_LABEL_VALUES).mean())

    # Avoid mistaking high-cardinality textual metadata for labels.
    low_cardinality_bonus = 1.0 if unique_count <= 100 else max(0.0, 100.0 / unique_count)
    score = known_ratio * low_cardinality_bonus
    return score, n_valid, unique_count


def recover_target_column(df, dataset_name):
    """
    Create/protect TARGET_COL using both header-based and value-based detection.

    Why this is needed:
    - CIC-IDS-2017 folder mode may contain dirty/mixed headers.
    - After concatenation, the actual labels can land in a different column than
      TARGET_COL, while TARGET_COL itself is empty for many/all rows.
    - We therefore recover labels before feature alias renaming and before any
      missing-label row filtering.
    """
    cols = list(df.columns)
    candidate_positions = []
    candidate_info = []

    # 1) Header/name based candidates.
    for idx, col in enumerate(cols):
        key = _compact_col_name(col)
        # Accept exact label aliases and common Pandas-mangled forms like Label.1.
        looks_like_label_name = (
            key in LABEL_NAME_KEYS
            or key.startswith("label")
            or key in {"class", "classification"}
        )
        if looks_like_label_name:
            candidate_positions.append(idx)
            candidate_info.append({
                "column": str(col),
                "position": idx,
                "reason": "name_match",
            })

    # 2) Value-based candidates. This catches cases where the header is corrupted
    # but the values clearly contain CIC labels such as BENIGN, DoS Hulk, etc.
    for idx, col in enumerate(cols):
        if idx in candidate_positions:
            continue
        # Prioritize object/string columns, but still allow mixed columns because
        # pandas may infer object when folders have inconsistent headers.
        if not (pd.api.types.is_object_dtype(df.iloc[:, idx]) or pd.api.types.is_string_dtype(df.iloc[:, idx])):
            continue
        score, n_valid, unique_count = _score_label_values(df.iloc[:, idx])
        if score >= 0.60 and n_valid > 0:
            candidate_positions.append(idx)
            candidate_info.append({
                "column": str(col),
                "position": idx,
                "reason": "value_match",
                "score": round(score, 4),
                "sample_non_empty": int(n_valid),
                "sample_unique": int(unique_count),
            })

    if not candidate_positions:
        print("    Target recovery          : no label-like column detected yet")
        return df, {
            "target_col": TARGET_COL,
            "candidate_columns": [],
            "non_empty_labels_after_recovery": 0,
            "status": "not_found",
        }

    # Merge all candidate columns row-wise. Empty strings and textual nulls are
    # treated as missing, then bfill selects the first available label per row.
    label_block = pd.concat(
        [_as_clean_label_text(df.iloc[:, pos]) for pos in candidate_positions],
        axis=1,
    )
    recovered_label = label_block.bfill(axis=1).iloc[:, 0]

    # Remove all candidate columns first, then insert one protected TARGET_COL.
    keep_positions = [i for i in range(len(cols)) if i not in set(candidate_positions)]
    df_out = df.iloc[:, keep_positions].copy()
    df_out[TARGET_COL] = recovered_label.values

    non_empty = int(_as_clean_label_text(df_out[TARGET_COL]).notna().sum())
    unique_preview = (
        _as_clean_label_text(df_out[TARGET_COL])
        .dropna()
        .astype(str)
        .value_counts()
        .head(10)
        .to_dict()
    )

    print(
        "    Target recovery          : "
        f"{len(candidate_positions)} candidate column(s), "
        f"{non_empty:,} non-empty label(s) recovered"
    )
    if non_empty == 0:
        print("    WARNING: Label candidates were found, but all recovered labels are empty.")
    else:
        print(f"    Label preview            : {unique_preview}")

    return df_out, {
        "target_col": TARGET_COL,
        "candidate_columns": candidate_info,
        "non_empty_labels_after_recovery": non_empty,
        "label_value_preview_top10": unique_preview,
        "status": "recovered" if non_empty > 0 else "empty_after_recovery",
    }


def _coalesce_duplicate_columns(df):
    """
    Coalesce duplicate column names created by mixed CICFlowMeter headers.

    This is important when a folder contains CSVs with headers such as
    " Label", "Label", "Tot Fwd Pkts", and "Total Fwd Packets". After
    cleaning/renaming, those variants become duplicate columns. Keeping only
    the first duplicate can accidentally discard the only populated Label
    column for some rows; therefore we combine duplicates row-wise using the
    first non-null value.
    """
    cols = list(df.columns)
    duplicated_names = sorted({c for c in cols if cols.count(c) > 1})
    if not duplicated_names:
        return df, []

    pieces = []
    seen = set()
    for i, col in enumerate(cols):
        if col in seen:
            continue

        if col in duplicated_names:
            positions = [j for j, c in enumerate(cols) if c == col]
            block = df.iloc[:, positions]
            # bfill(axis=1) chooses the first non-null value across duplicates.
            merged = block.bfill(axis=1).iloc[:, 0]
            merged.name = col
            pieces.append(merged)
        else:
            series = df.iloc[:, i]
            series.name = col
            pieces.append(series)

        seen.add(col)

    return pd.concat(pieces, axis=1), duplicated_names


def standardize_column_names(df, dataset_name):
    """
    Clean column names, rename known aliases, and coalesce duplicated headers.

    This step is intentionally done BEFORE metadata exclusion and label checking,
    so label/header variants can still be corrected and preserved.
    """
    original_cols = list(df.columns)
    initial_cleaned_cols = [_clean_col_name(c) for c in original_cols]
    df.columns = initial_cleaned_cols

    # Recover/protect the target column BEFORE feature alias renaming. This avoids
    # losing labels when concat(folder CSVs) creates dirty or split label columns.
    df, target_recovery_report = recover_target_column(df, dataset_name)
    cleaned_cols = list(df.columns)

    alias_map = {
        _compact_col_name(k): _clean_col_name(v)
        for k, v in COLUMN_RENAME_ALIASES_RAW.items()
    }

    rename_map = {}
    conflicts = []

    for col in cleaned_cols:
        target = alias_map.get(_compact_col_name(col))
        if not target or target == col:
            continue

        # Rename even if the target already exists. Any duplicate columns caused
        # by this are safely coalesced below instead of being dropped.
        if target in cleaned_cols:
            conflicts.append({
                "source": col,
                "target": target,
                "action": "renamed_then_coalesced_with_existing_target",
            })
        rename_map[col] = target

    if rename_map:
        df.rename(columns=rename_map, inplace=True)

    df, duplicated_cols = _coalesce_duplicate_columns(df)

    whitespace_changes = [
        {"from": old, "to": new}
        for old, new in zip(original_cols, initial_cleaned_cols)
        if old != new
    ]

    report = {
        "dataset_name": dataset_name,
        "whitespace_or_bom_fixed": whitespace_changes,
        "renamed_columns": rename_map,
        "rename_conflicts": conflicts,
        "duplicate_columns_coalesced_after_rename": duplicated_cols,
        "target_recovery_report": target_recovery_report,
        # Kept for backward readability with earlier generated reports.
        "duplicate_columns_dropped_after_rename": [],
    }

    if whitespace_changes or rename_map or conflicts or duplicated_cols:
        print(
            "    Column names normalized : "
            f"{len(whitespace_changes)} whitespace/BOM fix(es), "
            f"{len(rename_map)} alias rename(s), "
            f"{len(conflicts)} alias overlap(s), "
            f"{len(duplicated_cols)} duplicate column group(s) coalesced"
        )
    else:
        print("    Column names normalized : no changes needed")

    return df, report


def align_features_to_reference_schema(X, dataset_name):
    """
    Force numeric features to match REFERENCE_FEATURES exactly.

    - For REFERENCE_DATASET_NAME, the current numeric columns become the reference.
    - For all other datasets, missing reference features are added and non-reference
      features are optionally dropped. Final column order always follows reference.
    """
    global REFERENCE_FEATURES

    report = {
        "enabled": ENFORCE_REFERENCE_SCHEMA,
        "dataset_name": dataset_name,
        "reference_dataset_name": REFERENCE_DATASET_NAME,
        "reference_schema_file": REFERENCE_SCHEMA_FILE,
        "method": "disabled",
        "n_features_before": int(X.shape[1]),
        "n_features_after": int(X.shape[1]),
        "missing_reference_features_added": [],
        "extra_features_dropped": [],
        "extra_features_kept": [],
    }

    if not ENFORCE_REFERENCE_SCHEMA:
        return X, report

    # Load an existing schema for non-reference datasets if the runtime cache is empty.
    if REFERENCE_FEATURES is None and os.path.exists(REFERENCE_SCHEMA_FILE) and dataset_name != REFERENCE_DATASET_NAME:
        REFERENCE_FEATURES = joblib.load(REFERENCE_SCHEMA_FILE)

    # Create/refresh the reference schema from the designated reference dataset.
    if REFERENCE_FEATURES is None or dataset_name == REFERENCE_DATASET_NAME:
        REFERENCE_FEATURES = list(X.columns)
        joblib.dump(REFERENCE_FEATURES, REFERENCE_SCHEMA_FILE)
        report.update({
            "method": "reference_created_or_refreshed",
            "n_features_after": int(X.shape[1]),
            "reference_feature_count": len(REFERENCE_FEATURES),
        })
        print(
            f"    Reference schema set from {dataset_name}: "
            f"{len(REFERENCE_FEATURES)} numeric feature(s)"
        )
        return X.loc[:, REFERENCE_FEATURES], report

    ref_cols = list(REFERENCE_FEATURES)
    current_cols = list(X.columns)
    missing = [c for c in ref_cols if c not in current_cols]
    extra = [c for c in current_cols if c not in ref_cols]

    for col in missing:
        X[col] = MISSING_REFERENCE_FILL_VALUE

    if DROP_NON_REFERENCE_FEATURES:
        X = X.loc[:, ref_cols]
        extra_dropped = extra
        extra_kept = []
    else:
        X = X.loc[:, ref_cols + extra]
        extra_dropped = []
        extra_kept = extra

    report.update({
        "method": "aligned_to_reference",
        "n_features_after": int(X.shape[1]),
        "reference_feature_count": len(ref_cols),
        "missing_reference_features_added": missing,
        "extra_features_dropped": extra_dropped,
        "extra_features_kept": extra_kept,
        "missing_fill_value": MISSING_REFERENCE_FILL_VALUE,
        "drop_non_reference_features": DROP_NON_REFERENCE_FEATURES,
    })

    print(
        "    Feature schema aligned   : "
        f"{len(ref_cols)} reference feature(s), "
        f"{len(missing)} missing added, "
        f"{len(extra_dropped)} extra dropped, "
        f"{len(extra_kept)} extra kept"
    )

    return X, report


# =============================================================================
# STRATIFIED SUBSAMPLING  (Strategy B: dual-tier floor)
# =============================================================================

def stratified_subsample(X, y, n_target, label="subsample"):
    """
    Stratified subsample with HARD floor (drop rare) and SOFT floor (protect
    minority). Returns subsampled X, y as DataFrames/Series.
    """
    y_s = pd.Series(np.asarray(y)).reset_index(drop=True)
    counts = y_s.value_counts()
    n_total = len(y_s)

    print(f"\n  [{label}]")
    print(f"    Input    : {n_total:,} rows, {len(counts)} classes")

    if n_total <= n_target:
        print(f"    {n_total:,} ≤ target {n_target:,} — no subsampling needed.")
        return X.reset_index(drop=True), y_s, {"method": "none", "dropped": []}

    # ── HARD floor: drop rare classes ──
    rare = counts[counts < HARD_FLOOR]
    valid_classes = counts[counts >= HARD_FLOOR].index.tolist()
    dropped_classes = []
    if len(rare) > 0:
        print(f"    HARD floor ({HARD_FLOOR}) drops {len(rare)} class(es):")
        for c, n in rare.items():
            print(f"      • {c!r}: {n} samples → DROPPED")
            dropped_classes.append(str(c))

    valid_mask = y_s.isin(valid_classes).values
    X_valid = X.iloc[np.where(valid_mask)[0]].reset_index(drop=True)
    y_valid = y_s.iloc[np.where(valid_mask)[0]].reset_index(drop=True)
    counts_valid = y_valid.value_counts()
    n_valid = len(y_valid)

    if n_valid <= n_target:
        print(f"    After hard floor: {n_valid:,} ≤ target — keeping all.")
        return X_valid, y_valid, {"method": "hard_floor_only", "dropped": dropped_classes}

    # ── SOFT floor: compute per-class targets ──
    scale = n_target / counts_valid.sum()
    targets = {}
    soft_used = []
    for c, n in counts_valid.items():
        proportional = int(round(int(n) * scale))
        floor_cap    = min(SOFT_FLOOR, int(n))
        target       = max(proportional, floor_cap)
        if target > proportional:
            soft_used.append({"class": str(c), "proportional": proportional,
                              "kept": target, "available": int(n)})
        targets[c] = target

    # ── Sample ──
    if not soft_used:
        # Pure proportional → StratifiedShuffleSplit
        sss = StratifiedShuffleSplit(n_splits=1, train_size=n_target,
                                     random_state=RANDOM_STATE)
        idx, _ = next(sss.split(np.zeros(n_valid), y_valid))
        X_out = X_valid.iloc[idx].reset_index(drop=True)
        y_out = y_valid.iloc[idx].reset_index(drop=True)
        method = "StratifiedShuffleSplit"
        print(f"    Method   : {method} (no soft-floor needed)")
    else:
        # Per-class manual sampling
        rng = np.random.RandomState(RANDOM_STATE)
        parts = []
        for c, target in targets.items():
            class_idx = np.where(y_valid.values == c)[0]
            if target >= len(class_idx):
                parts.append(class_idx)
            else:
                parts.append(rng.choice(class_idx, size=target, replace=False))
        keep = np.sort(np.concatenate(parts))
        X_out = X_valid.iloc[keep].reset_index(drop=True)
        y_out = y_valid.iloc[keep].reset_index(drop=True)
        method = "per-class stratified"
        print(f"    Method   : {method} (soft floor inflated {len(soft_used)} class(es))")
        for s in soft_used:
            print(f"      • {s['class']}: proportional={s['proportional']} → kept={s['kept']}")

    print(f"    Output   : {len(X_out):,} rows ({len(X_out)/n_total*100:.1f}% of input)")
    print(f"    Classes  : {dict(y_out.value_counts())}")

    report = {
        "method": method,
        "dropped": dropped_classes,
        "soft_used": soft_used,
        "n_before": n_total,
        "n_after": len(X_out),
    }
    return X_out, y_out, report


# =============================================================================
# PREPROCESSING  (label-agnostic, applied before split)
# =============================================================================

def preprocess(df, dataset_name):
    print(f"\n  Preprocessing {dataset_name}...")

    # Normalize and rename feature/label columns before any column selection.
    df, column_rename_report = standardize_column_names(df, dataset_name)

    # Drop metadata columns
    drop = [c for c in EXCLUDE_COLS if c in df.columns]
    if drop:
        df.drop(columns=drop, inplace=True, errors="ignore")
        print(f"    Excluded {len(drop)} metadata columns")

    # Duplicates
    before = len(df)
    df.drop_duplicates(inplace=True)
    n_dup = before - len(df)
    if n_dup > 0:
        print(f"    Duplicates removed : {n_dup:,}")

    # Inf → NaN → drop feature cols with >50% missing → fill rest with median.
    # IMPORTANT: never apply the column-drop threshold to TARGET_COL. In folder
    # mode, mixed headers can otherwise make the target look sparse before all
    # label variants have been fully normalized/coalesced.
    df.replace([np.inf, -np.inf], np.nan, inplace=True)

    if TARGET_COL in df.columns:
        before_label_drop = len(df)
        protected_label_text = _as_clean_label_text(df[TARGET_COL])
        valid_label_mask = protected_label_text.notna()
        df = df.loc[valid_label_mask].reset_index(drop=True)
        dropped_missing_labels = before_label_drop - len(df)
        if dropped_missing_labels > 0:
            print(f"    Rows with missing labels dropped : {dropped_missing_labels:,}")

        y_keep = df[TARGET_COL].copy()
        feature_df = df.drop(columns=[TARGET_COL])
        feature_df.dropna(thresh=len(feature_df) * 0.5, axis=1, inplace=True)
        df = pd.concat([feature_df, y_keep], axis=1)
    else:
        df.dropna(thresh=len(df) * 0.5, axis=1, inplace=True)

    num_cols = df.select_dtypes(include=np.number).columns
    df[num_cols] = df[num_cols].fillna(df[num_cols].median())

    # Encode non-target categoricals
    cat_cols = [c for c in df.select_dtypes(include="object").columns if c != TARGET_COL]
    for col in cat_cols:
        df[col] = LabelEncoder().fit_transform(df[col].astype(str))

    print(f"    Shape after preprocess : {df.shape}")
    return df, column_rename_report


# =============================================================================
# LABEL ENCODING
# =============================================================================

class BinaryLabelEncoder:
    """Minimal label-encoder stand-in for binary mode (Normal/Attack).
    Defined at module level so joblib/pickle can serialize it."""
    classes_ = np.array(["Normal", "Attack"])


def encode_labels(y_raw):
    """
    Returns: y_encoded (int Series), label_encoder (or BinaryLE), class_names
    """
    y_str = y_raw.astype(str).str.strip()

    if BINARY_MODE:
        y_enc = y_str.apply(
            lambda x: 0 if x.lower() in ["benign", "normal"] else 1
        ).astype(np.int32)
        le = BinaryLabelEncoder()
        class_names = list(le.classes_)
        print(f"    Binary encoding → {dict(Counter(y_enc))}")
    else:
        le = LabelEncoder()
        y_enc = pd.Series(le.fit_transform(y_str), dtype=np.int32)
        class_names = list(le.classes_)
        print(f"    Multi-class → {len(class_names)} classes: {class_names}")

    return y_enc, le, class_names


# =============================================================================
# PROCESS ONE DATASET
# =============================================================================

def process_one_dataset(name, path):
    print("\n" + "="*70)
    print(f"  PREPARING: {name}")
    print(f"  Path: {path}")
    print("="*70)

    t0 = time.time()
    out_dir = os.path.join(OUTPUT_ROOT, name)
    os.makedirs(out_dir, exist_ok=True)

    # ── 1. Load ──
    print(f"\n  Loading {name}...")
    if os.path.isdir(path):
        files = sorted([f for f in os.listdir(path) if f.endswith(".csv")])
        dfs = []
        for f in files:
            print(f"    Reading: {f}")
            dfs.append(pd.read_csv(os.path.join(path, f), low_memory=False))
        df = pd.concat(dfs, ignore_index=True)
        print(f"    Loaded {len(files)} files.")
    else:
        df = pd.read_csv(path, low_memory=False)
    n_raw = len(df)
    print(f"    Raw shape : {df.shape[0]:,} × {df.shape[1]}")

    # ── 2. Preprocess ──
    df, column_rename_report = preprocess(df, name)

    # ── 3. Separate target ──
    if TARGET_COL not in df.columns:
        raise ValueError(f"'{TARGET_COL}' column not found in {name}. "
                         f"Available: {list(df.columns[:10])}...")

    y_raw = df[TARGET_COL]
    X = df.drop(columns=[TARGET_COL]).select_dtypes(include=np.number)

    # Align feature names/order to the reference dataset before sampling/splitting.
    X, feature_schema_report = align_features_to_reference_schema(X, name)
    all_numeric_cols = list(X.columns)

    # Save per-dataset column/schema report for auditability.
    with open(os.path.join(out_dir, "column_schema_report.json"), "w") as f:
        json.dump({
            "column_rename_report": column_rename_report,
            "feature_schema_report": feature_schema_report,
        }, f, indent=2, default=str)

    # Record original class distribution BEFORE encoding
    original_dist = dict(y_raw.astype(str).str.strip().value_counts())

    # ── 4. Encode labels ──
    y_enc, le, class_names = encode_labels(y_raw)

    if len(X) == 0 or len(y_enc) == 0:
        raise ValueError(
            f"No usable labelled rows remain for {name}. "
            f"Check column_schema_report.json and the raw CSV header/label columns."
        )
    if pd.Series(y_enc).nunique() < 2:
        raise ValueError(
            f"Only one class remains for {name}; stratified train/test split needs at least two classes."
        )

    # ── 5. Stratified subsample to ~SAMPLE_FRAC ──
    n_target = min(int(len(X) * SAMPLE_FRAC), MAX_ROWS)
    print(f"\n  Target sample size: {n_target:,} "
          f"({SAMPLE_FRAC*100:.0f}% of {len(X):,}, capped at {MAX_ROWS:,})")

    X_sub, y_sub, sample_report = stratified_subsample(
        X, y_enc, n_target, label=f"{name} sampling"
    )

    # ── 6. Stratified train/test split ──
    print(f"\n  Train/test split (stratified, test={TEST_SIZE*100:.0f}%)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X_sub, y_sub,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y_sub
    )
    X_train = X_train.reset_index(drop=True)
    X_test  = X_test.reset_index(drop=True)
    y_train = y_train.reset_index(drop=True)
    y_test  = y_test.reset_index(drop=True)

    print(f"    Train : {X_train.shape}  {dict(Counter(y_train))}")
    print(f"    Test  : {X_test.shape}  {dict(Counter(y_test))}")

    # ── 7. Save everything ──
    print(f"\n  Saving to {out_dir}/")

    # ── 7a. Pickle format (fast loading for Script 01) ──
    joblib.dump(X_train,          os.path.join(out_dir, "X_train_raw.pkl"))
    joblib.dump(X_test,           os.path.join(out_dir, "X_test_raw.pkl"))
    joblib.dump(y_train,          os.path.join(out_dir, "y_train.pkl"))
    joblib.dump(y_test,           os.path.join(out_dir, "y_test.pkl"))
    joblib.dump(le,               os.path.join(out_dir, "label_encoder.pkl"))
    joblib.dump(all_numeric_cols, os.path.join(out_dir, "all_numeric_cols.pkl"))

    # ── 7b. CSV format (human-readable, importable by other tools) ──
    # Merge features + label into a single CSV per partition.
    # Label column uses ORIGINAL class names for readability.
    if hasattr(le, 'classes_'):
        label_map = {i: c for i, c in enumerate(le.classes_)}
    else:
        label_map = {0: "Normal", 1: "Attack"}

    train_csv = X_train.copy()
    train_csv[TARGET_COL] = y_train.map(label_map).values
    train_csv.to_csv(os.path.join(out_dir, "train.csv"), index=False)

    test_csv = X_test.copy()
    test_csv[TARGET_COL] = y_test.map(label_map).values
    test_csv.to_csv(os.path.join(out_dir, "test.csv"), index=False)

    train_mb = os.path.getsize(os.path.join(out_dir, "train.csv")) / (1024**2)
    test_mb  = os.path.getsize(os.path.join(out_dir, "test.csv")) / (1024**2)
    print(f"  ✔ train.csv : {len(train_csv):,} rows ({train_mb:.1f} MB)")
    print(f"  ✔ test.csv  : {len(test_csv):,} rows ({test_mb:.1f} MB)")

    elapsed = time.time() - t0

    # Metadata report
    report = {
        "dataset_name": name,
        "source_path": path,
        "n_raw_rows": n_raw,
        "n_after_preprocess": len(X),
        "n_numeric_features": len(all_numeric_cols),
        "column_rename_report": column_rename_report,
        "feature_schema_report": feature_schema_report,
        "binary_mode": BINARY_MODE,
        "class_names": class_names,
        "original_distribution": {str(k): int(v) for k, v in original_dist.items()},
        "sample_frac": SAMPLE_FRAC,
        "sample_report": sample_report,
        "n_train": len(X_train),
        "n_test": len(X_test),
        "train_distribution": {str(k): int(v) for k, v in Counter(y_train).items()},
        "test_distribution": {str(k): int(v) for k, v in Counter(y_test).items()},
        "random_state": RANDOM_STATE,
        "elapsed_seconds": round(elapsed, 1),
    }
    joblib.dump(report, os.path.join(out_dir, "prep_report.pkl"))

    # Also save human-readable JSON
    with open(os.path.join(out_dir, "prep_report.json"), "w") as f:
        json.dump(report, f, indent=2, default=str)

    # ── 8. Class distribution plot ──
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    pd.Series(Counter(y_train)).sort_index().plot(
        kind="bar", ax=axes[0], color="steelblue", edgecolor="black", alpha=0.85)
    axes[0].set_title(f"{name} — Train ({len(X_train):,})", fontweight="bold")
    axes[0].set_ylabel("Count"); axes[0].tick_params(axis="x", rotation=45)

    pd.Series(Counter(y_test)).sort_index().plot(
        kind="bar", ax=axes[1], color="darkorange", edgecolor="black", alpha=0.85)
    axes[1].set_title(f"{name} — Test ({len(X_test):,})", fontweight="bold")
    axes[1].set_ylabel("Count"); axes[1].tick_params(axis="x", rotation=45)

    plt.suptitle(f"Class Distribution After Stratified Sampling ({SAMPLE_FRAC*100:.0f}%)",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "class_distribution.png"), dpi=150)
    plt.close()

    print(f"\n  ✔ {name} done in {elapsed:.1f}s")
    print(f"    {n_raw:,} → {len(X_sub):,} (sampled) → "
          f"{len(X_train):,} train + {len(X_test):,} test")

    return report


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("\n" + "="*70)
    print("  SCRIPT 00 — PREPARE DATASETS")
    print(f"  Datasets     : {len(DATASETS)}")
    print(f"  Sample frac  : {SAMPLE_FRAC*100:.0f}%")
    print(f"  Mode         : {'Binary' if BINARY_MODE else 'Multi-class'}")
    print(f"  Output       : ./{OUTPUT_ROOT}/")
    print("="*70)

    t_start = time.time()
    all_reports = {}

    for name, path in DATASETS.items():
        try:
            report = process_one_dataset(name, path)
            all_reports[name] = report
        except FileNotFoundError:
            print(f"\n  ✗ SKIPPED {name}: file not found at {path}")
        except Exception as e:
            print(f"\n  ✗ FAILED {name}: {e}")
            import traceback; traceback.print_exc()

    # ── Summary ──
    elapsed = time.time() - t_start
    print("\n" + "="*70)
    print("  SUMMARY")
    print("="*70)

    summary_rows = []
    for name, r in all_reports.items():
        summary_rows.append({
            "Dataset": name,
            "Raw Rows": f"{r['n_raw_rows']:,}",
            "Sampled": f"{r['sample_report'].get('n_after', r['n_train']+r['n_test']):,}",
            "Train": f"{r['n_train']:,}",
            "Test": f"{r['n_test']:,}",
            "Features": r['n_numeric_features'],
            "Classes": len(r['class_names']),
            "Dropped": len(r['sample_report'].get('dropped', [])),
            "Time": f"{r['elapsed_seconds']:.1f}s",
        })

    if summary_rows:
        df_summary = pd.DataFrame(summary_rows)
        print(f"\n{df_summary.to_string(index=False)}")
        df_summary.to_csv(os.path.join(OUTPUT_ROOT, "preparation_summary.csv"), index=False)

    print(f"""
  ╔═══════════════════════════════════════════════════════════╗
  ║  SCRIPT 00 COMPLETE ✔  ({elapsed:.1f}s total)
  ╠═══════════════════════════════════════════════════════════╣
  ║  Prepared {len(all_reports)}/{len(DATASETS)} datasets
  ║  Output: ./{OUTPUT_ROOT}/
  ║
  ║  Next steps — for EACH prepared dataset:
  ║
  ║    Edit 01_train_save_models.py:
  ║      PREPARED_DIR = "{OUTPUT_ROOT}/CIC17"
  ║
  ║    Then run:
  ║      python 01_train_save_models.py
  ║      python 02_load_explain_models.py
  ╚═══════════════════════════════════════════════════════════╝
    """)


if __name__ == "__main__":
    main()