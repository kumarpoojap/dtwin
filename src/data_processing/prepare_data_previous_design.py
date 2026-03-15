import os
import re
import ast
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import joblib

# -------------------------
# Config — EDIT THESE PATHS
# -------------------------
def resolve_data_file(path: Path) -> Path:
    """
    Return the first existing path among:
    - given path (absolute or relative)
    - same path with .txt suffix if none provided
    Relative paths are resolved from PROJECT_ROOT.
    """
    candidates = []
    p = Path(path)
    candidates.append(p)
    if p.suffix == "":
        candidates.append(p.with_suffix(".txt"))

    for cand in candidates:
        cand_abs = cand if cand.is_absolute() else PROJECT_ROOT / cand
        if cand_abs.exists():
            return cand_abs

    raise FileNotFoundError(
        "Could not find data file. Tried: "
        + ", ".join(str((PROJECT_ROOT / c) if not Path(c).is_absolute() else str(c)) for c in candidates)
    )

PROJECT_ROOT = Path(__file__).resolve().parents[2]
BASE_DIR = PROJECT_ROOT / "data" / "previousDesign"  # folder that holds your two files
# Match actual filenames in data/previousDesign
INROW_FILE = resolve_data_file(BASE_DIR / "inrow_cooler_sensor_door_opened")
NETBOTZ_FILE = resolve_data_file(BASE_DIR / "netbotz_sensor_values_door_opened_temperature")
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

if not INROW_FILE.exists() or not NETBOTZ_FILE.exists():
    raise FileNotFoundError(
        f"Expected data files not found. Looked for {INROW_FILE} and {NETBOTZ_FILE}. "
        "Ensure you're using the repo layout with data/previousDesign/*.txt present."
    )

# Target resampling cadence (dataset looks like 10s steps)
RESAMPLE_RULE = "10s"

# ----------------------------------
# Helpers: parsing & key normalization
# ----------------------------------

def safe_parse_line(line: str) -> Dict:
    """
    Lines are Python dict literals like:
    {'2022-06-03 15:00:00': [{'key': value}, {'key2': value2}, ...]}
    Use ast.literal_eval to safely parse.
    """
    line = line.strip()
    if not line:
        return {}
    return ast.literal_eval(line)

def clean_key_for_inrow(k: str) -> str:
    """
    Normalize in-row cooler keys:
    - remove unit brackets "[...]" (we keep units in the column name when helpful)
    - lower-case, snake_case
    - shorten vendor-specific prefixes
    - prefer metric columns (e.g., L/s) and drop US duplicates (CFM)
    """
    # Keep a copy for possible unit disambiguation
    original = k

    # Strip bracketed units for the machine-readable key
    k = re.sub(r"\[.*?\]", "", k)

    # Collapse vendor prefixes
    k = k.replace("airIRG2GroupStatus", "group_")
    k = k.replace("airIRG2RDT2Status", "rdt2_")

    # Clean punctuation/spacing
    k = k.replace("/", "_per_")
    k = re.sub(r"[^\w]+", "_", k).strip("_").lower()

    # Post-fix: some known friendly names
    # We'll also drop explicit US airflow later to avoid duplication.
    rename_map = {
        "group_status_airflowmetric": "airflow_lps",
        "group_status_airflowus_cubicfeet_m": "airflow_cfm",
        "group_status_cooloutput": "cool_output_kwh",           # keep label as provided by file
        "group_status_minrackinlettempmetric_c": "min_rack_inlet_c",
        "rdt2_status_cooldemand": "cool_demand_kwh",
        "rdt2_status_evaporatorfanspeed": "evap_fan_speed_pct",
        "rdt2_status_returnairtempmetric_c": "return_air_c",
        "rdt2_status_suctiontempmetric_c": "suction_c",
        "rdt2_status_supplyairtempmetric_c": "supply_air_c",
    }
    return rename_map.get(k, k)

def clean_key_for_netbotz(k: str) -> str:
    """
    Normalize NetBotz temperature sensor name to a tidy column, e.g.
    'Temperature (Cabin 3 - Back - 24U)' -> 'temp_cabin3_back_24u_c'
    'Temperature (1)' -> likely unused -> we will drop later
    """
    k = k.strip()

    # Keep "(1)" case to filter later
    if k == "Temperature (1)":
        return "temp_unused_channel"

    # Extract cabin/pos/U if available
    m = re.match(r"Temperature\s*\((.*?)\)", k)
    if m:
        inside = m.group(1)  # e.g., "Cabin 3 - Back - 24U"
        # standardize spaces
        inside = re.sub(r"\s*-\s*", "-", inside)
        # split pieces
        parts = inside.split("-")  # ["Cabin 3", "Back", "24U"]
        parts = [p.strip().lower() for p in parts]
        # normalize cabin number and U
        cabin = re.sub(r"\s+", "", parts[0]) if len(parts) > 0 else "cabin?"
        pos = parts[1] if len(parts) > 1 else "pos?"
        u = parts[2] if len(parts) > 2 else "u?"

        # 24U -> 24u (lowercase)
        u = u.lower()
        col = f"temp_{cabin}_{pos}_{u}_c"
        col = re.sub(r"[^\w]+", "_", col)
        return col
    else:
        # Fallback: sanitize everything
        col = k.lower()
        col = col.replace("temperature", "temp")
        col = re.sub(r"[^\w]+", "_", col)
        return col

def parse_inrow_file(path: Path) -> pd.DataFrame:
    """
    Return a DataFrame indexed by timestamp with cleaned in-row cooler columns.
    Drops US (CFM) airflow to keep metric L/s only.
    """
    records = []
    with open(path, "r") as f:
        for line in f:
            if not line.strip():
                continue
            d = safe_parse_line(line)
            if not d:
                continue
            # one timestamp per line
            ts = list(d.keys())[0]
            items = d[ts]  # list of {k: v}
            row = {"timestamp": pd.to_datetime(ts)}
            for kv in items:
                for k, v in kv.items():
                    ck = clean_key_for_inrow(k)
                    row[ck] = v
            records.append(row)

    df = pd.DataFrame(records).sort_values("timestamp").set_index("timestamp")

    # Prefer metric airflow; drop US (CFM) duplicate if present
    if "airflow_cfm" in df.columns and "airflow_lps" in df.columns:
        df = df.drop(columns=["airflow_cfm"])

    # Ensure numeric
    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # Resample to common cadence and forward-fill small gaps (1 step)
    df = (
        df.resample(RESAMPLE_RULE)
          .mean()
          .ffill(limit=1)
    )
    return df

def parse_netbotz_file(path: Path) -> pd.DataFrame:
    """
    Return a DataFrame indexed by timestamp with one column per NetBotz temperature probe.
    Drops 'unused' channels that read as 0.0.
    """
    records = []
    with open(path, "r") as f:
        for line in f:
            if not line.strip():
                continue
            d = safe_parse_line(line)
            if not d:
                continue
            ts = list(d.keys())[0]
            items = d[ts]  # list of {sensor_label: value}
            row = {"timestamp": pd.to_datetime(ts)}
            for kv in items:
                for k, v in kv.items():
                    ck = clean_key_for_netbotz(k)
                    row[ck] = v
            records.append(row)

    df = pd.DataFrame(records).sort_values("timestamp").set_index("timestamp")

    # Drop 'unused' channels (commonly named temp_unused_channel) and any all-zero columns
    drop_cols = [c for c in df.columns if "unused" in c]
    # also drop columns that are all zeros (likely unplugged)
    all_zero_cols = [c for c in df.columns if pd.to_numeric(df[c], errors="coerce").fillna(0).abs().sum() == 0.0]
    to_drop = sorted(set(drop_cols + all_zero_cols))
    if to_drop:
        df = df.drop(columns=to_drop)

    # Ensure numeric temperatures
    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # Filter out obviously invalid 0.0 temps (common when probes disconnect)
    df = df.mask(df == 0.0)

    # Resample and lightly forward-fill tiny gaps (1 step)
    df = (
        df.resample(RESAMPLE_RULE)
          .mean()
          .ffill(limit=1)
    )

    # OPTIONAL: keep only 'back' probes if present, or keep all
    # Here we keep all temperature probes that survived cleaning.
    return df

def zscore_normalize(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    """
    Fit a StandardScaler to the specified columns and return a DataFrame with
    additional *_z columns. Also save the scaler to artifacts for later reuse.
    """
    scaler = StandardScaler()
    X = df[cols].to_numpy()
    Z = scaler.fit_transform(X)
    z_cols = [f"{c}_z" for c in cols]
    df_z = df.copy()
    for i, c in enumerate(z_cols):
        df_z[c] = Z[:, i]

    # Persist scaler for training/inference parity
    joblib.dump(scaler, ARTIFACTS_DIR / "scaler_netbotz.pkl")
    return df_z

# -------------------
# Pipeline
# -------------------
inrow_df = parse_inrow_file(INROW_FILE)
# After parsing & resampling the in-row DF, add this mapping:
inrow_rename_map = {
    "group_airflowmetric": "airflow_lps",             # metric airflow
    "group_airflowus": "airflow_cfm",                 # will drop later
    "group_cooloutput": "cool_output_kwh",
    "group_minrackinlettempmetric": "min_rack_inlet_c",
    "rdt2_cooldemand": "cool_demand_kwh",
    "rdt2_evaporatorfanspeed": "evap_fan_speed_pct",
    "rdt2_returnairtempmetric": "return_air_c",
    "rdt2_suctiontempmetric": "suction_c",
    "rdt2_supplyairtempmetric": "supply_air_c",
}
inrow_df = inrow_df.rename(columns=inrow_rename_map)

# Prefer metric airflow; drop US airflow if present
if "airflow_cfm" in inrow_df.columns:
    inrow_df = inrow_df.drop(columns=["airflow_cfm"])

print(f"In-row cooler dataframe: {inrow_df.shape}, columns: {list(inrow_df.columns)}")

netbotz_df = parse_netbotz_file(NETBOTZ_FILE)
print(f"NetBotz dataframe: {netbotz_df.shape}, columns: {list(netbotz_df.columns)}")

# Merge on timestamp (inner join ensures strict alignment; use outer if you prefer)
merged = inrow_df.join(netbotz_df, how="inner")
print(f"Merged dataframe: {merged.shape}")

# Identify feature vs target columns
PREFERRED_FEATURES = [
    "airflow_lps",
    "cool_output_kwh",
    "cool_demand_kwh",
    "evap_fan_speed_pct",
    "return_air_c",
    "suction_c",
    "supply_air_c",
    "min_rack_inlet_c",
]

feature_cols = [c for c in PREFERRED_FEATURES if c in merged.columns]

# Optional safety fallback: if still empty (shouldn't be now), pick by pattern
if not feature_cols:
    patterns = ["airflow", "cool_output", "cool_demand", "fan_speed",
                "return_air", "suction", "supply_air", "min_rack_inlet"]
    for p in patterns:
        feature_cols.extend([c for c in merged.columns if p in c])
    # de-duplicate while preserving order
    seen = set()
    feature_cols = [c for c in feature_cols if not (c in seen or seen.add(c))]

# Targets: all NetBotz temperature columns (rear probes)
target_cols = [c for c in merged.columns if c.startswith("temp_cabin") and c.endswith("_c")]

# Clean rows where *all* targets are missing
merged = merged.dropna(subset=target_cols, how="all")

# Optional: forward-fill small gaps (1 step) for features only
merged[feature_cols] = merged[feature_cols].ffill(limit=1)

# Normalization for NetBotz temperatures only (create *_z columns)
merged_z = zscore_normalize(merged, target_cols)

# Save artifacts
out_parquet = ARTIFACTS_DIR / "previousDesign_merged.parquet"
out_csv = ARTIFACTS_DIR / "previousDesign_merged.csv"
spec_json = ARTIFACTS_DIR / "feature_target_spec.json"

merged_z.to_parquet(out_parquet, engine="pyarrow")
merged_z.to_csv(out_csv, index=True)

with open(spec_json, "w") as f:
    json.dump(
        {
            "resample_rule": RESAMPLE_RULE,
            "feature_cols": feature_cols,
            "target_cols_raw": target_cols,
            "target_cols_normalized": [f"{c}_z" for c in target_cols],
            "notes": [
                "Features come from in-row cooler telemetry.",
                "Targets are NetBotz rear temperature probes at various U-levels.",
                "Z-score normalization applied only to NetBotz columns; scaler saved to scaler_netbotz.pkl"
            ],
        },
        f,
        indent=2
    )


print("\nUsing features:")
for c in feature_cols: print("  -", c)
print("Total features:", len(feature_cols))

print("\nUsing target columns (sample):")
for c in target_cols[:5]: print("  -", c)
print("Total targets:", len(target_cols))

assert len(feature_cols) > 0, "No feature columns found—check renaming map."
assert len(target_cols) > 0, "No target columns found—check NetBotz parsing."


print(f"Saved: {out_parquet}")
print(f"Saved: {out_csv}")
print(f"Saved: {spec_json}")