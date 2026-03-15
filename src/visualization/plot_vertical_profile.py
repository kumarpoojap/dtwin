# src/visualization/plot_vertical_profile.py
from __future__ import annotations
import re
from typing import Iterable, Optional, Tuple, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# ---------------------------
# Helpers to work with U-levels
# ---------------------------

def extract_u_from_col(col: str) -> Optional[int]:
    """
    Extract integer U-level from a column like 'temp_cabin3_back_34u_c' -> 34.
    Returns None if not found.
    """
    m = re.search(r"_(\d{1,2})u_", col)
    if m:
        return int(m.group(1))
    return None


def select_netbotz_temp_cols(df: pd.DataFrame, cabin_filter: Optional[str] = None) -> List[str]:
    """
    Pick NetBotz temp columns. Optionally filter by cabin, e.g., 'cabin3'.
    """
    cols = [c for c in df.columns if c.startswith("temp_") and c.endswith("_c")]
    if cabin_filter:
        cols = [c for c in cols if cabin_filter.lower() in c.lower()]
    # Ensure there is a U-level in the name
    cols = [c for c in cols if extract_u_from_col(c) is not None]
    return sorted(cols, key=lambda c: extract_u_from_col(c))


def melt_vertical_profile(df_row: pd.Series, temp_cols: Iterable[str]) -> pd.DataFrame:
    """
    Convert a single timestamp row into a tidy dataframe with columns: ['U', 'temperature_c'].
    """
    data = []
    for c in temp_cols:
        u = extract_u_from_col(c)
        if u is None:
            continue
        val = df_row.get(c, np.nan)
        data.append((u, val))
    prof = pd.DataFrame(data, columns=["U", "temperature_c"]).dropna()
    prof = prof.sort_values("U")
    return prof


# ---------------------------
# Plots
# ---------------------------

def plot_vertical_profile_at_timestamp(
    df: pd.DataFrame,
    timestamp: pd.Timestamp,
    cabin_filter: Optional[str] = "cabin3",
    ax: Optional[plt.Axes] = None,
    title: Optional[str] = None,
) -> plt.Axes:
    """
    Line plot of vertical profile (temperature vs. U) at a given timestamp.
    """
    temp_cols = select_netbotz_temp_cols(df, cabin_filter=cabin_filter)
    if not temp_cols:
        raise ValueError("No NetBotz temperature columns found. Check column names or cabin_filter.")

    # Find nearest index if exact timestamp missing
    if timestamp not in df.index:
        nearest = df.index.get_indexer([timestamp], method="nearest")[0]
        timestamp = df.index[nearest]

    prof = melt_vertical_profile(df.loc[timestamp], temp_cols)

    if ax is None:
        _, ax = plt.subplots(figsize=(5, 6))

    ax.plot(prof["temperature_c"], prof["U"], marker="o", color="tab:red")
    ax.set_xlabel("Temperature (°C)")
    ax.set_ylabel("Rack U-level (U)")
    ax.set_ylim(prof["U"].min() - 1, prof["U"].max() + 1)
    ax.grid(True, alpha=0.3)
    final_title = title or f"Vertical profile at {timestamp}"
    ax.set_title(final_title)
    # In data centers, top U is physically higher. Invert y-axis so top appears at top of plot.
    ax.invert_yaxis()
    return ax


def plot_time_averaged_vertical_profile(
    df: pd.DataFrame,
    start: pd.Timestamp,
    end: pd.Timestamp,
    cabin_filter: Optional[str] = "cabin3",
    ax: Optional[plt.Axes] = None,
    title: Optional[str] = None,
) -> plt.Axes:
    """
    Average temperatures across a time window and plot temperature vs. U.
    """
    temp_cols = select_netbotz_temp_cols(df, cabin_filter=cabin_filter)
    if not temp_cols:
        raise ValueError("No NetBotz temperature columns found.")

    # Slice time window
    window = df.loc[start:end, temp_cols]
    avg_row = window.mean(axis=0)

    prof = melt_vertical_profile(avg_row, temp_cols)

    if ax is None:
        _, ax = plt.subplots(figsize=(5, 6))

    ax.plot(prof["temperature_c"], prof["U"], marker="o", color="tab:blue")
    ax.set_xlabel("Temperature (°C)")
    ax.set_ylabel("Rack U-level (U)")
    ax.set_ylim(prof["U"].min() - 1, prof["U"].max() + 1)
    ax.grid(True, alpha=0.3)
    final_title = title or f"Time-averaged vertical profile: {start} → {end}"
    ax.set_title(final_title)
    ax.invert_yaxis()
    return ax


def plot_vertical_profile_heatmap(
    df: pd.DataFrame,
    cabin_filter: Optional[str] = "cabin3",
    vmax: Optional[float] = None,
    vmin: Optional[float] = None,
    cmap: str = "inferno",
    figsize: Tuple[int, int] = (9, 5),
    title: Optional[str] = None,
) -> plt.Axes:
    """
    Heatmap of temperature over time (x-axis) and U-level (y-axis).
    This is great for spotting persistent hotspots and stratification.
    """
    temp_cols = select_netbotz_temp_cols(df, cabin_filter=cabin_filter)
    if not temp_cols:
        raise ValueError("No NetBotz temperature columns found.")

    # Build a 2D array: rows=U-levels (sorted), cols=time
    # We'll construct a DataFrame indexed by U with columns as timestamps.
    u_levels = [extract_u_from_col(c) for c in temp_cols]
    arr = df[temp_cols].T  # shape: n_cols x n_times
    arr.index = u_levels
    arr = arr.sort_index()  # sort by U
    # Now arr is U x time; seaborn heatmap expects a 2D table with rows/cols labels.
    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(
        arr,
        cmap=cmap,
        vmax=vmax,
        vmin=vmin,
        cbar_kws={"label": "Temperature (°C)"},
        ax=ax
    )
    ax.set_xlabel("Time")
    ax.set_ylabel("Rack U-level (U)")
    ax.set_title(title or "Rack vertical temperature heatmap")
    # Put the hottest U at the top visually (top of rack), so invert y
    ax.invert_yaxis()
    return ax