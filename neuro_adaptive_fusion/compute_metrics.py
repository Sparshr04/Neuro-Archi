"""compute_metrics.py — Standalone post-processing script for Table 1.

Reads ``flight_data_log.csv`` produced by ``data_logger.py`` and computes
the five performance metrics for the Neuro-Adaptive Sensor Fusion paper.

Metrics
───────
  1. Peak Z-Error (m)         — max |true_z - est_z| during Transition (10–20 s)
  2. RMSE (Transition)        — RMS of z_error during Transition phase
  3. Noise Gate Latency (ms)  — Δt between t=10.0 s and first noise_gate_active==True
  4. DNN Correction Factor    — mean & peak covariance_scale_factor during Transition
  5. Filter Stability         — "Stable" if Peak Z-Error < 1.0 m, else "Divergent"

Usage
─────
    # After data_logger.py has finished writing flight_data_log.csv:
    python compute_metrics.py

    # Or specify a custom CSV path:
    python compute_metrics.py --csv /path/to/flight_data_log.csv

Dependencies
────────────
    pip install pandas numpy
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# ── Phase boundaries (seconds) ────────────────────────────────────────────────
T_HOVER_END: float = 10.0
T_TRANSITION_END: float = 20.0

# ── Stability threshold ───────────────────────────────────────────────────────
STABILITY_THRESHOLD_M: float = 1.0  # Peak Z-Error must be below this


# ═══════════════════════════════════════════════════════════════════════════════
#  Data Loading & Validation
# ═══════════════════════════════════════════════════════════════════════════════


def load_and_validate(csv_path: Path) -> pd.DataFrame:
    """Load the CSV and perform basic sanity checks."""
    if not csv_path.exists():
        print(f"[ERROR] File not found: {csv_path}", file=sys.stderr)
        sys.exit(1)

    required_cols = {
        "timestamp",
        "true_z",
        "est_z",
        "z_error",
        "covariance_scale_factor",
        "noise_gate_active",
    }

    df = pd.read_csv(csv_path)

    missing = required_cols - set(df.columns)
    if missing:
        print(f"[ERROR] CSV is missing columns: {missing}", file=sys.stderr)
        sys.exit(1)

    # Ensure correct dtypes
    df["timestamp"] = df["timestamp"].astype(float)
    df["true_z"] = df["true_z"].astype(float)
    df["est_z"] = df["est_z"].astype(float)
    df["z_error"] = df["z_error"].astype(float)
    df["covariance_scale_factor"] = df["covariance_scale_factor"].astype(float)
    df["noise_gate_active"] = df["noise_gate_active"].astype(int).astype(bool)

    # Recompute z_error from raw columns (guards against rounding in CSV)
    df["z_error"] = df["true_z"] - df["est_z"]

    print(f"[INFO] Loaded {len(df):,} rows from {csv_path}")
    print(
        f"[INFO] Time range: {df['timestamp'].min():.3f}s → {df['timestamp'].max():.3f}s"
    )
    return df


# ═══════════════════════════════════════════════════════════════════════════════
#  Metric Calculations
# ═══════════════════════════════════════════════════════════════════════════════


def compute_peak_z_error(transition_df: pd.DataFrame) -> float:
    """Metric 1: Maximum absolute Z-error during the Transition phase."""
    if transition_df.empty:
        return float("nan")
    return float(np.max(np.abs(transition_df["z_error"].values)))


def compute_rmse_transition(transition_df: pd.DataFrame) -> float:
    """Metric 2: Root Mean Square Error of altitude during the Transition phase."""
    if transition_df.empty:
        return float("nan")
    errors = transition_df["z_error"].values
    return float(np.sqrt(np.mean(errors**2)))


def compute_noise_gate_latency(df: pd.DataFrame) -> float:
    """Metric 3: Time (ms) between t=10.0s and first noise_gate_active==True.

    Strategy
    ────────
    1. Find the row closest to t=10.0 s (noise injection start).
    2. Find the first row after t=10.0 s where noise_gate_active is True.
    3. Return the difference in milliseconds.

    Returns NaN if the noise gate never activates.
    """
    # Row closest to t=10.0 s
    idx_t10 = (df["timestamp"] - T_HOVER_END).abs().idxmin()
    t_injection_start = df.loc[idx_t10, "timestamp"]

    # First activation at or after the injection start
    after_t10 = df[df["timestamp"] >= t_injection_start]
    gate_active = after_t10[after_t10["noise_gate_active"] == True]  # noqa: E712

    if gate_active.empty:
        return float("nan")

    t_first_gate = gate_active.iloc[0]["timestamp"]
    latency_ms = (t_first_gate - t_injection_start) * 1000.0
    return float(latency_ms)


def compute_dnn_correction_factor(transition_df: pd.DataFrame) -> tuple[float, float]:
    """Metric 4: Average and peak DNN covariance_scale_factor during Transition.

    Returns (mean_delta_r, peak_delta_r).
    """
    if transition_df.empty:
        return float("nan"), float("nan")
    vals = transition_df["covariance_scale_factor"].values
    return float(np.mean(vals)), float(np.max(vals))


def compute_filter_stability(peak_z_error: float) -> str:
    """Metric 5: Stability verdict based on Peak Z-Error threshold."""
    if np.isnan(peak_z_error):
        return "UNKNOWN"
    return "Stable" if peak_z_error < STABILITY_THRESHOLD_M else "Divergent"


# ═══════════════════════════════════════════════════════════════════════════════
#  Formatted Table Output
# ═══════════════════════════════════════════════════════════════════════════════


def print_metrics_table(metrics: dict) -> None:
    """Print a publication-style metrics table to stdout."""
    # ── Column widths ──────────────────────────────────────────────────────────
    w_metric = 38
    w_value = 18
    w_unit = 14
    total = w_metric + w_value + w_unit + 6  # 6 for borders + padding

    def row(metric: str, value: str, unit: str) -> str:
        return f"│ {metric:<{w_metric}} │ {value:>{w_value}} │ {unit:<{w_unit}} │"

    print()
    print(f"╔{'═' * (total - 2)}╗")
    print(f"║{'  TABLE 1 — Neuro-Adaptive EKF Performance Metrics':^{total - 2}}║")
    print(f"╠{'═' * (total - 2)}╣")
    print(row("Metric", "Value", "Unit"))
    print(f"╠{'═' * (total - 2)}╣")

    # 1. Peak Z-Error
    peak_err = metrics["peak_z_error"]
    print(
        row(
            "1. Peak Z-Error (Transition)",
            f"{peak_err:.4f}" if not np.isnan(peak_err) else "N/A",
            "m",
        )
    )
    print(f"├{'─' * (total - 2)}┤")

    # 2. RMSE
    rmse = metrics["rmse_transition"]
    print(
        row(
            "2. RMSE — Transition Phase",
            f"{rmse:.4f}" if not np.isnan(rmse) else "N/A",
            "m",
        )
    )
    print(f"├{'─' * (total - 2)}┤")

    # 3. Noise Gate Latency
    latency = metrics["noise_gate_latency_ms"]
    print(
        row(
            "3. Noise Gate Latency",
            f"{latency:.2f}" if not np.isnan(latency) else "N/A (gate never fired)",
            "ms",
        )
    )
    print(f"├{'─' * (total - 2)}┤")

    # 4. DNN Correction Factor
    mean_dr = metrics["dnn_correction_mean"]
    peak_dr = metrics["dnn_correction_peak"]
    print(
        row(
            "4. DNN Correction Factor ΔR/R₀ (mean)",
            f"{mean_dr:.4f}" if not np.isnan(mean_dr) else "N/A",
            "dimensionless",
        )
    )
    print(
        row(
            "   DNN Correction Factor ΔR/R₀ (peak)",
            f"{peak_dr:.4f}" if not np.isnan(peak_dr) else "N/A",
            "dimensionless",
        )
    )
    print(f"├{'─' * (total - 2)}┤")

    # 5. Filter Stability
    stability = metrics["filter_stability"]
    stability_icon = "✓" if stability == "Stable" else "✗"
    print(
        row(
            "5. Filter Stability",
            f"{stability_icon}  {stability}",
            f"threshold < {STABILITY_THRESHOLD_M} m",
        )
    )

    print(f"╚{'═' * (total - 2)}╝")
    print()

    # ── Additional context ─────────────────────────────────────────────────────
    print("  Phase boundaries used:")
    print(f"    Hover      :  0.0 s – {T_HOVER_END:.1f} s")
    print(
        f"    Transition : {T_HOVER_END:.1f} s – {T_TRANSITION_END:.1f} s  ← metrics computed here"
    )
    print(f"    Cruise     : {T_TRANSITION_END:.1f} s – 30.0 s")
    print()


# ═══════════════════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════════════════


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute Neuro-Adaptive EKF performance metrics from flight_data_log.csv"
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=Path("flight_data_log.csv"),
        help="Path to the CSV file produced by data_logger.py (default: ./flight_data_log.csv)",
    )
    args = parser.parse_args()

    # ── Load data ─────────────────────────────────────────────────────────────
    df = load_and_validate(args.csv)

    # ── Slice Transition phase ────────────────────────────────────────────────
    transition_mask = (df["timestamp"] > T_HOVER_END) & (
        df["timestamp"] < T_TRANSITION_END
    )
    transition_df = df[transition_mask].copy()

    if transition_df.empty:
        print(
            f"[WARNING] No data found in Transition phase "
            f"({T_HOVER_END}s < t < {T_TRANSITION_END}s). "
            "Check that the CSV covers a full cycle.",
            file=sys.stderr,
        )

    print(
        f"[INFO] Transition phase rows: {len(transition_df):,} "
        f"({transition_df['timestamp'].min():.3f}s → "
        f"{transition_df['timestamp'].max():.3f}s)"
    )

    # ── Compute metrics ───────────────────────────────────────────────────────
    peak_z_error = compute_peak_z_error(transition_df)
    rmse = compute_rmse_transition(transition_df)
    latency_ms = compute_noise_gate_latency(df)
    mean_dr, peak_dr = compute_dnn_correction_factor(transition_df)
    stability = compute_filter_stability(peak_z_error)

    metrics = {
        "peak_z_error": peak_z_error,
        "rmse_transition": rmse,
        "noise_gate_latency_ms": latency_ms,
        "dnn_correction_mean": mean_dr,
        "dnn_correction_peak": peak_dr,
        "filter_stability": stability,
    }

    # ── Print formatted table ─────────────────────────────────────────────────
    print_metrics_table(metrics)

    # ── Also save metrics to a JSON sidecar for programmatic use ─────────────
    import json

    sidecar_path = args.csv.with_suffix(".metrics.json")
    with open(sidecar_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                k: (v if not (isinstance(v, float) and np.isnan(v)) else None)
                for k, v in metrics.items()
            },
            f,
            indent=2,
        )
    print(f"  Metrics also saved to: {sidecar_path}")


if __name__ == "__main__":
    main()
