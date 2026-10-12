"""Headless degradation plots; no training or inference side effects."""
from __future__ import annotations
import os
from pathlib import Path
import matplotlib
if not os.environ.get("MPLBACKEND"):
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
from .config import BASE_DIR, REVERSE_CLASS_MAPS

def plot_heat_exchanger_degradation(df: pd.DataFrame, save_path: Path | None = None) -> None:
    """Line chart of ACTUAL_UA and DUTY_kW over Time_Step (degradation trend)."""
    fig, ax1 = plt.subplots(figsize=(10, 5))

    color_ua = "#1f77b4"
    color_duty = "#ff7f0e"

    ax1.set_xlabel("Time Step")
    ax1.set_ylabel("ACTUAL_UA (kJ/C-h)", color=color_ua)
    ax1.plot(
        df["Time_Step"],
        df["ACTUAL_UA"],
        color=color_ua,
        marker="o",
        linewidth=2,
        label="ACTUAL_UA",
    )
    ax1.tick_params(axis="y", labelcolor=color_ua)
    ax1.grid(True, alpha=0.3)

    ax2 = ax1.twinx()
    ax2.set_ylabel("Heat duty (kW)", color=color_duty)
    ax2.plot(
        df["Time_Step"],
        df["DUTY_kW"],
        color=color_duty,
        marker="s",
        linewidth=2,
        linestyle="--",
        label="DUTY_kW",
    )
    ax2.tick_params(axis="y", labelcolor=color_duty)

    # Highlight health-state regions.
    status_labels = REVERSE_CLASS_MAPS["heat_exchanger"]
    for status_code in sorted(df["Target_Status"].unique()):
        mask = df["Target_Status"] == status_code
        if mask.any():
            ax1.axvspan(
                df.loc[mask, "Time_Step"].min(),
                df.loc[mask, "Time_Step"].max(),
                alpha=0.08,
                label=status_labels[status_code],
            )

    fig.suptitle(
        "Heat Exchanger - ACTUAL_UA and DUTY_kW Degradation Over Time",
        fontsize=13,
        fontweight="bold",
    )
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved heat exchanger EDA plot -> {save_path}")
        plt.close(fig)
    else:
        plt.show()


def plot_pump_surge_behavior(df: pd.DataFrame, save_path: Path | None = None) -> None:
    """
    Dual-axis plot: Pump_Efficiency (declining) vs surging Pump_Duty_kW
    and Outlet_Temp_C.
    """
    fig, ax1 = plt.subplots(figsize=(10, 5))

    color_eff = "#2ca02c"
    color_duty = "#d62728"
    color_temp = "#9467bd"

    ax1.set_xlabel("Time Step")
    ax1.set_ylabel("Pump Efficiency (%)", color=color_eff)
    ax1.plot(
        df["Time_Step"],
        df["Pump_Efficiency"],
        color=color_eff,
        marker="o",
        linewidth=2,
        label="Pump Efficiency",
    )
    ax1.tick_params(axis="y", labelcolor=color_eff)
    ax1.grid(True, alpha=0.3)

    ax2 = ax1.twinx()
    ax2.set_ylabel("Pump Duty (kW) / Outlet Temp (C)")
    ax2.plot(
        df["Time_Step"],
        df["Pump_Duty_kW"],
        color=color_duty,
        marker="s",
        linewidth=2,
        linestyle="--",
        label="Pump_Duty_kW",
    )
    ax2.plot(
        df["Time_Step"],
        df["Outlet_Temp_C"],
        color=color_temp,
        marker="^",
        linewidth=2,
        linestyle=":",
        label="Outlet_Temp_C",
    )

    status_labels = REVERSE_CLASS_MAPS["centrifugal_pump"]
    for status_code in sorted(df["Target_Status"].unique()):
        mask = df["Target_Status"] == status_code
        if mask.any():
            ax1.axvspan(
                df.loc[mask, "Time_Step"].min(),
                df.loc[mask, "Time_Step"].max(),
                alpha=0.08,
                label=status_labels[status_code],
            )

    fig.suptitle(
        "Centrifugal Pump - Duty and Outlet Temp Surge as Efficiency Drops",
        fontsize=13,
        fontweight="bold",
    )
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved pump EDA plot -> {save_path}")
        plt.close(fig)
    else:
        plt.show()


def run_exploratory_analysis(
    hx_df: pd.DataFrame, pump_df: pd.DataFrame, output_dir: Path | None = None
) -> None:
    """Generate and optionally persist EDA plots for both assets."""
    out = output_dir or BASE_DIR / "outputs"
    out.mkdir(parents=True, exist_ok=True)
    plot_heat_exchanger_degradation(hx_df, out / "eda_heat_exchanger.png")
    plot_pump_surge_behavior(pump_df, out / "eda_centrifugal_pump.png")
