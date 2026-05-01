"""
Compare two simulation runs — 4-subplot visualization.

Plots:
  1. Angle θ (degrees)
  2. Position x (m)
  3. Cart velocity ẋ (m/s)
  4. Angular velocity θ̇ (rad/s)
"""

import argparse
import json
import os
import sys

import numpy as np
import matplotlib.pyplot as plt
import yaml


def load_sim_data(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    frames = data["data"]
    dt = data["metadata"]["dt"]
    return {
        "time_s": np.array([f["time_s"] for f in frames]),
        "x": np.array([f["x"] for f in frames]),
        "x_dot": np.array([f["x_dot"] for f in frames]),
        "theta_deg": np.array([f["theta_deg"] for f in frames]),
        "theta_dot": np.array([f["theta_dot"] for f in frames]),
        "dt": dt,
        "metadata": data["metadata"],
    }


def compare_runs(config):
    cfg = config["compare"]
    run_1 = load_sim_data(cfg["run_1"]["path"])
    run_2 = load_sim_data(cfg["run_2"]["path"])
    label_1 = cfg["run_1"].get("label", "Run 1")
    label_2 = cfg["run_2"].get("label", "Run 2")
    t_start = cfg.get("time_start") or 0.0
    t_end = cfg.get("time_end")

    def filter_time(run, t_start, t_end):
        mask = run["time_s"] >= t_start
        if t_end is not None:
            mask &= run["time_s"] <= t_end
        return {k: (v[mask] if isinstance(v, np.ndarray) else v) for k, v in run.items()}

    run_1 = filter_time(run_1, t_start, t_end)
    run_2 = filter_time(run_2, t_start, t_end)

    fig, axes = plt.subplots(4, 1, figsize=tuple(cfg.get("figure_size", [14, 10])), sharex=True)
    colors = ["#2196F3", "#F44336"]
    series = [
        ("theta_deg", "Angle θ (°)"),
        ("x", "Position x (m)"),
        ("x_dot", "Cart velocity ẋ (m/s)"),
        ("theta_dot", "Angular velocity θ̇ (rad/s)"),
    ]
    for i, (key, ylabel) in enumerate(series):
        ax = axes[i]
        ax.plot(run_1["time_s"], run_1[key], color=colors[0], linewidth=1.2, label=label_1)
        ax.plot(run_2["time_s"], run_2[key], color=colors[1], linewidth=1.2, alpha=0.8, label=label_2)
        ax.axhline(0, color="gray", linestyle="--", alpha=0.3)
        ax.set_ylabel(ylabel)
        ax.legend(loc="upper right", fontsize=9)
        ax.grid(alpha=0.3)
    axes[0].set_title("Simulation Comparison", fontweight="bold")
    axes[-1].set_xlabel("Time (s)")
    fig.tight_layout()

    save_path = cfg.get("save_path")
    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        fig.savefig(save_path, dpi=cfg.get("dpi", 150))
        print(f"Saved comparison plot to {save_path}")
    else:
        plt.show()
    plt.close(fig)


def main():
    default_config = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "configs", "compare_config.yaml"))
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=default_config)
    args = parser.parse_args()
    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    compare_runs(config)


if __name__ == "__main__":
    main()
