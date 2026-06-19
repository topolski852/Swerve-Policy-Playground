# ──────────────────────────────────────────────────────────────────────────────
# plot_rewards.py
# Plot episode reward curves from a rewards CSV produced by train.py or
# train_scoring.py.
#
# Usage:
#   python plot_rewards.py                                  # auto-finds latest CSV
#   python plot_rewards.py path_following/logs/rewards_*.csv
#   python plot_rewards.py fuel_scoring/logs/rewards_*.csv
# ──────────────────────────────────────────────────────────────────────────────

import os
import sys
import glob
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


def rolling_mean(values, window):
    if len(values) < window:
        return values
    kernel = np.ones(window) / window
    return np.convolve(values, kernel, mode="valid")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("csv", nargs="?", default=None,
                        help="Path to rewards CSV (auto-detects latest if omitted)")
    parser.add_argument("--window", type=int, default=50,
                        help="Rolling average window in episodes (default: 50)")
    args = parser.parse_args()

    if args.csv:
        csv_path = args.csv
    else:
        # Search all experiment log directories
        candidates = (
            sorted(glob.glob("path_following/logs/rewards_*.csv")) +
            sorted(glob.glob("fuel_scoring/logs/rewards_*.csv")) +
            sorted(glob.glob("logs/rewards_*.csv"))   # legacy location
        )
        if not candidates:
            print("No reward logs found. Run train.py or train_scoring.py first.")
            sys.exit(1)
        csv_path = candidates[-1]
        print(f"Auto-selected: {csv_path}")

    data = np.loadtxt(csv_path, delimiter=",", skiprows=1)
    if data.ndim == 1:
        data = data[np.newaxis, :]

    timesteps = data[:, 0]
    rewards   = data[:, 1]

    fig, ax = plt.subplots(figsize=(12, 5))
    fig.patch.set_facecolor("#1e1e1e")
    ax.set_facecolor("#2a2a2a")

    ax.plot(timesteps, rewards, color="#4a90d9", alpha=0.25, linewidth=0.8,
            label="Episode reward (raw)")

    if len(rewards) >= args.window:
        smooth_rewards = rolling_mean(rewards, args.window)
        smooth_ts = timesteps[args.window - 1:]
        ax.plot(smooth_ts, smooth_rewards, color="#7ec8e3", linewidth=2.0,
                label=f"Rolling mean ({args.window} eps)")

    ax.set_xlabel("Timesteps", color="#cccccc")
    ax.set_ylabel("Episode Reward", color="#cccccc")
    ax.set_title(f"Swerve SAC Training — {os.path.basename(csv_path)}",
                 color="#eeeeee", fontsize=13)
    ax.tick_params(colors="#aaaaaa")
    for spine in ax.spines.values():
        spine.set_edgecolor("#555555")
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(
        lambda x, _: f"{int(x/1000)}k" if x >= 1000 else str(int(x))
    ))
    ax.legend(facecolor="#333333", edgecolor="#555555", labelcolor="#cccccc")
    ax.grid(True, color="#3a3a3a", linewidth=0.5)

    plt.tight_layout()

    out_path = csv_path.replace(".csv", ".png")
    plt.savefig(out_path, dpi=150, facecolor=fig.get_facecolor())
    print(f"Saved plot to: {out_path}")
    plt.show()


if __name__ == "__main__":
    main()
