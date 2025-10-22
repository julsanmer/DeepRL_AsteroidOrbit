import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
from glob import glob


million = 1e6


# Use LaTeX-like font
rc('font', family='serif', serif=['Computer Modern Roman'])
rc('text', usetex=True)
fontsize = 18
fontsize_legend = 15


def plot_reward2():
    # Directory containing multiple SAC monitor logs (one per seed)
    log_dir = "results/logs/"

    # Find all monitor files
    log_files = glob(os.path.join(log_dir, "*.monitor.csv"))

    data = []
    for f in log_files:
        df = pd.read_csv(f, comment="#")   # SB3 monitor has header comments
        # Cumulative timesteps = sum of episode lengths
        df["timesteps"] = np.cumsum(df["l"])
        data.append(df)

    # --- Resample to common x-axis (timesteps) ---
    # choose evaluation grid
    max_steps = min([df["timesteps"].max() for df in data])  # align on shortest run
    grid = np.linspace(0, max_steps, 200)

    interp_rewards = []
    for df in data:
        interp = np.interp(grid, df["timesteps"], df["r"])  # interpolate rewards
        interp_rewards.append(interp)

    interp_rewards = np.array(interp_rewards)
    mean = interp_rewards.mean(axis=0)
    std = interp_rewards.std(axis=0)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(grid, mean, label="Mean", color="blue")
    ax.fill_between(grid, mean-std, mean+std, color="blue", alpha=0.3, label="±1 std")
    ax.axhline(0, color="red", linestyle="--", linewidth=1.5, label="Max reward")
    ax.set_xlabel("Environment steps [-]", fontsize=fontsize)
    ax.set_ylabel("Episode total reward [-]", fontsize=fontsize)
    ax.set_xlim(0, grid[-1])
    ax.set_ylim(-8, 1)
    ax.legend(loc='lower right', fontsize=fontsize_legend)
    ax.tick_params(axis='x', labelsize=fontsize)
    ax.tick_params(axis='y', labelsize=fontsize)
    ax.grid()
    fig.tight_layout()
    plt.savefig("images/reward.pdf", bbox_inches="tight")


def plot_reward():
    # Directory containing multiple SAC monitor logs (one per seed)
    log_dir = "results/logs/"

    # Find all monitor files
    log_files = glob(os.path.join(log_dir, "*.monitor.csv"))

    data = []
    for f in log_files:
        df = pd.read_csv(f, comment="#")  # SB3 monitor has header comments
        df["timesteps"] = np.cumsum(df["l"])  # cumulative timesteps
        data.append(df)

    # --- Resample to a common x-axis (timesteps) ---
    max_steps = min(df["timesteps"].max() for df in data)  # align to shortest run
    grid = np.linspace(0, max_steps, 200)

    interp_rewards = []
    for df in data:
        interp = np.interp(grid, df["timesteps"], df["r"])  # interpolate rewards
        interp_rewards.append(interp)

    interp_rewards = np.array(interp_rewards)

    # Compute mean, min, and max across seeds
    mean = interp_rewards.mean(axis=0)
    min_r = interp_rewards.min(axis=0)
    max_r = interp_rewards.max(axis=0)

    # --- Plot ---
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(grid/million, mean, label="Mean", color="blue")
    ax.fill_between(grid/million, min_r, max_r, color="blue", alpha=0.3, label="Min–Max range")
    ax.axhline(0, color="red", linestyle="--", linewidth=2, label="Max. reward")
    ax.set_xlabel("Million steps [-]", fontsize=fontsize)
    ax.set_ylabel("Episode total reward [-]", fontsize=fontsize)
    ax.set_xlim(0, grid[-1]/million)
    ax.legend(loc='lower right', fontsize=fontsize_legend)
    ax.tick_params(axis='x', labelsize=fontsize)
    ax.tick_params(axis='y', labelsize=fontsize)
    ax.grid()
    fig.tight_layout()
    os.makedirs("images", exist_ok=True)
    plt.savefig("images/reward.pdf", bbox_inches="tight")
    plt.show()
