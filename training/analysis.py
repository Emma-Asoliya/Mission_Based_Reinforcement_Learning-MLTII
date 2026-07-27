"""
Analysis Script
================
Generates the plots needed for the report's Results Discussion section:
  - Cumulative reward curves (all 4 methods, best run each)
  - DQN loss curve / PG entropy curves (stability)
  - Convergence comparison (episodes to reach stable performance)
  - Generalization test (best models on unseen seeds)
"""

import os
import pandas as pd
import matplotlib.pyplot as plt

RESULTS_CSV = "logs/hyperparameter_results.csv"
PLOTS_DIR = "logs/plots"
os.makedirs(PLOTS_DIR, exist_ok=True)

# The results CSV has different column counts per algorithm, so we read it
# as raw rows rather than a single rectangular table.
COLUMN_NAMES = {
    "DQN": ["algorithm", "run_name", "learning_rate", "exploration_fraction",
            "mean_reward_last20", "mean_ep_length_last20", "training_time_sec"],
    "REINFORCE": ["algorithm", "run_name", "learning_rate", "n_steps",
                  "mean_reward_last20", "mean_ep_length_last20", "training_time_sec"],
    "PPO": ["algorithm", "run_name", "learning_rate", "clip_range", "ent_coef",
            "mean_reward_last20", "mean_ep_length_last20", "training_time_sec"],
    "A2C": ["algorithm", "run_name", "learning_rate", "n_steps",
            "mean_reward_last20", "mean_ep_length_last20", "training_time_sec"],
}


def load_results_by_algorithm():
    """Read the ragged CSV and split it into one clean DataFrame per algorithm."""
    with open(RESULTS_CSV, "r") as f:
        lines = [line.strip().split(",") for line in f.readlines() if line.strip()]

    by_algo = {"DQN": [], "REINFORCE": [], "PPO": [], "A2C": []}
    for row in lines:
        by_algo[row[0]].append(row)

    dfs = {}
    for algo, rows in by_algo.items():
        df = pd.DataFrame(rows, columns=COLUMN_NAMES[algo])
        # convert numeric columns from strings to actual numbers
        for col in df.columns:
            if col not in ("algorithm", "run_name"):
                df[col] = pd.to_numeric(df[col])
        dfs[algo] = df

    return dfs


def find_best_runs(dfs):
    """Return the run_name with the highest mean reward for each algorithm."""
    best = {}
    for algo, df in dfs.items():
        best_row = df.loc[df["mean_reward_last20"].idxmax()]
        best[algo] = best_row["run_name"]
    return best

def load_full_log(algo, run_name):
    """Load the complete episode-by-episode training log for one run."""
    log_dir = "logs/dqn" if algo == "DQN" else "logs/pg"
    monitor_path = f"{log_dir}/{run_name}.monitor.csv"
    df = pd.read_csv(monitor_path, skiprows=1)
    return df


def plot_cumulative_rewards(dfs):
    """Plot cumulative reward over episodes for each algorithm's best run, as subplots."""
    best_runs = find_best_runs(dfs)

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes = axes.flatten()

    for ax, (algo, run_name) in zip(axes, best_runs.items()):
        log = load_full_log(algo, run_name)
        cumulative_reward = log["r"].cumsum()

        ax.plot(cumulative_reward, color="tab:blue")
        ax.set_title(f"{algo} (best run: {run_name})")
        ax.set_xlabel("Episode")
        ax.set_ylabel("Cumulative Reward")
        ax.grid(alpha=0.3)

    plt.tight_layout()
    save_path = f"{PLOTS_DIR}/cumulative_rewards.png"
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Saved {save_path}")

import glob
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

def extract_tb_scalar(log_dir, run_name, tag):
    """Pull one scalar metric's full history out of a TensorBoard log file."""
    matches = sorted(glob.glob(f"{log_dir}/tensorboard/{run_name}_*"))
    if not matches:
        return None
    tb_path = matches[-1]  # pick the latest timestamped folder

    ea = EventAccumulator(tb_path)
    ea.Reload()

    if tag not in ea.Tags().get("scalars", []):
        return None

    events = ea.Scalars(tag)
    steps = [e.step for e in events]
    values = [e.value for e in events]
    return steps, values

def plot_stability(dfs):
    """Plot DQNs loss curve and PG methods entropy curves (training stability)."""
    best_runs = find_best_runs(dfs)

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes = axes.flatten()

    dqn_run = best_runs["DQN"]
    steps, values = extract_tb_scalar("logs/dqn", dqn_run, "train/loss")
    axes[0].plot(steps, values, color="tab:red")
    axes[0].set_title(f"DQN Loss Curve (best run: {dqn_run})")
    axes[0].set_xlabel("Training Step")
    axes[0].set_ylabel("Loss")
    axes[0].grid(alpha=0.3)

    for ax, algo in zip(axes[1:], ["REINFORCE", "PPO", "A2C"]):
        run_name = best_runs[algo]
        log_dir = "logs/pg"
        steps, values = extract_tb_scalar(log_dir, run_name, "train/entropy_loss")
        ax.plot(steps, values, color="tab:green")
        ax.set_title(f"{algo} Entropy Curve (best run: {run_name})")
        ax.set_xlabel("Training Step")
        ax.set_ylabel("Entropy Loss")
        ax.grid(alpha=0.3)

    plt.tight_layout()
    save_path = f"{PLOTS_DIR}/stability.png"
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Saved {save_path}")

def compute_convergence_episode(rewards, window=20, threshold=0.9):
    """
    Find the first episode where a rolling-average reward first reaches
    threshold * (final rolling-average reward) and stays there.
    """
    rolling = rewards.rolling(window=window).mean()
    final_level = rolling.iloc[-window:].mean() 

    target = threshold * final_level

    for i in range(len(rolling)):
        if rolling.iloc[i] >= target:
            return i

    return len(rolling)  # never quite stabilized


def plot_convergence(dfs):
    """Compare how many episodes each algorithm needed to reach stable performance."""
    best_runs = find_best_runs(dfs)
    convergence_episodes = {}

    for algo, run_name in best_runs.items():
        log_dir = "logs/dqn" if algo == "DQN" else "logs/pg"
        log = load_full_log(algo, run_name)
        convergence_episodes[algo] = compute_convergence_episode(log["r"])

    fig, ax = plt.subplots(figsize=(8, 5))
    algos = list(convergence_episodes.keys())
    values = list(convergence_episodes.values())

    bars = ax.bar(algos, values, color=["tab:red", "tab:orange", "tab:blue", "tab:green"])
    ax.set_ylabel("Episodes to Converge")
    ax.set_title("Episodes Required to Reach Stable Performance (90% of final level)")
    ax.grid(alpha=0.3, axis="y")

    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2, val + 1, str(val),
                ha="center", va="bottom")

    plt.tight_layout()
    save_path = f"{PLOTS_DIR}/convergence.png"
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Saved {save_path}")

    return convergence_episodes

from stable_baselines3 import DQN, PPO, A2C
from environment.custom_env import MentalHealthTriageEnv

MODEL_PATHS = {
    "DQN": "models/dqn",
    "REINFORCE": "models/pg",
    "PPO": "models/pg",
    "A2C": "models/pg",
}

ALGO_CLASSES = {
    "DQN": DQN,
    "REINFORCE": PPO,   # REINFORCE-style runs are PPO objects under the hood
    "PPO": PPO,
    "A2C": A2C,
}

UNSEEN_SEEDS = [1001, 1002, 1003, 1004, 1005]  # seeds never used during training


def test_generalization(dfs):
    """Run each algorithm's best model on unseen seeds and compare performance."""
    best_runs = find_best_runs(dfs)
    results = {}

    for algo, run_name in best_runs.items():
        model_path = f"{MODEL_PATHS[algo]}/{run_name}"
        model = ALGO_CLASSES[algo].load(model_path)

        seed_rewards = []
        for seed in UNSEEN_SEEDS:
            env = MentalHealthTriageEnv(max_requests=50, max_steps=200)
            obs, info = env.reset(seed=seed)
            terminated = truncated = False
            total_reward = 0.0

            while not (terminated or truncated):
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = env.step(int(action))
                total_reward += reward

            seed_rewards.append(total_reward)

        results[algo] = {
            "mean_reward": sum(seed_rewards) / len(seed_rewards),
            "min_reward": min(seed_rewards),
            "max_reward": max(seed_rewards),
            "per_seed": seed_rewards,
        }

    return results


def plot_generalization(results):
    """Bar chart comparing training performance vs. unseen-seed performance."""
    fig, ax = plt.subplots(figsize=(8, 5))
    algos = list(results.keys())
    means = [results[a]["mean_reward"] for a in algos]
    mins = [results[a]["min_reward"] for a in algos]
    maxs = [results[a]["max_reward"] for a in algos]

    errors = [[m - lo for m, lo in zip(means, mins)],
              [hi - m for m, hi in zip(means, maxs)]]

    ax.bar(algos, means, yerr=errors, capsize=5,
           color=["tab:red", "tab:orange", "tab:blue", "tab:green"])
    ax.set_ylabel("Reward on Unseen Seeds")
    ax.set_title("Generalization: Performance on 5 Unseen Initial States")
    ax.grid(alpha=0.3, axis="y")

    plt.tight_layout()
    save_path = f"{PLOTS_DIR}/generalization.png"
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Saved {save_path}")

if __name__ == "__main__":
    print("Loading hyperparameter sweep results...")
    dfs = load_results_by_algorithm()

    best_runs = find_best_runs(dfs)
    print("\nBest run per algorithm:")
    for algo, run_name in best_runs.items():
        best_reward = dfs[algo].loc[dfs[algo]["run_name"] == run_name, "mean_reward_last20"].values[0]
        print(f"  {algo}: {run_name} (reward = {best_reward:.2f})")

    print("\nGenerating cumulative reward plot...")
    plot_cumulative_rewards(dfs)

    print("Generating stability plot (DQN loss + PG entropy)...")
    plot_stability(dfs)

    print("Generating convergence comparison...")
    convergence = plot_convergence(dfs)
    print("Episodes to converge:", convergence)

    print("Testing generalization on unseen seeds...")
    gen_results = test_generalization(dfs)
    for algo, res in gen_results.items():
        print(f"  {algo}: mean={res['mean_reward']:.2f}, "
              f"min={res['min_reward']:.2f}, max={res['max_reward']:.2f}")
    plot_generalization(gen_results)

    print(f"\nAll plots saved to {PLOTS_DIR}/")