"""
Hyperparameter Sweep
Runs 10 hyperparameter combinations for each of the four algorithms
(DQN, REINFORCE-style PPO, PPO, A2C) and records results to a CSV file
for comparison in the report.
"""

import os
import csv
import time
import pandas as pd

from training.dqn_training import train_dqn
from training.pg_training import train_reinforce, train_ppo, train_a2c

RESULTS_CSV = "logs/hyperparameter_results.csv"

def summarize_run(log_dir, run_name, n_episodes=20):
    """Read a run's Monitior log and summarice its last N episodes."""
    monitor_path = f"{log_dir}/{run_name}.monitor.csv"

    if not os.path.exists(monitor_path):
        return {"mean_reward": None, "mean_ep_length": None}

    df = pd.read_csv(monitor_path, skiprows=1)
    tail = df.tail(n_episodes)

    return {
        "mean_reward": tail["r"].mean(),
        "mean_ep_length": tail["l"].mean(),
    }

def append_result(row: dict):
    """Append one run's results as a new row in the results CSV (creates file and header if needed)"""
    os.makedirs("logs", exist_ok=True)
    file_exists = os.path.exists(RESULTS_CSV)

    with open(RESULTS_CSV, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=row.keys())
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


DQN_CONFIGS = [
    {"learning_rate": 1e-4, "exploration_fraction": 0.1},
    {"learning_rate": 5e-4, "exploration_fraction": 0.1},
    {"learning_rate": 1e-3, "exploration_fraction": 0.1},
    {"learning_rate": 5e-3, "exploration_fraction": 0.1},
    {"learning_rate": 1e-2, "exploration_fraction": 0.1},
    {"learning_rate": 1e-3, "exploration_fraction": 0.05},
    {"learning_rate": 1e-3, "exploration_fraction": 0.2},
    {"learning_rate": 1e-3, "exploration_fraction": 0.4},
    {"learning_rate": 5e-4, "exploration_fraction": 0.2},
    {"learning_rate": 5e-3, "exploration_fraction": 0.3},
]


def run_dqn_sweep():
    for i, config in enumerate(DQN_CONFIGS):
        run_name = f"dqn_run{i+1}"
        print(f"\n=== DQN sweep {i+1}/10: {config} ===")

        start = time.time()
        train_dqn(
            total_timesteps=50_000,
            learning_rate=config["learning_rate"],
            exploration_fraction=config["exploration_fraction"],
            run_name=run_name,
        )
        elapsed = time.time() - start

        summary = summarize_run("logs/dqn", run_name)

        append_result({
            "algorithm": "DQN",
            "run_name": run_name,
            "learning_rate": config["learning_rate"],
            "exploration_fraction": config["exploration_fraction"],
            "mean_reward_last20": summary["mean_reward"],
            "mean_ep_length_last20": summary["mean_ep_length"],
            "training_time_sec": round(elapsed, 1),
        })


REINFORCE_CONFIGS = [
    {"learning_rate": 1e-4, "n_steps": 128},
    {"learning_rate": 5e-4, "n_steps": 128},
    {"learning_rate": 3e-4, "n_steps": 128},  
    {"learning_rate": 1e-3, "n_steps": 128},
    {"learning_rate": 5e-3, "n_steps": 128},
    {"learning_rate": 3e-4, "n_steps": 64},
    {"learning_rate": 3e-4, "n_steps": 256},
    {"learning_rate": 3e-4, "n_steps": 512},
    {"learning_rate": 1e-3, "n_steps": 256},
    {"learning_rate": 1e-4, "n_steps": 512},
]

PPO_CONFIGS = [
    {"learning_rate": 1e-4, "clip_range": 0.1, "ent_coef": 0.0},
    {"learning_rate": 3e-4, "clip_range": 0.2, "ent_coef": 0.01},  # same as our earlier default run
    {"learning_rate": 1e-3, "clip_range": 0.2, "ent_coef": 0.01},
    {"learning_rate": 3e-4, "clip_range": 0.1, "ent_coef": 0.01},
    {"learning_rate": 3e-4, "clip_range": 0.3, "ent_coef": 0.01},
    {"learning_rate": 3e-4, "clip_range": 0.2, "ent_coef": 0.0},
    {"learning_rate": 3e-4, "clip_range": 0.2, "ent_coef": 0.05},
    {"learning_rate": 5e-4, "clip_range": 0.3, "ent_coef": 0.02},
    {"learning_rate": 1e-3, "clip_range": 0.1, "ent_coef": 0.0},
    {"learning_rate": 1e-4, "clip_range": 0.3, "ent_coef": 0.05},
]

A2C_CONFIGS = [
    {"learning_rate": 1e-4, "n_steps": 5},
    {"learning_rate": 7e-4, "n_steps": 5},   # same as our earlier default run
    {"learning_rate": 1e-3, "n_steps": 5},
    {"learning_rate": 7e-4, "n_steps": 10},
    {"learning_rate": 7e-4, "n_steps": 20},
    {"learning_rate": 7e-4, "n_steps": 50},
    {"learning_rate": 1e-4, "n_steps": 20},
    {"learning_rate": 1e-3, "n_steps": 50},
    {"learning_rate": 5e-4, "n_steps": 10},
    {"learning_rate": 5e-4, "n_steps": 50},
]


def run_pg_sweep():
    for i, config in enumerate(REINFORCE_CONFIGS):
        run_name = f"reinforce_run{i+1}"
        print(f"\n=== REINFORCE sweep {i+1}/10: {config} ===")
        start = time.time()
        train_reinforce(total_timesteps=50_000, run_name=run_name, **config)
        elapsed = time.time() - start
        summary = summarize_run("logs/pg", run_name)
        append_result({
            "algorithm": "REINFORCE", "run_name": run_name,
            **config,
            "mean_reward_last20": summary["mean_reward"],
            "mean_ep_length_last20": summary["mean_ep_length"],
            "training_time_sec": round(elapsed, 1),
        })

    for i, config in enumerate(PPO_CONFIGS):
        run_name = f"ppo_run{i+1}"
        print(f"\n=== PPO sweep {i+1}/10: {config} ===")
        start = time.time()
        train_ppo(total_timesteps=50_000, run_name=run_name, **config)
        elapsed = time.time() - start
        summary = summarize_run("logs/pg", run_name)
        append_result({
            "algorithm": "PPO", "run_name": run_name,
            **config,
            "mean_reward_last20": summary["mean_reward"],
            "mean_ep_length_last20": summary["mean_ep_length"],
            "training_time_sec": round(elapsed, 1),
        })

    for i, config in enumerate(A2C_CONFIGS):
        run_name = f"a2c_run{i+1}"
        print(f"\n=== A2C sweep {i+1}/10: {config} ===")
        start = time.time()
        train_a2c(total_timesteps=50_000, run_name=run_name, **config)
        elapsed = time.time() - start
        summary = summarize_run("logs/pg", run_name)
        append_result({
            "algorithm": "A2C", "run_name": run_name,
            **config,
            "mean_reward_last20": summary["mean_reward"],
            "mean_ep_length_last20": summary["mean_ep_length"],
            "training_time_sec": round(elapsed, 1),
        })

if __name__ == "__main__":
    overall_start = time.time()

    print("Starting DQN sweep (10 runs)...")
    run_dqn_sweep()

    print("\nStarting policy-gradient sweep (30 runs: REINFORCE, PPO, A2C)...")
    run_pg_sweep()

    overall_elapsed = time.time() - overall_start
    print(f"\nAll 40 runs complete in {overall_elapsed/60:.1f} minutes.")
    print(f"Results saved to {RESULTS_CSV}")