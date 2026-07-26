"""
DQN Training Script
Trains a DQN (value-based) agent on the MentalHealthTriageEnv using 
Stable-Baselines3, and saves the trained model and the training logs to disk.
"""

import os
from stable_baselines3 import DQN
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import EvalCallback\

from environment.custom_env import MentalHealthTriageEnv

MODEL_DIR = "models/dqn"
LOG_DIR = "logs/dqn"
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

def make_env(log_path=None):
    env = MentalHealthTriageEnv(max_requests=50, max_steps=200)
    env = Monitor(env, log_path if log_path else LOG_DIR)
    return env

def train_dqn(
        total_timesteps: int = 50_000,
        learning_rate: float = 1e-3,
        gamma: float = 0.99,
        buffer_size: int = 10_000,
        batch_size: int = 64,
        exploration_fraction: float = 0.1,
        run_name: str = "dqn_default",
):
    """Train a single DQN aget with the given hyperparameters."""

    env = make_env(log_path=f"{LOG_DIR}/{run_name}")
    model = DQN(
        policy="MlpPolicy",
        env=env,
        learning_rate=learning_rate,
        gamma=gamma,
        buffer_size=buffer_size,
        batch_size=batch_size,
        exploration_fraction=exploration_fraction,
        verbose=1,
        tensorboard_log=f"{LOG_DIR}/tensorboard",
    )

    model.learn(total_timesteps=total_timesteps, tb_log_name=run_name)

    save_path = f"{MODEL_DIR}/{run_name}"
    model.save(save_path)
    print(f"Model saved to {save_path}")

    return model

if __name__ == "__main__":
    train_dqn(
        total_timesteps=50_000,
        learning_rate=1e-3,
        gamma=0.99,
        buffer_size=10_000,
        batch_size=64,
        exploration_fraction=0.1,
        run_name="dqn_default",
    )