""" 
Policy Gradient Training Script
Trains three policy-based agents on the MentalHealthTriageEnv using
Stable-Baselines3:
REINFORCE-style PPO (PPO configured to approximate vanilla REINFORCE)
Standard PPO (Proximal Policy Optimization)
A2C (Advantage Actor-Critic)
"""

import os
from stable_baselines3 import PPO, A2C
from stable_baselines3.common.monitor import Monitor

from environment.custom_env import MentalHealthTriageEnv

MODEL_DIR_PG = "models/pg"
LOG_DIR_PG = "logs/pg"
os.makedirs(MODEL_DIR_PG, exist_ok=True)
os.makedirs(LOG_DIR_PG, exist_ok=True)

def make_env():
    env = MentalHealthTriageEnv(max_requests=50, max_steps=200)
    env = Monitor(env, LOG_DIR_PG)
    return env

def train_reinforce(
        total_timesteps: int = 50_000,
        learning_rate: float = 3e-4,
        gamma: float = 0.99,
        n_steps: int = 256,
        run_name: str = "reinforce_default",

):
    env = make_env()
    model = PPO(
        policy="MlpPolicy",
        env=env,
        learning_rate=learning_rate,
        gamma=gamma,
        n_steps=n_steps,
        n_epochs=1,
        clip_range=10.0,
        ent_coef=0.0,
        verbose=1,
        tensorboard_log=f"{LOG_DIR_PG}/tensorboard",
    )

    model.learn(total_timesteps=total_timesteps, tb_log_name=run_name)

    save_path = f"{MODEL_DIR_PG}/{run_name}"
    model.save(save_path)
    print(f"Saved model to {save_path}")

    return model

def train_ppo(
        total_timesteps: int = 50_000,
        learning_rate: float = 3e-4,
        gamma:float = 0.99,
        n_steps: int = 256,
        clip_range: float = 0.2,
        ent_coef: float = 0.01,
        run_name: str = "ppo_default",
):

    env = make_env()
    model = PPO(
        policy="MlpPolicy",
        env=env,
        learning_rate=learning_rate,
        gamma=gamma,
        n_steps=n_steps,
        clip_range=clip_range,
        ent_coef=ent_coef,
        verbose=1,
        tensorboard_log=f"{LOG_DIR_PG}/tensorboard",
    )
    model.learn(total_timesteps=total_timesteps, tb_log_name=run_name)

    save_path = f"{MODEL_DIR_PG}/{run_name}"
    model.save(save_path)
    print(f"Saved model to {save_path}")

    return model


def train_a2c(
        total_timesteps: int = 50_000,
        learning_rate: float = 7e-4,
        gamma: float = 0.99,
        n_steps: int = 5,
        ent_coef: float = 0.01,
        run_name: str = "a2c_default",
):
    """Train an A2C (Advantage Actor-Critic) agent on the MentalHealthTriageEnv."""

    env = make_env()
    model = A2C(
        policy="MlpPolicy",
        env=env,
        learning_rate=learning_rate,
        gamma=gamma,
        n_steps=n_steps,
        ent_coef=ent_coef,
        verbose=1,
        tensorboard_log=f"{LOG_DIR_PG}/tensorboard",
    )

    model.learn(total_timesteps=total_timesteps, tb_log_name=run_name)

    save_path = f"{MODEL_DIR_PG}/{run_name}"
    model.save(save_path)
    print(f"Saved model to {save_path}")

    return model


if __name__ == "__main__":
    train_reinforce(run_name="reinforce_default")
    train_ppo(run_name="ppo_default")
    train_a2c(run_name="a2c_default")