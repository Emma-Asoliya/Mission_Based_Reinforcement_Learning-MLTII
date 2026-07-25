"""
main.py
Entry point for the project. Loads a trained agent and runs it in the
Pygame dashboard so you can watch its decision-making live.

Usage:
    uv run main.py
    uv run main.py --algo ppo
    uv run main.py --model models/ppo/ppo_default
"""

import argparse
from stable_baselines3 import DQN, PPO, A2C

from environment.custom_env import MentalHealthTriageEnv

ALGO_CLASSES = {
    "dqn": DQN,
    "ppo": PPO,
    "a2c": A2C,
}

DEFAULT_MODEL_PATHS = {
    "dqn": "models/dqn/dqn_default",
    "ppo": "models/pg/ppo_default",
    "reinforce": "models/pg/reinforce_default",
    "a2c": "models/pg/a2c_default",
}

def parse_args():
    parser = argparse.ArgumentParser(description="Watch a trained triage agent in action.")
    parser.add_argument("--algo", type=str, default="dqn", choices=ALGO_CLASSES.keys(),
                        help="Which algorithm's default model to load and run")
    parser.add_argument("--model", type=str, default=None,
                        help="Optional explicit path to a saved model (overrides --algo default).")
    parser.add_argument("--episodes", type=int, default=1,
                        help="How many episodes to run.")

    return parser.parse_args()

def run_agent(algo: str, model_path: str, episodes: int):
    """Load a trained model and run it in the environment with live rendering."""

    algo_class = ALGO_CLASSES[algo]
    model = algo_class.load(model_path)
    print(f"Loaded {algo.upper()} model from {model_path}")

    env = MentalHealthTriageEnv(max_requests=50, max_steps=200, render_mode="human")

    for ep in range(episodes):
        obs, info = env.reset(seed=ep)
        terminated = truncated = False
        total_reward = 0.0

        while not (terminated or truncated):
            action, _ = model.predict(obs, deterministic=True)
            action = int(action)
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            env.render()

        print(f"Episode {ep + 1} finsihed | "
              f"resolved={env.requests_resolved}/{env.max_requests} | "
              f"total_reward={total_reward:.2f}")

        env.close()

def main():
    args = parse_args()
    model_path = args.model if args.model else DEFAULT_MODEL_PATHS[args.algo]
    run_agent(algo=args.algo, model_path=model_path, episodes=args.episodes)

if __name__ == "__main__":
    main()