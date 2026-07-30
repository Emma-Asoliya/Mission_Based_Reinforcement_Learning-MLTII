# Mission-Based Reinforcement Learning: Mental Health Support Triage

Reinforcement learning is applied to a simulated mental-health support triage system, built around connecting African youth seeking mental health help with the right support resources under real-world capacity limitations.

## Overview

A custom Gymnasium environment simulates an agent that routes incoming help-seekers, based on urgency level, category, and wait time, to one of four support tiers (self-help content, peer counseling, professional counseling, or crisis hotline escalation), or defers them to a queue. Four reinforcement learning algorithms (DQN, REINFORCE, PPO, A2C) were implemented using Stable-Baselines3 and trained on this environment, then compared across a 40-run hyperparameter sweep.

## Project Structure

```
project_root/
├── pyproject.toml          # Dependency configuration (uv)
├── uv.lock                 # Locked dependency versions
├── main.py                 # Entry point runs a trained agent live with rendering
├── environment/
│   ├── custom_env.py       # The MentalHealthTriageEnv Gymnasium environment
│   └── rendering.py        # Pygame dashboard for visualizing the environment
├── training/
│   ├── dqn_training.py     # DQN training
│   ├── pg_training.py      # REINFORCE-style PPO, PPO, and A2C training
│   ├── hyperparameter_sweep.py  # Runs the full 40-run hyperparameter sweep
│   └── analysis.py         # Generates report plots (rewards, stability, convergence, generalization)
├── models/                 # Saved trained model files (.zip)
├── logs/                   # Training logs and generated plots
└── tests/                  # Basic environment sanity tests
```

## Setup

This project uses [uv](https://docs.astral.sh/uv/) for dependency management. After cloning:

```bash
uv sync
```

This installs all dependencies (gymnasium, stable-baselines3, pygame, torch, etc.) into a local virtual environment, no manual pip installs or environment setup required.

**Note:** requires Python 3.10–3.12. If your system defaults to a newer Python version, `uv sync` will automatically download and use a compatible interpreter.

## Running the Project

**Watch a trained agent run live** (opens a Pygame dashboard):

```bash
uv run python main.py --algo a2c --model models/pg/a2c_run3
```

Other options:
```bash
uv run python main.py --algo dqn                 # run the default DQN model
uv run python main.py --algo a2c --episodes 3     # run multiple episodes back to back
```

**Train a single model from scratch:**

```bash
uv run python training/dqn_training.py
uv run python training/pg_training.py
```

**Run the full 40-run hyperparameter sweep** (DQN ×10, REINFORCE ×10, PPO ×10, A2C ×10):

```bash
uv run python training/hyperparameter_sweep.py
```

This takes roughly 60–100 minutes and writes results to `logs/hyperparameter_results.csv`.

**Generate report plots** (cumulative rewards, DQN loss / PG entropy, convergence, generalization):

```bash
uv run python training/analysis.py
```

Plots are saved to `logs/plots/`.

## Results Summary

Across the hyperparameter sweep, A2C achieved the best overall performance (mean reward 492.37, fastest convergence at 103 episodes), closely followed by PPO (450.53). DQN was the most robust across a wide hyperparameter range, while REINFORCE, implemented as an unstabilized policy-gradient configuration, showed the instability expected without variance-reduction techniques. Full analysis is available in the accompanying report.

## Report & Video

- Report: https://docs.google.com/document/d/1XlQPwTmj-z7zMsQTbIScrzQYec6B8wuvT6d6UXKVcS8/edit?usp=sharing
- Demo Video: https://drive.google.com/file/d/1ayNeuT8dZNNcVZDUD3peADLr6CMaDnFM/view?usp=sharing