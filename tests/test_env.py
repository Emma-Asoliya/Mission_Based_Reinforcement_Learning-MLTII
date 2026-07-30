"""
Basic sanity tests for MentalHealthTriageEnv.

Run with:
    uv run python -m pytest tests/test_env.py -v

or, without pytest installed, just:
    uv run python tests/test_env.py
"""

import numpy as np
from environment.custom_env import MentalHealthTriageEnv, IDEAL_ACTION_FOR_URGENCY, MAX_CAPACITY


def test_reset_returns_valid_observation():
    env = MentalHealthTriageEnv(max_requests=20, max_steps=100)
    obs, info = env.reset(seed=42)
    assert env.observation_space.contains(obs), "Initial observation out of bounds"
    assert obs.shape == (15,), f"Expected 15-dim observation, got {obs.shape}"


def test_step_returns_valid_observation_and_types():
    env = MentalHealthTriageEnv(max_requests=20, max_steps=100)
    obs, info = env.reset(seed=42)
    obs, reward, terminated, truncated, info = env.step(0)

    assert env.observation_space.contains(obs), "Post-step observation out of bounds"
    assert isinstance(reward, (int, float)), "Reward should be numeric"
    assert isinstance(terminated, bool), "terminated should be a bool"
    assert isinstance(truncated, bool), "truncated should be a bool"


def test_episode_terminates():
    """An episode should end once all requests are resolved or max_steps is hit."""
    env = MentalHealthTriageEnv(max_requests=20, max_steps=200)
    obs, info = env.reset(seed=42)

    rng = np.random.default_rng(0)
    terminated = truncated = False
    steps = 0
    while not (terminated or truncated) and steps < 300:
        action = rng.integers(0, 5)
        obs, reward, terminated, truncated, info = env.step(action)
        steps += 1

    assert terminated or truncated, "Episode never ended within a reasonable number of steps"
    assert steps < 300, "Episode ran suspiciously long — possible infinite loop"


def test_capacity_never_exceeded():
    """No resource should ever hold more occupants than its MAX_CAPACITY."""
    env = MentalHealthTriageEnv(max_requests=30, max_steps=200)
    obs, info = env.reset(seed=1)

    rng = np.random.default_rng(1)
    terminated = truncated = False
    while not (terminated or truncated):
        action = rng.integers(0, 5)
        obs, reward, terminated, truncated, info = env.step(action)
        for r in range(4):
            assert len(env.occupied[r]) <= MAX_CAPACITY[r], f"Resource {r} exceeded capacity"


def test_smart_policy_outperforms_random_policy():
    """A policy that always routes to the 'ideal' resource should beat random actions."""

    def run_random_policy(seed):
        env = MentalHealthTriageEnv(max_requests=20, max_steps=200)
        obs, info = env.reset(seed=seed)
        rng = np.random.default_rng(seed)
        total_reward = 0.0
        terminated = truncated = False
        while not (terminated or truncated):
            action = rng.integers(0, 5)
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
        return total_reward

    def run_smart_policy(seed):
        env = MentalHealthTriageEnv(max_requests=20, max_steps=200)
        obs, info = env.reset(seed=seed)
        total_reward = 0.0
        terminated = truncated = False
        while not (terminated or truncated):
            req = env.current_request
            if req is None:
                action = 4
            else:
                ideal = IDEAL_ACTION_FOR_URGENCY[req["urgency"]]
                action = ideal if len(env.occupied[ideal]) < MAX_CAPACITY[ideal] else 4
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
        return total_reward

    random_reward = run_random_policy(seed=42)
    smart_reward = run_smart_policy(seed=42)

    assert smart_reward > random_reward, (
        f"Smart policy ({smart_reward:.2f}) should outperform random policy ({random_reward:.2f})"
    )


if __name__ == "__main__":
    # Allow running without pytest installed
    tests = [
        test_reset_returns_valid_observation,
        test_step_returns_valid_observation_and_types,
        test_episode_terminates,
        test_capacity_never_exceeded,
        test_smart_policy_outperforms_random_policy,
    ]
    for test in tests:
        test()
        print(f"PASSED: {test.__name__}")
    print("\nAll tests passed.")