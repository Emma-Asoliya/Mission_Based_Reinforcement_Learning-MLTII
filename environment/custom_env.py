"""
MentalHealthTriageEnv
======================
A custom Gymnasium environment simulating a mental-health support triage and
matching system for youth in Africa. An agent receives
a stream of help-seekers, each with an urgency level and a category, and must
route each one to an appropriate support resource under realistic capacity
constraints mirroring the real-world problem of connecting people who need
help with the people/services who can provide it.
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces


# ---- The four support resources our agent can route someone to ----
SELF_HELP, PEER, PROFESSIONAL, CRISIS, QUEUE_ACTION = 0, 1, 2, 3, 4
RESOURCE_NAMES = ["Self-Help", "Peer Counselor", "Professional", "Crisis Hotline"]

# ---- The categories a help-seeker's request might fall into ----
CATEGORY_NAMES = ["Anxiety", "Depression", "Crisis", "General"]
N_CATEGORIES = len(CATEGORY_NAMES)

# ---- How many people each resource can support AT ONCE ----
MAX_CAPACITY = np.array([20, 3, 2, 1], dtype=np.float32)

# ---- How many time-steps a resource stays occupied once assigned ----
SERVICE_DURATION = {
    SELF_HELP: 1,
    PEER: 3,
    PROFESSIONAL: 5,
    CRISIS: 2,
}

# ---- The "correct" resource tier for each urgency level ----
IDEAL_ACTION_FOR_URGENCY = {
    0: SELF_HELP,
    1: PEER,
    2: PROFESSIONAL,
    3: CRISIS,
}

# ---- Normalization horizons ----
MAX_WAIT = 15
MAX_QUEUE_LEN = 10
CRISIS_WAIT_LIMIT = 6   # steps a crisis case may wait before it's a system failure

OBS_SIZE = 15


class MentalHealthTriageEnv(gym.Env):
    """
    A custom Gymnasium environment simulating mental-health support triage:
    an agent receives help-seekers one at a time and must route each to an
    appropriate resource under realistic capacity constraints.
    """

    metadata = {"render_modes": ["human"], "render_fps": 4}

    def __init__(self, max_requests: int = 50, max_steps: int = 200, render_mode=None):
        super().__init__()

        self.max_requests = max_requests
        self.max_steps = max_steps
        self.render_mode = render_mode

        self.action_space = spaces.Discrete(5)
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(OBS_SIZE,), dtype=np.float32
        )

        self._renderer = None

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.rng = np.random.default_rng(seed)

        self.time_step = 0
        self.requests_generated = 0
        self.requests_resolved = 0
        self.occupied = {r: [] for r in range(4)}
        self.pending = []
        self.episode_reward = 0.0

        self._spawn_arrivals(initial=True)
        self.current_request = self._pop_priority_request()

        obs = self._get_obs()
        info = {"requests_resolved": 0}
        return obs, info

    def step(self, action):
        assert self.action_space.contains(action)

        self._advance_service_slots()

        terminated = False
        truncated = False
        req = self.current_request

        if req is None:
            self._spawn_arrivals()
            self.current_request = self._pop_priority_request()
            self.time_step += 1
            obs = self._get_obs()
            if self.time_step >= self.max_steps:
                truncated = True
            return obs, 0.0, terminated, truncated, {}

        reward = self._process_action(req, action)

        # crisis starvation check -- a safety failsafe
        for r in self.pending:
            if r["urgency"] == 3 and r["wait"] > CRISIS_WAIT_LIMIT:
                reward -= 25.0
                terminated = True

        self._spawn_arrivals()
        self.current_request = self._pop_priority_request()
        self.time_step += 1
        self.episode_reward += reward

        if self.requests_resolved >= self.max_requests:
            terminated = True
        if self.time_step >= self.max_steps:
            truncated = True

        obs = self._get_obs()
        info = {
            "requests_resolved": self.requests_resolved,
            "pending": len(self.pending),
        }
        return obs, reward, terminated, truncated, info

    def _spawn_arrivals(self, initial=False):
        """Add new help-seekers to the pending queue."""
        if self.requests_generated >= self.max_requests:
            return

        n_new = 3 if initial else self.rng.poisson(0.6)

        for _ in range(n_new):
            if self.requests_generated >= self.max_requests:
                break

            urgency = self.rng.choice([0, 1, 2, 3], p=[0.4, 0.3, 0.2, 0.1])
            category = self.rng.integers(0, N_CATEGORIES)

            if urgency == 3:
                category = 2  # "Crisis" category

            self.pending.append({
                "urgency": int(urgency),
                "category": int(category),
                "wait": 0,
            })
            self.requests_generated += 1

    def _pop_priority_request(self):
        """Pick the next request the agent must decide on right now."""
        if not self.pending:
            return None

        self.pending.sort(key=lambda r: (-r["urgency"], -r["wait"]))
        return self.pending.pop(0)

    def _advance_service_slots(self):
        """Tick down every occupied slot by one step; remove slots that finish."""
        for r in range(4):
            self.occupied[r] = [d - 1 for d in self.occupied[r] if d - 1 > 0]


    def _match_reward(self, req, resource):
        """Score how appropriate it was to route this request to this resource."""
        ideal = IDEAL_ACTION_FOR_URGENCY[req["urgency"]]
        distance = abs(ideal - resource)

        if distance == 0:
            base = 10.0
        elif distance == 1:
            base = -3.0
        elif distance == 2:
            base = -8.0
        else:
            base = -20.0

        if req["urgency"] == 3 and resource in (SELF_HELP, PEER):
            base -= 10.0
        if req["urgency"] == 0 and resource in (PROFESSIONAL, CRISIS):
            base -= 2.0

        return base

    def _process_action(self, req, action):
        """Handle the agent's chosen action for the current request. Returns reward."""
        reward = 0.0

        if action == QUEUE_ACTION:
            req["wait"] += 1
            reward -= 0.5 * (1 + req["urgency"])
            self.pending.append(req)

        else:
            resource = action
            if len(self.occupied[resource]) < MAX_CAPACITY[resource]:
                self.occupied[resource].append(SERVICE_DURATION[resource])
                reward += self._match_reward(req, resource)
                reward -= 0.15 * req["wait"] * (1 + req["urgency"])
                self.requests_resolved += 1
            else:
                reward -= 4.0
                req["wait"] += 1
                self.pending.append(req)

        return reward

    def _get_obs(self):
        req = self.current_request

        if req is None:
            urgency_n = 0.0
            cat_onehot = np.zeros(N_CATEGORIES, dtype=np.float32)
            wait_n = 0.0
        else:
            urgency_n = req["urgency"] / 3.0
            cat_onehot = np.eye(N_CATEGORIES, dtype=np.float32)[req["category"]]
            wait_n = min(req["wait"] / MAX_WAIT, 1.0)

        remaining_capacity = np.array([
            (MAX_CAPACITY[r] - len(self.occupied[r])) / MAX_CAPACITY[r]
            for r in range(4)
        ], dtype=np.float32)

        bucket_counts = np.zeros(4, dtype=np.float32)
        for r in self.pending:
            bucket_counts[IDEAL_ACTION_FOR_URGENCY[r["urgency"]]] += 1
        bucket_counts = np.minimum(bucket_counts / MAX_QUEUE_LEN, 1.0)

        time_progress = min(self.time_step / self.max_steps, 1.0)

        obs = np.concatenate([
            [urgency_n],
            cat_onehot,
            [wait_n],
            remaining_capacity,
            bucket_counts,
            [time_progress],
        ]).astype(np.float32)

        return obs

    def render(self):
        if self.render_mode is None:
            return

        if self._renderer is None:
            from environment.rendering import TriageRenderer
            self._renderer = TriageRenderer()

        return self._renderer.render(self)

    def close(self):
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None