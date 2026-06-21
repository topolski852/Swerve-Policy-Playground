# ──────────────────────────────────────────────────────────────────────────────
# path_randomizer/swerve_env.py
# Gymnasium environment for goal-conditioned swerve navigation.
#
# Each episode: random start position + N random waypoints.
# Observation encodes relative vectors to current and next waypoint so the
# policy generalises to any set of points — not locked to a specific field.
#
# Action space  : Box(3,) — [vx, vy, omega] normalized to [-1, 1]
#                 omega zeroed (translation-only phase)
# Observation   : 6-element vector — see OBS_LABELS
# Reward        : progress toward waypoints + arrival bonuses
# ──────────────────────────────────────────────────────────────────────────────

import math
import numpy as np
import gymnasium as gym
from gymnasium import spaces

from lib.kinematics import SwerveState
from lib.field_constants import (
    MAX_SPEED_MPS, FIELD_LENGTH, FIELD_WIDTH,
    ROBOT_BUMPER_HALF, IMPASSABLE_RECTS,
    WAYPOINT_PASS_RADIUS,
)
from path_randomizer.constants import (
    N_WAYPOINTS_MIN, N_WAYPOINTS_MAX, MAX_EPISODE_STEPS,
    RW_PROGRESS, RW_WAYPOINT_BONUS, RW_GOAL_BONUS,
    RW_TIME_PENALTY, RW_COLLISION_PENALTY,
)

FIELD_DIAGONAL = math.sqrt(FIELD_LENGTH ** 2 + FIELD_WIDTH ** 2)

OBS_DIM    = 6
OBS_LABELS = ["vx_n", "vy_n", "dx0_n", "dy0_n", "dx1_n", "dy1_n"]


class WaypointTracker:
    """Minimal tracker: advance current_idx when robot is within pass radius."""

    def __init__(self):
        self._waypoints  = []
        self.current_idx = 0   # named current_idx for renderer compatibility

    def reset(self, waypoints):
        self._waypoints  = list(waypoints)
        self.current_idx = 0

    @property
    def done(self):
        return self.current_idx >= len(self._waypoints)

    @property
    def current(self):
        i = min(self.current_idx, len(self._waypoints) - 1)
        return self._waypoints[i]

    @property
    def next_wp(self):
        i = min(self.current_idx + 1, len(self._waypoints) - 1)
        return self._waypoints[i]

    def update(self, robot_x, robot_y):
        advanced = 0
        while not self.done:
            wx, wy = self.current
            if math.hypot(robot_x - wx, robot_y - wy) < WAYPOINT_PASS_RADIUS:
                self.current_idx += 1
                advanced += 1
            else:
                break
        return advanced


class SwerveEnv(gym.Env):

    metadata = {"render_modes": ["human"], "render_fps": 60}

    def __init__(self, render_mode=None):
        super().__init__()
        self.render_mode = render_mode

        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(3,), dtype=np.float32
        )

        obs_low  = np.full(OBS_DIM, -1.0, dtype=np.float32)
        obs_high = np.full(OBS_DIM,  1.0, dtype=np.float32)
        self.observation_space = spaces.Box(obs_low, obs_high, dtype=np.float32)

        self._robot      = SwerveState()
        self._tracker    = WaypointTracker()
        self._waypoints  = []
        self._step_count = 0
        self._prev_dist  = 0.0
        self._renderer   = None

    # ──────────────────────────────────────────────────────────────────────────
    # Gymnasium API
    # ──────────────────────────────────────────────────────────────────────────

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)

        sx, sy = self._random_valid_pos()
        self._robot.reset(x=sx, y=sy, heading=0.0)

        n = int(self.np_random.integers(N_WAYPOINTS_MIN, N_WAYPOINTS_MAX + 1))
        self._waypoints = [self._random_valid_pos() for _ in range(n)]
        self._tracker.reset(self._waypoints)

        self._step_count = 0
        self._prev_dist  = math.hypot(
            sx - self._tracker.current[0],
            sy - self._tracker.current[1],
        )
        return self._get_obs(), {}

    def step(self, action: np.ndarray):
        action = np.clip(action, -1.0, 1.0).astype(np.float32)
        self._robot.step(
            float(action[0]) * MAX_SPEED_MPS,
            float(action[1]) * MAX_SPEED_MPS,
            0.0,
        )
        self._step_count += 1

        rx, ry = self._robot.x, self._robot.y

        # ── Waypoint progress ─────────────────────────────────────────────────
        progress      = 0.0
        waypoint_bonus = 0.0
        if not self._tracker.done:
            wx, wy   = self._tracker.current
            dist     = math.hypot(rx - wx, ry - wy)
            progress = self._prev_dist - dist          # positive = getting closer
            advanced = self._tracker.update(rx, ry)
            waypoint_bonus = RW_WAYPOINT_BONUS * advanced

            if not self._tracker.done:
                nwx, nwy     = self._tracker.current
                self._prev_dist = math.hypot(rx - nwx, ry - nwy)
            else:
                self._prev_dist = 0.0

        goal_done = self._tracker.done

        # ── Reward ────────────────────────────────────────────────────────────
        reward = (
            RW_PROGRESS * progress
            + waypoint_bonus
            + (RW_GOAL_BONUS if goal_done else 0.0)
            + RW_TIME_PENALTY
        )

        # ── Termination ───────────────────────────────────────────────────────
        collision  = self._check_collision()
        terminated = goal_done
        truncated  = (self._step_count >= MAX_EPISODE_STEPS) or collision

        if collision:
            reward += RW_COLLISION_PENALTY

        obs  = self._get_obs()
        info = {
            "waypoint_idx": self._tracker.current_idx,
            "n_waypoints":  len(self._waypoints),
            "progress":     progress,
        }

        if self.render_mode == "human":
            self.render()

        return obs, reward, terminated, truncated, info

    def render(self):
        if self._renderer is None:
            from lib.renderer import Renderer
            self._renderer = Renderer(waypoints=self._waypoints)
        self._renderer.set_waypoints(self._waypoints)
        self._renderer.draw(self._robot, self._tracker, self._get_module_states())

    def close(self):
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None

    # ──────────────────────────────────────────────────────────────────────────
    # Helpers
    # ──────────────────────────────────────────────────────────────────────────

    def _get_obs(self):
        vx_n = float(np.clip(self._robot.vx / MAX_SPEED_MPS, -1.0, 1.0))
        vy_n = float(np.clip(self._robot.vy / MAX_SPEED_MPS, -1.0, 1.0))
        rx, ry = self._robot.x, self._robot.y

        if self._tracker.done:
            dx0_n = dy0_n = dx1_n = dy1_n = 0.0
        else:
            wx0, wy0 = self._tracker.current
            dx0_n = float(np.clip((wx0 - rx) / FIELD_DIAGONAL, -1.0, 1.0))
            dy0_n = float(np.clip((wy0 - ry) / FIELD_DIAGONAL, -1.0, 1.0))
            wx1, wy1 = self._tracker.next_wp
            dx1_n = float(np.clip((wx1 - rx) / FIELD_DIAGONAL, -1.0, 1.0))
            dy1_n = float(np.clip((wy1 - ry) / FIELD_DIAGONAL, -1.0, 1.0))

        return np.array([vx_n, vy_n, dx0_n, dy0_n, dx1_n, dy1_n], dtype=np.float32)

    def _random_valid_pos(self):
        r = ROBOT_BUMPER_HALF
        pad = r + 0.1
        for _ in range(200):
            x = float(self.np_random.uniform(pad, FIELD_LENGTH - pad))
            y = float(self.np_random.uniform(pad, FIELD_WIDTH  - pad))
            ok = True
            for ox1, oy1, ox2, oy2 in IMPASSABLE_RECTS:
                if x > ox1 - r and x < ox2 + r and y > oy1 - r and y < oy2 + r:
                    ok = False
                    break
            if ok:
                return x, y
        return FIELD_LENGTH / 2, FIELD_WIDTH / 2  # fallback: midfield

    def _check_collision(self):
        rx, ry = self._robot.x, self._robot.y
        r = ROBOT_BUMPER_HALF
        if rx - r < 0 or rx + r > FIELD_LENGTH:
            return True
        if ry - r < 0 or ry + r > FIELD_WIDTH:
            return True
        for ox1, oy1, ox2, oy2 in IMPASSABLE_RECTS:
            if rx > ox1 - r and rx < ox2 + r and ry > oy1 - r and ry < oy2 + r:
                return True
        return False

    def _get_module_states(self):
        return self._robot.module_states
