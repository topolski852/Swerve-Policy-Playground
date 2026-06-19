# ──────────────────────────────────────────────────────────────────────────────
# fuel_scoring/swerve_env.py
# Gymnasium environment for the fuel scoring experiment.
#
# No path or waypoints — the robot is rewarded purely for the fuel loop:
#   Alliance Zone → score fuel from hopper (any speed)
#   Neutral Zone  → collect fuel while moving
#   Repeat
#
# Action space  : Box(3,) — [vx, vy, omega] normalized to [-1, 1]
#                 omega zeroed (translation-only phase)
# Observation   : 5-element vector — see OBS_LABELS
# Reward        : fuel scoring dominant; see _compute_reward()
# ──────────────────────────────────────────────────────────────────────────────

import math
import numpy as np
import gymnasium as gym
from gymnasium import spaces

from lib.kinematics import SwerveState
from lib.field_constants import (
    MAX_SPEED_MPS,
    FIELD_LENGTH, FIELD_WIDTH,
    ROBOT_BUMPER_HALF, IMPASSABLE_RECTS,
    BLUE_ALLIANCE_MAX_X, NEUTRAL_MIN_X, NEUTRAL_MAX_X,
)
from fuel_scoring.constants import (
    MAX_EPISODE_STEPS, SCORING_START_HOPPER,
    HOPPER_CAPACITY, FUEL_FILL_RATE_MAX, FUEL_FILL_MIN_SPEED, FUEL_SHOOT_RATE,
    RW_FUEL_SCORED, RW_FUEL_COLLECTED,
    RW_FULL_HOPPER_IN_NEUTRAL, RW_EMPTY_HOPPER_IN_ALLIANCE,
    RW_COLLISION_PENALTY,
)

# Starting position: Blue Alliance Zone, BumpRight corridor center
START_X = 3.50
START_Y = 2.55

OBS_DIM    = 5
OBS_LABELS = ["vx_n", "vy_n", "rx_n", "ry_n", "hopper_norm"]


class SwerveEnv(gym.Env):

    metadata = {"render_modes": ["human"], "render_fps": 60}

    def __init__(self, render_mode=None):
        super().__init__()
        self.render_mode = render_mode

        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(3,), dtype=np.float32
        )

        obs_low  = np.array([-1.0, -1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        obs_high = np.array([ 1.0,  1.0, 1.0, 1.0, 1.0], dtype=np.float32)
        self.observation_space = spaces.Box(obs_low, obs_high, dtype=np.float32)

        self._robot  = SwerveState()
        self._hopper = SCORING_START_HOPPER

        self._step_count = 0
        self._renderer   = None

    # ──────────────────────────────────────────────────────────────────────────
    # Gymnasium API
    # ──────────────────────────────────────────────────────────────────────────

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self._robot.reset(x=START_X, y=START_Y, heading=0.0)
        self._hopper     = SCORING_START_HOPPER
        self._step_count = 0
        return self._get_obs(), {}

    def step(self, action: np.ndarray):
        action = np.clip(action, -1.0, 1.0).astype(np.float32)

        cmd_vx    = float(action[0]) * MAX_SPEED_MPS
        cmd_vy    = float(action[1]) * MAX_SPEED_MPS
        cmd_omega = 0.0

        self._robot.step(cmd_vx, cmd_vy, cmd_omega)
        self._step_count += 1

        rx, ry = self._robot.x, self._robot.y
        speed  = math.hypot(self._robot.vx, self._robot.vy)

        # ── Fuel mechanics ────────────────────────────────────────────────────
        # Collect: log-scaled by speed — faster movement = more fuel per step.
        # rate = FUEL_FILL_RATE_MAX × log(speed/MIN) / log(MAX/MIN)
        # At MIN_SPEED: 0 fuel/step. At MAX_SPEED: FUEL_FILL_RATE_MAX fuel/step.
        fuel_collected = 0.0
        if NEUTRAL_MIN_X < rx < NEUTRAL_MAX_X and speed >= FUEL_FILL_MIN_SPEED:
            log_scale      = math.log(speed / FUEL_FILL_MIN_SPEED) / math.log(MAX_SPEED_MPS / FUEL_FILL_MIN_SPEED)
            fill_rate      = FUEL_FILL_RATE_MAX * log_scale
            collected      = min(HOPPER_CAPACITY - self._hopper, fill_rate)
            self._hopper  += collected
            fuel_collected = collected

        # Score: any speed, just be in the alliance zone
        fuel_scored = 0.0
        if rx < BLUE_ALLIANCE_MAX_X and self._hopper > 0:
            shot = min(self._hopper, FUEL_SHOOT_RATE)
            self._hopper -= shot
            fuel_scored   = shot

        # ── Reward ────────────────────────────────────────────────────────────
        reward, info = self._compute_reward(fuel_scored, fuel_collected)

        # ── Termination ───────────────────────────────────────────────────────
        collision = self._check_collision()
        terminated = False
        truncated  = self._step_count >= MAX_EPISODE_STEPS or collision

        if collision:
            reward += RW_COLLISION_PENALTY
            info["collision"] = True

        obs = self._get_obs()

        if self.render_mode == "human":
            self.render()

        return obs, reward, terminated, truncated, info

    def render(self):
        if self._renderer is None:
            from lib.renderer import Renderer
            self._renderer = Renderer(waypoints=None)
        self._renderer.draw(self._robot, None, self._get_module_states())

    def close(self):
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None

    # ──────────────────────────────────────────────────────────────────────────
    # Collision detection
    # ──────────────────────────────────────────────────────────────────────────

    def _check_collision(self) -> bool:
        rx, ry = self._robot.x, self._robot.y
        r = ROBOT_BUMPER_HALF

        if rx - r < 0 or rx + r > FIELD_LENGTH:
            return True
        if ry - r < 0 or ry + r > FIELD_WIDTH:
            return True

        for (ox1, oy1, ox2, oy2) in IMPASSABLE_RECTS:
            if rx > ox1 - r and rx < ox2 + r and ry > oy1 - r and ry < oy2 + r:
                return True

        return False

    # ──────────────────────────────────────────────────────────────────────────
    # Reward function
    # ──────────────────────────────────────────────────────────────────────────

    def _compute_reward(self, fuel_scored, fuel_collected):
        r_fuel_scored    = RW_FUEL_SCORED   * fuel_scored
        r_fuel_collected = RW_FUEL_COLLECTED * fuel_collected

        # Penalty: full hopper sitting in neutral — go score it
        r_full_hopper = (
            RW_FULL_HOPPER_IN_NEUTRAL
            if NEUTRAL_MIN_X < self._robot.x < NEUTRAL_MAX_X and self._hopper >= HOPPER_CAPACITY
            else 0.0
        )

        # Penalty: empty hopper sitting in alliance — go collect more
        r_empty_alliance = (
            RW_EMPTY_HOPPER_IN_ALLIANCE
            if self._robot.x < BLUE_ALLIANCE_MAX_X and self._hopper <= 0
            else 0.0
        )

        reward = r_fuel_scored + r_fuel_collected + r_full_hopper + r_empty_alliance

        info = {
            "fuel_scored":    fuel_scored,
            "fuel_collected": fuel_collected,
            "hopper_level":   self._hopper / HOPPER_CAPACITY,
            "r_fuel_scored":  r_fuel_scored,
        }
        return reward, info

    # ──────────────────────────────────────────────────────────────────────────
    # Observation builder
    # ──────────────────────────────────────────────────────────────────────────

    def _get_obs(self):
        vx_n = float(np.clip(self._robot.vx / MAX_SPEED_MPS, -1.0, 1.0))
        vy_n = float(np.clip(self._robot.vy / MAX_SPEED_MPS, -1.0, 1.0))
        rx_n = float(np.clip(self._robot.x  / FIELD_LENGTH,   0.0, 1.0))
        ry_n = float(np.clip(self._robot.y  / FIELD_WIDTH,    0.0, 1.0))
        hopper_norm = float(np.clip(self._hopper / HOPPER_CAPACITY, 0.0, 1.0))

        return np.array([vx_n, vy_n, rx_n, ry_n, hopper_norm], dtype=np.float32)

    def _get_module_states(self):
        return self._robot.module_states
