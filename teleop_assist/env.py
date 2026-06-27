# ──────────────────────────────────────────────────────────────────────────────
# teleop_assist/env.py
# Gymnasium environment for the Teleop Assist policy.
#
# The policy sits between the joystick and swerve.drive() on the real robot.
# It sees raw (drifty) joystick input and chassis state, then outputs
# ChassisSpeeds that match driver intent — smoothed and drift-corrected.
#
# Observation (OBS_DIM = 10):
#   [joy_x, joy_y, joy_rot,          raw joystick with per-episode drift
#    vx_n, vy_n, omega_n,            current chassis velocity, normalized
#    sin_h, cos_h,                   heading (avoids wrap discontinuity)
#    rx_n, ry_n]                     field position, normalized
#
# Action (3,):
#   [vx_n, vy_n, omega_n] normalized to [-1, 1], robot frame
#   → swerve.drive(ChassisSpeeds(vx_n*MAX_SPEED, vy_n*MAX_SPEED, omega_n*MAX_ANGULAR))
#
# Key training mechanics:
#   DRIFT INJECTION  — per-episode random stick offset; reward computed against
#                      true intent so policy learns to ignore noise-floor inputs.
#   SQUARE ROBOT     — AABB collision uses ROBOT_BUMPER_HALF on all sides,
#                      matching the real robot's bumper geometry.
#
# Phase 2 (obstacle avoidance) will add proximity rays back to the obs once
# joystick following is verified working on hardware.
# ──────────────────────────────────────────────────────────────────────────────

import math
import numpy as np
import gymnasium as gym
from gymnasium import spaces

from lib.kinematics import SwerveState
from lib.field_constants import (
    MAX_SPEED_MPS, MAX_ANGULAR_RPS,
    FIELD_LENGTH, FIELD_WIDTH,
    ROBOT_BUMPER_HALF, IMPASSABLE_RECTS,
)
from teleop_assist.constants import (
    MAX_EPISODE_STEPS,
    DRIFT_MAX, DRIFT_FLOOR,
    JOY_TARGET_REROLL_MIN, JOY_TARGET_REROLL_MAX, JOY_TARGET_ARRIVE_R,
    JOY_SPEED_MIN, JOY_SPEED_MAX, STOP_INTENT_PROB,
    RW_MATCH, RW_MATCH_K, RW_SMOOTH, RW_COLLISION,
)

OBS_DIM = 10
OBS_LABELS = [
    "joy_x", "joy_y", "joy_rot",
    "vx_n", "vy_n", "omega_n",
    "sin_h", "cos_h",
    "rx_n", "ry_n",
]


class TeleopAssistEnv(gym.Env):

    metadata = {"render_modes": ["human"], "render_fps": 60}

    def __init__(self, render_mode=None):
        super().__init__()
        self.render_mode = render_mode

        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(3,), dtype=np.float32
        )

        obs_low  = np.array(
            [-1, -1, -1,   # joystick
             -1, -1, -1,   # velocity
             -1, -1,       # sin/cos heading
              0,  0],      # position
            dtype=np.float32,
        )
        obs_high = np.array(
            [ 1,  1,  1,
              1,  1,  1,
              1,  1,
              1,  1],
            dtype=np.float32,
        )
        self.observation_space = spaces.Box(obs_low, obs_high, dtype=np.float32)

        self._robot = SwerveState()

        # Per-episode state
        self._drift        = np.zeros(3, dtype=np.float32)  # [dx, dy, drot]
        self._joy_target_x = 0.0
        self._joy_target_y = 0.0
        self._joy_speed    = 1.0
        self._joy_reroll_countdown = 0
        self._joy_is_stop  = False   # True → true intent is "hold still" this reroll period

        self._prev_action  = np.zeros(3, dtype=np.float32)
        self._step_count   = 0

        self._renderer = None

    # ──────────────────────────────────────────────────────────────────────────
    # Gymnasium API
    # ──────────────────────────────────────────────────────────────────────────

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)

        heading = float(self.np_random.uniform(-math.pi, math.pi))
        self._robot.reset(x=FIELD_LENGTH / 2.0, y=FIELD_WIDTH / 2.0, heading=heading)

        # Per-episode drift — fixed for the whole episode so the policy can't
        # just subtract the initial reading; it must learn to threshold.
        self._drift = self.np_random.uniform(
            -DRIFT_MAX, DRIFT_MAX, size=3
        ).astype(np.float32)

        self._joy_is_stop  = False
        self._sample_new_joy_target()

        self._prev_action  = np.zeros(3, dtype=np.float32)
        self._step_count   = 0

        return self._get_obs(), {}

    def step(self, action: np.ndarray):
        action = np.clip(action, -1.0, 1.0).astype(np.float32)

        self._update_joy_target()

        self._robot.step(
            float(action[0]) * MAX_SPEED_MPS,
            float(action[1]) * MAX_SPEED_MPS,
            float(action[2]) * MAX_ANGULAR_RPS,
        )
        self._step_count += 1

        collision  = self._check_collision()
        terminated = collision
        truncated  = (self._step_count >= MAX_EPISODE_STEPS)

        reward, info = self._compute_reward(action)

        if collision:
            reward += RW_COLLISION
            info["collision"] = True

        self._prev_action = action.copy()

        obs = self._get_obs()
        if self.render_mode == "human":
            self.render()

        return obs, reward, terminated, truncated, info

    def render(self):
        if self._renderer is None:
            from lib.renderer import Renderer
            self._renderer = Renderer()
        self._renderer.draw(self._robot, None, self._robot.module_states)

    def close(self):
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None

    # ──────────────────────────────────────────────────────────────────────────
    # Collision (square AABB)
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
    # Joystick intent simulation
    # ──────────────────────────────────────────────────────────────────────────

    def _update_joy_target(self):
        dx = self._joy_target_x - self._robot.x
        dy = self._joy_target_y - self._robot.y
        self._joy_reroll_countdown -= 1

        if math.hypot(dx, dy) < JOY_TARGET_ARRIVE_R or self._joy_reroll_countdown <= 0:
            self._sample_new_joy_target()

    def _sample_new_joy_target(self):
        self._joy_reroll_countdown = int(
            self.np_random.integers(JOY_TARGET_REROLL_MIN, JOY_TARGET_REROLL_MAX + 1)
        )

        # 20% of reroll periods are "hold still" — true intent is zero so the
        # policy learns to output zero velocity when the joystick is near neutral.
        if float(self.np_random.random()) < STOP_INTENT_PROB:
            self._joy_is_stop = True
            return

        self._joy_is_stop = False
        for _ in range(100):
            tx = float(self.np_random.uniform(1.0, FIELD_LENGTH - 1.0))
            ty = float(self.np_random.uniform(1.0, FIELD_WIDTH  - 1.0))
            if not self._pos_in_obstacle(tx, ty):
                self._joy_target_x = tx
                self._joy_target_y = ty
                break

        self._joy_speed = float(self.np_random.uniform(JOY_SPEED_MIN, JOY_SPEED_MAX))

    def _true_joy_robot(self) -> np.ndarray:
        """
        True joystick intent as a 2-vector in robot frame, magnitude = joy_speed.
        Returns zero during stop-intent periods (20% of reroll windows) so the
        policy is regularly trained on the "hold still" case.
        """
        if self._joy_is_stop:
            return np.zeros(2, dtype=np.float32)

        dx = self._joy_target_x - self._robot.x
        dy = self._joy_target_y - self._robot.y
        dist = math.hypot(dx, dy)

        if dist < 0.05:
            return np.zeros(2, dtype=np.float32)

        fx, fy = dx / dist, dy / dist

        cos_h = math.cos(self._robot.heading)
        sin_h = math.sin(self._robot.heading)
        rx =  fx * cos_h + fy * sin_h
        ry = -fx * sin_h + fy * cos_h

        return np.array([rx * self._joy_speed, ry * self._joy_speed], dtype=np.float32)

    # ──────────────────────────────────────────────────────────────────────────
    # Reward
    # ──────────────────────────────────────────────────────────────────────────

    def _compute_reward(self, action: np.ndarray):
        true_joy = self._true_joy_robot()

        # Target = what ChassisSpeeds.fromFieldRelativeSpeeds() would produce for
        # this joystick input, expressed in the same [-1,1] normalized action space.
        # true_joy is already in robot frame at the correct scale, so it IS the target.
        # Omega target is 0 — rotation is not trained in phase 1.
        target = np.array([true_joy[0], true_joy[1], 0.0], dtype=np.float32)

        mse     = float(np.mean((action - target) ** 2))
        r_match = RW_MATCH * math.exp(-RW_MATCH_K * mse)

        r_smooth = RW_SMOOTH * float(np.linalg.norm(action - self._prev_action))

        reward = r_match + r_smooth
        info = {
            "r_match":  r_match,
            "r_smooth": r_smooth,
            "mse":      mse,
        }
        return reward, info

    # ──────────────────────────────────────────────────────────────────────────
    # Observation
    # ──────────────────────────────────────────────────────────────────────────

    def _get_obs(self) -> np.ndarray:
        rx, ry, heading = self._robot.x, self._robot.y, self._robot.heading

        true_joy    = self._true_joy_robot()
        obs_joy_x   = float(np.clip(true_joy[0] + self._drift[0], -1.0, 1.0))
        obs_joy_y   = float(np.clip(true_joy[1] + self._drift[1], -1.0, 1.0))
        obs_joy_rot = float(np.clip(self._drift[2], -1.0, 1.0))

        vx_n    = self._robot.vx    / MAX_SPEED_MPS
        vy_n    = self._robot.vy    / MAX_SPEED_MPS
        omega_n = self._robot.omega / MAX_ANGULAR_RPS

        sin_h = math.sin(heading)
        cos_h = math.cos(heading)

        rx_n = rx / FIELD_LENGTH
        ry_n = ry / FIELD_WIDTH

        return np.array([
            obs_joy_x, obs_joy_y, obs_joy_rot,
            vx_n, vy_n, omega_n,
            sin_h, cos_h,
            rx_n, ry_n,
        ], dtype=np.float32)

    # ──────────────────────────────────────────────────────────────────────────
    # Helpers
    # ──────────────────────────────────────────────────────────────────────────

    def _pos_in_obstacle(self, x: float, y: float) -> bool:
        r = ROBOT_BUMPER_HALF
        if x < r or x > FIELD_LENGTH - r:
            return True
        if y < r or y > FIELD_WIDTH - r:
            return True
        for (ox1, oy1, ox2, oy2) in IMPASSABLE_RECTS:
            if ox1 - r < x < ox2 + r and oy1 - r < y < oy2 + r:
                return True
        return False

    def _sample_safe_pos(self):
        for _ in range(500):
            x = float(self.np_random.uniform(
                ROBOT_BUMPER_HALF + 0.1, FIELD_LENGTH - ROBOT_BUMPER_HALF - 0.1
            ))
            y = float(self.np_random.uniform(
                ROBOT_BUMPER_HALF + 0.1, FIELD_WIDTH  - ROBOT_BUMPER_HALF - 0.1
            ))
            if not self._pos_in_obstacle(x, y):
                return x, y
        return FIELD_LENGTH / 2.0, FIELD_WIDTH / 2.0
