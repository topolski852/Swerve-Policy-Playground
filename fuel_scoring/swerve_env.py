# ──────────────────────────────────────────────────────────────────────────────
# fuel_scoring/swerve_env.py
# Gymnasium environment for the fuel scoring experiment.
#
# Action space  : Box(3,) — [vx, vy, omega] normalized to [-1, 1]
#                 omega zeroed (translation-only phase)
# Observation   : 10-element vector — see OBS_LABELS
# Reward        : fuel scoring dominant; navigation secondary
#                 weights in fuel_scoring/constants.py
# ──────────────────────────────────────────────────────────────────────────────

import math
import numpy as np
import gymnasium as gym
from gymnasium import spaces

from lib.kinematics import SwerveState
from lib.field_constants import (
    MAX_SPEED_MPS,
    OFF_PATH_LIMIT, FIELD_LENGTH, FIELD_WIDTH,
    ROBOT_BUMPER_HALF, IMPASSABLE_RECTS,
    BLUE_ALLIANCE_MAX_X, NEUTRAL_MIN_X, NEUTRAL_MAX_X,
    BLUE_HUB_CENTER,
)
from fuel_scoring.field_path import (
    WAYPOINTS, NUM_WAYPOINTS, TOTAL_LENGTH,
    nearest_segment, progress_fraction, waypoint_relative, WaypointTracker,
)
from fuel_scoring.constants import (
    MAX_EPISODE_STEPS, SCORING_START_HOPPER,
    HOPPER_CAPACITY, FUEL_FILL_RATE, FUEL_FILL_MIN_SPEED, FUEL_SHOOT_RATE,
    RW_FUEL_SCORED, RW_FUEL_COLLECTED, RW_FULL_HOPPER_IN_NEUTRAL,
    RW_PROGRESS, RW_VEL_ALIGN, RW_CROSS_TRACK, RW_SMOOTH_VEL,
    RW_SPEED_MAGNITUDE, RW_TIME_PENALTY,
    RW_WAYPOINT_BONUS, RW_GOAL_BONUS, RW_OFF_PATH_PENALTY, RW_COLLISION_PENALTY,
)

OBS_DIM    = 10
OBS_LABELS = [
    "vx_n", "vy_n",
    "dx0", "dy0",
    "dx1", "dy1",
    "progress", "cross_track", "heading",
    "hopper_norm",
]


class SwerveEnv(gym.Env):

    metadata = {"render_modes": ["human"], "render_fps": 60}

    def __init__(self, render_mode=None):
        super().__init__()
        self.render_mode = render_mode

        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(3,), dtype=np.float32
        )

        obs_low  = np.array([-1, -1,
                              -FIELD_LENGTH, -FIELD_WIDTH,
                              -FIELD_LENGTH, -FIELD_WIDTH,
                               0.0, -OFF_PATH_LIMIT,
                              -math.pi,
                               0.0], dtype=np.float32)
        obs_high = np.array([ 1,  1,
                               FIELD_LENGTH,  FIELD_WIDTH,
                               FIELD_LENGTH,  FIELD_WIDTH,
                               1.0,  OFF_PATH_LIMIT,
                               math.pi,
                               1.0], dtype=np.float32)
        self.observation_space = spaces.Box(obs_low, obs_high, dtype=np.float32)

        self._robot   = SwerveState()
        self._tracker = WaypointTracker()
        self._hopper  = SCORING_START_HOPPER

        self._prev_action  = np.zeros(3, dtype=np.float32)
        self._prev_arc_pos = 0.0
        self._step_count   = 0

        self._renderer = None

    # ──────────────────────────────────────────────────────────────────────────
    # Gymnasium API
    # ──────────────────────────────────────────────────────────────────────────

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        sx, sy = WAYPOINTS[0]   # start at (3.50, 2.55) — Blue Alliance Zone
        self._robot.reset(x=sx, y=sy, heading=0.0)
        self._tracker.reset()
        self._hopper       = SCORING_START_HOPPER
        self._prev_action  = np.zeros(3, dtype=np.float32)
        res = nearest_segment(sx, sy)
        self._prev_arc_pos = res[5]   # arc_pos at index 5
        self._step_count   = 0
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
        fuel_collected = 0.0
        if NEUTRAL_MIN_X < rx < NEUTRAL_MAX_X and speed >= FUEL_FILL_MIN_SPEED:
            collected = min(HOPPER_CAPACITY - self._hopper, FUEL_FILL_RATE)
            self._hopper  += collected
            fuel_collected = collected

        fuel_scored = 0.0
        if rx < BLUE_ALLIANCE_MAX_X and self._hopper > 0 and speed >= FUEL_FILL_MIN_SPEED:
            shot = min(self._hopper, FUEL_SHOOT_RATE)
            self._hopper -= shot
            fuel_scored   = shot

        # ── Navigation ────────────────────────────────────────────────────────
        advanced = self._tracker.update(rx, ry)

        hint = max(0, self._tracker.current_idx - 1)
        result = nearest_segment(rx, ry, hint_seg=hint)
        seg_idx, _, _, _, dist, arc_pos, cross_sign = result
        cross_track = dist * cross_sign

        reward, info = self._compute_reward(
            seg_idx, arc_pos, cross_track, action, advanced,
            fuel_scored, fuel_collected
        )

        terminated = self._tracker.done
        collision  = self._check_collision()
        truncated  = (dist > OFF_PATH_LIMIT or
                      self._step_count >= MAX_EPISODE_STEPS or
                      collision)

        if dist > OFF_PATH_LIMIT:
            reward += RW_OFF_PATH_PENALTY
            info["off_path"] = True
        if collision:
            reward += RW_COLLISION_PENALTY
            info["collision"] = True

        self._prev_action  = action.copy()
        self._prev_arc_pos = arc_pos

        obs = self._get_obs()

        if self.render_mode == "human":
            self.render()

        return obs, reward, terminated, truncated, info

    def render(self):
        if self._renderer is None:
            from lib.renderer import Renderer
            from fuel_scoring.field_path import WAYPOINTS as _WP
            self._renderer = Renderer(waypoints=_WP)
        self._renderer.draw(self._robot, self._tracker, self._get_module_states())

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

    def _compute_reward(self, seg_idx, arc_pos, cross_track, action,
                        waypoints_advanced, fuel_scored, fuel_collected):
        # --- Fuel scoring (dominant signal) ---
        r_fuel_scored    = RW_FUEL_SCORED   * fuel_scored
        r_fuel_collected = RW_FUEL_COLLECTED * fuel_collected

        # Penalty for sitting in neutral with a full hopper
        hopper_full_in_neutral = (
            NEUTRAL_MIN_X < self._robot.x < NEUTRAL_MAX_X and
            self._hopper >= HOPPER_CAPACITY
        )
        r_full_hopper = RW_FULL_HOPPER_IN_NEUTRAL if hopper_full_in_neutral else 0.0

        # --- Navigation (secondary signals) ---
        arc_delta = arc_pos - self._prev_arc_pos
        if abs(arc_delta) < 0.001:
            arc_delta = 0.0
        r_progress = RW_PROGRESS * arc_delta

        tx, ty = self._tracker.target_waypoint()
        dx_wp  = tx - self._robot.x
        dy_wp  = ty - self._robot.y
        d_wp   = math.hypot(dx_wp, dy_wp)
        if d_wp > 1e-9:
            pdx, pdy = dx_wp / d_wp, dy_wp / d_wp
        else:
            pdx, pdy = 1.0, 0.0

        cos_h = math.cos(self._robot.heading)
        sin_h = math.sin(self._robot.heading)
        vx_w  = self._robot.vx * cos_h - self._robot.vy * sin_h
        vy_w  = self._robot.vx * sin_h + self._robot.vy * cos_h

        vel_align   = (vx_w * pdx + vy_w * pdy) / MAX_SPEED_MPS
        r_vel_align = RW_VEL_ALIGN * vel_align

        r_cross     = RW_CROSS_TRACK   * abs(cross_track)
        r_smooth    = RW_SMOOTH_VEL    * float(np.linalg.norm(action - self._prev_action))
        r_speed_mag = RW_SPEED_MAGNITUDE * float(np.linalg.norm(action))
        r_time      = RW_TIME_PENALTY
        r_waypoint  = RW_WAYPOINT_BONUS * waypoints_advanced
        r_goal      = RW_GOAL_BONUS if self._tracker.done else 0.0

        reward = (r_fuel_scored + r_fuel_collected + r_full_hopper +
                  r_progress + r_vel_align + r_cross +
                  r_smooth + r_speed_mag + r_time + r_waypoint + r_goal)

        info = {
            "r_fuel_scored":    r_fuel_scored,
            "r_fuel_collected": r_fuel_collected,
            "r_progress":       r_progress,
            "r_vel_align":      r_vel_align,
            "r_cross":          r_cross,
            "r_smooth":         r_smooth,
            "r_waypoint":       r_waypoint,
            "r_goal":           r_goal,
            "arc_pos":          arc_pos,
            "hopper_level":     self._hopper / HOPPER_CAPACITY,
            "fuel_scored":      fuel_scored,
            "fuel_collected":   fuel_collected,
        }
        return reward, info

    # ──────────────────────────────────────────────────────────────────────────
    # Observation builder
    # ──────────────────────────────────────────────────────────────────────────

    def _get_obs(self):
        rx, ry, heading = self._robot.x, self._robot.y, self._robot.heading

        vx_n = self._robot.vx / MAX_SPEED_MPS
        vy_n = self._robot.vy / MAX_SPEED_MPS

        t0 = min(self._tracker.current_idx, NUM_WAYPOINTS - 1)
        t1 = min(t0 + 1, NUM_WAYPOINTS - 1)
        dx0, dy0 = waypoint_relative(rx, ry, heading, t0)
        dx1, dy1 = waypoint_relative(rx, ry, heading, t1)

        result = nearest_segment(rx, ry)
        _, _, _, _, dist, arc_pos, cross_sign = result
        prog  = progress_fraction(arc_pos)
        cross = float(np.clip(dist * cross_sign, -OFF_PATH_LIMIT, OFF_PATH_LIMIT))

        hopper_norm = float(np.clip(self._hopper / HOPPER_CAPACITY, 0.0, 1.0))

        return np.array([
            vx_n, vy_n,
            dx0, dy0,
            dx1, dy1,
            prog,
            cross,
            heading,
            hopper_norm,
        ], dtype=np.float32)

    def _get_module_states(self):
        return self._robot.module_states
