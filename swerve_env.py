# ──────────────────────────────────────────────────────────────────────────────
# swerve_env.py
# Gymnasium environment for swerve path-following (Phase 1: translation only).
#
# Action space  : Box(3,) — [vx, vy, omega] normalized to [-1, 1]
#                 omega output is zeroed in Phase 1 (heading locked).
# Observation   : See _get_obs() for the full vector layout.
# Reward        : See _compute_reward() — all weights live in constants.py.
# ──────────────────────────────────────────────────────────────────────────────

import math
import numpy as np
import gymnasium as gym
from gymnasium import spaces

from kinematics import SwerveState
from field_path import (
    WAYPOINTS, NUM_WAYPOINTS, TOTAL_LENGTH,
    nearest_segment, progress_fraction, waypoint_relative, WaypointTracker,
)
from constants import (
    MAX_SPEED_MPS, MAX_ANGULAR_RPS,
    MAX_EPISODE_STEPS, OFF_PATH_LIMIT,
    FIELD_LENGTH, FIELD_WIDTH,
    ROBOT_BUMPER_HALF, IMPASSABLE_RECTS,
    # Reward weights
    RW_PROGRESS, RW_VEL_ALIGN, RW_CROSS_TRACK, RW_SMOOTH_VEL,
    RW_SPEED_MAGNITUDE, RW_TIME_PENALTY,
    RW_WAYPOINT_BONUS, RW_GOAL_BONUS, RW_OFF_PATH_PENALTY, RW_COLLISION_PENALTY,
)

# Observation vector indices (document here so swerve_env and render agree)
# [ vx_n, vy_n,               # 0-1  current robot velocity (normalized)
#   dx0, dy0,                  # 2-3  next waypoint relative pos (meters, robot frame)
#   dx1, dy1,                  # 4-5  waypoint+1 relative pos
#   progress,                  # 6    arc-length progress [0, 1]
#   cross_track,               # 7    signed cross-track error (meters, clamped)
#   heading ]                  # 8    robot heading (radians, kept for Phase 2 compat)
OBS_DIM = 9
OBS_LABELS = ["vx_n","vy_n","dx0","dy0","dx1","dy1","progress","cross_track","heading"]


class SwerveEnv(gym.Env):

    metadata = {"render_modes": ["human"], "render_fps": 60}

    def __init__(self, render_mode=None):
        super().__init__()
        self.render_mode = render_mode

        # Action: [vx, vy, omega] all in [-1, 1]; scaled by max limits in step()
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(3,), dtype=np.float32
        )

        # Observation bounds
        obs_low  = np.array([-1, -1,
                              -FIELD_LENGTH, -FIELD_WIDTH,
                              -FIELD_LENGTH, -FIELD_WIDTH,
                               0.0, -OFF_PATH_LIMIT,
                              -math.pi], dtype=np.float32)
        obs_high = np.array([ 1,  1,
                               FIELD_LENGTH,  FIELD_WIDTH,
                               FIELD_LENGTH,  FIELD_WIDTH,
                               1.0,  OFF_PATH_LIMIT,
                               math.pi], dtype=np.float32)
        self.observation_space = spaces.Box(obs_low, obs_high, dtype=np.float32)

        self._robot   = SwerveState()
        self._tracker = WaypointTracker()

        self._prev_action    = np.zeros(3, dtype=np.float32)
        self._prev_arc_pos   = 0.0
        self._step_count     = 0

        self._renderer = None   # created lazily on first render() call

    # ──────────────────────────────────────────────────────────────────────────
    # Gymnasium API
    # ──────────────────────────────────────────────────────────────────────────

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        sx, sy = WAYPOINTS[0]
        self._robot.reset(x=sx, y=sy, heading=0.0)
        self._tracker.reset()
        self._prev_action  = np.zeros(3, dtype=np.float32)
        _, _, _, _, _, arc, _ = nearest_segment(sx, sy)
        self._prev_arc_pos = arc
        self._step_count   = 0
        return self._get_obs(), {}

    def step(self, action: np.ndarray):
        action = np.clip(action, -1.0, 1.0).astype(np.float32)

        # Scale normalized action to physical units; zero omega in Phase 1
        cmd_vx    = float(action[0]) * MAX_SPEED_MPS
        cmd_vy    = float(action[1]) * MAX_SPEED_MPS
        cmd_omega = 0.0     # Phase 1: translation-only

        self._robot.step(cmd_vx, cmd_vy, cmd_omega)
        self._step_count += 1

        rx, ry = self._robot.x, self._robot.y
        advanced = self._tracker.update(rx, ry)

        # Restrict segment search to near the current tracker position.
        # Without this, the closed-loop start/end overlap lets the robot latch
        # onto segment 13→14 (arc_pos ~31 m) from the spawn point, giving a
        # massive false progress reward that teaches it to run the path backwards.
        hint = max(0, self._tracker.current_idx - 1)
        seg_idx, _, _, _, dist, arc_pos, cross_sign = nearest_segment(rx, ry, hint_seg=hint)
        cross_track = dist * cross_sign

        reward, info = self._compute_reward(
            seg_idx, arc_pos, cross_track, action, advanced
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
            from renderer import Renderer
            self._renderer = Renderer()
        self._renderer.draw(self._robot, self._tracker, self._get_module_states())

    def close(self):
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None

    # ──────────────────────────────────────────────────────────────────────────
    # Collision detection
    # ──────────────────────────────────────────────────────────────────────────

    def _check_collision(self) -> bool:
        """
        True if the robot overlaps an impassable zone or the field boundary.

        The robot is modelled as a circle with radius ROBOT_BUMPER_HALF.
        For rectangular obstacles, this is equivalent to checking whether the
        robot centre falls inside the obstacle expanded by ROBOT_BUMPER_HALF
        on all sides — a standard swept-circle vs AABB test.
        """
        rx, ry = self._robot.x, self._robot.y
        r = ROBOT_BUMPER_HALF

        # Field boundary walls
        if rx - r < 0 or rx + r > FIELD_LENGTH:
            return True
        if ry - r < 0 or ry + r > FIELD_WIDTH:
            return True

        # Impassable field structures (hubs and trenches)
        for (ox1, oy1, ox2, oy2) in IMPASSABLE_RECTS:
            if rx > ox1 - r and rx < ox2 + r and ry > oy1 - r and ry < oy2 + r:
                return True

        return False

    # ──────────────────────────────────────────────────────────────────────────
    # Reward function  <-- EDIT WEIGHTS IN constants.py
    # ──────────────────────────────────────────────────────────────────────────

    def _compute_reward(self, seg_idx, arc_pos, cross_track, action, waypoints_advanced):
        """
        Dense reward signal for path following.

        Components:
          progress     : arc-length delta this step — NEGATIVE if going backward,
                         so the agent is penalised for oscillating/reversing
          vel_align    : velocity dot-product with path direction, normalised to [-1,1]
                         this is the primary anti-oscillation term; gives a continuous
                         gradient even when arc_delta is near zero at segment boundaries
          cross_track  : penalty proportional to distance from path centreline
          smooth_vel   : penalty on change in action between steps (jerk)
          speed_mag    : tiny penalty on action magnitude
          waypoint     : bonus each time a waypoint is passed
          goal         : large bonus on path completion

        All weights are in constants.py under "Reward weights".
        """
        # --- Progress (arc-length delta, sign preserved) ---
        arc_delta = arc_pos - self._prev_arc_pos
        # Suppress sub-millimetre noise but keep genuine backward movement negative
        if abs(arc_delta) < 0.001:
            arc_delta = 0.0
        r_progress = RW_PROGRESS * arc_delta

        # --- Velocity alignment toward current target waypoint ---
        # Uses the tracker's live target, not the nearest segment.
        # This means once the robot passes wp1 and the target becomes wp2,
        # going backward toward wp1 gives NEGATIVE vel_align — closing the
        # wp1-loop exploit where the agent recycled the waypoint bonus.
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

        vel_align   = (vx_w * pdx + vy_w * pdy) / MAX_SPEED_MPS   # [-1, 1]
        r_vel_align = RW_VEL_ALIGN * vel_align

        # --- Other terms ---
        r_cross     = RW_CROSS_TRACK   * abs(cross_track)
        r_smooth    = RW_SMOOTH_VEL    * float(np.linalg.norm(action - self._prev_action))
        r_speed_mag = RW_SPEED_MAGNITUDE * float(np.linalg.norm(action))
        r_time      = RW_TIME_PENALTY  # flat per-step cost — punishes loitering
        r_waypoint  = RW_WAYPOINT_BONUS * waypoints_advanced
        r_goal      = RW_GOAL_BONUS if self._tracker.done else 0.0

        reward = (r_progress + r_vel_align + r_cross +
                  r_smooth + r_speed_mag + r_time + r_waypoint + r_goal)

        info = {
            "r_progress":   r_progress,
            "r_vel_align":  r_vel_align,
            "r_cross":      r_cross,
            "r_smooth":     r_smooth,
            "r_time":       r_time,
            "r_waypoint":   r_waypoint,
            "r_goal":       r_goal,
            "arc_pos":      arc_pos,
            "cross_track":  cross_track,
        }
        return reward, info

    # ──────────────────────────────────────────────────────────────────────────
    # Observation builder
    # ──────────────────────────────────────────────────────────────────────────

    def _get_obs(self):
        rx, ry, heading = self._robot.x, self._robot.y, self._robot.heading

        # Current velocity normalized to [-1, 1]
        vx_n = self._robot.vx / MAX_SPEED_MPS
        vy_n = self._robot.vy / MAX_SPEED_MPS

        # Next two waypoints in robot local frame
        t0 = min(self._tracker.current_idx, NUM_WAYPOINTS - 1)
        t1 = min(t0 + 1, NUM_WAYPOINTS - 1)
        dx0, dy0 = waypoint_relative(rx, ry, heading, t0)
        dx1, dy1 = waypoint_relative(rx, ry, heading, t1)

        # Arc-length progress and cross-track error
        _, _, _, _, dist, arc_pos, cross_sign = nearest_segment(rx, ry)
        prog = progress_fraction(arc_pos)
        cross = dist * cross_sign
        cross = float(np.clip(cross, -OFF_PATH_LIMIT, OFF_PATH_LIMIT))

        return np.array([
            vx_n, vy_n,
            dx0, dy0,
            dx1, dy1,
            prog,
            cross,
            heading,
        ], dtype=np.float32)

    def _get_module_states(self):
        """Returns current module (angle, speed) list for the renderer."""
        return self._robot.module_states
