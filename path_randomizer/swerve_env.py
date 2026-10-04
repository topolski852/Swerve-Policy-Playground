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
# Observation   : 8-element vector — see OBS_LABELS. Goal vectors are in units
#                 of GOAL_SCALE (6 m), length capped at 1 so direction is kept.
# Reward        : monotone approach reward + arrival bonuses (no milestone rings)
# ──────────────────────────────────────────────────────────────────────────────

import math
import numpy as np
import gymnasium as gym
from gymnasium import spaces

from lib.kinematics import SwerveState
from lib.accel_limiter import AccelLimiter
from lib.field_constants import (
    FIELD_LENGTH, FIELD_WIDTH, DT,
    ROBOT_BUMPER_HALF, IMPASSABLE_RECTS,
)
from path_randomizer.constants import (
    ROBOT_MAX_SPEED, SLIP_ACCEL_MPS2, TORQUE_ACCEL_MPS2, PASS_RADIUS,
    N_WAYPOINTS_MIN, N_WAYPOINTS_MAX, NODE_TIME_LIMIT_STEPS,
    MAX_WAYPOINT_DISTANCE, MIN_WAYPOINT_DISTANCE,
    RW_APPROACH, RW_WAYPOINT_BONUS, RW_GOAL_BONUS,
    RW_TIME_PENALTY, RW_COLLISION_PENALTY,
    OBSTACLE_DANGER_MARGIN, RW_OBSTACLE_PROXIMITY,
)

# Goal vectors are divided by the longest leg (6 m), not the 18.3 m field
# diagonal. Over the diagonal a 1–5 m leg became 0.05–0.27 and the pass radius
# 0.02, too small for the network to steer by; a 1-waypoint test learned
# noticeably faster with /6 m.
GOAL_SCALE = MAX_WAYPOINT_DISTANCE

OBS_DIM    = 8
OBS_LABELS = ["vx_n", "vy_n", "rx_n", "ry_n", "dx0_n", "dy0_n", "dx1_n", "dy1_n"]


def goal_vector(rx, ry, wx, wy):
    """Robot-to-waypoint vector in units of GOAL_SCALE. A vector longer than
    GOAL_SCALE is shrunk to length 1 (not clipped per axis) so it still points
    the right way; the next node can be up to 12 m away."""
    dx, dy = (wx - rx) / GOAL_SCALE, (wy - ry) / GOAL_SCALE
    n = math.hypot(dx, dy)
    if n > 1.0:
        dx, dy = dx / n, dy / n
    return float(dx), float(dy)


class WaypointTracker:
    """Minimal tracker: advance current_idx when robot is within pass radius."""

    def __init__(self):
        self._waypoints  = []
        self.current_idx = 0   # named current_idx for renderer compatibility

    def reset(self, waypoints, start_idx=0):
        self._waypoints  = list(waypoints)
        self.current_idx = start_idx

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
            if math.hypot(robot_x - wx, robot_y - wy) < PASS_RADIUS:
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

        obs_low  = np.array([-1., -1.,  0.,  0., -1., -1., -1., -1.], dtype=np.float32)
        obs_high = np.array([ 1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.], dtype=np.float32)
        self.observation_space = spaces.Box(obs_low, obs_high, dtype=np.float32)

        self._robot      = SwerveState()
        self._limiter    = AccelLimiter(SLIP_ACCEL_MPS2, TORQUE_ACCEL_MPS2, ROBOT_MAX_SPEED, DT)
        self._tracker    = WaypointTracker()
        self._waypoints  = []
        self._step_count = 0
        self._renderer   = None

        # Configurable difficulty — defaults to full training values.
        # test_randomizer.py may override these via setattr for diagnostic runs.
        self._n_waypoints_min = N_WAYPOINTS_MIN
        self._n_waypoints_max = N_WAYPOINTS_MAX
        self._wp_distance_max = MAX_WAYPOINT_DISTANCE

        # Monotone approach tracker: seeded to actual distance on each reset/advance
        # so the first step only earns reward for real progress (never inf).
        self._best_dist_to_wp = 0.0

    # ──────────────────────────────────────────────────────────────────────────
    # Gymnasium API
    # ──────────────────────────────────────────────────────────────────────────

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)

        sx, sy = self._random_valid_pos()
        self._robot.reset(x=sx, y=sy, heading=0.0)
        self._limiter.reset()

        n = int(self.np_random.integers(self._n_waypoints_min, self._n_waypoints_max + 1))
        # Chain each waypoint within _wp_distance_max of the previous so
        # the agent never has to cross the full field in one hop.
        nav_wps = []
        prev_x, prev_y = sx, sy
        for _ in range(n):
            wx, wy = self._random_pos_near(prev_x, prev_y, self._wp_distance_max)
            nav_wps.append((wx, wy))
            prev_x, prev_y = wx, wy

        # Prepend start as wp0 so the path overlay connects from the spawn point.
        # Tracker starts at index 1 — robot is already at wp0.
        self._waypoints = [(sx, sy)] + nav_wps
        self._tracker.reset(self._waypoints, start_idx=1)

        self._step_count = 0
        self._node_steps = 0   # steps spent on the current node
        # Seed best-dist to the actual starting distance so the first step only
        # earns reward for real progress, not for the inf → real_dist gap.
        if not self._tracker.done:
            wx, wy = self._tracker.current
            self._best_dist_to_wp = math.hypot(sx - wx, sy - wy)
        else:
            self._best_dist_to_wp = 0.0
        return self._get_obs(), {}

    def step(self, action: np.ndarray):
        action = np.clip(action, -1.0, 1.0).astype(np.float32)
        self._drive(float(action[0]) * ROBOT_MAX_SPEED, float(action[1]) * ROBOT_MAX_SPEED)
        self._step_count += 1
        self._node_steps += 1

        rx, ry = self._robot.x, self._robot.y

        # ── Approach reward + arrival ─────────────────────────────────────────
        approach_reward = 0.0
        waypoint_bonus  = 0.0
        if not self._tracker.done:
            wx, wy = self._tracker.current
            dist   = math.hypot(rx - wx, ry - wy)

            # Monotone approach reward: only fires when the robot sets a new
            # personal-best distance to the current waypoint. Back-and-forth
            # oscillation earns nothing because it can't beat the existing best.
            approach_reward       = max(0.0, self._best_dist_to_wp - dist) * RW_APPROACH
            self._best_dist_to_wp = min(self._best_dist_to_wp, dist)

            advanced = self._tracker.update(rx, ry)
            if advanced:
                self._node_steps = 0
                waypoint_bonus = RW_WAYPOINT_BONUS * advanced
                # Seed to actual distance to the new current waypoint, not inf.
                if not self._tracker.done:
                    wx2, wy2 = self._tracker.current
                    self._best_dist_to_wp = math.hypot(rx - wx2, ry - wy2)
                else:
                    self._best_dist_to_wp = 0.0

        goal_done = self._tracker.done

        # ── Obstacle proximity shaping ────────────────────────────────────────
        # AABB-to-AABB distance, consistent with _check_collision (square bumpers).
        # bx/by = gap from bumper face to obstacle face on each axis; 0 when touching.
        # bumper_dist = 0 at collision boundary, OBSTACLE_DANGER_MARGIN at outer edge.
        proximity_penalty = 0.0
        r        = ROBOT_BUMPER_HALF
        detect_r = r + OBSTACLE_DANGER_MARGIN
        for (ox1, oy1, ox2, oy2) in IMPASSABLE_RECTS:
            if (rx > ox1 - detect_r and rx < ox2 + detect_r and
                    ry > oy1 - detect_r and ry < oy2 + detect_r):
                bx = max(0.0, max(ox1 - rx, rx - ox2) - r)
                by = max(0.0, max(oy1 - ry, ry - oy2) - r)
                bumper_dist = math.hypot(bx, by)
                if bumper_dist < OBSTACLE_DANGER_MARGIN:
                    depth = 1.0 - bumper_dist / OBSTACLE_DANGER_MARGIN
                    proximity_penalty += RW_OBSTACLE_PROXIMITY * depth

        # ── Reward ────────────────────────────────────────────────────────────
        reward = (
            approach_reward
            + waypoint_bonus
            + (RW_GOAL_BONUS if goal_done else 0.0)
            + RW_TIME_PENALTY
            + proximity_penalty
        )

        # ── Termination ───────────────────────────────────────────────────────
        # A crash is a real ending (terminated), not a time limit (truncated).
        # SB3 treats truncated as "the episode was cut short, the future would
        # have continued" and adds the critic's guess of that future to the
        # crash penalty, so the agent never learned how bad a crash really is.
        collision  = self._check_collision()
        terminated = goal_done or collision
        truncated  = (self._node_steps >= NODE_TIME_LIMIT_STEPS) and not terminated

        if collision:
            reward += RW_COLLISION_PENALTY

        obs  = self._get_obs()
        info = {
            "waypoint_idx": self._tracker.current_idx,
            "n_waypoints":  len(self._waypoints),
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

    def _drive(self, cmd_vx, cmd_vy):
        """One 20 ms loop, the way 1507Base drives: the command goes through the
        same slip/torque acceleration limiter (Swerve.drive -> SwerveAccelLimiter),
        then the robot moves at the limited velocity. Field-relative; heading stays 0."""
        vx, vy = self._limiter.limit(cmd_vx, cmd_vy)
        self._robot.vx, self._robot.vy, self._robot.omega = vx, vy, 0.0
        self._robot.x += vx * DT
        self._robot.y += vy * DT

    def _get_obs(self):
        vx_n = float(np.clip(self._robot.vx / ROBOT_MAX_SPEED, -1.0, 1.0))
        vy_n = float(np.clip(self._robot.vy / ROBOT_MAX_SPEED, -1.0, 1.0))
        rx, ry = self._robot.x, self._robot.y
        rx_n = float(np.clip(rx / FIELD_LENGTH, 0.0, 1.0))
        ry_n = float(np.clip(ry / FIELD_WIDTH,  0.0, 1.0))

        if self._tracker.done:
            dx0_n = dy0_n = dx1_n = dy1_n = 0.0
        else:
            dx0_n, dy0_n = goal_vector(rx, ry, *self._tracker.current)
            dx1_n, dy1_n = goal_vector(rx, ry, *self._tracker.next_wp)

        return np.array([vx_n, vy_n, rx_n, ry_n, dx0_n, dy0_n, dx1_n, dy1_n],
                        dtype=np.float32)

    def _random_valid_pos(self):
        r = ROBOT_BUMPER_HALF
        pad = r + 0.1
        for _ in range(200):
            x = float(self.np_random.uniform(pad, FIELD_LENGTH - pad))
            y = float(self.np_random.uniform(pad, FIELD_WIDTH  - pad))
            if self._pos_valid(x, y, r):
                return x, y
        return FIELD_LENGTH / 2, FIELD_WIDTH / 2  # fallback: midfield

    def _random_pos_near(self, cx, cy, max_dist):
        """Random valid position within max_dist metres of (cx, cy)."""
        r = ROBOT_BUMPER_HALF
        pad = r + 0.1
        for _ in range(200):
            angle = float(self.np_random.uniform(0.0, 2.0 * math.pi))
            dist  = float(self.np_random.uniform(MIN_WAYPOINT_DISTANCE, max_dist))
            x = cx + dist * math.cos(angle)
            y = cy + dist * math.sin(angle)
            if x < pad or x > FIELD_LENGTH - pad: continue
            if y < pad or y > FIELD_WIDTH  - pad: continue
            if self._pos_valid(x, y, r):
                return x, y
        return self._random_valid_pos()  # fallback: unconstrained random

    def _pos_valid(self, x, y, r):
        for ox1, oy1, ox2, oy2 in IMPASSABLE_RECTS:
            if x > ox1 - r and x < ox2 + r and y > oy1 - r and y < oy2 + r:
                return False
        return True

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
