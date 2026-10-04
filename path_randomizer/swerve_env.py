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
# Observation   : 8 + N_RAYS values — see OBS_LABELS. Goal vectors are in units
#                 of GOAL_SCALE (6 m), length capped at 1 so direction is kept.
#                 Rays: clearance to field elements/walls (path_randomizer/rays.py).
# Reward        : progress + velocity alignment each step, arrival bonuses
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
from path_randomizer.rays import cast_rays
from path_randomizer.constants import (
    N_RAYS,
    ROBOT_MAX_SPEED, SLIP_ACCEL_MPS2, TORQUE_ACCEL_MPS2, PASS_RADIUS,
    NODE_TIME_LIMIT_STEPS, CURRICULUM,
    MAX_WAYPOINT_DISTANCE, MIN_WAYPOINT_DISTANCE,
    RW_PROGRESS, RW_VEL_ALIGN, RW_WAYPOINT_BONUS, RW_GOAL_BONUS,
    RW_TIME_PENALTY, RW_COLLISION_PENALTY,
)

# Goal vectors are divided by the longest leg (6 m), not the 18.3 m field
# diagonal. Over the diagonal a 1–5 m leg became 0.05–0.27 and the pass radius
# 0.02, too small for the network to steer by; a 1-waypoint test learned
# noticeably faster with /6 m.
GOAL_SCALE = MAX_WAYPOINT_DISTANCE

OBS_LABELS = (["vx_n", "vy_n", "rx_n", "ry_n", "dx0_n", "dy0_n", "dx1_n", "dy1_n"]
              + [f"ray{k}" for k in range(N_RAYS)])
OBS_DIM    = len(OBS_LABELS)


def goal_vector(rx, ry, wx, wy):
    """Robot-to-waypoint vector in units of GOAL_SCALE. A vector longer than
    GOAL_SCALE is shrunk to length 1 (not clipped per axis) so it still points
    the right way; the next node can be up to 12 m away."""
    dx, dy = (wx - rx) / GOAL_SCALE, (wy - ry) / GOAL_SCALE
    n = math.hypot(dx, dy)
    if n > 1.0:
        dx, dy = dx / n, dy / n
    return float(dx), float(dy)


def leg_clear(ax, ay, bx, by):
    """True if the robot can drive the straight line A->B without touching a
    field element (each element grown by the bumper half-width; slab test)."""
    r = ROBOT_BUMPER_HALF
    dx, dy = bx - ax, by - ay
    for ox1, oy1, ox2, oy2 in IMPASSABLE_RECTS:
        t0, t1 = 0.0, 1.0
        hit = True
        for lo, hi, p, d in ((ox1 - r, ox2 + r, ax, dx), (oy1 - r, oy2 + r, ay, dy)):
            if abs(d) < 1e-12:
                if p <= lo or p >= hi:
                    hit = False
                    break
            else:
                ta, tb = (lo - p) / d, (hi - p) / d
                if ta > tb:
                    ta, tb = tb, ta
                t0, t1 = max(t0, ta), min(t1, tb)
                if t0 >= t1:
                    hit = False
                    break
        if hit:
            return False
    return True


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

        obs_low  = np.array([-1., -1.,  0.,  0., -1., -1., -1., -1.] + [0.] * N_RAYS, dtype=np.float32)
        obs_high = np.array([ 1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.] + [1.] * N_RAYS, dtype=np.float32)
        self.observation_space = spaces.Box(obs_low, obs_high, dtype=np.float32)

        self._robot      = SwerveState()
        self._limiter    = AccelLimiter(SLIP_ACCEL_MPS2, TORQUE_ACCEL_MPS2, ROBOT_MAX_SPEED, DT)
        self._tracker    = WaypointTracker()
        self._waypoints  = []
        self._step_count = 0
        self._renderer   = None

        # Difficulty: a CURRICULUM stage. Defaults to the last (full) stage;
        # train_randomizer.py starts at 0 and moves up with set_stage().
        self.set_stage(len(CURRICULUM) - 1)

        # Distance to the current node last step, for the progress reward.
        self._prev_dist = 0.0

    # ──────────────────────────────────────────────────────────────────────────
    # Gymnasium API
    # ──────────────────────────────────────────────────────────────────────────

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)

        sx, sy = self._random_valid_pos()
        self._robot.reset(x=sx, y=sy, heading=0.0)
        self._limiter.reset()

        st = CURRICULUM[self.stage]
        n  = int(self.np_random.integers(st["n_min"], st["n_max"] + 1))
        # Chain each waypoint within max_dist of the previous so
        # the agent never has to cross the full field in one hop.
        nav_wps = []
        prev_x, prev_y = sx, sy
        for _ in range(n):
            wx, wy = self._random_pos_near(prev_x, prev_y, st["max_dist"], st["clear_legs"])
            nav_wps.append((wx, wy))
            prev_x, prev_y = wx, wy

        # Prepend start as wp0 so the path overlay connects from the spawn point.
        # Tracker starts at index 1 — robot is already at wp0.
        self._waypoints = [(sx, sy)] + nav_wps
        self._tracker.reset(self._waypoints, start_idx=1)

        self._step_count = 0
        self._node_steps = 0   # steps spent on the current node
        self._prev_dist  = self._dist_to_node()
        return self._get_obs(), {}

    def step(self, action: np.ndarray):
        action = np.clip(action, -1.0, 1.0).astype(np.float32)
        self._drive(float(action[0]) * ROBOT_MAX_SPEED, float(action[1]) * ROBOT_MAX_SPEED)
        self._step_count += 1
        self._node_steps += 1

        rx, ry = self._robot.x, self._robot.y

        # ── Progress + alignment + arrival ────────────────────────────────────
        progress_reward = 0.0
        align_reward    = 0.0
        waypoint_bonus  = 0.0
        if not self._tracker.done:
            wx, wy = self._tracker.current
            dist   = math.hypot(rx - wx, ry - wy)

            progress_reward = RW_PROGRESS * (self._prev_dist - dist)
            if dist > 1e-6:
                toward = (self._robot.vx * (wx - rx) + self._robot.vy * (wy - ry)) / dist
                align_reward = RW_VEL_ALIGN * toward / ROBOT_MAX_SPEED

            advanced = self._tracker.update(rx, ry)
            if advanced:
                self._node_steps = 0
                waypoint_bonus = RW_WAYPOINT_BONUS * advanced
            self._prev_dist = self._dist_to_node()   # to the new node if it advanced

        goal_done = self._tracker.done

        # ── Reward ────────────────────────────────────────────────────────────
        reward = (
            progress_reward
            + align_reward
            + waypoint_bonus
            + (RW_GOAL_BONUS if goal_done else 0.0)
            + RW_TIME_PENALTY
        )

        # ── Termination ───────────────────────────────────────────────────────
        # Crashing and giving up on a node are real endings (terminated), not
        # time limits (truncated). SB3 treats truncated as "the episode was cut
        # short, the future would have continued" and adds the critic's guess of
        # that future, so the agent never learned what these endings really cost.
        # On the robot, giving up on a node ends the route: every node after it
        # is lost too.
        collision  = self._check_collision()
        gave_up    = self._node_steps >= NODE_TIME_LIMIT_STEPS
        terminated = goal_done or collision or gave_up
        truncated  = False

        if collision:
            reward += RW_COLLISION_PENALTY

        obs  = self._get_obs()
        info = {
            "waypoint_idx": self._tracker.current_idx,
            "n_waypoints":  len(self._waypoints),
            "stage":        self.stage,
        }
        if terminated or truncated:
            info["outcome"] = ("crash" if collision else
                               "complete" if goal_done else "timeout")

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

    def set_stage(self, stage):
        """Pick the CURRICULUM stage used from the next reset() on."""
        self.stage = int(stage)

    def _dist_to_node(self):
        if self._tracker.done:
            return 0.0
        wx, wy = self._tracker.current
        return math.hypot(self._robot.x - wx, self._robot.y - wy)

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

        rays = cast_rays(rx, ry)
        return np.concatenate([
            np.array([vx_n, vy_n, rx_n, ry_n, dx0_n, dy0_n, dx1_n, dy1_n], dtype=np.float32),
            rays,
        ])

    def _random_valid_pos(self):
        r = ROBOT_BUMPER_HALF
        pad = r + 0.1
        for _ in range(200):
            x = float(self.np_random.uniform(pad, FIELD_LENGTH - pad))
            y = float(self.np_random.uniform(pad, FIELD_WIDTH  - pad))
            if self._pos_valid(x, y, r):
                return x, y
        return FIELD_LENGTH / 2, FIELD_WIDTH / 2  # fallback: midfield

    def _random_pos_near(self, cx, cy, max_dist, clear_leg=False):
        """Random valid position within max_dist metres of (cx, cy). With
        clear_leg, the straight line from (cx, cy) must also miss every field
        element (falls back to allowing a blocked leg if none is found)."""
        r = ROBOT_BUMPER_HALF
        pad = r + 0.1
        for attempt in range(400):
            angle = float(self.np_random.uniform(0.0, 2.0 * math.pi))
            dist  = float(self.np_random.uniform(MIN_WAYPOINT_DISTANCE, max_dist))
            x = cx + dist * math.cos(angle)
            y = cy + dist * math.sin(angle)
            if x < pad or x > FIELD_LENGTH - pad: continue
            if y < pad or y > FIELD_WIDTH  - pad: continue
            if not self._pos_valid(x, y, r): continue
            if clear_leg and attempt < 200 and not leg_clear(cx, cy, x, y):
                continue
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
