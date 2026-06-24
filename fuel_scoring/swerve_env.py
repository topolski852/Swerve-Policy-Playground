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
# Observation   : 8-element vector — see OBS_LABELS
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
    BLUE_HUB_CENTER,
)
from fuel_scoring.constants import (
    PHASE1_EPISODE_STEPS, MAX_EPISODE_STEPS, MATCH_STEP_DT, SCORING_START_HOPPER,
    HOPPER_CAPACITY, FUEL_FILL_RATE_MAX, FUEL_FILL_MIN_SPEED, FUEL_SHOOT_RATE,
    SCORING_OPTIMAL_DEPTH, SCORING_DEPTH_SIGMA,
    CONTRIBUTED_SCORE_NORM, TOTAL_SCORE_NORM,
    RW_FUEL_SCORED, RW_FUEL_COLLECTED,
    RW_FULL_HOPPER_IN_NEUTRAL, RW_EMPTY_HOPPER_IN_ALLIANCE,
    RW_NEUTRAL_IDLE,
    RW_MILESTONE_100, RW_MILESTONE_360, RW_MILESTONE_600, RW_MILESTONE_900,
    RW_JERK,
    RW_BUMP_IDLE, BUMP_IDLE_MAX_SPEED,
    RW_COLLISION_PENALTY,
)

# Default starting position: Blue Alliance Zone, BumpRight corridor center
START_X = 3.50
START_Y = 2.55

# Four alternate starting positions (path_following WP1/2/4/5) used when
# random_start=True. Two inside Blue Alliance Zone, two just past the bump.
RANDOM_STARTS = [
    (5.90, 5.60),   # WP1 — neutral side Blue BumpLeft
    (3.20, 5.60),   # WP2 — alliance side Blue BumpLeft top
    (3.20, 2.61),   # WP4 — alliance side Blue BumpRight bottom
    (5.90, 2.61),   # WP5 — neutral side Blue BumpRight
]

OBS_DIM    = 9
OBS_LABELS = ["vx_n", "vy_n", "rx_n", "ry_n", "hopper_norm",
              "contributed_norm", "total_norm", "hub_dist_norm", "time_remaining_norm"]

HUB_DIST_NORM_MAX = 8.0   # metres — normalises hub distance to [0, 1]


class SwerveEnv(gym.Env):

    metadata = {"render_modes": ["human"], "render_fps": 60}

    def __init__(self, render_mode=None, random_start=False):
        super().__init__()
        self.render_mode   = render_mode
        self._random_start = random_start

        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(3,), dtype=np.float32
        )

        obs_low  = np.array([-1.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0, 0.0], dtype=np.float32)
        obs_high = np.array([ 1.0,  1.0, 1.0, 1.0, 1.0, 1.0, 1.0,  1.0, 1.0], dtype=np.float32)
        self.observation_space = spaces.Box(obs_low, obs_high, dtype=np.float32)

        self._robot             = SwerveState()
        self._hopper            = SCORING_START_HOPPER
        self._contributed_score = 0.0
        self._total_score       = 0.0

        self._step_count             = 0
        self._max_episode_steps      = PHASE1_EPISODE_STEPS  # raised to MAX_EPISODE_STEPS at Phase 2
        self._milestone_100_reached  = False
        self._milestone_360_reached  = False
        self._milestone_600_reached  = False
        self._milestone_900_reached  = False
        self._last_action            = np.zeros(2, dtype=np.float32)
        self._collision_penalty      = RW_COLLISION_PENALTY  # mutable — raised by CollisionRampCallback
        self._renderer               = None

    # ──────────────────────────────────────────────────────────────────────────
    # Gymnasium API
    # ──────────────────────────────────────────────────────────────────────────

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if self._random_start:
            sx, sy = RANDOM_STARTS[self.np_random.integers(len(RANDOM_STARTS))]
            self._hopper = float(self.np_random.uniform(0.0, HOPPER_CAPACITY))
        else:
            sx, sy = START_X, START_Y
            self._hopper = SCORING_START_HOPPER
        self._robot.reset(x=sx, y=sy, heading=0.0)
        self._contributed_score     = 0.0
        self._total_score           = 0.0
        self._step_count            = 0
        self._milestone_100_reached = False
        self._milestone_360_reached = False
        self._milestone_600_reached = False
        self._milestone_900_reached = False
        self._last_action           = np.zeros(2, dtype=np.float32)
        return self._get_obs(), {}

    def step(self, action: np.ndarray):
        action = np.clip(action, -1.0, 1.0).astype(np.float32)

        action_xy    = action[:2]
        action_delta = float(np.mean(np.abs(action_xy - self._last_action)))
        self._last_action = action_xy.copy()

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

        # Score: alliance zone — hopper drains at full rate, scored fuel scaled by:
        #   speed_factor : stationary = 1.0 (best), full speed = 0.1
        #   dist_factor  : Gaussian bell centred at SCORING_OPTIMAL_DEPTH inside
        #                  alliance zone — too close OR too far reduces accuracy.
        #                  Peak at x ≈ 2.53 m; boundary (x=4.029) ≈ 32% efficiency.
        fuel_scored = 0.0
        if rx < BLUE_ALLIANCE_MAX_X - ROBOT_BUMPER_HALF and self._hopper > 0:
            shot = min(self._hopper, FUEL_SHOOT_RATE)
            self._hopper -= shot

            speed_factor   = max(0.1, 1.0 - 0.9 * (speed / MAX_SPEED_MPS))
            alliance_depth = max(0.0, BLUE_ALLIANCE_MAX_X - rx)
            dist_factor    = max(0.1, math.exp(
                -0.5 * ((alliance_depth - SCORING_OPTIMAL_DEPTH) / SCORING_DEPTH_SIGMA) ** 2
            ))
            fuel_scored    = shot * speed_factor * dist_factor

            self._contributed_score += fuel_scored
            self._total_score       += fuel_scored

        # ── Reward ────────────────────────────────────────────────────────────
        reward, info = self._compute_reward(fuel_scored, fuel_collected, speed, action_delta)

        # ── Termination ───────────────────────────────────────────────────────
        collision = self._check_collision()
        terminated = False
        truncated  = self._step_count >= self._max_episode_steps or collision

        if collision:
            reward += self._collision_penalty
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

    def _compute_reward(self, fuel_scored, fuel_collected, speed, action_delta):
        r_fuel_scored    = RW_FUEL_SCORED   * fuel_scored
        r_fuel_collected = RW_FUEL_COLLECTED * fuel_collected

        # Penalty: full hopper sitting in neutral — go score it
        r_full_hopper = (
            RW_FULL_HOPPER_IN_NEUTRAL
            if NEUTRAL_MIN_X < self._robot.x < NEUTRAL_MAX_X and self._hopper >= HOPPER_CAPACITY
            else 0.0
        )

        # Penalty: empty hopper anywhere outside the neutral collection zone.
        # Covers both the alliance zone AND the bump — continuous signal pushing
        # the robot all the way through the bump into neutral when hopper is empty.
        r_empty_alliance = (
            RW_EMPTY_HOPPER_IN_ALLIANCE
            if self._robot.x < NEUTRAL_MIN_X and self._hopper <= 0
            else 0.0
        )

        # Penalty: idling in neutral zone below collection speed with room in hopper
        # Forces the robot to either move fast enough to collect or leave the zone.
        r_neutral_idle = (
            RW_NEUTRAL_IDLE
            if NEUTRAL_MIN_X < self._robot.x < NEUTRAL_MAX_X
               and speed < FUEL_FILL_MIN_SPEED
               and self._hopper < HOPPER_CAPACITY
            else 0.0
        )

        # Penalty: stationary on the bump (transition zone, no scoring or collection here)
        # Transit at any real speed is free — only zero movement is penalised.
        r_bump_idle = (
            RW_BUMP_IDLE
            if BLUE_ALLIANCE_MAX_X <= self._robot.x <= NEUTRAL_MIN_X
               and speed < BUMP_IDLE_MAX_SPEED
            else 0.0
        )

        # One-time milestone bonuses (FRC ranking point equivalents)
        r_milestone = 0.0
        if not self._milestone_100_reached and self._contributed_score >= 100.0:
            r_milestone += RW_MILESTONE_100
            self._milestone_100_reached = True
        if not self._milestone_360_reached and self._contributed_score >= 360.0:
            r_milestone += RW_MILESTONE_360
            self._milestone_360_reached = True
        if not self._milestone_600_reached and self._contributed_score >= 600.0:
            r_milestone += RW_MILESTONE_600
            self._milestone_600_reached = True
        if not self._milestone_900_reached and self._contributed_score >= 900.0:
            r_milestone += RW_MILESTONE_900
            self._milestone_900_reached = True

        r_jerk = RW_JERK * action_delta

        reward = r_fuel_scored + r_fuel_collected + r_full_hopper + r_empty_alliance + r_neutral_idle + r_bump_idle + r_milestone + r_jerk

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
        vx_n = float(np.clip(self._robot.vx / MAX_SPEED_MPS,              -1.0, 1.0))
        vy_n = float(np.clip(self._robot.vy / MAX_SPEED_MPS,              -1.0, 1.0))
        rx_n = float(np.clip(self._robot.x  / FIELD_LENGTH,                0.0, 1.0))
        ry_n = float(np.clip(self._robot.y  / FIELD_WIDTH,                 0.0, 1.0))
        hopper_norm      = float(np.clip(self._hopper            / HOPPER_CAPACITY,        0.0, 1.0))
        contributed_norm = float(np.clip(self._contributed_score / CONTRIBUTED_SCORE_NORM, 0.0, 1.0))
        total_norm       = float(np.clip(self._total_score       / TOTAL_SCORE_NORM,       0.0, 1.0))
        # hub_dist_norm is -1.0 in the neutral zone — a clear sentinel meaning
        # "hub proximity is irrelevant here." Values [0, 1] are only meaningful
        # in and around the alliance zone, where 0 = at hub, 1 = 8 m away.
        if self._robot.x > NEUTRAL_MIN_X:
            hub_dist_norm = -1.0
        else:
            dist_to_hub   = math.hypot(self._robot.x - BLUE_HUB_CENTER[0],
                                       self._robot.y - BLUE_HUB_CENTER[1])
            hub_dist_norm = float(np.clip(dist_to_hub / HUB_DIST_NORM_MAX, 0.0, 1.0))

        # Countdown clock: 1.0 = match start, 0.0 = time expired
        time_remaining_norm = 1.0 - float(self._step_count) / float(self._max_episode_steps)

        return np.array([vx_n, vy_n, rx_n, ry_n, hopper_norm,
                         contributed_norm, total_norm, hub_dist_norm,
                         time_remaining_norm], dtype=np.float32)

    def match_time_str(self) -> str:
        """Remaining match time as 'M:SS' for HUD display."""
        remaining_s = (self._max_episode_steps - self._step_count) * MATCH_STEP_DT
        return f"{int(remaining_s // 60)}:{int(remaining_s % 60):02d}"

    def _get_module_states(self):
        return self._robot.module_states
