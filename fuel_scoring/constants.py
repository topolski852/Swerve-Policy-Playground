# ──────────────────────────────────────────────────────────────────────────────
# fuel_scoring/constants.py
# Parameters for the fuel scoring environment.
# No path navigation — pure zone-based fuel mechanics.
# ──────────────────────────────────────────────────────────────────────────────

# ── Episode parameters ─────────────────────────────────────────────────────────

MAX_EPISODE_STEPS     = 2000           # ~40 s at 20 ms per step
SCORING_START_HOPPER  = 8.0            # FRC 2026: robots start with 8 fuel loaded

# ── Hopper / fuel ─────────────────────────────────────────────────────────────

HOPPER_CAPACITY      = 60.0    # maximum fuel units stored
FUEL_FILL_RATE_MAX   = 1.5     # fuel/step at MAX_SPEED_MPS (log-scaled — see swerve_env)
FUEL_FILL_MIN_SPEED  = 0.5     # m/s — zero collection below this threshold
FUEL_SHOOT_RATE      = 0.4     # fuel units scored per step while in alliance zone (any speed)

# ── Score observation normalization ───────────────────────────────────────────

CONTRIBUTED_SCORE_NORM = 300.0   # realistic ceiling for one robot per episode
TOTAL_SCORE_NORM       = 1000.0  # ceiling for full alliance (reserved for multi-robot)

# ── Reward weights ─────────────────────────────────────────────────────────────

RW_FUEL_SCORED              =  5.0   # per fuel unit scored (dominant signal)
RW_FUEL_COLLECTED           =  0.3   # per fuel unit collected in neutral zone
RW_FULL_HOPPER_IN_NEUTRAL   = -0.5  # per step: hopper full but still in neutral zone
RW_EMPTY_HOPPER_IN_ALLIANCE = -0.5  # per step: hopper empty but still in alliance zone
RW_COLLISION_PENALTY        = -25.0 # one-time hit on wall/obstacle collision
