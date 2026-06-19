# ──────────────────────────────────────────────────────────────────────────────
# fuel_scoring/constants.py
# Parameters specific to the fuel scoring training environment.
# Shared field/robot constants live in lib/field_constants.py.
# ──────────────────────────────────────────────────────────────────────────────

# ── Episode parameters ─────────────────────────────────────────────────────────

MAX_EPISODE_STEPS     = 2000           # longer episodes — 3 full collect-score cycles
SCORING_START_HOPPER  = 8.0            # FRC 2026: robots start with 8 fuel loaded

# ── Hopper / fuel ─────────────────────────────────────────────────────────────

HOPPER_CAPACITY      = 60.0    # maximum fuel units stored
FUEL_FILL_RATE       = 0.5     # fuel units added per step while in neutral zone
FUEL_FILL_MIN_SPEED  = 0.5     # m/s — must be moving to collect
FUEL_SHOOT_RATE      = 0.4     # fuel units scored per step while in alliance zone

# ── Reward weights ─────────────────────────────────────────────────────────────

# Fuel mechanics — primary signals
RW_FUEL_SCORED             =  5.0    # per fuel unit deposited in hub (dominant reward)
RW_FUEL_COLLECTED          =  0.3    # per fuel unit picked up in neutral zone
RW_FULL_HOPPER_IN_NEUTRAL  = -0.5   # per step: hopper full but still in neutral zone

# Navigation — secondary signals to guide the shuttle
RW_PROGRESS          =  0.5
RW_VEL_ALIGN         =  0.3
RW_CROSS_TRACK       = -0.4
RW_SMOOTH_VEL        = -0.15
RW_SPEED_MAGNITUDE   = -0.01
RW_TIME_PENALTY      =  0.0    # no urgency — let the robot take its time
RW_WAYPOINT_BONUS    =  1.5
RW_GOAL_BONUS        = 10.0

# Termination penalties
RW_OFF_PATH_PENALTY  = -10.0
RW_COLLISION_PENALTY = -25.0
