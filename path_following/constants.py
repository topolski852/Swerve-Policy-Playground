# ──────────────────────────────────────────────────────────────────────────────
# path_following/constants.py
# Experiment-specific parameters for the path-following training system.
# Shared field/robot constants live in lib/field_constants.py.
# ──────────────────────────────────────────────────────────────────────────────

# ── Episode parameters ─────────────────────────────────────────────────────────

MAX_EPISODE_STEPS = 1500   # timeout in steps (~30 s at 20 ms)

# ── Reward weights ─────────────────────────────────────────────────────────────

RW_PROGRESS          =  1.0
RW_VEL_ALIGN         =  0.8
RW_CROSS_TRACK       = -0.6
RW_SMOOTH_VEL        = -0.30
RW_SPEED_MAGNITUDE   = -0.02
RW_TIME_PENALTY      = -0.02
RW_WAYPOINT_BONUS    =  6.0
RW_GOAL_BONUS        = 80.0
RW_OFF_PATH_PENALTY  = -10.0
RW_COLLISION_PENALTY = -25.0
