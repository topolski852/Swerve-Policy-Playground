# ──────────────────────────────────────────────────────────────────────────────
# path_randomizer/constants.py
# Parameters for the randomized waypoint navigation experiment.
# No field-specific mechanics — pure point-to-point locomotion.
# ──────────────────────────────────────────────────────────────────────────────

# ── Episode parameters ─────────────────────────────────────────────────────────

N_WAYPOINTS_MIN      = 3     # fewest waypoints per episode
N_WAYPOINTS_MAX      = 6     # most waypoints per episode
MAX_EPISODE_STEPS    = 3000  # ~60 s — generous budget for 3-6 waypoints
MAX_WAYPOINT_DISTANCE = 5.0  # metres — max distance between consecutive waypoints

# ── Reward weights ─────────────────────────────────────────────────────────────

# Progress: reward getting closer, 0 for moving away (never negative).
# Detours around obstacles cost time but not extra penalty.
RW_PROGRESS          =  1.5    # per metre closed toward current waypoint per step
RW_WAYPOINT_BONUS    = 20.0   # one-time bonus each time a waypoint is reached
RW_GOAL_BONUS        = 75.0   # bonus for completing all waypoints in the episode
RW_TIME_PENALTY      = -0.01  # per step — keeps the robot moving
RW_COLLISION_PENALTY = -25.0  # one-time on wall or obstacle contact
