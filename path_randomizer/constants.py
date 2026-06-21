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

# Milestone rings: one-time bonuses at 75%/50%/25% of MAX_WAYPOINT_DISTANCE.
# Each ring can only be earned once per waypoint — can't be re-triggered by
# backing off and re-approaching. Values increase toward the target so the agent
# is always incentivised to push to the next ring rather than hover.
MILESTONE_FRACTIONS  = [0.75, 0.50, 0.25]   # fraction of MAX_WAYPOINT_DISTANCE
MILESTONE_BONUSES    = [3.0,  5.0,  10.0]   # one-time reward at each ring

RW_WAYPOINT_BONUS    = 20.0   # one-time bonus on arrival (< WAYPOINT_PASS_RADIUS)
RW_GOAL_BONUS        = 75.0   # bonus for completing all waypoints in the episode
RW_TIME_PENALTY      = -0.01  # per step — keeps the robot moving
RW_COLLISION_PENALTY = -25.0  # one-time on wall or obstacle contact
