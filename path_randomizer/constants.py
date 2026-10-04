# ──────────────────────────────────────────────────────────────────────────────
# path_randomizer/constants.py
# Parameters for the randomized waypoint navigation experiment.
# No field-specific mechanics — pure point-to-point locomotion.
# ──────────────────────────────────────────────────────────────────────────────

# ── Robot: match 1507Base (SystemCore branch) ──────────────────────────────────
# The trained policy drives 1507Base's robot, so the sim robot uses its numbers.
# They live here (not lib/field_constants.py) so the other experiments keep theirs.

ROBOT_MAX_SPEED   = 5.04   # m/s — TunerConstants.kSpeedAt12Volts (SwerveConfig.MAX_SPEED)
SLIP_ACCEL_MPS2   = 15.0   # m/s² — SwerveConfig.SLIP_ACCEL_MPS2 (any change in velocity)
TORQUE_ACCEL_MPS2 = 10.0   # m/s² — SwerveConfig.TORQUE_ACCEL_MPS2 (from standstill; 0 at top speed)
PASS_RADIUS       = 0.40   # m — kAuto.Accuracy.NORMAL, the default when a node gives no accuracy

# ── Episode parameters ─────────────────────────────────────────────────────────

MAX_EPISODE_STEPS     = 3000  # ~60 s — generous budget for 3-12 waypoints

N_WAYPOINTS_MIN       = 3     # fewest waypoints per episode
N_WAYPOINTS_MAX       = 12    # most waypoints per episode
MAX_WAYPOINT_DISTANCE = 6.0   # metres — practical FRC ceiling; > 6 m flagged as unexpected in app
MIN_WAYPOINT_DISTANCE = 1.0   # metres — floor so the robot must physically move between points
                               # (must exceed PASS_RADIUS)

# ── Reward weights ─────────────────────────────────────────────────────────────

# Approach reward: earned per metre of new closest-approach progress toward the
# current waypoint. Monotone — only fires when the robot sets a new personal best
# distance, so back-and-forth oscillation cannot farm infinite reward.
RW_APPROACH          =  2.0   # per metre of new closest-approach progress

RW_WAYPOINT_BONUS    = 100.0  # one-time bonus on arrival (< PASS_RADIUS)
RW_GOAL_BONUS        = 75.0   # bonus for completing all waypoints in the episode
RW_TIME_PENALTY      = -0.03  # per step — creates urgency to commit and advance
RW_COLLISION_PENALTY     = -75.0  # one-time on wall or obstacle contact
OBSTACLE_DANGER_MARGIN   =  0.15  # metres of warning zone beyond robot bumper (~6 in)
RW_OBSTACLE_PROXIMITY    = -0.5   # per-step at collision boundary; 0 at outer danger edge
