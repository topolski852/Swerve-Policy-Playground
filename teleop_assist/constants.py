# ──────────────────────────────────────────────────────────────────────────────
# teleop_assist/constants.py
# Experiment-specific parameters for the teleop-assist policy trainer.
# Shared field/robot constants live in lib/field_constants.py.
# ──────────────────────────────────────────────────────────────────────────────

# ── Episode parameters ─────────────────────────────────────────────────────────

MAX_EPISODE_STEPS     = 1500   # 30 s at 20 ms — matches PRT timeout

# ── Joystick simulation ────────────────────────────────────────────────────────

DRIFT_MAX             = 0.12   # max per-axis drift magnitude injected per episode
DRIFT_FLOOR           = 0.10   # true joy magnitude below this = "stop" intent;
                                # policy must learn to ignore inputs at this scale

JOY_TARGET_REROLL_MIN = 40     # min steps before picking a new random intent target
JOY_TARGET_REROLL_MAX = 80     # max steps before picking a new random intent target
JOY_TARGET_ARRIVE_R   = 0.80   # meters — reroll target when robot gets this close
JOY_SPEED_MIN         = 0.30   # minimum speed fraction (relative to MAX_SPEED_MPS)
JOY_SPEED_MAX         = 1.00   # maximum speed fraction — driver can push full stick

# ── Raycaster ──────────────────────────────────────────────────────────────────

RAY_MAX_DISTANCE      = 2.5    # meters — rays beyond this read as fully clear
DANGER_ZONE_NORM      = 0.80   # danger starts at 80% of RAY_MAX_DISTANCE (2.0 m);
                                # approach penalty scales from 0 at edge → full at contact

# ── Reward weights ─────────────────────────────────────────────────────────────
#
# Design intent:
#   - In open field:  intent reward dominates → policy faithfully follows joystick
#   - Approaching obstacle at full speed:
#       approach penalty overwhelms intent reward → policy slows down
#   - Drift-magnitude joystick with no real intent:
#       still_when_drift penalty → policy stays still
#   - Collision: large terminal penalty → policy avoids contact hard

RW_INTENT           =  1.2    # reward per step for matching joystick direction × magnitude
RW_APPROACH         = -4.0    # penalty scaling for moving toward a nearby obstacle
RW_SMOOTH           = -0.25   # jerk penalty — borrowed from PRT RW_SMOOTH_VEL
RW_COLLISION        = -50.0   # terminal penalty — larger than PRT -25 (safety-critical)
RW_STILL_WHEN_DRIFT = -0.8    # penalty per unit speed when true intent is zero
