# ──────────────────────────────────────────────────────────────────────────────
# teleop_assist/constants.py
# Experiment-specific parameters for the teleop-assist policy trainer.
# Shared field/robot constants live in lib/field_constants.py.
# ──────────────────────────────────────────────────────────────────────────────

# ── Episode parameters ─────────────────────────────────────────────────────────

MAX_EPISODE_STEPS     = 1500   # 30 s at 20 ms — matches PRT timeout

# ── Joystick simulation ────────────────────────────────────────────────────────

DRIFT_MAX             = 0.07   # max per-axis drift; vector magnitude ≤ 0.07*√2 ≈ 0.10
DRIFT_FLOOR           = 0.12   # true joy magnitude below this = "stop" intent;
                                # set above max-drift magnitude so drift alone never
                                # exceeds the floor and gets mistaken for real intent

JOY_TARGET_REROLL_MIN = 40     # min steps before picking a new random intent target
JOY_TARGET_REROLL_MAX = 80     # max steps before picking a new random intent target
JOY_TARGET_ARRIVE_R   = 0.80   # meters — reroll target when robot gets this close
JOY_SPEED_MIN         = 0.30   # minimum speed fraction (relative to MAX_SPEED_MPS)
JOY_SPEED_MAX         = 1.00   # maximum speed fraction — driver can push full stick
STOP_INTENT_PROB      = 0.35   # fraction of reroll periods where true intent is "hold still"

# ── Reward weights ─────────────────────────────────────────────────────────────
#
# Phase 1 (current): joystick following only.
# Phase 2 (future):  add proximity rays + RW_APPROACH back for obstacle avoidance.

RW_INTENT           =  1.2    # reward per step for matching joystick direction × magnitude
RW_SMOOTH           = -0.25   # jerk penalty
RW_COLLISION        = -50.0   # terminal penalty
RW_STILL_WHEN_DRIFT = -8.0    # penalty per unit speed when true intent is zero
