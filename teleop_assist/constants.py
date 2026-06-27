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
# Phase 1 (current): explicit match against fromFieldRelativeSpeeds target.
#   r_match  = RW_MATCH * exp(-RW_MATCH_K * MSE(action, target))
#   Peaks at RW_MATCH for perfect match, decays toward 0 for large deviation.
#   Handles both "move" and "stop" cases uniformly — no separate stillness term.
#
# Phase 2 (future): add proximity rays + RW_APPROACH for obstacle avoidance.

RW_MATCH        =  2.0    # peak reward per step for perfect joystick match
RW_MATCH_K      =  8.0    # exponential decay; MSE of 0.25 → ~27% of peak reward
RW_SMOOTH       = -0.25   # jerk penalty
RW_COLLISION    = -50.0   # terminal penalty
