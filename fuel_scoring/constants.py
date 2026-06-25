# ──────────────────────────────────────────────────────────────────────────────
# fuel_scoring/constants.py
# Parameters for the fuel scoring environment.
# No path navigation — pure zone-based fuel mechanics.
# ──────────────────────────────────────────────────────────────────────────────

# ── Episode parameters ─────────────────────────────────────────────────────────

MATCH_STEP_DT         = 0.02          # seconds per simulation step (50 Hz)
AUTO_PERIOD_STEPS     = 1000          # first 20 s = autonomous period
PHASE1_EPISODE_STEPS  = 2000          # ~40 s — short episodes during random-start exploration
MAX_EPISODE_STEPS     = 8000          # 160 s = full 2:40 FRC 2026 match at 20 ms/step
SCORING_START_HOPPER  = 8.0           # FRC 2026: robots start with 8 fuel loaded

# ── Hopper / fuel ─────────────────────────────────────────────────────────────

HOPPER_CAPACITY      = 60.0    # maximum fuel units stored
FUEL_FILL_RATE_MAX   = 1.0     # fuel/step at MAX_SPEED_MPS (log-scaled — see swerve_env)
FUEL_FILL_MIN_SPEED  = 0.5     # m/s — zero collection below this threshold
FUEL_SHOOT_RATE      = 0.4     # fuel units scored per step while in alliance zone (any speed)
SCORING_OPTIMAL_DEPTH = 1.5   # metres inside alliance zone for peak shot accuracy (x ≈ 2.53 m)
SCORING_DEPTH_SIGMA   = 1.0   # sweet-spot spread — ±1 sigma gives ~61% efficiency

# ── Score observation normalization ───────────────────────────────────────────

CONTRIBUTED_SCORE_NORM = 1500.0  # ceiling for one robot per 8000-step match (~25 cycles × 60 fuel)
TOTAL_SCORE_NORM       = 2000.0  # ceiling for full alliance (reserved for multi-robot)

# ── Reward weights ─────────────────────────────────────────────────────────────

RW_FUEL_SCORED              =  5.0   # per fuel unit scored (after shot multiplier)
RW_FUEL_COLLECTED           =  0.3   # per fuel unit collected in neutral zone
RW_FULL_HOPPER_IN_NEUTRAL   = -0.5  # per step: hopper full but still in neutral zone
RW_EMPTY_HOPPER_IN_ALLIANCE = -0.5  # per step: hopper empty but still in alliance zone
RW_NEUTRAL_IDLE             = -0.3  # per step: in neutral zone, below collection speed, hopper not full
RW_MILESTONE_100            =  50.0  # one-time bonus when contributed_score reaches 100  (~cycle 2)
RW_MILESTONE_360            = 150.0  # one-time bonus when contributed_score reaches 360  (~cycle 6)
RW_MILESTONE_600            = 300.0  # one-time bonus when contributed_score reaches 600  (~cycle 10)
RW_MILESTONE_900            = 500.0  # one-time bonus when contributed_score reaches 900  (~cycle 20, ~67% through 8k match)
RW_JERK                     = -0.1  # per unit of mean |Δaction| (vx/vy normalized 0-2 range)
RW_BUMP_IDLE                = -0.5  # per step: stationary on the bump (4.029–5.223 m) — robot should transit, not camp
BUMP_IDLE_MAX_SPEED         =  0.1  # m/s — below this on the bump triggers the penalty (transit at any real speed is free)
OBSTACLE_DANGER_MARGIN      =  0.15  # metres beyond robot bumper (~6 in) — keeps corridor center penalty-free
RW_OBSTACLE_PROXIMITY       = -4.0  # per-step at collision boundary; 0 at outer danger edge
RW_COLLISION_PENALTY        = -50.0  # starting value — ramped to -200 at Phase 3 via CollisionRampCallback
