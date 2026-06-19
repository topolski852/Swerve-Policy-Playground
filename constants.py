# ──────────────────────────────────────────────────────────────────────────────
# constants.py
# All tunable parameters in one place. Edit here, nowhere else.
# ──────────────────────────────────────────────────────────────────────────────

import math

# ── Robot geometry ─────────────────────────────────────────────────────────────

# Physical dimensions from Nodes.java (1507 robot, 2026 season).
ROBOT_BUMPER_HALF = 27 / 2 * 0.0254      # 0.34290 m — outer bumper half-width, used for chassis rendering
MODULE_OFFSET     = 10.7375 * 0.0254     # 0.27273 m — module position from robot center

# Module offsets from robot center (meters), WPILib order: FL, FR, BL, BR.
# Each entry is (x_offset, y_offset) in robot frame (x=forward, y=left).
MODULE_OFFSETS = [
    ( MODULE_OFFSET,  MODULE_OFFSET),   # FL
    ( MODULE_OFFSET, -MODULE_OFFSET),   # FR
    (-MODULE_OFFSET,  MODULE_OFFSET),   # BL
    (-MODULE_OFFSET, -MODULE_OFFSET),   # BR
]

# ── Drive limits ───────────────────────────────────────────────────────────────

MAX_SPEED_MPS       = 4.5   # maximum translational speed (m/s)
MAX_ANGULAR_RPS     = 3.5   # maximum angular speed (rad/s)  — unused in Phase 1

# First-order velocity lag coefficient (0 < alpha <= 1).
# actual = alpha * commanded + (1-alpha) * previous
# Lower = more lag (sluggish), higher = snappier.
VELOCITY_ALPHA      = 0.35

# Module angle is held at last value when speed is below this threshold (anti-jitter).
SPEED_DEADBAND      = 0.05  # m/s

# ── Physics ────────────────────────────────────────────────────────────────────

DT = 0.02   # simulation timestep (seconds) — matches WPILib 20 ms loop

# ── Field dimensions ───────────────────────────────────────────────────────────

# FRC 2026 Rebuilt field. LENGTH = long axis (X), WIDTH = short axis (Y).
# "Height" is avoided — the field is 2D and height would imply a Z dimension.
FIELD_LENGTH =  16.54   # meters, x-axis (Blue DS → Red DS)
FIELD_WIDTH  =   8.21   # meters, y-axis (right wall → left wall from Blue DS)

# ── Environment / training ─────────────────────────────────────────────────────

MAX_EPISODE_STEPS   = 1500          # timeout in steps (~30 s at 20 ms)
WAYPOINT_PASS_RADIUS = 0.65         # meters — advance to next waypoint when within this
OFF_PATH_LIMIT       = 2.5          # meters from nearest path point → episode failure

# ── Reward weights ─────────────────────────────────────────────────────────────
# This is the block you'll tune most often. All coefficients are floats.

RW_PROGRESS          =  1.0    # reward per meter of arc-length progress per step
                               # (allows negative — penalizes backward movement)
RW_VEL_ALIGN         =  0.8    # reward for velocity toward the current target waypoint [-1,1]
                               # uses waypoint direction, not segment — prevents wp1-loop exploit
RW_CROSS_TRACK       = -0.6    # penalty per meter of signed cross-track error (absolute)
RW_SMOOTH_VEL        = -0.30   # penalty on L2 norm of (action - prev_action)
RW_SPEED_MAGNITUDE   = -0.02   # small penalty on |action| to discourage max-speed defaults
RW_TIME_PENALTY      = -0.02   # per-step cost — creates urgency, prevents infinite looping
RW_WAYPOINT_BONUS    =  2.0    # bonus each time a waypoint is reached
RW_GOAL_BONUS        = 20.0    # bonus on successful path completion
RW_OFF_PATH_PENALTY  = -10.0   # one-time penalty on off-path termination

# ── Renderer ───────────────────────────────────────────────────────────────────

FIELD_IMAGE          = "assets/field_2026.png"

# Source image pixel coordinates of the two field boundary corners,
# measured with Paint on assets/field_2026.png (3902x1584 source).
# FIELD_CORNER_BL: pixel for field (0, 0)                    — Blue DS, bottom-left
# FIELD_CORNER_TR: pixel for field (FIELD_LENGTH, FIELD_WIDTH) — Red DS, top-right
# The full image is displayed and these two points anchor the coordinate transform,
# so driver stations and surrounding carpet remain visible.
# Adjust if the path overlay doesn't land on the right structures.
FIELD_CORNER_BL      = (522, 1487)   # field (0.0 m, 0.0 m)
FIELD_CORNER_TR      = (3378,  92)   # field (16.54 m, 8.21 m)

RENDER_SCALE         = 60.0    # pixels per meter
WINDOW_PADDING       = 40      # pixels of border around the field

ROBOT_COLOR          = (220, 220, 220)   # chassis fill
ROBOT_BORDER_COLOR   = ( 50,  50,  50)  # chassis outline
MODULE_COLOR         = ( 80,  80,  80)  # module housing fill
PATH_COLOR           = ( 60, 130, 200)  # path line
WAYPOINT_COLOR       = (255, 200,  50)  # waypoint markers
ROBOT_HEADING_COLOR  = (255,  80,  80)  # heading indicator on chassis

# Module state arrows (AdvantageScope style)
ARROW_MAX_PIXELS     = 38      # arrow length at MAX_SPEED_MPS
ARROW_COLOR          = ( 80, 220, 120)  # arrow shaft/head
ARROW_WIDTH          = 3       # shaft thickness in pixels
ARROW_HEAD_SIZE      = 8       # triangle side length in pixels

# Speed indicator ring on each module housing
MODULE_RING_SLOW     = ( 60,  60,  60)
MODULE_RING_FAST     = (120, 220, 120)

WINDOW_TITLE         = "Swerve Policy Playground — Team 1507"
TARGET_FPS           = 60
