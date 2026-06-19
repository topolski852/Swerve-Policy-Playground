# ──────────────────────────────────────────────────────────────────────────────
# lib/field_constants.py
# Shared field geometry, robot parameters, drive limits, and renderer settings.
# Imported by both the path_following and fuel_scoring experiment packages.
# ──────────────────────────────────────────────────────────────────────────────

import math

# ── Robot geometry ─────────────────────────────────────────────────────────────

ROBOT_BUMPER_HALF = 27 / 2 * 0.0254      # 0.34290 m — outer bumper half-width
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
MAX_ANGULAR_RPS     = 3.5   # maximum angular speed (rad/s)

VELOCITY_ALPHA      = 0.35  # first-order velocity lag (0 < alpha <= 1)
SPEED_DEADBAND      = 0.05  # m/s — module angle held at last value below this

# ── Physics ────────────────────────────────────────────────────────────────────

DT = 0.02   # simulation timestep (seconds) — matches WPILib 20 ms loop

# ── Field dimensions ───────────────────────────────────────────────────────────

FIELD_LENGTH =  16.54   # meters, x-axis (Blue DS → Red DS)
FIELD_WIDTH  =   8.21   # meters, y-axis (right wall → left wall from Blue DS)

# ── Field obstacles ────────────────────────────────────────────────────────────
# Impassable zones as axis-aligned rectangles: (x_min, y_min, x_max, y_max) meters.
IMPASSABLE_RECTS = [
    # ── Hubs ──────────────────────────────────────────────────────────────────
    ( 4.029, 3.438,  5.223, 4.632),   # Blue hub
    (11.317, 3.438, 12.511, 4.632),   # Red hub

    # ── Trenches ──────────────────────────────────────────────────────────────
    ( 4.029, 6.556,  5.223, 8.210),   # Blue trench, top wall
    ( 4.029, 0.000,  5.223, 1.654),   # Blue trench, bottom wall
    (11.317, 6.556, 12.511, 8.210),   # Red trench, top wall
    (11.317, 0.000, 12.511, 1.654),   # Red trench, bottom wall
]

# ── Hub scoring targets ────────────────────────────────────────────────────────
BLUE_HUB_CENTER = (4.626, 4.035)
RED_HUB_CENTER  = (11.914, 4.035)

# ── Zone X boundaries ─────────────────────────────────────────────────────────
BLUE_ALLIANCE_MAX_X = 4.029    # rx < this → Blue Alliance Zone
NEUTRAL_MIN_X       = 5.223    # rx > this (and < NEUTRAL_MAX_X) → Neutral Zone
NEUTRAL_MAX_X       = 11.317
RED_ALLIANCE_MIN_X  = 12.511   # rx > this → Red Alliance Zone

# ── Environment shared parameters ─────────────────────────────────────────────

WAYPOINT_PASS_RADIUS = 0.65   # meters — advance to next waypoint when within this
OFF_PATH_LIMIT       = 2.5    # meters from path before episode terminates

# ── Renderer ───────────────────────────────────────────────────────────────────

FIELD_IMAGE          = "assets/field_2026.png"

FIELD_CORNER_BL      = (522, 1487)   # pixel for field (0.0 m, 0.0 m)
FIELD_CORNER_TR      = (3378,  92)   # pixel for field (FIELD_LENGTH, FIELD_WIDTH)

RENDER_SCALE         = 60.0    # pixels per meter
WINDOW_PADDING       = 40      # pixels of border around the field

ROBOT_COLOR          = (220, 220, 220)
ROBOT_BORDER_COLOR   = ( 50,  50,  50)
MODULE_COLOR         = ( 80,  80,  80)
PATH_COLOR           = ( 60, 130, 200)
WAYPOINT_COLOR       = (255, 200,  50)
ROBOT_HEADING_COLOR  = (255,  80,  80)

ARROW_MAX_PIXELS     = 38
ARROW_COLOR          = ( 80, 220, 120)
ARROW_WIDTH          = 3
ARROW_HEAD_SIZE      = 8

MODULE_RING_SLOW     = ( 60,  60,  60)
MODULE_RING_FAST     = (120, 220, 120)

WINDOW_TITLE         = "Swerve Policy Playground — Team 1507"
TARGET_FPS           = 60
