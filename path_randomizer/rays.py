# ──────────────────────────────────────────────────────────────────────────────
# path_randomizer/rays.py
# Distance rays: how far the robot can drive in each direction before it hits
# a field element or wall. They let the policy SEE obstacles instead of
# memorizing where they are on one year's field.
#
# Works on any polygon (2025 hexagons, 2026 rectangles, ...): every field
# element is a list of corners, the same shape 1507Base keeps in
# Nodes.FieldElements.CORNERS. The robot code casts the same rays:
#
#   for each direction k (angle = 360° * k / N_RAYS, field-relative, 0° = +x):
#       d = distance from robot center along the ray to the nearest
#           polygon edge or field wall (RAY_RANGE if none closer)
#       ray_k = clamp(d - ROBOT_BUMPER_HALF, 0, RAY_RANGE) / RAY_RANGE
#
# 0.0 = bumper touching (or nearly), 1.0 = nothing within RAY_RANGE.
# ──────────────────────────────────────────────────────────────────────────────

import math
import numpy as np

from lib.field_constants import FIELD_LENGTH, FIELD_WIDTH, ROBOT_BUMPER_HALF, IMPASSABLE_RECTS
from path_randomizer.constants import N_RAYS, RAY_RANGE


def rect_corners(x1, y1, x2, y2):
    return [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]


# Field elements as polygons (corner lists). 2026: all rectangles.
FIELD_POLYGONS = [rect_corners(*r) for r in IMPASSABLE_RECTS]
FIELD_WALLS    = rect_corners(0.0, 0.0, FIELD_LENGTH, FIELD_WIDTH)


def _segments(polygons):
    segs = []
    for poly in polygons:
        for i in range(len(poly)):
            (ax, ay), (bx, by) = poly[i], poly[(i + 1) % len(poly)]
            segs.append((ax, ay, bx, by))
    return np.array(segs, dtype=np.float64)


_SEGS   = _segments(FIELD_POLYGONS + [FIELD_WALLS])
_ANGLES = 2.0 * math.pi * np.arange(N_RAYS) / N_RAYS
RAY_DIRS = np.stack([np.cos(_ANGLES), np.sin(_ANGLES)], axis=1)   # (N_RAYS, 2)


def cast_rays(rx, ry):
    """Returns N_RAYS values in [0, 1] (see the header for the formula)."""
    ax, ay, bx, by = _SEGS.T
    ex, ey = bx - ax, by - ay                 # segment direction
    wx, wy = ax - rx, ay - ry                 # robot -> segment start
    out = np.empty(N_RAYS, dtype=np.float32)
    for k, (dx, dy) in enumerate(RAY_DIRS):
        # Solve robot + t*d = a + u*e  for t >= 0 and 0 <= u <= 1.
        denom = dx * ey - dy * ex
        with np.errstate(divide="ignore", invalid="ignore"):
            t = (wx * ey - wy * ex) / denom
            u = (wx * dy - wy * dx) / denom
        hit = (np.abs(denom) > 1e-12) & (t >= 0.0) & (u >= 0.0) & (u <= 1.0)
        d = float(t[hit].min()) if hit.any() else RAY_RANGE + ROBOT_BUMPER_HALF
        out[k] = min(max(d - ROBOT_BUMPER_HALF, 0.0), RAY_RANGE) / RAY_RANGE
    return out
