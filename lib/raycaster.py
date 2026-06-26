# ──────────────────────────────────────────────────────────────────────────────
# lib/raycaster.py
# 8-direction AABB raycasting for obstacle proximity.
#
# Rays originate from the robot center in robot frame (x=forward, y=left).
# All obstacles — field walls and impassable rects — are pre-expanded by
# ROBOT_BUMPER_HALF so that a returned distance of 0.0 means the robot edge
# is exactly at the collision boundary (matching _check_collision() in the envs).
# ──────────────────────────────────────────────────────────────────────────────

import math
import numpy as np
from lib.field_constants import (
    IMPASSABLE_RECTS, FIELD_LENGTH, FIELD_WIDTH, ROBOT_BUMPER_HALF,
)
from teleop_assist.constants import RAY_MAX_DISTANCE

# ── Ray directions (robot frame, x=forward, y=left, 0°=forward, CCW) ──────────

_ANGLES_DEG = [0, 45, 90, 135, 180, 225, 270, 315]
RAY_DIRS = np.array(
    [[math.cos(math.radians(a)), math.sin(math.radians(a))] for a in _ANGLES_DEG],
    dtype=np.float32,
)  # shape (8, 2)

# ── Pre-expanded obstacle rectangles ──────────────────────────────────────────
# Expand every obstacle by ROBOT_BUMPER_HALF on all sides so the distance a
# ray reports equals the distance the *robot edge* is from that surface.

_r   = ROBOT_BUMPER_HALF
_BIG = 200.0   # half-extent for wall slabs (large enough to always cover the field)

_EXPANDED_RECTS = [
    # Field walls — each is a thin slab on one side of the field
    (-_BIG,             -_BIG,          _r,                  _BIG),           # left  wall (x < r)
    (FIELD_LENGTH - _r,  -_BIG,          _BIG,                _BIG),          # right wall
    (-_BIG,             -_BIG,          _BIG,                 _r),            # bottom wall (y < r)
    (-_BIG,  FIELD_WIDTH - _r,          _BIG,                 _BIG),          # top   wall
]
for (ox1, oy1, ox2, oy2) in IMPASSABLE_RECTS:
    _EXPANDED_RECTS.append((ox1 - _r, oy1 - _r, ox2 + _r, oy2 + _r))


def _ray_aabb_dist(ox: float, oy: float,
                   dx: float, dy: float,
                   x1: float, y1: float, x2: float, y2: float) -> float:
    """
    Distance from ray origin (ox, oy) / direction (dx, dy) to AABB [x1,x2]×[y1,y2].
    Returns RAY_MAX_DISTANCE when there is no forward intersection.
    Assumes the robot center is OUTSIDE the AABB (post-collision guard).
    """
    INF = 1e9

    if abs(dx) > 1e-9:
        tx1 = (x1 - ox) / dx
        tx2 = (x2 - ox) / dx
    else:
        # Ray is parallel to x-axis — inside x-slab iff x1 <= ox <= x2
        if x1 <= ox <= x2:
            tx1, tx2 = -INF, INF
        else:
            return RAY_MAX_DISTANCE

    if abs(dy) > 1e-9:
        ty1 = (y1 - oy) / dy
        ty2 = (y2 - oy) / dy
    else:
        if y1 <= oy <= y2:
            ty1, ty2 = -INF, INF
        else:
            return RAY_MAX_DISTANCE

    tmin = max(min(tx1, tx2), min(ty1, ty2))
    tmax = min(max(tx1, tx2), max(ty1, ty2))

    # No intersection, or intersection is entirely behind the ray
    if tmax < 1e-6 or tmin > tmax:
        return RAY_MAX_DISTANCE

    t = tmin if tmin >= 0.0 else tmax
    if t < 0.0:
        return RAY_MAX_DISTANCE

    return min(float(t), RAY_MAX_DISTANCE)


def cast_rays(rx: float, ry: float, heading: float) -> np.ndarray:
    """
    Cast 8 rays from (rx, ry) with the robot's heading.

    Returns ndarray shape (8,), distances in metres to the nearest surface in
    each of the 8 directions.  Index mapping:
        0=forward  1=front-left  2=left   3=back-left
        4=back     5=back-right  6=right  7=front-right

    distance == 0.0  → robot edge touching that surface (collision boundary)
    distance == RAY_MAX_DISTANCE → nothing within range
    """
    cos_h = math.cos(heading)
    sin_h = math.sin(heading)
    distances = np.full(8, RAY_MAX_DISTANCE, dtype=np.float32)

    for i, (rdx, rdy) in enumerate(RAY_DIRS):
        # Robot frame → world frame rotation
        wdx = rdx * cos_h - rdy * sin_h
        wdy = rdx * sin_h + rdy * cos_h

        best = RAY_MAX_DISTANCE
        for (x1, y1, x2, y2) in _EXPANDED_RECTS:
            d = _ray_aabb_dist(rx, ry, wdx, wdy, x1, y1, x2, y2)
            if d < best:
                best = d

        distances[i] = best

    return distances
