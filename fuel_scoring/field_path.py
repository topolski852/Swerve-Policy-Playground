# ──────────────────────────────────────────────────────────────────────────────
# fuel_scoring/field_path.py
# Simple alliance ↔ neutral shuttle path for fuel scoring training.
#
# The robot shuttles horizontally at Y=2.55, the center of the BumpRight
# corridor (gap between hub bottom at Y=3.438 and trench top at Y=1.654).
# Three collect-score cycles per episode.
# ──────────────────────────────────────────────────────────────────────────────

import math
from lib.field_constants import WAYPOINT_PASS_RADIUS


# ── Waypoints ─────────────────────────────────────────────────────────────────

ALLIANCE_X  = 3.50
NEUTRAL_X   = 8.27
CORRIDOR_Y  = 2.55   # center of BumpRight gap

WAYPOINTS = [
    (ALLIANCE_X, CORRIDOR_Y),   # 0  start — blue alliance zone (score initial fuel)
    (NEUTRAL_X,  CORRIDOR_Y),   # 1  neutral zone — collect fuel
    (ALLIANCE_X, CORRIDOR_Y),   # 2  blue alliance zone — score
    (NEUTRAL_X,  CORRIDOR_Y),   # 3  neutral zone — collect
    (ALLIANCE_X, CORRIDOR_Y),   # 4  blue alliance zone — score
    (NEUTRAL_X,  CORRIDOR_Y),   # 5  neutral zone — collect
    (ALLIANCE_X, CORRIDOR_Y),   # 6  blue alliance zone — final score
]

NUM_WAYPOINTS = len(WAYPOINTS)


# ── Arc-length parameterization ───────────────────────────────────────────────

_SEG_LENGTHS = [
    math.hypot(WAYPOINTS[i+1][0] - WAYPOINTS[i][0],
               WAYPOINTS[i+1][1] - WAYPOINTS[i][1])
    for i in range(NUM_WAYPOINTS - 1)
]
_CUM_LENGTHS = [0.0]
for _l in _SEG_LENGTHS:
    _CUM_LENGTHS.append(_CUM_LENGTHS[-1] + _l)
TOTAL_LENGTH = _CUM_LENGTHS[-1]


# ── Path query utilities ───────────────────────────────────────────────────────

def nearest_segment(rx, ry, hint_seg=0, window=4):
    """Returns (seg_idx, px, py, t, dist, arc_pos, cross_sign) for nearest point on path."""
    best_dist = float("inf")
    best = (0, WAYPOINTS[0][0], WAYPOINTS[0][1], 0.0, 0.0, 0.0, 1.0)

    lo = max(0, hint_seg)
    hi = min(NUM_WAYPOINTS - 2, hint_seg + window)

    for i in range(lo, hi + 1):
        ax, ay = WAYPOINTS[i]
        bx, by = WAYPOINTS[i + 1]
        dx, dy = bx - ax, by - ay
        seg_len = _SEG_LENGTHS[i]
        if seg_len < 1e-9:
            continue

        t = ((rx - ax) * dx + (ry - ay) * dy) / (seg_len ** 2)
        t = max(0.0, min(1.0, t))
        px = ax + t * dx
        py = ay + t * dy
        dist = math.hypot(rx - px, ry - py)

        if dist < best_dist:
            best_dist = dist
            arc_pos = _CUM_LENGTHS[i] + t * seg_len
            cross = (rx - ax) * dy - (ry - ay) * dx
            cross_sign = 1.0 if cross >= 0 else -1.0
            best = (i, px, py, t, dist, arc_pos, cross_sign)

    return best


def progress_fraction(arc_pos):
    return arc_pos / TOTAL_LENGTH if TOTAL_LENGTH > 0 else 0.0


def waypoint_relative(rx, ry, heading, wp_idx):
    """Returns (dx, dy) from robot to waypoint in world frame."""
    wx, wy = WAYPOINTS[min(wp_idx, NUM_WAYPOINTS - 1)]
    return wx - rx, wy - ry


# ── Waypoint tracker ──────────────────────────────────────────────────────────

class WaypointTracker:
    """Monotonically advancing waypoint tracker."""

    def __init__(self):
        self.current_idx = 0
        self.done = False

    def reset(self):
        self.current_idx = 0
        self.done = False

    def target_waypoint(self):
        idx = min(self.current_idx, NUM_WAYPOINTS - 1)
        return WAYPOINTS[idx]

    def update(self, rx, ry):
        advanced = 0
        while self.current_idx < NUM_WAYPOINTS:
            wx, wy = WAYPOINTS[self.current_idx]
            if math.hypot(rx - wx, ry - wy) <= WAYPOINT_PASS_RADIUS:
                self.current_idx += 1
                advanced += 1
            else:
                break
        if self.current_idx >= NUM_WAYPOINTS:
            self.done = True
        return advanced
