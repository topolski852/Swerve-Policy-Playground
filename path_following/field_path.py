# ──────────────────────────────────────────────────────────────────────────────
# path_following/field_path.py
# 13-waypoint figure-8 path for the path-following experiment.
# ──────────────────────────────────────────────────────────────────────────────

import math
from lib.field_constants import WAYPOINT_PASS_RADIUS


# ── Waypoints ─────────────────────────────────────────────────────────────────

WAYPOINTS = [
    # ── Midfield crossing ─────────────────────────────────────────────────────
    ( 8.27, 4.10),   # 0  start/end — midfield crossing

    # ── Blue loop (left side) — counterclockwise ──────────────────────────────
    ( 5.90, 5.60),   # 1  neutral side of Blue BumpLeft
    ( 3.20, 5.60),   # 2  alliance side — Blue BumpLeft top
    ( 2.00, 4.10),   # 3  arch midpoint — deep into Blue alliance zone
    ( 3.20, 2.61),   # 4  alliance side — Blue BumpRight bottom
    ( 5.90, 2.61),   # 5  neutral side of Blue BumpRight

    # ── Midfield crossing (second pass) ───────────────────────────────────────
    ( 8.27, 4.10),   # 6  midfield crossing

    # ── Red loop (right side) — clockwise ────────────────────────────────────
    (10.64, 5.60),   # 7  neutral side of Red BumpLeft
    (13.34, 5.60),   # 8  alliance side — Red BumpLeft top
    (14.54, 4.10),   # 9  arch midpoint — deep into Red alliance zone
    (13.34, 2.61),   # 10 alliance side — Red BumpRight bottom
    (10.64, 2.61),   # 11 neutral side of Red BumpRight

    # ── Back to start ─────────────────────────────────────────────────────────
    ( 8.27, 4.10),   # 12 closed loop
]


# ── Arc-length parameterization ───────────────────────────────────────────────

def _build_arc_lengths(waypoints):
    arcs = [0.0]
    for i in range(1, len(waypoints)):
        dx = waypoints[i][0] - waypoints[i-1][0]
        dy = waypoints[i][1] - waypoints[i-1][1]
        arcs.append(arcs[-1] + math.hypot(dx, dy))
    return arcs


ARC_LENGTHS   = _build_arc_lengths(WAYPOINTS)
TOTAL_LENGTH  = ARC_LENGTHS[-1]
NUM_WAYPOINTS = len(WAYPOINTS)


# ── Path query utilities ───────────────────────────────────────────────────────

def nearest_segment(x: float, y: float, hint_seg: int = 0, window: int = 4):
    """
    Find the path segment (i, i+1) closest to (x, y).

    Returns: (seg_idx, t, cx, cy, dist, arc_pos, cross_sign)
    """
    best_dist  = float('inf')
    best_seg   = 0
    best_t     = 0.0
    best_cx    = WAYPOINTS[0][0]
    best_cy    = WAYPOINTS[0][1]
    best_arc   = 0.0
    best_cross = 0.0

    start = max(0, hint_seg - 1)
    end   = min(len(WAYPOINTS) - 1, hint_seg + window)
    for i in range(start, end):
        ax, ay = WAYPOINTS[i]
        bx, by = WAYPOINTS[i+1]

        seg_dx = bx - ax
        seg_dy = by - ay
        seg_len_sq = seg_dx**2 + seg_dy**2
        if seg_len_sq < 1e-12:
            continue

        t = ((x - ax) * seg_dx + (y - ay) * seg_dy) / seg_len_sq
        t = max(0.0, min(1.0, t))

        cx = ax + t * seg_dx
        cy = ay + t * seg_dy
        dist = math.hypot(x - cx, y - cy)

        if dist < best_dist:
            best_dist  = dist
            best_seg   = i
            best_t     = t
            best_cx    = cx
            best_cy    = cy
            best_arc   = ARC_LENGTHS[i] + t * math.hypot(seg_dx, seg_dy)
            best_cross = (seg_dx * (y - ay) - seg_dy * (x - ax))

    cross_sign = 1.0 if best_cross >= 0.0 else -1.0
    return best_seg, best_t, best_cx, best_cy, best_dist, best_arc, cross_sign


def progress_fraction(arc_pos: float) -> float:
    return arc_pos / TOTAL_LENGTH if TOTAL_LENGTH > 0 else 0.0


def waypoint_relative(robot_x: float, robot_y: float,
                      robot_heading: float, wp_idx: int):
    """Returns the position of waypoint wp_idx in the robot's local frame."""
    wx, wy = WAYPOINTS[wp_idx]
    dx_w = wx - robot_x
    dy_w = wy - robot_y
    cos_h = math.cos(-robot_heading)
    sin_h = math.sin(-robot_heading)
    dx_l =  dx_w * cos_h - dy_w * sin_h
    dy_l =  dx_w * sin_h + dy_w * cos_h
    return dx_l, dy_l


# ── Waypoint tracker ──────────────────────────────────────────────────────────

class WaypointTracker:
    """Monotonically advancing waypoint pointer."""

    def __init__(self):
        self.current_idx = 1   # skip spawn point, target first real waypoint

    def reset(self):
        self.current_idx = 1

    @property
    def done(self) -> bool:
        return self.current_idx >= NUM_WAYPOINTS

    def target_waypoint(self):
        idx = min(self.current_idx, NUM_WAYPOINTS - 1)
        return WAYPOINTS[idx]

    def lookahead_waypoint(self, n=1):
        idx = min(self.current_idx + n, NUM_WAYPOINTS - 1)
        return WAYPOINTS[idx]

    def update(self, robot_x: float, robot_y: float) -> int:
        advanced = 0
        while not self.done:
            tx, ty = self.target_waypoint()
            if math.hypot(robot_x - tx, robot_y - ty) < WAYPOINT_PASS_RADIUS:
                self.current_idx += 1
                advanced += 1
            else:
                break
        return advanced
