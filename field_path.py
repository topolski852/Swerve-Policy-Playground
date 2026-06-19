# ──────────────────────────────────────────────────────────────────────────────
# field_path.py
# Hardcoded FRC-style field path and arc-length utilities.
#
# Path layout (field is 16.54 m wide x 8.21 m tall, origin = bottom-left):
#
#   Start near bottom-left, drive a long straight toward center-right,
#   sweep around the far end (tight turn), come back along the top,
#   S-curve through the middle, finish near start.
#
# ──────────────────────────────────────────────────────────────────────────────

import math
from constants import WAYPOINT_PASS_RADIUS


# ── Waypoints ─────────────────────────────────────────────────────────────────
# (x, y) in meters, field frame (origin bottom-left, x=right, y=up).
# Edit these to reshape the path; all downstream code uses WAYPOINTS.

WAYPOINTS = [
    # ── Midfield crossing ─────────────────────────────────
    ( 8.27, 4.10),   # 0  start/end — midfield crossing

    # ── Approach Blue BumpLeft ────────────────────────────
    ( 6.50, 5.60),   # 1  neutral zone, aligned to BumpLeft corridor
    ( 5.40, 5.60),   # 2  BUMP_NEUTRAL  — Blue BumpLeft entry  (from Nodes.java)
    ( 3.70, 5.60),   # 3  BUMP_ALLIANCE — Blue BumpLeft exit   (from Nodes.java)

    # ── Blue alliance zone: loop left around blue hub ─────
    ( 2.80, 5.20),   # 4  above-left of blue hub
    ( 2.00, 4.10),   # 5  left of blue hub center
    ( 2.80, 3.00),   # 6  below-left of blue hub

    # ── Cross Blue BumpRight back to neutral ──────────────
    ( 3.70, 2.61),   # 7  BUMP_ALLIANCE — Blue BumpRight entry (Y-flip of BumpLeft)
    ( 5.40, 2.61),   # 8  BUMP_NEUTRAL  — Blue BumpRight exit
    ( 6.50, 2.60),   # 9  neutral zone exit

    # ── Midfield crossing (second pass) ───────────────────
    ( 8.27, 4.10),   # 10 midfield crossing

    # ── Approach Red BumpRight ────────────────────────────
    ( 9.80, 2.60),   # 11 neutral zone, aligned to Red BumpRight corridor
    (11.14, 2.61),   # 12 BUMP_NEUTRAL  — Red BumpRight entry  (X-flip of Blue)
    (12.84, 2.61),   # 13 BUMP_ALLIANCE — Red BumpRight exit

    # ── Red alliance zone: loop right around red hub ──────
    (13.70, 3.00),   # 14 below-right of red hub
    (14.50, 4.10),   # 15 right of red hub center
    (13.70, 5.20),   # 16 above-right of red hub

    # ── Cross Red BumpLeft back to neutral ────────────────
    (12.84, 5.60),   # 17 BUMP_ALLIANCE — Red BumpLeft entry
    (11.14, 5.60),   # 18 BUMP_NEUTRAL  — Red BumpLeft exit
    ( 9.80, 5.60),   # 19 neutral zone return

    # ── Back to start ─────────────────────────────────────
    ( 8.27, 4.10),   # 20 closed loop
]


# ── Arc-length parameterization ───────────────────────────────────────────────

def _build_arc_lengths(waypoints):
    """
    Precompute cumulative arc lengths for each waypoint.
    Returns list of length len(waypoints), with arc_lengths[0] = 0.
    """
    arcs = [0.0]
    for i in range(1, len(waypoints)):
        dx = waypoints[i][0] - waypoints[i-1][0]
        dy = waypoints[i][1] - waypoints[i-1][1]
        arcs.append(arcs[-1] + math.hypot(dx, dy))
    return arcs


ARC_LENGTHS  = _build_arc_lengths(WAYPOINTS)
TOTAL_LENGTH = ARC_LENGTHS[-1]
NUM_WAYPOINTS = len(WAYPOINTS)


# ── Path query utilities ───────────────────────────────────────────────────────

def nearest_segment(x: float, y: float, hint_seg: int = 0, window: int = 4):
    """
    Find the path segment (i, i+1) closest to (x, y).

    hint_seg : start of search window (typically tracker.current_idx - 1).
               Limits the search so the robot can't accidentally latch onto
               a distant segment with a high arc position — the main cause
               of the closed-loop backwards-running exploit.
    window   : number of segments to search ahead of hint_seg.

    Returns:
        seg_idx    : index of the segment start waypoint
        t          : parameter in [0, 1] along that segment
        cx, cy     : closest point on segment
        dist       : perpendicular distance to that closest point
        arc_pos    : arc-length position of the closest point
        cross_sign : +1 if robot is to the left of the path direction, -1 right
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

        # Project (x,y) onto segment
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
            # Cross product: positive = robot left of path direction
            best_cross = (seg_dx * (y - ay) - seg_dy * (x - ax))

    cross_sign = 1.0 if best_cross >= 0.0 else -1.0
    return best_seg, best_t, best_cx, best_cy, best_dist, best_arc, cross_sign


def progress_fraction(arc_pos: float) -> float:
    """Normalize arc position to [0, 1]."""
    return arc_pos / TOTAL_LENGTH if TOTAL_LENGTH > 0 else 0.0


def waypoint_relative(robot_x: float, robot_y: float,
                      robot_heading: float, wp_idx: int):
    """
    Returns the position of waypoint wp_idx in the robot's local frame
    as (dx_local, dy_local).
    """
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
    """
    Monotonically advancing waypoint pointer.

    Advances to the next waypoint when the robot enters WAYPOINT_PASS_RADIUS.
    Never regresses — prevents the agent from backing up for reward.
    """

    def __init__(self):
        self.current_idx = 0    # index of the *next* waypoint to reach
        self._advance_to_first_non_start()

    def _advance_to_first_non_start(self):
        # Start at waypoint 1 (skip the spawn point itself)
        self.current_idx = 1

    def reset(self):
        self.current_idx = 1

    @property
    def done(self) -> bool:
        return self.current_idx >= NUM_WAYPOINTS

    def target_waypoint(self):
        """(x, y) of the current target waypoint."""
        idx = min(self.current_idx, NUM_WAYPOINTS - 1)
        return WAYPOINTS[idx]

    def lookahead_waypoint(self, n=1):
        """(x, y) of a waypoint n steps ahead of the current target."""
        idx = min(self.current_idx + n, NUM_WAYPOINTS - 1)
        return WAYPOINTS[idx]

    def update(self, robot_x: float, robot_y: float) -> int:
        """
        Check if the robot has passed the current target waypoint.
        Advances if within pass radius. Returns number of waypoints advanced.
        """
        advanced = 0
        while not self.done:
            tx, ty = self.target_waypoint()
            dist = math.hypot(robot_x - tx, robot_y - ty)
            if dist < WAYPOINT_PASS_RADIUS:
                self.current_idx += 1
                advanced += 1
            else:
                break
        return advanced
