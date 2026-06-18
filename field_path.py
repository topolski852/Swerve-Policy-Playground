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
    ( 1.5,  1.5),    # 0  start (bottom-left zone)
    ( 4.0,  1.5),    # 1  long straight — mid-field bottom
    ( 7.0,  1.5),    # 2
    (10.0,  1.5),    # 3  approaching far end
    (12.5,  2.0),    # 4  beginning of far-end curve
    (14.5,  3.5),    # 5  sweeping right turn (tight)
    (14.8,  5.5),    # 6  top of tight turn
    (13.5,  7.0),    # 7  exit of tight turn, heading left
    (10.5,  7.2),    # 8  long return along top
    ( 7.5,  7.0),    # 9
    ( 5.5,  6.0),    # 10 S-curve entry (drop toward center)
    ( 4.5,  4.5),    # 11 S-curve mid
    ( 3.5,  3.0),    # 12 S-curve exit
    ( 2.0,  2.5),    # 13 closing approach
    ( 1.5,  1.5),    # 14 back to start (closed loop)
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

def nearest_segment(x: float, y: float):
    """
    Find the path segment (i, i+1) closest to (x, y).

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

    for i in range(len(WAYPOINTS) - 1):
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
        return self.current_idx >= NUM_WAYPOINTS - 1

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
