# ──────────────────────────────────────────────────────────────────────────────
# path_randomizer/geometry.py
# Straight-line clearance checks against the field elements.
# ──────────────────────────────────────────────────────────────────────────────

from lib.field_constants import ROBOT_BUMPER_HALF, IMPASSABLE_RECTS


def leg_clear(ax, ay, bx, by):
    """True if the robot can drive the straight line A->B without touching a
    field element (each element grown by the bumper half-width; slab test)."""
    r = ROBOT_BUMPER_HALF
    dx, dy = bx - ax, by - ay
    for ox1, oy1, ox2, oy2 in IMPASSABLE_RECTS:
        t0, t1 = 0.0, 1.0
        hit = True
        for lo, hi, p, d in ((ox1 - r, ox2 + r, ax, dx), (oy1 - r, oy2 + r, ay, dy)):
            if abs(d) < 1e-12:
                if p <= lo or p >= hi:
                    hit = False
                    break
            else:
                ta, tb = (lo - p) / d, (hi - p) / d
                if ta > tb:
                    ta, tb = tb, ta
                t0, t1 = max(t0, ta), min(t1, tb)
                if t0 >= t1:
                    hit = False
                    break
        if hit:
            return False
    return True
