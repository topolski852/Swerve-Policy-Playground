# ──────────────────────────────────────────────────────────────────────────────
# path_randomizer/pathing.py
# Shortest drivable path from the robot to a node, going around field elements.
# Used only for the training reward; the robot never runs this.
#
# Why: the progress and alignment rewards used to measure straight-line
# distance. When a hub sits between the robot and the node, the right move is
# around it, which makes the straight-line distance GROW, so the reward
# punished the correct detour. Measuring along the shortest drivable path
# rewards going around instead.
#
# How: the shortest path around rectangles always bends at their corners, so
# the only points that matter are the corners of each field element, grown by
# the bumper half-width plus a small margin (a "visibility graph"). From the
# node we run Dijkstra over corners that can see each other; then from any
# robot position the path length is
#     straight line, if the robot can see the node, else
#     min over corners c the robot can see:  |robot - c| + (path c -> node)
# ──────────────────────────────────────────────────────────────────────────────

import heapq
import math

from lib.field_constants import FIELD_LENGTH, FIELD_WIDTH, ROBOT_BUMPER_HALF, IMPASSABLE_RECTS
from path_randomizer.geometry import leg_clear

CORNER_MARGIN = 0.05   # metres beyond the bumper; paths may skim corners this close

_g = ROBOT_BUMPER_HALF + CORNER_MARGIN
_CORNERS = []
for (x1, y1, x2, y2) in IMPASSABLE_RECTS:
    for cx, cy in ((x1 - _g, y1 - _g), (x2 + _g, y1 - _g), (x2 + _g, y2 + _g), (x1 - _g, y2 + _g)):
        # Corners outside the drivable field (e.g. trench walls touching the
        # field wall) can't be driven to.
        if ROBOT_BUMPER_HALF <= cx <= FIELD_LENGTH - ROBOT_BUMPER_HALF and \
           ROBOT_BUMPER_HALF <= cy <= FIELD_WIDTH - ROBOT_BUMPER_HALF:
            _CORNERS.append((cx, cy))

# Corner-to-corner edges never change, so find them once.
_EDGES = [[] for _ in _CORNERS]
for i, (ax, ay) in enumerate(_CORNERS):
    for j in range(i + 1, len(_CORNERS)):
        bx, by = _CORNERS[j]
        if leg_clear(ax, ay, bx, by):
            d = math.hypot(bx - ax, by - ay)
            _EDGES[i].append((j, d))
            _EDGES[j].append((i, d))


class PathToNode:
    """Shortest drivable path to one node. Build once per node, query every step."""

    def __init__(self, nx, ny):
        self.nx, self.ny = nx, ny
        # Dijkstra from the node over the corner graph.
        dist = [math.inf] * len(_CORNERS)
        heap = []
        for i, (cx, cy) in enumerate(_CORNERS):
            if leg_clear(cx, cy, nx, ny):
                dist[i] = math.hypot(nx - cx, ny - cy)
                heap.append((dist[i], i))
        heapq.heapify(heap)
        while heap:
            d, i = heapq.heappop(heap)
            if d > dist[i]:
                continue
            for j, w in _EDGES[i]:
                if d + w < dist[j]:
                    dist[j] = d + w
                    heapq.heappush(heap, (dist[j], j))
        self._corner_dist = dist

    def query(self, rx, ry):
        """Returns (path length, (x, y) of the first point to drive toward)."""
        if leg_clear(rx, ry, self.nx, self.ny):
            return math.hypot(self.nx - rx, self.ny - ry), (self.nx, self.ny)
        best, target = math.inf, None
        for (cx, cy), cd in zip(_CORNERS, self._corner_dist):
            if cd == math.inf:
                continue
            total = math.hypot(cx - rx, cy - ry) + cd
            if total < best and leg_clear(rx, ry, cx, cy):
                best, target = total, (cx, cy)
        if target is None:   # boxed in (shouldn't happen): fall back to straight line
            return math.hypot(self.nx - rx, self.ny - ry), (self.nx, self.ny)
        return best, target
