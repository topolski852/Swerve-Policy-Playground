# ──────────────────────────────────────────────────────────────────────────────
# renderer.py
# Pygame 2D renderer shared by live training (render_mode="human") and render.py
# playback script.
# ──────────────────────────────────────────────────────────────────────────────

import math
import pygame
from field_path import WAYPOINTS
from constants import (
    FIELD_WIDTH, FIELD_HEIGHT,
    RENDER_SCALE, WINDOW_PADDING,
    ROBOT_HALF_WIDTH_X, ROBOT_HALF_WIDTH_Y, MODULE_OFFSETS,
    ROBOT_COLOR, ROBOT_BORDER_COLOR, MODULE_COLOR,
    PATH_COLOR, WAYPOINT_COLOR, ROBOT_HEADING_COLOR,
    ARROW_MAX_PIXELS, ARROW_COLOR, ARROW_WIDTH, ARROW_HEAD_SIZE,
    MODULE_RING_SLOW, MODULE_RING_FAST,
    MAX_SPEED_MPS, TARGET_FPS, WINDOW_TITLE,
)


def _field_to_screen(fx: float, fy: float, scale: float, pad: int, field_h: float):
    """Convert field coordinates (m) to Pygame screen pixels."""
    sx = int(fx * scale) + pad
    # Flip y: field y=0 is bottom, screen y=0 is top
    sy = int((field_h - fy) * scale) + pad
    return sx, sy


class Renderer:

    def __init__(self):
        if not pygame.get_init():
            pygame.init()

        self._scale = RENDER_SCALE
        self._pad   = WINDOW_PADDING
        self._fw    = FIELD_WIDTH
        self._fh    = FIELD_HEIGHT

        w = int(FIELD_WIDTH  * RENDER_SCALE) + 2 * WINDOW_PADDING
        h = int(FIELD_HEIGHT * RENDER_SCALE) + 2 * WINDOW_PADDING

        self.screen = pygame.display.set_mode((w, h))
        pygame.display.set_caption(WINDOW_TITLE)
        self.clock  = pygame.time.Clock()
        self._font  = pygame.font.SysFont("consolas", 13)

    # ── Public API ─────────────────────────────────────────────────────────────

    def draw(self, robot_state, tracker, module_states, info=None):
        """
        Full-frame draw call.

        robot_state  : SwerveState
        tracker      : WaypointTracker
        module_states: list of (angle_rad, speed_mps) — FL, FR, BL, BR
        info         : optional dict of scalars to display in the HUD
        """
        self._handle_events()
        self.screen.fill((30, 30, 30))

        self._draw_field_border()
        self._draw_path(tracker.current_idx)
        self._draw_robot(robot_state, module_states)
        if info:
            self._draw_hud(info)

        pygame.display.flip()
        self.clock.tick(TARGET_FPS)

    def close(self):
        pygame.quit()

    # ── Drawing helpers ────────────────────────────────────────────────────────

    def _fs(self, fx, fy):
        return _field_to_screen(fx, fy, self._scale, self._pad, self._fh)

    def _handle_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.close()
                raise SystemExit

    def _draw_field_border(self):
        rect = pygame.Rect(
            self._pad, self._pad,
            int(self._fw * self._scale),
            int(self._fh * self._scale)
        )
        pygame.draw.rect(self.screen, (55, 55, 55), rect)
        pygame.draw.rect(self.screen, (90, 90, 90), rect, 2)

    def _draw_path(self, active_wp_idx: int):
        pts = [self._fs(wx, wy) for wx, wy in WAYPOINTS]

        # Draw path segments — completed segments slightly dimmed
        for i in range(len(pts) - 1):
            color = PATH_COLOR if i >= active_wp_idx - 1 else (40, 80, 130)
            pygame.draw.line(self.screen, color, pts[i], pts[i+1], 2)

        # Waypoint markers
        for i, (sx, sy) in enumerate(pts):
            if i == 0 or i == len(pts) - 1:
                pygame.draw.circle(self.screen, (255, 100, 100), (sx, sy), 6)
            elif i < active_wp_idx:
                pygame.draw.circle(self.screen, (60, 100, 60), (sx, sy), 4)
            elif i == active_wp_idx:
                pygame.draw.circle(self.screen, WAYPOINT_COLOR, (sx, sy), 7)
                pygame.draw.circle(self.screen, (255, 255, 255), (sx, sy), 7, 1)
            else:
                pygame.draw.circle(self.screen, (100, 100, 100), (sx, sy), 4)

    def _draw_robot(self, robot, module_states):
        rx, ry   = robot.x, robot.y
        heading  = robot.heading
        cos_h    = math.cos(heading)
        sin_h    = math.sin(heading)

        # ── Chassis body ──────────────────────────────────────────────────────
        hw_x = ROBOT_HALF_WIDTH_X * self._scale
        hw_y = ROBOT_HALF_WIDTH_Y * self._scale

        # Four corners in robot frame, rotated to world, then to screen
        corners_robot = [
            ( ROBOT_HALF_WIDTH_X,  ROBOT_HALF_WIDTH_Y),
            ( ROBOT_HALF_WIDTH_X, -ROBOT_HALF_WIDTH_Y),
            (-ROBOT_HALF_WIDTH_X, -ROBOT_HALF_WIDTH_Y),
            (-ROBOT_HALF_WIDTH_X,  ROBOT_HALF_WIDTH_Y),
        ]
        corners_screen = []
        for (cx, cy) in corners_robot:
            wx = rx + cx * cos_h - cy * sin_h
            wy = ry + cx * sin_h + cy * cos_h
            corners_screen.append(self._fs(wx, wy))

        pygame.draw.polygon(self.screen, ROBOT_COLOR, corners_screen)
        pygame.draw.polygon(self.screen, ROBOT_BORDER_COLOR, corners_screen, 2)

        # ── Heading indicator (front edge highlight) ──────────────────────────
        pygame.draw.line(self.screen, ROBOT_HEADING_COLOR,
                         corners_screen[0], corners_screen[1], 3)

        # ── Module housings + state arrows ────────────────────────────────────
        for i, ((mx, my), (m_angle, m_speed)) in enumerate(
                zip(MODULE_OFFSETS, module_states)):
            # Module center in world frame
            wx = rx + mx * cos_h - my * sin_h
            wy = ry + mx * sin_h + my * cos_h
            sx, sy = self._fs(wx, wy)

            # Speed-coloured ring behind housing
            t = min(m_speed / MAX_SPEED_MPS, 1.0)
            ring_color = tuple(
                int(MODULE_RING_SLOW[c] + t * (MODULE_RING_FAST[c] - MODULE_RING_SLOW[c]))
                for c in range(3)
            )
            pygame.draw.circle(self.screen, ring_color, (sx, sy), 9)

            # Module housing (small filled square, rotated with chassis)
            mod_hw = 6
            mod_corners_r = [(-mod_hw, -mod_hw), (mod_hw, -mod_hw),
                              (mod_hw,  mod_hw), (-mod_hw,  mod_hw)]
            mod_screen = []
            for (mcx, mcy) in mod_corners_r:
                # These are pixel offsets, so scale is already applied
                rot_x = mcx * cos_h - mcy * sin_h
                rot_y = mcx * sin_h + mcy * cos_h
                mod_screen.append((sx + rot_x, sy - rot_y))
            pygame.draw.polygon(self.screen, MODULE_COLOR, mod_screen)
            pygame.draw.polygon(self.screen, ROBOT_BORDER_COLOR, mod_screen, 1)

            # AdvantageScope-style arrow: direction=module angle, length~speed
            if m_speed > 0.05:
                arrow_len = (m_speed / MAX_SPEED_MPS) * ARROW_MAX_PIXELS

                # Module angle is in robot frame; convert to world/screen frame
                world_angle = m_angle + heading
                # Screen y is flipped, so negate the y component
                adx =  math.cos(world_angle) * arrow_len
                ady = -math.sin(world_angle) * arrow_len

                ex = sx + adx
                ey = sy + ady

                pygame.draw.line(self.screen, ARROW_COLOR,
                                 (sx, sy), (int(ex), int(ey)), ARROW_WIDTH)

                # Arrow head (triangle)
                perp_angle = world_angle + math.pi / 2
                hx = ARROW_HEAD_SIZE * math.cos(perp_angle)
                hy = ARROW_HEAD_SIZE * math.sin(perp_angle)
                tip    = (int(ex), int(ey))
                left   = (int(ex - adx * 0.4 + hx), int(ey - ady * 0.4 - hy))
                right  = (int(ex - adx * 0.4 - hx), int(ey - ady * 0.4 + hy))
                pygame.draw.polygon(self.screen, ARROW_COLOR, [tip, left, right])

    def _draw_hud(self, info: dict):
        y = self._pad + 4
        for key, val in info.items():
            if isinstance(val, float):
                text = f"{key}: {val:+.3f}"
            else:
                text = f"{key}: {val}"
            surf = self._font.render(text, True, (200, 200, 200))
            self.screen.blit(surf, (self._pad + 4, y))
            y += 16
