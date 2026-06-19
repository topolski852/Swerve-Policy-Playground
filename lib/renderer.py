# ──────────────────────────────────────────────────────────────────────────────
# lib/renderer.py
# Pygame 2D renderer shared by all experiments.
# Waypoints are passed at construction so any experiment's path can be drawn.
# ──────────────────────────────────────────────────────────────────────────────

import os
import math
import pygame
from lib.field_constants import (
    FIELD_LENGTH, FIELD_WIDTH,
    RENDER_SCALE, WINDOW_PADDING, FIELD_IMAGE, FIELD_CORNER_BL, FIELD_CORNER_TR,
    ROBOT_BUMPER_HALF, MODULE_OFFSETS, IMPASSABLE_RECTS,
    ROBOT_COLOR, ROBOT_BORDER_COLOR, MODULE_COLOR,
    PATH_COLOR, WAYPOINT_COLOR, ROBOT_HEADING_COLOR,
    ARROW_MAX_PIXELS, ARROW_COLOR, ARROW_WIDTH, ARROW_HEAD_SIZE,
    MODULE_RING_SLOW, MODULE_RING_FAST,
    MAX_SPEED_MPS, TARGET_FPS, WINDOW_TITLE,
)


class Renderer:

    def __init__(self, waypoints=None, record_path=None):
        """
        waypoints   : list of (x, y) tuples defining the path to draw.
                      Pass None to draw no path overlay.
        record_path : if set, writes all frames to an MP4 at this path.
        """
        self._waypoints = waypoints or []

        if not pygame.get_init():
            pygame.init()

        self._pad = WINDOW_PADDING

        _src_field_px = FIELD_CORNER_TR[0] - FIELD_CORNER_BL[0]
        self._disp_scale = RENDER_SCALE * FIELD_LENGTH / _src_field_px

        field_img_raw = pygame.image.load(FIELD_IMAGE)
        src_w = field_img_raw.get_width()
        src_h = field_img_raw.get_height()
        disp_w = int(src_w * self._disp_scale)
        disp_h = int(src_h * self._disp_scale)

        w = disp_w + 2 * WINDOW_PADDING
        h = disp_h + 2 * WINDOW_PADDING

        self.screen = pygame.display.set_mode((w, h))
        pygame.display.set_caption(WINDOW_TITLE)
        self.clock  = pygame.time.Clock()
        self._font  = pygame.font.SysFont("consolas", 13)

        self._field_img = pygame.transform.smoothscale(field_img_raw.convert(), (disp_w, disp_h))

        self._video_writer = None
        if record_path is not None:
            import imageio
            rec_dir = os.path.dirname(record_path)
            if rec_dir:
                os.makedirs(rec_dir, exist_ok=True)
            self._video_writer = imageio.get_writer(record_path, fps=50)

    # ── Public API ─────────────────────────────────────────────────────────────

    def draw(self, robot_state, tracker, module_states, info=None):
        """
        Full-frame draw call.

        robot_state  : SwerveState
        tracker      : WaypointTracker
        module_states: list of (angle_rad, speed_mps) — FL, FR, BL, BR
        info         : optional dict of scalars to display in the HUD
        """
        if not self._alive:
            return
        self._handle_events()
        if not self._alive:
            return
        self.screen.fill((30, 30, 30))

        self._draw_field_border()
        self._draw_obstacles()
        if self._waypoints and tracker is not None:
            self._draw_path(tracker.current_idx)
        self._draw_robot(robot_state, module_states)
        if info:
            self._draw_hud(info)

        pygame.display.flip()
        self.clock.tick(TARGET_FPS)

        if self._video_writer is not None:
            frame = pygame.surfarray.array3d(self.screen).transpose(1, 0, 2)
            self._video_writer.append_data(frame)

    def close(self):
        if self._video_writer is not None:
            self._video_writer.close()
            self._video_writer = None
        pygame.quit()

    # ── Drawing helpers ────────────────────────────────────────────────────────

    def _fs(self, fx: float, fy: float):
        """Field coords (m) → screen pixels using the two calibration corners."""
        t_x = fx / FIELD_LENGTH
        t_y = fy / FIELD_WIDTH
        img_x = FIELD_CORNER_BL[0] + t_x * (FIELD_CORNER_TR[0] - FIELD_CORNER_BL[0])
        img_y = FIELD_CORNER_BL[1] + t_y * (FIELD_CORNER_TR[1] - FIELD_CORNER_BL[1])
        sx = int(img_x * self._disp_scale) + self._pad
        sy = int(img_y * self._disp_scale) + self._pad
        return sx, sy

    def _handle_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                self._alive = False
                return

    @property
    def _alive(self):
        return getattr(self, "_window_alive", True)

    @_alive.setter
    def _alive(self, v):
        self._window_alive = v

    def _draw_field_border(self):
        self.screen.blit(self._field_img, (self._pad, self._pad))

    def _draw_obstacles(self):
        """Semi-transparent red overlay on each impassable zone."""
        for (ox1, oy1, ox2, oy2) in IMPASSABLE_RECTS:
            sx1, sy1 = self._fs(ox1, oy1)
            sx2, sy2 = self._fs(ox2, oy2)
            rx = min(sx1, sx2)
            ry = min(sy1, sy2)
            rw = abs(sx2 - sx1)
            rh = abs(sy2 - sy1)
            if rw < 1 or rh < 1:
                continue
            overlay = pygame.Surface((rw, rh), pygame.SRCALPHA)
            overlay.fill((220, 50, 50, 60))
            self.screen.blit(overlay, (rx, ry))
            pygame.draw.rect(self.screen, (220, 50, 50), (rx, ry, rw, rh), 1)

    def _draw_path(self, active_wp_idx: int):
        pts = [self._fs(wx, wy) for wx, wy in self._waypoints]

        for i in range(len(pts) - 1):
            color = PATH_COLOR if i >= active_wp_idx - 1 else (40, 80, 130)
            pygame.draw.line(self.screen, color, pts[i], pts[i+1], 2)

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

        bh = ROBOT_BUMPER_HALF
        corners_robot = [( bh,  bh), ( bh, -bh), (-bh, -bh), (-bh,  bh)]
        corners_screen = []
        for (cx, cy) in corners_robot:
            wx = rx + cx * cos_h - cy * sin_h
            wy = ry + cx * sin_h + cy * cos_h
            corners_screen.append(self._fs(wx, wy))

        pygame.draw.polygon(self.screen, ROBOT_COLOR, corners_screen)
        pygame.draw.polygon(self.screen, ROBOT_BORDER_COLOR, corners_screen, 2)

        pygame.draw.line(self.screen, ROBOT_HEADING_COLOR,
                         corners_screen[0], corners_screen[1], 3)

        for i, ((mx, my), (m_angle, m_speed)) in enumerate(
                zip(MODULE_OFFSETS, module_states)):
            wx = rx + mx * cos_h - my * sin_h
            wy = ry + mx * sin_h + my * cos_h
            sx, sy = self._fs(wx, wy)

            t = min(m_speed / MAX_SPEED_MPS, 1.0)
            ring_color = tuple(
                int(MODULE_RING_SLOW[c] + t * (MODULE_RING_FAST[c] - MODULE_RING_SLOW[c]))
                for c in range(3)
            )
            pygame.draw.circle(self.screen, ring_color, (sx, sy), 9)

            mod_hw = 6
            mod_corners_r = [(-mod_hw, -mod_hw), (mod_hw, -mod_hw),
                              (mod_hw,  mod_hw), (-mod_hw,  mod_hw)]
            mod_screen = []
            for (mcx, mcy) in mod_corners_r:
                rot_x = mcx * cos_h - mcy * sin_h
                rot_y = mcx * sin_h + mcy * cos_h
                mod_screen.append((sx + rot_x, sy - rot_y))
            pygame.draw.polygon(self.screen, MODULE_COLOR, mod_screen)
            pygame.draw.polygon(self.screen, ROBOT_BORDER_COLOR, mod_screen, 1)

            if m_speed > 0.05:
                arrow_len = (m_speed / MAX_SPEED_MPS) * ARROW_MAX_PIXELS
                world_angle = m_angle + heading
                adx =  math.cos(world_angle) * arrow_len
                ady = -math.sin(world_angle) * arrow_len
                ex = sx + adx
                ey = sy + ady

                pygame.draw.line(self.screen, ARROW_COLOR,
                                 (sx, sy), (int(ex), int(ey)), ARROW_WIDTH)

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
