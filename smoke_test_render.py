"""
smoke_test_render.py
Drives the robot with a rule-based controller to verify the path-following
renderer looks correct before training. Close the window to exit.
"""

import math
import pygame
import numpy as np

from path_following.swerve_env import SwerveEnv
from path_following.field_path import WAYPOINTS
from lib.renderer import Renderer
from lib.field_constants import MAX_SPEED_MPS


def main():
    env      = SwerveEnv()
    renderer = Renderer(waypoints=WAYPOINTS)
    obs, _   = env.reset()

    print("Smoke-test renderer: simple rule-based controller following the path.")
    print("Close the window to exit.")

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        rx = env._robot.x
        ry = env._robot.y
        tx, ty = env._tracker.target_waypoint()

        dx = tx - rx
        dy = ty - ry
        dist = math.hypot(dx, dy)

        if dist > 0.01:
            speed = min(2.5, dist * 1.5)
            vx_n = (dx / dist) * speed / MAX_SPEED_MPS
            vy_n = (dy / dist) * speed / MAX_SPEED_MPS
        else:
            vx_n = vy_n = 0.0

        action = np.array([vx_n, vy_n, 0.0], dtype="float32")
        obs, reward, terminated, truncated, info = env.step(action)

        hud = {
            "x":      round(env._robot.x, 2),
            "y":      round(env._robot.y, 2),
            "wp":     env._tracker.current_idx,
            "reward": round(reward, 3),
        }

        renderer.draw(env._robot, env._tracker, env._get_module_states(), info=hud)

        if terminated or truncated:
            print("Episode ended — resetting.")
            obs, _ = env.reset()

    renderer.close()
    env.close()


if __name__ == "__main__":
    main()
