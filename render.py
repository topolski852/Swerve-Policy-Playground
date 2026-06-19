# ──────────────────────────────────────────────────────────────────────────────
# render.py
# Load a saved checkpoint and render the robot driving the path in Pygame.
#
# Usage:
#   python render.py path_following/checkpoints/swerve_final
#   python render.py path_following/checkpoints/swerve_50000_steps --speed 0.5
# ──────────────────────────────────────────────────────────────────────────────

import argparse
import time
import pygame

from stable_baselines3 import SAC
from path_following.swerve_env import SwerveEnv
from path_following.field_path import WAYPOINTS, TOTAL_LENGTH
from lib.renderer import Renderer


def main():
    parser = argparse.ArgumentParser(description="Render a trained path-following checkpoint")
    parser.add_argument("checkpoint", help="Path to .zip checkpoint file")
    parser.add_argument("--episodes", type=int, default=5,
                        help="Number of episodes to render (default: 5)")
    parser.add_argument("--speed", type=float, default=1.0,
                        help="Playback speed multiplier (0.5=half speed, 2.0=double)")
    args = parser.parse_args()

    env      = SwerveEnv()
    renderer = Renderer(waypoints=WAYPOINTS)

    print(f"Loading checkpoint: {args.checkpoint}")
    model = SAC.load(args.checkpoint, device="cpu")

    for ep in range(args.episodes):
        obs, _ = env.reset()
        done   = False
        ep_reward = 0.0
        step = 0

        print(f"\nEpisode {ep + 1}/{args.episodes}")

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            ep_reward += reward
            done = terminated or truncated
            step += 1

            hud = {
                "step":      step,
                "reward":    ep_reward,
                "progress":  round(float(info.get("arc_pos", 0)) / TOTAL_LENGTH * 100, 1),
                "cross(m)":  round(float(info.get("cross_track", 0)), 3),
            }

            renderer.draw(env._robot, env._tracker, env._get_module_states(), info=hud)

            if args.speed < 1.0:
                time.sleep(0.02 / args.speed - 0.02)

        status = "SUCCESS" if terminated else "TIMEOUT/OFF-PATH"
        print(f"  {status}  steps={step}  total_reward={ep_reward:.2f}")

    renderer.close()
    env.close()


if __name__ == "__main__":
    main()
