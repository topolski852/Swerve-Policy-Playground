# ──────────────────────────────────────────────────────────────────────────────
# render_teleop.py
# Watch a trained teleop-assist policy drive the simulated field.
#
# Usage:
#   python render_teleop.py teleop_assist/teleop_final.zip
#   python render_teleop.py teleop_assist/teleop_final.zip --episodes 10
#   python render_teleop.py teleop_assist/teleop_final.zip --speed 0.5
#
# HUD columns:
#   step / ep_reward       — episode progress
#   joy / out              — joystick intent magnitude vs. policy output magnitude
#   r_intent / r_approach  — main reward components (positive = good, negative = bad)
#   r_still / r_smooth     — secondary penalties
#   rays                   — 8 proximity distances in metres (F FL L BL B BR R FR)
#                            "!" marks rays inside the 2.0 m danger zone
# ──────────────────────────────────────────────────────────────────────────────

import argparse
import time

import numpy as np
import pygame
from stable_baselines3 import SAC

from teleop_assist.env import TeleopAssistEnv
from teleop_assist.constants import RAY_MAX_DISTANCE, DANGER_ZONE_NORM
from lib.renderer import Renderer

RAY_LABELS = ["F", "FL", "L", "BL", "B", "BR", "R", "FR"]
DANGER_DIST = DANGER_ZONE_NORM * RAY_MAX_DISTANCE   # 2.0 m


def main():
    parser = argparse.ArgumentParser(description="Render a trained teleop-assist checkpoint")
    parser.add_argument("checkpoint", help="Path to .zip checkpoint (e.g. teleop_assist/teleop_final.zip)")
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--speed", type=float, default=1.0,
                        help="Playback speed multiplier (0.5 = half speed)")
    args = parser.parse_args()

    env      = TeleopAssistEnv()
    renderer = Renderer()

    print(f"Loading: {args.checkpoint}")
    model = SAC.load(args.checkpoint, device="cpu")
    print("Ready. Close the window or press Ctrl+C to stop.\n")

    for ep in range(args.episodes):
        obs, _    = env.reset()
        done      = False
        ep_reward = 0.0
        step      = 0

        print(f"Episode {ep + 1}/{args.episodes}")

        while not done:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    renderer.close()
                    env.close()
                    return

            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            ep_reward += reward
            done = terminated or truncated
            step += 1

            # Build compact ray string — "!" prefix = inside danger zone
            rays = env._ray_distances
            ray_str = " ".join(
                f"{'!' if rays[i] < DANGER_DIST else ' '}{RAY_LABELS[i]}:{rays[i]:.1f}"
                for i in range(8)
            )

            out_mag = float(np.linalg.norm(action[:2]))

            hud = {
                "step":       step,
                "ep_reward":  round(ep_reward, 1),
                "joy":        round(info.get("joy_mag", 0.0), 2),
                "out":        round(out_mag, 2),
                "r_intent":   round(info.get("r_intent",   0.0), 3),
                "r_approach": round(info.get("r_approach", 0.0), 3),
                "r_still":    round(info.get("r_still",    0.0), 3),
                "r_smooth":   round(info.get("r_smooth",   0.0), 3),
                "rays":       ray_str,
            }

            renderer.draw(env._robot, None, env._robot.module_states, info=hud)

            if args.speed < 1.0:
                time.sleep(0.02 / args.speed - 0.02)

        status = "COLLISION" if terminated else "TIMEOUT"
        print(f"  {status}  steps={step}  reward={ep_reward:.2f}")

    renderer.close()
    env.close()


if __name__ == "__main__":
    main()
