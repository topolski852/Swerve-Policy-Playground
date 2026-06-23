# ──────────────────────────────────────────────────────────────────────────────
# test_randomizer.py
# Interactive model tester for the path_randomizer experiment.
#
# Usage:
#   python test_randomizer.py                        # final model, Phase 6, 10 episodes
#   python test_randomizer.py --phase 3              # test at Phase 3 difficulty
#   python test_randomizer.py --n-episodes 25
#   python test_randomizer.py --checkpoint path_randomizer/checkpoints/randomizer_500000_steps.zip
#   python test_randomizer.py --stochastic           # stochastic policy instead of deterministic
#
# Controls:
#   Close the window or press Q to stop early.
# ──────────────────────────────────────────────────────────────────────────────

import argparse
import sys
import pygame
from stable_baselines3 import SAC
from path_randomizer.swerve_env import SwerveEnv

DEFAULT_CHECKPOINT = "path_randomizer/checkpoints/randomizer_final.zip"

# Diagnostic difficulty levels for testing — not curriculum phases.
# Training uses level 3 (full difficulty) throughout.
PHASE_SETTINGS = {
    1: dict(_n_waypoints_min=3, _n_waypoints_max=5,  _wp_distance_max=2.0),  # easy
    2: dict(_n_waypoints_min=3, _n_waypoints_max=8,  _wp_distance_max=4.0),  # medium
    3: dict(_n_waypoints_min=3, _n_waypoints_max=12, _wp_distance_max=6.0),  # full (training distribution)
}


def run_episode(model, env, renderer, episode_num, deterministic):
    """Run one episode. Returns (ep_reward, steps, completed, quit_requested)."""
    obs, _ = env.reset()
    renderer.set_waypoints(env._waypoints)

    # Pump a few frames so the window draws before the agent moves
    for _ in range(5):
        pygame.event.pump()
        renderer.draw(env._robot, env._tracker, env._get_module_states())

    ep_reward  = 0.0
    step       = 0
    terminated = False

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return ep_reward, step, terminated, True
            if event.type == pygame.KEYDOWN and event.key == pygame.K_q:
                return ep_reward, step, terminated, True

        action, _ = model.predict(obs, deterministic=deterministic)
        obs, reward, terminated, truncated, info = env.step(action)
        ep_reward += reward
        step += 1

        hud = {
            "episode":  episode_num,
            "step":     step,
            "reward":   round(ep_reward, 2),
            "waypoint": f"{info['waypoint_idx']}/{info['n_waypoints']}",
        }
        renderer.draw(env._robot, env._tracker, env._get_module_states(), info=hud)

        if terminated or truncated:
            # Hold the final frame for a moment so the user can see the result
            for _ in range(30):
                pygame.event.pump()
                renderer.draw(env._robot, env._tracker, env._get_module_states(), info=hud)
            break

    return ep_reward, step, terminated, False


def main():
    parser = argparse.ArgumentParser(description="Test a trained path_randomizer model with live rendering.")
    parser.add_argument("--checkpoint",  type=str,  default=DEFAULT_CHECKPOINT,
                        help=f"model zip to load (default: {DEFAULT_CHECKPOINT})")
    parser.add_argument("--n-episodes",  type=int,  default=10,
                        help="number of episodes to run (default: 10)")
    parser.add_argument("--phase",       type=int,  default=3, choices=[1, 2, 3],
                        help="difficulty: 1=easy, 2=medium, 3=full/training (default: 3)")
    parser.add_argument("--stochastic",  action="store_true",
                        help="use stochastic policy (default: deterministic)")
    args = parser.parse_args()

    deterministic = not args.stochastic
    settings      = PHASE_SETTINGS[args.phase]

    print(f"Checkpoint : {args.checkpoint}")
    difficulty = {1: "easy", 2: "medium", 3: "full (training distribution)"}
    print(f"Difficulty : {args.phase} ({difficulty[args.phase]})  —  "
          f"{settings['_n_waypoints_min']}–{settings['_n_waypoints_max']} waypoints, "
          f"max {settings['_wp_distance_max']} m apart")
    print(f"Policy     : {'deterministic' if deterministic else 'stochastic'}")
    print(f"Episodes   : {args.n_episodes}")
    print()

    model = SAC.load(args.checkpoint, device="cpu")

    env = SwerveEnv()
    for k, v in settings.items():
        setattr(env, k, v)

    from lib.renderer import Renderer
    renderer = Renderer(waypoints=None)

    rewards    = []
    completions = 0

    for ep in range(1, args.n_episodes + 1):
        ep_reward, steps, completed, quit_req = run_episode(
            model, env, renderer, ep, deterministic
        )

        result = "COMPLETE" if completed else "TIMEOUT "
        wp_done = env._tracker.current_idx - 1   # -1 because index 0 is the spawn point
        wp_total = len(env._waypoints) - 1
        print(f"  Ep {ep:3d}: {result}  steps={steps:4d}  "
              f"reward={ep_reward:7.1f}  waypoints={wp_done}/{wp_total}")

        rewards.append(ep_reward)
        if completed:
            completions += 1

        if quit_req:
            print("\nWindow closed — stopping early.")
            break

    if rewards:
        n = len(rewards)
        print(f"\n{'─'*52}")
        print(f"  Episodes run : {n}")
        print(f"  Completed    : {completions}/{n}  ({100*completions//n}%)")
        print(f"  Mean reward  : {sum(rewards)/n:.1f}")
        print(f"  Max reward   : {max(rewards):.1f}")
        print(f"  Min reward   : {min(rewards):.1f}")
        print(f"{'─'*52}")

    renderer.close()
    env.close()


if __name__ == "__main__":
    main()
