# ──────────────────────────────────────────────────────────────────────────────
# test_randomizer.py
# Interactive model tester for the path_randomizer experiment.
#
# Usage:
#   python test_randomizer.py                        # final model, Phase 6, 10 episodes
#   python test_randomizer.py --phase 1              # test on curriculum stage 1 (easiest)
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
from train_randomizer import episode_outcome

DEFAULT_CHECKPOINT = "path_randomizer/checkpoints/randomizer_final.zip"

# --phase picks a training curriculum stage (1 = easiest). The last is the full
# training distribution.
from path_randomizer.constants import CURRICULUM


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
    parser.add_argument("--phase",       type=int,  default=len(CURRICULUM),
                        choices=range(1, len(CURRICULUM) + 1),
                        help=f"curriculum stage 1-{len(CURRICULUM)}; {len(CURRICULUM)} = full (default)")
    parser.add_argument("--stochastic",  action="store_true",
                        help="use stochastic policy (default: deterministic)")
    args = parser.parse_args()

    deterministic = not args.stochastic
    settings      = CURRICULUM[args.phase - 1]

    print(f"Checkpoint : {args.checkpoint}")
    print(f"Difficulty : stage {args.phase} of {len(CURRICULUM)}  —  "
          f"{settings['n_min']}–{settings['n_max']} waypoints, max {settings['max_dist']} m apart, "
          f"{'clear legs only' if settings['clear_legs'] else 'legs may cross field elements'}")
    print(f"Policy     : {'deterministic' if deterministic else 'stochastic'}")
    print(f"Episodes   : {args.n_episodes}")
    print()

    model = SAC.load(args.checkpoint, device="cpu")

    env = SwerveEnv()
    env.set_stage(args.phase - 1)

    from lib.renderer import Renderer
    renderer = Renderer(waypoints=None)

    rewards    = []
    completions = 0

    for ep in range(1, args.n_episodes + 1):
        ep_reward, steps, completed, quit_req = run_episode(
            model, env, renderer, ep, deterministic
        )

        outcome = episode_outcome(env)
        completed = outcome == "COMPLETE"   # terminated is also True on a crash
        result = f"{outcome:8s}"
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
