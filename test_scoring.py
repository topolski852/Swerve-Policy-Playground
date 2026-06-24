# ──────────────────────────────────────────────────────────────────────────────
# test_scoring.py
# Interactive model tester for the fuel_scoring experiment.
#
# Usage:
#   python test_scoring.py                         # final model, 10 episodes
#   python test_scoring.py --n-episodes 25
#   python test_scoring.py --checkpoint fuel_scoring/checkpoints/scoring_500000_steps.zip
#   python test_scoring.py --stochastic            # stochastic policy instead of deterministic
#   python test_scoring.py --random-start          # random spawn instead of fixed match start
#
# Controls:
#   Close the window or press Q to stop early.
# ──────────────────────────────────────────────────────────────────────────────

import argparse
import pygame
from stable_baselines3 import SAC
from fuel_scoring.swerve_env import SwerveEnv, HUB_DIST_NORM_MAX
from fuel_scoring.constants import MAX_EPISODE_STEPS, AUTO_PERIOD_STEPS

DEFAULT_CHECKPOINT = "fuel_scoring/checkpoints/scoring_final.zip"

MILESTONES = [
    (900, "MILESTONE_900"),
    (600, "MILESTONE_600"),
    (360, "MILESTONE_360"),
    (100, "MILESTONE_100"),
]


def peak_milestone(fuel_scored):
    for threshold, label in MILESTONES:
        if fuel_scored >= threshold:
            return label
    return "none"


def run_episode(model, env, renderer, episode_num, deterministic):
    """Run one episode. Returns (ep_reward, steps, collision, fuel_scored, quit_requested)."""
    obs, _ = env.reset()

    for _ in range(5):
        pygame.event.pump()
        renderer.draw(env._robot, None, env._get_module_states())

    ep_reward    = 0.0
    total_scored = 0.0
    step         = 0
    collision    = False

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return ep_reward, step, collision, total_scored, True
            if event.type == pygame.KEYDOWN and event.key == pygame.K_q:
                return ep_reward, step, collision, total_scored, True

        action, _ = model.predict(obs, deterministic=deterministic)
        obs, reward, terminated, truncated, info = env.step(action)
        ep_reward    += reward
        total_scored += info.get("fuel_scored", 0.0)
        step         += 1

        if info.get("collision"):
            collision = True

        hub_raw = float(obs[7])
        period  = "AUTO" if step <= AUTO_PERIOD_STEPS else "TELEOP"
        hud = {
            "episode":      episode_num,
            "match_time":   env.match_time_str(),
            "period":       period,
            "reward":       round(ep_reward, 2),
            "hopper":       round(info.get("hopper_level", 0) * 60, 1),
            "total_scored": round(total_scored, 1),
            "hub_dist":     "neutral" if hub_raw < 0 else f"{hub_raw * HUB_DIST_NORM_MAX:.1f}m",
        }
        renderer.draw(env._robot, None, env._get_module_states(), info=hud)

        if terminated or truncated:
            for _ in range(30):
                pygame.event.pump()
                renderer.draw(env._robot, None, env._get_module_states(), info=hud)
            break

    return ep_reward, step, collision, total_scored, False


def main():
    parser = argparse.ArgumentParser(description="Test a trained fuel_scoring model with live rendering.")
    parser.add_argument("--checkpoint",   type=str,  default=DEFAULT_CHECKPOINT,
                        help=f"model zip to load (default: {DEFAULT_CHECKPOINT})")
    parser.add_argument("--n-episodes",   type=int,  default=10,
                        help="number of episodes to run (default: 10)")
    parser.add_argument("--stochastic",   action="store_true",
                        help="use stochastic policy (default: deterministic)")
    parser.add_argument("--random-start", action="store_true",
                        help="randomise spawn position (default: fixed match start)")
    args = parser.parse_args()

    deterministic = not args.stochastic

    print(f"Checkpoint   : {args.checkpoint}")
    print(f"Policy       : {'deterministic' if deterministic else 'stochastic'}")
    print(f"Start        : {'random' if args.random_start else 'fixed (3.50, 2.55)'}")
    print(f"Episode len  : {MAX_EPISODE_STEPS} steps (full 2:40 match)")
    print(f"Episodes     : {args.n_episodes}")
    print()

    model = SAC.load(args.checkpoint, device="cpu")

    env = SwerveEnv(random_start=args.random_start)
    env._max_episode_steps = MAX_EPISODE_STEPS

    from lib.renderer import Renderer
    renderer = Renderer(waypoints=None)

    rewards      = []
    fuels        = []
    collisions   = 0

    for ep in range(1, args.n_episodes + 1):
        ep_reward, steps, had_collision, fuel_scored, quit_req = run_episode(
            model, env, renderer, ep, deterministic
        )

        result = "COLLISION" if had_collision else "COMPLETE "
        peak   = peak_milestone(fuel_scored)
        print(f"  Ep {ep:3d}: {result}  steps={steps:4d}  "
              f"reward={ep_reward:8.1f}  fuel={fuel_scored:6.1f}  peak={peak}")

        rewards.append(ep_reward)
        fuels.append(fuel_scored)
        if had_collision:
            collisions += 1

        if quit_req:
            print("\nWindow closed — stopping early.")
            break

    if rewards:
        n = len(rewards)
        print(f"\n{'─'*60}")
        print(f"  Episodes run  : {n}")
        print(f"  Collisions    : {collisions}/{n}  ({100*collisions//n}%)")
        print(f"  Mean reward   : {sum(rewards)/n:.1f}")
        print(f"  Max reward    : {max(rewards):.1f}")
        print(f"  Min reward    : {min(rewards):.1f}")
        print(f"  Mean fuel     : {sum(fuels)/n:.1f}")
        print(f"  Max fuel      : {max(fuels):.1f}")
        ms_counts = {label: sum(1 for f in fuels if f >= t) for t, label in MILESTONES}
        for label, count in ms_counts.items():
            print(f"  {label:<18}: {count}/{n}  ({100*count//n}%)")
        print(f"{'─'*60}")

    renderer.close()
    env.close()


if __name__ == "__main__":
    main()
