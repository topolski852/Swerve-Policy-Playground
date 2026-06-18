# ──────────────────────────────────────────────────────────────────────────────
# train.py
# SAC training script. Run this to train the swerve path-following policy.
#
# Usage:
#   python train.py                              # train silently
#   python train.py --render-eval                # pop a window every 20k steps
#   python train.py --render-eval --eval-freq 10000   # more frequent eval window
#   python train.py --resume checkpoints/swerve_100000_steps.zip
#
# Outputs:
#   checkpoints/swerve_<N>_steps.zip         model snapshots every CHECKPOINT_FREQ steps
#   logs/rewards.csv                         timestep, episode reward for plotting
# ──────────────────────────────────────────────────────────────────────────────

import os
import csv
import argparse
import numpy as np
from datetime import datetime

from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback

from swerve_env import SwerveEnv

# ── Training hyperparameters ───────────────────────────────────────────────────

TOTAL_TIMESTEPS   = 500_000
CHECKPOINT_FREQ   = 10_000      # save a .zip every N steps
EVAL_FREQ_DEFAULT = 20_000      # render an eval episode every N steps (--render-eval)
LOG_DIR           = "logs"
CHECKPOINT_DIR    = "checkpoints"

SAC_KWARGS = dict(
    policy         = "MlpPolicy",
    learning_rate  = 3e-4,
    buffer_size    = 200_000,
    learning_starts= 5_000,
    batch_size     = 256,
    tau            = 0.005,
    gamma          = 0.99,
    train_freq     = 1,
    gradient_steps = 1,
    policy_kwargs  = dict(net_arch=[256, 256]),
    verbose        = 1,
    device         = "cpu",
)


# ── Render-eval callback ───────────────────────────────────────────────────────

class RenderEvalCallback(BaseCallback):
    """
    Every eval_freq training steps, opens a Pygame window, runs one
    deterministic episode, then CLOSES the window immediately after.

    The window only exists for the duration of the episode (~20 s).
    It never sits frozen between evals, which caused the Not Responding crash.
    """

    def __init__(self, eval_freq: int):
        super().__init__()
        self._eval_freq  = eval_freq
        self._last_eval  = 0

    def _on_step(self) -> bool:
        if self.num_timesteps - self._last_eval >= self._eval_freq:
            self._last_eval = self.num_timesteps
            self._run_rendered_episode()
        return True

    def _run_rendered_episode(self):
        import pygame
        from renderer import Renderer

        print(f"\n[Render eval @ step {self.num_timesteps:,}]")

        env      = SwerveEnv()
        renderer = Renderer()
        obs, _   = env.reset()
        done     = False
        ep_reward = 0.0
        step      = 0

        while not done:
            # Pump events so Windows doesn't mark the window as frozen
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    done = True
                    break

            if done:
                break

            action, _ = self.model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            ep_reward += reward
            done = terminated or truncated
            step += 1

            hud = {
                "train_step": self.num_timesteps,
                "eval_step":  step,
                "reward":     round(ep_reward, 2),
                "progress%":  round(info.get("arc_pos", 0) /
                                    __import__("field_path").TOTAL_LENGTH * 100, 1),
            }
            renderer.draw(env._robot, env._tracker, env._get_module_states(), info=hud)

        status = "SUCCESS" if terminated else ("QUIT" if step == 0 else "TIMEOUT/OFF-PATH")
        print(f"  {status}  steps={step}  reward={ep_reward:.2f}")

        # Close immediately — no frozen window left behind
        renderer.close()
        env.close()


# ── Reward logger callback ─────────────────────────────────────────────────────

class RewardLogger(BaseCallback):
    """Writes (timestep, episode_reward) rows to logs/rewards.csv."""

    def __init__(self, log_path: str):
        super().__init__()
        self._path = log_path
        self._ep_reward = 0.0
        self._f   = None
        self._writer = None

    def _on_training_start(self):
        os.makedirs(os.path.dirname(self._path), exist_ok=True)
        self._f      = open(self._path, "w", newline="")
        self._writer = csv.writer(self._f)
        self._writer.writerow(["timestep", "episode_reward"])

    def _on_step(self) -> bool:
        reward = self.locals.get("rewards", [0])[0]
        done   = self.locals.get("dones",   [False])[0]
        self._ep_reward += reward
        if done:
            self._writer.writerow([self.num_timesteps, round(self._ep_reward, 4)])
            self._f.flush()
            self._ep_reward = 0.0
        return True

    def _on_training_end(self):
        if self._f:
            self._f.close()


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to a checkpoint .zip to resume from")
    parser.add_argument("--steps", type=int, default=TOTAL_TIMESTEPS)
    parser.add_argument("--render-eval", action="store_true",
                        help="Open a Pygame window for one eval episode every --eval-freq steps")
    parser.add_argument("--eval-freq", type=int, default=EVAL_FREQ_DEFAULT,
                        help="How often (in training steps) to run a rendered eval episode")
    args = parser.parse_args()

    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    reward_csv = os.path.join(LOG_DIR, f"rewards_{timestamp}.csv")
    print(f"Logging rewards to: {reward_csv}")

    env = SwerveEnv()

    checkpoint_cb = CheckpointCallback(
        save_freq   = CHECKPOINT_FREQ,
        save_path   = CHECKPOINT_DIR,
        name_prefix = "swerve",
        verbose     = 1,
    )
    reward_cb = RewardLogger(reward_csv)

    callbacks = [checkpoint_cb, reward_cb]

    if args.render_eval:
        callbacks.append(RenderEvalCallback(eval_freq=args.eval_freq))
        print(f"Render-eval ON: window opens/closes every {args.eval_freq:,} steps.")

    if args.resume:
        print(f"Resuming from: {args.resume}")
        model = SAC.load(args.resume, env=env, **{k: v for k, v in SAC_KWARGS.items()
                                                   if k != "verbose"})
        model.verbose = 1
    else:
        model = SAC(env=env, **SAC_KWARGS)

    print(f"\nStarting training for {args.steps:,} timesteps on CPU.")
    print("Checkpoints saved every", CHECKPOINT_FREQ, "steps to", CHECKPOINT_DIR)
    print("Press Ctrl+C to stop early — latest checkpoint is kept.\n")

    try:
        model.learn(
            total_timesteps     = args.steps,
            callback            = callbacks,
            reset_num_timesteps = (args.resume is None),
            progress_bar        = True,
        )
    except KeyboardInterrupt:
        print("\nTraining interrupted by user.")

    final_path = os.path.join(CHECKPOINT_DIR, "swerve_final.zip")
    model.save(final_path)
    print(f"\nFinal model saved to: {final_path}")
    env.close()


if __name__ == "__main__":
    main()
