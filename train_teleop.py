# ──────────────────────────────────────────────────────────────────────────────
# train_teleop.py
# SAC training script for the teleop-assist policy.
#
# Usage:
#   python train_teleop.py                       # train silently
#   python train_teleop.py --render-eval         # pop a window every 20k steps
#   python train_teleop.py --resume teleop_assist/checkpoints/teleop_100000_steps.zip
#
# Outputs:
#   teleop_assist/checkpoints/   model snapshots every CHECKPOINT_FREQ steps
#   teleop_assist/logs/          rewards CSV for plotting
#
# Machine: Ryzen 9 9950X (16c/32t) + RTX 5080 16 GB — CUDA 13 / torch 2.12
# ──────────────────────────────────────────────────────────────────────────────

import os
import csv
import argparse
from datetime import datetime

import numpy as np
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv

from teleop_assist.env import TeleopAssistEnv
from lib.field_constants import MAX_SPEED_MPS, MAX_ANGULAR_RPS

# ── Parallelism ────────────────────────────────────────────────────────────────
# Ryzen 9 9950X has 16 physical cores / 32 threads.
# 32 SubprocVecEnv workers saturates the physical cores; SubprocVecEnv beats
# DummyVecEnv by ~35% at this env size (measured in bench_teleop.py).
N_ENVS = 32

# ── Training hyperparameters ───────────────────────────────────────────────────

TOTAL_TIMESTEPS   = 5_000_000
CHECKPOINT_FREQ   = 10_000      # total timesteps between checkpoints
EVAL_FREQ_DEFAULT = 20_000
LOG_DIR           = "teleop_assist/logs"
CHECKPOINT_DIR    = "teleop_assist/checkpoints"

EVAL_STEPS = [
    1_000, 5_000, 10_000, 25_000, 50_000,
    100_000, 200_000, 500_000, 1_000_000, 2_000_000, 3_000_000, 5_000_000,
]

SAC_KWARGS = dict(
    policy          = "MlpPolicy",
    learning_rate   = 3e-4,
    buffer_size     = 1_000_000,
    learning_starts = 5_000,
    # batch_size=4096 is nearly free vs 512 — GPU is underutilised per step;
    # the bottleneck is CPU-side SB3 Python overhead, not GPU compute.
    batch_size      = 4096,
    tau             = 0.005,
    gamma           = 0.99,
    train_freq      = 1,
    # gradient_steps=8 (not -1=32) — 8 updates per outer step instead of 32.
    # Each step costs ~47ms of CPU overhead regardless of batch size, so doing
    # 32 steps per outer step (940ms) is what made the last run take 2h50m.
    # 8 steps × 47ms = 376ms/outer-step. With batch_size 8× larger the total
    # sample exposure is similar; fewer steps = ~4× faster wall-clock.
    gradient_steps  = 4,
    policy_kwargs   = dict(net_arch=[256, 256]),
    verbose         = 1,
    device          = "cuda",     # RTX 5080, CUDA 13, 16 GB VRAM
)


# ── Reward logger ──────────────────────────────────────────────────────────────

class RewardLogger(BaseCallback):
    """Writes (timestep, episode_reward) rows to CSV — handles multi-env."""

    def __init__(self, log_path: str):
        super().__init__()
        self._path       = log_path
        self._ep_rewards = None   # per-env accumulators, shape (n_envs,)
        self._f          = None
        self._writer     = None

    def _on_training_start(self):
        os.makedirs(os.path.dirname(self._path), exist_ok=True)
        self._f      = open(self._path, "w", newline="")
        self._writer = csv.writer(self._f)
        self._writer.writerow(["timestep", "episode_reward"])
        self._ep_rewards = np.zeros(self.training_env.num_envs)

    def _on_step(self) -> bool:
        rewards = self.locals.get("rewards", np.zeros(self.training_env.num_envs))
        dones   = self.locals.get("dones",   np.zeros(self.training_env.num_envs, dtype=bool))
        self._ep_rewards += rewards
        for i, done in enumerate(dones):
            if done:
                self._writer.writerow([self.num_timesteps, round(float(self._ep_rewards[i]), 4)])
                self._f.flush()
                self._ep_rewards[i] = 0.0
        return True

    def _on_training_end(self):
        if self._f:
            self._f.close()


# ── Breakdown logger (prints reward components at milestone steps) ─────────────

class BreakdownLogger(BaseCallback):
    """Runs one deterministic episode at each milestone and prints reward breakdown."""

    def __init__(self, eval_steps=None):
        super().__init__()
        self._targets = sorted(eval_steps or EVAL_STEPS)

    def _on_step(self) -> bool:
        if self._targets and self.num_timesteps >= self._targets[0]:
            self._targets.pop(0)
            self._run_breakdown()
        return True

    def _run_breakdown(self):
        env    = TeleopAssistEnv()
        obs, _ = env.reset()
        done   = False

        totals = {
            "r_match": 0.0, "r_still": 0.0,
            "r_dir": 0.0, "r_smooth": 0.0, "collision": 0,
        }
        steps = 0

        while not done:
            action, _ = self.model.predict(obs, deterministic=True)
            obs, _, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            steps += 1
            for k in ("r_match", "r_still", "r_dir", "r_smooth"):
                totals[k] += info.get(k, 0.0)
            if info.get("collision"):
                totals["collision"] += 1

        env.close()
        print(
            f"\n[breakdown @ {self.num_timesteps:>7,}]  steps={steps}  "
            f"match={totals['r_match']:+.1f}  still={totals['r_still']:+.1f}  "
            f"dir={totals['r_dir']:+.1f}  smooth={totals['r_smooth']:+.1f}  "
            f"collisions={totals['collision']}"
        )


# ── Render-eval callback ───────────────────────────────────────────────────────

class RenderEvalCallback(BaseCallback):
    """Opens a Pygame window for one deterministic episode every eval_freq steps."""

    def __init__(self, eval_freq: int):
        super().__init__()
        self._eval_freq = eval_freq
        self._last_eval = 0

    def _on_step(self) -> bool:
        if self.num_timesteps - self._last_eval >= self._eval_freq:
            self._last_eval = self.num_timesteps
            self._run_rendered_episode()
        return True

    def _run_rendered_episode(self):
        import pygame
        from lib.renderer import Renderer

        print(f"\n[Render eval @ step {self.num_timesteps:,}]")

        env      = TeleopAssistEnv()
        renderer = Renderer()
        obs, _   = env.reset()
        done     = False
        ep_reward = 0.0
        step      = 0

        for _ in range(5):
            pygame.event.pump()
            renderer.draw(env._robot, None, env._robot.module_states)

        while not done:
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
                "joy_mag":    round(info.get("joy_mag", 0.0), 2),
                "approach":   round(info.get("r_approach", 0.0), 2),
            }
            renderer.draw(env._robot, None, env._robot.module_states, info=hud)

        status = "COLLISION" if terminated else "TIMEOUT"
        print(f"  {status}  steps={step}  reward={ep_reward:.2f}")

        renderer.close()
        env.close()


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to a checkpoint .zip to resume from")
    parser.add_argument("--steps", type=int, default=TOTAL_TIMESTEPS)
    parser.add_argument("--render-eval", action="store_true",
                        help="Open a Pygame window for one eval episode every --eval-freq steps")
    parser.add_argument("--eval-freq", type=int, default=EVAL_FREQ_DEFAULT)
    args = parser.parse_args()

    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)

    timestamp  = datetime.now().strftime("%Y%m%d_%H%M")
    reward_csv = os.path.join(LOG_DIR, f"rewards_{timestamp}.csv")
    print(f"Logging rewards to: {reward_csv}")

    env = make_vec_env(TeleopAssistEnv, n_envs=N_ENVS, vec_env_cls=SubprocVecEnv)

    # CheckpointCallback counts outer steps, so divide by N_ENVS to hit
    # CHECKPOINT_FREQ total timesteps between saves.
    checkpoint_cb = CheckpointCallback(
        save_freq   = max(1, CHECKPOINT_FREQ // N_ENVS),
        save_path   = CHECKPOINT_DIR,
        name_prefix = "teleop",
        verbose     = 1,
    )
    callbacks = [checkpoint_cb, RewardLogger(reward_csv), BreakdownLogger()]

    if args.render_eval:
        callbacks.append(RenderEvalCallback(eval_freq=args.eval_freq))
        print(f"Render-eval ON: window opens/closes every {args.eval_freq:,} steps.")

    if args.resume:
        print(f"Resuming from: {args.resume}")
        resume_kwargs = {k: v for k, v in SAC_KWARGS.items()
                         if k not in ("verbose", "policy_kwargs")}
        model = SAC.load(args.resume, env=env, **resume_kwargs)
        model.verbose = 1
    else:
        model = SAC(env=env, **SAC_KWARGS)

    print(f"\nStarting teleop-assist training for {args.steps:,} timesteps.")
    print(f"Device: {SAC_KWARGS['device']}  |  Parallel envs: {N_ENVS}")
    print(f"Checkpoints saved every {CHECKPOINT_FREQ:,} steps to {CHECKPOINT_DIR}/")
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

    final_path = os.path.join(CHECKPOINT_DIR, "teleop_final.zip")
    model.save(final_path)
    print(f"\nFinal model saved to: {final_path}")
    env.close()


if __name__ == "__main__":
    main()
