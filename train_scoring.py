# ──────────────────────────────────────────────────────────────────────────────
# train_scoring.py
# SAC training script for the fuel scoring experiment.
#
# Usage:
#   python train_scoring.py                       # train silently
#   python train_scoring.py --render-eval          # pop a window every 20k steps
#   python train_scoring.py --render-eval --eval-freq 10000
#   python train_scoring.py --resume fuel_scoring/checkpoints/scoring_100000_steps.zip
#
# Outputs:
#   fuel_scoring/checkpoints/   model snapshots every CHECKPOINT_FREQ steps
#   fuel_scoring/logs/          rewards CSV for plotting
#   fuel_scoring/recordings/    MP4s at RECORD_STEPS (with --render-capture)
# ──────────────────────────────────────────────────────────────────────────────

import os
import csv
import argparse
from datetime import datetime

from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback

from fuel_scoring.swerve_env import SwerveEnv

# ── Training hyperparameters ───────────────────────────────────────────────────

TOTAL_TIMESTEPS   = 2_000_000
CHECKPOINT_FREQ   = 10_000
EVAL_FREQ_DEFAULT = 20_000
LOG_DIR           = "fuel_scoring/logs"
CHECKPOINT_DIR    = "fuel_scoring/checkpoints"
RECORDINGS_DIR    = "fuel_scoring/recordings"

RECORD_STEPS = [
    500, 1_000, 2_000,
    5_000, 8_000, 12_000,
    18_000, 25_000, 35_000, 50_000,
    75_000, 100_000, 150_000,
    200_000, 300_000, 500_000,
    750_000, 1_000_000, 1_500_000, 2_000_000,
]

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

        env      = SwerveEnv()
        renderer = Renderer(waypoints=None)
        obs, _   = env.reset()
        done     = False
        ep_reward = 0.0
        step      = 0

        for _ in range(5):
            pygame.event.pump()
            renderer.draw(env._robot, None, env._get_module_states())

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
                "train_step":  self.num_timesteps,
                "eval_step":   step,
                "reward":      round(ep_reward, 2),
                "hopper":      round(info.get("hopper_level", 0) * 60, 1),
                "fuel_scored": round(info.get("fuel_scored", 0), 2),
            }
            renderer.draw(env._robot, None, env._get_module_states(), info=hud)

        status = "SUCCESS" if terminated else ("QUIT" if step == 0 else "TIMEOUT")
        print(f"  {status}  steps={step}  reward={ep_reward:.2f}")

        renderer.close()
        env.close()


# ── Recording callback ────────────────────────────────────────────────────────

class RecordEvalCallback(BaseCallback):
    """Records one deterministic episode as MP4 at each step in RECORD_STEPS."""

    def __init__(self, record_steps=None, recordings_dir=RECORDINGS_DIR):
        super().__init__()
        self._targets = sorted(record_steps or RECORD_STEPS)
        self._dir = recordings_dir
        os.makedirs(recordings_dir, exist_ok=True)

    def _on_step(self) -> bool:
        if self._targets and self.num_timesteps >= self._targets[0]:
            target = self._targets.pop(0)
            self._record_episode(target)
        return True

    def _record_episode(self, step_count):
        import pygame
        from lib.renderer import Renderer

        path = os.path.join(self._dir, f"eval_{step_count:07d}_steps.mp4")
        print(f"\n[Recording @ step {self.num_timesteps:,}] -> {path}")

        env      = SwerveEnv()
        renderer = Renderer(waypoints=None, record_path=path)
        obs, _   = env.reset()
        done     = False
        ep_reward = 0.0
        step      = 0

        for _ in range(5):
            pygame.event.pump()
            renderer.draw(env._robot, None, env._get_module_states())

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
                "train_step":  step_count,
                "eval_step":   step,
                "reward":      round(ep_reward, 2),
                "hopper":      round(info.get("hopper_level", 0) * 60, 1),
                "fuel_scored": round(info.get("fuel_scored", 0), 2),
            }
            renderer.draw(env._robot, None, env._get_module_states(), info=hud)

        status = "SUCCESS" if terminated else "TIMEOUT"
        print(f"  {status}  frames={step}  reward={ep_reward:.2f}  saved: {path}")

        renderer.close()
        env.close()


# ── Reward logger callback ─────────────────────────────────────────────────────

class RewardLogger(BaseCallback):
    """Writes (timestep, episode_reward) rows to a CSV file."""

    def __init__(self, log_path: str):
        super().__init__()
        self._path = log_path
        self._ep_reward = 0.0
        self._f      = None
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
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--steps", type=int, default=TOTAL_TIMESTEPS)
    parser.add_argument("--render-eval", action="store_true")
    parser.add_argument("--eval-freq", type=int, default=EVAL_FREQ_DEFAULT)
    parser.add_argument("--render-capture", action="store_true")
    args = parser.parse_args()

    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)

    timestamp  = datetime.now().strftime("%Y%m%d_%H%M")
    reward_csv = os.path.join(LOG_DIR, f"rewards_{timestamp}.csv")
    print(f"Logging rewards to: {reward_csv}")

    env = SwerveEnv()

    checkpoint_cb = CheckpointCallback(
        save_freq   = CHECKPOINT_FREQ,
        save_path   = CHECKPOINT_DIR,
        name_prefix = "scoring",
        verbose     = 1,
    )
    reward_cb = RewardLogger(reward_csv)
    callbacks = [checkpoint_cb, reward_cb]

    if args.render_eval:
        callbacks.append(RenderEvalCallback(eval_freq=args.eval_freq))
        print(f"Render-eval ON: window opens/closes every {args.eval_freq:,} steps.")

    if args.render_capture:
        rec_dir = os.path.join(RECORDINGS_DIR, f"run_{timestamp}")
        callbacks.append(RecordEvalCallback(recordings_dir=rec_dir))
        print(f"Render-capture ON: MP4s will be saved to {rec_dir}/")

    if args.resume:
        print(f"Resuming from: {args.resume}")
        resume_kwargs = {k: v for k, v in SAC_KWARGS.items()
                         if k not in ("verbose", "policy_kwargs")}
        model = SAC.load(args.resume, env=env, **resume_kwargs)
        model.verbose = 1
    else:
        model = SAC(env=env, **SAC_KWARGS)

    print(f"\nStarting fuel scoring training for {args.steps:,} timesteps on CPU.")
    print(f"Checkpoints saved every {CHECKPOINT_FREQ} steps to {CHECKPOINT_DIR}/")
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

    final_path = os.path.join(CHECKPOINT_DIR, "scoring_final.zip")
    model.save(final_path)
    print(f"\nFinal model saved to: {final_path}")
    env.close()


if __name__ == "__main__":
    main()
