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
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv

from fuel_scoring.swerve_env import SwerveEnv, HUB_DIST_NORM_MAX
from fuel_scoring.constants import MAX_EPISODE_STEPS, AUTO_PERIOD_STEPS

# ── Training hyperparameters ───────────────────────────────────────────────────

TOTAL_TIMESTEPS       = 5_000_000
CHECKPOINT_FREQ       = 10_000
EVAL_FREQ_DEFAULT     = 20_000
N_ENVS                = 4
PHASE2_TIMESTEPS      = 250_000   # switch from random starts → fixed match start
ENTROPY_SCHEDULE      = [
    (750_000,  -1.5),   # phase 3: tighten — less random wandering
    (1_500_000, -0.5),  # phase 4: exploit — execute consistently
]
COLLISION_SCHEDULE    = [
    (750_000,  -200.0), # ramp collision penalty once agent knows how to score
]
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
    buffer_size    = 400_000,
    learning_starts= 20_000,
    batch_size     = 512,
    tau            = 0.005,
    gamma          = 0.99,
    train_freq     = 1,
    gradient_steps = -1,
    policy_kwargs  = dict(net_arch=[256, 256]),
    verbose        = 1,
    device         = "cuda",
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

        env           = SwerveEnv()
        env._max_episode_steps = MAX_EPISODE_STEPS
        renderer      = Renderer(waypoints=None)
        obs, _        = env.reset()
        done          = False
        ep_reward     = 0.0
        total_scored  = 0.0
        step          = 0

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
            ep_reward    += reward
            total_scored += info.get("fuel_scored", 0)
            done          = terminated or truncated
            step         += 1

            hub_raw = float(obs[7])
            period  = "AUTO" if step <= AUTO_PERIOD_STEPS else "TELEOP"
            hud = {
                "match_time":   env.match_time_str(),
                "period":       period,
                "train_step":   self.num_timesteps,
                "reward":       round(ep_reward, 2),
                "hopper":       round(info.get("hopper_level", 0) * 60, 1),
                "total_scored": round(total_scored, 1),
                "hub_dist":     "neutral" if hub_raw < 0 else f"{hub_raw * HUB_DIST_NORM_MAX:.1f}m",
            }
            renderer.draw(env._robot, None, env._get_module_states(), info=hud)

        status = "SUCCESS" if terminated else ("QUIT" if step == 0 else "TIMEOUT")
        print(f"  {status}  steps={step}  reward={ep_reward:.2f}")

        renderer.close()
        env.close()


# ── Entropy annealing callback ────────────────────────────────────────────────

class EntropyAnnealCallback(BaseCallback):
    """
    Lowers SAC's target_entropy at scheduled timesteps, shifting the policy
    from exploration toward exploitation without resetting weights or buffer.

    Schedule: list of (timestep, target_entropy) pairs in ascending order.
    Default (auto) target is -dim(action) = -3 for this env.
    """

    def __init__(self, schedule=ENTROPY_SCHEDULE):
        super().__init__()
        self._schedule = sorted(schedule, key=lambda x: x[0])
        self._idx = 0

    def _on_step(self) -> bool:
        while self._idx < len(self._schedule):
            ts, target = self._schedule[self._idx]
            if self.num_timesteps >= ts:
                self.model.target_entropy = float(target)
                self._idx += 1
                print(f"\n[Entropy phase @ step {self.num_timesteps:,}]  target_entropy → {target}")
            else:
                break
        return True


# ── Collision penalty ramp callback ───────────────────────────────────────────

class CollisionRampCallback(BaseCallback):
    """
    Starts with a small collision penalty so early exploration isn't catastrophic,
    then ramps to the full penalty once the agent has learned to score.
    Uses set_attr to push the new value into all SubprocVecEnv workers.
    """

    def __init__(self, schedule=COLLISION_SCHEDULE):
        super().__init__()
        self._schedule = sorted(schedule, key=lambda x: x[0])
        self._idx = 0

    def _on_step(self) -> bool:
        while self._idx < len(self._schedule):
            ts, penalty = self._schedule[self._idx]
            if self.num_timesteps >= ts:
                self.training_env.set_attr("_collision_penalty", penalty)
                self._idx += 1
                print(f"\n[Collision ramp @ step {self.num_timesteps:,}]  collision penalty → {penalty}")
            else:
                break
        return True


# ── Phase curriculum callback ─────────────────────────────────────────────────

class PhaseCallback(BaseCallback):
    """
    Phase 1 (0 → PHASE2_TIMESTEPS): random starts, random hopper — learn the general loop.
    Phase 2 (PHASE2_TIMESTEPS → end): fixed match start (3.50, 2.55), hopper=8 — specialise.
    """

    def __init__(self, switch_at: int = PHASE2_TIMESTEPS):
        super().__init__()
        self._switch_at = switch_at
        self._switched  = False

    def _on_step(self) -> bool:
        if not self._switched and self.num_timesteps >= self._switch_at:
            self.training_env.set_attr("_random_start", False)
            self.training_env.set_attr("_max_episode_steps", MAX_EPISODE_STEPS)
            self._switched = True
            print(f"\n[Phase 2 @ step {self.num_timesteps:,}]  Fixed start (3.50, 2.55), hopper=8, episodes → {MAX_EPISODE_STEPS} steps")
        return True


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

        env          = SwerveEnv()
        env._max_episode_steps = MAX_EPISODE_STEPS
        renderer     = Renderer(waypoints=None, record_path=path)
        obs, _       = env.reset()
        done         = False
        ep_reward    = 0.0
        total_scored = 0.0
        step         = 0

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
            ep_reward    += reward
            total_scored += info.get("fuel_scored", 0)
            done          = terminated or truncated
            step         += 1

            hub_raw = float(obs[7])
            period  = "AUTO" if step <= AUTO_PERIOD_STEPS else "TELEOP"
            hud = {
                "match_time":   env.match_time_str(),
                "period":       period,
                "train_step":   step_count,
                "reward":       round(ep_reward, 2),
                "hopper":       round(info.get("hopper_level", 0) * 60, 1),
                "total_scored": round(total_scored, 1),
                "hub_dist":     "neutral" if hub_raw < 0 else f"{hub_raw * HUB_DIST_NORM_MAX:.1f}m",
            }
            renderer.draw(env._robot, None, env._get_module_states(), info=hud)

        status = "SUCCESS" if terminated else "TIMEOUT"
        print(f"  {status}  frames={step}  reward={ep_reward:.2f}  saved: {path}")

        renderer.close()
        env.close()


# ── Reward logger callback ─────────────────────────────────────────────────────

class RewardLogger(BaseCallback):
    """Writes (timestep, episode_reward) rows to a CSV file, one row per episode across all envs."""

    def __init__(self, log_path: str):
        super().__init__()
        self._path = log_path
        self._ep_rewards: list[float] = []
        self._f      = None
        self._writer = None

    def _on_training_start(self):
        os.makedirs(os.path.dirname(self._path), exist_ok=True)
        self._f      = open(self._path, "w", newline="")
        self._writer = csv.writer(self._f)
        self._writer.writerow(["timestep", "episode_reward"])
        self._ep_rewards = [0.0] * self.training_env.num_envs

    def _on_step(self) -> bool:
        rewards = self.locals.get("rewards", [])
        dones   = self.locals.get("dones",   [])
        for i, (r, d) in enumerate(zip(rewards, dones)):
            self._ep_rewards[i] += float(r)
            if d:
                self._writer.writerow([self.num_timesteps, round(self._ep_rewards[i], 4)])
                self._f.flush()
                self._ep_rewards[i] = 0.0
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

    env = make_vec_env(SwerveEnv, n_envs=N_ENVS, vec_env_cls=SubprocVecEnv,
                       env_kwargs=dict(random_start=True))

    checkpoint_cb = CheckpointCallback(
        save_freq   = max(CHECKPOINT_FREQ // N_ENVS, 1),
        save_path   = CHECKPOINT_DIR,
        name_prefix = "scoring",
        verbose     = 1,
    )
    reward_cb = RewardLogger(reward_csv)
    callbacks = [checkpoint_cb, reward_cb, PhaseCallback(), EntropyAnnealCallback(), CollisionRampCallback()]

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

    print(f"\nStarting fuel scoring training for {args.steps:,} timesteps  |  {N_ENVS} envs  |  device=cuda.")
    print(f"Phase 1: random starts, 2000-step episodes  (0 → {PHASE2_TIMESTEPS:,})")
    print(f"Phase 2: fixed start,   8000-step episodes  ({PHASE2_TIMESTEPS:,} → end)  [full 2:40 match]")
    for ts, ent in ENTROPY_SCHEDULE:
        print(f"  entropy → {ent} at step {ts:,}")
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

    final_path = os.path.join(CHECKPOINT_DIR, "scoring_final.zip")
    model.save(final_path)
    print(f"\nFinal model saved to: {final_path}")
    env.close()


if __name__ == "__main__":
    main()
