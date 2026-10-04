# ──────────────────────────────────────────────────────────────────────────────
# train_randomizer.py
# SAC training script for goal-conditioned waypoint navigation.
#
# Usage:
#   python train_randomizer.py                          # single env, silent
#   python train_randomizer.py --n-envs 2               # 2 parallel envs (PC)
#   python train_randomizer.py --render-eval             # pop a window every 20k steps
#   python train_randomizer.py --render-eval --eval-freq 10000
#   python train_randomizer.py --resume path_randomizer/checkpoints/randomizer_100000_steps.zip
#
# Outputs:
#   path_randomizer/checkpoints/   model snapshots every CHECKPOINT_FREQ steps
#   path_randomizer/logs/          rewards CSV for plotting
#   path_randomizer/recordings/    MP4s at RECORD_STEPS (with --render-capture)
# ──────────────────────────────────────────────────────────────────────────────

import os
import csv
import argparse
from datetime import datetime

from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv

from path_randomizer.swerve_env import SwerveEnv
from path_randomizer.constants import (
    NODE_TIME_LIMIT_STEPS, MIN_WAYPOINT_DISTANCE,
    CURRICULUM, PROMOTE_COMPLETE_RATE, PROMOTE_WINDOW,
)

# ── Training hyperparameters ───────────────────────────────────────────────────

TOTAL_TIMESTEPS   = 7_000_000
CHECKPOINT_FREQ   = 10_000
EVAL_FREQ_DEFAULT = 20_000
N_ENVS            = 2
LOG_DIR           = "path_randomizer/logs"
CHECKPOINT_DIR    = "path_randomizer/checkpoints"
RECORDINGS_DIR    = "path_randomizer/recordings"


# Same milestones as train.py so path_following and randomizer clips line up side by side.
# Every milestone recording drives the same route (per curriculum stage) so the
# clips can be compared side by side. With a new random route each time, an easy
# route at 200k and a hard one at 300k looked like the policy had gotten worse.
# Seed 1507's stage-2 route: 10 nodes, 2 legs that must go around a field element.
RECORD_SEED = 1507

RECORD_STEPS = [
    500, 1_000, 2_000,
    5_000, 8_000, 12_000,
    18_000, 25_000, 35_000, 50_000,
    75_000, 100_000, 150_000,
    200_000, 300_000, 500_000,
    1_000_000, 2_000_000,
]

SAC_KWARGS = dict(
    policy          = "MlpPolicy",
    learning_rate   = 3e-4,
    buffer_size     = 1_000_000,
    learning_starts = 10_000,
    batch_size      = 256,
    tau             = 0.005,
    gamma           = 0.99,
    train_freq      = 1,
    gradient_steps  = 1,
    policy_kwargs   = dict(net_arch=[256, 256]),
    verbose         = 1,
    device          = "cpu",
)



def episode_outcome(env):
    """Label how an eval episode ended. terminated covers both finishing and
    crashing, so check the robot itself rather than the flags."""
    if env._check_collision():
        return "CRASH"
    if env._tracker.done:
        return "COMPLETE"
    if env._node_steps >= NODE_TIME_LIMIT_STEPS:
        return "TIMEOUT"   # gave up on a node after 5 s, like RouteRunner
    return "STOPPED"


# ── Curriculum callback ───────────────────────────────────────────────────────

class CurriculumCallback(BaseCallback):
    """Moves every training env up one CURRICULUM stage once
    PROMOTE_COMPLETE_RATE of the last PROMOTE_WINDOW episodes finished.
    Never moves back down. Logs the stage and outcome rates to TensorBoard/console."""

    def __init__(self, start_stage=0):
        super().__init__()
        self.stage    = start_stage
        self._recent  = []   # outcome strings, newest last

    def _on_training_start(self):
        self.training_env.env_method("set_stage", self.stage)
        print(f"Curriculum: starting at stage {self.stage} {CURRICULUM[self.stage]}")

    def _on_step(self) -> bool:
        for done, info in zip(self.locals["dones"], self.locals["infos"]):
            if done and "outcome" in info and info.get("stage") == self.stage:
                self._recent.append(info["outcome"])
        self._recent = self._recent[-PROMOTE_WINDOW:]

        if self._recent:
            n = len(self._recent)
            for k in ("complete", "crash", "timeout"):
                self.logger.record(f"curriculum/{k}_rate", self._recent.count(k) / n)
        self.logger.record("curriculum/stage", self.stage)

        if (self.stage < len(CURRICULUM) - 1 and len(self._recent) >= PROMOTE_WINDOW and
                self._recent.count("complete") / len(self._recent) >= PROMOTE_COMPLETE_RATE):
            self.stage += 1
            self._recent = []
            self.training_env.env_method("set_stage", self.stage)
            print(f"\n[Curriculum @ step {self.num_timesteps:,}] -> stage {self.stage} {CURRICULUM[self.stage]}")
        return True


# ── Render-eval callback ───────────────────────────────────────────────────────

class RenderEvalCallback(BaseCallback):
    """Opens a Pygame window for one deterministic episode every eval_freq steps."""

    def __init__(self, eval_freq: int, curriculum=None):
        super().__init__()
        self._eval_freq = eval_freq
        self._last_eval = 0
        self._curriculum = curriculum

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
        if self._curriculum is not None:
            env.set_stage(self._curriculum.stage)
        renderer = Renderer(waypoints=None)
        obs, _   = env.reset()
        renderer.set_waypoints(env._waypoints)

        done      = False
        ep_reward = 0.0
        step      = 0

        for _ in range(5):
            pygame.event.pump()
            renderer.draw(env._robot, env._tracker, env._get_module_states())

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
                "waypoint":   f"{info['waypoint_idx']}/{info['n_waypoints']}",
            }
            renderer.draw(env._robot, env._tracker, env._get_module_states(), info=hud)

        status = episode_outcome(env)
        print(f"  {status}  stage={env.stage}  steps={step}  reward={ep_reward:.2f}  "
              f"waypoints={env._tracker.current_idx}/{len(env._waypoints)}")

        renderer.close()
        env.close()


# ── Recording callback ────────────────────────────────────────────────────────

class RecordEvalCallback(BaseCallback):
    """Records one deterministic episode as MP4 at each step in RECORD_STEPS."""

    def __init__(self, record_steps=None, recordings_dir=RECORDINGS_DIR, curriculum=None):
        super().__init__()
        self._curriculum = curriculum
        self._targets = sorted(record_steps or RECORD_STEPS)
        self._dir = recordings_dir
        os.makedirs(recordings_dir, exist_ok=True)

    def _on_training_start(self):
        # A resumed run starts at its checkpoint's step count: skip milestones
        # already behind it instead of recording them all on the first step.
        self._targets = [t for t in self._targets if t > self.num_timesteps]

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
        if self._curriculum is not None:
            env.set_stage(self._curriculum.stage)
        renderer = Renderer(waypoints=None, record_path=path)
        obs, _   = env.reset(seed=RECORD_SEED)
        renderer.set_waypoints(env._waypoints)

        done      = False
        ep_reward = 0.0
        step      = 0

        for _ in range(5):
            pygame.event.pump()
            renderer.draw(env._robot, env._tracker, env._get_module_states())

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
                "train_step": step_count,
                "eval_step":  step,
                "reward":     round(ep_reward, 2),
                "waypoint":   f"{info['waypoint_idx']}/{info['n_waypoints']}",
            }
            renderer.draw(env._robot, env._tracker, env._get_module_states(), info=hud)

        status = episode_outcome(env)
        print(f"  {status}  stage={env.stage}  frames={step}  reward={ep_reward:.2f}  saved: {path}")

        renderer.close()
        env.close()


# ── Reward logger callback ─────────────────────────────────────────────────────

class RewardLogger(BaseCallback):
    """Writes (timestep, episode_reward) rows to a CSV file.
    Works with both single env and SubprocVecEnv (tracks each env separately)."""

    def __init__(self, log_path: str):
        super().__init__()
        self._path       = log_path
        self._ep_rewards = None   # list of floats, one per env — set on training start
        self._f          = None
        self._writer     = None

    def _on_training_start(self):
        n = self.training_env.num_envs
        self._ep_rewards = [0.0] * n
        os.makedirs(os.path.dirname(self._path), exist_ok=True)
        self._f      = open(self._path, "w", newline="")
        self._writer = csv.writer(self._f)
        self._writer.writerow(["timestep", "episode_reward", "stage"])

    def _on_step(self) -> bool:
        rewards = self.locals.get("rewards", [0])
        dones   = self.locals.get("dones",   [False])
        infos   = self.locals.get("infos",   [{}] * len(dones))
        for i, (r, done, info) in enumerate(zip(rewards, dones, infos)):
            self._ep_rewards[i] += r
            if done:
                self._writer.writerow([self.num_timesteps, round(self._ep_rewards[i], 4),
                                       info.get("stage", -1)])
                self._f.flush()
                self._ep_rewards[i] = 0.0
        return True

    def _on_training_end(self):
        if self._f:
            self._f.close()


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume",         type=str,  default=None)
    parser.add_argument("--steps",          type=int,  default=TOTAL_TIMESTEPS)
    parser.add_argument("--n-envs",         type=int,  default=N_ENVS,
                        help=f"parallel envs via SubprocVecEnv (default {N_ENVS})")
    parser.add_argument("--render-eval",    action="store_true")
    parser.add_argument("--eval-freq",      type=int,  default=EVAL_FREQ_DEFAULT)
    parser.add_argument("--render-capture", action="store_true")
    parser.add_argument("--stage",          type=int,  default=0,
                        help=f"curriculum stage to start at, 0-{len(CURRICULUM) - 1} "
                             f"(use the last stage printed when resuming)")
    args = parser.parse_args()

    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)

    timestamp  = datetime.now().strftime("%Y%m%d_%H%M")
    reward_csv = os.path.join(LOG_DIR, f"rewards_{timestamp}.csv")
    print(f"Logging rewards to: {reward_csv}")

    env = make_vec_env(SwerveEnv, n_envs=args.n_envs, vec_env_cls=SubprocVecEnv)
    print(f"Using {args.n_envs} parallel envs (SubprocVecEnv).")

    checkpoint_cb = CheckpointCallback(
        save_freq   = max(1, CHECKPOINT_FREQ // max(args.n_envs, 1)),
        save_path   = CHECKPOINT_DIR,
        name_prefix = "randomizer",
        verbose     = 1,
    )
    reward_cb = RewardLogger(reward_csv)
    curriculum_cb = CurriculumCallback(start_stage=args.stage)
    callbacks = [checkpoint_cb, reward_cb, curriculum_cb]

    if args.render_eval:
        callbacks.append(RenderEvalCallback(eval_freq=args.eval_freq, curriculum=curriculum_cb))
        print(f"Render-eval ON: window opens/closes every {args.eval_freq:,} steps.")

    if args.render_capture:
        rec_dir = os.path.join(RECORDINGS_DIR, f"run_{timestamp}")
        callbacks.append(RecordEvalCallback(recordings_dir=rec_dir, curriculum=curriculum_cb))
        print(f"Render-capture ON: MP4s will be saved to {rec_dir}/")

    if args.resume:
        print(f"Resuming from: {args.resume}")
        resume_kwargs = {k: v for k, v in SAC_KWARGS.items()
                         if k not in ("verbose", "policy_kwargs")}
        model = SAC.load(args.resume, env=env, **resume_kwargs)
        model.verbose = 1
        # Checkpoints don't save the replay buffer, so a resumed run starts with
        # an empty one. learning_starts counts total steps (already past 10k), so
        # without this it would train from the first step on a handful of
        # samples. Refill the buffer first, as a fresh run does.
        model.learning_starts = model.num_timesteps + SAC_KWARGS["learning_starts"]
    else:
        model = SAC(env=env, **SAC_KWARGS)

    print(f"\nStarting randomized-waypoint training for {args.steps:,} timesteps.")
    print(f"Envs: {args.n_envs}  |  device: {SAC_KWARGS['device']}")
    print(f"Curriculum: {len(CURRICULUM)} stages, last = "
          f"{CURRICULUM[-1]['n_min']}-{CURRICULUM[-1]['n_max']} waypoints, "
          f"{MIN_WAYPOINT_DISTANCE}-{CURRICULUM[-1]['max_dist']} m apart; "
          f"{NODE_TIME_LIMIT_STEPS} steps per node")
    print(f"Checkpoints saved every {CHECKPOINT_FREQ:,} steps to {CHECKPOINT_DIR}/")
    print("Press Ctrl+C to stop early — latest checkpoint is kept.\n")

    try:
        model.learn(
            total_timesteps     = args.steps,
            callback            = callbacks,
            reset_num_timesteps = (args.resume is None),
            progress_bar        = True,
        )
    except (KeyboardInterrupt, EOFError, BrokenPipeError):
        print("\nTraining interrupted by user.")
    finally:
        final_path = os.path.join(CHECKPOINT_DIR, "randomizer_final.zip")
        model.save(final_path)
        print(f"\nFinal model saved to: {final_path}")
        try:
            env.close()
        except Exception:
            pass


if __name__ == "__main__":
    main()
