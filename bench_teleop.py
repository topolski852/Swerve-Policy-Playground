"""
bench_teleop.py — find optimal N_ENVS, vec_env type, gradient_steps, batch_size.
Run from repo root:  python bench_teleop.py
"""

import time
import numpy as np
import torch

WARMUP  = 300
MEASURE = 2000

def bench_rollout(n_envs, vec_cls, label):
    from stable_baselines3.common.env_util import make_vec_env
    from teleop_assist.env import TeleopAssistEnv
    env = make_vec_env(TeleopAssistEnv, n_envs=n_envs, vec_env_cls=vec_cls)
    env.reset()
    acts = np.array([env.action_space.sample() for _ in range(n_envs)])
    for _ in range(WARMUP):
        env.step(acts)
    t0 = time.perf_counter()
    for _ in range(MEASURE):
        env.step(acts)
    elapsed = time.perf_counter() - t0
    fps = MEASURE * n_envs / elapsed
    ms  = elapsed * 1000 / MEASURE
    print(f"  {label:<42}  {fps:>9,.0f} steps/s   {ms:.2f} ms/outer-step")
    env.close()
    return fps

def bench_gradient(batch_size, label):
    """Time pure gradient updates by running a real (tiny) training session."""
    from stable_baselines3.common.env_util import make_vec_env
    from stable_baselines3.common.vec_env import DummyVecEnv
    from stable_baselines3 import SAC
    from teleop_assist.env import TeleopAssistEnv

    PRIME   = 6_000   # steps to fill buffer before timing
    MEASURE_GRAD = 400

    env = make_vec_env(TeleopAssistEnv, n_envs=1, vec_env_cls=DummyVecEnv)
    model = SAC("MlpPolicy", env, device="cuda",
                batch_size=batch_size, buffer_size=10_000,
                learning_starts=PRIME, gradient_steps=1,
                train_freq=1, verbose=0)

    # Prime: fill the buffer and warm up the optimizer
    model.learn(total_timesteps=PRIME + 200, progress_bar=False)
    torch.cuda.synchronize()

    # Now time pure gradient steps by continuing training but only counting GPU time
    # We run learn() for MEASURE_GRAD more steps with train_freq=1, gradient_steps=1
    t0 = time.perf_counter()
    model.learn(total_timesteps=PRIME + 200 + MEASURE_GRAD,
                reset_num_timesteps=False, progress_bar=False)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0

    # elapsed includes both env steps and gradient steps, but env steps are ~0.02ms each
    # so for MEASURE_GRAD steps: env_time ≈ MEASURE_GRAD * 0.02ms (negligible)
    ms = elapsed * 1000 / MEASURE_GRAD
    print(f"  {label:<42}  {ms:.2f} ms/step  (env+grad combined; env≈0.02ms)")
    env.close()
    return ms


if __name__ == "__main__":
    from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv

    print("\n" + "="*70)
    print("  ROLLOUT SPEED  (env stepping only, no gradient updates)")
    print("="*70)
    print("  --- DummyVecEnv (sequential, no IPC) ---")
    bench_rollout(1,   DummyVecEnv,  "DummyVecEnv   n_envs=1")
    bench_rollout(16,  DummyVecEnv,  "DummyVecEnv   n_envs=16")
    bench_rollout(32,  DummyVecEnv,  "DummyVecEnv   n_envs=32")
    bench_rollout(64,  DummyVecEnv,  "DummyVecEnv   n_envs=64")
    bench_rollout(128, DummyVecEnv,  "DummyVecEnv   n_envs=128")
    bench_rollout(256, DummyVecEnv,  "DummyVecEnv   n_envs=256")
    print()
    print("  --- SubprocVecEnv (parallel processes, has IPC overhead) ---")
    bench_rollout(16,  SubprocVecEnv,"SubprocVecEnv n_envs=16")
    bench_rollout(32,  SubprocVecEnv,"SubprocVecEnv n_envs=32")
    bench_rollout(64,  SubprocVecEnv,"SubprocVecEnv n_envs=64")
    bench_rollout(128, SubprocVecEnv,"SubprocVecEnv n_envs=128")

    print("\n" + "="*70)
    print("  GRADIENT STEP SPEED  (RTX 5080, 1 env, train_freq=1 grad_steps=1)")
    print("="*70)
    for bs in [256, 512, 1024, 2048, 4096, 8192]:
        bench_gradient(bs, f"batch_size={bs}")

    print("\n" + "="*70)
    print("  SUMMARY / HOW TO READ THIS")
    print("="*70)
    print("""
  The total outer-step time is:   env_time + gradient_steps * grad_time

  Optimal n_envs:
    Pick the n_envs where SubprocVecEnv stops gaining or falls behind
    DummyVecEnv — that's the IPC crossover point.

  Optimal gradient_steps:
    gradient_steps = env_outer_step_time_ms / grad_time_ms
    keeps the GPU fully busy without it being the bottleneck.

  Optimal batch_size:
    Largest batch where ms/step doesn't significantly increase
    (GPU is still underutilized — bigger batch is free).
""")
