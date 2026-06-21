"""
verify_env_randomizer.py
Sanity checks for the path_randomizer environment.
"""

import numpy as np
from path_randomizer.swerve_env import SwerveEnv, OBS_DIM

def check(label, cond):
    tag = "PASS" if cond else "FAIL"
    print(f"  [{tag}]  {label}")
    return cond

print("\n--- SwerveEnv (path_randomizer) sanity checks ---")

env = SwerveEnv()
obs, _ = env.reset()

check(f"obs shape is correct (expected {OBS_DIM})",  obs.shape == (OBS_DIM,))
check("obs dtype is float32",   obs.dtype == np.float32)
check("obs within declared bounds",
      np.all(obs >= env.observation_space.low) and
      np.all(obs <= env.observation_space.high))
check("action space shape",     env.action_space.shape == (3,))
check("waypoints generated",         len(env._waypoints) >= 4)   # N nav wps + 1 start
check("start is waypoint 0",         env._waypoints[0] == (env._robot.x, env._robot.y))
check("tracker begins at index 1",   env._tracker.current_idx == 1)

rewards   = []
completed = 0
episodes  = 0
max_eps   = 20

obs, _ = env.reset()
for _ in range(max_eps):
    ep_reward = 0.0
    for _ in range(3000):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        ep_reward += reward
        if terminated:
            completed += 1
        if terminated or truncated:
            rewards.append(ep_reward)
            episodes += 1
            obs, _ = env.reset()
            break

check("all episodes terminated or truncated", episodes == max_eps)
check("obs always within bounds during rollout",
      np.all(obs >= env.observation_space.low) and
      np.all(obs <= env.observation_space.high))

if rewards:
    print(f"\n  Random-policy episode rewards over {episodes} eps:")
    print(f"    mean={np.mean(rewards):.2f}  min={np.min(rewards):.2f}  max={np.max(rewards):.2f}")
    print(f"    episodes with all waypoints reached: {completed}/{episodes}")

env.close()
print()
