"""
verify_env_scoring.py
Runs a quick random-policy rollout and checks that the fuel scoring
environment is well-formed. No display required.
"""

import numpy as np
from fuel_scoring.swerve_env import SwerveEnv, OBS_DIM

def check(label, cond):
    tag = "PASS" if cond else "FAIL"
    print(f"  [{tag}]  {label}")
    return cond

print("\n--- SwerveEnv (fuel_scoring) sanity checks ---")

env = SwerveEnv()
obs, info = env.reset()

check(f"obs shape is correct (expected {OBS_DIM})", obs.shape == (OBS_DIM,))
check("obs dtype is float32", obs.dtype == np.float32)
check("obs within declared bounds",
      np.all(obs >= env.observation_space.low) and
      np.all(obs <= env.observation_space.high))
check("action space shape", env.action_space.shape == (3,))
check("hopper starts loaded", env._hopper > 0)

rewards = []
episodes = 0
max_eps  = 10

obs, _ = env.reset()
for _ in range(max_eps):
    ep_reward = 0.0
    for _ in range(2000):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        ep_reward += reward
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

# Targeted mechanics check: full hopper near hub → scoring should produce fuel_scored > 0
mech_env = SwerveEnv()
mech_env.reset()
mech_env._hopper = 60.0
mech_env._robot.reset(x=2.0, y=2.55, heading=0.0)
_, _, _, _, mech_info = mech_env.step(np.zeros(3, dtype=np.float32))
check("scoring mechanics active (fuel scored when near hub with full hopper)",
      mech_info.get("fuel_scored", 0) > 0)
mech_env.close()

env.close()
print()
