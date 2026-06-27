"""
verify_env_teleop.py
Sanity checks for TeleopAssistEnv. No display required.
Run from the repo root: python verify_env_teleop.py
Should print 7 PASSes with no errors.
"""

import numpy as np
from teleop_assist.env import TeleopAssistEnv, OBS_DIM
from teleop_assist.constants import MAX_EPISODE_STEPS, DRIFT_FLOOR

def check(label, cond):
    tag = "PASS" if cond else "FAIL"
    print(f"  [{tag}]  {label}")
    return cond

print(f"\n--- TeleopAssistEnv sanity checks (OBS_DIM={OBS_DIM}) ---\n")

env = TeleopAssistEnv()
obs, _ = env.reset()

check(f"obs shape is ({OBS_DIM},)",    obs.shape == (OBS_DIM,))
check("obs dtype is float32",          obs.dtype == np.float32)
check("obs within declared bounds",
      np.all(obs >= env.observation_space.low) and
      np.all(obs <= env.observation_space.high))
check("action space shape is (3,)",    env.action_space.shape == (3,))

# ── Step with zero action — should not crash ───────────────────────────────────
_, rew, term, trunc, info = env.step(np.zeros(3, dtype=np.float32))
check("zero action step OK",           not (term or trunc))
check("info has expected keys",
      {"r_intent", "r_still", "r_smooth", "joy_mag"}.issubset(info.keys()))

# ── Obs stays in bounds over a random-policy rollout ──────────────────────────
obs, _ = env.reset()
in_bounds = True
episodes  = 0
for _ in range(5):
    for _ in range(MAX_EPISODE_STEPS + 10):
        action = env.action_space.sample()
        obs, _, term, trunc, _ = env.step(action)
        if not (np.all(obs >= env.observation_space.low) and
                np.all(obs <= env.observation_space.high)):
            in_bounds = False
        if term or trunc:
            episodes += 1
            obs, _ = env.reset()
            break
check("obs always in bounds during random rollout", in_bounds)

env.close()

# ── Intent reward is positive when action aligns with joystick ────────────────
env2 = TeleopAssistEnv()
obs, _ = env2.reset()
intent_ok = False
for _ in range(200):
    joy_xy = obs[:2].copy()
    joy_mag = float(np.linalg.norm(joy_xy))
    if joy_mag > DRIFT_FLOOR + 0.05:
        # Action that exactly matches the observed joystick direction
        aligned = np.array([joy_xy[0], joy_xy[1], 0.0], dtype=np.float32)
        _, _, _, _, info = env2.step(aligned)
        if info.get("r_intent", 0.0) > 0:
            intent_ok = True
            break
    obs, _, term, trunc, _ = env2.step(env2.action_space.sample())
    if term or trunc:
        obs, _ = env2.reset()
check("intent reward > 0 when action aligns with joystick", intent_ok)
env2.close()

# ── Still penalty fires during a stop-intent period ───────────────────────────
env3 = TeleopAssistEnv()
env3.reset()
env3._joy_is_stop = True   # force stop-intent period
_, _, _, _, info = env3.step(np.array([1.0, 0.0, 0.0], dtype=np.float32))
check("still penalty fires during stop-intent when robot moves",
      info.get("r_still", 0.0) < 0)
env3.close()

# ── Collision terminates episode ───────────────────────────────────────────────
env4 = TeleopAssistEnv()
env4.reset()
# Place the robot near the left wall and drive it straight into the wall
env4._robot.reset(x=0.35, y=3.0, heading=0.0)
terminated = False
for _ in range(30):
    _, rew, term, trunc, _ = env4.step(np.array([-1.0, 0.0, 0.0], dtype=np.float32))
    if term:
        terminated = True
        break
check(f"collision terminates episode (last_rew={rew:.1f})", terminated)
env4.close()

print()
