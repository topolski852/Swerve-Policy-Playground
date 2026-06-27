"""
verify_env_teleop.py
Sanity checks for TeleopAssistEnv. No display required.
Run from the repo root: python verify_env_teleop.py
Should print 10 PASSes with no errors.
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

# ── Step with zero action — should not crash ──────────────────────────────────
_, rew, term, trunc, info = env.step(np.zeros(3, dtype=np.float32))
check("zero action step OK",           not (term or trunc))
check("info has expected keys",
      {"r_match", "r_still", "r_dir", "r_smooth", "target_mag"}.issubset(info.keys()))

# ── Obs stays in bounds over a random-policy rollout ──────────────────────────
obs, _ = env.reset()
in_bounds = True
for _ in range(5):
    for _ in range(MAX_EPISODE_STEPS + 10):
        action = env.action_space.sample()
        obs, _, term, trunc, _ = env.step(action)
        if not (np.all(obs >= env.observation_space.low) and
                np.all(obs <= env.observation_space.high)):
            in_bounds = False
        if term or trunc:
            obs, _ = env.reset()
            break
check("obs always in bounds during random rollout", in_bounds)

env.close()

# ── r_match is highest for a perfectly aligned action ────────────────────────
env2 = TeleopAssistEnv()
obs, _ = env2.reset()
match_ok = False
for _ in range(200):
    joy_xy = obs[:2].copy()
    joy_mag = float(np.linalg.norm(joy_xy))
    if joy_mag > DRIFT_FLOOR + 0.05:
        # Perfect match: action = observed joystick (true joy + drift ≈ true joy)
        perfect  = np.array([joy_xy[0], joy_xy[1], 0.0], dtype=np.float32)
        opposite = np.array([-joy_xy[0], -joy_xy[1], 0.0], dtype=np.float32)
        _, _, _, _, info_perfect  = env2.step(perfect)
        _, _, _, _, info_opposite = env2.step(opposite)
        if (info_perfect.get("r_match", 0.0) > info_opposite.get("r_match", 0.0) and
                info_perfect.get("r_match", 0.0) > 0):
            match_ok = True
            break
    obs, _, term, trunc, _ = env2.step(env2.action_space.sample())
    if term or trunc:
        obs, _ = env2.reset()
check("r_match higher for aligned action than opposite", match_ok)
env2.close()

# ── Direction penalty fires when going opposite to intent ─────────────────────
env3 = TeleopAssistEnv()
env3.reset()
env3._robot.reset(x=env3._robot.x, y=env3._robot.y, heading=0.0)  # face east
env3._joy_is_stop = False
env3._joy_target_x = env3._robot.x + 3.0   # target is east → robot-frame +x
env3._joy_target_y = env3._robot.y
env3._joy_speed = 0.8
env3._joy_reroll_countdown = 999
# Action [-1, 0, 0] drives west (robot-frame -x) = opposite to intent
_, _, _, _, info = env3.step(np.array([-1.0, 0.0, 0.0], dtype=np.float32))
check("r_dir is negative when going opposite to intent",
      info.get("r_dir", 0.0) < 0)
env3.close()

# ── still penalty fires and r_match peaks when stopped during stop intent ─────
env4 = TeleopAssistEnv()
env4.reset()
env4._joy_is_stop = True
_, _, _, _, info_still  = env4.step(np.zeros(3, dtype=np.float32))
_, _, _, _, info_moving = env4.step(np.array([1.0, 0.0, 0.0], dtype=np.float32))
check("r_match higher for zero output than full-speed during stop intent",
      info_still.get("r_match", 0.0) > info_moving.get("r_match", 0.0))
check("r_still is zero when not moving during stop intent",
      info_still.get("r_still", -1.0) == 0.0)
check("r_still is negative when moving during stop intent",
      info_moving.get("r_still", 0.0) < 0.0)
env4.close()

# ── Collision terminates episode ──────────────────────────────────────────────
env5 = TeleopAssistEnv()
env5.reset()
env5._robot.reset(x=0.35, y=3.0, heading=0.0)
terminated = False
for _ in range(30):
    _, rew, term, trunc, _ = env5.step(np.array([-1.0, 0.0, 0.0], dtype=np.float32))
    if term:
        terminated = True
        break
check(f"collision terminates episode (last_rew={rew:.1f})", terminated)
env5.close()

print()
