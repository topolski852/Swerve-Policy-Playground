<div align="center">
  <img src="assets/banner.svg" alt="Swerve Policy Playground" width="100%"/>
</div>

<br/>

**Swerve Policy Playground** is a standalone reinforcement learning sandbox built by [Team 1507 – Warlocks](https://warlocks1507.com). A simulated swerve-drive robot learns to navigate an FRC-style field using Soft Actor-Critic (SAC) training — with no hand-written control logic. The goal is a visual teaching tool showing students how a policy evolves from random stumbling to coordinated movement across hundreds of thousands of training steps.

---

## Experiments

The repo contains four progressive experiments, each building on the last:

| Experiment | Folder | Description |
|---|---|---|
| Path Following | `path_following/` | Fixed figure-8 path around both Alliance Hubs — the original demo |
| Path Randomizer | `path_randomizer/` | Random waypoint chains (up to 12 points) anywhere on the field; becomes 1507Base's `Driver.POLICY` |
| Fuel Scoring | `fuel_scoring/` | Collect fuel in the neutral zone, return to score it at the hub |
| Teleop Assist | `teleop_assist/` | Policy learns to mirror driver joystick input while avoiding collisions |

---

## Quick Start

This repo targets the **2027 (SystemCore) season**. Open the folder in **WPILib VS Code 2027**.
It's a plain Python project with no Gradle build, so the only extension it needs is Python
(`ms-python.python`), which VS Code will suggest. The committed `.vscode/settings.json` points VS Code at `.venv`.

### Install dependencies
```bash
python -m venv .venv
.venv\Scripts\activate            # Windows (VS Code terminals activate it automatically)

# 1. torch: pick ONE
pip install torch --index-url https://download.pytorch.org/whl/cpu                                   # laptop / CPU
pip install "torch==2.7.1+cu118" --index-url https://download.pytorch.org/whl/cu118 --no-deps       # GTX 1050 Ti

# 2. everything else
pip install -r requirements.txt
```

> **Note:** `torch>=2.8` dropped Pascal (CC 6.1) support. Pin `torch==2.7.1+cu118` on Pascal hardware.

> **Packages already installed globally?** `python -m venv --system-site-packages .venv` reuses them
> (no download needed).

### Train a policy

```bash
# Path following (original experiment)
python train.py

# Randomized waypoint navigation
python train_randomizer.py

# Fuel collect/score loop
python train_scoring.py

# Teleop-assist policy
python train_teleop.py
```

All trainers support `--render-eval` (live Pygame window) and `--render-capture` (auto-record MP4 snapshots at training milestones).

```bash
python train_scoring.py --render-capture
```

### Verify an environment
```bash
python verify_env.py           # path_following
python verify_env_randomizer.py
python verify_env_scoring.py   # should show 7 PASSes
python verify_env_teleop.py
```

### Watch any checkpoint
```bash
python render.py checkpoints/swerve_final
python render.py checkpoints/swerve_50000_steps --speed 0.5   # half speed
```

### Plot the reward curve
```bash
python plot_rewards.py   # auto-selects the latest log in logs/
```

---

## Experiment Details

### Path Following (`path_following/`)

The original demo. A fixed figure-8 arc-length parameterized path loops around both Alliance Hubs. The agent earns reward for arc-length progress and velocity alignment, and is penalized for cross-track error and time.

### Path Randomizer (`path_randomizer/`)

Each episode generates a fresh chain of waypoints placed randomly on the field. The agent must reach them in order (within 0.40 m), giving up on any node it can't reach in 5 s. The trained policy becomes the `Driver.POLICY` route driver in 1507Base, so the sim robot uses 1507Base's numbers: 5.04 m/s top speed and a port of its `SwerveAccelLimiter` (`lib/accel_limiter.py`).

- **Observation (16):** velocity, position, vectors to the current and next node (÷ 6 m), and 8 distance rays: how far the robot can drive in each direction before it hits a field element or wall (`path_randomizer/rays.py`, works on any polygon).
- **Reward:** metres of progress toward the node + velocity pointed at it each step (both measured along the shortest path around field elements, so detours aren't punished), +100 per node, +75 for finishing, −10 for a crash. Crashing or giving up on a node ends the episode, as it ends the route on the robot.
- **Curriculum:** starts with 1–3 nodes up to 3 m apart on clear legs and moves up a stage when 75% of recent episodes finish; the last stage is 3–12 nodes 0.5–6 m apart, including legs the robot must steer around a hub or trench. `test_randomizer.py --phase N` tests on stage N.
- **Best model:** every 10k steps the trainer drives the same 50 routes and keeps `checkpoints/best_stage<N>.zip`; skill swings between checkpoints, so the last one is often not the best. The 2026-10-04 1M-step run reached 78% complete / 5% crash on fresh full-difficulty routes.

The commit history explains each design choice; the 2026-10 commits record why the first version never learned.

### Fuel Scoring (`fuel_scoring/`)

A game-mechanic experiment. The robot must:
1. **Collect** — enter the neutral zone at speed to pick up fuel (log-scaled up to 1.0 fuel/step)
2. **Score** — return to the alliance zone and drain the hopper near the hub

Penalties discourage idling with a full hopper or loitering empty in the alliance zone.

### Teleop Assist (`teleop_assist/`)

The policy learns to shadow a simulated driver joystick signal. The observation includes raw joystick intent; the reward peaks when the robot's actual velocity matches `fromFieldRelativeSpeeds(joy_x, joy_y)`. Runs live against the 1507Labs robot simulator via NetworkTables (see NT Bridge below).

---

## NT Bridge (Teleop Assist → 1507Labs Sim)

Connects the trained teleop-assist policy to the Java robot simulator over NT4.

```bash
# 1. Start the robot sim
./gradlew simulateJava   # in 1507Labs

# 2. Start the bridge
python -m teleop_assist.nt_bridge

# 3. Press Right Bumper in the sim to enable policy assist
```

For a real robot (once SystemCore ships):
```bash
python -m teleop_assist.nt_bridge --host 10.15.7.2
```

---

## File Overview

```
train.py / train_*.py        SAC training loops for each experiment
render.py / render_teleop.py Pygame checkpoint playback
verify_env_*.py              Quick sanity checks for each Gymnasium env
plot_rewards.py              Reward curve from training CSV logs
bench_teleop.py              Latency benchmark for the teleop policy
lib/
  kinematics.py              WPILib-equivalent swerve IK + discretize
  renderer.py                Shared Pygame renderer
  field_constants.py         Field geometry (zone bounds, hub positions)
  raycaster.py               Proximity ray sensor for obstacle avoidance
path_following/              Experiment 1: fixed figure-8 path
path_randomizer/             Experiment 2: random waypoint navigation
fuel_scoring/                Experiment 3: collect/score fuel loop
teleop_assist/               Experiment 4: joystick mirroring + NT bridge
```

---

## Tech Stack

| Layer | Technology |
|---|---|
| RL algorithm | [Stable-Baselines3](https://github.com/DLR-RM/stable-baselines3) SAC |
| Environment interface | [Gymnasium](https://gymnasium.farama.org/) |
| Swerve kinematics | Custom Python — WPILib `SwerveDriveKinematics` equivalent |
| Visualization | [Pygame](https://www.pygame.org/) |
| Reward plotting | [matplotlib](https://matplotlib.org/) |
| Video recording | [imageio](https://imageio.readthedocs.io/) + [imageio-ffmpeg](https://github.com/imageio/imageio-ffmpeg) |
| NT bridge | [pyntcore](https://github.com/robotpy/mostrobotpy) (ships with robotpy) |

Built by [Team 1507 – Warlocks](https://warlocks1507.com).
