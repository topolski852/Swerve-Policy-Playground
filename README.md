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
| Path Randomizer | `path_randomizer/` | Random waypoint chains (3–12 points) anywhere on the full field |
| Fuel Scoring | `fuel_scoring/` | Collect fuel in the neutral zone, return to score it at the hub |
| Teleop Assist | `teleop_assist/` | Policy learns to mirror driver joystick input while avoiding collisions |

---

## Quick Start

### Install dependencies
```bash
pip install "stable-baselines3==2.8.0" gymnasium pygame numpy imageio
```

For GPU training on a Pascal-era card (GTX 1050 Ti / CC 6.1):
```bash
pip install "torch==2.7.1+cu118" --index-url https://download.pytorch.org/whl/cu118 --no-deps
```

> **Note:** `torch>=2.8` dropped Pascal (CC 6.1) support. Pin `torch==2.7.1+cu118` on Pascal hardware.

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

Each episode generates a fresh chain of 3–12 waypoints placed randomly across the field (1–6 m apart). The agent must navigate all of them in order. A monotone approach-progress reward prevents the agent from farming reward by oscillating near a waypoint.

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
