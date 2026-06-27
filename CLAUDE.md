# Swerve Policy Playground — Claude Context

## Project
FRC 2026 robot policy trainer using Stable-Baselines3 SAC + custom Gymnasium environments.
Long-term goal: Electron desktop app where students tune reward weights and run experiments.
Even longer-term: 6-robot multi-agent match simulation.

## Repo structure
```
train.py                  # path_following trainer (run from root)
train_scoring.py          # fuel scoring trainer (run from root)
lib/                      # shared: kinematics, renderer, field constants
path_following/           # experiment 1: figure-8 path navigation
fuel_scoring/             # experiment 2: fuel collect/score loop (active)
```

## Active experiment: fuel_scoring

### Observation space (OBS_DIM = 8)
| # | Name | Range | Meaning |
|---|---|---|---|
| 0 | vx_n | [-1,1] | velocity X normalized |
| 1 | vy_n | [-1,1] | velocity Y normalized |
| 2 | rx_n | [0,1] | X position normalized |
| 3 | ry_n | [0,1] | Y position normalized |
| 4 | hopper_norm | [0,1] | current hopper / 60 |
| 5 | contributed_norm | [0,1] | fuel scored this episode / 300 |
| 6 | total_norm | [0,1] | same as contributed (reserved for multi-robot) |
| 7 | hub_dist_norm | [0,1] | distance to Blue hub / 8 m; **0.0 when in neutral zone** (hub irrelevant during collection) |

### Fuel mechanics
- **Collect**: neutral zone (5.223 < x < 11.317), speed >= 0.5 m/s → log-scaled up to 1.0 fuel/step at max speed
- **Score**: alliance zone (x < 4.029), hopper > 0 → hopper drains 0.4/step; fuel scored = 0.4 × speed_factor × dist_factor
  - `speed_factor` = max(0.1, 1.0 − 0.9 × speed/max_speed) — stationary = 1.0, full speed = 0.1
  - `dist_factor`  = max(0.5, 1.0 − 0.5 × max(0, (dist_to_hub − 1 m) / 5 m)) — within 1 m = 1.0, at 6 m = 0.5
- **Penalties**: full hopper in neutral = -0.5/step; empty hopper in alliance = -0.5/step; neutral idle (below collect speed, hopper not full) = -0.3/step; collision = -25
- **Milestones**: +50 bonus at 100 fuel scored; +150 bonus at 360 fuel scored (one-time per episode)

### Training config (train_scoring.py)
- `learning_starts = 20_000`
- `device = "cpu"` — change to `"cuda"` on PC with GTX 1050 Ti (4GB VRAM)
- `buffer_size = 200_000`, `batch_size = 256`
- Training env uses `random_start=True` (4 safe starting positions, random hopper 0-60)
- Eval/render callbacks always use fixed start (3.50, 2.55) with hopper=8

### 4 random starting positions
- (5.90, 5.60) — neutral side Blue BumpLeft
- (3.20, 5.60) — alliance side Blue BumpLeft top
- (3.20, 2.61) — alliance side Blue BumpRight bottom
- (5.90, 2.61) — neutral side Blue BumpRight

### Training progress (as of 2026-06-19, laptop run)
- Reached ~77k steps before moving to PC for thermal/performance reasons
- Mean reward still negative (-200 to -400 range) — expected at this stage
- Best single episode: +249 — agent occasionally finds the fuel loop
- Target: ~150-200k steps before mean reward turns consistently positive

## PC hardware (gaming PC, Kelly's desk)
- **GPU**: NVIDIA GeForce GTX 1050 Ti (4GB VRAM, Pascal / compute 6.1)
- **CPU**: Intel i7-6700K @ 4.00GHz, 4 cores / 8 threads
- **RAM**: 16 GB
- **NVIDIA driver**: ~581.x (supports CUDA up to 12.8)

## Setting up on a new machine

```bash
git clone <repo>
pip install "stable-baselines3==2.8.0" gymnasium pygame numpy imageio
# GTX 1050 Ti (Pascal CC 6.1) requires cu118 — newer CUDA builds dropped Pascal support:
pip install "torch==2.7.1+cu118" --index-url https://download.pytorch.org/whl/cu118 --no-deps
python verify_env_scoring.py   # should show 7 PASSes
python train_scoring.py --render-capture
```

## NT bridge (teleop-assist policy → 1507Labs sim)

NT bridge uses `ntcore` which ships with `robotpy` (already installed as `pyntcore`).
No additional install needed.

Run order:
1. `./gradlew simulateJava` in 1507Labs (starts the robot sim + NT server on 127.0.0.1)
2. `python -m teleop_assist.nt_bridge` from this directory
3. Open AdvantageScope, connect to `127.0.0.1`
4. Press **Right Bumper** in the sim to enable policy assist

For real robot (once SystemCore ships):
```bash
python -m teleop_assist.nt_bridge --host 10.15.7.2
```

Change `device = "cpu"` → `device = "cuda"` in `train_scoring.py` before running on a CUDA machine.

**CUDA note**: `torch>=2.8` dropped Pascal (CC 6.1) support. Pin `torch==2.7.1+cu118` + `stable-baselines3==2.8.0` on this machine. Do NOT run `pip install --upgrade` without re-pinning torch.

**SubprocVecEnv target**: 4 parallel envs on i7-6700K (leaves headroom for GPU-side training loop).

## Next planned work
- SubprocVecEnv: parallel environments (4-8 envs on PC, 100 on future dev box)
- Phase 3: enable omega (rotation), robot must face hub to score
