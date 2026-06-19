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

### Observation space (OBS_DIM = 7)
| # | Name | Range | Meaning |
|---|---|---|---|
| 0 | vx_n | [-1,1] | velocity X normalized |
| 1 | vy_n | [-1,1] | velocity Y normalized |
| 2 | rx_n | [0,1] | X position normalized |
| 3 | ry_n | [0,1] | Y position normalized |
| 4 | hopper_norm | [0,1] | current hopper / 60 |
| 5 | contributed_norm | [0,1] | fuel scored this episode / 300 |
| 6 | total_norm | [0,1] | same as contributed (reserved for multi-robot) |

### Fuel mechanics
- **Collect**: neutral zone (5.223 < x < 11.317), speed >= 0.5 m/s → log-scaled up to 1.5 fuel/step at max speed
- **Score**: alliance zone (x < 4.029), any speed, hopper > 0 → 0.4 fuel/step drained, 5.0 reward/fuel
- **Penalties**: full hopper in neutral = -0.5/step; empty hopper in alliance = -0.5/step; collision = -25

### Training config (train_scoring.py)
- `learning_starts = 20_000`
- `device = "cpu"` — change to `"cuda"` on PC with GTX 1080 Ti
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

## Setting up on a new machine

```bash
git clone <repo>
pip install stable-baselines3 gymnasium pygame numpy imageio
# For GTX 1080 Ti (CUDA 12.1 — adjust cu version to match installed CUDA):
pip install torch --index-url https://download.pytorch.org/whl/cu121
python verify_env_scoring.py   # should show 7 PASSes
python train_scoring.py --render-capture
```

Change `device = "cpu"` → `device = "cuda"` in `train_scoring.py` before running on a CUDA machine.

## Next planned work
- SubprocVecEnv: parallel environments (4-8 envs on PC, 100 on future dev box)
- Phase 3: enable omega (rotation), robot must face hub to score
