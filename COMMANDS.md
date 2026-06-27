# Training Command Reference

All commands run from the repo root. Linux uses `python3`, Windows uses `python`.

---

## Path Following (`train.py`)
Default: 2,000,000 steps | Checkpoints: `path_following/checkpoints/`

### Linux
```bash
# Fresh run
python3 train.py

# With live eval window
python3 train.py --render-eval

# With eval window, faster eval checks
python3 train.py --render-eval --eval-freq 10000

# Save MP4 recordings during training
python3 train.py --render-capture

# Resume from checkpoint
python3 train.py --resume path_following/checkpoints/swerve_100000_steps.zip

# Resume + custom step count
python3 train.py --resume path_following/checkpoints/swerve_100000_steps.zip --steps 500000
```

### Windows
```bat
python train.py
python train.py --render-eval
python train.py --render-eval --eval-freq 10000
python train.py --render-capture
python train.py --resume path_following\checkpoints\swerve_100000_steps.zip
python train.py --resume path_following\checkpoints\swerve_100000_steps.zip --steps 500000
```

---

## Fuel Scoring (`train_scoring.py`)
Default: 5,000,000 steps | Checkpoints: `fuel_scoring/checkpoints/`

### Linux
```bash
# Fresh run
python3 train_scoring.py

# With live eval window
python3 train_scoring.py --render-eval

# With eval window, faster eval checks
python3 train_scoring.py --render-eval --eval-freq 10000

# Save MP4 recordings during training
python3 train_scoring.py --render-capture

# Resume from checkpoint
python3 train_scoring.py --resume fuel_scoring/checkpoints/scoring_100000_steps.zip

# Resume + render capture
python3 train_scoring.py --resume fuel_scoring/checkpoints/scoring_100000_steps.zip --render-capture
```

### Windows
```bat
python train_scoring.py
python train_scoring.py --render-eval
python train_scoring.py --render-eval --eval-freq 10000
python train_scoring.py --render-capture
python train_scoring.py --resume fuel_scoring\checkpoints\scoring_100000_steps.zip
python train_scoring.py --resume fuel_scoring\checkpoints\scoring_100000_steps.zip --render-capture
```

---

## Path Randomizer (`train_randomizer.py`)
Default: 7,000,000 steps, 2 parallel envs | Checkpoints: `path_randomizer/checkpoints/`

### Linux
```bash
# Fresh run
python3 train_randomizer.py

# With live eval window
python3 train_randomizer.py --render-eval

# With eval window, faster eval checks
python3 train_randomizer.py --render-eval --eval-freq 10000

# Save MP4 recordings during training
python3 train_randomizer.py --render-capture

# More parallel envs (e.g. 4 on a beefy machine)
python3 train_randomizer.py --n-envs 4

# Resume from checkpoint
python3 train_randomizer.py --resume path_randomizer/checkpoints/randomizer_100000_steps.zip

# Resume with more envs
python3 train_randomizer.py --resume path_randomizer/checkpoints/randomizer_100000_steps.zip --n-envs 4
```

### Windows
```bat
python train_randomizer.py
python train_randomizer.py --render-eval
python train_randomizer.py --render-eval --eval-freq 10000
python train_randomizer.py --render-capture
python train_randomizer.py --n-envs 4
python train_randomizer.py --resume path_randomizer\checkpoints\randomizer_100000_steps.zip
python train_randomizer.py --resume path_randomizer\checkpoints\randomizer_100000_steps.zip --n-envs 4
```

---

## Teleop Assist (`train_teleop.py`)
Default: 3,000,000 steps | Checkpoints: `teleop_assist/checkpoints/`

### Linux
```bash
# Fresh run
python3 train_teleop.py

# With live eval window
python3 train_teleop.py --render-eval

# With eval window, faster eval checks
python3 train_teleop.py --render-eval --eval-freq 10000

# Resume from checkpoint
python3 train_teleop.py --resume teleop_assist/checkpoints/teleop_100000_steps.zip

# Resume + render eval
python3 train_teleop.py --resume teleop_assist/checkpoints/teleop_100000_steps.zip --render-eval
```

### Windows
```bat
python train_teleop.py
python train_teleop.py --render-eval
python train_teleop.py --render-eval --eval-freq 10000
python train_teleop.py --resume teleop_assist\checkpoints\teleop_100000_steps.zip
python train_teleop.py --resume teleop_assist\checkpoints\teleop_100000_steps.zip --render-eval
```

---

## Render / Replay

### Linux
```bash
# Replay path-following checkpoint
python3 render.py path_following/checkpoints/swerve_final.zip
python3 render.py path_following/checkpoints/swerve_final.zip --episodes 10
python3 render.py path_following/checkpoints/swerve_final.zip --speed 0.5

# Replay teleop-assist checkpoint
python3 render_teleop.py teleop_assist/teleop_final.zip
python3 render_teleop.py teleop_assist/teleop_final.zip --episodes 10
python3 render_teleop.py teleop_assist/teleop_final.zip --speed 0.5
```

### Windows
```bat
python render.py path_following\checkpoints\swerve_final.zip
python render.py path_following\checkpoints\swerve_final.zip --episodes 10
python render.py path_following\checkpoints\swerve_final.zip --speed 0.5

python render_teleop.py teleop_assist\teleop_final.zip
python render_teleop.py teleop_assist\teleop_final.zip --episodes 10
python render_teleop.py teleop_assist\teleop_final.zip --speed 0.5
```

---

## Verify Environments

### Linux
```bash
python3 verify_env.py
python3 verify_env_scoring.py
python3 verify_env_randomizer.py
python3 verify_env_teleop.py
```

### Windows
```bat
python verify_env.py
python verify_env_scoring.py
python verify_env_randomizer.py
python verify_env_teleop.py
```

---

## NT Bridge (Teleop Assist → 1507Labs Sim)

### Linux
```bash
# Connect to local sim (127.0.0.1)
python3 -m teleop_assist.nt_bridge

# Connect to real robot
python3 -m teleop_assist.nt_bridge --host 10.15.7.2
```

### Windows
```bat
python -m teleop_assist.nt_bridge
python -m teleop_assist.nt_bridge --host 10.15.7.2
```

---

## Flags Summary

| Flag | Type | Description |
|------|------|-------------|
| `--resume <path>` | str | Resume training from a checkpoint .zip |
| `--steps <n>` | int | Override total training timesteps |
| `--render-eval` | flag | Open a live window during eval callbacks |
| `--eval-freq <n>` | int | How often (steps) to run eval (default 20000) |
| `--render-capture` | flag | Save MP4s at record intervals (scoring/randomizer/path) |
| `--n-envs <n>` | int | Parallel SubprocVecEnv count (randomizer only, default 2) |
| `--episodes <n>` | int | Episodes to render (render scripts only, default 5) |
| `--speed <f>` | float | Playback speed multiplier (render scripts only, default 1.0) |
