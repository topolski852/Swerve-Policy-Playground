# ──────────────────────────────────────────────────────────────────────────────
# teleop_assist/nt_bridge.py
# NetworkTables bridge: reads robot state from the WPILib sim (or real robot),
# runs the teleop-assist SAC policy, and writes back ChassisSpeeds.
#
# Run from the Swerve-Policy-Playground root:
#   python -m teleop_assist.nt_bridge [--host 127.0.0.1]
#
# ntcore ships with robotpy (pyntcore) — no extra install needed.
#
# NT topic layout (matches PolicyAssist.java):
#   Reads:   policy/input/joy_left_y, joy_left_x, joy_right_x   (raw, no deadband)
#            policy/input/heading, vx, vy, omega, rx, ry
#   Writes:  policy/output/vx, vy, omega
#            policy/heartbeat   (time.monotonic(), double)
# ──────────────────────────────────────────────────────────────────────────────

import argparse
import math
import time

import numpy as np

try:
    import ntcore
except ImportError:
    raise SystemExit("ntcore not found — run: pip install robotpy")

from stable_baselines3 import SAC

from lib.field_constants import MAX_SPEED_MPS, MAX_ANGULAR_RPS, FIELD_LENGTH, FIELD_WIDTH

# ── Model path ────────────────────────────────────────────────────────────────
MODEL_PATH = "teleop_assist/teleop_final"

# ── Loop timing ───────────────────────────────────────────────────────────────
DT = 0.02   # 50 Hz — matches WPILib 20 ms loop


def _rotate_to_robot_frame(
    joy_field_x: float, joy_field_y: float, heading: float
) -> tuple[float, float]:
    """
    Rotate field-frame joystick (x=forward, y=left) into robot frame.
    Matches _true_joy_robot() in env.py.
    """
    cos_h = math.cos(heading)
    sin_h = math.sin(heading)
    rx =  joy_field_x * cos_h + joy_field_y * sin_h
    ry = -joy_field_x * sin_h + joy_field_y * cos_h
    return float(np.clip(rx, -1.0, 1.0)), float(np.clip(ry, -1.0, 1.0))


def _build_obs(
    joy_rf_x: float, joy_rf_y: float, joy_rot: float,
    vx: float, vy: float, omega: float,
    heading: float,
    rx: float, ry: float,
) -> np.ndarray:
    """
    Build the 10-element observation vector matching env.py OBS_LABELS order.
    """
    vx_n    = float(np.clip(vx    / MAX_SPEED_MPS,   -1.0, 1.0))
    vy_n    = float(np.clip(vy    / MAX_SPEED_MPS,   -1.0, 1.0))
    omega_n = float(np.clip(omega / MAX_ANGULAR_RPS, -1.0, 1.0))
    sin_h   = math.sin(heading)
    cos_h   = math.cos(heading)
    rx_n    = float(np.clip(rx / FIELD_LENGTH, 0.0, 1.0))
    ry_n    = float(np.clip(ry / FIELD_WIDTH,  0.0, 1.0))

    return np.array([
        joy_rf_x, joy_rf_y, joy_rot,
        vx_n, vy_n, omega_n,
        sin_h, cos_h,
        rx_n, ry_n,
    ], dtype=np.float32)


def run(host: str) -> None:
    print(f"Loading policy from {MODEL_PATH} ...")
    model = SAC.load(MODEL_PATH)
    print("Policy loaded.")

    inst = ntcore.NetworkTableInstance.getDefault()
    inst.startClient4("teleop-assist-bridge")
    inst.setServer(host)
    print(f"Connecting to NT server at {host} ...")

    root = inst.getTable("policy")
    inp  = root.getSubTable("input")
    out  = root.getSubTable("output")

    # Subscribers — Java publishes raw (undeadbanded) joystick + robot state
    sub_joy_y   = inp.getDoubleTopic("joy_left_y").subscribe(0.0)
    sub_joy_x   = inp.getDoubleTopic("joy_left_x").subscribe(0.0)
    sub_joy_rot = inp.getDoubleTopic("joy_right_x").subscribe(0.0)
    sub_heading = inp.getDoubleTopic("heading").subscribe(0.0)
    sub_vx      = inp.getDoubleTopic("vx").subscribe(0.0)
    sub_vy      = inp.getDoubleTopic("vy").subscribe(0.0)
    sub_omega   = inp.getDoubleTopic("omega").subscribe(0.0)
    sub_rx      = inp.getDoubleTopic("rx").subscribe(0.0)
    sub_ry      = inp.getDoubleTopic("ry").subscribe(0.0)

    # Publishers — Python writes ChassisSpeeds + heartbeat
    pub_out_vx    = out.getDoubleTopic("vx").publish()
    pub_out_vy    = out.getDoubleTopic("vy").publish()
    pub_out_omega = out.getDoubleTopic("omega").publish()
    pub_hb        = root.getDoubleTopic("heartbeat").publish()

    print("Bridge running — press Ctrl+C to stop.\n")

    while True:
        t0 = time.monotonic()

        joy_y   = sub_joy_y.get()
        joy_x   = sub_joy_x.get()
        joy_rot = sub_joy_rot.get()
        heading = sub_heading.get()
        vx      = sub_vx.get()
        vy      = sub_vy.get()
        omega   = sub_omega.get()
        rx      = sub_rx.get()
        ry      = sub_ry.get()

        joy_rf_x, joy_rf_y = _rotate_to_robot_frame(joy_y, joy_x, heading)

        obs = _build_obs(
            joy_rf_x, joy_rf_y, joy_rot,
            vx, vy, omega,
            heading,
            rx, ry,
        )

        action, _ = model.predict(obs, deterministic=True)

        pub_out_vx   .set(float(action[0]) * MAX_SPEED_MPS)
        pub_out_vy   .set(float(action[1]) * MAX_SPEED_MPS)
        pub_out_omega.set(float(action[2]) * MAX_ANGULAR_RPS)
        pub_hb       .set(time.monotonic())

        elapsed = time.monotonic() - t0
        if DT - elapsed > 0:
            time.sleep(DT - elapsed)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Teleop-assist NT bridge")
    parser.add_argument(
        "--host", default="127.0.0.1",
        help="NT server host (127.0.0.1 for sim; 10.15.7.2 for real robot)"
    )
    args = parser.parse_args()

    try:
        run(args.host)
    except KeyboardInterrupt:
        print("\nBridge stopped.")
