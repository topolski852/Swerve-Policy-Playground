# ──────────────────────────────────────────────────────────────────────────────
# teleop_assist/nt_bridge.py
# NetworkTables bridge: reads robot state from the WPILib sim (or real robot),
# runs the teleop-assist SAC policy, and writes back ChassisSpeeds.
#
# Run from the Swerve-Policy-Playground root:
#   python -m teleop_assist.nt_bridge [--host 127.0.0.1]
#
# Dependencies (not in the default requirements.txt — install once):
#   pip install robotpy-ntcore
#
# NT topic layout (matches PolicyAssist.java):
#   Reads:   policy/input/joy_left_y, joy_left_x, joy_right_x
#            policy/input/heading, vx, vy, omega, rx, ry
#   Writes:  policy/output/vx, vy, omega
#            policy/heartbeat            (current time.monotonic(), double)
#            policy/debug/prox_f .. prox_fr  (8 proximity rays, for AdvantageScope)
# ──────────────────────────────────────────────────────────────────────────────

import argparse
import math
import time

import numpy as np

try:
    import ntcore
except ImportError:
    raise SystemExit(
        "ntcore is not installed.\n"
        "Run:  pip install robotpy"
    )

from stable_baselines3 import SAC

from lib.field_constants import MAX_SPEED_MPS, MAX_ANGULAR_RPS, FIELD_LENGTH, FIELD_WIDTH
from lib.raycaster import cast_rays
from teleop_assist.constants import RAY_MAX_DISTANCE

# ── Model path ────────────────────────────────────────────────────────────────
MODEL_PATH = "teleop_assist/teleop_final"

# ── Observation labels (must match env.py OBS_LABELS order exactly) ───────────
_PROX_LABELS = ["prox_f", "prox_fl", "prox_l", "prox_bl",
                "prox_b", "prox_br", "prox_r", "prox_fr"]

# ── Loop timing ───────────────────────────────────────────────────────────────
DT = 0.02   # 50 Hz — matches WPILib 20 ms loop


def _rotate_to_robot_frame(
    joy_field_x: float, joy_field_y: float, heading: float
) -> tuple[float, float]:
    """
    Rotate field-frame joystick (x=forward, y=left) into robot frame.
    Matches the _true_joy_robot() rotation in env.py:
        rx =  fx * cos_h + fy * sin_h
        ry = -fx * sin_h + fy * cos_h
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
    ray_distances: np.ndarray,
) -> np.ndarray:
    """
    Build the 18-element observation vector in the exact order from env.py L334-340.
    All inputs must already be in the correct frame (robot-frame joy, robot-relative velocity).
    """
    vx_n    = np.clip(vx    / MAX_SPEED_MPS,    -1.0, 1.0)
    vy_n    = np.clip(vy    / MAX_SPEED_MPS,    -1.0, 1.0)
    omega_n = np.clip(omega / MAX_ANGULAR_RPS,  -1.0, 1.0)
    sin_h   = math.sin(heading)
    cos_h   = math.cos(heading)
    rx_n    = np.clip(rx / FIELD_LENGTH, 0.0, 1.0)
    ry_n    = np.clip(ry / FIELD_WIDTH,  0.0, 1.0)
    prox    = np.clip(ray_distances / RAY_MAX_DISTANCE, 0.0, 1.0)

    return np.array([
        joy_rf_x, joy_rf_y, joy_rot,
        vx_n, vy_n, omega_n,
        sin_h, cos_h,
        rx_n, ry_n,
        *prox,
    ], dtype=np.float32)


def run(host: str) -> None:
    print(f"Loading policy from {MODEL_PATH} ...")
    model = SAC.load(MODEL_PATH)
    print("Policy loaded.")

    # ── NT setup ──────────────────────────────────────────────────────────────
    inst = ntcore.NetworkTableInstance.getDefault()
    inst.startClient4("teleop-assist-bridge")
    inst.setServer(host)
    print(f"Connecting to NT server at {host} ...")

    root        = inst.getTable("policy")
    inp         = root.getSubTable("input")
    out         = root.getSubTable("output")
    dbg         = root.getSubTable("debug")

    # Subscribers — Java publishes these
    sub_joy_y   = inp.getDoubleTopic("joy_left_y").subscribe(0.0)
    sub_joy_x   = inp.getDoubleTopic("joy_left_x").subscribe(0.0)
    sub_joy_rot = inp.getDoubleTopic("joy_right_x").subscribe(0.0)
    sub_heading = inp.getDoubleTopic("heading").subscribe(0.0)
    sub_vx      = inp.getDoubleTopic("vx").subscribe(0.0)
    sub_vy      = inp.getDoubleTopic("vy").subscribe(0.0)
    sub_omega   = inp.getDoubleTopic("omega").subscribe(0.0)
    sub_rx      = inp.getDoubleTopic("rx").subscribe(0.0)
    sub_ry      = inp.getDoubleTopic("ry").subscribe(0.0)

    # Publishers — Python writes these, Java reads them
    pub_out_vx    = out.getDoubleTopic("vx").publish()
    pub_out_vy    = out.getDoubleTopic("vy").publish()
    pub_out_omega = out.getDoubleTopic("omega").publish()
    pub_hb        = root.getDoubleTopic("heartbeat").publish()

    prox_pubs = [dbg.getDoubleTopic(label).publish() for label in _PROX_LABELS]

    print("Bridge running — press Ctrl+C to stop.\n")

    while True:
        t0 = time.monotonic()

        # ── Read robot state ──────────────────────────────────────────────────
        joy_y   = sub_joy_y.get()
        joy_x   = sub_joy_x.get()
        joy_rot = sub_joy_rot.get()
        heading = sub_heading.get()
        vx      = sub_vx.get()
        vy      = sub_vy.get()
        omega   = sub_omega.get()
        rx      = sub_rx.get()
        ry      = sub_ry.get()

        # ── Short-circuit: zero joystick → stop immediately, skip inference ──
        # The policy hasn't learned stillness reliably yet. Bypassing inference
        # when the driver isn't commanding movement prevents ghost motion while
        # training continues. Remove this block once the policy converges.
        joy_mag = math.hypot(joy_y, joy_x)
        if joy_mag < DRIFT_FLOOR:
            pub_out_vx   .set(0.0)
            pub_out_vy   .set(0.0)
            pub_out_omega.set(joy_rot * MAX_ANGULAR_RPS)
            pub_hb       .set(time.monotonic())
            elapsed = time.monotonic() - t0
            if DT - elapsed > 0:
                time.sleep(DT - elapsed)
            continue

        # ── Raycasting (uses field obstacles from lib/field_constants.py) ──────
        ray_distances = cast_rays(rx, ry, heading)

        # ── Rotate joy to robot frame ─────────────────────────────────────────
        joy_rf_x, joy_rf_y = _rotate_to_robot_frame(joy_y, joy_x, heading)

        # ── Build observation ─────────────────────────────────────────────────
        obs = _build_obs(
            joy_rf_x, joy_rf_y, joy_rot,
            vx, vy, omega,
            heading,
            rx, ry,
            ray_distances,
        )

        # ── Inference ─────────────────────────────────────────────────────────
        action, _ = model.predict(obs, deterministic=True)

        out_vx    = float(action[0]) * MAX_SPEED_MPS
        out_vy    = float(action[1]) * MAX_SPEED_MPS
        out_omega = float(action[2]) * MAX_ANGULAR_RPS

        # ── Publish ───────────────────────────────────────────────────────────
        pub_out_vx   .set(out_vx)
        pub_out_vy   .set(out_vy)
        pub_out_omega.set(out_omega)
        pub_hb       .set(time.monotonic())

        for i, pub in enumerate(prox_pubs):
            pub.set(float(ray_distances[i]))

        # ── 50 Hz pacing ──────────────────────────────────────────────────────
        elapsed = time.monotonic() - t0
        sleep   = DT - elapsed
        if sleep > 0:
            time.sleep(sleep)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Teleop-assist NT bridge")
    parser.add_argument(
        "--host", default="127.0.0.1",
        help="NT server address (default: 127.0.0.1 for sim; use 10.15.7.2 for real robot)"
    )
    args = parser.parse_args()

    try:
        run(args.host)
    except KeyboardInterrupt:
        print("\nBridge stopped.")
