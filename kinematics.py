# ──────────────────────────────────────────────────────────────────────────────
# kinematics.py
# Swerve inverse kinematics and physics integration.
# Matches WPILib SwerveDriveKinematics behavior.
# ──────────────────────────────────────────────────────────────────────────────

import math
from constants import (
    MODULE_OFFSETS, MAX_SPEED_MPS, VELOCITY_ALPHA,
    SPEED_DEADBAND, DT
)


def discretize(vx: float, vy: float, omega: float, dt: float):
    """
    Port of WPILib ChassisSpeeds.discretize().

    Corrects for the arc a robot traces when it translates and rotates
    simultaneously within a single timestep. Without this, the robot drifts
    off a straight line when omega != 0.

    Returns corrected (vx, vy, omega) — omega is unchanged.
    """
    if abs(omega) < 1e-9:
        return vx, vy, omega

    # Desired delta pose in robot frame over dt
    dx     = vx * dt
    dy     = vy * dt
    dtheta = omega * dt

    # SE(2) log map: compute the twist whose exponential equals (dx, dy, dtheta)
    cos_dt  = math.cos(dtheta)
    sin_dt  = math.sin(dtheta)
    cos_m1  = cos_dt - 1.0          # cos(dtheta) - 1

    if abs(cos_m1) < 1e-9:
        # Near zero rotation — sinc approximation avoids divide-by-zero
        half_theta_by_tan = 1.0 - (dtheta * dtheta) / 12.0
    else:
        half_theta = dtheta / 2.0
        half_theta_by_tan = -(half_theta * sin_dt) / cos_m1

    # Rotate the translation part by the log-map correction angle
    corr_angle = math.atan2(-half_theta, half_theta_by_tan)
    corr_mag   = math.hypot(half_theta_by_tan, half_theta)

    rot_cos = math.cos(corr_angle)
    rot_sin = math.sin(corr_angle)

    tx = (dx * rot_cos - dy * rot_sin) * corr_mag
    ty = (dx * rot_sin + dy * rot_cos) * corr_mag

    return tx / dt, ty / dt, omega


def swerve_ik(vx: float, vy: float, omega: float,
              module_offsets=None, max_speed=MAX_SPEED_MPS):
    """
    Swerve inverse kinematics: chassis velocity → per-module (angle, speed).

    Inputs are in ROBOT frame (x=forward, y=left).
    Returns a list of (angle_rad, speed_mps) for each module,
    in WPILib order: FL, FR, BL, BR.

    Applies wheel-speed desaturation so no module exceeds max_speed while
    preserving the translational-to-rotational ratio (WPILib behavior).
    """
    if module_offsets is None:
        module_offsets = MODULE_OFFSETS

    states = []
    for (rx, ry) in module_offsets:
        # Per-module velocity vector (robot frame)
        vx_m = vx - omega * ry
        vy_m = vy + omega * rx
        speed = math.hypot(vx_m, vy_m)
        angle = math.atan2(vy_m, vx_m)
        states.append([angle, speed])

    # Desaturate: scale all speeds proportionally if any exceeds max_speed
    max_s = max(s for _, s in states)
    if max_s > max_speed:
        scale = max_speed / max_s
        states = [[a, s * scale] for a, s in states]

    return [(a, s) for a, s in states]


def integrate_pose(x: float, y: float, heading: float,
                   vx: float, vy: float, omega: float,
                   dt: float = DT):
    """
    Integrate robot pose one timestep forward.

    vx, vy, omega are in ROBOT frame; integration converts to world frame.
    Returns (new_x, new_y, new_heading).
    """
    cos_h = math.cos(heading)
    sin_h = math.sin(heading)

    # Rotate robot-frame velocity into world frame
    vx_w = vx * cos_h - vy * sin_h
    vy_w = vx * sin_h + vy * cos_h

    new_x       = x       + vx_w * dt
    new_y       = y       + vy_w * dt
    new_heading = heading + omega * dt   # wrapped by caller if needed

    return new_x, new_y, new_heading


class SwerveState:
    """
    Holds the full robot state and steps it forward each control cycle.

    Applies ChassisSpeeds.discretize() and first-order velocity lag
    before computing module states, matching real WPILib behavior.
    """

    def __init__(self, x=0.0, y=0.0, heading=0.0):
        self.x       = x
        self.y       = y
        self.heading = heading

        # Actual (lagged) chassis velocities, robot frame
        self.vx    = 0.0
        self.vy    = 0.0
        self.omega = 0.0

        # Last module states for anti-jitter and smoothness penalty
        self._last_module_angles = [0.0] * 4

    @property
    def module_states(self):
        """Current module (angle, speed) tuple list — FL, FR, BL, BR."""
        return swerve_ik(self.vx, self.vy, self.omega)

    def step(self, cmd_vx: float, cmd_vy: float, cmd_omega: float):
        """
        Advance one 20 ms step.

        cmd_* are the commanded chassis velocities in robot frame.
        Returns list of (angle_rad, speed_mps) per module after the step.
        """
        # 1. Apply ChassisSpeeds.discretize() for skew correction
        d_vx, d_vy, d_omega = discretize(cmd_vx, cmd_vy, cmd_omega, DT)

        # 2. First-order velocity lag (simulates acceleration limits)
        self.vx    = VELOCITY_ALPHA * d_vx    + (1.0 - VELOCITY_ALPHA) * self.vx
        self.vy    = VELOCITY_ALPHA * d_vy    + (1.0 - VELOCITY_ALPHA) * self.vy
        self.omega = VELOCITY_ALPHA * d_omega + (1.0 - VELOCITY_ALPHA) * self.omega

        # 3. Compute module states with anti-jitter angle hold
        raw_states = swerve_ik(self.vx, self.vy, self.omega)
        module_states = []
        for i, (angle, speed) in enumerate(raw_states):
            if speed < SPEED_DEADBAND:
                angle = self._last_module_angles[i]   # hold last angle
            else:
                self._last_module_angles[i] = angle
            module_states.append((angle, speed))

        # 4. Integrate pose
        self.x, self.y, self.heading = integrate_pose(
            self.x, self.y, self.heading,
            self.vx, self.vy, self.omega
        )

        return module_states

    def reset(self, x=0.0, y=0.0, heading=0.0):
        self.x, self.y, self.heading = x, y, heading
        self.vx = self.vy = self.omega = 0.0
        self._last_module_angles = [0.0] * 4
