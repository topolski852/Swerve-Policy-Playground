# ──────────────────────────────────────────────────────────────────────────────
# lib/accel_limiter.py
# Python port of 1507Base's SwerveAccelLimiter (translation only).
#   ../1507Base/src/main/java/org/team1507/lib/core/swerve/SwerveAccelLimiter.java
#
# The real robot runs every drive command through this limiter, so a policy
# trained without it learns starts and stops the robot can't make.
#
# Two limits, applied in this order each 20 ms loop, on FIELD-relative velocity:
#   1. Slip   (m/s²): the change in velocity, in any direction, is capped so
#                     the wheels keep grip. Speeding up, slowing down, turning.
#   2. Torque (m/s² from standstill): speeding up is capped by what the motors
#                     can deliver; the cap shrinks to zero at top speed.
# The Java version also limits turning rate (angular); omega isn't trained yet.
# ──────────────────────────────────────────────────────────────────────────────

import math


class AccelLimiter:

    def __init__(self, slip_accel, torque_accel, max_speed, dt):
        self.slip_accel   = slip_accel
        self.torque_accel = torque_accel
        self.max_speed    = max_speed
        self.dt           = dt
        self.reset()

    def reset(self, vx=0.0, vy=0.0):
        self._last_vx = vx
        self._last_vy = vy

    def limit(self, vx, vy):
        """Returns (vx, vy) limited so it changes no faster than the limits allow."""
        last_vx, last_vy = self._last_vx, self._last_vy

        # 1. Slip: cap the size of the velocity change, in any direction.
        dx, dy     = vx - last_vx, vy - last_vy
        max_change = self.slip_accel * self.dt
        change     = math.hypot(dx, dy)
        if change > max_change:
            k  = max_change / change
            vx = last_vx + k * dx
            vy = last_vy + k * dy

        # 2. Torque: cap speeding up; less is available the faster the robot goes.
        last_speed = math.hypot(last_vx, last_vy)
        speed      = math.hypot(vx, vy)
        available  = max(0.0, self.torque_accel * self.dt * (1.0 - last_speed / self.max_speed))
        if speed - last_speed > available and speed > 1e-6:
            k   = (last_speed + available) / speed
            vx *= k
            vy *= k

        self._last_vx, self._last_vy = vx, vy
        return vx, vy
