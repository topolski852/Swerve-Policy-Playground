"""
verify_kinematics.py
Run this before training to confirm the swerve IK is correct.
"""

import math
from kinematics import swerve_ik, SwerveState
from constants import MAX_SPEED_MPS

def check(label, got, expected, tol=0.01):
    ok = all(abs(g - e) < tol for g, e in zip(got, expected))
    tag = "PASS" if ok else "FAIL"
    print(f"  [{tag}]  {label}")
    if not ok:
        print(f"         got:      {[round(v,4) for v in got]}")
        print(f"         expected: {[round(v,4) for v in expected]}")

print("\n--- swerve_ik sanity checks ---")

# Pure forward (vx=1, vy=0, omega=0) -> all modules at 0 rad
states = swerve_ik(1.0, 0.0, 0.0)
angles = [a for a, _ in states]
check("Pure forward  -> all modules at 0 deg",
      angles, [0.0, 0.0, 0.0, 0.0])

# Pure left strafe (vx=0, vy=1, omega=0) -> all modules at 90 deg
states = swerve_ik(0.0, 1.0, 0.0)
angles = [a for a, _ in states]
check("Pure strafe   -> all modules at 90 deg",
      angles, [math.pi/2]*4)

# Diagonal (vx=1, vy=1, omega=0) -> all modules at 45 deg
states = swerve_ik(1.0, 1.0, 0.0)
angles = [a for a, _ in states]
check("Diagonal      -> all modules at 45 deg",
      angles, [math.pi/4]*4)

# Rotate in place (vx=0, vy=0, omega=1) -> modules tangent to orbit radius
# FL (+x,+y): vx_m=-ry=-0.31, vy_m=+rx=+0.31  -> 135 deg
# FR (+x,-y): vx_m=+0.31,     vy_m=+0.31       ->  45 deg
# BL (-x,+y): vx_m=-0.31,     vy_m=-0.31       -> -135 deg
# BR (-x,-y): vx_m=+0.31,     vy_m=-0.31       ->  -45 deg
states = swerve_ik(0.0, 0.0, 1.0)
angles = [a for a, _ in states]
expected_rot = [
    math.atan2( 0.31, -0.31),   # FL
    math.atan2( 0.31,  0.31),   # FR
    math.atan2(-0.31, -0.31),   # BL
    math.atan2(-0.31,  0.31),   # BR
]
check("Rotate in place -> modules tangent to orbit", angles, expected_rot)

# Speed desaturation: commanding above max should clamp all modules
states = swerve_ik(MAX_SPEED_MPS, MAX_SPEED_MPS, 0.0)
speeds = [s for _, s in states]
ok = all(s <= MAX_SPEED_MPS + 0.001 for s in speeds)
print(f"  [{'PASS' if ok else 'FAIL'}]  Desaturation -> max module speed {max(speeds):.3f} m/s  (limit {MAX_SPEED_MPS})")

print("\n--- SwerveState integration checks ---")

# 1 s of pure forward should move x, no y drift
state = SwerveState(x=0.0, y=0.0, heading=0.0)
for _ in range(50):     # 50 * 0.02 s = 1 s
    state.step(1.0, 0.0, 0.0)

print(f"  After 1 s at vx=1: x={state.x:.3f} m (expect ~0.7-0.9 due to lag), "
      f"y={state.y:.4f} m (expect ~0)")
y_ok = abs(state.y) < 0.01
print(f"  [{'PASS' if y_ok else 'FAIL'}]  No lateral drift during pure forward drive")

# 1 s of pure rotation should not translate
state.reset()
for _ in range(50):
    state.step(0.0, 0.0, 1.0)

xy_ok = abs(state.x) < 0.01 and abs(state.y) < 0.01
print(f"  After 1 s at omega=1: heading={state.heading:.3f} rad (expect ~0.4-0.6), "
      f"x={state.x:.4f}, y={state.y:.4f}")
print(f"  [{'PASS' if xy_ok else 'FAIL'}]  No translation during pure rotation")

print()
