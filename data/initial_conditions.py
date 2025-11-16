# initial_conditions.py
import numpy as np
from source.constants import omega_hover

def make_initial_state():
    # positions (inertial)
    x0, y0, z0 = 0.0, 0.0, 0.0
    # velocities
    vx0, vy0, vz0 = 0.0, 0.0, 0.0
    # Euler angles
    phi0, theta0, psi0 = 0.0, 0.0, 0.0
    # angular rates
    p0, q0, r0 = 0.0, 0.0, 0.0
    # rotor speeds (hover guess)
    omega1_0 = omega_hover
    omega2_0 = omega_hover
    omega3_0 = omega_hover
    omega4_0 = omega_hover

    # pack into one state vector however you’ve defined it
    return np.array([
        x0, y0, z0,
        vx0, vy0, vz0,
        phi0, theta0, psi0,
        p0, q0, r0,
        omega1_0, omega2_0, omega3_0, omega4_0
    ])
