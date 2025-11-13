import numpy as np
from sympy import *
from data.constants import params
# dynamics.py


# ----------------------------------------
#   THRUST IN BODY FRAME
# ----------------------------------------
def thrust_body(omegas):
    """
    Compute total thrust vector expressed in the BODY frame.
    Thrust = k * (ω1^2 + ω2^2 + ω3^2 + ω4^2)
    Direction: +body-z
    """
    k = params["k_thrust"]

    omega1, omega2, omega3, omega4 = omegas
    total_thrust = k * (omega1**2 + omega2**2 + omega3**2 + omega4**2)

    return np.array([0.0, 0.0, total_thrust])


# ----------------------------------------
#   TRANSLATIONAL ACCELERATION (EOM)
# ----------------------------------------
def translational_accel(state, R_body_to_inertial):
    """
    Computes inertial-frame linear acceleration.
    state = [x,y,z, vx,vy,vz, phi,theta,psi, p,q,r, ω1,ω2,ω3,ω4]
    """
    m = params["mtot"]
    g = params["g"]

    # unpack state
    x, y, z, vx, vy, vz, phi, theta, psi = state[:9]
    omegas = state[-4:]

    # rotate thrust from body -> inertial
    R = R_body_to_inertial(phi, theta, psi)
    T_B = thrust_body(omegas)

    # gravity (always inertial frame)
    F_g = np.array([0.0, 0.0, -m * g])

    # Newton's 2nd law
    a = (F_g + R @ T_B) / m

    return a





