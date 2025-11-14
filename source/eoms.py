import numpy as np
from sympy import *
from source.constants import params
from source.math import R

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

# ----------------------------------------
#   Body Torques
# ----------------------------------------
def torque_body(omegas):
    """
    Compute roll, pitch, and yaw torques from rotor speeds.
    Using simplified quadcopter model.
    """
    L     = params["L"]        # this should really be your arm length
    kT    = params["k_thrust"]
    bdrag = params["k_drag"]

    omega1, omega2, omega3, omega4 = omegas

    # roll torque (x-body axis)
    tau_phi = L * kT * (omega1**2 - omega3**2)

    # pitch torque (y-body axis)
    tau_theta = L * kT * (omega2**2 - omega4**2)

    # yaw torque (z-body axis)
    tau_psi = bdrag * (omega1**2 - omega2**2 + omega3**2 - omega4**2)

    return np.array([tau_phi, tau_theta, tau_psi], dtype=float)


# ----------------------------------------
#   ROTATIONAL ACCELERATION (EOM)
# ----------------------------------------
def rotational_accel(state):
    """
    Computes body-frame angular accelerations [p_dot, q_dot, r_dot].
    state ordering:
    [x, y, z,
     vx, vy, vz,
     phi, theta, psi,
     p, q, r,
     omega1, omega2, omega3, omega4]
    """

    Ixx = params["Ixx"]
    Iyy = params["Iyy"]
    Izz = params["Izz"]

    # unpack body rates
    p, q, r = state[9:12]

    # rotor speeds
    omegas = state[-4:]

    # get torques from motors
    tau_phi, tau_theta, tau_psi = torque_body(omegas)

    # Euler rigid-body equations (diagonal inertia)
    p_dot = (tau_phi   / Ixx) - ((Iyy - Izz) / Ixx) * q * r
    q_dot = (tau_theta / Iyy) - ((Izz - Ixx) / Iyy) * p * r
    r_dot = (tau_psi   / Izz) - ((Ixx - Iyy) / Izz) * p * q

    return np.array([p_dot, q_dot, r_dot], dtype=float)

