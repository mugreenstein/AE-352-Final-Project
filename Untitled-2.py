# %%
import numpy as np

mtot = 0.943 #kg
m_motor = .052 #kg
m_prop = .01 #kg
mr = m_motor + m_prop # kg
md = mtot - 4*(m_motor + m_prop) #kg
L = .225 #m
I = (m_motor + m_prop) * L**2 #kg*m^2
h1 = 1 #m
g = 9.81        # gravity
rho = 1.225         # air density
k_drag = 1e-6 # drag coefficient
Ixx = 2 * I
Iyy = 2 * I
Izz = 4 * I
# pick hover rotor speed (guess or from data sheet)
omega_hover = 400.0  # rad/s, example

# thrust coefficient from hover condition: 4 k ω_h^2 = m g
k_thrust = mtot * g / (4 * omega_hover**2)

params = {
    "mtot": mtot,
    "md": md,
    "mr": mr,
    "I": I,
    "h1": h1,
    "g": g,
    "rho": rho,
    "L": L,
    "omega_hover": omega_hover,
    "k_thrust": k_thrust,
    "k_drag": k_drag,
    "Ixx": Ixx,
    "Iyy": Iyy,
    "Izz": Izz,
}




# %%
from sympy import symbols, Matrix, cos, sin

phi, theta, psi = symbols('phi theta psi')

R = Matrix([
    [cos(phi)*cos(psi) - cos(theta)*sin(phi)*sin(psi),
     -cos(psi)*sin(phi) - cos(phi)*cos(theta)*sin(psi),
     sin(theta)*sin(psi)],
    
    [cos(theta)*cos(psi)*sin(phi) + cos(phi)*sin(psi),
     cos(phi)*cos(psi) - sin(phi)*sin(psi),
     -cos(psi)*sin(theta)],
    
    [sin(phi)*sin(theta),
     cos(phi)*sin(theta),
     cos(theta)]
])

# %%
import numpy as np

def euler_rates(phi, theta, psi, p, q, r):
    """
    Returns [phi_dot, theta_dot, psi_dot] for 3-2-1 Euler angles.
    """

    phi_dot   = p + q*np.sin(phi)*np.tan(theta) + r*np.cos(phi)*np.tan(theta)
    theta_dot =     q*np.cos(phi)               - r*np.sin(phi)
    psi_dot   =     q*np.sin(phi)/np.cos(theta) + r*np.cos(phi)/np.cos(theta)

    return np.array([phi_dot, theta_dot, psi_dot])


# %%
import numpy as np
from sympy import *

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



# %%
# INITIAL CONDITIONS
dt = 0.005
start_time = 0
end_time = 10
times = np.arange(start_time, end_time + dt, dt)

x     = np.array([0., 0., 10.])
xdot  = np.zeros(3)
angles = np.zeros(3)   # [phi, theta, psi]

# random disturbance
deviation = 100
omega = np.deg2rad(2*np.random.rand(3)*deviation - deviation)  # [p,q,r]

# Assume rotor speeds squared (inputs) — placeholder
def input_func(t):
    return np.array([400**2, 400**2, 400**2, 400**2], dtype=float)

# SIMULATION LOOP
for t in times:

    omega_sq = input_func(t)

    # linear acceleration
    a = acceleration(omega_sq, angles, xdot, m, g, kT)

    # angular acceleration
    omegadot = angular_acceleration(omega_sq, omega, I, L, b, kT)

    # integrate rotational motion
    omega = omega + dt * omegadot
    angles = angles + dt * euler_rates(*angles, *omega)

    # integrate linear motion
    xdot = xdot + dt * a
    x = x + dt * xdot


# %% [markdown]
# Main Jupyter Notebook
# 
# Use to run code from source folder, and plot results
# 

# %% [markdown]
# # INITIAL CONDITIONS
# x     = np.array([0., 0., 10.])
# xdot  = np.zeros(3)
# angles = np.zeros(3)   # [phi, theta, psi]
# 
# # random disturbance
# deviation = 100
# omega = np.deg2rad(2*np.random.rand(3)*deviation - deviation)  # [p,q,r]
# 
# # Assume rotor speeds squared (inputs) — placeholder
# def input_func(t):
#     return np.array([400**2, 400**2, 400**2, 400**2], dtype=float)
# 
# # SIMULATION LOOP
# for t in times:
# 
#     omega_sq = input_func(t)
# 
#     # linear acceleration
#     a = acceleration(omega_sq, angles, xdot, m, g, kT)
# 
#     # angular acceleration
#     omegadot = angular_acceleration(omega_sq, omega, I, L, b, kT)
# 
#     # integrate rotational motion
#     omega = omega + dt * omegadot
#     angles = angles + dt * euler_rates(*angles, *omega)
# 
#     # integrate linear motion
#     xdot = xdot + dt * a
#     x = x + dt * xdot
# 

# %% [markdown]
# 


