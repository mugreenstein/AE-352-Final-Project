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


