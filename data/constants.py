#from sympy import *
#phi, theta, psi = symbols('phi theta psi')

#Rotation Matrix (ZYZ Euler Angles)
#Rz = Matrix([[cos(psi), -sin(psi), 0],
              #[sin(psi), cos(psi), 0],
              #[0, 0, 1]])
#Ry = Matrix([[cos(theta), 0, sin(theta)],
              #[0, 1, 0],
              #[-sin(theta), 0, cos(theta)]])
#Rx = Matrix([[1, 0, 0],
            #[0, cos(phi), -sin(phi)],
            #[0, sin(phi),  cos(phi)]])
#R1 = Rz * Ry * Rx
#print(R)


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

print(R)

##############
import numpy as np

mtot = 0.943 #kg
m_motor = .052 #kg
m_prop = .01 #kg
mr = m_motor + m_prop # kg
md = mtot - 4*(m_motor + m_prop) #kg
r = .225 #m
I = (m_motor + m_prop) * r**2 #kg*m^2
h1 = 1 #m
g = 9.81        # gravity
rho = 1.225         # air density

# pick hover rotor speed (guess or from data sheet)
omega_hover = 400.0  # rad/s, example

# thrust coefficient from hover condition: 4 k ω_h^2 = m g
k = mtot * g / (4 * omega_hover**2)

params = {
    "mtot": mtot,
    "md": md,
    "mr": mr,
    "I": I,
    "h1": h1,
    "g": g,
    "rho": rho,
    "omega_hover": omega_hover,
    "k": k
}


