
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
