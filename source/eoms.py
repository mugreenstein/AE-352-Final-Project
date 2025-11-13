import numpy as np
from sympy import *
from data.initial_conditions import constants

# EOMs for the dynamical system
def skycrane(u):
    """
    Returns the derivative of the state function using the
    equations of motion for the system without damping.
    """
    x, xdot, theta, thetadot = u 
    m1, m2, L, k, b, g = constants

    A = Matrix([[m1+m2, m2*L*cos(theta)],
                [cos(theta)/L, 1]])
    B = Matrix([[m2*L*thetadot**2*sin(theta)-k*x],
                [-g/L * sin(theta)]])

    O = A.inv() @ B
    xddot = O[0]
    thetaddot = O[1]
    udot = np.array([xdot, xddot, thetadot, thetaddot])
    return udot

def skycrane_damping(u):
    """
    Returns the derivative of the state function using the
    equations of motion for the system with damping.
    """
    x, xdot, theta, thetadot = u 
    m1, m2, L, k, b, g = constants

    A = Matrix([[m1+m2, m2*L*cos(theta)],
                [cos(theta)/L, 1]])
    B = Matrix([[m2*L*thetadot**2*sin(theta)-k*x-b*xdot],
                [-g/L * sin(theta)]])

    O = A.inv() @ B
    xddot = O[0]
    thetaddot = O[1]
    udot = np.array([xdot, xddot, thetadot, thetaddot])
    return udot

def set_constants(new_constants):
    """
    Update the module-level 'constants' tuple so skycrane() and
    skycrane_damping() pick up the current (m1, m2, L, k, b, g).
    """
    global constants
    constants = new_constants