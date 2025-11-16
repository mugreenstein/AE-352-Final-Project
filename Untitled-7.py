# %%
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Physical parameters
mtot = 0.943
m_motor = 0.052
m_prop  = 0.01
mr = m_motor + m_prop
md = mtot - 4*(m_motor + m_prop)
L = 0.225
I_single = (m_motor + m_prop) * L**2

h1 = 1
g = 9.81
rho = 1.225
k_drag = 1e-6

Ixx = 2 * I_single
Iyy = 2 * I_single
Izz = 4 * I_single

omega_hover = 400
k_thrust = mtot * g / (4 * omega_hover**2)

params = {
    "mtot": mtot, "md": md, "mr": mr,
    "I": I_single, "h1": h1, "g": g, "rho": rho,
    "L": L, "omega_hover": omega_hover,
    "k_thrust": k_thrust, "k_drag": k_drag,
    "Ixx": Ixx, "Iyy": Iyy, "Izz": Izz
}


# %%
battery_voltage = 11.1
battery_capacity_Ah = 1.5
battery_energy_J = battery_voltage * battery_capacity_Ah * 3600.0

R_rotor = 0.10
A_disk = np.pi * R_rotor**2

# ideal induced power approximately
cP_rotor = (k_thrust**1.5) / np.sqrt(2 * rho * A_disk)

params.update({
    "battery_voltage": battery_voltage,
    "battery_capacity_Ah": battery_capacity_Ah,
    "battery_energy_J": battery_energy_J,
    "R_rotor": R_rotor,
    "A_disk": A_disk,
    "cP_rotor": cP_rotor
})

# %%
def rotation(phi, theta, psi):
    cphi, sphi = np.cos(phi), np.sin(phi)
    cth,  sth  = np.cos(theta), np.sin(theta)
    cpsi, spsi = np.cos(psi), np.sin(psi)

    return np.array([
        [cpsi*cth,  spsi*cth,   -sth],
        [cpsi*sth*sphi - spsi*cphi,   spsi*sth*sphi + cpsi*cphi,   cth*sphi],
        [cpsi*sth*cphi + spsi*sphi,   spsi*sth*cphi - cpsi*sphi,   cth*cphi]
    ])

def euler_rates(phi, theta, psi, p, q, r):
    phi_dot   = p + q*np.sin(phi)*np.tan(theta) + r*np.cos(phi)*np.tan(theta)
    theta_dot =     q*np.cos(phi)               - r*np.sin(phi)
    psi_dot   =     q*np.sin(phi)/np.cos(theta) + r*np.cos(phi)/np.cos(theta)
    return np.array([phi_dot, theta_dot, psi_dot])

def thrust_body(omegas):
    kT = params["k_thrust"]
    return np.array([0, 0, kT * np.sum(omegas**2)])

def torque_body(omegas):
    L = params["L"]
    kT = params["k_thrust"]
    b = params["k_drag"]

    w1, w2, w3, w4 = omegas
    tau_phi   = L * kT * (w1**2 - w3**2)
    tau_theta = L * kT * (w2**2 - w4**2)
    tau_psi   = b * (w1**2 - w2**2 + w3**2 - w4**2)
    return np.array([tau_phi, tau_theta, tau_psi])


# %%
def rotor_power(omega):
    return params["cP_rotor"] * omega**3

def total_power(omegas):
    return np.sum([rotor_power(w) for w in omegas])


# %%
def total_thrust(omegas):
    return params["k_thrust"] * np.sum(omegas**2)

def acceleration(omegas, angles, xdot, params):
    m = params["mtot"]
    g = params["g"]
    phi, theta, psi = angles
    R = rotation(phi, theta, psi)

    T = total_thrust(omegas)
    T_I = R @ np.array([0, 0, T])
    gravity = np.array([0, 0, -g])
    return gravity + T_I/m

def angular_acceleration(omega_sq, omega, params):
    p, q, r = omega
    Ixx, Iyy, Izz = params["Ixx"], params["Iyy"], params["Izz"]
    tau_phi, tau_theta, tau_psi = torque_body(omega_sq)

    p_dot = (tau_phi/Ixx)  - ((Iyy - Izz)/Ixx)*q*r
    q_dot = (tau_theta/Iyy) - ((Izz - Ixx)/Iyy)*p*r
    r_dot = (tau_psi/Izz)  - ((Ixx - Iyy)/Izz)*p*q

    return np.array([p_dot, q_dot, r_dot])


# %%
def motor_speeds_from_thrust_torques(T, tau_phi, tau_theta, tau_psi):
    kT = params["k_thrust"]
    L  = params["L"]
    b  = params["k_drag"]

    A = np.array([
        [ kT,      kT,      kT,      kT     ],
        [ L*kT,    0.0,    -L*kT,    0.0    ],
        [ 0.0,     L*kT,    0.0,    -L*kT   ],
        [ b,      -b,       b,      -b      ]
    ])
    u = np.array([T, tau_phi, tau_theta, tau_psi])

    omega_sq = np.linalg.solve(A, u)
    omega_sq = np.clip(omega_sq, 0.0, None)
    return np.sqrt(omega_sq)


# %%
def hover_input(t):
    ω = params["omega_hover"]
    return np.array([ω, ω, ω, ω])


# %%
def simulate_hover(dt=0.01, T=120):
    times = np.arange(0, T, dt)

    x = np.array([0.,0.,1.])
    xdot = np.zeros(3)
    angles = np.zeros(3)
    omega_body = np.zeros(3)

    Xs = []
    Angs = []
    Omegas = []

    for t in times:
        omegas = hover_input(t)

        a = acceleration(omegas, angles, xdot, params)
        omegadot = angular_acceleration(omegas, omega_body, params)

        omega_body += dt * omegadot
        angles     += dt * euler_rates(*angles, *omega_body)
        xdot       += dt * a
        x          += dt * xdot

        Xs.append(x.copy())
        Angs.append(angles.copy())
        Omegas.append(omega_body.copy())

    return times, np.array(Xs), np.array(Angs), np.array(Omegas)

t_hov, Xs_hov, Angs_hov, Omegas_hov = simulate_hover()


# %%
def simulate_hover_power(T=120.0, dt=0.01):
    times = np.arange(0, T, dt)

    x = np.array([0.0, 0.0, 1.0])
    xdot = np.zeros(3)
    angles = np.zeros(3)
    omega_body = np.zeros(3)

    E_used = 0.0
    E_batt = params["battery_energy_J"]

    Xs = []
    Angs = []
    Powers = []
    Energies = []

    for t in times:
        omegas = hover_input(t)

        # dynamics
        a = acceleration(omegas, angles, xdot, params)
        omegadot = angular_acceleration(omegas, omega_body, params)

        omega_body += dt * omegadot
        angles     += dt * euler_rates(*angles, *omega_body)
        xdot       += dt * a
        x          += dt * xdot

        # power / energy
        P = total_power(omegas)
        E_used += P * dt

        Xs.append(x.copy())
        Angs.append(angles.copy())
        Powers.append(P)
        Energies.append(E_used)

        if E_used >= E_batt:
            break

    return (
        np.array(times[:len(Xs)]),
        np.array(Xs),
        np.array(Angs),
        np.array(Powers),
        np.array(Energies)
    )

times_h, Xs_h, Angles_h, Powers_h, Energies_h = simulate_hover_power()


# %% [markdown]
# SIMULATE HOVER!

# %%
# ---------------------------------------------------------
# HOVER INPUT FUNCTION
# ---------------------------------------------------------
def hover_input(t):
    """
    Constant rotor speeds equal to hover RPM.
    From: 4*k_thrust*omega_hover^2 = m*g
    """
    omega_h = params["omega_hover"]
    return np.array([omega_h, omega_h, omega_h, omega_h])


# %%
def simulate_hover_power(T=120.0, dt=0.005):
    times = np.arange(0.0, T+dt, dt)

    x = np.array([0.0, 0.0, 1.0])
    xdot = np.zeros(3)
    angles = np.zeros(3)
    omega_body = np.zeros(3)

    Xs, Angles, Omegas = [], [], []
    Powers = []
    Energies = []

    E_used = 0.0   # cumulative joules
    E_batt = params["battery_energy_J"]

    for t in times:
        omegas = hover_input(t)

        # dynamics
        a = acceleration(omegas, angles, xdot, params)
        omegadot = angular_acceleration(omegas, omega_body, params)

        omega_body += dt * omegadot
        angles     += dt * euler_rates(*angles, *omega_body)
        xdot       += dt * a
        x          += dt * xdot

        # power + energy
        P = total_power(omegas)
        E_used += P * dt

        # stop if battery dies
        if E_used >= E_batt:
            print(f"🔋 Battery depleted at t = {t:.2f} seconds")
            break

        # log
        Xs.append(x.copy())
        Angles.append(angles.copy())
        Omegas.append(omega_body.copy())
        Powers.append(P)
        Energies.append(E_used)

    return (
        np.array(times[:len(Xs)]),
        np.array(Xs),
        np.array(Angles),
        np.array(Omegas),
        np.array(Powers),
        np.array(Energies)
    )


# %%
# ============================================================
#                    POWER MODEL PLOTS
# ============================================================

# ---------------------- Altitude ----------------------
plt.figure(figsize=(10,5))
plt.plot(times_h, Xs_h[:,2], label="z(t)")
plt.axhline(1.0, color='r', linestyle='--', label="Target 1 m")
plt.title("Hover Altitude (With Power Model)")
plt.xlabel("Time (s)")
plt.ylabel("z (m)")
plt.grid(True)
plt.legend()
plt.show()


# ---------------------- Euler Angles ----------------------
plt.figure(figsize=(10,5))
plt.plot(times_h, Angles_h[:,0], label="phi")
plt.plot(times_h, Angles_h[:,1], label="theta")
plt.plot(times_h, Angles_h[:,2], label="psi")
plt.legend()
plt.grid(True)
plt.title("Euler Angles During Powered Hover")
plt.xlabel("Time (s)")
plt.ylabel("Angle (rad)")
plt.ylim(-0.01, 0.01)
plt.show()


# ---------------- Euler Angles (1e-16 zoom) ----------------
plt.figure(figsize=(10,5))
plt.plot(times_h, Angles_h[:,0], label="phi")
plt.plot(times_h, Angles_h[:,1], label="theta")
plt.plot(times_h, Angles_h[:,2], label="psi")
plt.legend()
plt.grid(True)
plt.title("Euler Angles During Powered Hover (Zoomed to Numerical Noise)")
plt.xlabel("Time (s)")
plt.ylabel("Angle (rad)")
plt.ylim(-2e-16, 2e-16)
plt.show()


# ---------------------- Power Consumption ----------------------
plt.figure(figsize=(10,5))
plt.plot(times_h, Powers_h, color='blue')
plt.ylabel("Power (W)")
plt.xlabel("Time (s)")
plt.title("Hover Power Consumption")
plt.grid(True)
plt.show()


# ---------------------- Energy Used vs Battery ----------------------
E_batt = params["battery_energy_J"]

plt.figure(figsize=(10,5))
plt.plot(times_h, Energies_h, label="Energy Used [J]", color='purple')
plt.axhline(E_batt, color='red', linestyle='--', label="Battery Capacity")
plt.ylabel("Energy (J)")
plt.xlabel("Time (s)")
plt.title("Energy Usage During Hover")
plt.grid(True)
plt.legend()
plt.show()


# ---------------------- Battery Remaining ----------------------
batt_remaining = np.maximum(E_batt - Energies_h, 0)

plt.figure(figsize=(10,5))
plt.plot(times_h, batt_remaining, color='green', label="Remaining Energy")
plt.axhline(0, color='red', linestyle='--')
plt.ylabel("Energy (J)")
plt.xlabel("Time (s)")
plt.title("Battery Remaining Over Time")
plt.grid(True)
plt.legend()
plt.show()


# ---------------------- Battery Remaining (Zoomed) ----------------------
plt.figure(figsize=(12,5))
plt.plot(times_h, batt_remaining, color='green', linewidth=2, label="Remaining Energy")
plt.title("Battery Remaining Over Time (Zoomed In)")
plt.xlabel("Time (s)")
plt.ylabel("Energy (J)")

plt.ylim(batt_remaining.min() - 500, batt_remaining.max() + 500)

plt.grid(True)
plt.legend()
plt.show()


# %%
# ============================================================
#                    POWER MODEL PLOTS
# ============================================================

# ---------------------- Altitude ----------------------
plt.figure(figsize=(10,5))
plt.plot(times_h, Xs_h[:,2], label="z(t)")
plt.axhline(1.0, color='r', linestyle='--', label="Target 1 m")
plt.title("Hover Altitude (With Power Model)")
plt.xlabel("Time (s)")
plt.ylabel("z (m)")
plt.grid(True)
plt.legend()
plt.show()


# ---------------------- Euler Angles ----------------------
plt.figure(figsize=(10,5))
plt.plot(times_h, Angles_h[:,0], label="phi")
plt.plot(times_h, Angles_h[:,1], label="theta")
plt.plot(times_h, Angles_h[:,2], label="psi")
plt.legend()
plt.grid(True)
plt.title("Euler Angles During Powered Hover")
plt.xlabel("Time (s)")
plt.ylabel("Angle (rad)")
plt.ylim(-0.01, 0.01)
plt.show()


# ---------------- Euler Angles (1e-16 zoom) ----------------
plt.figure(figsize=(10,5))
plt.plot(times_h, Angles_h[:,0], label="phi")
plt.plot(times_h, Angles_h[:,1], label="theta")
plt.plot(times_h, Angles_h[:,2], label="psi")
plt.legend()
plt.grid(True)
plt.title("Euler Angles During Powered Hover (Zoomed to Numerical Noise)")
plt.xlabel("Time (s)")
plt.ylabel("Angle (rad)")
plt.ylim(-2e-16, 2e-16)
plt.show()


# ---------------------- Power Consumption ----------------------
plt.figure(figsize=(10,5))
plt.plot(times_h, Powers_h, color='blue')
plt.ylabel("Power (W)")
plt.xlabel("Time (s)")
plt.title("Hover Power Consumption")
plt.grid(True)
plt.show()


# ---------------------- Energy Used vs Battery ----------------------
E_batt = params["battery_energy_J"]

plt.figure(figsize=(10,5))
plt.plot(times_h, Energies_h, label="Energy Used [J]", color='purple')
plt.axhline(E_batt, color='red', linestyle='--', label="Battery Capacity")
plt.ylabel("Energy (J)")
plt.xlabel("Time (s)")
plt.title("Energy Usage During Hover")
plt.grid(True)
plt.legend()
plt.show()


# ---------------------- Battery Remaining ----------------------
batt_remaining = np.maximum(E_batt - Energies_h, 0)

plt.figure(figsize=(10,5))
plt.plot(times_h, batt_remaining, color='green', label="Remaining Energy")
plt.axhline(0, color='red', linestyle='--')
plt.ylabel("Energy (J)")
plt.xlabel("Time (s)")
plt.title("Battery Remaining Over Time")
plt.grid(True)
plt.legend()
plt.show()


# ---------------------- Battery Remaining (Zoomed) ----------------------
plt.figure(figsize=(12,5))
plt.plot(times_h, batt_remaining, color='green', linewidth=2, label="Remaining Energy")
plt.title("Battery Remaining Over Time (Zoomed In)")
plt.xlabel("Time (s)")
plt.ylabel("Energy (J)")

plt.ylim(batt_remaining.min() - 500, batt_remaining.max() + 500)

plt.grid(True)
plt.legend()
plt.show()


# %%


# %%



