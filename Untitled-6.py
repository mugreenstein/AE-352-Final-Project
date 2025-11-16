# %%
import numpy as np
import matplotlib.pyplot as plt

# Physical parameters
mtot = 0.943
m_motor = 0.052
m_prop  = 0.01
mr = m_motor + m_prop
md = mtot - 4*(m_motor + m_prop)
L = 0.225   # arm length
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
# ---------------------------------------------------------
# POWER + BATTERY PARAMETERS
# ---------------------------------------------------------

battery_voltage = 11.1              # V (typical 3S LiPo)
battery_capacity_Ah = 1.5           # 1500 mAh
battery_energy_J = battery_voltage * battery_capacity_Ah * 3600.0  # Joules

# Rotor disk assumptions for ideal induced power
R_rotor = 0.10
A_disk = np.pi * R_rotor**2

# Ideal induced power coefficient (from momentum theory)
# T = kT * sum(omega^2)
# P = T^(3/2) / sqrt(2 ρ A)
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


# %%
def euler_rates(phi, theta, psi, p, q, r):
    phi_dot   = p + q*np.sin(phi)*np.tan(theta) + r*np.cos(phi)*np.tan(theta)
    theta_dot =     q*np.cos(phi)               - r*np.sin(phi)
    psi_dot   =     q*np.sin(phi)/np.cos(theta) + r*np.cos(phi)/np.cos(theta)
    return np.array([phi_dot, theta_dot, psi_dot])


# %%
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
def rotor_power(omega, params=params):
    """Ideal induced rotor power P = cP * omega^3 per rotor."""
    return params["cP_rotor"] * omega**3

def total_power(omegas, params=params):
    """Sum of power from all 4 rotors."""
    return np.sum([rotor_power(w) for w in omegas])


# %%
def total_thrust(omegas, params=params):
    kT = params["k_thrust"]
    return kT * np.sum(omegas**2)

def acceleration(omegas, angles, xdot, params):
    m = params["mtot"]
    g = params["g"]

    phi, theta, psi = angles
    R = rotation(phi, theta, psi)

    # correct total thrust magnitude
    T = total_thrust(omegas, params)

    # thrust direction = body z mapped to world
    T_I = R @ np.array([0, 0, T])

    gravity = np.array([0, 0, -g])

    return gravity + T_I/m


def angular_acceleration(omega_sq, omega, params):
    p, q, r = omega
    Ixx, Iyy, Izz = params["Ixx"], params["Iyy"], params["Izz"]

    tau = torque_body(omega_sq)
    tau_phi, tau_theta, tau_psi = tau

    p_dot = (tau_phi   / Ixx) - ((Iyy - Izz)/Ixx) * q * r
    q_dot = (tau_theta / Iyy) - ((Izz - Ixx)/Iyy) * p * r
    r_dot = (tau_psi   / Izz) - ((Ixx - Iyy)/Izz) * p * q

    return np.array([p_dot, q_dot, r_dot])


# %%
dt = 0.005
start_time = 0
end_time = 120  # or however long you want
times = np.arange(start_time, end_time, dt)

# State storage
Xs = []
Vels = []
Angles = []
Omegas = []

# Initial conditions
x = np.array([0., 0., 0.])
xdot = np.zeros(3)
angles = np.zeros(3)
omega = np.zeros(3)    # NO random disturbance

def input_func(t):
    return np.array([400, 400, 400, 400])  # constant hover

for t in times:
    omegas = input_func(t)

    a = acceleration(omegas, angles, xdot, params)
    omegadot = angular_acceleration(omegas, omega, params)

    omega += dt * omegadot
    angles += dt * euler_rates(*angles, *omega)
    xdot += dt * a
    x += dt * xdot

    # store
    Xs.append(x.copy())
    Vels.append(xdot.copy())
    Angles.append(angles.copy())
    Omegas.append(omega.copy())

Xs = np.array(Xs)
Angles = np.array(Angles)
Omegas = np.array(Omegas)


# %%
def motor_speeds_from_thrust_torques(T, tau_phi, tau_theta, tau_psi, params=params):
    """
    Given desired total thrust T and body torques (tau_phi, tau_theta, tau_psi),
    solve for the rotor speeds [ω1, ω2, ω3, ω4].

    Returns ω's in rad/s.
    """
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

    # solve A * [ω1², ω2², ω3², ω4²] = u
    omega_sq = np.linalg.solve(A, u)

    # numerical safety: no negative due to tiny floating noise
    omega_sq = np.clip(omega_sq, 0.0, None)

    return np.sqrt(omega_sq)


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
# ---------------------------------------------------------
# RUN HOVER WITH POWER MODEL
# ---------------------------------------------------------
T_hover = 120.0   # 2 minutes
dt = 0.01         # or your preferred dt

(times_h,
 Xs_h,
 Angles_h,
 Omegas_h,
 Powers_h,
 Energies_h) = simulate_hover_power(T=T_hover, dt=dt)






# %%
# ---------------------------------------------------------
# ALTITUDE PLOT
# ---------------------------------------------------------
plt.figure(figsize=(10,5))
plt.plot(times_h, Xs_h[:,2], label="z(t)")
plt.axhline(1.0, color='r', linestyle='--', label="Target 1 m")
plt.title("Hover Altitude with Full Controller + Power Model")
plt.xlabel("Time (s)")
plt.ylabel("z (m)")
plt.grid(True)
plt.legend()
plt.show()


# ---------------------------------------------------------
# EULER ANGLES
# ---------------------------------------------------------
plt.figure(figsize=(10,5))
plt.plot(times_h, Angles_h[:,0], label="phi")
plt.plot(times_h, Angles_h[:,1], label="theta")
plt.plot(times_h, Angles_h[:,2], label="psi")
plt.legend()
plt.grid(True)
plt.ylim(-0.01, 0.01)
plt.title("Euler Angles During Hover")
plt.show()


# ---------------------------------------------------------
# Euler angles (ultra-zoomed, shows 1e-16 numerical noise)
# ---------------------------------------------------------

plt.figure(figsize=(12,5))
plt.plot(times_h, Angles_h[:,0], label="phi")
plt.plot(times_h, Angles_h[:,1], label="theta")
plt.plot(times_h, Angles_h[:,2], label="psi")

plt.title("Euler Angles During Hover (Zoomed to Numerical Noise Level)")
plt.xlabel("Time (s)")
plt.ylabel("Angle (rad)")

# THIS zoom shows the noisy wiggles at ~1e-16
plt.ylim(-2e-16, 2e-16)

plt.grid(True)
plt.legend()
plt.show()


# ---------------------------------------------------------
# 3D TRAJECTORY (VISIBLE RANGE)
# ---------------------------------------------------------
from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure(figsize=(8,6))
ax = fig.add_subplot(111, projection='3d')

ax.plot(Xs_h[:,0], Xs_h[:,1], Xs_h[:,2], color='blue', linewidth=2)
ax.scatter(Xs_h[-1,0], Xs_h[-1,1], Xs_h[-1,2], color='red', s=80)

ax.set_title("3D Hover Trajectory (Full Controller)")
ax.set_xlabel("X (m)")
ax.set_ylabel("Y (m)")
ax.set_zlabel("Z (m)")

max_range = 0.05
ax.set_xlim(-max_range, max_range)
ax.set_ylim(-max_range, max_range)
ax.set_zlim(1 - max_range, 1 + max_range)

plt.show()


# ---------------------------------------------------------
# EXTREME ZOOM (Goldilocks Zoom 5e-14)
# ---------------------------------------------------------
fig = plt.figure(figsize=(7,6))
ax = fig.add_subplot(111, projection='3d')

ax.plot(Xs_h[:,0], Xs_h[:,1], Xs_h[:,2], color='blue', linewidth=1.5)
ax.scatter(Xs_h[-1,0], Xs_h[-1,1], Xs_h[-1,2], color='red', s=60)

ax.set_title("3D Hover Trajectory (Zoom = 5e-14)")
ax.set_xlabel("X (m)")
ax.set_ylabel("Y (m)")
ax.set_zlabel("Z (m)")

x0 = np.mean(Xs_h[:,0])
y0 = np.mean(Xs_h[:,1])
z0 = np.mean(Xs_h[:,2])

zoom = 5e-14  # change to 5e-15 or 5e-13 if needed

ax.set_xlim(x0 - zoom, x0 + zoom)
ax.set_ylim(y0 - zoom, y0 + zoom)
ax.set_zlim(z0 - zoom, z0 + zoom)

plt.show()

# ---------------------------------------------------------
# 3D Hover Trajectory (Zoom = 5e-16)
# ---------------------------------------------------------
fig = plt.figure(figsize=(6,6))
ax = fig.add_subplot(111, projection='3d')

ax.plot(
    Xs_h[:,0], Xs_h[:,1], Xs_h[:,2],
    color='blue', linewidth=1.5, label='Hover trajectory'
)

ax.scatter(
    Xs_h[-1,0], Xs_h[-1,1], Xs_h[-1,2],
    color='red', s=60, label='Final hover point'
)

ax.set_title("3D Hover Trajectory (Zoom = 5e−16)")
ax.set_xlabel("X (m)")
ax.set_ylabel("Y (m)")
ax.set_zlabel("Z (m)")

# Zero-centered zoom around the hover point
x0 = np.mean(Xs_h[:,0])
y0 = np.mean(Xs_h[:,1])
z0 = np.mean(Xs_h[:,2])

zoom = 5e-16
ax.set_xlim(x0 - zoom, x0 + zoom)
ax.set_ylim(y0 - zoom, y0 + zoom)
ax.set_zlim(z0 - zoom, z0 + zoom)

ax.legend()
plt.show()



# %%
# ============================================================
#   EULER ANGLES — ZOOMED TO SEE NUMERICAL NOISE (1e-16)
# ============================================================

plt.figure(figsize=(10,5))
plt.plot(times_h, Angles_h[:,0], label="phi")
plt.plot(times_h, Angles_h[:,1], label="theta")
plt.plot(times_h, Angles_h[:,2], label="psi")
plt.legend()
plt.grid(True)
plt.title("Euler Angles During Hover (Zoomed to Numerical Noise Level)")
plt.ylabel("Angle (rad)")
plt.xlabel("Time (s)")
plt.ylim(-2e-16, 2e-16)  
plt.show()



# ============================================================
#   3D TRAJECTORY — GOLDILOCKS SUPER-ZOOM LIKE YOUR OLD PLOT
# ============================================================

fig = plt.figure(figsize=(7,6))
ax = fig.add_subplot(111, projection='3d')

ax.plot(
    Xs_h[:,0],
    Xs_h[:,1],
    Xs_h[:,2],
    color='blue',
    linewidth=1.5,
    label='Hover trajectory'
)

ax.scatter(
    Xs_h[-1,0],
    Xs_h[-1,1],
    Xs_h[-1,2],
    color='red',
    s=60,
    label='Final hover point'
)

ax.set_title("3D Hover Trajectory (Zoom = 5e-16)")
ax.set_xlabel("X (m)")
ax.set_ylabel("Y (m)")
ax.set_zlabel("Z (m)")

# --- Compute mean position (very close to 0,0,1) ---
x0 = np.mean(Xs_h[:,0])
y0 = np.mean(Xs_h[:,1])
z0 = np.mean(Xs_h[:,2])

zoom = 5e-16   # exact zoom level from your old good figure

ax.set_xlim(x0 - zoom, x0 + zoom)
ax.set_ylim(y0 - zoom, y0 + zoom)
ax.set_zlim(z0 - zoom, z0 + zoom)

ax.legend()
plt.show()


# %%
# ---------------------------------------------------------
# POWER CONSUMPTION OVER TIME
# ---------------------------------------------------------
plt.figure(figsize=(10,5))
plt.plot(times_h, Powers_h, color='blue')
plt.ylabel("Power (W)")
plt.xlabel("Time (s)")
plt.title("Hover Power Consumption")
plt.grid(True)
plt.show()


# ---------------------------------------------------------
# ENERGY USED VS BATTERY CAPACITY
# ---------------------------------------------------------
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


# ---------------------------------------------------------
# BATTERY REMAINING
# ---------------------------------------------------------
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

# ---------------------------------------------------------
# Battery remaining — zoomed-in view
# ---------------------------------------------------------

plt.figure(figsize=(12,5))
plt.plot(times_h, batt_remaining, color='green', linewidth=2, label="Remaining Energy")

plt.title("Battery Remaining Over Time (Zoomed In)")
plt.xlabel("Time (s)")
plt.ylabel("Energy (J)")

# match your desired zoom: around 54k – 60k
plt.ylim(batt_remaining.min() - 500, batt_remaining.max() + 500)

plt.grid(True)
plt.legend()
plt.show()



# %%


# %%


# %%


# %%



