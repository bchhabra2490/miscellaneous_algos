import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

# Constants
G = 6.6743e-11
M_e = 5.972e24  # Earth mass
R_earth = 6371e3  # radius in meters
R_safe = 50  # safe distance in meters

# Initial conditions: satellite
x0_s, y0_s = R_earth + 500e3, 0  # 500 km orbit
vx0_s, vy0_s = 0, 7670  # approx circular speed
# Debris
x0_d, y0_d = R_earth + 500e3, 1000
vx0_d, vy0_d = 0, 7670


def dynamics(t, y):
    x_s, y_s, vx_s, vy_s, x_d, y_d, vx_d, vy_d = y

    # Satellite acceleration
    r_s = np.sqrt(x_s**2 + y_s**2)
    ax_s = -G * M_e * x_s / r_s**3
    ay_s = -G * M_e * y_s / r_s**3

    # Debris acceleration
    r_d = np.sqrt(x_d**2 + y_d**2)
    ax_d = -G * M_e * x_d / r_d**3
    ay_d = -G * M_e * y_d / r_d**3

    return [vx_s, vy_s, ax_s, ay_s, vx_d, vy_d, ax_d, ay_d]


# Initial state vector
y0 = [x0_s, y0_s, vx0_s, vy0_s, x0_d, y0_d, vx0_d, vy0_d]

t_span = (0, 6000)  # seconds (~1.5 hr)
t_eval = np.linspace(*t_span, 2000)

sol = solve_ivp(dynamics, t_span, y0, t_eval=t_eval)

# Check collision
dist = np.sqrt((sol.y[0] - sol.y[4]) ** 2 + (sol.y[1] - sol.y[5]) ** 2)
collision_times = sol.t[dist < R_safe]

if len(collision_times) > 0:
    print(f"Potential collision at t = {collision_times[0]:.1f} s")
else:
    print("No collision predicted.")

# Plot
plt.plot(sol.y[0], sol.y[1], label="Satellite")
plt.plot(sol.y[4], sol.y[5], label="Debris")
plt.scatter(0, 0, color="blue", label="Earth", s=200)
plt.xlabel("x (m)")
plt.ylabel("y (m)")
plt.title("Satellite & Debris Trajectories")
plt.axis("equal")
plt.grid(True)
plt.legend()
plt.show()
