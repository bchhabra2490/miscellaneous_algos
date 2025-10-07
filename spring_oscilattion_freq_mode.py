import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import lagrange

# True motion
t = np.linspace(0, 4, 100)
x_true = np.cos(t)

# Sample points
t_sample = np.array([0, 1, 2, 3, 4])
x_sample = np.cos(t_sample)

# Polynomial interpolation
poly = lagrange(t_sample, x_sample)
x_poly = poly(t)

# Frequency-based approximation (single-mode cosine)
x_freq = np.cos(t)  # same as true, since single-mode

# Plot
plt.figure(figsize=(10, 5))
plt.plot(t, x_true, "k-", label="True motion")
plt.plot(t, x_poly, "r--", label="Polynomial approx")
plt.plot(t, x_freq, "b:", label="Frequency approx")
plt.scatter(t_sample, x_sample, c="k", zorder=5)
plt.title("Mass-Spring System: Polynomial vs Frequency Approx")
plt.xlabel("Time t")
plt.ylabel("Displacement x(t)")
plt.legend()
plt.show()

# Compare velocities
v_true = -np.sin(t)
v_poly = np.polyder(poly)(t)
v_freq = -np.sin(t)

plt.figure(figsize=(10, 5))
plt.plot(t, v_true, "k-", label="True velocity")
plt.plot(t, v_poly, "r--", label="Polynomial velocity")
plt.plot(t, v_freq, "b:", label="Frequency velocity")
plt.title("Velocity Comparison")
plt.xlabel("Time t")
plt.ylabel("Velocity dx/dt")
plt.legend()
plt.show()
