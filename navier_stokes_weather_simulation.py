import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Grid size
lat_size = 10
lon_size = 10
layers = 3

# Time steps
frames = 200
dt = 0.1  # time step size

# Initialize fields
np.random.seed(42)
T = np.zeros((layers, lat_size, lon_size))  # Temperature
P = np.zeros((layers, lat_size, lon_size))  # Pressure
u = np.zeros((layers, lat_size, lon_size))  # East-West wind
v = np.zeros((layers, lat_size, lon_size))  # North-South wind

# Initial temperature gradient (warm equator, cold poles)
for j in range(lat_size):
    base_temp = 30 - abs(j - lat_size / 2) * 3  # warmer in the middle
    for k in range(layers):
        T[k, j, :] = base_temp - k * 6  # colder in upper layers
T += np.random.randn(*T.shape) * 0.5  # small noise

# Initial random wind
u += np.random.randn(*u.shape) * 0.1
v += np.random.randn(*v.shape) * 0.1

# Initial random pressure
P[:] = 1013 + np.random.randn(*P.shape) * 0.5  # hPa

# Parameters
diffusion_rate = 0.05
coriolis_strength = 0.05
vertical_mix = 0.01


def apply_diffusion(field):
    """Simple 2D diffusion on each layer."""
    new_field = field.copy()
    for k in range(layers):
        for j in range(lat_size):
            for i in range(lon_size):
                neighbors = []
                for dj, di in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    nj = (j + dj) % lat_size
                    ni = (i + di) % lon_size
                    neighbors.append(field[k, nj, ni])
                new_field[k, j, i] += diffusion_rate * (np.mean(neighbors) - field[k, j, i])
    return new_field


def apply_advection(scalar, u, v):
    """Move scalar field according to winds (upwind scheme)."""
    new_scalar = scalar.copy()
    for k in range(layers):
        for j in range(lat_size):
            for i in range(lon_size):
                prev_j = max(0, min(lat_size - 1, int(j - v[k, j, i] * dt)))
                prev_i = (i - int(u[k, j, i] * dt)) % lon_size
                new_scalar[k, j, i] = scalar[k, prev_j, prev_i]
    return new_scalar


def step():
    global T, P, u, v

    # Advection
    T = apply_advection(T, u, v)
    P = apply_advection(P, u, v)

    # Diffusion
    T = apply_diffusion(T)
    u = apply_diffusion(u)
    v = apply_diffusion(v)

    # Pressure gradient force (simplified)
    for k in range(layers):
        grad_y, grad_x = np.gradient(P[k])
        u[k] -= grad_x * 0.01
        v[k] -= grad_y * 0.01

    # Coriolis effect
    for k in range(layers):
        u_k = u[k].copy()
        v_k = v[k].copy()
        u[k] += coriolis_strength * v_k
        v[k] -= coriolis_strength * u_k

    # Vertical mixing
    for k in range(1, layers):
        mix_T = vertical_mix * (T[k - 1] - T[k])
        T[k] += mix_T
        T[k - 1] -= mix_T

        mix_u = vertical_mix * (u[k - 1] - u[k])
        u[k] += mix_u
        u[k - 1] -= mix_u

        mix_v = vertical_mix * (v[k - 1] - v[k])
        v[k] += mix_v
        v[k - 1] -= mix_v

    # Heating/cooling toward equilibrium
    for j in range(lat_size):
        equator_heat = 0.01 * (5 - abs(j - lat_size / 2))
        T[0, j, :] += equator_heat
        T[0, j, :] -= 0.005 * (T[0, j, :] - 15)


# ==== Visualization ====
fig, ax = plt.subplots(figsize=(6, 6))
lat = np.arange(lat_size)
lon = np.arange(lon_size)
lon_grid, lat_grid = np.meshgrid(lon, lat)

temp_plot = ax.imshow(T[0], cmap="coolwarm", vmin=-10, vmax=35)
quiver = ax.quiver(lon_grid, lat_grid, u[0], v[0], scale=1, scale_units="xy")
contour = ax.contour(lon_grid, lat_grid, P[0], colors="black", linewidths=0.5)
ax.set_title("Toy Weather Simulation with Pressure Contours")
ax.set_xticks([])
ax.set_yticks([])


def update(frame):
    global contour
    step()
    temp_plot.set_data(T[0])
    quiver.set_UVC(u[0], v[0])

    # Remove old contours
    for c in ax.collections:
        if isinstance(c, plt.matplotlib.collections.LineCollection):
            c.remove()

    # Draw new contours
    contour = ax.contour(lon_grid, lat_grid, P[0], colors="black", linewidths=0.5)

    ax.set_title(f"Toy Weather Simulation - Step {frame}")
    return [temp_plot, quiver]


ani = animation.FuncAnimation(fig, update, frames=frames, interval=100, blit=True)
plt.show()
