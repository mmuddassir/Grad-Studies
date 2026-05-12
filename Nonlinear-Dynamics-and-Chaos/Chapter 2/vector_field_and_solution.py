import math
import numpy as np
import matplotlib.pyplot as plt
from RK4 import RK4

dt = 0.001
t = np.arange(0,10,dt)
r=2

func = lambda x: r*x - x**2

N = len(t)
results=[]

for n in [0,0.001,0.01,0.1,0.5,1,2]:

    results.append(RK4(func,n,dt,N))

# 2. Setup the Grid
t_grid = np.linspace(0, 10, 25)
x_grid = np.linspace(0, 2.1, 25)
T, X = np.meshgrid(t_grid, x_grid)

# 3. Compute Slopes
dt_field = np.ones_like(X)    # change in t (dt)
dx_field = func(X)           # change in x (dx)

# Normalize the vectors (so they are all the same length)
norm = np.sqrt(dt_field**2 + dx_field**2)
dt_field /= norm
dx_field /= norm


fig, (ax1, ax2) = plt.subplots(2,1, figsize=(8, 10))
# --- Left: Vector field + t vs x ---
magnitude = np.abs(dx_field)
ax1.quiver(T, X, dt_field, dx_field, magnitude,
           cmap='plasma', alpha=0.8,
           angles='xy', scale_units='xy', scale=5)

for j in range(len(results)):
    ax1.plot(t, results[j], color="black", linewidth=2)

ax1.set_xlabel('Time (t)')
ax1.set_ylabel('State (x)')
ax1.set_title('Vector Field & RK4 Solutions')
ax1.grid(True, alpha=0.3)

# --- Right: x vs xdot ---
x_vals = np.linspace(-2, 2, 500)
xdot_vals = func(x_vals)

ax2.plot(x_vals, xdot_vals, color='royalblue', linewidth=2.5)
ax2.axhline(0, color='gray', linewidth=0.8, linestyle='--')

ax2.set_xlabel('State (x)')
ax2.set_ylabel(r'$\dot{x}$')
ax2.set_title(r'Phase Portrait: $x$ vs $\dot{x}$')
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(r'Nonlinear-Dynamics-and-Chaos\Chapter 2\plot.png')
