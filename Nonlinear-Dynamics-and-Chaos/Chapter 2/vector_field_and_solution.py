import math
import numpy as np
import matplotlib.pyplot as plt
from RK4 import RK4

dt = 0.001
t = np.arange(0,10,dt)

func = np.log

N = len(t)
results=[]

for n in np.arange(0,10,np.pi/10):

    results.append(RK4(func,n,dt,N))

# 2. Setup the Grid
t_grid = np.linspace(0, 10, 25)
x_grid = np.linspace(0, 10, 25)
T, X = np.meshgrid(t_grid, x_grid)

# 3. Compute Slopes
dt_field = np.ones_like(X)    # change in t (dt)
dx_field = func(X)           # change in x (dx)

# Normalize the vectors (so they are all the same length)
norm = np.sqrt(dt_field**2 + dx_field**2)
dt_field /= norm
dx_field /= norm

# 4. Plotting
plt.figure(figsize=(10, 6))

# angles='xy' and scale_units='xy' align arrows with the axes
# 'scale' controls the length of the arrows (higher number = shorter arrows)
plt.quiver(T, X, dt_field, dx_field, color='gray', alpha=0.4, 
           angles='xy', scale_units='xy', scale=5)

#alternate way to visualize vector field.
#plt.streamplot(T, X, dt_field, dx_field, color='gray', linewidth=1, 
              # density=0.8, arrowstyle='->', arrowsize=1.5)

# Plot RK4 Result
for j in range(0,len(results)):
    plt.plot(t, results[j], color='blue', linewidth=2)




# Aesthetics
plt.xlabel('Time (t)')
plt.ylabel('State (x)')
plt.title(' Vector Field & RK4 Solution')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()