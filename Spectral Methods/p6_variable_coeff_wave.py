import matplotlib.pyplot as plt
import numpy as np



def w_fft(v, N):
   
    v_hat = np.fft.fft(v)

    k = np.concatenate((np.arange(0, N // 2), [0], np.arange(-N // 2 + 1, 0)))


    w_hat = 1j * k * v_hat

    return np.real(np.fft.ifft(w_hat))



def p6_fft_3d(N, tmax=8.0):
    h = 2 * np.pi / N
    x = h * np.arange(1, N + 1)
    dt = h / 4

    c = 0.2 + np.sin(x - 1) ** 2
    v = np.exp(-100 * (x - 1) ** 2)
    vold = np.exp(-100 * (x - dt - 1) ** 2)
    
    nsteps = int(round(tmax / dt))
    t_vals = np.linspace(0, tmax, nsteps + 1)
    
    # Pre-allocate 2D array to store v at every time step
    V = np.zeros((nsteps + 1, N))
    V[0, :] = v  # Initial condition at t = 0

    for step in range(1, nsteps + 1):
        w = w_fft(v, N)
        vnew = vold - 2 * dt * c * w
        vold, v = v, vnew
        V[step, :] = v  # Store snapshot at current time step

    return x, t_vals, V



# 1. Run the updated simulation
x, t, V = p6_fft_3d(N=128, tmax=8.0)

# 2. Create a 2D coordinate grid for (x, t)
X, T = np.meshgrid(x, t)

# 3. Create 3D Surface Plot
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')

# Plot the surface
surf = ax.plot_surface(X, T, V,  cmap='viridis', edgecolor='none',alpha=0.4)

# Labels and view angle
ax.set_xlabel('Position (x)')
ax.set_ylabel('Time (t)')
ax.set_zlabel('Wave Amplitude (v)')
ax.set_zlim(0,4)
ax.set_title('Spatiotemporal Wave Evolution v(x, t)')

# Add colorbar and adjust viewpoint
ax.view_init(elev=30, azim=-60)  # Rotate angle for better view

plt.show()