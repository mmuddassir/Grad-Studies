"""
p6_variable_coeff_wave.py

Chapter 3: A variable-coefficient wave equation solved by combining FFT-based
spectral differentiation in space with an explicit leapfrog scheme in time.

PDE:   u_t + c(x) u_x = 0,      c(x) = 0.2 + sin(x - 1)^2

The spatial derivative u_x is computed spectrally at every time step via the
FFT differentiation of p5; time-stepping uses the leapfrog (midpoint) method:

    u^{n+1} = u^{n-1} - 2*dt*c(x)*u_x^n

with a single Euler half-step to start the two-level scheme.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def spectral_derivative(v):
    N = len(v)
    v_hat = np.fft.fft(v)
    k = np.fft.fftfreq(N, d=1.0 / N)
    if N % 2 == 0:
        k[N // 2] = 0.0
    w_hat = 1j * k * v_hat
    return np.real(np.fft.ifft(w_hat))


# Grid
N = 128
h = 2 * np.pi / N
x = h * np.arange(1, N + 1)
c = 0.2 + np.sin(x - 1.0) ** 2

# Time-stepping parameters
dt = 8.0 / N          # CFL-safe step (mirrors Trefethen's h*8/N choice)
t = 0.0
tmax = 8.0
nsteps = int(round(tmax / dt))

# Initial condition: a Gaussian bump
u = np.exp(-3 * (x - np.pi / 2) ** 2)
uold = np.exp(-3 * (x - dt - np.pi / 2) ** 2)  # approx. solution one step in the past

snapshots = [u.copy()]
times = [0.0]
snap_every = nsteps // 4

for n in range(1, nsteps + 1):
    w = spectral_derivative(u)
    unew = uold - 2 * dt * c * w
    uold, u = u, unew
    t += dt
    if n % snap_every == 0:
        snapshots.append(u.copy())
        times.append(t)

plt.figure(figsize=(7, 5))
for tt, snap in zip(times, snapshots):
    plt.plot(x, snap, label=f"t = {tt:.2f}")
plt.legend(fontsize=8)
plt.xlabel("x")
plt.ylabel("u(x,t)")
plt.title("Variable-coefficient wave equation, FFT spectral differentiation")
plt.tight_layout()
plt.savefig("p6_variable_coeff_wave.png", dpi=150)
print("Saved p6_variable_coeff_wave.png")
