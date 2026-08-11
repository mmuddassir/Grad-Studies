
import numpy as np
import matplotlib
import matplotlib.pyplot as plt


def w_fft(v, N):
    
    v_hat = np.fft.fft(v)
    k = np.concatenate((np.arange(0, N // 2), [0], np.arange(-N // 2 + 1, 0)))
    w_hat = 1j * k * v_hat
    return np.real(np.fft.ifft(w_hat))


def p6_fft(N, tmax=8.0):
   
    h = 2 * np.pi / N
    x = h * np.arange(1, N + 1)
    dt = h / 4

    c = 0.2 + np.sin(x - 1) ** 2
    v = np.exp(-100 * (x - 1) ** 2)

    vold = np.exp(-100 * (x - dt - 1) ** 2)

    steps = int(round(tmax / dt))

    for _ in range(steps):
        w = w_fft(v, N)
        vnew = vold - 2 * dt * c * w
        vold, v = v, vnew
    return x, v


def p6_with_leapfrog(N, tmax=8.0):
    h = 2 * np.pi / N
    x = h * np.arange(1, N + 1)
    dt = h / 4

    c = 0.2 + np.sin(x - 1) ** 2
    v = np.exp(-100 * (x - 1) ** 2)

    vold = np.exp(-100 * (x - dt - 1) ** 2)
    steps = int(round(tmax / dt))


    for _ in range(steps):
        w = (np.roll(v, -1) - np.roll(v, 1)) / (2 * h) 
        vnew = vold - 2 * dt * c * w
        vold, v = v, vnew
    return x, v


Ns = [128, 256]
results = {}
print("N      max|u_spectral - u_FD|      max|u_spectral|")
for N in Ns:
    x_spec, v_spec = p6_fft(N)
    x_leap, v_leap = p6_with_leapfrog(N)
    results[N] = (x_spec, v_spec, x_leap, v_leap)
    err = np.max(np.abs(v_spec - v_leap))
    print(f"{N:4d}   {err:16.5f}          {np.max(np.abs(v_spec)):.5f}")


fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)

for ax, N in zip(axes, Ns):

    x_spec, v_spec, x_leap, v_leap = results[N]

    ax.plot(x_spec, v_spec, 'k-', lw=1.8, label='spectral (Program 6)')

    ax.plot(x_leap, v_leap, color='crimson', lw=1.2, label='2nd-order leapfrog FD')

    ax.set_title(f'N={N}')

    ax.set_xlabel('x')

    ax.legend(fontsize=8)

axes[0].set_ylabel(r'u(x, $t_{final}$)')
plt.tight_layout()


plt.savefig('Spectral Methods\Assignment 1\Figures\Ch3Q8.pdf', dpi=150)

print("Done. Plot saved to Ch3Q8.pdf")