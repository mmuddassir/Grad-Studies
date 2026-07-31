"""
p7_accuracy_periodic_diff.py

Chapter 3/4: Numerically illustrate how the *smoothness* of a periodic
function controls the convergence rate of spectral differentiation.

Three test functions of increasing smoothness:
  1. v(x) = |sin(x)|^3        -> finitely many continuous derivatives
                                  => only algebraic (power-law) convergence
  2. v(x) = exp(-sin(x/2)^{-2}) style bump  -> C^infinity but not analytic
                                  => convergence faster than any power of N,
                                     but not geometric
  3. v(x) = exp(sin(x))       -> analytic and periodic
                                  => geometric (spectral) convergence

We plot max error vs N on a log-linear / log-log axis to visually
distinguish algebraic decay (straight line on log-log) from spectral decay
(straight line on log-linear / semilog).
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def fourier_diff_fft(v):
    N = len(v)
    v_hat = np.fft.fft(v)
    k = np.fft.fftfreq(N, d=1.0 / N)
    if N % 2 == 0:
        k[N // 2] = 0.0
    w_hat = 1j * k * v_hat
    return np.real(np.fft.ifft(w_hat))


def func_low_smoothness(x):
    v = np.abs(np.sin(x)) ** 3
    vprime = 3 * np.sin(x) * np.abs(np.sin(x)) * np.cos(x)
    return v, vprime


def func_c_infinity(x):
    # smooth periodic bump, C^infinity but not analytic
    eps = 1e-14
    s = np.sin(x / 2) ** 2 + eps
    v = np.exp(-1.0 / s)
    # numerically differentiate via a very fine reference FFT for the "exact" derivative
    return v, None  # exact derivative supplied numerically below


def func_analytic(x):
    v = np.exp(np.sin(x))
    vprime = np.cos(x) * v
    return v, vprime


Ns = [8, 16, 32, 64, 128, 256]

results = {"low_smoothness": [], "C_infinity": [], "analytic": []}

# reference solution for the C-infinity bump via a very fine grid
N_ref = 4096
h_ref = 2 * np.pi / N_ref
x_ref = h_ref * np.arange(1, N_ref + 1)
v_ref, _ = func_c_infinity(x_ref)
w_ref = fourier_diff_fft(v_ref)

for N in Ns:
    h = 2 * np.pi / N
    x = h * np.arange(1, N + 1)

    v, vprime = func_low_smoothness(x)
    w = fourier_diff_fft(v)
    results["low_smoothness"].append(np.max(np.abs(w - vprime)))

    v, _ = func_c_infinity(x)
    w = fourier_diff_fft(v)
    w_true = w_ref[:: N_ref // N][:N]  # subsample reference derivative at matching points
    results["C_infinity"].append(np.max(np.abs(w - w_true)))

    v, vprime = func_analytic(x)
    w = fourier_diff_fft(v)
    results["analytic"].append(np.max(np.abs(w - vprime)))

fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
for name, errs in results.items():
    axes[0].loglog(Ns, errs, "o-", label=name)
    axes[1].semilogy(Ns, errs, "o-", label=name)
axes[0].set_xlabel("N"); axes[0].set_ylabel("max error"); axes[0].set_title("log-log")
axes[1].set_xlabel("N"); axes[1].set_ylabel("max error"); axes[1].set_title("log-linear")
for ax in axes:
    ax.legend(fontsize=8)
    ax.grid(True, which="both", alpha=0.3)
plt.tight_layout()
plt.savefig("p7_accuracy_periodic_diff.png", dpi=150)
print("Saved p7_accuracy_periodic_diff.png")
for name, errs in results.items():
    print(name, ["%.2e" % e for e in errs])
