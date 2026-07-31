"""
p12_accuracy_chebyshev_diff.py

Chapter 5: Measure convergence of Chebyshev spectral differentiation as a
function of N, for functions of varying smoothness on [-1, 1], mirroring
the periodic-case experiment of p7 but for non-periodic Chebyshev grids.

Test functions:
  1. v(x) = |x|^3              -> limited smoothness -> algebraic convergence
  2. v(x) = exp(-x^{-2}) style -> C^infinity but not analytic (here we use
                                   a runge-like rational function, which is
                                   analytic on [-1,1] but has poles nearby
                                   in the complex plane -> geometric
                                   convergence with a rate set by the
                                   distance to the nearest pole)
  3. v(x) = exp(x)*sin(5x)     -> entire function -> very fast (faster than
                                   any fixed geometric rate) convergence
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def cheb(N):
    if N == 0:
        return np.array([[0.0]]), np.array([1.0])
    x = np.cos(np.pi * np.arange(N + 1) / N)
    c = np.ones(N + 1)
    c[0] = 2.0
    c[N] = 2.0
    c = c * (-1.0) ** np.arange(N + 1)
    X = np.tile(x, (N + 1, 1)).T
    dX = X - X.T
    D = np.outer(c, 1.0 / c) / (dX + np.eye(N + 1))
    D = D - np.diag(D.sum(axis=1))
    return D, x


def func_low_smoothness(x):
    v = np.abs(x) ** 3
    vprime = 3 * x * np.abs(x)
    return v, vprime


def func_runge(x):
    v = 1.0 / (1.0 + 16.0 * x ** 2)
    vprime = -32.0 * x / (1.0 + 16.0 * x ** 2) ** 2
    return v, vprime


def func_entire(x):
    v = np.exp(x) * np.sin(5 * x)
    vprime = np.exp(x) * (np.sin(5 * x) + 5 * np.cos(5 * x))
    return v, vprime


Ns = [4, 8, 16, 32, 64, 128]
results = {"low_smoothness": [], "runge_rational": [], "entire": []}

for N in Ns:
    D, x = cheb(N)

    v, vprime = func_low_smoothness(x)
    results["low_smoothness"].append(np.max(np.abs(D @ v - vprime)))

    v, vprime = func_runge(x)
    results["runge_rational"].append(np.max(np.abs(D @ v - vprime)))

    v, vprime = func_entire(x)
    results["entire"].append(np.max(np.abs(D @ v - vprime)))

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
plt.savefig("p12_accuracy_chebyshev_diff.png", dpi=150)
print("Saved p12_accuracy_chebyshev_diff.png")
for name, errs in results.items():
    print(name, ["%.2e" % e for e in errs])
