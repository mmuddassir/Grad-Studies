"""
p9_equispaced_vs_chebyshev.py

Chapter 5: Polynomial interpolation of a smooth-but-not-too-smooth function
(the classic Runge function) on equispaced points versus Chebyshev points,
demonstrating that:
  - equispaced-point interpolation diverges near the endpoints as the
    degree increases (Runge phenomenon),
  - Chebyshev-point interpolation converges uniformly.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def runge(x):
    return 1.0 / (1.0 + 25.0 * x ** 2)


def equispaced_nodes(N):
    return np.linspace(-1, 1, N + 1)


def chebyshev_nodes(N):
    j = np.arange(N + 1)
    return np.cos(j * np.pi / N)


xx = np.linspace(-1, 1, 2000)
true_vals = runge(xx)

fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

for N in [6, 10, 16, 20]:
    xe = equispaced_nodes(N)
    ve = runge(xe)
    pe = np.polyval(np.polyfit(xe, ve, N), xx)
    axes[0].plot(xx, pe, label=f"N={N}")

axes[0].plot(xx, true_vals, "k--", linewidth=1, label="exact")
axes[0].set_ylim(-1, 2)
axes[0].set_title("Equispaced points (Runge phenomenon)")
axes[0].legend(fontsize=7)

for N in [6, 10, 16, 20]:
    xc = chebyshev_nodes(N)
    vc = runge(xc)
    pc = np.polyval(np.polyfit(xc, vc, N), xx)
    axes[1].plot(xx, pc, label=f"N={N}")

axes[1].plot(xx, true_vals, "k--", linewidth=1, label="exact")
axes[1].set_ylim(-0.2, 1.2)
axes[1].set_title("Chebyshev points (uniform convergence)")
axes[1].legend(fontsize=7)

plt.tight_layout()
plt.savefig("p9_equispaced_vs_chebyshev.png", dpi=150)
print("Saved p9_equispaced_vs_chebyshev.png")

# Also report max interpolation error vs N for both node types
print("\nMax error over [-1,1] vs N:")
print(f"{'N':>4} {'equispaced':>14} {'chebyshev':>14}")
for N in [6, 10, 14, 18, 22, 26, 30]:
    xe = equispaced_nodes(N)
    pe = np.polyval(np.polyfit(xe, runge(xe), N), xx)
    err_e = np.max(np.abs(pe - true_vals))

    xc = chebyshev_nodes(N)
    pc = np.polyval(np.polyfit(xc, runge(xc), N), xx)
    err_c = np.max(np.abs(pc - true_vals))

    print(f"{N:4d} {err_e:14.4e} {err_c:14.4e}")
