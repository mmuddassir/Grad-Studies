"""
p10_nodal_polynomial_contours.py

Chapter 5: For a set of interpolation nodes {x_0, ..., x_N} on [-1, 1], the
"nodal polynomial" is

    phi(z) = prod_k (z - x_k)

Its behavior in the complex plane controls interpolation error (Runge-type
analysis). We compare the nodal polynomials for equispaced and Chebyshev
nodes by plotting contours of log|phi(z)| in the complex plane: for
Chebyshev nodes the level curves cluster tightly and evenly around [-1,1]
(small, well-controlled growth off the real axis), while for equispaced
nodes they bulge dramatically near the endpoints -- the geometric root
cause of the Runge phenomenon.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def equispaced_nodes(N):
    return np.linspace(-1, 1, N + 1)


def chebyshev_nodes(N):
    j = np.arange(N + 1)
    return np.cos(j * np.pi / N)


N = 20
xr = np.linspace(-1.4, 1.4, 300)
yr = np.linspace(-1.0, 1.0, 300)
X, Y = np.meshgrid(xr, yr)
Z = X + 1j * Y

fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

for ax, nodes_fn, title in [
    (axes[0], equispaced_nodes, "Equispaced nodes"),
    (axes[1], chebyshev_nodes, "Chebyshev nodes"),
]:
    nodes = nodes_fn(N)
    phi = np.ones_like(Z)
    for xk in nodes:
        phi = phi * (Z - xk)
    logphi = np.log(np.abs(phi) + 1e-300) / N  # normalize by degree

    cs = ax.contour(X, Y, logphi, levels=20)
    ax.plot(nodes, np.zeros_like(nodes), "r.", markersize=6)
    ax.set_title(f"{title}, N={N}")
    ax.set_xlabel("Re(z)")
    ax.set_ylabel("Im(z)")
    ax.set_aspect("equal")

plt.tight_layout()
plt.savefig("p10_nodal_polynomial_contours.png", dpi=150)
print("Saved p10_nodal_polynomial_contours.png")
