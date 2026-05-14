"""
p1.py - Convergence of fourth-order finite differences

Translated from MATLAB Program 1 (Spectral Methods in MATLAB).
For various N, sets up a periodic grid on [-pi, pi], constructs a
4th-order finite difference differentiation matrix, and plots the
max-norm error vs. N on a log-log scale.
"""

import numpy as np
import scipy.sparse as sp
import matplotlib.pyplot as plt

# Grid sizes: N = 2^3, 2^4, ..., 2^12
Nvec = 2 ** np.arange(3, 13)

fig, ax = plt.subplots(figsize=(7, 5))

for N in Nvec:
    h = 2 * np.pi / N
    x = -np.pi + np.arange(1, N + 1) * h   # N evenly-spaced points on (-pi, pi]

    u      = np.exp(np.sin(x))              # test function
    uprime = np.cos(x) * u                  # exact derivative

    # ------------------------------------------------------------------
    # Build the 4th-order skew-symmetric differentiation matrix.
    #
    # The standard 2nd-order centered difference uses neighbors at ±1.
    # The 4th-order stencil adds neighbors at ±2 to cancel the O(h^2)
    # error term:
    #
    #   w_j ≈ ( 2/3*(u_{j+1}-u_{j-1}) - 1/12*(u_{j+2}-u_{j-2}) ) / h
    #
    # In matrix form this means:
    #   +2/3 on the  +1 superdiagonal,  -2/3 on the  -1 subdiagonal
    #   -1/12 on the +2 superdiagonal, +1/12 on the  -2 subdiagonal
    # with periodic (wrap-around) entries in the corners.
    # ------------------------------------------------------------------

    e = np.ones(N)

    # Diagonal indices for periodic wrap-around (using mod N)
    idx = np.arange(N)
    row_p1 = idx
    col_p1 = (idx + 1) % N   # +1 neighbor (wraps: N-1 -> 0)
    row_p2 = idx
    col_p2 = (idx + 2) % N   # +2 neighbor (wraps at edges)

    # Build upper part (+2/3 at +1, -1/12 at +2)
    D = (sp.csr_matrix((2 * e / 3,  (row_p1, col_p1)), shape=(N, N))
       - sp.csr_matrix((e / 12,     (row_p2, col_p2)), shape=(N, N)))

    # Make skew-symmetric: D = (D - D^T) / h
    D = (D - D.T) / h

    # ------------------------------------------------------------------
    # Compute max-norm (infinity norm) of the error
    # ------------------------------------------------------------------
    error = np.linalg.norm(D @ u - uprime, np.inf)
    ax.loglog(N, error, '.', markersize=15, color='steelblue')

# Reference line ~ N^{-4} to show expected convergence rate
ax.loglog(Nvec, Nvec.astype(float) ** (-4), '--', color='gray', label=r'$N^{-4}$')
ax.text(105, 5e-8, r'$N^{-4}$', fontsize=14, color='gray')

ax.set_xlabel('N')
ax.set_ylabel('Error')
ax.set_title('Convergence of 4th-order finite differences')
ax.grid(True, which='both', linestyle=':', alpha=0.6)
ax.legend()

plt.tight_layout()
plt.savefig('Spectral Methods\Figures\p1_convergence.pdf', dpi=150)
plt.show()
print("Done. Plot saved to p1_convergence.png")
