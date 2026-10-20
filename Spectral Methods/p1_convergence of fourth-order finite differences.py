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


Nvec = 2 ** np.arange(7, 25)

fig, ax = plt.subplots(figsize=(7, 5))

for N in Nvec:
    h = 2 * np.pi / N
    x = -np.pi + np.arange(1, N + 1) * h   

    u = np.exp(np.sin(x))   # test function
    uprime = np.cos(x) * u       # exact derivative

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
#ax.loglog(Nvec, Nvec.astype(float) ** (-4), '--', color='gray', label=r'$N^{-4}$')
#ax.text(105, 5e-8, r'$N^{-4}$', fontsize=14, color='gray')

ax.set_xlabel('N')
ax.set_ylabel('Error')
ax.set_title('Convergence of finite differences')
ax.grid(True, which='both', linestyle=':', alpha=0.6)
ax.legend()

plt.tight_layout()


#plt.show()
print("Done. Plot saved to convergence.pdf")



S = lambda u, k: np.roll(u, -k)

schemes = {
    "forward O(h)":   lambda u, h: (S(u,1) - u) / h,
    "forward O(h²)":  lambda u, h: (-3*u + 4*S(u,1) - S(u,2)) / (2*h),
    "forward O(h⁴)":  lambda u, h: (-25*u + 48*S(u,1) - 36*S(u,2)
                                    + 16*S(u,3) - 3*S(u,4)) / (12*h),
    "central O(h²)":  lambda u, h: (S(u,1) - S(u,-1)) / (2*h),
    "central O(h⁴)":  lambda u, h: (8*(S(u,1) - S(u,-1))
                                    - (S(u,2) - S(u,-2))) / (12*h),
}
for name, D in schemes.items():
    errs = [np.max(np.abs(D(np.exp(np.sin(x)), h) - np.cos(x)*np.exp(np.sin(x))))
            for N in Nvec
            for h, x in [(2*np.pi/N, -np.pi + 2*np.pi*np.arange(N)/N)]]
    ax.loglog(Nvec, errs, 'o-', label=name)
plt.legend()
plt.savefig('Spectral Methods\Figures\convergence.pdf', dpi=150)
plt.show()

