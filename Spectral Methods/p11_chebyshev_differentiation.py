"""
p11_chebyshev_differentiation.py

Chapter 5: Build the Chebyshev spectral differentiation matrix on the
Chebyshev-Gauss-Lobatto points

    x_j = cos(j*pi/N),   j = 0, ..., N

and use it to differentiate a smooth, non-periodic test function on
[-1, 1]:  v(x) = exp(x) * sin(5x).

The Chebyshev differentiation matrix D_N (Trefethen's "cheb" function) has
entries:

    D_00 = (2N^2+1)/6,  D_NN = -(2N^2+1)/6
    D_jj = -x_j / (2(1-x_j^2)),           for 1 <= j <= N-1
    D_ij = (c_i/c_j) * (-1)^(i+j) / (x_i - x_j),   i != j

    where c_0 = c_N = 2, c_j = 1 otherwise.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def cheb(N):
    """Chebyshev differentiation matrix and Chebyshev-Lobatto points."""
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


if __name__ == "__main__":
    N = 16
    D, x = cheb(N)

    v = np.exp(x) * np.sin(5 * x)
    vprime_exact = np.exp(x) * (np.sin(5 * x) + 5 * np.cos(5 * x))
    vprime_numeric = D @ v

    print("x_j          v'(x_j) exact     v'(x_j) spectral      error")
    for xi, ve, vn in zip(x, vprime_exact, vprime_numeric):
        print(f"{xi:8.4f}   {ve:14.6f}   {vn:14.6f}   {abs(ve - vn):.2e}")

    xx = np.linspace(-1, 1, 400)
    plt.figure(figsize=(6, 4.5))
    plt.plot(xx, np.exp(xx) * (np.sin(5 * xx) + 5 * np.cos(5 * xx)), "-", label="exact v'")
    plt.plot(x, vprime_numeric, "o", label="Chebyshev spectral derivative")
    plt.legend()
    plt.title(f"Chebyshev differentiation, N = {N}")
    plt.tight_layout()
    plt.savefig("p11_chebyshev_differentiation.png", dpi=150)
    print("\nSaved p11_chebyshev_differentiation.png")
