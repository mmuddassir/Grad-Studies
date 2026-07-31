"""
p4_periodic_diff_matrix.py

Chapter 2/3: Periodic spectral differentiation using the explicit
differentiation matrix D_N built from the cotangent (Dirichlet-kernel)
formula. For an even number of points N on the periodic grid x_j = j*h,
h = 2*pi/N, the matrix is the skew-symmetric Toeplitz matrix with

    D_{ij} = 0                                  if i == j
    D_{ij} = (1/2) * (-1)^(i-j) * cot((i-j)h/2)  if i != j

Applying D to a sampled vector v approximates v'(x_j) with spectral
(faster than any power of h) accuracy, provided v is smooth and periodic.

We test this on two functions:
  1. v(x) = |sin(x)|^3            (three continuous derivatives -> algebraic
                                    convergence, not spectral)
  2. v(x) = exp(sin(x))           (analytic and periodic -> spectral,
                                    exponential convergence)
"""
import numpy as np
from scipy.linalg import toeplitz


def fourier_diff_matrix(N):
    """Build the N x N periodic spectral differentiation matrix."""
    h = 2 * np.pi / N
    x = h * np.arange(1, N + 1)
    col = np.zeros(N)
    idx = np.arange(1, N)
    col[1:] = 0.5 * (-1.0) ** idx / np.tan(idx * h / 2)
    row = np.concatenate(([col[0]], col[1:][::-1]))
    D = toeplitz(col, row)
    return D, x


def test_function_1(x):
    v = np.abs(np.sin(x)) ** 3
    vprime = 3 * np.sin(x) * np.abs(np.sin(x)) * np.cos(x)
    return v, vprime


def test_function_2(x):
    v = np.exp(np.sin(x))
    vprime = np.cos(x) * v
    return v, vprime


if __name__ == "__main__":
    print("Function 1: v(x) = |sin(x)|^3  (limited smoothness)")
    for N in [4, 8, 16, 32, 64, 128]:
        D, x = fourier_diff_matrix(N)
        v, vprime = test_function_1(x)
        err = np.max(np.abs(D @ v - vprime))
        print(f"  N = {N:4d}    max error = {err:.4e}")

    print("\nFunction 2: v(x) = exp(sin(x))  (analytic, periodic)")
    for N in [4, 8, 16, 32, 64, 128]:
        D, x = fourier_diff_matrix(N)
        v, vprime = test_function_2(x)
        err = np.max(np.abs(D @ v - vprime))
        print(f"  N = {N:4d}    max error = {err:.4e}")
