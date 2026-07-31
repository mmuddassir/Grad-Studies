"""
p8_harmonic_oscillator_eigenvalues.py

Chapter 3: Use a periodic Fourier spectral differentiation matrix on a
sufficiently large domain to approximate the eigenvalues of the quantum
harmonic oscillator operator

    L u = -u'' + x^2 u        on the whole real line,

whose exact eigenvalues are lambda_n = 2n + 1, n = 0, 1, 2, ...

Because the eigenfunctions (Hermite functions) decay like exp(-x^2/2), a
periodic approximation on a large-enough interval [-L, L] with periodic
boundary conditions is an excellent approximation to the true whole-line
problem, and the discretized operator -D2 + diag(x^2) converges spectrally
fast to the true low-lying eigenvalues as N increases.
"""
import numpy as np
from scipy.linalg import toeplitz


def fourier_diff_matrices(N, L):
    """First and second order periodic differentiation matrices on [-L, L)."""
    h = 2 * np.pi / N
    x = h * np.arange(1, N + 1) - np.pi
    x = x * (L / np.pi)  # rescale periodic grid [-pi,pi) to [-L, L)
    scale = np.pi / L

    col1 = np.zeros(N)
    idx = np.arange(1, N)
    col1[1:] = 0.5 * (-1.0) ** idx / np.tan(idx * h / 2)
    row1 = np.concatenate(([col1[0]], col1[1:][::-1]))
    D1 = toeplitz(col1, row1) * scale

    col2 = np.zeros(N)
    col2[0] = -np.pi ** 2 / (3 * h ** 2) - 1.0 / 6.0
    col2[1:] = -0.5 * (-1.0) ** idx / np.sin(idx * h / 2) ** 2
    D2 = toeplitz(col2) * scale ** 2

    return D1, D2, x


if __name__ == "__main__":
    L = 12.0  # domain half-width; eigenfunctions are negligible beyond this
    print("N     lowest 5 numerical eigenvalues        exact (2n+1)")
    for N in [40, 60, 80, 100]:
        _, D2, x = fourier_diff_matrices(N, L)
        Lop = -D2 + np.diag(x ** 2)
        eigvals = np.sort(np.linalg.eigvalsh(Lop))
        approx = eigvals[:5]
        exact = np.array([2 * n + 1 for n in range(5)])
        print(f"N={N:4d}  {np.round(approx, 4)}   exact {exact}")
