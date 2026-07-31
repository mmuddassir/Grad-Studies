"""
p5_fft_differentiation.py

Chapter 3: Repeat the periodic spectral differentiation test from p4, but
using the FFT/DFT instead of an explicit dense differentiation matrix.

Algorithm:
  1. Take the DFT of the sampled data:      v_hat = fft(v)
  2. Multiply by i*k in Fourier space:      w_hat = i*k * v_hat
     (with the Nyquist mode zeroed out when N is even, since it has no
     well-defined odd derivative / to keep the result real and symmetric)
  3. Inverse transform:                      w = ifft(w_hat)

This computes the same object as D_N @ v in p4, but in O(N log N) instead
of O(N^2), which is the entire reason spectral methods are practical.
"""
import numpy as np


def fourier_diff_fft(v):
    """First derivative of a periodic sampled vector v via the DFT."""
    N = len(v)
    v_hat = np.fft.fft(v)
    k = np.fft.fftfreq(N, d=1.0 / N)  # integer wavenumbers 0,1,...,N/2,-N/2+1,...,-1
    if N % 2 == 0:
        k[N // 2] = 0.0  # zero the Nyquist mode
    w_hat = 1j * k * v_hat
    w = np.fft.ifft(w_hat)
    return np.real(w)


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
        h = 2 * np.pi / N
        x = h * np.arange(1, N + 1)
        v, vprime = test_function_1(x)
        w = fourier_diff_fft(v)
        err = np.max(np.abs(w - vprime))
        print(f"  N = {N:4d}    max error = {err:.4e}")

    print("\nFunction 2: v(x) = exp(sin(x))  (analytic, periodic)")
    for N in [4, 8, 16, 32, 64, 128]:
        h = 2 * np.pi / N
        x = h * np.arange(1, N + 1)
        v, vprime = test_function_2(x)
        w = fourier_diff_fft(v)
        err = np.max(np.abs(w - vprime))
        print(f"  N = {N:4d}    max error = {err:.4e}")
