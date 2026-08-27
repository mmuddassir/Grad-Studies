
import numpy as np
import time
from scipy.linalg import toeplitz


def periodic_diff_matrix(N):
   
    h = 2 * np.pi / N

    col = np.zeros(N)
    col[1:] = 0.5 * (-1.0) ** np.arange(1, N) / np.tan(np.arange(1, N) * h / 2)

    row = np.zeros(N)
    row[0] = col[0]
    row[1:] = col[N - 1:0:-1]


    return toeplitz(col, row)


def w_fft(v, N):
   
    v_hat = np.fft.fft(v)

    k = np.concatenate((np.arange(0, N // 2), [0], np.arange(-N // 2 + 1, 0)))


    w_hat = 1j * k * v_hat

    return np.real(np.fft.ifft(w_hat))


def p6_fft(N, tmax=8.0):
   
    h = 2 * np.pi / N
    x = h * np.arange(1, N + 1)
    dt = h / 4


    c = 0.2 + np.sin(x - 1) ** 2
    v = np.exp(-100 * (x - 1) ** 2)


    vold = np.exp(-100 * (x - dt - 1) ** 2)
    nsteps = int(round(tmax / dt))


    for _ in range(nsteps):
        w = w_fft(v, N)
        vnew = vold - 2 * dt * c * w
        vold, v = v, vnew

    return x, v


def p6_matrix(N, tmax=8.0):

    D = periodic_diff_matrix(N)
    h = 2 * np.pi / N
    x = h * np.arange(1, N + 1)
    dt = h / 4


    c = 0.2 + np.sin(x - 1) ** 2
    v = np.exp(-100 * (x - 1) ** 2)


    vold = np.exp(-100 * (x - dt - 1) ** 2)
    nsteps = int(round(tmax / dt))


    for _ in range(nsteps):
        w = D @ v
        vnew = vold - 2 * dt * c * w
        vold, v = v, vnew


    return x, v



print("N      FFT total(s)   matrix total(s)   matrix/FFT ratio")
nsteps = 50
for N in [128, 256, 512, 1024, 2048, 4096]:
    v = np.random.rand(N)
    D = periodic_diff_matrix(N)

    t0 = time.perf_counter()
    for _ in range(nsteps):
        w = w_fft(v, N)
    t1 = time.perf_counter()
    for _ in range(nsteps):
        w = D @ v
    t2 = time.perf_counter()

    print(f"{N:<5d} {t1-t0:10.5f} {t2-t1:15.5f} {(t2-t1)/(t1-t0):18.2f}")

print("Specifically for the function in p6\n")

print("N       FFT total(s)   matrix total(s)   matrix/FFT ratio")
for N in [128, 256, 512, 1024]:  
    t0 = time.perf_counter()
    x_fft, v_fft = p6_fft(N)
    t1 = time.perf_counter()
    
    x_mat, v_mat = p6_matrix(N)
    t2 = time.perf_counter()

    print(f"{N:<5d} {t1-t0:10.5f} {t2-t1:15.5f} {(t2-t1)/(t1-t0):18.2f}")