
import numpy as np
import time
from sympy import isprime, factorint
import matplotlib.pyplot as plt


def time_fft(N, reps=200):
    """Average clock time for one N-point FFT."""
    v = np.random.rand(N) + 1j * np.random.rand(N)
    np.fft.fft(v)                       
    t0 = time.perf_counter()
    for _ in range(reps):
        np.fft.fft(v)
    t1 = time.perf_counter()
    return (t1 - t0) / reps


def largest_prime_factor(N):
    return max(factorint(N).keys())


#First part of the question: Find average time for a large range of N
ks = np.arange(0, 16)
Ns_pow2 = 2 ** ks
times_pow2 = np.array([time_fft(int(N), reps=200) for N in Ns_pow2])



# Second part: find correlation between prime N or N with large prime factors and average FFT time
Ns_range = np.arange(500, 521)

times_range = np.array([time_fft(int(N), reps=300) for N in Ns_range])

primes_mask = np.array([isprime(int(N)) for N in Ns_range])

lpf = np.array([largest_prime_factor(int(N)) for N in Ns_range])

print("\nN     prime?   largest prime factor   time (s)")
for N, t, p, f in zip(Ns_range, times_range, primes_mask, lpf):
    print(f"{N:<6d}{str(p):<15s}{f:<17d}{t:<.3e}")


print("\nCorrelation with being prime:      ",
      np.corrcoef(primes_mask.astype(float), times_range)[0, 1])


print("Correlation with largest prime factor:",
      np.corrcoef(lpf.astype(float), times_range)[0, 1])





fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))

axes[0].loglog(Ns_pow2, times_pow2, 'o-')

axes[0].set_xlabel('N')
axes[0].set_ylabel('time per FFT (s)')
axes[0].grid(True, which='both', alpha=0.3)

#sc = axes[1].scatter(Ns_range, times_range * 1e6,
                      #s=80, zorder=3, marker=',')
axes[1].plot(Ns_range, times_range * 1e6, marker='o', color='blue', alpha=0.5, zorder=1)
for N, t, p in zip(Ns_range, times_range, primes_mask):
    if p:
        axes[1].scatter(N, t * 1e6, facecolors='none',
                         s=180, linewidths=2, zorder=4)
axes[1].set_xlabel('N')
axes[1].set_ylabel(r'time per FFT ($\mu$s)')
axes[1].set_title(r'FFT time vs $N=500,...520$')
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('Spectral Methods\Assignment 1\Figures\Ch3Q5.pdf', dpi=150)
print("Done. Plot saved to Ch3Q5.pdf")