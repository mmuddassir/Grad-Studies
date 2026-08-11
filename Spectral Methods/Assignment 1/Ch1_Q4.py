""" 
This program makes changes to program 1 in Spectral Methods to add a timer 
and change the range of N from 2^12 to 2^16.

Results can be viewed in the pdf with the matching file name. 
"""


import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import toeplitz
import time

N_vec = 2**np.arange(3,16)
errors = []
times = []

for N in N_vec:

    t0 = time.perf_counter()  #added this as a timer start time
    
    h = 2 * np.pi / N
    x = -np.pi + np.arange(1, N + 1) * h
    u = np.exp(np.sin(x))
    uprime = np.cos(x) * u
    
    #D is built differently here, I found a better way to build the matrix
    j = np.arange(1, N)
    column = np.zeros(N)
    column[1:] = 0.5 * ((-1) ** j) / np.tan(j * h / 2)
    first_row = np.concatenate(([column[0]], column[:0:-1]))
    D = toeplitz(column, first_row)
    
    
    numeric_Du = D @ u
    error = np.linalg.norm(numeric_Du - uprime, np.inf)
    
    
    elapsed = time.perf_counter() - t0 #finds elapsed time for run with N in Nvec
    
    errors.append(error)
    times.append(elapsed)

N_vals = np.array(N_vec)
errors = np.array(errors)
times = np.array(times)


plt.figure(figsize=(12, 5))

#For N vs error
plt.subplot(1, 2, 1)
plt.semilogy(N_vec, errors, 'b.-', markersize=8)
plt.grid(True, which="both", ls="--")
plt.xlabel('N')
plt.ylabel('Max Error')

#For N vs time
plt.subplot(1, 2, 2)
plt.loglog(N_vec, times, 'r.-', label='Measured Time')

plt.grid(True, which="both", ls="--")
plt.xlabel('N')
plt.ylabel('Time (seconds)')

plt.legend()

plt.tight_layout()
plt.savefig('Spectral Methods\Assignment 1\Figures\Ch1Q4.pdf', dpi=150)
print("Done. Plot saved to Ch1Q4.pdf")