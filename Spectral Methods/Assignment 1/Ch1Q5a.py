import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import toeplitz

plt.figure(figsize=(8, 5))

# Loop over N from 2 to 1000; ~500 steps
for N in range(2, 1001, 2):
    h = 2 * np.pi / N
    x = -np.pi + np.arange(1, N + 1) * h
    
    # Test function and its exact derivative
    u = np.exp(np.sin(x)**2)
    uprime = 2*np.cos(x) * np.sin(x)*u
    
    
    j = np.arange(1, N)
    column = np.zeros(N)
    column[1:] = 0.5 * ((-1) ** j) / np.tan(j * h / 2)  # cot(a) = 1/tan(a). jh/2 = j*pi/N has range (0, pi)
    
    first_row = np.concatenate(([column[0]], column[:0:-1]))
    D = toeplitz(column, first_row)
    
    
    error = np.max(np.abs(D @ u - uprime))
    
    
    plt.loglog(N, error, 'b.', markersize=5)

plt.grid(True, which="both", ls="--")

plt.xlabel('N')
plt.ylabel('error')


plt.title('Convergence of spectral differentiation')

plt.savefig('Spectral Methods\Assignment 1\Figures\Ch1Q5a.pdf', dpi=150)
print("Done. Plot saved to Ch1Q5a.pdf")