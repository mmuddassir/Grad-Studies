import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import toeplitz

plt.figure(figsize=(8, 5))

# Loop over even N from 2 to 100
for N in range(2, 101, 2):
    h = 2 * np.pi / N
    x = -np.pi + np.arange(1, N + 1) * h
    
    # Test function and its exact derivative
    u = np.exp(np.sin(x))
    uprime = np.cos(x) * u
    
    # Construct periodic spectral differentiation matrix column
    j = np.arange(1, N)
    column = np.zeros(N)
    column[1:] = 0.5 * ((-1) ** j) / np.tan(j * h / 2)  # cot(a) = 1/tan(a)
    
    # Construct Toeplitz differentiation matrix D
    # Row matching MATLAB's column([1, N:-1:2])
    first_row = np.concatenate(([column[0]], column[:0:-1]))
    D = toeplitz(column, first_row)
    
    # Calculate infinity norm error max(abs(D*u - uprime))
    error = np.linalg.norm(D @ u - uprime, np.inf)
    
    # Plot maximum error vs N on log-log scale
    plt.loglog(N, error, 'b.', markersize=8)

plt.grid(True, which="both", ls="--")
plt.xlabel('N')
plt.ylabel('error')
plt.title('Convergence of spectral differentiation')
plt.savefig('Spectral Methods\Figures\p2_convergence.pdf', dpi=150)
print("Done. Plot saved to p2_convergence.pdf")