import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import toeplitz

plt.figure(figsize=(8, 5))


for N in range(2, 1001, 2):
    h = 2 * np.pi / N
    x = -np.pi + np.arange(1, N + 1) * h
    
    # Test function and its exact derivative
    u = np.exp(np.sin(x)*np.abs(np.sin(x)))
    uprime = 2*np.abs(np.sin(x)) * np.cos(x)*u 

    """ Wolfram initially gave a function with sin(x) in the denominator for uprime. 
        This led to most of the datapoints being NaN because of overflow error
        due to division by 0 whenever x was a multiple of pi.
        
        Used Claude to derive uprime and had it show the working to understand how it got there."""
    
    
    j = np.arange(1, N)
    column = np.zeros(N)
    column[1:] = 0.5 * ((-1) ** j) / np.tan(j * h / 2)  
    
    
    first_row = np.concatenate(([column[0]], column[:0:-1]))
    D = toeplitz(column, first_row)
    
    
    error = np.max(np.abs(D @ u - uprime))
    
    
    plt.loglog(N, error, 'b.', markersize=8)

plt.grid(True, which="both", ls="--")

plt.xlabel('N')
plt.ylabel('error')

plt.title('Convergence of spectral differentiation')

plt.savefig('Spectral Methods\Assignment 1\Figures\Ch1Q5b.pdf', dpi=150)
print("Done. Plot saved to Ch1Q5b.pdf")