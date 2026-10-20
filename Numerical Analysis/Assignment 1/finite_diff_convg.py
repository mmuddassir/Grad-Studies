

import numpy as np
import scipy.sparse as sp
import matplotlib.pyplot as plt


Nvec = 2 ** np.arange(7, 25)

fig, ax = plt.subplots(figsize=(7, 5))


ax.set_xlabel('N')
ax.set_ylabel('Error')
ax.set_title('Convergence of finite differences')
ax.grid(True, which='both', linestyle=':', alpha=0.6)
ax.legend()

plt.tight_layout()

#The roll method allows us to shift values by a certain amount and bypasses the need to create 
# sparse matrices for each finite difference scheme. 
S = lambda u, k: np.roll(u, -k) # lambda is just shorthand for defining a function in one line


# The different schemes only differ by the weight of each term and how many terms there are in total.
#So define them in a dictionary as:
schemes = { 
    "forward O(h)":   lambda u, h: (S(u,1) - u) / h,
    "forward O(h²)":  lambda u, h: (-3*u + 4*S(u,1) - S(u,2)) / (2*h),
    "forward O(h⁴)":  lambda u, h: (-25*u + 48*S(u,1) - 36*S(u,2)
                                    + 16*S(u,3) - 3*S(u,4)) / (12*h),
    "central O(h²)":  lambda u, h: (S(u,1) - S(u,-1)) / (2*h),
    "central O(h⁴)":  lambda u, h: (8*(S(u,1) - S(u,-1))
                                    - (S(u,2) - S(u,-2))) / (12*h),
}

#Now calculate the maximum error on the grid of the approximation compared to the exact derivative



for name, D in schemes.items():
    errs = []
    for N in Nvec:
        # Define step size h and grid points x for the current N
        h = 2 * np.pi / N
        x = -np.pi + 2 * np.pi * np.arange(N) / N
        f = np.exp(np.sin(x))
        df = np.cos(x)*np.exp(np.sin(x))


        # Compute the error for this grid
        err = np.max(np.abs(D(f, h) - df))
        
        # Add it to our list of errors
        errs.append(err)

    #Plot the errors vs N for each scheme
    ax.loglog(Nvec, errs, 'o-', label=name)
plt.legend()
plt.savefig('Numerical Analysis\Assignment 1\convergence.pdf', dpi=150)
plt.show()
print("Done. Plot saved to convergence.pdf")

