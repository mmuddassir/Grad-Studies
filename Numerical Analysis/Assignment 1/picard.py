import numpy as np
import matplotlib.pyplot as plt


t1 = 1.0


N_values = 10**np.arange(2,8)
e_N_list = []


for N in N_values:
    
    t = np.linspace(0, t1, N)
    dt = t[1] - t[0]
    
    # Exact solution for comparison later
    u_exact = np.sqrt(4 + 2 * t) - 1
    
    # Initial guess u^(0)(t) = 1
    u = np.ones(N)
    
    prev_diff = float('inf')
    
   
    for k in range(100): # max 100 iterations as a safety net
        integrand = 1.0 / (1.0 + u)
        
        # trapezoidal integration
        integral = np.zeros(N)
        for i in range(1, N):
            integral[i] = integral[i-1] + 0.5 * dt * (integrand[i-1] + integrand[i])
        
        u_new = 1.0 + integral
        
        # Check change from the previous iteration
        diff = np.max(np.abs(u_new - u))
        
        # If the iteration error starts going back up (numerical noise), stop
        if diff > prev_diff:
            break
            
        prev_diff = diff
        u = u_new
        
        # Stop if we've reached machine precision
        if diff < 1e-14:
            break

    # Once converged, calculate the minimal error e_N 
    # (max absolute difference between our converged numerical solution and the exact solution)
    e_N = np.max(np.abs(u - u_exact))
    e_N_list.append(e_N)
    
    print(f"N = {N:5d} -> Minimal Error (e_N) = {e_N:.2e}")


plt.figure(figsize=(8, 5))
plt.loglog(N_values, e_N_list, 'bo-', linewidth=2, markersize=6, label='Minimal Error ($e_N$)')
plt.xlabel('Number of Grid Points ($N$, Log Scale)')
plt.ylabel('Minimal Error ($e_N$, Log Scale)')
plt.title('Plot of $e_N$ versus $N$')
plt.grid(True, which="both", ls="--")
plt.legend()
plt.savefig("Numerical Analysis\Assignment 1\picard.pdf")
plt.show()