import numpy as np
import scipy.linalg
import matplotlib.pyplot as plt

def solve_pde_with_history(u_init, mu, n, dt, tf):
    h = 2.0 / n
    x = np.linspace(-1, 1, n + 1)
    size = n - 1
    r = dt / (h ** 2) # common term in all coefficients in M
    
    # INITIALIZE: M = matrix of coefficients
    main_diag = (1.0 + 2.0 * r) * np.ones(size)
    off_diag = -r * np.ones(size - 1) 
    M = np.diag(main_diag) + np.diag(off_diag, -1) + np.diag(off_diag, 1)
    
    # Find M = LU (Done ONCE outside the loop)
    P, L, U = scipy.linalg.lu(M)
    
    # Apply BCs u(-1) = u(1) = 0
    u_init[0] = 0.0
    u_init[-1] = 0.0
    u_int = u_init[1:n].copy()
    
    steps = int(tf / dt)
    save_interval = max(1, steps // 10)  # Save about 10 snapshots between t=0 and tf
                                        # 'max' avoids errors in case steps < 10
    
    snap_time = [0.0]
    solutions = [u_init.copy()]
    
    # Time loop
    for step in range(steps):
        # F[j] = u[j,t] + dt * mu * sin(u[j,t])
        b = u_int + dt * mu * np.sin(u_int)
        
        # Solve Ly = b, Ux = y
        y = scipy.linalg.solve_triangular(L, P @ b, lower=True, unit_diagonal=True)
        u_int = scipy.linalg.solve_triangular(U, y, lower=False)
        
       
        if (step + 1) % save_interval == 0 or (step + 1) == steps:
            u_full = np.zeros(n + 1)
            u_full[1:n] = u_int
            solutions.append(u_full.copy())
            snap_time.append((step + 1) * dt)
            
    return x, snap_time, solutions



#Test run with some IC and also plotting intermediate time steps
if __name__ == "__main__":
    n = 400
    dt = 0.005
    tf = 0.5
    mu = 1.0
    
    x = np.linspace(-1, 1, n + 1)
    
    #Enter ICs
    u_init = np.sin(x+2.45) - x
    
    
    x_grid, snap_times, history = solve_pde_with_history(u_init, mu, n, dt, tf)


    #Used Claude to help make plots look nicer
    
    plt.figure(figsize=(8, 5))
    for t_val, u_sol in zip(snap_times, history):
        plt.plot(x_grid, u_sol, label=f't = {t_val:.2f}')
        
    plt.xlabel("x")
    plt.ylabel("u(x, t)")
    plt.title("Heat Equation Solutions with Implicit-Explicit Finite Difference methods")
    plt.grid(True)
    plt.legend()
    plt.savefig("Numerical Analysis\Heat_Eq_Solns.pdf")
    plt.show()