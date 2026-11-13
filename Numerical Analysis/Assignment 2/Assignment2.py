import numpy as np
import scipy.linalg
import matplotlib.pyplot as plt
import matplotlib as mpl
import os

def solve_pde(u_init, mu, n, dt, tf):
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
    
    steps = int(round(tf / dt))
    t = np.arange(steps + 1) * dt
    max_u = np.zeros(steps + 1)      # max u at every time level
    max_u[0] = np.max(u_int)
    
    # Time loop
    for step in range(steps):
        # F[j] = u[j,t] + dt * mu * sin(u[j,t])
        b = u_int + dt * mu * np.sin(u_int)
        
        # Solve Ly = b, Ux = y
        y = scipy.linalg.solve_triangular(L, P @ b, lower=True, unit_diagonal=True)
        u_int = scipy.linalg.solve_triangular(U, y, lower=False)
        
        max_u[step + 1] = np.max(u_int)
    
    u_final = np.zeros(n + 1)
    u_final[1:n] = u_int
    return x, t, max_u, u_final

def run_convergence_tests():

    tf = 1.0
    mu = 2.0
    u_fn = lambda x: np.cos((np.pi / 2) * x)

    print("\n--- SPATIAL CONVERGENCE TEST ---")
    # Fix a tiny time step so only spatial errors matter
    dt_fixed = 0.0001
    n_values = [50, 100, 200, 400]
    
    # Get a "reference" solution using a very fine grid (n = 800)
    n_ref = 800
    x_ref = np.linspace(-1, 1, n_ref + 1)
    na, nb, nc, u_ref = solve_pde(u_fn(x_ref), mu, n_ref, dt_fixed, tf)
    
    spatial_errors = []
    for n in n_values:
        h = 2.0 / n
        x_grid = np.linspace(-1, 1, n + 1)
        na, nb, nc, u_numerical = solve_pde(u_fn(x_grid), mu, n, dt_fixed, tf)
        
        # Compare numerical solution to every 2nd, 4th, or 8th point of the reference solution
        step_size = n_ref // n
        u_ref_matched = u_ref[::step_size] # Basically makes it so that the the comparison points
                                            # are in the "same" positions as the reference grid.
        
        # Find the maximum error
        error = np.max(np.abs(u_numerical - u_ref_matched))
        spatial_errors.append(error)
        print(f"n = {n}, h = {h:.4f}, Error = {error:.5e}")

    # Estimate slope (should be close to 2.0)
    h_vals = [2.0 / n for n in n_values]
    spatial_rate = np.polyfit(np.log(h_vals), np.log(spatial_errors), 1)[0]
    print(f"Estimated Rate of Convergence for the spatial test: {spatial_rate:.2f} (Expected: ~2.0 for the central scheme)")


    print("\n--- TEMPORAL CONVERGENCE TEST ---")
    # Set a large spatial grid so only time errors matter
    n_fixed = 400
    x_grid = np.linspace(-1, 1, n_fixed + 1)
    dts = [0.04, 0.02, 0.01, 0.005]
    
    # Get a reference solution with an very tiny time step
    dt_ref = 0.0005
    na, nb, nc, u_ref_time = solve_pde(u_fn(x_grid), mu, n_fixed, dt_ref, tf)
    
    time_errors = []
    for dt in dts:
        na, nb, nc, u_numerical = solve_pde(u_fn(x_grid), mu, n_fixed, dt, tf)
        
        # Find the maximum error compared to the reference time solution
        error = np.max(np.abs(u_numerical - u_ref_time))
        time_errors.append(error)
        print(f"dt = {dt}, Error = {error:.5e}")

    # Estimate slope (should be close to 1.0)
    time_rate = np.polyfit(np.log(dts), np.log(time_errors), 1)[0]
    print(f"Estimated Rate of Convergence for the time test: {time_rate:.2f} (Expected: ~1.0 for the forward scheme)")


if __name__ == "__main__":
    n = 400
    dt = 0.002
    tf = 15.0
    mu_crit = np.pi**2 / 4

    # 37 values total: fine steps near mu_crit, wider steps up to 40
    mu_vals = np.array([2.5, 2.55, 2.6, 2.65, 2.7, 2.8, 2.9, 3.0] + list(range(4, 41, 2)))

    x = np.linspace(-1, 1, n + 1)
    u0 = 0.1 * np.cos((np.pi / 2) * x)

    cmap = mpl.cm.viridis
    norm = mpl.colors.Normalize(vmin=mu_vals.min(), vmax=mu_vals.max())
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 4.5), constrained_layout=True)

    max_final = []
    for i, mu in enumerate(mu_vals):
        x_grid, t, max_u, u_final = solve_pde(u0.copy(), mu, n, dt, tf)
        max_final.append(max_u[-1])

        # Plot only 10 evenly spaced curves (every 3rd item) on ax2 and ax3
        if i % 3 == 0:
            color = cmap(norm(mu))
            ax2.semilogy(t, max_u, color=color)
            ax3.plot(x_grid, u_final, color=color)

    # Plot 1: Bifurcation scaling (uses all mu values)
    delta_mu = mu_vals - mu_crit
    ax1.loglog(delta_mu, max_final, 'o-', markersize=4, label="Numerical")
    
    mu_th = np.linspace(delta_mu.min(), delta_mu.max(), 300)
    ax1.loglog(mu_th, (4 * np.sqrt(2) / np.pi) * np.sqrt(mu_th), 'r--', label="Theory")
    ax1.axhline(np.pi, color='k', linestyle=':', label=r"$u=\pi$")
    ax1.set_ylim(0.1, 10)
    ax1.set_xlabel(r"$\mu - \mu_c$")
    ax1.set_ylabel(r"$\max_x\, u(x, t_f)$")
    ax1.set_title(r"Steady-state amplitude vs $\mu - \mu_c$")
    ax1.grid(True, which="both", alpha=0.5)
    ax1.legend()

    # Plot 2: Time evolution
    ax2.set_xlabel(r"$t$")
    ax2.set_ylabel(r"$\max_x\, u(x, t)$")
    ax2.set_title(r"Time evolution of $\max\, u(x, t)$")
    ax2.grid(True, which="both", alpha=0.5)

    # Plot 3: Final spatial profiles
    ax3.axhline(np.pi, color='red', linestyle='--', label=r"$u=\pi$")
    ax3.set_xlabel(r"$x$")
    ax3.set_ylabel(r"$u(x, t_f)$")
    ax3.set_title(r"Final spatial profile $u(x, t_f)$")
    ax3.grid(True, alpha=0.5)
    ax3.legend(loc="upper right")

    sm = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
    fig.colorbar(sm, ax=[ax1, ax2, ax3], label=r"$\mu$")

    os.makedirs("Numerical Analysis/Assignment 2", exist_ok=True)
    fig.savefig("Numerical Analysis/Assignment 2/plots.pdf")
    plt.show()

    run_convergence_tests()