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
    max_u = np.zeros(steps + 1)      # max_x u at every time level
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


if __name__ == "__main__":
    n = 400
    dt = 0.005
    tf = 10.0
    mu_crit = np.pi**2 / 4
    mu_vals = np.arange(3, 6,0.05)

    x = np.linspace(-1, 1, n + 1)
    u0 = np.cos((np.pi / 2) * x)

    cmap = mpl.cm.viridis
    norm = mpl.colors.Normalize(vmin=mu_vals.min(), vmax=mu_vals.max())

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 4), constrained_layout=True)

    max_final = []   # one entry per mu
    for i, mu in enumerate(mu_vals):
        x_grid, t, max_u, u_final = solve_pde(u0.copy(), mu, n, dt, tf)
        color = cmap(norm(mu))

        max_final.append(max_u[-1])
        ax2.semilogy(t, max_u, color=color)
        ax3.plot(x_grid, u_final, color=color)
        ax3.axhline(np.pi, color='red', linestyle='--', linewidth=1.5, label=r"$u=\pi$")

        print(f"mu = {mu:.1f} done ({i+1}/{len(mu_vals)})")


#Plotting

    ax1.loglog(mu_vals, max_final, 'o-')
    ax1.set_xlabel(r"$\mu$")
    ax1.set_ylabel(r"log($\max\, u(x, t_f)$)")
    ax1.set_title(r"Max of $u$ at $t=t_f$ vs $\mu$")
    ax1.grid(True)

    ax2.set_xlabel(r"$t$")
    ax2.set_ylabel(r"$\max\, u(x, t)$")
    ax2.set_title(r"Max of $u$ vs $t$ (one curve per $\mu$)")
    ax2.grid(True)

    ax3.set_xlabel(r"$x$")
    ax3.set_ylabel(r"$u(x, t_f)$")
    ax3.set_title(r"Final solution (one curve per $\mu$)")
    ax3.grid(True)

#Theoretical Plot for small mu. Not applicable to large mu
    mu_th = np.linspace(mu_vals.min()+0.01, mu_vals.max(), 300)
    u_th = (4*np.sqrt(2)/np.pi)*np.sqrt(mu_th - mu_crit)
    ax1.loglog(mu_th, u_th, 'r--', label=r"$\frac{4 \sqrt{2}}{\pi}\sqrt{\mu-\mu_c}$")
    ax1.legend()   

    sm = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
    fig.colorbar(sm, ax=[ax2, ax3], label=r"$\mu$")

    
    fig.savefig("Numerical Analysis/mu_vs_max_u.pdf")
    #plt.show()