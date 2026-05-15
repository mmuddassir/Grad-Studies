"""
Two-Parameter Bifurcation Boundary Diagram
===========================================
Sweeps two parameters (p, q) and maps out regions in parameter space
by the number of fixed points of  xdot = f(x, p, q).

The boundaries between regions are the bifurcation curves.

Usage
-----
  python bifurcation_2param.py
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.lines import Line2D
from scipy.ndimage import gaussian_filter


# ─────────────────────────────────────────────────────────────────────────────
# 1.  DEFINE YOUR ODE  ← EDIT THIS
# ─────────────────────────────────────────────────────────────────────────────
def xdot(x, p, q):
    """Right-hand side of xdot = f(x, p, q)."""
    #return p - x**2 + q*x       # saddle-node shifted by q
    # Other examples:
    #return p*x - x**3 + q     # pitchfork with imperfection q
    return x*(p - x) + q*x    # logistic with linear coupling


# ─────────────────────────────────────────────────────────────────────────────
# 2.  PARAMETER & STATE RANGES  ← EDIT THIS
# ─────────────────────────────────────────────────────────────────────────────
P_MIN, P_MAX = -2.0,  2.0    # horizontal axis range
Q_MIN, Q_MAX = -3.0,  3.0    # vertical axis range
X_MIN, X_MAX = -6.0,  6.0    # x range to search for fixed points

P_NAME = "p"                  # horizontal axis label
Q_NAME = "q"                  # vertical axis label


# ─────────────────────────────────────────────────────────────────────────────
# 3.  RESOLUTION  ← EDIT THIS (higher = slower but sharper boundaries)
# ─────────────────────────────────────────────────────────────────────────────
N_P    = 400    # grid points along p axis
N_Q    = 400    # grid points along q axis
N_X    = 400    # x scan resolution for fixed-point counting


# ─────────────────────────────────────────────────────────────────────────────
# Core: count fixed points at each (p, q)
# ─────────────────────────────────────────────────────────────────────────────
def count_fixed_points(f, p_val, q_val, x_scan):
    """Count sign changes of f(x, p, q) → number of fixed points."""
    y = f(x_scan, p_val, q_val)
    sign_changes = np.sum(y[:-1] * y[1:] < 0)
    return int(sign_changes)


def compute_grid(f, p_vals, q_vals, x_scan):
    """Build a 2D grid of fixed-point counts over (p, q) space."""
    grid = np.zeros((len(q_vals), len(p_vals)), dtype=int)
    total = len(p_vals) * len(q_vals)
    done  = 0
    print_every = total // 20

    for i, q in enumerate(q_vals):
        for j, p in enumerate(p_vals):
            grid[i, j] = count_fixed_points(f, p, q, x_scan)
            done += 1
            if done % print_every == 0:
                print(f"  {100*done//total}% …", flush=True)

    return grid


# ─────────────────────────────────────────────────────────────────────────────
# Plotting
# ─────────────────────────────────────────────────────────────────────────────
def plot_boundary(grid, p_vals, q_vals, p_name, q_name):
    fig, ax = plt.subplots(figsize=(8, 6))
    fig.patch.set_facecolor('#0f0f11')
    ax.set_facecolor('#0f0f11')

    n_fp_vals = np.unique(grid)

    # Color regions by number of fixed points
    region_colors = {
        0: '#1a1a2e',   # dark navy  — no fixed points
        1: '#16213e',   # deep blue  — one fixed point
        2: '#0f3460',   # medium blue — two fixed points
        3: '#1a472a',   # dark green  — three fixed points
        4: '#2d4a1e',   # olive       — four fixed points
    }
    default_color = '#2c2c3e'

    # Build a custom colormap from the region colors
    n = len(n_fp_vals)
    color_list = [region_colors.get(v, default_color) for v in sorted(n_fp_vals)]
    cmap = mcolors.ListedColormap(color_list)
    # one bound per gap between unique values, always len(n_fp_vals)+1 bounds
    bounds = np.concatenate([[n_fp_vals[0] - 0.5],
                              (n_fp_vals[:-1] + n_fp_vals[1:]) / 2,
                              [n_fp_vals[-1] + 0.5]])
    norm = mcolors.BoundaryNorm(bounds, cmap.N)

    extent = [p_vals[0], p_vals[-1], q_vals[0], q_vals[-1]]
    im = ax.imshow(grid, origin='lower', extent=extent, aspect='auto',
                   cmap=cmap, norm=norm, interpolation='nearest')

    # Draw bifurcation boundaries as contour lines between regions
    BOUNDARY_COLOR = '#e0e0ff'
    for threshold in range(int(n_fp_vals.min()), int(n_fp_vals.max()) + 1):
        try:
            ax.contour(p_vals, q_vals, grid,
                       levels=[threshold - 0.5],
                       colors=[BOUNDARY_COLOR],
                       linewidths=1.2,
                       alpha=0.9)
        except Exception:
            pass

    # Colorbar
    cbar = fig.colorbar(im, ax=ax, pad=0.02)
    cbar.set_label('# fixed points', color='#ccc', fontsize=11)
    cbar.ax.yaxis.set_tick_params(color='#aaa')
    cbar.set_ticks(sorted(n_fp_vals))
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color='#aaa', fontsize=10)
    cbar.outline.set_edgecolor('#444')

    # Labels & style
    ax.set_xlabel(p_name, color='#ccc', fontsize=13)
    ax.set_ylabel(q_name, color='#ccc', fontsize=13)
    ax.set_title('Two-parameter bifurcation diagram', color='#eee', fontsize=14, pad=12)
    ax.tick_params(colors='#aaa', labelsize=10)
    for spine in ax.spines.values():
        spine.set_color('#444')

    ax.axhline(0, color='#ffffff', linewidth=0.4, alpha=0.3)
    ax.axvline(0, color='#ffffff', linewidth=0.4, alpha=0.3)

    plt.tight_layout()
    plt.savefig(r'Nonlinear-Dynamics-and-Chaos\Chapter 3\Figures\bifurcation_2param.pdf', dpi=150, bbox_inches='tight',
                facecolor=fig.get_facecolor())
    print("Saved → bifurcation_2param.png")
    


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    p_vals = np.linspace(P_MIN, P_MAX, N_P)
    q_vals = np.linspace(Q_MIN, Q_MAX, N_Q)
    x_scan = np.linspace(X_MIN, X_MAX, N_X)

    # Vectorise xdot over x so the scan is fast
    def f_vec(x_arr, p, q):
        return np.array([xdot(x, p, q) for x in x_arr])

    print(f"Computing {N_P}×{N_Q} grid …")
    grid = compute_grid(f_vec, p_vals, q_vals, x_scan)

    vals, counts = np.unique(grid, return_counts=True)
    for v, c in zip(vals, counts):
        pct = 100 * c / grid.size
        print(f"  {v} fixed point(s): {pct:.1f}% of parameter space")

    plot_boundary(grid, p_vals, q_vals, P_NAME, Q_NAME)