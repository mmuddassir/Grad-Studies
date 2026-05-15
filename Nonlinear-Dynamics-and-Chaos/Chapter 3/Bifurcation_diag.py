"""
Bifurcation Diagram Generator
==============================
Define any 1-D ODE  xdot = f(x, p)  and sweep a parameter p
to produce a bifurcation diagram.

Usage
-----
  python bifurcation.py

Customise the three sections marked  ← EDIT THIS  below.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


# ─────────────────────────────────────────────────────────────────────────────
# 1.  DEFINE YOUR ODE  ← EDIT THIS
# ─────────────────────────────────────────────────────────────────────────────
def xdot(x, p):
    """Right-hand side of xdot = f(x, p)."""
    #return p - x**2          # saddle-node normal form (default example)
    # Other examples to try:
    #return p*x + x**3      # supercritical pitchfork
    #return p*x - x**2      # transcritical
    return x*(p - x)       # logistic  (p = carrying capacity / growth rate)


# ─────────────────────────────────────────────────────────────────────────────
# 2.  PARAMETER & STATE RANGES  ← EDIT THIS
# ─────────────────────────────────────────────────────────────────────────────
P_MIN, P_MAX   = -1.0,  2.0   # parameter sweep range
X_MIN, X_MAX   = -3.0,  3.0   # x range to search for fixed points
PARAM_NAME     = "p"           # label shown on x-axis
STATE_NAME     = "x"           # label shown on y-axis


# ─────────────────────────────────────────────────────────────────────────────
# 3.  SOLVER SETTINGS  ← EDIT THIS (optional)
# ─────────────────────────────────────────────────────────────────────────────
N_PARAM   = 800    # number of parameter values to sweep
N_XGRID   = 300    # grid resolution for fixed-point search
FP_TOL    = 1e-6   # tolerance for declaring two fixed points equal


# ─────────────────────────────────────────────────────────────────────────────
# Core algorithm
# ─────────────────────────────────────────────────────────────────────────────
def find_fixed_points(f, p_val, x_vals):
    """
    Find fixed points of  xdot = f(x, p_val)  by scanning for sign changes
    in f and refining each root with bisection.
    Returns list of (x_fp, stability) where stability ∈ {'stable', 'unstable', 'half-stable'}.
    """
    from scipy.optimize import brentq

    y = np.array([f(x, p_val) for x in x_vals])
    fps = []

    for i in range(len(y) - 1):
        # sign change → bracket contains a root
        if y[i] * y[i+1] < 0:
            try:
                x_fp = brentq(lambda x: f(x, p_val), x_vals[i], x_vals[i+1], xtol=1e-10)
            except ValueError:
                continue
            fps.append(x_fp)
        # touching zero without crossing → half-stable / degenerate
        elif abs(y[i]) < FP_TOL * 10:
            fps.append(x_vals[i])

    # remove duplicates
    unique = []
    for xf in fps:
        if all(abs(xf - xu) > FP_TOL for xu in unique):
            unique.append(xf)

    results = []
    dx = (x_vals[-1] - x_vals[0]) / len(x_vals) * 0.01
    for xf in unique:
        df = (f(xf + dx, p_val) - f(xf - dx, p_val)) / (2 * dx)
        if df < -FP_TOL:
            stab = 'stable'
        elif df > FP_TOL:
            stab = 'unstable'
        else:
            stab = 'half-stable'
        results.append((xf, stab))

    return results


def compute_bifurcation(f, p_range, x_range, n_p, n_x):
    """Sweep parameter and collect fixed points with their stability."""
    p_vals  = np.linspace(*p_range, n_p)
    x_scan  = np.linspace(*x_range, n_x)

    stable_p, stable_x       = [], []
    unstable_p, unstable_x   = [], []
    halfstable_p, halfstable_x = [], []

    for p in p_vals:
        for xf, stab in find_fixed_points(f, p, x_scan):
            if stab == 'stable':
                stable_p.append(p);     stable_x.append(xf)
            elif stab == 'unstable':
                unstable_p.append(p);   unstable_x.append(xf)
            else:
                halfstable_p.append(p); halfstable_x.append(xf)

    return (np.array(stable_p),    np.array(stable_x),
            np.array(unstable_p),  np.array(unstable_x),
            np.array(halfstable_p),np.array(halfstable_x))


# ─────────────────────────────────────────────────────────────────────────────
# Plotting
# ─────────────────────────────────────────────────────────────────────────────
def plot_bifurcation(data, param_name, state_name):
    sp, sx, up, ux, hp, hx = data

    fig, ax = plt.subplots(figsize=(9, 5))
    fig.patch.set_facecolor('#0f0f11')
    ax.set_facecolor('#0f0f11')

    STABLE_C    = '#4dabf7'   # blue
    UNSTABLE_C  = '#ff6b6b'   # red
    HALF_C      = '#ffd43b'   # yellow

    DOT = 6

    if len(sx):
        ax.scatter(sp, sx, s=DOT, color=STABLE_C,   linewidths=0, label='Stable',      zorder=3)
    if len(ux):
        ax.scatter(up, ux, s=DOT, color=UNSTABLE_C, linewidths=0, label='Unstable',    zorder=3)
    if len(hx):
        ax.scatter(hp, hx, s=DOT, color=HALF_C,     linewidths=0, label='Half-stable', zorder=3)

    # grid
    ax.grid(color='#ffffff', alpha=0.06, linewidth=0.5)
    ax.spines[['top','right']].set_visible(False)
    for sp_ in ax.spines.values():
        sp_.set_color('#444')

    ax.tick_params(colors='#aaa', labelsize=10)
    ax.set_xlabel(param_name, color='#ccc', fontsize=13)
    ax.set_ylabel(f'{state_name}*  (fixed points)', color='#ccc', fontsize=13)
    ax.set_title('Bifurcation diagram', color='#eee', fontsize=15, pad=14)

    legend_elements = [
        Line2D([0],[0], marker='o', color='w', markerfacecolor=STABLE_C,   markersize=7, label='Stable',      linestyle='None'),
        Line2D([0],[0], marker='o', color='w', markerfacecolor=UNSTABLE_C, markersize=7, label='Unstable',    linestyle='None'),
        Line2D([0],[0], marker='o', color='w', markerfacecolor=HALF_C,     markersize=7, label='Half-stable', linestyle='None'),
    ]
    ax.legend(handles=legend_elements, facecolor='#1a1a1e', edgecolor='#444',
              labelcolor='#ccc', fontsize=10, markerscale=1.2)

    plt.tight_layout()
    plt.savefig(r'Nonlinear-Dynamics-and-Chaos\Chapter 3\Figures\bifurcation_diagram.pdf', dpi=150, bbox_inches='tight',
                facecolor=fig.get_facecolor())
    print("Saved → bifurcation_diagram.png")
    


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    print(f"Sweeping {PARAM_NAME} ∈ [{P_MIN}, {P_MAX}]  with {N_PARAM} steps …")
    data = compute_bifurcation(
        xdot,
        p_range=(P_MIN, P_MAX),
        x_range=(X_MIN, X_MAX),
        n_p=N_PARAM,
        n_x=N_XGRID,
    )
    n_fp = len(data[0]) + len(data[2]) + len(data[4])
    print(f"Found {n_fp} fixed-point instances total.")
    plot_bifurcation(data, PARAM_NAME, STATE_NAME)