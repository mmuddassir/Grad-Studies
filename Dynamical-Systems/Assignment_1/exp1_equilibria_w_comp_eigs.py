import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import brentq
from g_simul import Jac
from glucose_model import equilibria

a, c, b = 5.0, 1.0, 1.0  # Control parameters

Gstar = -(a+b)**2 / (4.0*a*b)  # No equilibria for G < G*
print(f"G* = {Gstar}")

Gs = np.linspace(Gstar + 1e-6, 3.0, 300)


def plus_eigs(G):
    """Eigenvalues of the Jacobian on the plus branch at a given G."""
    p = [a, c, b, G]
    eq_plus, _ = equilibria(p)
    return np.linalg.eigvals(Jac(eq_plus, p))


def max_real(G):
    return plus_eigs(G).real.max()


v_plus, v_minus, u_plus, u_minus = [], [], [], []
stable, complex_eig, max_re = [], [], []

for G in Gs:
    p = [a, c, b, G]
    eq_plus, eq_minus = equilibria(p)

    u_plus.append(eq_plus[0]);   v_plus.append(eq_plus[1])
    u_minus.append(eq_minus[0]); v_minus.append(eq_minus[1])

    eigenvalues = np.linalg.eigvals(Jac(eq_plus, p))
    max_re.append(eigenvalues.real.max())
    stable.append(eigenvalues.real.max() < 0)
    complex_eig.append(np.any(np.abs(eigenvalues.imag) > 1e-9))

v_plus = np.array(v_plus);   u_plus = np.array(u_plus)
v_minus = np.array(v_minus); u_minus = np.array(u_minus)
stable = np.array(stable)
complex_eig = np.array(complex_eig)
max_re = np.array(max_re)

if stable.all():
    print("The plus branch is stable for every G tested.")
else:
    print("The plus branch is UNSTABLE for G between %.3f and %.3f"
          % (Gs[~stable].min(), Gs[~stable].max()))

# ---- Hopf bifurcation detection -------------------------------------------
# A Hopf point is where the real part of a COMPLEX pair crosses zero.
# We look for sign changes of max Re(lambda) between neighbouring G values,
# then refine each one with a root finder.
hopf = []  # list of (G, u, v)
for i in range(len(Gs) - 1):
    if max_re[i] * max_re[i+1] < 0:
        G_h = brentq(max_real, Gs[i], Gs[i+1])
        eigs = plus_eigs(G_h)
        # Only a Hopf if the eigenvalues at the crossing are a complex pair
        # (a real eigenvalue crossing zero would be a different bifurcation)
        if np.any(np.abs(eigs.imag) > 1e-6):
            eq_h, _ = equilibria([a, c, b, G_h])
            hopf.append((G_h, eq_h[0], eq_h[1]))
            print(f"Hopf bifurcation at G = {G_h:.5f}, "
                  f"omega = {np.abs(eigs.imag).max():.4f}")

# ---- Saddle-node: plus and minus branches collide at G* --------------------
sn_G = Gs[0]
sn_u = 0.5 * (u_plus[0] + u_minus[0])
sn_v = 0.5 * (v_plus[0] + v_minus[0])
print(f"Saddle-node at G = {Gstar:.5f}")

# ---- Plotting ---------------------------------------------------------------
# Line style encodes the eigenvalue type, colour encodes stability:
#   solid  = real eigenvalues      dotted = complex eigenvalues (spirals)
#   green  = stable                red    = unstable
masks = {
    "stable, real":      (stable & ~complex_eig,   'g', '-'),
    "stable, complex":   (stable & complex_eig,    'g', ':'),
    "unstable, real":    (~stable & ~complex_eig,  'r', '-'),
    "unstable, complex": (~stable & complex_eig,   'r', ':'),
}

fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
for ax, plus, minus, name, idx in [(axes[0], v_plus, v_minus, "v", 2),
                                   (axes[1], u_plus, u_minus, "u", 1)]:
    for label, (m, col, ls) in masks.items():
        ax.plot(Gs, np.where(m, plus, np.nan), color=col, ls=ls, lw=2,
                label=f"plus branch ({label})")
    ax.plot(Gs, minus, 'k--', label="minus branch (saddle)")
    ax.axvline(Gstar, color='b', ls=':', alpha=0.4)

    # Saddle-node marker
    ax.plot(Gstar, sn_v if name == "v" else sn_u, 'bs', ms=9,
            label="saddle-node (G*)", zorder=5)

    # Hopf markers
    for k, (G_h, u_h, v_h) in enumerate(hopf):
        ax.plot(G_h, v_h if name == "v" else u_h, 'o', color='gold',
                mec='k', ms=10, zorder=5,
                label="Hopf" if k == 0 else None)

    ax.set_xlabel("G"); ax.set_ylabel(name + " at equilibrium")

axes[0].axhline(-1, color='grey', ls='-.')  # v = -1 means zero glucose
axes[0].legend(fontsize=7)
fig.suptitle("Equilibria: a=%g, c=%g, b=%g" % (a, c, b))
plt.tight_layout()
plt.savefig("Dynamical-Systems/exp1_equilibria.pdf", dpi=150)
plt.show()