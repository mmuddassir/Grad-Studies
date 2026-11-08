import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import brentq
from g_simul import Jac
from glucose_model import equilibria

# ---- Parameters ----
a = 0.1
c = 1.0
b = 1.0

# No equilibria exist for G below G*
Gstar = -(a + b)**2 / (4.0 * a * b)
print("G* =", Gstar)

# Values of G to scan (start just above G* so the equilibria exist)
Gs = np.linspace(Gstar + 1e-6, 3.0, 300)


# ---- Function used by the root finder ----
def max_real(G):
    """Largest real part of the eigenvalues on the plus branch at this G."""
    p = [a, c, b, G]
    eq_plus, eq_minus = equilibria(p)
    eigenvalues = np.linalg.eigvals(Jac(eq_plus, p))
    return eigenvalues.real.max()


# ---- Loop over G and store the results ----
u_plus = []
v_plus = []
u_minus = []
v_minus = []
max_re = []      # largest real part of the eigenvalues
stable = []      # True if the plus branch is stable
complex_eig = [] # True if the eigenvalues are complex (spiral)

for G in Gs:
    p = [a, c, b, G]
    eq_plus, eq_minus = equilibria(p)

    u_plus.append(eq_plus[0])
    v_plus.append(eq_plus[1])
    u_minus.append(eq_minus[0])
    v_minus.append(eq_minus[1])

    eigenvalues = np.linalg.eigvals(Jac(eq_plus, p))
    max_re.append(eigenvalues.real.max())
    stable.append(eigenvalues.real.max() < 0)
    complex_eig.append(np.any(np.abs(eigenvalues.imag) > 1e-9))

# Turn the lists into arrays so we can do maths and masking on them
u_plus = np.array(u_plus)
v_plus = np.array(v_plus)
u_minus = np.array(u_minus)
v_minus = np.array(v_minus)
max_re = np.array(max_re)
stable = np.array(stable)
complex_eig = np.array(complex_eig)

# ---- Is the plus branch stable? ----
if stable.all():
    print("The plus branch is stable for every G tested.")
else:
    print("The plus branch is UNSTABLE for G between %.3f and %.3f"
          % (Gs[~stable].min(), Gs[~stable].max()))

# ---- Find Hopf bifurcations ----
# A Hopf point is where the real part of a COMPLEX pair of eigenvalues crosses zero.
# Step 1: look for a sign change in max_re between neighbouring G values.
# Step 2: use brentq to find the exact G where it crosses zero.
# Step 3: check the eigenvalues there are complex (a real eigenvalue
#         crossing zero would be a different type of bifurcation).
hopf_G = []
hopf_u = []
hopf_v = []

for i in range(len(Gs) - 1):
    if max_re[i] * max_re[i + 1] < 0:
        G_h = brentq(max_real, Gs[i], Gs[i + 1])

        p = [a, c, b, G_h]
        eq_h, eq_minus_h = equilibria(p)
        eigenvalues = np.linalg.eigvals(Jac(eq_h, p))

        if np.any(np.abs(eigenvalues.imag) > 1e-6):
            hopf_G.append(G_h)
            hopf_u.append(eq_h[0])
            hopf_v.append(eq_h[1])
            print("Hopf bifurcation at G = %.5f, omega = %.4f"% (G_h, np.abs(eigenvalues.imag).max()))

# ---- Saddle-node point ----
# At G = G* the two branches meet (the discriminant is zero)
sn_u = -(a + b) / (2.0 * a)
sn_v = -(a + b) / (2.0 * b)
print("Saddle-node at G = %.5f" % Gstar)

# ---- Split the plus branch into four pieces for plotting ----
# Colour shows stability (green = stable, red = unstable)
# Line style shows eigenvalue type (solid = real, dotted = complex)
# np.where keeps the value where the condition is True and puts nan
# (which matplotlib doesn't draw) everywhere else.
stable_real = stable & ~complex_eig
stable_complex = stable & complex_eig
unstable_real = ~stable & ~complex_eig
unstable_complex = ~stable & complex_eig

# ---- Plot ----
fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

# Left plot: v against G
ax = axes[0]
ax.plot(Gs, np.where(stable_real, v_plus, np.nan), 'g-', lw=2, label="plus branch (stable, real)")
ax.plot(Gs, np.where(stable_complex, v_plus, np.nan), 'g:', lw=2, label="plus branch (stable, complex)")
ax.plot(Gs, np.where(unstable_real, v_plus, np.nan), 'r-', lw=2, label="plus branch (unstable, real)")
ax.plot(Gs, np.where(unstable_complex, v_plus, np.nan), 'r:', lw=2, label="plus branch (unstable, complex)")
ax.plot(Gs, v_minus, 'k--', label="minus branch (saddle)")

ax.axvline(Gstar, color='b', ls=':', alpha=0.4)

ax.plot(Gstar, sn_v, 'bs', ms=9, label="saddle-node (G*)")

ax.plot(hopf_G, hopf_v, 'o', color='gold', mec='k', ms=10, label="Hopf")

ax.axhline(-1, color='grey', ls='-.')  # v = -1 means zero glucose

ax.set_xlabel("G")

ax.set_ylabel("v at equilibrium")

ax.legend(fontsize=7)

# Right plot: u against G
ax = axes[1]
ax.plot(Gs, np.where(stable_real, u_plus, np.nan), 'g-', lw=2)
ax.plot(Gs, np.where(stable_complex, u_plus, np.nan), 'g:', lw=2)
ax.plot(Gs, np.where(unstable_real, u_plus, np.nan), 'r-', lw=2)
ax.plot(Gs, np.where(unstable_complex, u_plus, np.nan), 'r:', lw=2)
ax.plot(Gs, u_minus, 'k--')

ax.axvline(Gstar, color='b', ls=':', alpha=0.4)

ax.plot(Gstar, sn_u, 'bs', ms=9)

ax.plot(hopf_G, hopf_u, 'o', color='gold', mec='k', ms=10)

ax.set_xlabel("G")

ax.set_ylabel("u at equilibrium")

fig.suptitle("Equilibria: a=%g, c=%g, b=%g" % (a, c, b))
plt.tight_layout()
plt.savefig("Dynamical-Systems/Assignment_2/PDFs/1_param.pdf", dpi=150)
