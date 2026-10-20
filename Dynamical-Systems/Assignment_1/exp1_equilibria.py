import numpy as np
import matplotlib.pyplot as plt
from g_simul import Jac
from glucose_model import equilibria

a, c, b = 0.1, 1.0, 1.0 # Control parameters

Gstar = -(a+b)**2 / (4.0*a*b) # No equilibria for G < G*
print(f"G* = {Gstar}")


Gs = np.linspace(Gstar + 1e-6, 3.0, 300)

v_plus = []
v_minus = []
u_plus = []
u_minus = []
stable = []

for G in Gs:
    p = [a,c,b,G]
    eq_plus, eq_minus = equilibria(p) #Find equilibrium points for different G values while keeping the other parameters constant

    u_plus.append(eq_plus[0])
    u_minus.append(eq_minus[0])

    v_plus.append(eq_plus[1])
    v_minus.append(eq_minus[1])

    eigenvalues = np.linalg.eigvals(Jac(eq_plus,p))

    stable.append(eigenvalues.real.max() < 0) #If the plus branch is stable, this is True

v_plus = np.array(v_plus); u_plus = np.array(u_plus)
v_minus = np.array(v_minus); u_minus = np.array(u_minus)
stable = np.array(stable)

if stable.all():
    print("The plus branch is stable for every G tested.")
else:
    print("The plus branch is UNSTABLE for G between %.3f and %.3f"
          % (Gs[~stable].min(), Gs[~stable].max()))


#Used Claude's help for this next part just to speed things up and make it look nicer


fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
for ax, plus, minus, name in [(axes[0], v_plus, v_minus, "v"), (axes[1], u_plus, u_minus, "u")]:
    ax.plot(Gs, np.where(stable, plus, np.nan), 'g-', label="plus branch (stable)")
    ax.plot(Gs, np.where(~stable, plus, np.nan), 'r-', label="plus branch (unstable)")
    ax.plot(Gs, minus, 'k--', label="minus branch (saddle)")
    ax.axvline(Gstar, color='b', ls=':', label="G*")
    ax.axvline(-1, color='grey', ls=':', label="G = -1")
    ax.set_xlabel("G"); ax.set_ylabel(name + " at equilibrium")
axes[0].axhline(-1, color='grey', ls='-.')   # v = -1 means zero glucose
axes[0].legend(fontsize=8)
fig.suptitle("Equilibria: a=%g, c=%g, b=%g" % (a, c, b))
plt.savefig("Dynamical-Systems/exp1_equilibria.pdf", dpi=150)
plt.show()






