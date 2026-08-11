
import numpy as np
import matplotlib.pyplot as plt


def hat(x):
    return np.where(np.abs(x) <= 3, 1 - np.abs(x) / 3, 0.0)

def square_wave(x):
    return np.where(np.abs(x) <= 3, 1.0, 0.0)

funcs = {"square wave": square_wave, "hat function": hat}


def sinc_interp(xx, xj, vj, h):
    T = (xx[:, None] - xj[None, :]) / h
    return np.sinc(T) @ vj


hs = [2.0**-3, 2.0**-4, 2.0**-5, 2.0**-6]  # h = 1/8, 1/16, 1/32, 1/64


xx = np.arange(-10, 10, 5e-4) + 1.3e-4

errors = {name: [] for name in funcs}


for h in hs:

    xj = np.arange(-3 - 2 * h, 3 + 2 * h, h)    

    for name, f in funcs.items():

        vj = f(xj)

        p = sinc_interp(xx, xj, vj, h)

        err = np.max(np.abs(f(xx) - p))

        errors[name].append(err)

#Plotting

plt.figure(figsize=(6, 5))

plt.loglog(hs, errors["square wave"], "o-", label="square wave (jump)")
plt.loglog(hs, errors["hat function"], "s-", label="hat function (corner)")


plt.gca().invert_xaxis()

plt.xlabel("h")
plt.ylabel("max error 'over R'")


plt.title("Sinc-interpolation error vs. grid spacing h")

plt.legend()
plt.grid(True, which="both", ls=":")


plt.savefig('Spectral Methods\Assignment 1\Figures\Ch2Q7.pdf', dpi=150)
print("Done. Plot saved to Ch2Q7.pdf")