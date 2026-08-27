"""Band Limited Interpolation"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

h = 1
xmax = 10.0
x = np.arange(-xmax, xmax + h, h)          # coarse data grid
xx = np.arange(-xmax - h / 20, xmax + h / 20 + h / 10, h / 10)  # fine plotting grid

def sinc_interp(xx, x, v, h):
    """Band-limited (sinc) interpolant of samples v at points x, evaluated on xx."""
    p = np.zeros_like(xx)
    for xj, vj in zip(x, v):
        p += vj * np.sinc((xx - xj) / h)
    return p

datasets = {
    "hat_function": np.where(np.abs(x) <= 3, 1 - np.abs(x) / 3, 0.0),
    "single_spike": (x == 0).astype(float),
    "square_wave": np.where(np.abs(x) <= 3, 1.0, 0.0)
}

fig, axes = plt.subplots(1, 3, figsize=(11, 4))
for ax, (name, v) in zip(axes, datasets.items()):
    p = sinc_interp(xx, x, v, h)
    ax.plot(x, v, "o", label="samples")
    ax.plot(xx, p, "-", label="band-limited interpolant")
    ax.set_title(name.replace("_", " "))
    ax.set_xlim(-xmax, xmax)
    ax.legend(fontsize=8)

plt.tight_layout()
plt.savefig('Spectral Methods\Figures\p3_bandlimited_interpolation.pdf', dpi=150)
print("Saved p3_bandlimited_interpolation.pdf")
