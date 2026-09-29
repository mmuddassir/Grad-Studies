import matplotlib.pyplot as plt
import numpy as np

t = np.linspace(0,10)

x = np.exp(t) # Easier to write x in equations 
              # instead of np.exp(t) every time

plt.figure()
for d in [-10,-5,-2.5,0,0.25, 0.5, 1, 2,10,20]:
    u = d*x / (1 + d*x)
    plt.plot(t, u, label=f'd={d}')
plt.xlabel('t')
plt.ylabel('u')
plt.legend()
plt.savefig("Sol_plot.pdf")
