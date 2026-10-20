# exp2_step_response.py -- start from u0 = [u, v], switch on constant G, scan G.
import numpy as np
import matplotlib.pyplot as plt
from glucose_model import run_orbit, equilibria

a, c, b = 0.1, 1.0, 1.0                     # <-- change these to try other controllers
u0 = [0,-1.5]                              # <-- change the initial condition: [u, v]
Gstar = -(a + b)**2 / (4.0*a*b)
print("a=%g c=%g b=%g   G* = %.4f   start u0=%s" % (a, c, b, Gstar, u0))

# Part 1: table of outcomes for many G values
Gs = np.linspace(-3.0, 7.0, 200)              # <-- change the G range here
#print("\n   G      outcome")
lowest_controlled = None
for G in Gs:
    outcome, t, u, v = run_orbit(u0, [a, c, b, G])
    #print("%6.3f   %s" % (G, outcome))
    if outcome == "controlled" and lowest_controlled is None:
        lowest_controlled = G
print("\nLowest G in the list that was controlled:", lowest_controlled)
