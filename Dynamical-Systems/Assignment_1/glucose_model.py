#This is the base code used for all experiments to find equilibria,
# and finding orbits to plot. 

import numpy as np
from g_simul import RK2step
import matplotlib.pyplot as plt

def equilibria(p):
    """Equilibria of the model, using the same formulas as g_simul.py.
    p = [a, c, b, G].  Returns (eq_plus, eq_minus), which are ([u_plus,v_plus], [u_minus,v_minus]).
    If there are no equilibria (D < 0) it returns (None, None)."""
    a, c, b, G = p
    D = (a + b)**2 + 4.0*a*b*G
    if D < 0:
        return None, None
    s = np.sqrt(D)
    eq_plus  = np.array([(-(a+b)+s)/(2.0*a), (-(a+b)+s)/(2.0*b)])   # the "normal" equilibrium
    eq_minus = np.array([(-(a+b)-s)/(2.0*a), (-(a+b)-s)/(2.0*b)])   # the saddle
    return eq_plus, eq_minus

def run_orbit(u0, p, T=100.0, h=0.002):
    """Time-step from u0 = [u, v] and decide what happened. Returns
    (outcome, times, u values, v values), where outcome is one of:
      'controlled' : settled on the equilibrium and v never went below -1
      'recovered'  : settled on the equilibrium, but v went below -1 on the way
      'crash'      : v fell below -100 (glucose crashed)
      'unsettled'  : none of the above within time T"""
    p = np.array(p, dtype=float)
    u = np.array(u0, dtype=float)
    eq_plus, eq_minus = equilibria(p)

    times = [0.0]
    us = [u[0]]
    vs = [u[1]]
    outcome = "unsettled"

    for step in range(1, int(T/h) + 1):
        u = RK2step(u, h, p)
        times.append(step*h)
        us.append(u[0])
        vs.append(u[1])

        # crash: if v too negative (or numbers blew up)
        if u[1] < -100.0 or abs(u[0]) > 1e6 or not np.isfinite(u).all():
            outcome = "crash"
            break
        # settled: close to the equilibrium (checked every 50 steps to save time)
        if eq_plus is not None and step % 50 == 0:
            if np.hypot(u[0]-eq_plus[0], u[1]-eq_plus[1]) < 1e-3:
                outcome = "controlled" if min(vs) > -1.0 else "recovered"
                break

    return outcome, np.array(times), np.array(us), np.array(vs)


