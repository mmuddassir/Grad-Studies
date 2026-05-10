import matplotlib.pyplot as plt
import math
import numpy as np
import sympy as sp

    

def RK4(f,x0,dt,N):
    
    x = np.zeros(N)
    x[0] = x0


    for i in range(0,N-1):

        k1 = f(x[i])*dt
        k2 = f(x[i]+0.5*k1)*dt
        k3 = f(x[i]+0.5*k2)*dt
        k4 = f(x[i]+k3)*dt

        x[i+1] = x[i] + (1/6)*(k1 + 2*k2 + 2*k3 + k4)
    
    return x

