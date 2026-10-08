import numpy as np
import matplotlib.pyplot as plt


nPar = 4
nDof = 2


def rhs(u,p):
    r = np.zeros((nDof,))
    f = -u[1]-u[0]*(u[1]+1.0)+p[3]
    r[0] = -p[0]*u[0]+p[1]*f+p[2]*u[1]
    r[1] = f
    return r


def Jac(u,p):
    J = np.zeros((nDof,nDof))
    dfdu0 = -u[1]-1.0
    dfdu1 = -1.0 - u[0]
    J[0,0] = -p[0]+p[1]*dfdu0
    J[0,1] = p[1]*dfdu1+p[2]
    J[1,0] = dfdu0
    J[1,1] = dfdu1
    return J


def Newt_Raph(x0, e_f, e_x, N,p):
    #Initialize variables
    x = x0
    conv = False
    par = p

    for i in range(0,N):
        F = rhs(x,par)
        DF = Jac(x,par)

        dx = np.linalg.solve(DF,-F)

        x += dx

        print(f"||F|| = {np.linalg.norm(F)}, ||dx|| = {np.linalg.norm(dx)}")

        if np.linalg.norm(dx) < e_x and np.linalg.norm(F) < e_f:
            conv = True
            break


    if conv == True:
        return x
    else:
        print("No convergence!")
        return x


def test():
    p = np.zeros((nPar,))
    p[0] = 0.1
    p[1] = 1.0
    p[2] = 1.0
    p[3] = 1.6
    x0= [1.7, 0.5]

    print(Newt_Raph(x0,10**-13, 10**-13, 10, p))




test()