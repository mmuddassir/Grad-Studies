# Time-stepper for the glucose control toy model. By L. van Veen, Ontario Tech U, 2026.
# MCSC6120/MATH4041 Special Topics: Qualitative Analysis of Dynamical Systems.
import numpy as np
import matplotlib.pyplot as plt

nDof = 2
nPar = 4

# Time derivatives of the model (non-dimensionalized). In: current state (u,v) (float of shape (nDof,)) and parameters p (float of shape (nPar,)).
# Out: derivatives r (float of shape (nDof,))
def rhs(u,p):
    r = np.zeros((nDof,))
    f = -u[1]-u[0]*(u[1]+1.0)+p[3]
    r[0] = -p[0]*u[0]+p[1]*f+p[2]*u[1]
    r[1] = f
    return r

# Jacobian. In: current state (u,v) (float of shape (nDof,)) and parameters p (float of shape (nPar,)).
# Out: J (float of shape (nDof,nDof)).
def Jac(u,p):
    J = np.zeros((nDof,nDof))
    dfdu0 = -u[1]-1.0
    dfdu1 = -1.0 - u[0]
    J[0,0] = -p[0]+p[1]*dfdu0
    J[0,1] = p[1]*dfdu1+p[2]
    J[1,0] = dfdu0
    J[1,1] = dfdu1
    return J

def dfdp(u,p):
    b = np.zeros((nDof,1))
    b[0,0] = p
    b[1,0] = 1
    return b



def tangent(X,p):

    u = X[0:2,0]
    p[3] = X[2,0]

    F = rhs(u,p)
    J = Jac(u,p)
    b = dfdp(u,p)
    DF = np.concatenate((J,b),axis = 1)
    c1 = DF[0,1]*DF[1,2] - DF[0,2]*DF[1,1]
    c2 = DF[0,0]*DF[1,2] - DF[0,2]*DF[1,0]
    c3 = DF[0,0]*DF[1,1] - DF[0,1]*DF[1,0]

    T = np.array([[c1],[c2],[c3]])
    T/= np.linalg.norm(T,2)

    return T

def correct(X0, p, nCorr, eps_x, eps_f):

    X = X0
    conv = False
    for it in range(nCorr):
        u = X[0:2,0]
        p[3] = X[2,0]
        F = rhs(u,p)
        J = Jac(u,p)
        b = dfdp(u,p)
        DF = np.concatenate((J,b),axis=1)

        mu = np.linalg.solve(DF@np.transpose(DF),2.0*F)
        print(f"mu = {mu}")

        dX = -0.5*np.transpose(DF)@mu
        print(f"dX = {dX})
              
        X += dX

        print(f"||F|| = {np.linalg.norm(F,2)}, ||dX|| = {np.linalg.norm(dX,2)})

        if np.linalg.norm(F,2)<eps_f and np.linalg.norm(dX,2)<eps_x:
            conv = True
            return X
        
        
        

nStep = 160

stepSize = 0.1

eps_x = 1e-12
eps_f = 1e-12
nCorr = 5

p = np.Zeros((nPar,))

p[0]=0.1
p[1] = 1.0
p[2]=1.0
p[3] = 1.6

X = np.zeros((nDof+1,1))

X[0]=1.3
X[1]= 0.13
X[2] = 1.6

curve  = np.zeros((nStep,3))
eigs = np.zeros((nStep,2), dtype = complex)

for step in range(nStep):
    X, J, conv = correct(X, p, nCorr, eps_x, eps_f)
    if conv == False:
        print('correction failed to converge on step %d, exiting')
