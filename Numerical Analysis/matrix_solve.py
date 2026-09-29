import numpy as np
import matplotlib.pyplot as plt
import sympy as sp

n=5

A = np.empty((n,n))

x = np.empty((n))

y = A @ x

import numpy as np

result = x

print(np.array2string(result, precision=16, floatmode='fixed'))