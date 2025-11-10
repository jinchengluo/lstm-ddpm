import numpy as np

### GRAY SCOTT

F = 0.028 # baseline 0.04
F_values = np.linspace(0.025, 0.06, 10)
k = 0.06 # baseline 0.06
D_u = 2e-5 # baseline 2e-5
D_v = 1e-5 # balise 1e-5
x0 = -1 
x1 = 1
N = 256
time_length = 3000