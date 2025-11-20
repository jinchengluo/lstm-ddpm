import numpy as np

### GRAY SCOTT

F_DEFAULT = 0.0545 # baseline 0.04
F_values = np.linspace(0.025, 0.06, 10)
k_DEFAULT = 0.062 # baseline 0.06
k_values = np.linspace(0.055, 0.065, 10)
Du_DEFAULT = 2e-5 # baseline 2e-5
Dv_DEFAULT = 1e-5 # balise 1e-5
x0 = -1 
x1 = 1
N = 256
time_length = 15000