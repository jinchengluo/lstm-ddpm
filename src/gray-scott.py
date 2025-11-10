import matplotlib.pyplot as plt
import numpy as np
import os
import subprocess as sp
import time
from tqdm import tqdm

from constants import *

def laplacian(u,h):
    return (u[2:, 1:-1] + u[1:-1, 2:] + u[:-2, 1:-1] + u[1:-1, :-2] - 4 * u[1:-1, 1:-1]) / h**2

def periodic_bound_conditions(u):
    u[0, :] = u[-2, :]
    u[-1, :] = u[1, :]
    u[:, 0] = u[:, -2]
    u[:, -1] = u[:, 1]

class GrayScott:
    def __init__(self, F=0.04, k=0.06, D_u=2e-5, D_v=1e-5, x0=-1, x1=1, N=256, gen=False):
        self.F = F
        self.k = k
        self.D_u = D_u
        self.D_v = D_v
        self.count_frame = 0
        self.gen = gen

        Nnodes = N + 1
        
        range = x1 - x0
        dx = range / N
        self.x0 = x0
        self.x1 = x1
        self.dx = dx
        self.dt = dx**2 / (5*max(D_u, D_v))

        self.x, self.y = np.meshgrid(np.linspace(x0-dx, x1+dx, Nnodes+2), np.linspace(x0-dx, x1+dx, Nnodes+2))

        self.U = np.zeros((Nnodes+2, Nnodes+2))
        self.V = np.zeros((Nnodes+2, Nnodes+2))
        self.U[1:-1, 1:-1] = 1 - np.exp(-160*((self.x[1:-1, 1:-1]+0.05)**2 + (self.y[1:-1, 1:-1]+0.05)**2))
        self.V[1:-1, 1:-1] = np.exp(-160*((self.x[1:-1, 1:-1]-0.05)**2 + (self.y[1:-1, 1:-1]-0.05)**2))

        # noise_strength = 0.05 # between 0.005 and 0.05
        # self.u[1:-1, 1:-1] += noise_strength * (np.random.rand(*self.u[1:-1, 1:-1].shape) - 0.5)
        # self.v[1:-1, 1:-1] += noise_strength * (np.random.rand(*self.v[1:-1, 1:-1].shape) - 0.5)

        periodic_bound_conditions(self.U)
        periodic_bound_conditions(self.V)
    
    def create_frame(self, time):

        x = self.x[1:-1, 1:-1]
        y = self.y[1:-1, 1:-1]
        U = self.U[1:-1, 1:-1]
        V = self.V[1:-1, 1:-1]

        fig, ax = plt.subplots(1, 2, figsize=(16, 8))
        ax[0].contourf(x, y, U, levels=50, cmap='jet')
        ax[1].contourf(x, y, V, levels=50, cmap='jet')
        fig.suptitle(f"time = {time:e}")
        lim = (self.x0, self.x1)
        species = ("U", "V")
        for a, l in zip(ax, species):
            a.set_title(f"Species {l}")
            a.set_xlabel("x")
            a.set_ylabel("y")
            a.set_xlim(lim)
            a.set_ylim(lim)
            a.set_aspect('equal')

        fig.savefig(os.path.join(f"../frames/f{self.F:.3f}", f"frame_{self.count_frame:06d}.png"), dpi=400)
        plt.close(fig)
        self.count_frame += 1

    def step(self):
        U_view = self.U[1:-1, 1:-1]
        V_view = self.V[1:-1, 1:-1]

        u_v2 = U_view * V_view * V_view
        self.U[1:-1, 1:-1] += self.dt * (self.D_u * laplacian(self.U, self.dx) - u_v2 + self.F * (1 - U_view))
        self.V[1:-1, 1:-1] += self.dt * (self.D_v * laplacian(self.V, self.dx) + u_v2 - (self.F + self.k) * V_view)

        periodic_bound_conditions(self.U)
        periodic_bound_conditions(self.V)

    def forward(self, t0, t1):
        t = t0
        s = 0

        if not os.path.exists("../frames"):
            os.makedirs("../frames")
        if not os.path.exists(f"../frames/f{self.F:.3f}"):
            os.makedirs(f"../frames/f{self.F:.3f}")

        self.create_frame(t)

        total_steps = int((t1 - t0) / self.dt)

        with tqdm(total=total_steps, desc=f"F={self.F:.3f}, k={self.k:.3f}", ncols=100) as pbar:
            while t < t1:
                if self.gen and s > 0 :
                    if s % 100 == 0:
                        self.create_frame(t)
                self.step()
                t += self.dt
                if (t1 - t) < self.dt:
                    self.dt = t1 - t
                s += 1
                if s % 10 == 0:
                    pbar.update(10)
        
        self.create_frame(t)
        return self.U, self.V
    
if __name__ == "__main__":
    for F_test in F_values:
        grayscott = GrayScott(F=F_test, k=k, D_u=D_u, D_v=D_v, x0=x0, x1=x1, N=N)
        t0 = time.perf_counter()
        U, V = grayscott.forward(0, time_length)
        t1 = time.perf_counter()
        print(f"Execution time for a {time_length} ms sequence : {t1-t0} seconds")