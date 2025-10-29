import matplotlib.pyplot as plt
import numpy as np
import os
import time

def laplacian(u,h):
    return (u[2:, 1:-1] + u[1:-1, 2:] + u[:-2, 1:-1] + u[1:-1, :-2] - 4 * u[1:-1, 1:-1]) / h**2

def periodic_bound_conditions(u):
    u[0, :] = u[-2, :]
    u[-1, :] = u[1, :]
    u[:, 0] = u[:, -2]
    u[:, -1] = u[:, 1]

class GrayScott:
    def __init__(self, F=0.04, k=0.06, D_u=2e-5, D_v=1e-5, x0=-1, x1=1, N=256):
        self.F = F
        self.k = k
        self.D_u = D_u
        self.D_v = D_v
        self.image_frame = 0

        Nnodes = N + 1
        
        range = x1 - x0
        dx = range / N
        self.x0 = x0
        self.x1 = x1
        self.dx = dx
        self.dt = dx**2 / (5*max(D_u, D_v))

        self.x, self.y = np.meshgrid(np.linspace(x0-dx, x1+dx, Nnodes+2), np.linspace(x0-dx, x1+dx, Nnodes+2))

        # Concentration definition
        self.u = np.zeros((Nnodes+2, Nnodes+2))
        self.v = np.zeros((Nnodes+2, Nnodes+2))
        self.u[1:-1, 1:-1] = 1 - np.exp(-80*((self.x[1:-1, 1:-1]+0.05)**2 + (self.y[1:-1, 1:-1]+0.05)**2))
        self.v[1:-1, 1:-1] = np.exp(-80*((self.x[1:-1, 1:-1]-0.05)**2 + (self.y[1:-1, 1:-1]-0.05)**2))

        noise_strength = 0.05 # between 0.005 and 0.05
        self.u[1:-1, 1:-1] += noise_strength * (np.random.rand(*self.u[1:-1, 1:-1].shape) - 0.5)
        self.v[1:-1, 1:-1] += noise_strength * (np.random.rand(*self.v[1:-1, 1:-1].shape) - 0.5)

        periodic_bound_conditions(self.u)
        periodic_bound_conditions(self.v)
    
    def _dump(self, time, both=True):
        x = self.x[1:-1, 1:-1]
        y = self.y[1:-1, 1:-1]
        U = self.u[1:-1, 1:-1]
        V = self.v[1:-1, 1:-1]
        if both:
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
        else:
            fig, ax = plt.subplots(1, 1, figsize=(6, 6))
            ax.contourf(x, y, V, levels=50, cmap='jet')
            fig.suptitle(f"time = {time:e}")
            lim = (self.x0, self.x1)
            ax.set_title(f"Species V")
            ax.set_xlabel("x")
            ax.set_ylabel("y")
            ax.set_xlim(lim)
            ax.set_ylim(lim)
            ax.set_aspect('equal')
        fig.savefig(os.path.join(".", f"frame_{self.image_frame:06d}.png"), dpi=400)
        plt.close(fig)
        self.image_frame += 1

    def update(self):
        u_view = self.u[1:-1, 1:-1]
        v_view = self.v[1:-1, 1:-1]

        u_v2 = u_view * v_view * v_view
        self.u[1:-1, 1:-1] += self.dt * (self.D_u * laplacian(self.u, self.dx) - u_v2 + self.F * (1 - u_view))
        self.v[1:-1, 1:-1] += self.dt * (self.D_v * laplacian(self.v, self.dx) + u_v2 - (self.F + self.k) * v_view)

        periodic_bound_conditions(self.u)
        periodic_bound_conditions(self.v)

    def integrate(self, t0, t1):
        t = t0
        s = 0

        self._dump(t)
        while t < t1:
            if s % 100 == 0:
                print(f"step={s}; time={t:e}")
            self.update()
            t += self.dt
            if (t1 - t) < self.dt:
                self.dt = t1 - t
            s += 1
        self._dump(t)

if __name__ == "__main__":
    
    F = 0.04
    k=0.06
    D_u=2e-5
    D_v=1e-5
    x0=-1
    x1=1
    N=256
    time_length = 1500
    
    grayscott = GrayScott(F=F, k=k, D_u=D_u, D_v=D_v, x0=x0, x1=x1, N=N)

    t0 = time.perf_counter()
    grayscott.integrate(0, time_length)
    t1 = time.perf_counter()

    print(f"Execution time for a {time_length} ms sequence : {t1-t0} seconds")