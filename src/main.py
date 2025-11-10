import numpy as np
import matplotlib.pyplot as plt

from grayscott import GrayScott
from constants import *

def spatial_power_spectrum(field):
    # Remove edges if periodic padding was used
    f = field[1:-1, 1:-1]

    # FFT
    F = np.fft.fftshift(np.fft.fft2(f))
    P = np.abs(F)**2

    # Radial average
    ny, nx = f.shape
    cy, cx = ny//2, nx//2
    Y, X = np.ogrid[:ny, :nx]
    r = np.sqrt((X - cx)**2 + (Y - cy)**2)
    r = r.astype(int)
    r_max = r.max()

    radial_ps = np.bincount(r.ravel(), weights=P.ravel()) / np.bincount(r.ravel())

    return radial_ps

def dominant_wavelength(radial_ps, dx):
    peak_index = np.argmax(radial_ps[1:]) + 1
    wavelength = 1 / peak_index * dx
    return wavelength


if __name__ == "__main__":
    gs = GrayScott(F=F, k=k)
    U, V = gs.forward(0, time_length)
    radial_ps = spatial_power_spectrum(V)
    plt.plot(radial_ps)
    plt.yscale("log")
    plt.title("Radial Power Spectrum")
    plt.xlabel("Spatial frequency")
    plt.show()
