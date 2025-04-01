import numpy as np
import Camera
import matplotlib.pyplot as plt
from scipy.special import j1, j0
from scipy.interpolate import RegularGridInterpolator
from scipy.interpolate import interp1d, griddata
from typing import Tuple, List
import h5py
import os
from scipy import integrate
from functools import partial
from scipy.optimize import curve_fit
from scipy.stats import poisson


imgs, metadata = Camera.getImagesFrom_h5_files("D:/simulationImages/20us")

'''average image'''
average = np.mean(imgs, axis = 0)
plt.imshow(average)
plt.show()

'''captured photons emitted by the atom (without image noise)'''

photonCount = [val["grid 1, number of photons before added noise"] for val in metadata.values()]
unique, counts = np.unique(photonCount, return_counts=True)

# Fit the histogram data to the Poisson distribution
params, _ = curve_fit(poisson.pmf, unique, counts / len(photonCount), p0=[np.mean(photonCount)])

# Plot the fitted Poisson distribution
plt.scatter(unique, poisson.pmf(unique, *params), label=f'Poisson fit (mean = {params[0]})')
plt.scatter(unique, counts / len(photonCount), label='simulated data')
plt.legend()
plt.show()


a=0