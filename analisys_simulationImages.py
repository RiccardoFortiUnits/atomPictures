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
from scipy.stats import norm



# import numpy as np
# import matplotlib.pyplot as plt
# from scipy.optimize import curve_fit
# from scipy.stats import poisson

# repetitions = 5000
# scattRate = 3
# dt=.00001
# totalTime = 1
# nSteps = int(totalTime // dt)
# oldNumScatter = np.sum(np.random.random((nSteps, repetitions))<scattRate*dt, axis=0)

# newTimings = np.random.exponential(1/scattRate, (nSteps,repetitions))
# newTimings = np.cumsum(newTimings, axis=0)
# newNumScatter = np.sum(newTimings<totalTime, axis=0)

# plt.plot(*np.unique(oldNumScatter, return_counts=True), alpha=0.5, label='Old Method')
# plt.plot(*np.unique(newNumScatter, return_counts=True), alpha=0.5, label='New Method')
# plt.legend()
# plt.show()


def Gauss(x, A, B, C):
    y = A*np.exp(-1*B*(x-C)**2)
    return y
def get_xyFromSignal(signal, returnParamters = False):
    x=np.arange(len(signal),dtype=float) - np.argmax(signal)
    parameters, covariance = curve_fit(Gauss, x, signal, p0=[1, 1, 0])
    mean = parameters[2]
    x -= mean
    parameters[2] = 0
    if returnParamters:
        return x, signal, parameters
    return x, signal
def getAverageRoi(image, roi, tweezerMinPixelDistance = 10, atomPeakMinValue = 1):
    maxes = maximum_filter(image, size=tweezerMinPixelDistance)
    centers = np.where(np.logical_and((maxes == image), (image > atomPeakMinValue))) - np.repeat(roi//2,2)[:,None]
    finalImage=np.zeros((roi,roi))
    for center in centers.T:
        finalImage += image[center[0]:center[0]+roi, center[1]:center[1]+roi]
    finalImage /= len(centers[0])
    return finalImage
def getAverageCrossSection(image, tweezerMinPixelDistance = 10, atomPeakMinValue = 1):
    maxes = maximum_filter(image, size=tweezerMinPixelDistance)
    centers = np.array(np.where(np.logical_and((maxes == image), (image > atomPeakMinValue))))

    xx=[]
    yy=[]
    for center in centers.T:
        x,y, parameters=get_xyFromSignal(image[center[0], :], returnParamters=True)
        xx.append(x)
        yy.append(y)
    xx = np.array(xx)
    yy = np.array(yy)
    x = np.linspace(np.min(xx), np.max(xx), int(np.ceil(np.max(xx)-np.min(xx))))
    y = np.zeros_like(x)
    for i in range(len(xx)):
        y += interp1d(xx[i], yy[i], kind='linear', fill_value="extrapolate")(x)
    y /= len(xx)
    return x,y





imgs, metadata = Camera.getImagesFrom_h5_files("D:/simulationImages/10us_smallImages/pictures/testing tweezer")
imgs = np.array(imgs)

'''average image'''
average = np.mean(imgs, axis = 0)
plt.imshow(average)
plt.show()

'''captured photons emitted by the atom (without image noise)'''

photonCount = [val["grid 1, number of photons before added noise"] for val in metadata.values()]
unique, counts = np.unique(photonCount, return_counts=True)

params, _ = curve_fit(poisson.pmf, unique, counts / len(photonCount), p0=[np.mean(photonCount)])

plt.plot(unique, poisson.pmf(unique, *params), label=f'Poisson fit (mean = {params[0]})')
plt.plot(unique, counts / len(photonCount), label='simulated data')
plt.legend()
plt.show()

'''image section, to check the gaussian fit'''
for signal, direction in zip([average[:,average.shape[1]//2], average[average.shape[0]//2,:]], ["x","y"]):
    x,signal, parameters = get_xyFromSignal(signal, returnParamters=True)
    plt.plot(x, signal, label=direction)
    plt.plot(x, Gauss(x,*parameters), label=f"fit {direction}")

plt.legend()
plt.show()

'''histogram of number of photons in roy'''

roi = 3
metadata = list(metadata.values())[0]
atomPosition = metadata["atoms_InitialCoordinates"][0]
atomPosition[0] *=-1
pixelSize = metadata["camera_pixelSize"]
pixelNumber = imgs.shape[1]
halfImageSize = pixelNumber*pixelSize/2
atomPixels = np.array([(atomPosition[0]+halfImageSize) / pixelSize,
              (atomPosition[1]+halfImageSize) / pixelSize]).astype(int) - roi//2
atomPixels = imgs[:,atomPixels[0]:atomPixels[0]+roi,atomPixels[1]:atomPixels[1]+roi]
photonsInRoi = np.sum(atomPixels, axis=(1,2))

plt.hist(photonsInRoi, bins=int(np.max(photonsInRoi)-np.min(photonsInRoi)))
plt.show()

'''comparison with experimental images'''
from scipy.ndimage import maximum_filter
with h5py.File("d:/simulationImages/real images/10_tweezers_mean_images.h5", 'r') as f:
    freeSpaceIllumTimes = np.array(f['Free space illumination times'])
    firstImageInTrap = np.array(f['first image in trap (6us ill time) average'])
    secondImageFreeSpace = np.array(f['second image free space average'])
    secondImageFreeSpace_Notes = dict(f['second image free space average'].attrs)

roiFromSimulations = np.mean(atomPixels, axis=0)
roiFromData = getAverageRoi(secondImageFreeSpace[0], roi)
difference = roiFromData - roiFromSimulations

print(f"average photons in simulation roi: {np.sum(roiFromSimulations)}")
print(f"average photons in experiment roi: {np.sum(roiFromData)}")

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
vmin = min(roiFromSimulations.min(), roiFromData.min(), difference.min())
vmax = max(roiFromSimulations.max(), roiFromData.max(), difference.max())

im0 = axes[0].imshow(roiFromSimulations, vmin=vmin, vmax=vmax)
axes[0].set_title("ROI from Simulations")
im1 = axes[1].imshow(roiFromData, vmin=vmin, vmax=vmax)
axes[1].set_title("ROI from Data")
im2 = axes[2].imshow(difference, vmin=vmin, vmax=vmax)
axes[2].set_title("Difference (Data - Simulations)")

fig.colorbar(im0, ax=axes, orientation='vertical', fraction=0.02, pad=0.04)
plt.show()

plt.plot(*getAverageCrossSection(firstImageInTrap), label="averaged data")
plt.plot(*get_xyFromSignal(average[average.shape[0]//2,:]), label="simulation")
plt.legend()
plt.show()

# for i in range(len(secondImageFreeSpace)):
#     Camera.save_h5_image(f"D:/simulationImages/real images/2ndImagefreeSpace_{freeSpaceIllumTimes[i]:.1e}.h5", secondImageFreeSpace[i], illumTime = freeSpaceIllumTimes[i], **secondImageFreeSpace_Notes)
