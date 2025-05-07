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
import scipy.optimize as opt
import ArQuS_analysis_lib as Ar


'''----------------------------------------general values----------------------------------------'''
comparisonRoi = 7
'''----------------------------------------experimental data----------------------------------------'''
exp_file = "D:/simulationImages/real images/171_10Tweezer_inTrap_12us_2Images"
internalPath = [f"images/Orca/fluorescence {i}/frame" for i in [1,2]]
exp_cai = Camera.doubleCameraAtomImage(exp_file, exp_file, *internalPath)
exp_cai.calcTweezerPositions(tweezerMinPixelDistance = 10, atomPeakMinValue = 1.8)

sti = exp_cai.getSurelyTrappedAtoms(photonThreshold = 8, roi = 3)
exp_z = exp_cai.first.getAtomROI_fromBooleanArray(comparisonRoi, sti, averageOnAtoms=True)
exp_x,exp_y = np.meshgrid(*[np.arange(0,comparisonRoi) for _ in range(2)], indexing="ij")
exp_x = exp_x[None,:,:] - comparisonRoi//2 - (exp_cai.first.tweezerPositions - exp_cai.first.tweezerPixels)[0][:,None,None]
exp_y = exp_y[None,:,:] - comparisonRoi//2 - (exp_cai.first.tweezerPositions - exp_cai.first.tweezerPixels)[1][:,None,None]


'''----------------------------------------simulation data----------------------------------------'''
sim_file = "D:/simulationImages/Yt171_12us/pictures"
sim_cai = Camera.cameraAtomImages(sim_file)
sim_cai.calcTweezerPositions(tweezerMinPixelDistance = 10, atomPeakMinValue = 1.8)
sim_z = sim_cai.getAtomROI(comparisonRoi)
sim_x,sim_y = np.meshgrid(*[np.arange(0,comparisonRoi) for _ in range(2)], indexing="ij")
tweezerCenters = np.array([el["tweezer_centers"][0][:2] for el in sim_cai.metadata.values()])
pixelSize = list(sim_cai.metadata.values())[0]["camera_pixelSize"]
tweezerCenters /= pixelSize
tweezerCenters[:,0] *= -1#for how the simulation works now, the x axis gets flipped when acquiring the image, so I put a negative sign in the position of the tweezer
sim_x = sim_x[None,:,:] - comparisonRoi//2 - tweezerCenters[:,0][:,None,None]
sim_y = sim_y[None,:,:] - comparisonRoi//2 - tweezerCenters[:,1][:,None,None]



coordinates = [[sim_x,sim_y,sim_z], [exp_x,exp_y,exp_z]]
names = ["simulation", "experiment"]

'''----------------------------------------azimutal average----------------------------------------'''

plt.figure("azimutal average")
for coords, name in zip(coordinates, names):
    r,az = Camera.cameraAtomImages.azimuthal_average(*coords,(0,0),30)
    plt.plot(r,az, label = name)
plt.legend()
plt.show()

'''----------------------------------------3D gaussian fit----------------------------------------'''

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
for coords, name in zip(coordinates, names):
    ax.scatter(*coords, label = name)

# xy = np.vstack((sim_x.flatten(),sim_y.flatten()))
# x_grid = np.linspace(np.min(sim_x),np.max(sim_x),100)
# y_grid = np.linspace(np.min(sim_y),np.max(sim_y),100)
# xv, yv = np.meshgrid(x_grid,y_grid)
# initial_guess = np.asarray([np.max(sim_z), 0, 0, 1, 1,np.min(sim_z)])
# p_opt,p_cov = opt.curve_fit(Ar.Gaussian_2D, xy, sim_z.flatten(), p0 = initial_guess)
# fitted_gaussian = Ar.Gaussian_2D((xv,yv),*p_opt).reshape(100,100)
# ax.contourf(xv,yv,fitted_gaussian,100,alpha=0.5,cmap='plasma')

plt.legend()
plt.show()


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



'''
exp_cai = Camera.doubleCameraAtomImage("D:/simulationImages/real images/smallerSet - 171_10Tweezer_inTrap_7us_2Images", "D:/simulationImages/real images/smallerSet - 171_10Tweezer_inTrap_7us_2Images", "images/Orca/fluorescence 1/frame", "images/Orca/fluorescence 2/frame")
exp_cai.calcTweezerPositions()
imgs = exp_cai.first.getAtomROI_fromBooleanArray(7, exp_cai.getSurelyTrappedAtoms(8, 3))
'''
imgs, metadata = Camera.getImagesFrom_h5_files("D:/simulationImages/Yt171_7us/pictures/")
imgs = np.array(imgs)
#'''

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
    x=np.linspace(x[0],x[-1],len(x)*10)
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

plt.hist(photonsInRoi, bins=int(np.max(photonsInRoi)-np.min(photonsInRoi)+1), label="photons in ROI")

unique, counts = np.unique(photonsInRoi, return_counts=True)
params, _ = curve_fit(poisson.pmf, unique, counts / len(photonsInRoi), p0=[np.mean(photonsInRoi)])
plt.plot(unique, len(photonsInRoi) * poisson.pmf(unique, *params), label=f'Poisson fit (mean = {params[0]})')
plt.legend()
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
