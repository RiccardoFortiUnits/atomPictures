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
import matplotlib.patches as patches

from scipy.ndimage import maximum_filter


'''----------------------------------------general values----------------------------------------'''
comparisonRoi = 7 
pixelSize = 4.6e-6 / 8

coordinates = []#[exp_x,exp_y,exp_z]]
names = []#"experiment"]
valuesForEllpiticFit = []
nOfImages = None

def gaussian(x, A, sigma, offset):
	return A * np.exp(-2 * ((x) / sigma) ** 2) + offset
def gaussian2D(xy, A,x0,y0, sigma_x, sigma_y, offset, angle = 0):
	x,y=xy[0]*np.cos(angle) + xy[1]*np.sin(angle), xy[1]*np.cos(angle) - xy[0]*np.sin(angle)

	return A * gaussian(x-x0,1,sigma_x,0) * gaussian(y-y0,1,sigma_y,0) + offset
def getFittedGaussian(x,y):
	p0 = [np.max(z), x[np.argmax(y.flatten())], 1, np.min(y)]
	params, cov = curve_fit(gaussian, r*8, az, p0=p0)
	return params
def getFittedGaussian2D(x,y,z):
	center = np.unravel_index(np.argmax(z.flatten()), x.shape)
	p0 = [np.max(z), x[*center], y[*center], (x.max()-x.min())/8, (y.max()-y.min())/8, np.min(z),0]
	params, cov = curve_fit(gaussian2D, np.row_stack((x.flatten(),y.flatten())), z.flatten(), p0=p0)
	return params
	
	
'''----------------------------------------experimental data----------------------------------------'''

exp_files = ["D:/simulationImages/real images/171_10Tweezer_inTrap_12us_2Images"]
internalPath = [f"images/Orca/fluorescence {i}/frame" for i in [1,2]]
for exp_file in exp_files:
	exp_cai = Camera.doubleCameraAtomImage(exp_file, exp_file, *internalPath, maxPictures=nOfImages)

	exp_cai.calcTweezerPositions(tweezerMinPixelDistance = 10, atomPeakMinValue = 1.8)

	sti = exp_cai.getSurelyTrappedAtoms(photonThreshold = 8, roi = 3)
	exp_z = exp_cai.first.getAtomROI_fromBooleanArray(comparisonRoi, sti, averageOnAtoms=True)
	exp_x,exp_y = exp_cai.first.getAtomCoordinates(comparisonRoi, pixelSize)
	A,x0,y0,sigma_x,sigma_y,offset,angle = getFittedGaussian2D(exp_x,exp_y,exp_z)
	if sigma_x > sigma_y:
		sigma_x,sigma_y = sigma_y,sigma_x
		angle += np.pi/2
	valuesForEllpiticFit.append([sigma_x,sigma_y, angle])
	print(f"exp_file: {exp_file}, A = {A}, x0 = {x0}, y0 = {y0}, sigma_x = {sigma_x}, sigma_y = {sigma_y}, offset = {offset}, angle = {angle}")
	exp_x = exp_x - x0
	exp_y = exp_y - y0
	coordinates.append([exp_x,exp_y,exp_z])
	names.append(exp_file.split("/")[-1])

'''----------------------------------------simulation data----------------------------------------'''
'''
sim_files = [
  "D:/simulationImages/correctScattering_Yt171_12us_10tweezerArray/12us_pictures",
  "D:/simulationImages/correctScattering_Yt171_12us_10tweezerArray/5us_pictures",
  "D:/simulationImages/correctScattering_Yt171_12us_10tweezerArray_z_lattice/pictures",
  "D:/simulationImages/correctScattering_Yt171_12us_10tweezerArray_atomUnmovable/pictures",
]
names += ["12us", "5us", "12us_z-lattice", "12us_fixedAtom"]
'''
sim_files = [
	# "D:/simulationImages/magnified_Yt171_20us_10tweezerArray/correctMagnification_outOfFocus0.0e+00m_7us_pictures",
	# "D:/simulationImages/magnified_Yt171_20us_10tweezerArray/correctMagnification_outOfFocus0.0e+00m_12us_pictures",
	# "D:/simulationImages/magnified_Yt171_20us_10tweezerArray/correctMagnification_outOfFocus0.0e+00m_20us_pictures",
	# "D:/simulationImages/magnified_Yt171_20us_10tweezerArray_atomUnmovable/correctMagnification_outOfFocus0.0e+00m_12us_pictures",
	# "D:/simulationImages/magnified_Yt171_20us_10tweezerArray/correctMagnification_outOfFocus0.0e+00m_12us_pictures_10reduction",
	# "D:/simulationImages/magnified_Yt171_20us_10tweezerArray_z_lattice/correctMagnification_outOfFocus0.0e+00m_12us_pictures_10reduction",
	"D:/simulationImages/magnified_Yt171_20us_10tweezerArray_atomUnmovable/correctMagnification_outOfFocus0.0e+00m_12us_pictures_10reduction",

	# "D:/simulationImages/magnified_Yt171_20us_10tweezerArray/correctMagnification_outOfFocus0.0e+00m_7us_pictures",
	# "D:/simulationImages/magnified_Yt171_20us_10tweezerArray/correctMagnification_outOfFocus0.0e+00m_12us_pictures_10reduction",
#   "D:/simulationImages/correctScattering_Yt171_12us_10tweezerArray/12us_pictures",
#   "D:/simulationImages/correctScattering_Yt171_12us_10tweezerArray/outOfFocus1.0e-06m_12us_pictures",
#   "D:/simulationImages/correctScattering_Yt171_12us_10tweezerArray/outOfFocus2.0e-06m_12us_pictures",
#   "D:/simulationImages/correctScattering_Yt171_12us_10tweezerArray/outOfFocus3.0e-06m_12us_pictures",
#   "D:/simulationImages/correctScattering_Yt171_12us_10tweezerArray/outOfFocus4.0e-06m_12us_pictures",
]
names += ["std simulation", "offset 2um", "offset 4um"]#,"simulation_smallerPixel", "simul. un-movable atom_smallerPixel"]           #[f"simulation" for i in range(len(sim_files))]#[f"offset {i}um" for i in [4]]#[0,4,1,2,3]]
#'''
for sim_file in sim_files:
	sim_cai = Camera.cameraAtomImages(sim_file, maxPictures=nOfImages)
	metadata = list(sim_cai.metadata.values())[0]
	pixelScale = int(pixelSize * 8 / metadata["camera_pixelSize"])
	expectedPeakValue = metadata["number of photons hitting grid 0"] * 2.8 / 380
	comparisonRoi = int(comparisonRoi * pixelScale)
	sim_cai.calcTweezerPositions(tweezerMinPixelDistance = 10 * pixelScale, atomPeakMinValue = expectedPeakValue / pixelScale)
	sim_z = sim_cai.getAtomROI(comparisonRoi, averageOnAtoms=True)
	sim_x,sim_y = sim_cai.getAtomCoordinates(comparisonRoi, pixelSize / pixelScale)
	A,x0,y0,sigma_x,sigma_y,offset, angle = getFittedGaussian2D(sim_x,sim_y,sim_z)
	if sigma_x > sigma_y:
		sigma_x,sigma_y = sigma_y,sigma_x
		angle += np.pi/2
	valuesForEllpiticFit.append([sigma_x,sigma_y, angle])
	print(f"sim_file: {sim_file}, A = {A}, x0 = {x0}, y0 = {y0}, sigma_x = {sigma_x}, sigma_y = {sigma_y}, offset = {offset}, angle = {angle}")
	
	sim_x = sim_x - x0
	sim_y = sim_y - y0
	
	coordinates.append([sim_x,sim_y,sim_z])
	comparisonRoi = int(np.round(comparisonRoi / pixelScale))
	# names.append(sim_file.split("/")[-1])

	# imageSize = metadata["camera_pixelSize"] * np.array( metadata["camera_pixelNumber"])
	# obtainedTweezerPositions = ((sim_cai.tweezerPositions / np.array([sim_cai.images.shape[1],sim_cai.images.shape[2]])[:,None])-.5)*imageSize[:,None]
	# actualTweezerPositions = metadata["tweezer_centers"][:,:2]
	# actualTweezerPositions = np.column_stack((actualTweezerPositions[:,1], -actualTweezerPositions[:,0])).T
	# errors = np.linalg.norm((actualTweezerPositions - obtainedTweezerPositions), axis=0)
	# print(f"{sim_file}: average error in the positions of the tweezers: {np.mean(errors):.2e} m ({np.mean(errors)/pixelSize:.3f} pixels)")
	# print(f"{sim_file}: maximum error in the positions of the tweezers: {np.max(errors):.2e} m ({np.max(errors)/pixelSize:.3f} pixels)")
	
'''
for i in range(len(sim_files)):
	coordinates[i][2] = coordinates[i][2] / np.max(coordinates[i][2])
'''
'''----------------------------------------azimutal average----------------------------------------'''
# plt.figure("Elliptic Fits")
# plt.scatter(0,0,c="red")
# ax = plt.gca()
# for i, (sigma_x, sigma_y, angle, x0,y0) in enumerate(valuesForEllpiticFit):
# 	ellipse = patches.Ellipse(
# 		(0,0), 
# 		width=2*sigma_x, 
# 		height=2*sigma_y, 
# 		angle=np.degrees(-angle), 
# 		fill=False, edgecolor='C{}'.format(i), label=names[i] if i < len(names) else None
# 	)
# 	plt.gca().add_patch(ellipse)

# idealPsf = patches.Circle((0,0), radius = 270e-9, fill=False, label="ideal PSF", color="black", linestyle='--')
# plt.gca().add_patch(idealPsf)
# # ax.set_aspect('equal')
# plt.xlabel("x")
# plt.ylabel("y")
# plt.legend()
# plt.title("Elliptic Gaussian Fits")
# plt.grid()
# plt.show()

# plt.scatter(exp_x, exp_y, c=exp_z, cmap='viridis')
# plt.scatter(0,0,c="red")
# for i in np.linspace(.2e-6,2e-6,10):
# 	circle = plt.Circle((0, 0), radius=i, color='orange', fill=False, linestyle='--')
# 	plt.gca().add_patch(circle)
# plt.legend()


'''----------------------------------------azimutal average----------------------------------------'''

plt.figure("azimutal average")
# names = ["5us", "12us", "12us_real"]
for i, (coords, name, valForEllipse) in enumerate(zip(coordinates, names, valuesForEllpiticFit)):
	if False and i < len(exp_files):
		r,az = Camera.cameraAtomImages.elliptical_average(*coords,valForEllipse[2], valForEllipse[0]/valForEllipse[1], (0,0),dividestandardBin=2)
	else:
		r,az = Camera.cameraAtomImages.azimuthal_average(*coords,(0,0),dividestandardBin=2)
	# az /= np.max(az)
	# Initial guess: amplitude, mean, stddev, offset
	p0 = [np.max(az), pixelSize, np.min(az)]
	params, cov = curve_fit(gaussian, r*8, az, p0=p0)
	print(params)
	# az = (az-params[-1]) / params[0] + params[-1]
	# params[0] = 1
	line, = plt.plot(r*8, az, label=name)
	plt.plot(r*8, gaussian(r*8, *params), label=f"{name}: gaussian fit: {params[1]*1e9/8:.1f}nm", linestyle=':', color=line.get_color())
# blur_z0,metadata = Camera.load_h5_image("d:/simulationImages/blurs/399nm/camera_atomZ=0.00e+00.h5", returnMetadata=True)
# blur_z0 = np.abs(blur_z0)**2
# blur_z0 = blur_z0[blur_z0.shape[0]//2,:]/np.sum(blur_z0)
# blur_z0 = blur_z0 / np.max(blur_z0) * 35 * 35/23
# blurSize = metadata["range"]
# x = np.linspace(-blurSize/2, (blurSize*(blur_z0.shape[0]-1)/blur_z0.shape[0])-blurSize/2, blur_z0.shape[0])
# pixel_response = np.where(np.abs(x) <= pixelSize*8/2, 1, 0)
# plt.plot(x, np.where(np.logical_and(x>=-2e-10,x <= pixelSize*8), 1, 0) * np.max(blur_z0), label='pixel')
# pixel_response = pixel_response / np.sum(pixel_response)  # Normalize
# blur_z0_pixel = np.convolve(blur_z0, pixel_response, mode='same')
# plt.plot(x,blur_z0, label = "blur at z=0 (normalized)")
# plt.plot(x,blur_z0_pixel, label = "blur convolved with pixel")
plt.xlabel("radius (m), camera space")
plt.ylabel("intensity")
plt.xlim(left=0)
plt.grid()
plt.legend()
plt.title("comparison with different offsets")
plt.show()



exp_fit = Camera.cameraAtomImages.findCenterOfGaussianSetOfImages(coords[0], coords[1], coords[2])


'''----------------------------------------3D intensity comparison----------------------------------------'''

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
for coords, name in zip(coordinates[0:1], names[0:1]):
	x,y,z=Camera.filterScatterToGrid(coords[0].flatten(), coords[1].flatten(), coords[2].flatten(),[comparisonRoi,comparisonRoi])
	ax.plot_trisurf(coords[0].flatten(), coords[1].flatten(), coords[2].flatten(), label=name, alpha=0.7)
	# ax.scatter(x,y,z, label = name)

plt.legend()
plt.title("3D intensity comparison")
plt.show()


'''----------------------------------------average intensity comparison----------------------------------------'''
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
plt.suptitle("Intensity Comparison (sub-pixel)", fontsize=16)  # Add an overall title
xx, yy, zz = [[], [], []]
for coords, name in zip(coordinates, names):	
	x, y, z = Camera.filterScatterToGrid(coords[0].flatten(), coords[1].flatten(), coords[2].flatten(),[comparisonRoi,comparisonRoi])
	xx.append(x)
	yy.append(y)
	zz.append(z)
difference = zz[1] - zz[0]
vmin = min(zz[0].min(), zz[1].min(), difference.min())
vmax = max(zz[0].max(), zz[1].max(), difference.max())

for i in [0, 1]:
	axes[i].imshow(zz[i], vmin=vmin, vmax=vmax)
	axes[i].set_title(names[i])
im2 = axes[2].imshow(difference, vmin=vmin, vmax=vmax)
axes[2].set_title(f"Difference")

fig.colorbar(im2, ax=axes, orientation='vertical', fraction=0.02, pad=0.04)
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

# '''histogram of number of photons in roy'''

# roi = 3
# metadata = list(metadata.values())[0]
# atomPosition = metadata["atoms_InitialCoordinates"][0]
# atomPosition[0] *=-1
# pixelSize = metadata["camera_pixelSize"]
# pixelNumber = imgs.shape[1]
# halfImageSize = pixelNumber*pixelSize/2
# atomPixels = np.array([(atomPosition[0]+halfImageSize) / pixelSize,
# 			  (atomPosition[1]+halfImageSize) / pixelSize]).astype(int) - roi//2
# atomPixels = imgs[:,atomPixels[0]:atomPixels[0]+roi,atomPixels[1]:atomPixels[1]+roi]
# photonsInRoi = np.sum(atomPixels, axis=(1,2))

# plt.hist(photonsInRoi, bins=int(np.max(photonsInRoi)-np.min(photonsInRoi)+1), label="photons in ROI")

# unique, counts = np.unique(photonsInRoi, return_counts=True)
# params, _ = curve_fit(poisson.pmf, unique, counts / len(photonsInRoi), p0=[np.mean(photonsInRoi)])
# plt.plot(unique, len(photonsInRoi) * poisson.pmf(unique, *params), label=f'Poisson fit (mean = {params[0]})')
# plt.legend()
# plt.show()

# '''comparison with experimental images'''
# with h5py.File("d:/simulationImages/real images/10_tweezers_mean_images.h5", 'r') as f:
# 	freeSpaceIllumTimes = np.array(f['Free space illumination times'])
# 	firstImageInTrap = np.array(f['first image in trap (6us ill time) average'])
# 	secondImageFreeSpace = np.array(f['second image free space average'])
# 	secondImageFreeSpace_Notes = dict(f['second image free space average'].attrs)

# roiFromSimulations = np.mean(atomPixels, axis=0)
# roiFromData = getAverageRoi(secondImageFreeSpace[0], roi)
# difference = roiFromData - roiFromSimulations

# print(f"average photons in simulation roi: {np.sum(roiFromSimulations)}")
# print(f"average photons in experiment roi: {np.sum(roiFromData)}")

# fig, axes = plt.subplots(1, 3, figsize=(15, 5))
# vmin = min(roiFromSimulations.min(), roiFromData.min(), difference.min())
# vmax = max(roiFromSimulations.max(), roiFromData.max(), difference.max())

# im0 = axes[0].imshow(roiFromSimulations, vmin=vmin, vmax=vmax)
# axes[0].set_title("ROI from Simulations")
# im1 = axes[1].imshow(roiFromData, vmin=vmin, vmax=vmax)
# axes[1].set_title("ROI from Data")
# im2 = axes[2].imshow(difference, vmin=vmin, vmax=vmax)
# axes[2].set_title("Difference (Data - Simulations)")

# fig.colorbar(im0, ax=axes, orientation='vertical', fraction=0.02, pad=0.04)
# plt.show()

# plt.plot(*getAverageCrossSection(firstImageInTrap), label="averaged data")
# plt.plot(*get_xyFromSignal(average[average.shape[0]//2,:]), label="simulation")
# plt.legend()
# plt.show()

# # for i in range(len(secondImageFreeSpace)):
# #     Camera.save_h5_image(f"D:/simulationImages/real images/2ndImagefreeSpace_{freeSpaceIllumTimes[i]:.1e}.h5", secondImageFreeSpace[i], illumTime = freeSpaceIllumTimes[i], **secondImageFreeSpace_Notes)
