
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import j1, j0
from scipy.interpolate import RegularGridInterpolator
from scipy.interpolate import interp1d, griddata
from typing import Tuple, List, Union
import h5py
import os
from scipy import integrate
from functools import partial
import inspect
from scipy.optimize import curve_fit
from scipy.ndimage import maximum_filter, gaussian_filter,uniform_filter
from mpl_toolkits.mplot3d import Axes3D
from scipy import constants, signal
import scipy.optimize as opt
import ArQuS_analysis_lib as Ar
from scipy.ndimage import convolve

def plot2D(Z, x_range, y_range, figureTitle=""):
	if Z.dtype == np.complex128:
		plt.figure(figureTitle + " (abs value)")
		plt.imshow((np.abs(Z.T)), extent=(y_range[0], y_range[1], x_range[0], x_range[1]), origin='lower', cmap='viridis', aspect='auto')
		plt.colorbar()
		plt.show()
		Z=np.angle(Z)
		figureTitle = f"{figureTitle} (angle)"
		
	plt.figure(figureTitle)
	plt.imshow(Z.T, extent=(y_range[0], y_range[1], x_range[0], x_range[1]), origin='lower', cmap='viridis', aspect='auto')
	plt.show()
def plot2D_function(function, x_range, y_range, resolution_x, resolution_y, figureTitle=""):
	X,Y=np.meshgrid(np.linspace(x_range[0],x_range[1],resolution_x),np.linspace(y_range[0],y_range[1],resolution_y))
	signature = inspect.signature(function)
	if len(signature.parameters) == 1:
		Z=function(np.stack((X,Y), axis= -1))
	else:
		Z=function(X,Y)
	plot2D(Z, x_range, y_range, figureTitle)

def plot1D_function(function, range, resolution, figureTitle=""):
	x = np.linspace(range[0],range[1],resolution)
	y=function(x)
	if y.dtype == np.complex128:
		plt.figure(figureTitle + " (abs value)")
		plt.plot(x,np.abs(y))
		plt.show()
		y=np.angle(y)
		figureTitle = f"{figureTitle} (angle)"
		
	plt.figure(figureTitle)
	plt.plot(x,y)
	plt.show()
def ainy(u, v, r=1):
	# Calculate the 2D Fourier transform
	rho = np.sqrt(u**2 + v**2)
	result = 2 * np.pi * r**2 * j1(2 * np.pi * r * rho) / (2 * np.pi * r * rho +.1)
	return np.abs(result)

def blur(r,z, k,E0,f,w,R):
	s = np.shape(r)
	r=r.flatten()
	z=z.flatten()
	def functionToIntegrate(x,r,z):
		return x*j0(k*r*x/f)*np.exp(
			-(x/w)**2 - 0.5j*k*z*(x/f)**2
		)
	usedR = min(R, 4*w)
	x = np.repeat(np.linspace(0,usedR,1000)[None,:],len(r), axis=0)
	integral = np.sum(functionToIntegrate(x, r[:,None],z[:,None]), axis=1) * usedR/x.shape[1]
	retVal = np.abs(E0 * k / (2 * np.pi * f) * np.exp(-1j * k * z) * integral)**2
	return retVal.reshape(s)

def CoordinateChange_extended(data_tuple, x0, y0, z0, angleXY, angleXZ, angleYZ):
	(x, y, z) = data_tuple
	x, y, z = x - x0, y - y0, z - z0
	
	# Rotation around the XY plane
	y_Prime = y * np.cos(angleXY) + x * np.sin(angleXY)
	x_temp = x * np.cos(angleXY) - y * np.sin(angleXY)
	
	# Rotation around the XZ plane
	x_Prime = x_temp * np.cos(angleXZ) - z * np.sin(angleXZ)
	z_temp = x_temp * np.sin(angleXZ) + z * np.cos(angleXZ)
	
	
	# Rotation around the YZ plane
	z_Prime = y_Prime * np.sin(angleYZ) + z_temp * np.cos(angleYZ)
	y_Prime = y_Prime * np.cos(angleYZ) - z_temp * np.sin(angleYZ)
	
	return x_Prime, y_Prime, z_Prime
def reverseCoordinateChange(data_tuple, x0, y0, z0, angleXY, angleXZ, angleYZ):
	#reverts the coordinate change done by CoordinateChange_extended
	(x, y, z) = data_tuple
	angleXY = - angleXY
	angleXZ = - angleXZ
	angleYZ = - angleYZ
 
	# Rotation around the YZ plane
	z_temp = y * np.sin(angleYZ) + z * np.cos(angleYZ)
	y_temp = y * np.cos(angleYZ) - z * np.sin(angleYZ)

	# Rotation around the XZ plane
	x_temp = x * np.cos(angleXZ) - z_temp * np.sin(angleXZ)
	z_Prime = x * np.sin(angleXZ) + z_temp * np.cos(angleXZ)

	# Rotation around the XY plane
	y_Prime = x_temp * np.sin(angleXY) + y_temp * np.cos(angleXY)
	x_Prime = x_temp * np.cos(angleXY) - y_temp * np.sin(angleXY)

	return x_Prime + x0, y_Prime + y0, z_Prime + z0
def fft2_butBetter(image, imageSize, transformSize, transform_n = None, paddingValue = 0):
	n = np.array(image.shape)
	if transform_n is None:
		transform_n = n
	L=imageSize
	M=transformSize
	m=transform_n
	P=m/M
	usedPixels = L/P * m
	decimation = n / usedPixels
	padding = np.ceil(decimation).astype(int)
	image = np.pad(image, ((padding[0], padding[0]), (padding[1], padding[1])), mode='constant', constant_values=paddingValue)
	
	filteredImage = uniform_filter(image, decimation)
	imageSize = imageSize * np.array(image.shape) / n
	n = np.array(image.shape)
	def f(x,y):
		interpolator = RegularGridInterpolator(
			(np.linspace(0,imageSize[i],n[i]) for i in [0,1]), filteredImage, bounds_error=False, fill_value=paddingValue
		)
		points = np.stack((x, y), axis = -1)
		im = interpolator(points)
		return im
	x,y=np.meshgrid(*[np.linspace(0,P[i],transform_n[i]) for i in [0,1]], indexing='ij')
	CorrectSizedImage = f(x,y)
	return np.fft.fft2(CorrectSizedImage)
		
def mirrorSymmetricImage(image, mirrorFirstLines = False):
	'''
	given an image f(x>=0,y>=0), it mirrors it along the x and y axes.
	if mirrorFirstLines is false, the first row and column are treated as the values of f(0,y) and f(x,0), and thus they won't be mirrored.
	'''
	toFlip = image if mirrorFirstLines else image[:,1:]
	image = np.concatenate((np.flip(toFlip,axis=1), image), axis=1)
	toFlip = image if mirrorFirstLines else image[1:,:]
	image = np.concatenate((np.flip(toFlip,axis=0), image), axis=0)
	return image

def filterScatterToGrid(x,y,z,gridPoints):
	mins = [np.min(x), np.min(y)]
	maxs = [np.max(x), np.max(y)]
	X,Y = np.meshgrid(*[np.linspace(mins[i], maxs[i], gridPoints[i]) for i in [0,1]], indexing='ij')
	Z=np.zeros(X.size)
	rowIndexes = ((x - mins[0]) / (maxs[0] - mins[0]) * gridPoints[0]).astype(int)
	columnIndexes = ((y - mins[1]) / (maxs[1] - mins[1]) * gridPoints[1]).astype(int)
	rowIndexes[rowIndexes == gridPoints[0]] -= 1
	columnIndexes[columnIndexes == gridPoints[1]] -= 1
	pointIndex = rowIndexes * X.shape[1] + columnIndexes
	np.add.at(Z, pointIndex, z)
	unique, count = np.unique(pointIndex, axis=0, return_counts=True)
	Z[unique] /= count
	Z=Z.reshape(X.shape)
	return X,Y,Z



def save_h5_image(imageName, image, **metadata):	
	with h5py.File(imageName, 'w') as f:
		f.create_dataset('image', data=image)
		f.attrs.update(metadata)

def load_h5_image(path, internalPath = None, returnMetadata = False):
	'''
		gets an image from the specified path
		also returns a dictionary {fileName:metadata} of all the metadata contained in the files
	'''
	with h5py.File(path, 'r') as f:		
		if internalPath is None:
			image = np.array(f['image'])
		else:
			image = np.array(f[internalPath])
		if returnMetadata:
			metadata = dict(f.attrs)
			return image, metadata

	return image
def getImagesFrom_h5_files(folderPath, internalPath = None, maxPictures = None):
	'''
		gets all the images contained in folderPath.
		also returns a dictionary {fileName:metadata} of all the metadata contained in the files
	'''
	files = [f for f in os.listdir(folderPath) if f.endswith('.h5')]
	if not files:
		raise ValueError("No h5 files found in the specified folder.")
	if maxPictures is not None:
		files = files[:maxPictures]
	if len(files) > 400:
		print(f"reading all the images from path {folderPath}, it may take a while")
	images = []
	metadata = {}

	for file in files:
		with h5py.File(os.path.join(folderPath, file), 'r') as f:
			if internalPath is None:
				image = np.array(f['image'])
			else:
				image = np.array(f[internalPath])
			images.append(image)
			metadata[file] = dict(f.attrs)

	return images, metadata

def getIndexesAndFractionalPosition(values, grid):
	'''
	find the values i, p so that
	for each j, values[j] =  grid[i[j]] + p[j] * (grid[i[j]+1] - grid[i[j]]),
	'''
	i = np.searchsorted(grid, values, side='right') - 1
	i[i >= len(grid) -1] = len(grid) - 2
	x0 = grid[i]
	x1 = grid[i + 1]
	p = (values - x0) / (x1 - x0)
	return i, p
'''---------------------------------------------------------------------------------'''

class cameraAtomImages:
	@staticmethod
	def Gauss(x, A, B, C):
		y = A*np.exp(-1*B*(x-C)**2)
		return y
	@staticmethod
	def Gauss2D(xy, A, B, Cx, Cy):
		xy = np.stack((xy[...,0] - Cx, xy[...,1] - Cy), axis = -1)
		r = np.linalg.norm(xy, axis=-1)
		return cameraAtomImages.Gauss(r,A,B,0).flatten()
	@staticmethod
	def findCenterOfGaussianSignal(signal):
		parameters, _ = curve_fit(cameraAtomImages.Gauss, np.arange(0,len(signal)), signal, p0=[np.max(signal), 1, np.argmax(np.abs(signal))])
		return parameters[2]
	@staticmethod
	def findCenterOfGaussianImage(image):
		x,y = np.meshgrid(*[np.arange(0,image.shape[i]) for i in range(2)], indexing="ij")
		center = np.unravel_index(np.argmax(np.abs(image)), image.shape)
		parameters, _ = curve_fit(Ar.Gaussian_2D, np.vstack((x.flatten(),y.flatten())), image.flatten(), p0=[np.max(image), *center, 1,1, 0])
		# print(parameters[3], parameters[4])
		return parameters[1], parameters[2]
	@staticmethod
	def findCenterOfGaussianSetOfImages(x,y,z):
		center = np.unravel_index(np.argmax(np.abs(z)), z.shape)[-2:]
		parameters, _ = curve_fit(Ar.Gaussian_2D, np.vstack((x.flatten(),y.flatten())), z.flatten(), p0=[np.max(z), *center, 1,1, 0])
		return parameters[1], parameters[2]
	


	def __init__(self, folder, internalPath = None, maxPictures = None):
		self.folder = folder
		self.images, self.metadata = getImagesFrom_h5_files(folder, internalPath, maxPictures)
		self.images = np.array(self.images)

	def averageImage(self):
		average = np.mean(self.images, axis = 0)
		return average
	@staticmethod
	def getTweezerPositions(image, tweezerMinPixelDistance = 10, atomPeakMinValue = 1.8):
		maxes = maximum_filter(image, size=tweezerMinPixelDistance)
		peaks = (maxes == image).astype(int)
		#let's consider the possibility that 2 closeby pixels could have the same value (it happened to me, so it's not impossible).
		#let's give unique values to each peak, so that by re-applying the max filter we omit closeby peaks
		peaks *= np.arange(1,peaks.size+1).reshape(peaks.shape)
		maxes = maximum_filter(peaks, size=tweezerMinPixelDistance)
		peaks = maxes == peaks

		pixelCenters = np.array(np.where(np.logical_and(peaks, (image > atomPeakMinValue))))
		#let's find the actual center by fitting a gaussian on the roi
		centers = pixelCenters - np.repeat(tweezerMinPixelDistance//2.,2)[:,None]
		
		for i, center in enumerate(pixelCenters.T):
			corner = center - np.repeat(tweezerMinPixelDistance//2,2)
			# for axis in [0,1]:
			# 	section =	image[corner[0]:corner[0]+tweezerMinPixelDistance, center[1]] if axis == 0 else \
			# 				image[center[0], corner[1]:corner[1]+tweezerMinPixelDistance]
				
			# 	centers[axis,i] += cameraAtomImages.findCenterOfGaussianSignal(section)
			section = image[corner[0]:corner[0]+tweezerMinPixelDistance, corner[1]:corner[1]+tweezerMinPixelDistance]
			centers[:,i] += np.array(cameraAtomImages.findCenterOfGaussianImage(section))
		return centers

	def calcTweezerPositions(self, tweezerMinPixelDistance = 10, atomPeakMinValue = 1.8):
		image = self.averageImage()
		centers = self.getTweezerPositions(image, tweezerMinPixelDistance, atomPeakMinValue)
				
		self._tweezerPositions = centers
		self._tweezerPixels = np.round(centers).astype(int)

	@staticmethod
	def azimuthal_average(x,y,z, center = None, n_bins=None, dividestandardBin = 1):
		if center is None:
			idx = np.unravel_index(np.argmax(z), x.shape)
			center = x[idx],y[idx]
		x0,y0 = center
		if n_bins is None:
			n_bins = int((z.shape[1] + z.shape[2]) * dividestandardBin)

		r = np.hypot(x-x0,y-y0) 
		r_bins = np.linspace(r.min(),r.max(),n_bins+1)
		r_bins_center = 0.5*(r_bins[:-1]+r_bins[1:])

		z_avg = np.zeros(n_bins)
		binMap = ((r - r.min()) / (r_bins[1]-r_bins[0])).astype(int)
		binMap[binMap==n_bins] = n_bins-1
		np.add.at(z_avg, binMap, z)
		unique, count = np.unique(binMap, return_counts=True)
		z_avg[unique] /= count

		return r_bins_center, z_avg
	@staticmethod
	def elliptical_average(x,y,z, angle, axisRatio, center = None, n_bins=None, dividestandardBin = 1):
		'''
		angle:angle of the axis that will not be normalized (i.e,, which is already normalized)
		axisRatio = length of the normalized axis / length of the perpendicular axis
		'''
		if center is None:
			idx = np.unravel_index(np.argmax(z), x.shape)
			center = x[idx],y[idx]
		x0,y0 = center
		x,y=x-x0,y-y0
		x,y=x*np.cos(angle) + y*np.sin(-angle), y*np.cos(angle) - x*np.sin(-angle)
		y *= axisRatio
		z /= axisRatio
		return cameraAtomImages.azimuthal_average(x,y,z, center = (0,0), n_bins=n_bins, dividestandardBin=dividestandardBin)

	@property
	def tweezerPositions(self):
		if not hasattr(self, "_tweezerPositions"):
			self.calcTweezerPositions()
		return self._tweezerPositions
	@property
	def tweezerPixels(self):
		if not hasattr(self, "_tweezerPixels"):
			self.calcTweezerPositions()
		return self._tweezerPixels
	
	def getTrappedAtoms(self, photonThreshold, roi = 3):
		'''returns a nImg * nTweezers boolean array, where element [i,j] states if the trap j of image i is filled or not'''
		corners = self.tweezerPixels - np.repeat(roi//2,2)[:,None]
		trappedAtoms = np.zeros((len(self.images), len(corners.T)), dtype=bool)
		for i,corner in enumerate(corners.T):
			atomsPerRoi = np.sum(self.images[:,corner[0]:corner[0]+roi,corner[1]:corner[1]+roi], axis=(1,2))
			trappedAtoms[:,i] = atomsPerRoi >= photonThreshold
		return trappedAtoms
	
	def getAtomROI(self, roi, imageIdx = None, atomIdx = None, averageOnAtoms = False):
		return self.getAtomROIOnImages(self.images, self.tweezerPixels, roi, imageIdx, atomIdx, averageOnAtoms)
	def getAverageAtomROI(self, roi, atomIdx = None):
		return self.getAtomROIOnImages(self.averageImage()[None,:,:], self.tweezerPixels, roi, None, atomIdx, False)
	def getAtomROI_fromBooleanArray(self, roi, array, averageOnAtoms = False):
		imgs, atoms = np.where(array)
		return self.getAtomROIOnImages(self.images, self.tweezerPixels, roi, imgs, atoms, averageOnAtoms)
	def getAtomCoordinates(self, roi, pixelSize = 1, atomsIdx = None):
		'''
		returns the coordinates of the atoms in the specified roi. 
		if atomsIdx is not specified, all the valid indexes will be considered
		'''
		if atomsIdx is None:
			atomsIdx = np.arange(0, len(self.tweezerPixels.T))
		x,y = np.meshgrid(*[np.arange(0,roi) for _ in range(2)], indexing="ij")
		x = (x[None,:,:] - roi//2 - (self.tweezerPositions - self.tweezerPixels)[0][:,None,None]) * pixelSize
		y = (y[None,:,:] - roi//2 - (self.tweezerPositions - self.tweezerPixels)[1][:,None,None]) * pixelSize
		
		return x[atomsIdx,:,:],y[atomsIdx,:,:]
	@staticmethod
	def getAtomROIOnImages(images, tweezerPixels, roi, imageIdx = None, atomIdx = None, averageOnAtoms = False):
		'''
		returns the sub-images from the specified images containing the specified atoms. 

		if imageIdx and/or atomIdx are not specified, all the valid indexes will be considered, and imageIdx and atomIdx will be meshed together, to obtain all the combinations of index couples.
		
		if averageOnAtoms is true, all the images of the same atom will be averaged, resulting in an nOfAtoms long vector of images
		'''
		mesh = False
		if imageIdx is None:
			imageIdx = np.arange(0, len(images))
			mesh = True
		if atomIdx is None:
			atomIdx = np.arange(0, len(tweezerPixels.T))
			mesh = True
		if mesh:
			imageIdx, atomIdx = np.meshgrid(imageIdx, atomIdx, indexing='ij')
			imageIdx = imageIdx.flatten()
			atomIdx = atomIdx.flatten()
		corners = tweezerPixels - np.repeat(roi//2,2)[:,None]
		corners = corners[:, atomIdx]
		allImages = images[imageIdx[:,None,None],
					 np.round(np.linspace(corners[0],corners[0]+roi-1, roi)).T.astype(int)[:,:,None],
					 np.round(np.linspace(corners[1],corners[1]+roi-1, roi)).T.astype(int)[:,None,:]]
		if averageOnAtoms:
			averagedAtomImages = np.zeros((len(tweezerPixels.T), roi,roi))
			np.add.at(averagedAtomImages, atomIdx, allImages)
			unique, count = np.unique(atomIdx, return_counts=True)
			averagedAtomImages = averagedAtomImages[unique]
			averagedAtomImages /= count[:,None,None]
			return averagedAtomImages
		
		return allImages


	def getTweezerImage(self, roi, tweezerMinPixelDistance = 10, atomPeakMinValue = 1):
		'''returns a boolean image that is True where the roi of an atom should be'''
		img = np.zeros_like(self.images[0], dtype=bool)
		self.calcTweezerPositions(tweezerMinPixelDistance, atomPeakMinValue)
		corners = self.tweezerPixels - np.repeat(roi//2,2)[:,None]
		for corner in corners.T:
			img[corner[0]:corner[0]+roi, corner[1]:corner[1]+roi] = True
		return img

	

class doubleCameraAtomImage:
	def __init__(self,firstPath, secondPath, firstInternalPath = None, secondInternalPath = None, maxPictures = None):
		self.first = cameraAtomImages(firstPath, firstInternalPath, maxPictures)
		self.second = cameraAtomImages(secondPath, secondInternalPath, maxPictures)
	
	def calcTweezerPositions(self, tweezerMinPixelDistance = 10, atomPeakMinValue = 1.8):
		self.first.calcTweezerPositions(tweezerMinPixelDistance, atomPeakMinValue)
		self.second.calcTweezerPositions(tweezerMinPixelDistance, atomPeakMinValue)

	def getTrappedAtomBehaviour(self, photonThreshold, roi = 3):
		'''returns a nImg * nTweezers int array, where element [i,j] states the "filling transition" of the tweezer j of image i.
		For example, trappAtomBehav[i,j] = 2 = 0b10, => the tweezer was filled in the first image, and empty in the second'''
		a0 = self.first.getTrappedAtoms(photonThreshold, roi)
		a1 = self.second.getTrappedAtoms(photonThreshold, roi)
		return a0.astype(int) << 1 | a1.astype(int)
	
	def getSurelyTrappedAtoms(self, photonThreshold = 10, roi = 3):
		trappAtomBehav = self.getTrappedAtomBehaviour(photonThreshold, roi)
		return trappAtomBehav == 0b11
'''---------------------------------------------------------------------------------'''

class pixelGrid:
	def __init__(self, xsize,ysize,nofXpixels,nofYpixels, PSF, magnification = 1):
		#let's always work with the center of the grid being addressed as (0,0), so that it's easier to change from grid to grid
		#magnification is applied to the entering coordinates, so the sizes in x and y are already in the "magnified" dimension
		self.xsize = xsize
		self.ysize = ysize
		self.nofXpixels = nofXpixels
		self.nofYpixels = nofYpixels
		self.pixels = np.zeros((nofXpixels,nofYpixels))
		self.PSF = PSF
		self.magnification = magnification

	def _normalizeCoordinate(self,x,y, removeOutOfBoundaryValues = True):
		# Normalize the coordinates to the pixel grid
		x_normalized = np.round((x / self.xsize + .5) * (self.nofXpixels - 1)).astype(int)
		y_normalized = np.round((y / self.ysize + .5) * (self.nofYpixels - 1)).astype(int)
		
		if removeOutOfBoundaryValues:
			insideBoundaries = np.logical_and(
				np.logical_and(x_normalized >= 0, x_normalized <= self.nofXpixels - 1),
				np.logical_and(y_normalized >= 0, y_normalized <= self.nofYpixels - 1))
			x_normalized = x_normalized[insideBoundaries]
			y_normalized = y_normalized[insideBoundaries]
		else:
			# Ensure the indices are within the valid range
			x_normalized = np.clip(x_normalized, 0, self.nofXpixels - 1)
			y_normalized = np.clip(y_normalized, 0, self.nofYpixels - 1)
		return x_normalized, y_normalized
	@property
	def currentNumberOfPhotons(self):
		return np.sum(self.pixels)

	def get_pixel(self, x, y):
		x_normalized, y_normalized = self._normalizeCoordinate(x,y)
		return self.pixels[x_normalized, y_normalized]

	def __call__(self, x, y):
		return self.get_pixel(x, y)

	def set_pixel(self, x, y, value):
		x_normalized, y_normalized = self._normalizeCoordinate(x,y)
		
		self.pixels[x_normalized, y_normalized] = value
	def getRawPositions(self):
		#gives a list of the raw position of all the lit pixels.
		coordinates = np.argwhere(self.pixels > 0)
		counts = self.pixels[self.pixels > 0].astype(int)
		coordinates = np.repeat(coordinates, counts, axis=0).astype(float)
		coordinates[:,0] = (coordinates[:,0] / (self.nofXpixels - 1) - .5) * self.xsize
		coordinates[:,1] = (coordinates[:,1] / (self.nofYpixels - 1) - .5) * self.ysize
		return coordinates
	def fillFromLens(self, rawPhotonPositions):
		'''
		rawPhotonPositions: array of 2D positions (could also be a 3D position, if the PSF is also dependent on the z-coordinate)

		returns a dictionary with some extra info (though the default pixel grid only returns a non-empty dictionary if there's no photon to work on)
		'''
		magnification = np.array([self.magnification,self.magnification,1])
		rawPhotonPositions = rawPhotonPositions * magnification[None,:rawPhotonPositions.shape[1]]
		if len(rawPhotonPositions) == 0:
			return {"warning:" : "no photons hitting the lens"}
		if len(rawPhotonPositions[0]) == 3:
			psfPhotonPosition = self.PSF(rawPhotonPositions[:,:2], -rawPhotonPositions[:,2])# * transformedLensRadius / lensRadius
		else:
			psfPhotonPosition = self.PSF(rawPhotonPositions)# * transformedLensRadius / lensRadius
		x_normalized, y_normalized = self._normalizeCoordinate(psfPhotonPosition[:,0], psfPhotonPosition[:,1])
		np.add.at(self.pixels, (x_normalized, y_normalized), 1)
		return {}
	def fillFromOtherGrid(self, inputGrid : 'pixelGrid'):
		psfPositions = self.PSF(inputGrid.getRawPositions() * self.magnification)
		x_normalized, y_normalized = self._normalizeCoordinate(psfPositions[:,0], psfPositions[:,1])
		np.add.at(self.pixels, (x_normalized, y_normalized), 1)
	def clear(self):
		self.pixels = np.zeros((self.nofXpixels,self.nofYpixels))

class cMosGrid(pixelGrid):
	def __init__(self, xsize, ysize, nofXpixels, nofYpixels, PSF, noisePictureFilePath, imageStart = (0,0), imageSizes = None, magnification = 1):

		super().__init__(xsize, ysize, nofXpixels, nofYpixels, PSF, magnification)
		self.setRandomPixelNoises(noisePictureFilePath, imageStart, imageSizes)

	def setRandomPixelNoises(self, noisePictureFilePath, imageStart = (0,0), imageSizes = None):
		self.pixelNoises = cMosGrid.getRandomPixelNoises(self.pixels.size, noisePictureFilePath, imageStart, imageSizes)    \
								.reshape((self.pixels.shape[0],self.pixels.shape[1], -1))
	def fillFromLens(self, rawPhotonPositions):#given a photon at position (lensRadius,0), the expected pixel to be hit will be the one with coordinates (transformedLensRadius,0)
		extraData = super().fillFromLens(rawPhotonPositions)
		extraData["number of photons before added noise"] = self.currentNumberOfPhotons
		self.addNoise()
		return extraData

	def addNoise(self):
		self.pixels += self.pixelNoises[np.arange(self.pixels.shape[0])[:, None], np.arange(self.pixels.shape[1]), np.random.randint(self.pixelNoises.shape[2], size=self.pixels.shape)]

	@staticmethod	
	def getPictures(path, imageStart = (0,0), imageSizes = None, pictureLimit = None):
		files_to_analyse = os.listdir(path)
		if pictureLimit is not None:
			files_to_analyse = files_to_analyse[:pictureLimit]
		
		raw_images = None
		def find_image_dataset(group, prefix="images"):
				for key in group:
					if prefix == None or key == prefix:
						item = group[key]
						if isinstance(item, h5py.Group):
							# If it's a group, recurse into it
							result = find_image_dataset(item, None)
							if result is not None:
								return result
						elif isinstance(item, h5py.Dataset):
							# If it's a dataset and the path starts with "images/"
							return item.name
				# If not found at this level
				return None
		for i, file in enumerate(files_to_analyse):
			f = h5py.File(path+file, 'r')
			# Recursively find the dataset path that starts with "images/"
			img_path = find_image_dataset(f, prefix="images")
			if img_path is None:
				raise ValueError("No dataset found under 'images/' in the file.")
			img = np.asarray(f[img_path])
			if raw_images is None:
				if imageSizes is None:
					imageSizes = img.shape
				if imageSizes[0] < 0: imageSizes = (img.shape[0] + 1 + imageSizes[0] - imageStart[0], imageSizes[1])
				if imageSizes[1] < 0: imageSizes = (imageSizes[0], img.shape[1] + 1 + imageSizes[1] - imageStart[1])
				raw_images = np.zeros((len(files_to_analyse), imageSizes[0], imageSizes[1]), dtype=np.uint8)
			
			raw_images[i] = img[imageStart[0]:imageStart[0] + imageSizes[0], imageStart[1]:imageStart[1] + imageSizes[1]].astype(np.uint8)
		return raw_images#so we won't occupy too much memory (we'll still occupy a lot though)
	@staticmethod
	def getRandomPixelNoises(nOfPixels, path, imageStart = (0,0), imageSizes = None):
		images = cMosGrid.getPictures(path, imageStart, imageSizes)
		pixels = images.reshape((images.shape[0], -1)).T
		pixels = pixels[np.random.randint(0, pixels.shape[0]-1, nOfPixels)]
		return pixels

		
class refreshing_cMosGrid(cMosGrid):
	'''
	normal cMosGrids decide the pixel noise when they are initialized (more accurately, they decide which pixels of the real data set to copy),
	but if you want to have a wider range of possible images, it's better to "change" the pixels each time, otherwise you might have bias due 
	to a "bad" pixel always being in a certain position, or other similar problems.
	This subclass just gets random pixel values for each imaging
	'''	
	def __init__(self, xsize, ysize, nofXpixels, nofYpixels, PSF, noisePictureFilePath, imageStart = (0,0), imageSizes = None, magnification = 1):
		super().__init__(xsize, ysize, nofXpixels, nofYpixels, PSF, noisePictureFilePath, imageStart, imageSizes, magnification)
		self.pixelNoises = self.pixelNoises.flatten()
		
	def addNoise(self):
		self.pixels += np.random.choice(self.pixelNoises, self.pixels.shape, )

class fixed_cMosGrid(cMosGrid):
	'''
	in this cMos variant, each pixel copies the pixel that has its position in the original images.
	the sizes of the images will be the same as the original images
	'''	
	def __init__(self, xsize, ysize, PSF, noisePictureFilePath, magnification = 1):
		firstImage = cMosGrid.getPictures(noisePictureFilePath, pictureLimit=1)
		super().__init__(xsize, ysize, *firstImage.shape[1:], PSF, noisePictureFilePath, (0,0), None, magnification)
		
	def addNoise(self):
		self.pixels += self.pixelNoises[np.arange(self.pixels.shape[0])[:, None], np.arange(self.pixels.shape[1]), np.random.randint(self.pixelNoises.shape[2], size=self.pixels.shape)]
	def setRandomPixelNoises(self, noisePictureFilePath, imageStart = (0,0), imageSizes = None):
		self.pixelNoises = fixed_cMosGrid.getRandomPixelNoises(self.pixels.size, noisePictureFilePath, imageStart, imageSizes)    \
								.reshape((self.pixels.shape[0],self.pixels.shape[1], -1))
	@staticmethod
	def getRandomPixelNoises(nOfPixels, path, imageStart = (0,0), imageSizes = None):
		images = cMosGrid.getPictures(path, imageStart, imageSizes)
		pixels = images.reshape((images.shape[0], -1)).T
		return pixels

class strayLight_cMosGrid(refreshing_cMosGrid):
	def __init__(self, xsize, ysize, nofXpixels, nofYpixels, PSF, noisePictureFilePath, avoidanceRoi = 5, tweezerMinPixelDistance = 10, atomPeakMinValue = 1, magnification = 1):
		pixelGrid.__init__(self, xsize, ysize, nofXpixels, nofYpixels, PSF, magnification)
		self.setRandomPixelNoises(noisePictureFilePath, avoidanceRoi, tweezerMinPixelDistance, atomPeakMinValue)

	def setRandomPixelNoises(self, noisePictureFilePath, avoidanceRoi = 5, tweezerMinPixelDistance = 10, atomPeakMinValue = 1):
		self.pixelNoises = strayLight_cMosGrid.getRandomPixelNoises(noisePictureFilePath, avoidanceRoi, tweezerMinPixelDistance, atomPeakMinValue)
	
	@staticmethod
	def getRandomPixelNoises(path, avoidanceRoi = 5, tweezerMinPixelDistance = 10, atomPeakMinValue = 1):
		cai = cameraAtomImages(path)
		emptyPixels = ~ cai.getTweezerImage(avoidanceRoi, tweezerMinPixelDistance, atomPeakMinValue)
		images = cai.images.reshape((len(cai.images),-1))
		images = images[:,emptyPixels.flatten()]
		return images.flatten()

class Camera:
	def __init__(self, position, orientation, radius, pixelGrids: Union[Tuple[pixelGrid], List[pixelGrid], pixelGrid], focusDistance=None):
		self.position = np.array(position)
		self.orientation = orientation
		self.radius = radius
		self.focusDistance = focusDistance
		if not isinstance(pixelGrids, tuple):
			pixelGrids = (pixelGrids,)
		self.pixelGrids : Tuple[pixelGrid] = pixelGrids
	
	@staticmethod
	def intersect_yz_plane(start_points, direction_vectors):
		x0, y0, z0 = start_points.T
		dx, dy, dz = direction_vectors.T
				
		# Calculate the parameter t for the intersection with the YZ plane (x = 0)        
		t = -x0 / dx #it shouldn't happen that dx==0, because we should only have the photons that are oriented towards the camera

		# Calculate the intersection points
		y = y0 + t * dy
		z = z0 + t * dz
		
		# Combine the results into an array of intersection points
		return np.vstack((y, z)).T

	@staticmethod
	def hitsSpecifiedLens(startPoints, directionVectors, lensPosition, lensAngle, lensRadius, returnHitIndexes = False, focusDistance = None):
		if len(startPoints) == 0:
			if returnHitIndexes:
				return startPoints, np.array([], dtype=int)
			else:
				return startPoints
		#rotate all the points and directions to the lens's reference frame
		startPoints = np.array([CoordinateChange_extended(point, *lensPosition, *lensAngle) for point in startPoints])
		directionVectors = np.array([CoordinateChange_extended(direction, 0,0,0, *lensAngle) for direction in directionVectors])
		#let's see which photons have the correct orientation to hit the lens
		correctOriented = np.where(np.logical_and((directionVectors[:, 0] > 0), (startPoints[:, 0] < 0)))[0]
		#let's see which photons hit the lens
		hittingPositions = Camera.intersect_yz_plane(startPoints[correctOriented], directionVectors[correctOriented])
		actuallyHitting = np.where(np.linalg.norm(hittingPositions, axis=1) <= lensRadius)[0]
		if focusDistance is None:
			#the focus is perfect, so the camera position of the photon is the projection of the photon start point on the camera
			hittingPositions = startPoints[correctOriented][:,[1,2,0]]
		else:
			#todo continue
			pass
		if returnHitIndexes:
			return hittingPositions[actuallyHitting], correctOriented[actuallyHitting]
		return hittingPositions[actuallyHitting]
	
	def hitLens(self, photonStartPoints, photonDirections, returnHitIndexes = False):
		return self.hitsSpecifiedLens(photonStartPoints, photonDirections, self.position, self.orientation, self.radius, returnHitIndexes, self.focusDistance)	

	def takePicture(self, photonStartPoints, photonDirections, plot = False, saveToFile : str = None, **additionalAttributesToSave):
		initialNOfPhotons = len(photonDirections)
		hittingPositions = self.hitLens(photonStartPoints, photonDirections)
		nOfPhotonHittingLens = len(hittingPositions)
		nOfPhotonsHittingGrid = []
		extraGridInfos = []
		for grid in self.pixelGrids:
			grid.clear()
			extraGridInfos.append(grid.fillFromLens(hittingPositions))
			hittingPositions = grid.getRawPositions()
			nOfPhotonsHittingGrid.append(len(hittingPositions))
		
		image = self.pixelGrids[-1].pixels
		if plot:
			plt.figure(figsize=(14, 12))  # Ensure the plot is always square
			plt.imshow(image.T, origin='lower', cmap = 'Purples', aspect='auto')
			plt.colorbar(label='Intensity')
			plt.xlabel('x')
			plt.ylabel('y')
			plt.title('2D Function Plot')
			plt.show()
		if saveToFile is not None:
			additionalAttributesToSave['initial number of photons'] = initialNOfPhotons
			additionalAttributesToSave['number of photons hitting lens'] = nOfPhotonHittingLens
			for i in range(len(self.pixelGrids)):
				additionalAttributesToSave[f'number of photons hitting grid {i}'] = nOfPhotonsHittingGrid[i]
				for key, val in extraGridInfos[i].items():
					additionalAttributesToSave[f'grid {i}, {key}'] = val

			save_h5_image(saveToFile, image, **additionalAttributesToSave)
		return image
	@staticmethod
	def blurFromImages(folderPath):
		images, metadata = getImagesFrom_h5_files(folderPath)
		zGrid = np.array([val['zAtom'] for val in metadata.values()])
		ordered = np.argsort(zGrid)
		images = np.array(images)[ordered]
		zGrid = zGrid[ordered]
		xymin = -np.mean([val['range'] for val in metadata.values()])/2
		xymax = -xymin
		xystep = (xymax-xymin)/images.shape[1]
		images = np.abs(images)**2
		def getFromImages(x,y,z):
			z,_ = getIndexesAndFractionalPosition(z, zGrid)
			x = ((x-xymin)/(xymax-xymin) * (len(images[0])-1)).astype(int)
			y = ((y-xymin)/(xymax-xymin) * (len(images[0][0])-1)).astype(int)
			return images[z,x,y]
		# plot2D_function(lambda x,z: getFromImages(x,0,z), [xymin,xymax], [zmin,zmax], len(images[0]), len(images))
		f = randExtractor.distribFunFromPDF_2D_1D(getFromImages, [[xymin,xymax],[xymin,xymax],None], [xystep,xystep,zGrid])
		return f

class randExtractor:
	def __init__(self, distribFun, plotFunction = None):
		self.distribFun = distribFun
	
	def __call__(self, *args, **kwds):
		return self.distribFun(*args, **kwds)
	# def distribFunFromPDF(pdf, ranges, steps):
	#     #todo not working yet

	#     # Create a meshgrid for the given ranges and steps
	#     grids = [np.linspace(r[0], r[1], 1+int(np.ceil((r[1]-r[0])/s))) for r, s in zip(ranges, steps)]
	#     meshed_grids = np.meshgrid(*grids, indexing='ij')
	#     grid_points = np.stack(meshed_grids, axis=-1)

	#     # Evaluate the PDF at each point in the grid
	#     pdf_values = pdf(*[grid_points[..., i] for i in range(len(ranges))])
	#     return RegularGridInterpolator(grids, pdf_values)
	@staticmethod
	def getGrids(ranges, steps):
		return [(
			np.array(s) 
				if isinstance(s, list) or isinstance(s, tuple) or isinstance(s, np.ndarray)
		  	else np.linspace(r[0], r[1], 1+int(np.ceil((r[1]-r[0])/s)))
		  ) for r, s in zip(ranges, steps)]
	@staticmethod
	def interpolate2D_semigrid(x,y,z,valuesToInterpolate):
		'''
		interpolation for a function for which you have computed some values in a semi-grid 
			(instead of calculating the values for equally distanced values of x and y, only 
			the x values are equally distanced, and there's no limitation on the y values)
		x: n-array
		y: nxm-array
		z: nxm-array, z[i][j] = f(x[i], y[i][j])
		valuesToInterpolate: px2 array
		'''
		#let's find where the values would be in the x axis, and how far they are from the corresponding value in x
		i, p = getIndexesAndFractionalPosition(valuesToInterpolate[:, 0], x)

		averageY = (y[i].T*(1-p)+y[i+1].T*p).T
		averageZ = (z[i].T*(1-p)+z[i+1].T*p).T
		return np.array([np.interp(valuesToInterpolate[:,1][j], averageY[j], averageZ[j]) for j in range(len(i))])
	@staticmethod
	def interpolate3D_semigrid(x,y,z, f,valuesToInterpolate):
		'''
		x: n-array
		y: m-array
		z: nxmxp-array
		f: nxmxp-array, f[i][j][k] = f(x[i], y[j], z[i][j][k])
		valuesToInterpolate: qx3 array
		'''
		
		x_i, x_p = getIndexesAndFractionalPosition(valuesToInterpolate[:, 0], x)
		y_i, y_p = getIndexesAndFractionalPosition(valuesToInterpolate[:, 1], y)
		x_p, y_p = x_p[:,None], y_p[:,None]
		averageZ =	z[x_i	,	y_i]	* (1-x_p)	* (1-y_p)	+ \
			  		z[x_i+1	,	y_i]	* (x_p)		* (1-y_p)	+ \
			  		z[x_i	,	y_i+1]	* (1-x_p)	* (y_p)		+ \
					z[x_i+1	,	y_i+1]	* (x_p)		* (y_p)
		averageF =	f[x_i	,	y_i]	* (1-x_p)	* (1-y_p)	+ \
			  		f[x_i+1	,	y_i]	* (x_p)		* (1-y_p)	+ \
			  		f[x_i	,	y_i+1]	* (1-x_p)	* (y_p)		+ \
					f[x_i+1	,	y_i+1]	* (x_p)		* (y_p)
		return np.array([np.interp(valuesToInterpolate[:,2][j], averageZ[j], averageF[j]) for j in range(len(averageZ))])


	@staticmethod
	def distribFunFromPDF_2D(pdf, ranges, steps):        
		# Create a meshgrid for the given ranges and steps
		grids = randExtractor.getGrids(ranges, steps)
		meshed_grids = np.meshgrid(*grids, indexing='ij')
		grid_points = np.stack(meshed_grids, axis=-1)

		pdf_values = pdf(*[grid_points[..., i] for i in range(len(ranges))])

		#generic probability of finding x (independently from the value of y)
		pdf_values_x = np.sum(pdf_values, axis=1)
		cdf_values_x = np.cumsum(pdf_values_x)
		cdf_values_x /= cdf_values_x[-1]
		cdf_values_x[0] = 0#if the first value is not 0, when we then interpolate it is possible that the random number is lower than this value, and thus the interpolation will return an invalid number

		#probability of finding y, given a fixed value for x
		cdf_values_y = np.cumsum(pdf_values, axis = 1)
		cdf_values_y = (cdf_values_y.T / cdf_values_y[ :,-1]).T
		inverse_cdf_x_interp = interp1d(cdf_values_x, grids[0], kind='linear', fill_value="extrapolate")
		cdf_values_y[:,0]=0#if the first value is not 0, when we then interpolate it is possible that the random number is lower than this value, and thus the interpolation will return NaN
		inverse_y_points = np.dstack((meshed_grids[0], cdf_values_y)).reshape(-1, 2)#[[grids[0][i], cdf_values_y[i][j]] for i in range(len(grids[0])) for j in range(len(cdf_values_y[i]))]
		inverse_y_values = np.reshape(meshed_grids[1].T, (-1))

		def get_x_y(offsets, *t):
			rand = np.random.random(np.shape(offsets))
			x = inverse_cdf_x_interp(rand[:,0])
			y = randExtractor.interpolate2D_semigrid(grids[0], cdf_values_y, meshed_grids[1], np.column_stack((x,rand[:,1])))
			return offsets + np.column_stack((x,y))
		return randExtractor(get_x_y)
	
	@staticmethod
	def distribFunFromPDF_3D(pdf, ranges, steps):        
		# Create a meshgrid for the given ranges and steps
		grids = randExtractor.getGrids(ranges, steps)
		meshed_grids = np.meshgrid(*grids, indexing='ij')
		grid_points = np.stack(meshed_grids, axis=-1)

		pdf_values = pdf(*[grid_points[..., i] for i in range(len(ranges))])

		#generic probability of finding x (independently from the value of y)
		pdf_values_x = np.sum(pdf_values, axis=(1,2))
		cdf_values_x = np.cumsum(pdf_values_x)
		cdf_values_x /= cdf_values_x[-1]
		cdf_values_x[0] = 0#if the first value is not 0, when we then interpolate it is possible that the random number is lower than this value, and thus the interpolation will return an invalid number
		inverse_cdf_x_interp = interp1d(cdf_values_x, grids[0], kind='linear', fill_value="extrapolate")

		#probability of finding y, given a fixed value for x
		pdf_values_y_givenX = np.sum(pdf_values, axis=2)
		cdf_values_y = np.cumsum(pdf_values_y_givenX, axis = 1)
		cdf_values_y = (cdf_values_y.T / cdf_values_y[ :,-1]).T
		
		cdf_values_y[:,0]=0#if the first value is not 0, when we then interpolate it is possible that the random number is lower than this value, and thus the interpolation will return NaN

		#probability of finding z, given a fixed value for x and y
		cdf_values_z = np.cumsum(pdf_values, axis = 2)
		cdf_values_z = (cdf_values_z.T / cdf_values_z[:, :,-1].T).T
		cdf_values_z[:,:,0]=0#if the first value is not 0, when we then interpolate it is possible that the random number is lower than this value, and thus the interpolation will return NaN

		def get_x_y(offsets, *t):
			rand = np.random.random(np.shape(offsets))
			x = inverse_cdf_x_interp(rand[:,0])
			y = randExtractor.interpolate2D_semigrid(grids[0], cdf_values_y, meshed_grids[1][:,:,0], np.column_stack((x,rand[:,1])))
			z = randExtractor.interpolate3D_semigrid(grids[0], grids[1], cdf_values_z, meshed_grids[2], np.column_stack((x, y, rand[:,2])))
			return offsets + np.column_stack((x,y,z))
		return randExtractor(get_x_y)
	
	@staticmethod
	def distribFunFromradiusPDF_2D_1D(pdf, xrange, xstep, trange, tstep):
		#use this distribution generator for 2D radial functions (r=sqrt(x^2+y^2)) with an extra control dimension t (PDF(r,t) | integr(PDF(x,t) dx) = 1 for each t)
		#the generated extractor will take as inputs the value t (and an offset for (x,y)) and return a random value (x,y)
		f = randExtractor.distribFunFromPDF_1D_1D(pdf, xrange, xstep, trange, tstep)
		ff=f.distribFun
		def get_x_y(offset, t):
			randAngle = np.random.random(np.shape(t)) * 2 * np.pi
			r = ff(np.zeros_like(t),t)
			return offset + np.column_stack((r*np.cos(randAngle), r*np.sin(randAngle)))
		f.distribFun = get_x_y
		return f
	

	@staticmethod
	def distribFunFromPDF_1D_1D(pdf, xrange, xstep, trange, tstep):
		#use this distribution generator for 1D functions with an extra control dimension t (PDF(x,t) | integr(PDF(x,t) dx) = 1 for each t)
		#the generated extractor will take as inputs the value t (and an offset for x) and return a random value x
		# Create a meshgrid for the given ranges and steps
		ranges = [xrange,trange]
		steps = [xstep,tstep]
		grids = randExtractor.getGrids(ranges, steps)
		meshed_grids = np.meshgrid(*grids, indexing='ij')
		grid_points = np.stack(meshed_grids, axis=-1)

		pdf_values = pdf(*[grid_points[..., i] for i in range(len(ranges))])	
		cdf_values_x = np.cumsum(pdf_values, axis = 0)
		cdf_values_x -= cdf_values_x[0,:]
		cdf_values_x /= cdf_values_x[-1,:]
		
		def get_x(offsets,t):
			rand = np.random.random(np.shape(t))
			# y = griddata(inverse_y_points, inverse_y_values, np.column_stack((x,rand[:,1])), method='linear')
			x = randExtractor.interpolate2D_semigrid(grids[1], cdf_values_x.T, meshed_grids[0].T, np.column_stack((t,rand)))
			return offsets + x
		return randExtractor(get_x)
	@staticmethod
	def distribFunFromPDF_1D(pdf, ranges, steps):
		#use this distribution generator for 1D functions (PDF(x) | integr(PDF(x) dx) = 1)
		#the generated extractor accepts a will return a random value x that follows the given PDF

		grid = randExtractor.getGrids([ranges], [steps])[0]

		pdf_values = pdf(grid)
		cdf_values_x = np.cumsum(pdf_values)
		cdf_values_x -= cdf_values_x[0]
		cdf_values_x /= cdf_values_x[-1]
		interp = interp1d(cdf_values_x, grid, kind='linear', fill_value="extrapolate")
		def get_x(offsets):
			rand = np.random.random(np.shape(offsets))
			x = interp(rand)
			return offsets + x
		return randExtractor(get_x)
	@staticmethod
	def cellDistributionFromPDF_ND(pdf, ranges, steps):
		'''
		returns a very simple random extraction function: It computes the probability on a grid of points. 
		To extract the result it chooses a random point of the grid (according to the given PDF) and 
		adds a random displacement inside its "cell"
		'''
		grids = randExtractor.getGrids(ranges, steps)
		meshed_grids = np.meshgrid(*grids, indexing='ij')
		grid_points = np.stack(meshed_grids, axis=-1)
		
		pdf_values = pdf(*[grid_points[..., i] for i in range(len(ranges))])
		
		cdf_values = np.cumsum(pdf_values.flatten())
		cdf_values /= cdf_values[-1]
		def getValue(offset):
			chooseCell = np.random.random(len(offset))
			randomOffset = np.random.random(np.shape(offset))
			indexes = np.searchsorted(cdf_values, chooseCell)
			indexes = np.array(np.unravel_index(indexes, pdf_values.shape))

			#we'll choose a random point in the cell
			cells = meshed_grids[indexes]#np.array([grid_points[i] for i in np.unravel_index(index, pdf_values.shape)])
			cellSizes = meshed_grids[indexes + 1] - cells
			cell += cellSizes * (np.random.random(len(ranges)) * cellSizes)
			return offset + cell
		return randExtractor(getValue)

	@staticmethod
	def randomLosts(lostProbability):
		def removeLost(data):
			mask = np.random.rand(data.shape[0]) > lostProbability
			return data[mask]
		return randExtractor(removeLost)
	
	@staticmethod
	def distribFunFromPDF_2D_1D(pdf, xyt_ranges, xyt_steps):        
		#use this distribution generator for 2D functions with an extra control dimension t (PDF(x,y,t) | integr(PDF(x,y,t) dxdy) = 1 for each t)
		#the generated extractor will take as inputs the value t (and an offset for (x,y)) and return a random value (x,y)
		# Create a meshgrid for the given ranges and steps
		grids = randExtractor.getGrids(xyt_ranges, xyt_steps)
		meshed_grids = np.meshgrid(*grids, indexing='ij')
		grid_points = np.stack(meshed_grids, axis=-1)

		pdf_values = pdf(*[grid_points[..., i] for i in range(len(xyt_ranges))])

		#generic probability of finding x (independently from the value of y)
		pdf_values_x = np.sum(pdf_values, axis=1)
		cdf_values_x = np.cumsum(pdf_values_x, axis=0)
		cdf_values_x -= cdf_values_x[0,:]
		cdf_values_x /= cdf_values_x[-1,:]

		#probability of finding y, given a fixed value for x
		cdf_values_y = np.cumsum(pdf_values, axis = 1)
		cdf_values_y -= cdf_values_y[:,0,][:,None,:]
		cdf_values_y /= cdf_values_y[:,-1,:][:,None,:]

		cdf_values_y = np.swapaxes(cdf_values_y, 1, 2)
		meshed_grids[1] = np.swapaxes(meshed_grids[1], 1, 2)

		def get_x_y(offsets, t):
			rand = np.random.random(np.shape(offsets))
			x = randExtractor.interpolate2D_semigrid(grids[2], cdf_values_x.T, meshed_grids[0][:,0,:].T, np.column_stack((t,rand[:,0])))
			
			y = randExtractor.interpolate3D_semigrid(grids[0], grids[2], cdf_values_y, meshed_grids[1], np.column_stack((x,t,rand[:,1])))
			return offsets + np.column_stack((x,y))
		return randExtractor(get_x_y)

class experimentViewer:
	'''to do analysis of pre-computed runs, I don't want to create the entire experiment object 
	(which would also require to include a bunch of very heavy libraries). This class contains 
	all the necessary to load/save/show data'''
	# lastTimings:            array[nOfTimes][nOfAtoms] time of each instant for each atom
	# lastPositons:           array[nOfTimes][nOfAtoms][3] positions of each atom at each time frame
	# lastHits:               array{timeIndex, atomIndex, laserIndex}[nOfHits] all the recorded, specifies the time, atom and laser involved in the hit
	# lastGeneratedPhotons:   array[nOfHits][3] the generated photons for each hit

	def saveAcquisition(self, fileName, **metadata):        
		with h5py.File(fileName, 'w') as f:
			f.create_dataset("lastPositons", data = self.lastPositons)
			f.create_dataset("lastHits", data = np.array(self.lastHits))
			f.create_dataset("lastGeneratedPhotons", data = self.lastGeneratedPhotons)
			f.create_dataset("lastTimings", data = self.lastTimings)
			if hasattr(self, "tweezerPositions"):
				f.create_dataset("tweezerPositions", data = self.tweezerPositions)
			f.attrs.update(metadata)

	@property
	def hasHits(self):
		return len(self.lastHits) > 0

	def loadAcquisition(self, fileName):        
		with h5py.File(fileName, 'r') as f:
			self.lastPositons = np.array(f['lastPositons'])
			self.lastHits = np.array(f['lastHits'])
			if self.hasHits:
				self.lastHits = (self.lastHits[0], self.lastHits[1], self.lastHits[2])
			self.lastGeneratedPhotons = np.array(f['lastGeneratedPhotons'])
			if 'lastTimings' in f.keys():
				self.lastTimings = np.array(f['lastTimings'])
			if 'tweezerPositions' in f.keys():
				self.tweezerPositions = np.array(f['tweezerPositions'])
			metadata = dict(f.attrs)
			return metadata
		
	def getScatteredPhotons(self, acquisitionTime = None):
		if self.hasHits:
			if acquisitionTime is not None:
				higherUsableIndexes = np.array([np.searchsorted(self.lastTimings[:,i], acquisitionTime) for i in range(self.lastTimings.shape[1])])
				usableIndexes = self.lastHits[0] < higherUsableIndexes[self.lastHits[1]]
				usableHits = tuple([self.lastHits[i][usableIndexes] for i in range(len(self.lastHits))])
				usablePhotons = self.lastGeneratedPhotons[usableIndexes]
			else:
				usableHits = self.lastHits
				usablePhotons = self.lastGeneratedPhotons
			startPositions = self.lastPositons[usableHits[0:2]]
			directions = usablePhotons
		else:
			startPositions = np.zeros((0,3))
			directions = np.zeros((0,3))
		return startPositions, directions
	def positionsAtTime(self, time):
		positions = np.zeros(self.lastPositons.shape[1:])
		for atom_idx in range(len(positions)):
			timeIndex = np.searchsorted(self.lastTimings[:,atom_idx], time)
			positions[atom_idx] = self.lastPositons[timeIndex, atom_idx]
		return positions
	def plotTrajectoriesAndCameraAcquisition(self, camera : Camera):
		hitPositions, hitIdx = camera.hitLens(self.lastPositons[self.lastHits[0], self.lastHits[1]], self.lastGeneratedPhotons, returnHitIndexes=True)
		hitCamera = np.zeros((len(self.lastHits[0])), dtype=bool)
		hitCamera[hitIdx] = True
		fig = plt.figure()
		ax = fig.add_subplot(111, projection='3d')
		min_bounds = np.min(self.lastPositons, axis=(0, 1))
		max_bounds = np.max(self.lastPositons, axis=(0, 1))
		maxRange = np.max(max_bounds - min_bounds)
		baseQuiverLength = maxRange / 20
		min_bounds = (max_bounds + min_bounds) / 2 - maxRange / 2
		max_bounds = min_bounds + maxRange
		for atom_idx in range(self.lastPositons.shape[1]):
			ax.plot(self.lastPositons[:, atom_idx, 0], self.lastPositons[:, atom_idx, 1], self.lastPositons[:, atom_idx, 2], label=f'Atom {atom_idx+1}')
			ax.scatter(self.lastPositons[0, atom_idx, 0], self.lastPositons[0, atom_idx, 1], self.lastPositons[0, atom_idx, 2])
		if self.hasHits:
			for laser_idx in range(np.max(self.lastHits[2])+1):
				laserHits = np.where(self.lastHits[2] == laser_idx)[0]
				if len(laserHits) > 0:
					time_idx = self.lastHits[0][laserHits]
					atom_idx = self.lastHits[1][laserHits]
					ax.scatter(self.lastPositons[time_idx, atom_idx, 0], self.lastPositons[time_idx, atom_idx, 1], self.lastPositons[time_idx, atom_idx, 2], 
							label=f'laser {laser_idx+1} hits', s=5)
					for h in range(len(laserHits)):
						position = self.lastPositons[time_idx[h], atom_idx[h]]
						directions = self.lastGeneratedPhotons[laserHits[h]] * baseQuiverLength
						isAHit = hitCamera[laserHits[h]]
						if isAHit:
							ax.quiver(position[0], position[1], position[2], 
										directions[0], directions[1], directions[2], color='red')
							# ax.quiver(self.lastPositons[time_idx, atom_idx, 0], self.lastPositons[time_idx, atom_idx, 1], self.lastGeneratedPhotons[time_idx, atom_idx, 0])
				
		ax.set_xlabel('X Position (m)')
		ax.set_ylabel('Y Position (m)')
		ax.set_zlabel('Z Position (m)')
		ax.set_xlim(min_bounds[0], max_bounds[0])
		ax.set_ylim(min_bounds[1], max_bounds[1])
		ax.set_zlim(min_bounds[2], max_bounds[2])
		ax.set_title('3D Trajectories of Atoms')
		ax.legend()
		plt.show()
	def plotTrajectories(self):
		fig = plt.figure()
		ax = fig.add_subplot(111, projection='3d')
		min_atomBounds = np.nanmin(self.lastPositons, axis=0)
		max_atomBounds = np.nanmax(self.lastPositons, axis=0)
		atomRanges = max_atomBounds - min_atomBounds
		min_bounds = np.nanmin(self.lastPositons, axis=(0, 1))
		max_bounds = np.nanmax(self.lastPositons, axis=(0, 1))
		maxRange = np.maximum(np.nanmax(max_bounds - min_bounds), 1e-9)
		quiverLength = maxRange / 20
		min_bounds = (max_bounds + min_bounds) / 2 - maxRange / 2
		max_bounds = min_bounds + maxRange
		for atom_idx in range(self.lastPositons.shape[1]):
			ax.plot(self.lastPositons[:, atom_idx, 0], self.lastPositons[:, atom_idx, 1], self.lastPositons[:, atom_idx, 2], label=f'Atom {atom_idx+1}')
			ax.scatter(*self.lastPositons[0, atom_idx, :], s=20)
			if hasattr (self, "tweezerPositions"):
				ax.scatter(*self.tweezerPositions[atom_idx], s=30, color=plt.gca().lines[-1].get_color())
		if self.hasHits:
			for laser_idx in range(np.max(self.lastHits[2])+1):
				laserHits = np.where(self.lastHits[2] == laser_idx)[0]
				if len(laserHits) > 0:
					time_idx = self.lastHits[0][laserHits]
					atom_idx = self.lastHits[1][laserHits]
					ax.scatter(self.lastPositons[time_idx, atom_idx, 0], self.lastPositons[time_idx, atom_idx, 1], self.lastPositons[time_idx, atom_idx, 2], 
							label=f'laser {laser_idx+1} hits', s=5)
					# for h in range(len(laserHits)):
					# 	position = self.lastPositons[time_idx[h], atom_idx[h]]
					# 	directions = self.lastGeneratedPhotons[laserHits[h]] * quiverLength
					# 	ax.quiver(position[0], position[1], position[2], 
					# 			directions[0], directions[1], directions[2], color='red')
		# 			# ax.quiver(self.lastPositons[time_idx, atom_idx, 0], self.lastPositons[time_idx, atom_idx, 1], self.lastGeneratedPhotons[time_idx, atom_idx])

		ax.set_xlabel('X Position (m)')
		ax.set_ylabel('Y Position (m)')
		ax.set_zlabel('Z Position (m)')
		ax.set_xlim(min_bounds[0], max_bounds[0])
		ax.set_ylim(min_bounds[1], max_bounds[1])
		ax.set_zlim(min_bounds[2], max_bounds[2])
		ax.set_title('3D Trajectories of Atoms')
		ax.legend()
		plt.show()


if __name__ == '__main__':

	# positions  = np.array([[0,0,0],[0,0,0],[0,0.5,0],   [0,-0.5,0],   [0,.2,.2]])
	# directions = np.array([[1,0,0],[1,0,0],[.8,-0.6,0], [.8,-0.6,0],  [1,0,0]  ])

	# q=Camera.hitsSpecifiedLens(positions, directions, [1,0,0],[0,0,0], 1)
	# print(q)

	# pg = pixelGrid(.5,.5,9,9,lambda x:x)
	# pg.fillFromLens(q,1,1)
	# print(pg.pixels)

	# f=randExtractor.distribFunFromPDF_2D(ainy, [[-2,2]]*2, [.03]*2)

	# # plt.plot(x,y)
		
	# q=f(np.random.random(1000), np.random.random(1000))
	# plt.scatter(q[0],q[1], alpha=.03)
	# plt.show()


	# positions  = np.array([[0,0,0],[0,0,0],[0,0.5,0],   [0,-0.5,0],   [0,.2,.2]])
	# directions = np.array([[1,0,0],[1,0,0],[.8,-0.6,0], [.8,-0.6,0],  [1,0,0]  ])
	# positions  = np.array([[0,0,0]]*100000)
	# directions = np.array([[1,0,0]]*100000)

	# q=Camera.hitsSpecifiedLens(positions, directions, [1,0,0],[0,0,0], 1)
	# print(q)
	# f = randExtractor.distribFunFromPDF_2D(ainy, [[-2,2]]*2, [.01,0.0075])
	# pg = pixelGrid(4,4,50,50, f)
	# pg.fillFromLens(q)
	# plt.imshow(pg.pixels)
	# plt.show()
	# print(pg.pixels)

	# c = cMosGrid(1,1,100,100, lambda x:x, "Orca_testing/shots/", imageStart = (10,10), imageSizes = (-10,-10))
	# c.fillFromLens(np.array([[0,0]]))
	# plt.imshow(c.pixels)
	# plt.show()

	# r=np.linspace(0,1)
	# z=np.linspace(-2,2)
	# R,Z = np.meshgrid(r,z)
	# R=R.reshape((-1,))
	# Z=Z.reshape((-1,))
	# res = blur(R,Z,1,1,1,1,1)
	# res = res.reshape((len(r),len(z)))
	# res = np.abs(res)
	# plt.imshow(res)
	# plt.show()

	# def f11(x,t):
	# 	return np.abs((x-t)**2)
	# f=randExtractor.distribFunFromPDF_1D_1D(f11,[0,1],.05,[0,1],.01)
	# waist = 1e-3 
	# power = 10e-3

	# dt = 5e-9
	# detuning = 0#-5.5*trajlib.MHz
	# Lambda = 399e-9
	# k = 2*np.pi/Lambda
	# tweezerPower = 10e-3
	# E0 = np.sqrt(tweezerPower*753.4606273337396)
	# effectiveFocalLength = 25.5e-3
	# tweezerWaist = 1e-4
	# objective_Ray = 15.3e-3

	# # startPositions, directions = exp.getScatteredPhotons()
	# # # G = randExtractor.distribFunFromPDF_2D(lambda x,y: gauss(x, y,1,0,1e-8), [[-1e-9,1e-9]]*2, [5e-11]*2)
	# #'''
	# M = 8
	# finalPixelSize = 4.6e-6
	# finalNOfPixels = 40
	# finalCameraSize = finalNOfPixels * finalPixelSize
	# initialCameraSize = finalCameraSize / M#if we considered all the pixels, the camera size should be == 2*lensRadius = 32e-3 m
	# initialPixelSize = finalPixelSize / M
	# lensPosition = effectiveFocalLength
	# lensRadius = 16e-3
	# f11=lambda r,z:blur(r,z, k, E0, lensPosition, tweezerWaist, lensRadius)
	# plot2D_function(f11, [-0, lensRadius/100], [-50.1/k, 50/k], 50, 50)

	# f=randExtractor.distribFunFromPDF_1D_1D(f11,np.array([0,1])*.02, 0.0005, np.array([-1,1])*1, 0.01)
	# # # f(0,np.repeat(0.5,3))
	# """
	# for t in np.linspace(-.99,0.99,10):		
	# 	plt.plot(np.sort(f(0,np.repeat(t,10000))))
	# """
	# t=np.random.random(10000)
	# x=f(np.zeros_like(t),t)
	# plt.scatter(t,x, alpha=.03)
	# #"""
	# plt.show()
	# def f2(x,y):
	# 	b = np.logical_and(x<=0, np.logical_and(x >= -1, np.logical_and(y<=1, y>=-1)))
	# 	return b.astype(float)
	# f=randExtractor.cellDistributionFromPDF_ND(f2,[[-5,5],[-5,5]],[1,1])
	
	# t=np.random.random(10000)
	# x=f(np.zeros((10000,2)))
	# plt.scatter(x[0],x[1], alpha=.03)
	# plt.show()



	# images, metadata = getImagesFrom_h5_files("blurs/")
	# z = np.array([val['zAtom'] for val in metadata.values()])
	# ordered = np.argsort(z)
	# images = np.array(images)[ordered]
	# z = z[ordered]
	# zmin, zmax = z[0], z[-1]
	# zstep = z[1] - z[0]
	# xymin = -np.mean([val['range'] for val in metadata.values()])/2
	# xymax = -xymin
	# xystep = (xymax-xymin)/images.shape[1]
	# images = np.abs(images)**2
	# def getFromImages(x,y,z):
	# 	z = ((z-zmin)/(zmax-zmin) * (len(images)-1)).astype(int)
	# 	x = ((x-xymin)/(xymax-xymin) * (len(images[0])-1)).astype(int)
	# 	y = ((y-xymin)/(xymax-xymin) * (len(images[0][0])-1)).astype(int)
	# 	return images[z,x,y]
	# # plot2D_function(lambda x,z: getFromImages(x,0,z), [xymin,xymax], [zmin,zmax], len(images[0]), len(images))
	# f = randExtractor.distribFunFromPDF_2D_1D(getFromImages, [[xymin,xymax],[xymin,xymax],[zmin,zmax]], [xystep,xystep,zstep])
	# z = np.repeat(0,1000)
	# xy=f(np.zeros((len(z),2)), z)
	# plt.scatter(xy[:,0],xy[:,1], alpha=.03)
	# plt.show()

	# cai = cameraAtomImages("D:/simulationImages/fakeAtomArray/pictures")
	# cai.calcTweezerPositions(5,3)
	# # plt.imshow(cai.averageImage())
	# # plt.show()
	# plt.imshow(cai.getTweezerImage(3))
	# plt.show()
	
	# grid = strayLight_cMosGrid(20,20,20,20,lambda x:x, "D:/simulationImages/fakeAtomArray/pictures/", 5, 5, 3)
	# for i in range(10):
	# 	grid.fillFromLens(np.zeros((0,2)))
	# 	plt.imshow(grid.pixels)
	# 	plt.show()
	# 	grid.clear()


	# cai = cameraAtomImages("D:/simulationImages/real images/smallerSet - 171_10Tweezer_inTrap_7us_2Images", "images/Orca/fluorescence 1/frame")
	# cai.calcTweezerPositions()

	# # plt.imshow(cai.averageImage())
	# # plt.show()
	# # q = cai.getAtomROI(5, np.array([15,94]),np.array([0,1,2]))
	# q = cai.getAtomROI_fromBooleanArray(7, cai.getTrappedAtoms(8, 3))
	# plt.imshow(np.mean(q,axis=0))
	# plt.show()

	# file = "D:/simulationImages/real images/smallerSet - 171_10Tweezer_inTrap_7us_2Images"
	# internalPath = [f"images/Orca/fluorescence {i}/frame" for i in [1,2]]
	# cai = doubleCameraAtomImage(file, file, *internalPath)
	# cai.calcTweezerPositions()

	
	# plt.imshow(cai.first.averageImage().T)
	# plt.scatter(*(cai.first.tweezerPositions),s=1,c='red')
	# plt.show()

	# q = cai.first.getAtomROI(7)
	# plt.imshow(np.mean(q,axis=0))
	# plt.show()

	# q = cai.first.getAtomROI_fromBooleanArray(7, cai.first.getTrappedAtoms(8, 3))
	# plt.imshow(np.mean(q,axis=0))
	# plt.show()
	# q = cai.second.getAtomROI_fromBooleanArray(7, cai.second.getTrappedAtoms(8, 3))
	# plt.imshow(np.mean(q,axis=0))
	# plt.show()
	
	# q = cai.first.getAtomROI_fromBooleanArray(7, cai.getSurelyTrappedAtoms(8, 3))
	# plt.imshow(np.mean(q,axis=0))
	# plt.show()

	# fig = plt.figure()
	# ax = fig.add_subplot(111, projection='3d')	
	# roi = 7
	# x,y = np.meshgrid(*[np.arange(0,roi) for _ in range(2)], indexing="ij")
	# x = x[None,:,:] - roi//2 - (cai.first.tweezerPositions - cai.first.tweezerPixels)[0][:,None,None]
	# y = y[None,:,:] - roi//2 - (cai.first.tweezerPositions - cai.first.tweezerPixels)[1][:,None,None]
	# for i in ["allTweezers", "fullTweezers"]:
	# 	if i=="allTweezers":
	# 		z = cai.first.getAverageAtomROI(roi)
	# 	else:
	# 		sti = cai.getSurelyTrappedAtoms(8, 3)
	# 		z = cai.first.getAtomROI_fromBooleanArray(roi, sti, averageOnAtoms=True)

	# 	# Flatten the arrays for plotting
	# 	x_flat = x.flatten()
	# 	y_flat = y.flatten()
	# 	z_flat = z.flatten()

	# 	# Plot the surface
	# 	ax.scatter(x,y,z, label=i)
	# 	if i=="fullTweezers":
	# 		xy = np.vstack((x.flatten(),y.flatten()))
	# 		x_grid = np.linspace(np.min(x),np.max(x),100)
	# 		y_grid = np.linspace(np.min(y),np.max(y),100)
	# 		xv, yv = np.meshgrid(x_grid,y_grid)
	# 		initial_guess = np.asarray([np.max(z), 0, 0, 1, 1,np.min(z)])
	# 		p_opt,p_cov = opt.curve_fit(Ar.Gaussian_2D, xy, z.flatten(), p0 = initial_guess)
	# 		fitted_gaussian = Ar.Gaussian_2D((xv,yv),*p_opt).reshape(100,100)
	# 		ax.contourf(xv,yv,fitted_gaussian,100,alpha=0.5,cmap='plasma', label=f"{i} gaussian fit")

	# ax.set_xlabel('X')
	# ax.set_ylabel('Y')
	# ax.set_zlabel('Z')
	# ax.set_title('3D Function Plot')
	# plt.legend()
	# plt.show()


		
	# roi = 7
	# x,y = np.meshgrid(*[np.arange(0,roi) for _ in range(2)], indexing="ij")
	# x = x[None,:,:] - roi//2 - (cai.first.tweezerPositions - cai.first.tweezerPixels)[0][:,None,None]
	# y = y[None,:,:] - roi//2 - (cai.first.tweezerPositions - cai.first.tweezerPixels)[1][:,None,None]
	# for i in ["allTweezers", "fullTweezers"]:
	# 	if i=="allTweezers":
	# 		z = cai.first.getAverageAtomROI(roi)
	# 	else:
	# 		sti = cai.getSurelyTrappedAtoms(8, 3)
	# 		z = cai.first.getAtomROI_fromBooleanArray(roi, sti, averageOnAtoms=True)

	# 	r,az = cameraAtomImages.azimuthal_average(x,y,z,(0,0),30)
	# 	plt.plot(r,az, label = i)
	# plt.legend()
	# plt.show()

	# def f(xy):
	# 	r=np.linalg.norm(xy,axis=-1)
	# 	return Ar.Gaussian(r, 1, 0,2,0)
	# xy = np.random.uniform(-1,0,(10000,2))
	# x,y=xy.T
	# z=f(xy)
	# X,Y,Z = filterScatterToGrid(x,y,z,[10,9])
	# plt.imshow(Z)
	# plt.show()
	
	# exp = experimentViewer()
	# exp.loadAcquisition("d:/simulationImages/Yt171_12us_10tweezerArray_with_z_lattice/simulation/xsimulation_2.h5")
	# exp.plotTrajectories()
	# directions = np.random.uniform(-1,1,(10000,3))
	# norm = np.linalg.norm(directions,axis=1)
	# i=0
	# toExtract = norm >= 1
	# while(np.sum(toExtract) > 0 and i < 100):
	# 	directions[toExtract] = np.random.uniform(-1,1,(toExtract.sum(),3))
	# 	norm = np.linalg.norm(directions,axis=1)
	# 	toExtract = norm >= 1
	# 	i+=1
	# 	print(i)
	# directions /= norm[:,None]
	# positions = np.zeros_like(directions)
	# lensDistance = 25.5e-3
	# lensRadius = 16e-3
	# NA=np.sin(np.arctan(lensRadius/lensDistance))
	# print(f"{len(Camera.hitsSpecifiedLens(positions, directions, [lensDistance,0,0],[0,0,0], lensRadius)) / 10000}, {.5*(1-np.sqrt(1-NA**2))}")

	# def update_metadata_in_h5_files(folder_path, metadata_key, new_value):
	# 	"""
	# 	Updates the specified metadata key with a new value in all h5 files in the given folder.

	# 	Args:
	# 		folder_path (str): Path to the folder containing h5 files.
	# 		metadata_key (str): The metadata key to update.
	# 		new_value: The new value to set for the metadata key.
	# 	"""
	# 	files = [f for f in os.listdir(folder_path) if f.endswith('.h5')]
	# 	if not files:
	# 		raise ValueError("No h5 files found in the specified folder.")

	# 	for file in files:
	# 		file_path = os.path.join(folder_path, file)
	# 		with h5py.File(file_path, 'r+') as f:
	# 			if metadata_key in f.attrs:
	# 				f.attrs[metadata_key] = new_value
	# 				print(f"Updated {metadata_key} in {file}")
	# 			else:
	# 				print(f"{metadata_key} not found in {file}")
	# update_metadata_in_h5_files("D:/simulationImages/blurs", "range", 4.6e-6 * 8)

	# exp = experimentViewer()
	# exp.loadAcquisition("d:/simulationImages/correctScattering_Yt171_12us_10tweezerArray/simulation/\/.h5")
	# exp.getScatteredPhotons(6e-6)

	# img = np.array([[1,2,3],[4,5,6],[7,8,9]])
	# print(mirrorSymmetricImage(img))
	# print(mirrorSymmetricImage(img, True))
	# blur_z0,metadata = load_h5_image("d:/simulationImages/blurs/399nm/camera_atomZ=0.00e+00.h5", returnMetadata=True)
	# blur_z0 = np.abs(blur_z0)**2
	# blur_z0 = blur_z0[blur_z0.shape[0]//2,:]/np.sum(blur_z0)
	# size = metadata["requestedRange"]
	# pixelSize = 4.6e-6
	# # Create a pixel response function (rectangular function of width pixelSize)
	# x = np.linspace(-size/2, size/2, blur_z0.shape[0])
	# pixel_response = np.where(np.abs(x) <= pixelSize/2, 1, 0)
	# plt.plot(x, pixel_response * np.max(blur_z0), label='pixel')
	# pixel_response = pixel_response / np.sum(pixel_response)  # Normalize

	# # Convolve blur_z0 with the pixel response
	# blur_z0_pixel = np.convolve(blur_z0, pixel_response, mode='same')
	# def gaussian(x, A, mu, sigma, offset):
	# 	return A * np.exp(-2 * ((x - mu) / sigma) ** 2) + offset

	# # Initial guess: amplitude, mean, stddev, offset
	# p0 = [np.max(blur_z0_pixel), 0, pixelSize, np.min(blur_z0_pixel)]
	# params, cov = curve_fit(gaussian, x, blur_z0_pixel, p0=p0)

	# fit_curve = gaussian(x, *params)
	# plt.plot(x, fit_curve, label='Gaussian fit')
	# print("Fitted parameters: A = {:.3g}, mu = {:.3g}, sigma = {}, offset = {:.3g}".format(*params))
	# plt.plot(x, blur_z0, label='blur function (camera space)')
	# plt.plot(x, blur_z0_pixel, label='Convolved with pixel')
	
	# plt.legend()
	# plt.show()
	# files = [f"D:/simulationImages/magnified_Yt171_20us_10tweezerArray/simulation/simulation_{i}.h5" for i in range(0, 1)]
	# exp = experimentViewer()
	# for file in files:
	# 	metadata = exp.loadAcquisition(file)
	# 	for i in range(exp.lastPositons.shape[1]):
	# 		with h5py.File(file.replace(".h5",f"atom{i}.h5"), 'w') as f:
	# 			data = exp.lastPositons[:, i, :]
	# 			data = data[np.logical_not(np.isnan(data).any(axis=1))]
	# 			center = metadata["tweezer_centers"][i]
	# 			data -= center[None,:]
				
	# 			plt.plot(
	# 				data[:, 2], label=f'Atom {i+1}')
	# 			f.create_dataset("positions", data=exp.lastPositons[:, i, :])
	# 			f.close()
	# 	plt.show()
	# def testPDF(nx, ny, nz):
	# 	return np.exp(-nx) * np.exp(-ny) * np.exp(-nz*.5)
	# G = randExtractor.distribFunFromPDF_3D(lambda x,y,z: testPDF(x,y,z), [[0,4]]*3, [5e-2]*3)

	def testPDF(x,y,z):
		# if x>2:
		# 	return 1 if y<1 else 0.1
		# if x>1:
		# 	return .5 if z>1 else 0.1
		# return .25	
		result = np.zeros_like(x)
		result[np.logical_and(x>2,y<1)] = 1
		result[np.logical_and(x>2,y>=1)] = 0.1
		result[np.logical_and(x>1,np.logical_and(x<=2,z>1))] = 0.5
		result[np.logical_and(x>1,np.logical_and(x<=2,z<=1))] = 0.1
		result[x<=1] = .25
		return result
	G = randExtractor.distribFunFromPDF_3D(lambda x,y,z: testPDF(x,y,z), [[0,3],[0,2],[0,2]], [5e-2]*3)
	
	# def testPDF(nx, ny, nz):
	# 	return np.exp(-(nx**2+ny**2+.05*nz**2))	
	# G = randExtractor.distribFunFromPDF_3D(lambda x,y,z: testPDF(x,y,z), [[-5,5]]*3, [5e-2]*3)

	extractedPoints = np.zeros((10000,3))
	extractedPoints = G(extractedPoints)

	# Then add this for 3D scatter plot:
	fig = plt.figure(figsize=(10, 8))
	ax = fig.add_subplot(111, projection='3d')
	ax.scatter(extractedPoints[:, 0], extractedPoints[:, 1], extractedPoints[:, 2], alpha=0.1)
	ax.set_xlabel('X')
	ax.set_ylabel('Y')
	ax.set_zlabel('Z')
	ax.set_title('3D Distribution')
	plt.show()
	# for i in range(len(extractedPoints[0])):
	# 	# extractedPoints[:,i] = G(np.zeros((2,1)))

	plt.scatter(extractedPoints[:,0], extractedPoints[:,1],alpha=.1)
	plt.show()
	plt.scatter(extractedPoints[:,0], extractedPoints[:,2],alpha=.1)
	plt.show()
	plt.scatter(extractedPoints[:,1], extractedPoints[:,2],alpha=.1)
	plt.show()
	# extractedPoints = np.zeros((10000,3))
	# extractedPoints = G(extractedPoints)
	# plt.scatter(extractedPoints[:,1], extractedPoints[:,2],alpha=.1)
	# plt.show()
	pass
