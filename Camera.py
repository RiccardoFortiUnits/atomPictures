
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
def plot2D(Z, x_range, y_range, figureTitle=""):
	if Z.dtype == np.complex128:
		plt.figure(figureTitle + " (abs value)")
		plt.imshow((np.abs(Z.T)), extent=(y_range[0], y_range[1], x_range[0], x_range[1]), origin='lower', cmap='viridis', aspect='auto')
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


def save_h5_image(imageName, image, **metadata):	
	with h5py.File(imageName, 'w') as f:
		f.create_dataset('image', data=image)
		f.attrs.update(metadata)

def load_h5_image(path, returnMetadata = False):
	'''
		gets an image that is the average of all the images contained in folderPath.
		also returns a dictionary {fileName:metadata} of all the metadata contained in the files
	'''
	with h5py.File(path, 'r') as f:
		image = np.array(f['image'])
		if returnMetadata:
			metadata = dict(f.attrs)
			return image, metadata

	return image
def getImagesFrom_h5_files(folderPath):
	'''
		gets all the images contained in folderPath.
		also returns a dictionary {fileName:metadata} of all the metadata contained in the files
	'''
	files = [f for f in os.listdir(folderPath) if f.endswith('.h5')]
	if not files:
		raise ValueError("No h5 files found in the specified folder.")

	images = []
	metadata = {}

	for file in files:
		with h5py.File(os.path.join(folderPath, file), 'r') as f:
			image = np.array(f['image'])
			images.append(image)
			metadata[file] = dict(f.attrs)

	return images, metadata

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
		coordinates[:,0] = (coordinates[:,0] / (self.nofXpixels - 1) - .5) * self.xsize * self.magnification
		coordinates[:,1] = (coordinates[:,1] / (self.nofYpixels - 1) - .5) * self.ysize * self.magnification
		return coordinates
	def fillFromLens(self, rawPhotonPositions):
		'''
		rawPhotonPositions: array of 2D positions (could also be a 3D position, if the PSF is also dependent on the z-coordinate)

		returns a dictionary with some extra info (though the default pixel grid only returns a non-empty dictionary if there's no photon to work on)
		'''
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
		psfPositions = self.PSF(inputGrid.getRawPositions())
		x_normalized, y_normalized = self._normalizeCoordinate(psfPositions[:,0], psfPositions[:,1])
		np.add.at(self.pixels, (x_normalized, y_normalized), 1)
	def clear(self):
		self.pixels = np.zeros((self.nofXpixels,self.nofYpixels))
	@staticmethod
	def looseShotsForQuantumEfficiency(shotsCoordinates, QE : float):
		mask = np.random.rand(shotsCoordinates.shape[1]) > QE
		return shotsCoordinates[mask]

class cMosGrid(pixelGrid):
	def __init__(self, xsize, ysize, nofXpixels, nofYpixels, PSF, noisePictureFilePath, imageStart = (0,0), imageSizes = None):

		super().__init__(xsize, ysize, nofXpixels, nofYpixels, PSF)
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
	def getPictures(path, imageStart = (0,0), imageSizes = None):
		files_to_analyse = os.listdir(path)
		
		raw_images = None
		for i, file in enumerate(files_to_analyse):
			f = h5py.File(path+file, 'r')
			img = np.asarray(f['images/Orca/Test_image/frame'])
			if raw_images is None:
				if imageSizes is None:
					imageSizes = img.shape
				if imageSizes[0] < 0: imageSizes = (img.shape[0] + 1 + imageSizes[0] - imageStart[0], imageSizes[1])
				if imageSizes[1] < 0: imageSizes = (imageSizes[0], img.shape[1] + 1 + imageSizes[1] - imageStart[1])
				raw_images = np.zeros((len(files_to_analyse), imageSizes[0], imageSizes[1]), dtype=np.uint8)
			
			raw_images[i] = img[imageStart[0]:imageStart[0] + imageSizes[0], imageStart[1]:imageStart[1] + imageSizes[1]].astype(np.uint8)
		return raw_images#so we won't occupy too much memory (we'll still occupy a lot though)
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
	def __init__(self, xsize, ysize, nofXpixels, nofYpixels, PSF, noisePictureFilePath, imageStart = (0,0), imageSizes = None):
		super().__init__(xsize, ysize, nofXpixels, nofYpixels, PSF, noisePictureFilePath, imageStart, imageSizes)
		self.pixelNoises = self.pixelNoises.flatten()
		
	def addNoise(self):
		self.pixels += np.random.choice(self.pixelNoises, self.pixels.shape, )


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
		z = np.array([val['zAtom'] for val in metadata.values()])
		ordered = np.argsort(z)
		images = np.array(images)[ordered]
		z = z[ordered]
		zmin, zmax = z[0], z[-1]
		zstep = z[1] - z[0]
		xymin = -np.mean([val['range'] for val in metadata.values()])/2
		xymax = -xymin
		xystep = (xymax-xymin)/images.shape[1]
		images = np.abs(images)**2
		def getFromImages(x,y,z):
			z = ((z-zmin)/(zmax-zmin) * (len(images)-1)).astype(int)
			x = ((x-xymin)/(xymax-xymin) * (len(images[0])-1)).astype(int)
			y = ((y-xymin)/(xymax-xymin) * (len(images[0][0])-1)).astype(int)
			return images[z,x,y]
		# plot2D_function(lambda x,z: getFromImages(x,0,z), [xymin,xymax], [zmin,zmax], len(images[0]), len(images))
		f = randExtractor.distribFunFromPDF_2D_1D(getFromImages, [[xymin,xymax],[xymin,xymax],[zmin,zmax]], [xystep,xystep,zstep])
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
		#let's find where the values would be in the x axis
		interpolatedIndex = (valuesToInterpolate[:,0]-x[0])*(len(x)-1)/(x[-1]-x[0])
		i = interpolatedIndex.astype(int)
		p = interpolatedIndex - i.astype(float)
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
		#let's find where the values would be in the x axis
		interpolatedIndex = (valuesToInterpolate[:,0]-x[0])*(len(x)-1)/(x[-1]-x[0])
		x_i = interpolatedIndex.astype(int)
		x_p = (interpolatedIndex - x_i.astype(float))[:,None]
		interpolatedIndex = (valuesToInterpolate[:,1]-y[0])*(len(y)-1)/(y[-1]-y[0])
		y_i = interpolatedIndex.astype(int)
		y_p = (interpolatedIndex - y_i.astype(float))[:,None]
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
		grids = [np.linspace(r[0], r[1], 1+int(np.ceil((r[1]-r[0])/s))) for r, s in zip(ranges, steps)]
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
		grids = [np.linspace(r[0], r[1], 1+int(np.ceil((r[1]-r[0])/s))) for r, s in zip(ranges, steps)]
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

		grid = np.linspace(ranges[0], ranges[1], 1+int(np.ceil((ranges[1]-ranges[0])/steps)))

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
		grids = [np.linspace(r[0], r[1], 1+int(np.ceil((r[1]-r[0])/s))) for r, s in zip(ranges, steps)]
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
		grids = [np.linspace(r[0], r[1], 1+int(np.ceil((r[1]-r[0])/s))) for r, s in zip(xyt_ranges, xyt_steps)]
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
	# lastPositons:           array[nOfTimes][nOfAtoms][3] positions of each atom at each time frame
	# lastHits:               array{timeIndex, atomIndex, laserIndex}[nOfHits] all the recorded, specifies the time, atom and laser involved in the hit
	# lastGeneratedPhotons:   array[nOfHits][3] the generated photons for each hit

	def saveAcquisition(self, fileName, **metadata):        
		with h5py.File(fileName, 'w') as f:
			f.create_dataset("lastPositons", data = self.lastPositons)
			f.create_dataset("lastHits", data = np.array(self.lastHits))
			f.create_dataset("lastGeneratedPhotons", data = self.lastGeneratedPhotons)
			f.create_dataset("lastTimings", data = self.lastTimings)
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
			metadata = dict(f.attrs)
			return metadata
		
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
		min_bounds = np.nanmin(self.lastPositons, axis=(0, 1))
		max_bounds = np.nanmax(self.lastPositons, axis=(0, 1))
		max_bounds = np.nanmax(self.lastPositons, axis=(0, 1))
		maxRange = np.nanmax(max_bounds - min_bounds)
		quiverLength = maxRange / 20
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
						directions = self.lastGeneratedPhotons[laserHits[h]] * quiverLength
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

	
	pass
