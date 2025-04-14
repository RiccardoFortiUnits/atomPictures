# import Simulations_Libraries.trajectory_library as trajlib
import numpy as np
import matplotlib.pyplot as plt
from Camera import *
from scipy.stats import poisson
from scipy.optimize import curve_fit
# import Simulations_Libraries.general_library as genlib
from scipy.interpolate import RegularGridInterpolator
from numpy.fft import fft2, fftshift


def h(xyz, k):
	'''
	Fresnel approximation of the free space impulse response function
	'''
	return np.exp(1j*k*xyz[...,2] + .5j*k/xyz[...,2]*(xyz[...,0]**2+xyz[...,1]**2)) * k/(2j*np.pi*xyz[...,2])
def hreal(xyz,k):
	'''
	free space impulse response function without approximation
	'''
	r=np.linalg.norm(xyz,axis=-1)
	cosTheta=xyz[...,2]/r
	return np.exp(1j*k*r)*cosTheta/r

def h_separated(xyz, k):
	return k/(2j*np.pi*xyz[...,2]), k*xyz[...,2] + .5*k/xyz[...,2]*(xyz[...,0]**2+xyz[...,1]**2)
def hreal_separated(xyz,k):
	'''
	free space impulse response function without approximation
	'''
	r=np.linalg.norm(xyz,axis=-1)
	cosTheta=xyz[...,2]/r
	return cosTheta/r, k*r
def impulseExpansionInFreeSpace(xyz1, U0, xyz0, k):
	return U0(xyz0) * h(xyz1-xyz0, k)

def boundingBox(map):
	''' gets the index coordinates of a rectangle that includes all the elements of map that are different from 0'''
	vals = np.where(map)
	minx=np.min(vals[1])
	maxx=np.max(vals[1])
	miny=np.min(vals[0])
	maxy=np.max(vals[0])
	return ((minx, maxx), (miny, maxy))

def expandMap(map, maxExceptedValue, ranges, extensionMultiplier = 2, reductionadder = 1, reductionMultiplier = 20):
	'''returns a new search range for the estimate of the acceptable ranges of a map,
	so that the new map (generated with an external function that uses the new search range) 
	should contains more (and hopefully all the acceptable region(s?) of the function... Sorry, 
	I haven't explained myself... Oh well, hopefully I'll remember what this actually does'''
	mm = map < maxExceptedValue
	minValues = [ranges[0][0], ranges[1][0]]
	pixelSizes = [(max-min)/map.shape[i] for i, (min,max) in enumerate(ranges)]
	if not np.any(mm):#no valid pixel?
		# #Let's assume that there will be a valid value in the center, and just return the size of one pixel centered in the map
		# pixelHalfSizes = [(max-min)*reductionadder*.5/map.shape[i] for i, (min,max) in enumerate(ranges)]
		# mapCenter = [(max+min)/2 for (min,max) in ranges]
		# return ((mapCenter[0]-pixelHalfSizes[0],mapCenter[0]+pixelHalfSizes[0]),(mapCenter[1]-pixelHalfSizes[1],mapCenter[1]+pixelHalfSizes[1]))
		
		#Let's find an estimate for the minimum, and let's move the range there
		axes=[0,1]
		limits = [[0,len(map)-1], [0,len(map[0])-1]]
		onAnyLimit=False
		newCenters = []
		localMins = np.unravel_index(np.argmin(np.where(np.isnan(map), np.inf, map)), map.shape)
		localMins = [localMins[1],localMins[0]]
		for min, axis, limit in zip(localMins, axes,limits):
			onLimit = -1
			for i, lim in enumerate(limit):
				if min == lim:
					onLimit = i
			if onLimit != -1:
				#let's move the range half a range to the direction of the mimimum
				onAnyLimit = True
				newCenters.append(map.shape[onLimit] * (onLimit))
			else:
				newCenters.append(min)
			newCenters[-1] = newCenters[-1] * pixelSizes[axis] + minValues[axis]

		if onAnyLimit:
			#let's not reduce the size of the ranges. It's better to first have an estimate of the position of the mimimum, and then reduce the size to center it
			newHalfSizes = [(max-min)*.5*extensionMultiplier for i, (min,max) in enumerate(ranges)]
		else:
			#we have the miminum inside the frame. Let's center it
			#since when we zoom the value of the function will be reduced, if we zoom too much we might extend the "0" area outside of the zoom.
			minVal = map[localMins[1],localMins[0]] / maxExceptedValue
			expectedMaxVal = minVal / np.sqrt(map.size)
			if expectedMaxVal < 1:
				newHalfSizes = [(max-min)*.5/minVal for (min,max) in (ranges)]
			else:
				newHalfSizes = [(max-min)*.5/map.shape[i] for i, (min,max) in enumerate(ranges)]
		return ((newCenters[0]-newHalfSizes[0],newCenters[0]+newHalfSizes[0]),(newCenters[1]-newHalfSizes[1],newCenters[1]+newHalfSizes[1]))

		
	((minx, maxx), (miny, maxy)) = boundingBox(mm)
	centerX=(minx+maxx)/2
	centerY=(miny+maxy)/2
	indexes = [minx, maxx, miny, maxy]
	signs = [-1,1,-1,1]
	centers =[centerX,centerX,centerY,centerY]
	limits = [0,len(map)-1,0,len(map[0])-1]
	minValues = [minValues[0],minValues[0],minValues[1],minValues[1]]
	pixelSizes = [pixelSizes[0],pixelSizes[0],pixelSizes[1],pixelSizes[1]]
	newValues = []
	for val, center, sign, limit, min, pixelSize in zip(indexes,centers,signs,limits,minValues,pixelSizes):
		if val==limit:#acceptable range possibly extending outside of the current range?
			newIndex = center + extensionMultiplier * (val - center)
		else:
			newIndex = val + sign * reductionadder
		newValues.append(min + pixelSize * newIndex)
	return ((newValues[0],newValues[1]),(newValues[2],newValues[3]))

def expandMap_old(map, maxExceptedValue, ranges, extensionMultiplier = 2, reductionadder = 1, reductionMultiplier = 20):
	'''returns a new search range for the estimate of the acceptable ranges of a map,
	so that the new map (generated with an external function that uses the new search range) 
	should contains more (and hopefully all the acceptable region(s?) of the function... Sorry, 
	I haven't explained myself... Oh well, hopefully I'll remember what this actually does'''
	mm = map < maxExceptedValue
	minValues = [ranges[0][0], ranges[1][0]]
	pixelSizes = [(max-min)/map.shape[i] for i, (min,max) in enumerate(ranges)]
	if not np.any(mm):#no valid pixel?
		# #Let's assume that there will be a valid value in the center, and just return the size of one pixel centered in the map
		# pixelHalfSizes = [(max-min)*reductionadder*.5/map.shape[i] for i, (min,max) in enumerate(ranges)]
		# mapCenter = [(max+min)/2 for (min,max) in ranges]
		# return ((mapCenter[0]-pixelHalfSizes[0],mapCenter[0]+pixelHalfSizes[0]),(mapCenter[1]-pixelHalfSizes[1],mapCenter[1]+pixelHalfSizes[1]))
		
		#Let's find an estimate for the minimum, and let's move the range there
		axes=[0,1]
		limits = [[0,len(map)-1], [0,len(map[0])-1]]
		onAnyLimit=False
		newCenters = []
		localMins = np.unravel_index(np.argmin(np.where(np.isnan(map), np.inf, map)), map.shape)
		localMins = [localMins[1],localMins[0]]
		for min, axis, limit in zip(localMins, axes,limits):
			onLimit = -1
			for i, lim in enumerate(limit):
				if min == lim:
					onLimit = i
			if onLimit != -1:
				#let's move the range half a range to the direction of the mimimum
				onAnyLimit = True
				newCenters.append(map.shape[onLimit] * (onLimit))
			else:
				newCenters.append(min)
			newCenters[-1] = newCenters[-1] * pixelSizes[axis] + minValues[axis]

		if onAnyLimit:
			#let's not reduce the size of the ranges. It's better to first have an estimate of the position of the mimimum, and then reduce the size to center it
			newHalfSizes = [(max-min)*.5*extensionMultiplier for i, (min,max) in enumerate(ranges)]
		else:
			#we have the miminum inside the frame. Let's center it
			newHalfSizes = [(max-min)*reductionMultiplier*.5/map.shape[i] for i, (min,max) in enumerate(ranges)]
		return ((newCenters[0]-newHalfSizes[0],newCenters[0]+newHalfSizes[0]),(newCenters[1]-newHalfSizes[1],newCenters[1]+newHalfSizes[1]))

		
	((minx, maxx), (miny, maxy)) = boundingBox(mm)
	centerX=(minx+maxx)/2
	centerY=(miny+maxy)/2
	indexes = [minx, maxx, miny, maxy]
	signs = [-1,1,-1,1]
	centers =[centerX,centerX,centerY,centerY]
	limits = [0,len(map)-1,0,len(map[0])-1]
	minValues = [minValues[0],minValues[0],minValues[1],minValues[1]]
	pixelSizes = [pixelSizes[0],pixelSizes[0],pixelSizes[1],pixelSizes[1]]
	newValues = []
	for val, center, sign, limit, min, pixelSize in zip(indexes,centers,signs,limits,minValues,pixelSizes):
		if val==limit:#acceptable range possibly extending outside of the current range?
			newIndex = center + extensionMultiplier * (val - center)
		else:
			newIndex = val + sign * reductionadder
		newValues.append(min + pixelSize * newIndex)
	return ((newValues[0],newValues[1]),(newValues[2],newValues[3]))
		
def MapStatus(map, maxExceptedValue,fillRatio):
	mm = map < maxExceptedValue
	if not np.any(mm):
		return "noVal"
	((minx, maxx), (miny, maxy)) = boundingBox(mm)
	limits = np.array([0,len(map)-1,0,len(map[0])-1])
	indexes = np.array([minx, maxx, miny, maxy])
	if np.any(limits==indexes):
		return "notComplete"
	validRatio = np.count_nonzero(mm) / map.size
	if validRatio < fillRatio:
		return "tooSmall"
	return "ok"


# def q(ranges,sizes):
# 	X,Y=np.meshgrid(*([np.linspace(min,max,sizes[i]) for i, (min,max) in enumerate(ranges)]))

# 	return ((X+.2)**2+4*(Y-0.5)**2 < 4).astype(int)

# ranges = [[-1,1],[0,4]]
# sizes=[30,20]

# m=q([[-3,3],[-3,3]],sizes)
# plt.imshow(m)
# plt.show()
# for i in range(10):
# 	m=q(ranges,sizes)
# 	plt.imshow(m)
# 	plt.show()
# 	ranges = expandMap(m, ranges)
# print(expandMap(m, ranges))

def impulseExpansionInFreeSpace_separated(xyz1, U0, xyz0, k):
	rh, ah = h_separated(xyz1-xyz0, k)
	ru, au = U0(xyz0)
	r=ru*rh
	a=au+ah
	return r, a
def impulseExpansionInFreeSpace_separatedAndWithValidRange(xyz1, U0, xyz0, k):
	rh, ah = h_separated(xyz1-xyz0, k)
	ru, au = U0(xyz0)
	r=ru*rh
	a=au+ah
	grad_x, grad_y = np.gradient(a, axis=(0, 1))
	grad = np.sqrt(grad_x**2 + grad_y**2)
	# Estimate the position of the local minimum using the gradient
	# grad_magnitude = np.sqrt(grad_x**2 + grad_y**2)
	# grad_direction_x = -grad_x / (grad_magnitude + 1e-10)  # Avoid division by zero
	# grad_direction_y = -grad_y / (grad_magnitude + 1e-10)

	# # Compute a weighted average of the gradient directions to estimate the local minimum
	# weights = 1 / (grad_magnitude + 1e-10)  # Higher weights for smaller gradients
	# estimated_min_x = np.sum(grad_direction_x * weights) / np.sum(weights)
	# estimated_min_y = np.sum(grad_direction_y * weights) / np.sum(weights)

	# # Add the estimated offset to the grid center
	# grid_center_x = xyz0[..., 0].mean()
	# grid_center_y = xyz0[..., 1].mean()
	# estimated_min_position = (grid_center_x + estimated_min_x, grid_center_y + estimated_min_y)
	# print("Estimated position of local minimum:", estimated_min_position)
	a = np.nan_to_num(a, nan=0)
	return r * np.exp(1j * a), grad

from testDiffraction_thirdAttempt import mappedFunction

def expandInFreeSpace_separated(U0, xy0_ranges,z0=0, k=1):
	'''
	returns the convolution U1(x1,y1,z1) of the field U0(x,y,z=z0) with h(x,y,z) to obtain the value of U1(x,y,z)
	the integral is calculated in the specified ranges
	'''
	nOfSamplesPerDimension = 200
	nOfSamplesForMapSearch = 50
	def get_xyz0(ranges, sizes = nOfSamplesForMapSearch):
		X,Y = np.meshgrid(*[np.linspace(min,max,sizes)for (min,max) in ranges])
		Z=z0 * np.ones_like(X)
		return np.stack((X,Y,Z), axis=2)
	def alternateRange(l,i):
		if i%2==0:
			return range(l)
		return range(l-1,-1,-1)
	scale=1
	xyz0 = get_xyz0(xy0_ranges)
	def U1(xyz1):
		if len(xyz1.shape) == 2:#list of points instead of a grid of points?
			xyz1 = np.reshape(xyz1, (xyz1.shape[0],1,xyz1.shape[1]))
		nonlocal xyz0, scale, xy0_ranges
		integral = np.zeros(xyz1.shape[:-1], dtype=np.complex128)
		startCenter, startSize = mappedFunction.rangesToCenterAndSize(xy0_ranges)
		for i in range(len(xyz1)):
			print(i)
			for j in alternateRange(len(xyz1[i]),i):
				if j==7:
					i += 0

				def getOnlyAngle(xy0):
					xyz0 = np.concatenate((xy0, z0*np.ones(xy0.shape[:-1])[...,None]), axis = -1)
					_, angle = impulseExpansionInFreeSpace_separated(xyz1[i, j, :], U0, xyz0, k)
					return angle
				def getImpulseExpansion(xy0):
					xyz0 = np.concatenate((xy0, z0*np.ones(xy0.shape[:-1])[...,None]), axis = -1)
					ray, angle = impulseExpansionInFreeSpace_separated(xyz1[i, j, :], U0, xyz0, k)
					angle = np.nan_to_num(angle, nan=0)
					return ray * np.exp(1j * angle)
				
				startMap = mappedFunction(getOnlyAngle, startCenter, startSize, np.repeat(nOfSamplesForMapSearch,2))
				zoomedMaps = startMap.getZoomedFlatSections(np.pi*4,.3)
				integ = 0
				for map in zoomedMaps:
					map.f = getImpulseExpansion
					map.resolution = np.repeat(nOfSamplesPerDimension,2)
					integ += map.integral()
				integral[i][j] = integ
		return integral * k/(2j*np.pi)
	return U1

def expandInFreeSpace_separated_old(U0, xy0_ranges,z0=0, k=1):
	'''
	returns the convolution U1(x1,y1,z1) of the field U0(x,y,z=z0) with h(x,y,z) to obtain the value of U1(x,y,z)
	the integral is calculated in the specified ranges
	'''
	nOfSamplesPerDimension = 200
	nOfSamplesForMapSearch = 50
	def get_xyz0(ranges, sizes = nOfSamplesForMapSearch):
		X,Y = np.meshgrid(*[np.linspace(min,max,sizes)for (min,max) in ranges])
		Z=z0 * np.ones_like(X)
		return np.stack((X,Y,Z), axis=2)
	def alternateRange(l,i):
		if i%2==0:
			return range(l)
		return range(l-1,-1,-1)
	gridSize = nOfSamplesPerDimension**2
	scale=1
	xyz0 = get_xyz0(xy0_ranges)
	def U1(xyz1):
		if len(xyz1.shape) == 2:#list of points instead of a grid of points?
			xyz1 = np.reshape(xyz1, (xyz1.shape[0],1,xyz1.shape[1]))
		nonlocal xyz0, scale, xy0_ranges
		newRange = xy0_ranges
		integral = np.zeros(xyz1.shape[:-1], dtype=np.complex128)
		plotGradiens=False
		gradLim =np.pi/2
		for i in range(len(xyz1)):
			for j in alternateRange(len(xyz1[i]),i):
				if i==2 and j==9:
					i += 0
				newRange = xy0_ranges
				xyz0 = get_xyz0(newRange)
				repeats = 10
				# newRange = ((newRange[0][0]-xyz1[i, j, 0],newRange[0][1]-xyz1[i, j, 0]),(newRange[1][0]-xyz1[i, j, 1],newRange[1][1]-xyz1[i, j, 1]))
				# newRange = ((newRange[0][0]+xyz1[i, j, 0],newRange[0][1]+xyz1[i, j, 0]),(newRange[1][0]+xyz1[i, j, 1],newRange[1][1]+xyz1[i, j, 1]))
				foundOneGoodRange = False
				foundOneSmallRange = False
				while repeats>0:
					allImpulseResponses, validSections = impulseExpansionInFreeSpace_separated(xyz1[i, j, :], U0, xyz0, k)
					if plotGradiens:
						plt.imshow(validSections//gradLim)
						'''
						plotGradiens=True
						plt.imshow(validSections/gradLim)
						'''
						plt.show()
					ms = MapStatus(validSections, gradLim, 0.2)
					if ms != "ok":
						if ms != "noVal":
							if ms == "tooSmall":
								if foundOneSmallRange:
									break
								foundOneSmallRange = True
							#we found some good values, but it's still not perfect
							foundOneGoodRange = True
						elif foundOneGoodRange:
							#while zooming out to get all the "valid" values, we ended up increasing the difference in values too 
							#much => we can't obtain a map with all the good values (or better, there were no good values to begin 
							#with, and we found something only because we zoomed so much). Let's just break
							break
						newRange = expandMap(validSections, gradLim, newRange, 1.5,8)
						xyz0 = get_xyz0(newRange)
					else:
						break
					repeats -= 1
				
				# newRange = ((newRange[0][0]-xyz1[i, j, 0],newRange[0][1]-xyz1[i, j, 0]),(newRange[1][0]-xyz1[i, j, 1],newRange[1][1]-xyz1[i, j, 1]))
				# newRange = ((newRange[0][0]+xyz1[i, j, 0],newRange[0][1]+xyz1[i, j, 0]),(newRange[1][0]+xyz1[i, j, 1],newRange[1][1]+xyz1[i, j, 1]))
				if repeats == 0:
					print(f"failed to converge at point ({i}, {j})")
					# newRange = xy0_ranges
					# xyz0 = get_xyz0(newRange)
					# allImpulseResponses, validSections = impulseExpansionInFreeSpace_separated(xyz1[i, j, :], U0, xyz0, k)
					# validSections = (np.logical_not(np.isnan(validSections))).astype(float)
				
				xyz0 = get_xyz0(newRange, nOfSamplesPerDimension)
				
				allImpulseResponses, validSections = impulseExpansionInFreeSpace_separated(xyz1[i, j, :], U0, xyz0, k)
				# validSections[np.isnan(validSections)] = np.inf
				# minValidSections = np.min(validSections)
				# validSections = (validSections//gradLim < (minValidSections // gradLim) + 1).astype(float)
				validSections = (validSections < gradLim).astype(float)
				integral[i][j] = np.sum(allImpulseResponses * validSections) / gridSize * np.prod(np.array([(max-min)*scale for min,max in newRange]))

		return integral * k/(2j*np.pi)
	return U1

def expandInFreeSpace(U0, xy0_ranges,z0=0, k=1):
	'''
	returns the convolution U1(x1,y1,z1) of the field U0(x,y,z=z0) with h(x,y,z) to obtain the value of U1(x,y,z)
	the integral is calculated in the specified ranges
	'''
	nOfSamplesPerDimension = 100
	X,Y = np.meshgrid(*[np.linspace(min,max,nOfSamplesPerDimension) for min,max in xy0_ranges])
	X=X.flatten()[None,:]
	Y=Y.flatten()[None,:]
	Z=z0 * np.ones_like(X)
	xyz0 = np.stack((X,Y,Z), axis=2)
	def U1(xyz1):
		s=xyz1.shape
		xyz1 = np.reshape(xyz1, (-1, 1, 3))
		xyz0_mod = xyz0# + np.stack((xyz1[:,:,0],xyz1[:,:,1],np.zeros_like(xyz1[:,:,0])), axis=2)
		allImpulseResponses = impulseExpansionInFreeSpace(xyz1, U0, xyz0_mod , k)# * (xyz0[...,0]**2 + xyz0[...,1]**2 < xy0_ranges[0][1]*xy0_ranges[1][1]).astype(float)

		integral = np.sum(allImpulseResponses, axis=1)
		integral *= np.prod(np.array([(max-min) / nOfSamplesPerDimension for min,max in xy0_ranges]))
		integral = np.reshape(integral, s[:-1]) * k/(2j*np.pi)
		return integral
	return U1

def passThroughLens_separated(U0, k,f,R):
	'''returns the function U1(x,y) after a lens of focal length f'''
	def U1(xy):
		ru, au = U0(xy)
		al = -.5*k/f*(xy[...,0]**2 + xy[...,1]**2)
		outside = (xy[...,0]**2 + xy[...,1]**2 > R**2)
		a = (au+al)
		a[outside] = np.nan
		ru[outside] = 0
		return ru, a
	return U1

def passThroughLens(U0, k,f,R):
	'''returns the function U1(x,y) after a lens of focal length f'''
	def U1(xy):
		return U0(xy) * np.exp(-.5j*k/f*(xy[...,0]**2 + xy[...,1]**2)) * (xy[...,0]**2 + xy[...,1]**2 < R**2).astype(float)
	return U1

def addStaticZ(function, z):
	def f(xy):
		return function(np.stack((xy[...,0],xy[...,1],z*np.ones(xy.shape[:-1])),axis=-1))
	return f
def addStaticY(function, y):
	def f(xz):
		return function(np.stack((xz[...,0],y*np.ones(xz.shape[:-1]),xz[...,1]),axis=-1))
	return f
def addStaticYZ(function, y, z):
	def f(x):
		return function(np.stack((x,y*np.ones(x.shape),z*np.ones(x.shape)),axis=-1))
	return f
def toComplex(a):
	def u(*x):
		q=a(*x)
		return q[0]*np.exp(q[1]*1j)
	return u

def gridFunction(startingFunction, ranges, z, nOfValuesPerDimension = None, returnGrid = False, gridGeneration=np.linspace):
	if nOfValuesPerDimension is None:
		nOfValuesPerDimension = [50] * len(ranges)
	grids = [gridGeneration(ranges[i][0], ranges[i][1], nOfValuesPerDimension[i]) for i in range(len(ranges))]
	meshed_grids = np.meshgrid(*grids)
	grid_points = np.stack((*meshed_grids, np.ones_like(meshed_grids[0])*z), axis=-1)

	values = startingFunction(grid_points)
	interpolator = RegularGridInterpolator([grid for grid in grids], values)
	if returnGrid:
		return interpolator, values
	return interpolator

def gridFunction_separated(startingFunction, ranges, z, nOfValuesPerDimension = None, returnGrid = False, gridGeneration=np.linspace):
	if nOfValuesPerDimension is None:
		nOfValuesPerDimension = [50] * len(ranges)
	grids = [gridGeneration(ranges[i][0], ranges[i][1], nOfValuesPerDimension[i]) for i in range(len(ranges))]
	meshed_grids = np.meshgrid(*grids)
	grid_points = np.stack((*meshed_grids, np.ones_like(meshed_grids[0])*z), axis=-1)

	r,a = startingFunction(grid_points)
	r_interpolator = RegularGridInterpolator([grid for grid in grids], r)
	a_interpolator = RegularGridInterpolator([grid for grid in grids], a)
	if returnGrid:
		return (r_interpolator, a_interpolator), (r,a)
	return (r_interpolator, a_interpolator)

f0 = 25.5e-3
pow = 1
R0 = 16e-3
f1 = 200e-3
R1 = 27e-3
lam = 399e-9#1e-6#f0/3#399e-9
k=2*np.pi/lam

z0=0#f0
z1=z0+f1
z2=z1+f1
effectiveR_lens0=R0
effectiveR_lens1=R1

def U0_separated(k):
	def U0(xyz):
		r=np.linalg.norm(xyz, axis=-1)
		cosAlpha = xyz[...,2]/r	
		return cosAlpha / r, k*r
	return U0

def dipoleField_separated(k):
	def U0(xyz):
		r=np.linalg.norm(xyz, axis=-1)
		theta = np.arcsin(xyz[...,1]/r)
		cosAlpha = xyz[...,2]/r
		return (1+np.cos(2*theta))*cosAlpha / r, k*r
	return U0

def gaussian_beam_separated(w0, k):
	"""
	gaussian beam propagating in axis z and focus at z=0
	"""
	lambda_ =  2 * np.pi / k
	zR = np.pi * w0**2 / lambda_
	def U0(xyz):
		r = np.linalg.norm(xyz[...,:2], axis=-1)
		z = xyz[...,2]
		wz = w0 * np.sqrt(1 + (z / zR)**2)
		Rz = z * (1 + (zR / z)**2)
		phi = np.arctan(z / zR)

		amplitude = (w0 / wz) * np.exp(-(r**2) / wz**2)
		phase = -(k * z + k * (r**2) / (2 * Rz) - phi)
		phase = np.nan_to_num(phase, nan=0)
		return amplitude, phase
	return U0

'''------------------------------------------------------------------------------------------'''
# U0=dipoleField_separated(R0/10,k)
# U_afterLens0=passThroughLens_separated(U0, k, f0, R0)
# # plot2D_function(addStaticY(toComplex(U0),0), [-effectiveR_lens0,effectiveR_lens0],[0,z0],50,50, "U_beforeLens1")
# # plot2D_function(addStaticZ(toComplex(U_afterLens0),z0), [-effectiveR_lens0,effectiveR_lens0],[-effectiveR_lens0,effectiveR_lens0],50,50, "U_afterLens0")
# # U_afterLens=U0#lambda xyz:impulseExpansionInFreeSpace_separated(xyz,U0, np.array([0,0,0]), k)

# U_beforeLens1=expandInFreeSpace_separated(U_afterLens0, 
# 							[[-effectiveR_lens0,effectiveR_lens0],[-effectiveR_lens0,effectiveR_lens0]], z0, k)
# plot2D_function(addStaticY(U_beforeLens1,0), [0,R0],[z0+f1*.5,z0+f1*1.5],50,50, "U_beforeLens1_xz")
# # plot1D_function(addStaticYZ(U_beforeLens1,0,z0+f0+lam*50), [0,1e-5],100, "U_beforeLens1_xz")
# # plot1D_function(addStaticYZ(U_beforeLens1,0,z1), [0,R0*.001],100, "U_beforeLens1_xz")
# # plot2D_function(addStaticZ(U_beforeLens1,z1), [-0,.006],[-0,.006],20,20, "U_beforeLens1_xy")
# # plot2D_function(addStaticZ(U_beforeLens1,z1), [-0,R0*.36],[-0,R0*.36],100,100, "U_beforeLens1_xy")
# # plot2D_function(addStaticZ(U_beforeLens1,z1), [-0,R0*1],[-0,R0*1],100,10, "U_beforeLens1_xy")


'''------------------------------save images------------------------------------------------------------'''
# for i in range(4):
# 	z0=f0-5e-8*(i)
# 	z1=z0+f1
# 	z2=z1+f1
# 	U_beforeLens0 = dipoleField_separated(k)
# 	U_afterLens0 = passThroughLens_separated(U_beforeLens0, k, f0, R0)
# 	U_beforeLens1 = expandInFreeSpace_separated(U_afterLens0, 
# 								[[-R0,R0],[-R0,R0]], z0, k)
# 	effectiveR_lens1=0.0055
# 	x, y=np.meshgrid(*[np.linspace(0,effectiveR_lens1,400) for i in range(2)])
# 	xy=np.stack((x,y), axis=2)
# 	# '''
# 	fieldBeforeLens1 = addStaticZ(U_beforeLens1, z1)(xy)
# 	save_h5_image(f"fieldBeforeLens1_{z0}_{effectiveR_lens1}.h5", fieldBeforeLens1, f0=f0,ray=np.max(xy))
# 	'''
# 	fieldBeforeLens1 = load_h5_image(f"fieldBeforeLens1_{f0}_{effectiveR_lens1}.h5")
# 	#'''
# 	fieldBeforeLens1 = np.concatenate((np.flip(fieldBeforeLens1[1:,:], axis=(0)), fieldBeforeLens1), axis=0)
# 	fieldBeforeLens1 = np.concatenate((np.flip(fieldBeforeLens1[:,1:], axis=(1)), fieldBeforeLens1), axis=1)
# 	# plot2D(fieldBeforeLens1,[-effectiveR_lens1,effectiveR_lens1],[-effectiveR_lens1,effectiveR_lens1], "fieldBeforeLens1")
# 	objective_Ray = fieldBeforeLens1.shape[0] * lam * f1 / effectiveR_lens1 / 2
# 	U_objective = fftshift(fft2(fieldBeforeLens1))
# 	# plot2D(U_objective,[-objective_Ray,objective_Ray],[-objective_Ray,objective_Ray], "fieldBeforeLens1")
# a=0
# U_afterLens1=passThroughLens_separated(U_beforeLens1, k, f1, R1)

# U_objective = expandInFreeSpace_separated(U_afterLens1, 
# 							[[-effectiveR_lens1,effectiveR_lens1],[-effectiveR_lens1,effectiveR_lens1]], z0, k)

# plot2D_function(addStaticY(U_objective,0), [0,effectiveR_lens0/10],[z2-f0*.9,z2+f0*.9],51,51, "U_objective")


'''-----------------------------------test with gaussian beam-------------------------------------------------------'''
w0=100e-6
f=50e-3
R=10e-3
zlens=100e-3
lam=830e-9
k=2*np.pi/lam

U0=gaussian_beam_separated(w0,k)
plot2D_function(addStaticY(toComplex(U0),0), [-0,.4e-3],[0,zlens],51,51, "U_beforeLens1")

U_afterLens0=passThroughLens_separated(U0, k, f, R)
# plot2D_function(addStaticZ(toComplex(U_afterLens0),z0), [-R,R],[-R,R],50,50, "U_afterLens0")

U_final = expandInFreeSpace_separated(U_afterLens0,[[-R,R],[-R,R]],zlens,k)
plot2D_function(addStaticY(U_final,0), [0,.4e-3],[120e-3,200e-3],50,50, "U_beforeLens1_xz")