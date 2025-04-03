# import Simulations_Libraries.trajectory_library as trajlib
import numpy as np
import matplotlib.pyplot as plt
from Camera import *
from scipy.stats import poisson
from scipy.optimize import curve_fit
# import Simulations_Libraries.general_library as genlib
from scipy.interpolate import RegularGridInterpolator


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

def impulseExpansionInFreeSpace(xyz1, U0, xyz0, k):
	return U0(xyz0) * h(xyz1-xyz0, k)

def boundingBox(map):
	''' gets the index coordinates of a rectangle that includes all the elements of map that are different from 0'''
	vals = np.where(map != 0)
	minx=np.min(vals[1])
	maxx=np.max(vals[1])
	miny=np.min(vals[0])
	maxy=np.max(vals[0])
	return ((minx, maxx), (miny, maxy))

def expandMap(map, realRanges, extensionMultiplier = 2, reductionadder = 1):
	'''returns a new search range for the estimate of the acceptable ranges of a map,
	so that the new map (generated with an external function that uses the new search range) 
	should contains more (and hopefully all the acceptable region(s?) of the function... Sorry, 
	I haven't explained myself... Oh well, hopefully I'll remember what this actually does'''
	if np.max(map) == 0:#no valid pixel?
		#Let's assume that there will be a valid value in the center, and just return the size of one pixel centered in the map
		pixelHalfSizes = [(max-min)*reductionadder*.5/map.shape[i] for i, (min,max) in enumerate(realRanges)]
		mapCenter = [(max+min)/2 for (min,max) in realRanges]
		return ((mapCenter[0]-pixelHalfSizes[0],mapCenter[0]+pixelHalfSizes[0]),(mapCenter[1]-pixelHalfSizes[1],mapCenter[1]+pixelHalfSizes[1]))
		
	((minx, maxx), (miny, maxy)) = boundingBox(map)
	centerX=(minx+maxx)/2
	centerY=(miny+maxy)/2
	indexes = [minx, maxx, miny, maxy]
	signs = [-1,1,-1,1]
	centers =[centerX,centerX,centerY,centerY]
	limits = [0,len(map)-1,0,len(map[0])-1]
	minValues = [realRanges[0][0], realRanges[1][0]]
	pixelSizes = [(max-min)/map.shape[i] for i, (min,max) in enumerate(realRanges)]
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
		
def isMapOk(map,fillRatio):
	if np.max(map) == 0:
		return False
	((minx, maxx), (miny, maxy)) = boundingBox(map)
	limits = np.array([0,len(map)-1,0,len(map[0])-1])
	indexes = np.array([minx, maxx, miny, maxy])
	if np.any(limits==indexes):
		return False
	validRatio = np.sum(map) / map.size
	return validRatio > fillRatio


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
	grad_x, grad_y = np.gradient(a, axis=(0, 1))
	gradLim =np.pi/8
	grad = np.sqrt(grad_x**2+grad_y**2)
	plt.imshow(grad // gradLim)
	plt.show()
	return r * np.exp(1j * a), (np.abs(grad) < (gradLim)).astype(int)

def expandInFreeSpace_separated(U0, xy0_ranges,z0=0, k=1):
	'''
	returns the convolution U1(x1,y1,z1) of the field U0(x,y,z=z0) with h(x,y,z) to obtain the value of U1(x,y,z)
	the integral is calculated in the specified ranges
	'''
	nOfSamplesPerDimension = 100
	def get_xyz0(ranges):
		X,Y = np.meshgrid(*[np.linspace(min,max,nOfSamplesPerDimension)for (min,max) in ranges])
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
		nonlocal xyz0, scale, xy0_ranges
		newRange = xy0_ranges
		integral = np.zeros(xyz1.shape[:-1], dtype=np.complex128)
		for i in range(len(xyz1)):
			for j in alternateRange(len(xyz1[i]),i):
				if i==1:
					i += 0
				repeats = 10
				# newRange = ((newRange[0][0]-xyz1[i, j, 0],newRange[0][1]-xyz1[i, j, 0]),(newRange[1][0]-xyz1[i, j, 1],newRange[1][1]-xyz1[i, j, 1]))
				# newRange = ((newRange[0][0]+xyz1[i, j, 0],newRange[0][1]+xyz1[i, j, 0]),(newRange[1][0]+xyz1[i, j, 1],newRange[1][1]+xyz1[i, j, 1]))
				while repeats>0:
					allImpulseResponses, validSections = impulseExpansionInFreeSpace_separated(xyz1[i, j, :], U0, xyz0, k)
					
					if not isMapOk(validSections, 0.3):
						newRange = expandMap(validSections, newRange, 1.1,4)
						xyz0 = get_xyz0(newRange)
					else:
						break
					repeats -= 1
				
				# newRange = ((newRange[0][0]-xyz1[i, j, 0],newRange[0][1]-xyz1[i, j, 0]),(newRange[1][0]-xyz1[i, j, 1],newRange[1][1]-xyz1[i, j, 1]))
				# newRange = ((newRange[0][0]+xyz1[i, j, 0],newRange[0][1]+xyz1[i, j, 0]),(newRange[1][0]+xyz1[i, j, 1],newRange[1][1]+xyz1[i, j, 1]))
				if repeats == 0:
					print("failed to converge")
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
		rl = (xy[...,0]**2 + xy[...,1]**2 < R**2).astype(float)
		return ru*rl, au+al
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

z0=f0+50
z1=z0+f1
z2=z1+f1
effectiveR_lens0=R0/10
effectiveR_lens1=R1

def U0_separated(k):
	def U0(xyz):
		r=np.linalg.norm(xyz, axis=-1)
		cosAlpha = xyz[...,2]/r	
		return cosAlpha / r**2, k*r
	return U0


U0=U0_separated(k)
U_afterLens0=passThroughLens_separated(U0, k, f0, R0)
# plot2D_function(addStaticY(toComplex(U0),0), [-effectiveR_lens0,effectiveR_lens0],[0,z0],50,50, "U_beforeLens1")
# plot2D_function(addStaticZ(toComplex(U_afterLens0),z0), [-effectiveR_lens0,effectiveR_lens0],[-effectiveR_lens0,effectiveR_lens0],50,50, "U_afterLens0")
# U_afterLens=U0#lambda xyz:impulseExpansionInFreeSpace_separated(xyz,U0, np.array([0,0,0]), k)

U_beforeLens1=expandInFreeSpace_separated(U_afterLens0, 
							[[-effectiveR_lens0,effectiveR_lens0],[-effectiveR_lens0,effectiveR_lens0]], z0, k)
plot2D_function(addStaticY(U_beforeLens1,0), [-effectiveR_lens0,effectiveR_lens0],[z0+f0*.9,z0+f0*1.1],50,50, "U_beforeLens1_xz")
# plot2D_function(addStaticZ(U_beforeLens1,z0), [-effectiveR_lens0,effectiveR_lens0],[-effectiveR_lens0,effectiveR_lens0],50,50, "U_beforeLens1_xy")
# U_afterLens1=passThroughLens_separated(U_beforeLens1, k, f1, R1)

# U_objective = expandInFreeSpace_separated(U_afterLens1, 
# 							[[-effectiveR_lens1,effectiveR_lens1],[-effectiveR_lens1,effectiveR_lens1]], z0, k)

# plot2D_function(addStaticY(U_objective,0), [0,effectiveR_lens0/10],[z2-f0*.9,z2+f0*.9],51,51, "U_objective")

