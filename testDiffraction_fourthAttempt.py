# import Simulations_Libraries.trajectory_library as trajlib
import numpy as np
import matplotlib.pyplot as plt
from Camera import *
from scipy.stats import poisson
from scipy.optimize import curve_fit
# import Simulations_Libraries.general_library as genlib
from scipy.interpolate import RegularGridInterpolator
from numpy.fft import fft2, fftshift
from testDiffraction_thirdAttempt import mappedFunction
import os

def img(*x):
	plt.imshow(*x)
	plt.show()

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

'''-------------------------------------------------basic fields---------------------------------------------------------'''

def pointField(k):
	'''
	field of a particle in (0,0,0)
	'''
	def U0(xyz):
		# '''
		r=np.linalg.norm(xyz, axis=-1)
		cosAlpha = xyz[...,2]/r	
		return k / (2j*np.pi) * cosAlpha / r, k*r
		'''
		return h(xyz,k)
		#'''
	return U0

def dipoleField(k):
	'''
	field of a dipole in (0,0,0), oriented in the y axis
	'''
	def U0(xyz):
		r=np.linalg.norm(xyz, axis=-1)
		theta = np.arcsin(xyz[...,1]/r)
		'''
		cosAlpha = xyz[...,2]/r
		return k / (2j*np.pi) * (1+np.cos(2*theta)) * cosAlpha / r, k*r		
		'''
		R,A=h(xyz,k)
		return (1+np.cos(2*theta)) * R, A
		#'''
	return U0

def gaussian_beam(w0, k):
	"""
	gaussian beam propagating in axis z and focus in (0,0,0)
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
		phase = (k * z + k * (r**2) / (2 * Rz) - phi)
		phase = np.nan_to_num(phase, nan=0)
		return amplitude, phase
	return U0


def passThroughLens(U0, k,f,R):
	'''
	returns the function U1(x,y) after a lens of focal length f
	'''
	def U1(xy):
		ru, au = U0(xy)
		al = -.5*k/f*(xy[...,0]**2 + xy[...,1]**2)
		outside = (xy[...,0]**2 + xy[...,1]**2 > R**2)
		a = (au+al)
		a[outside] = np.nan
		ru[outside] = 0
		return ru, a
	return U1

'''-------------------------------------------------expansion formulas---------------------------------------------------------'''
def h(xyz, k):
	#'''
	return k/(2j*np.pi*xyz[...,2]), k*xyz[...,2] + .5*k/xyz[...,2]*(xyz[...,0]**2+xyz[...,1]**2)
	'''
	r=np.linalg.norm(xyz,axis=-1)
	cosTheta=xyz[...,2]/r
	return cosTheta/r * k / (2j*np.pi), k*r
	#'''
def impulseExpansionInFreeSpace(xyz1, U0, xyz0, k):
	rh, ah = h(xyz1-xyz0, k)
	ru, au = U0(xyz0)
	r=ru*rh
	a=au+ah
	return r, a


def expandInFreeSpace(U0, xy0_ranges,z0=0, k=1):
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
			for j in range(len(xyz1[i])):
				def getOnlyAngle(xy0):
					xyz0 = np.concatenate((xy0, z0*np.ones(xy0.shape[:-1])[...,None]), axis = -1)
					_, angle = impulseExpansionInFreeSpace(xyz1[i, j, :], U0, xyz0, k)
					return angle
				def getImpulseExpansion(xy0):
					xyz0 = np.concatenate((xy0, z0*np.ones(xy0.shape[:-1])[...,None]), axis = -1)
					ray, angle = impulseExpansionInFreeSpace(xyz1[i, j, :], U0, xyz0, k)
					angle = np.nan_to_num(angle, nan=0)
					return ray * np.exp(1j * angle)
				
				startMap = mappedFunction(getOnlyAngle, startCenter, startSize, np.repeat(nOfSamplesForMapSearch,2))
				'''
				zoomedMaps = [startMap]
				'''
				zoomedMaps = startMap.getZoomedFlatSections(np.pi*8,.3)
				#'''
				integ = 0
				if len(zoomedMaps) != 0:
					zoomedMaps = zoomedMaps
				for map in zoomedMaps:
					map.f = getImpulseExpansion
					map.resolution = np.repeat(nOfSamplesPerDimension,2)
					integ += map.integral()
				integral[i][j] = integ
		return integral
	return U1


'''-----------------------------------test with gaussian beam-------------------------------------------------------'''
w0=50e-6
f=70e-3
R=w0*20
zlens=150e-3
lam=830e-9
k=2*np.pi/lam

U0=gaussian_beam(w0,k)
# plot2D_function(addStaticY(toComplex(U0),0), [-0,w0*2],[0,zlens],51,51, "U_beforeLens")

U_afterLens0=passThroughLens(U0, k, f, R)
# plot2D_function(addStaticZ(toComplex(U_afterLens0),z0), [-.2e-3,.2e-3],[-.2e-3,.2e-3],50,50, "U_afterLens")

U_final = expandInFreeSpace(U_afterLens0,[[-R,R],[-R,R]],zlens,k)
plot2D_function(addStaticY(U_final,0), [0,w0],[200e-3,300e-3],10,20, "U_afterLens")


# U_final = expandInFreeSpace(U0,[[-R,R],[-R,R]],-200e-3,k)
# plot2D_function(addStaticY(toComplex(U0),0), [-0,w0*2],[-50e-3,200e-3],20,20, "U_theoretical")
# plot2D_function(addStaticY(U_final,0), [0,w0*2],[-50e-3,200e-3],20,20, "U_calculated")


'''-----------------------------------double lens-------------------------------------------------------'''

# f0 = 25.5e-3
# pow = 1
# R0 = 16e-3
# f1 = 200e-3
# R1 = 27e-3
# lam = 899e-9#1e-6#f0/3#399e-9
# k=2*np.pi/lam

# pixelSize = 4.6e-6
# requestedRange = pixelSize * 8
# resolution = 50
# range_beforeLens1 = resolution * lam * f1 / requestedRange
# print(requestedRange, resolution, range_beforeLens1)
# zAtom = np.linspace(-10e-6, 10e-6, 41)
# contour = np.zeros((len(zAtom), resolution))
# for i,z in enumerate(zAtom):
	
# 	file_name = f"beforeLens1_atomZ={z:.2e}.h5"
# 	if False or not os.path.exists(file_name):
# 		zLens0=f0+z
# 		zLens1=zLens0+f0+f1
# 		zObjective=zLens1+f1

# 		U_beforeLens0 = dipoleField(k)
# 		U_afterLens0 = passThroughLens(U_beforeLens0, k, f0, R0)

# 		U_beforeLens1 = expandInFreeSpace(U_afterLens0,[[-R0,R0],[-R0,R0]],zLens0,k)
# 		# plot2D_function(addStaticY(U_beforeLens1,0), [0,R0*1.1],[zLens0 + .5*f1, zLens0 + 1.5*f1],20,20, "U_beforeLens1_xz")
# 		# plot2D_function(addStaticZ(U_beforeLens1,zLens1), [0,R0*1.1],[0,R0*1.1],20,20, "U_beforeLens1_xy")

# 		mappedFun_beforeLens1 = mappedFunction(addStaticZ(U_beforeLens1,zLens1), np.array([0,0]), np.repeat(range_beforeLens1, 2), np.repeat(resolution,2))
# 		U_beforeLens1_calculated = mappedFun_beforeLens1(mappedFun_beforeLens1.getXY())
# 		# plot2D(U_beforeLens1_calculated,[-range_beforeLens1/2,range_beforeLens1/2],[-range_beforeLens1/2,range_beforeLens1/2], "fieldAtObjective")
		
# 		save_h5_image(file_name, U_beforeLens1_calculated, zAtom = z, range = mappedFun_beforeLens1.size[0], resolution = mappedFun_beforeLens1.resolution[0])

# 	#let's get it from the file
# 	U_beforeLens1_calculated, metadata = load_h5_image(file_name, returnMetadata=True)

# 	Range = metadata['range']
# 	imageResolution = metadata['resolution']

# 	U_objective = fftshift(fft2(U_beforeLens1_calculated))
# 	contour[i] = np.abs(U_objective[int(resolution/2),:])

# 	# objective_Ray = resolution[0] * lam * f1 / Range[0] / 2
# 	# plot2D(U_objective,[-requestedRange/2,requestedRange/2],[-requestedRange/2,requestedRange/2], "fieldAtObjective")

# plot2D(contour,[-requestedRange/2,requestedRange/2],[-2e-6,2e-6], "fieldAtObjective")


'''-----------------------double lens with gaussian beam----------------------'''

# w0=1e-6
# f0 = 25.5e-3
# pow = 1
# R0 = 16e-3
# f1 = 200e-3
# R1 = 27e-3
# lam = 899e-9#1e-6#f0/3#399e-9
# k=2*np.pi/lam

# requestedRange = 80e-6
# resolution = 50
# range_beforeLens1 = resolution * lam * f1 / requestedRange
# print(requestedRange, resolution, range_beforeLens1)
	
# file_name = f"gaussianBeam.h5"
# if True or not os.path.exists(file_name):
# 	zLens0=f0
# 	zLens1=zLens0+f0+f1
# 	zObjective=zLens1+f1

# 	U_beforeLens0 = gaussian_beam(w0,k)
# 	U_afterLens0 = passThroughLens(U_beforeLens0, k, f0, R0)

# 	U_beforeLens1 = expandInFreeSpace(U_afterLens0,[[-R0,R0],[-R0,R0]],zLens0,k)

# 	mappedFun_beforeLens1 = mappedFunction(addStaticZ(U_beforeLens1,zLens1), np.array([0,0]), np.repeat(range_beforeLens1, 2), np.repeat(resolution,2))
# 	U_beforeLens1_calculated = mappedFun_beforeLens1(mappedFun_beforeLens1.getXY())
# 	# plot2D(U_beforeLens1_calculated,[-range_beforeLens1/2,range_beforeLens1/2],[-range_beforeLens1/2,range_beforeLens1/2], "fieldAtObjective")
	
# 	save_h5_image(file_name, U_beforeLens1_calculated, range = mappedFun_beforeLens1.size, resolution = mappedFun_beforeLens1.resolution)

# #let's get it from the file
# U_beforeLens1_calculated, metadata = load_h5_image(file_name, returnMetadata=True)

# Range = metadata['range']
# resolution = metadata['resolution']

# U_objective = fftshift(fft2(U_beforeLens1_calculated))

# plot2D(U_objective, [-requestedRange/2,requestedRange/2], [-requestedRange/2,requestedRange/2])

# plt.plot(np.linspace(-requestedRange/2,requestedRange/2, resolution[0]),np.abs(U_objective[int(resolution[0]/2),:]))
# plt.show()

# a=0