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
from joblib import Parallel, delayed

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
		ray,phase=a(*x)
		phase = np.nan_to_num(phase, nan=0)
		return ray*np.exp(phase*1j)
	return u
def toRayAngle(a):
	def u(*x):
		q=a(*x)
		return np.abs(q), np.angle(q)
	return u

'''-------------------------------------------------basic fields---------------------------------------------------------'''

def pointField(k):
	'''
	field of a particle in (0,0,0)
	'''
	def U0(xyz):
		'''
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

def planeWave(k):
	'''
	planar wave travelling along z
	'''
	def U0(xyz):
		'''
		r=np.linalg.norm(xyz, axis=-1)
		cosAlpha = xyz[...,2]/r	
		return k / (2j*np.pi) * cosAlpha / r, k*r
		'''
		return np.ones(xyz.shape[:-1]), k*xyz[...,2]
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
	n=1.8
	R1=f/(n-1)
	#R2=np.inf
	def U1(xy):
		ru, au = U0(xy)
		r = np.linalg.norm(xy[...,:2], axis=-1)
		'''
		d = R1*(1-np.cos(np.arcsin(r/R1)))
		al = - k * (n-1) * d
		'''
		al = -.5*k/f*(r**2)
		#'''
		outside = (r**2 > R**2)
		a = (au+al)
		a[outside] = np.nan
		ru[outside] = 0
		return ru, a
	return U1

'''-------------------------------------------------expansion formulas---------------------------------------------------------'''
def h(xyz, k):
	# '''
	return k/(2j*np.pi*xyz[...,2]), k*xyz[...,2] + .5*k/xyz[...,2]*(xyz[...,0]**2+xyz[...,1]**2)
	'''
	r=np.linalg.norm(xyz,axis=-1)
	cosTheta=xyz[...,2]/r
	return cosTheta/r**2 * k / (2j*np.pi), k*r
	# z=xyz[...,2]
	# return -1/(2*np.pi)*z/r**2*(1j*k-1/r), k*r
	#'''
def impulseExpansionInFreeSpace(xyz1, U0, xyz0, k):
	rh, ah = h(xyz1-xyz0, k)
	ru, au = U0(xyz0)
	r=ru*rh
	a=au+ah
	return r, a


def expandInFreeSpace_parallel(U0, xy0_ranges,z0=0, k=1):
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
		# Parallelize the outer loop using joblib

		def compute_integral(i):
			nonlocal xyz1
			print(i)
			row_integral = np.zeros(xyz1.shape[1], dtype=np.complex128)
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
				zoomedMaps = startMap.getZoomedFlatSections(np.pi*8,.3)
				integ = 0
				if len(zoomedMaps) != 0:
					zoomedMaps = zoomedMaps
				for map in zoomedMaps:
					map.f = getImpulseExpansion
					map.resolution = np.repeat(nOfSamplesPerDimension,2)
					integ += map.integral()
				row_integral[j] = integ
			return row_integral

		results = Parallel(n_jobs=-1, prefer="threads")(delayed(compute_integral)(i) for i in range(len(xyz1)))

		return np.array(results)
	return U1

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

if __name__ == "__main__":
	'''-----------------------------------test with gaussian beam-------------------------------------------------------'''
	# w0=50e-6
	# f=70e-3
	# R=w0*20
	# zlens=150e-3
	# lam=830e-9
	# k=2*np.pi/lam

	# U0=gaussian_beam(w0,k)
	# # plot2D_function(addStaticY(toComplex(U0),0), [-0,w0*2],[0,zlens],51,51, "U_beforeLens")

	# U_afterLens0=passThroughLens(U0, k, f, R)
	# # plot2D_function(addStaticZ(toComplex(U_afterLens0),z0), [-.2e-3,.2e-3],[-.2e-3,.2e-3],50,50, "U_afterLens")

	# U_final = expandInFreeSpace(U_afterLens0,[[-R,R],[-R,R]],zlens,k)
	# plot2D_function(addStaticY(U_final,0), [0,w0],[200e-3,300e-3],10,20, "U_afterLens")


	# U_final = expandInFreeSpace(U0,[[-R,R],[-R,R]],-200e-3,k)
	# plot2D_function(addStaticY(toComplex(U0),0), [-0,w0*2],[-50e-3,200e-3],20,20, "U_theoretical")
	# plot2D_function(addStaticY(U_final,0), [0,w0*2],[-50e-3,200e-3],20,20, "U_calculated")


	'''-----------------------------------double lens-------------------------------------------------------'''

	f0 = 25.5e-3
	pow = 1
	R0 = 16e-3
	f1 = 200e-3
	R1 = 27e-3
	lam = 399e-9#1e-6#f0/3#399e-9
	k=2*np.pi/lam

	pixelSize = 4.6e-6
	requestedRange = pixelSize * 16
	resolution_beforeLens1 = 50
	resolution_objective = 150
	range_beforeLens1 = 2*R1#15e-3 * 0.07
	rangeForFft = resolution_objective * lam * f1 / requestedRange

	metadata = {
		"f0" 						: f0,
		"pow" 						: pow,
		"R0" 						: R0,
		"f1" 						: f1,
		"R1" 						: R1,
		"lam" 						: lam,
		"k" 						: k,
		"pixelSize"					: pixelSize,
		"requestedRange"			: requestedRange,
		"resolution_objective"		: resolution_objective,
		"resolution_beforeLens1"	: resolution_beforeLens1,
		"range_beforeLens1"			: range_beforeLens1,
		"rangeForFft"				: rangeForFft,
	}
	beforeLens_path = "D:/simulationImages/blurs/noApprox_399nm/beforeLens/"
	objective_path = "D:/simulationImages/blurs/noApprox_399nm/largerBlur/"

	if not os.path.exists(beforeLens_path):
		os.makedirs(beforeLens_path)
	if not os.path.exists(objective_path):
		os.makedirs(objective_path)

	zAtom = np.linspace(0, 8e-6, 3)
	contour = np.zeros((len(zAtom), resolution_objective))
	for i,z in enumerate(zAtom):
		
		beforLens_file_name = f"{beforeLens_path}beforeLens1_atomZ={z:.2e}.h5"
		if not os.path.exists(beforLens_file_name):
			metadata["zLens0"] = zLens0=f0+z
			metadata["zLens1"] = zLens1=zLens0+600e-3#f0+f1
			metadata["zObjective"] = zObjective=zLens1+f1

			U_beforeLens0 = pointField(k)#gaussian_beam(2e-3,k)
			U_afterLens0 = passThroughLens(U_beforeLens0, k, f0, R0)
			# plot2D_function(addStaticY(toComplex(U_beforeLens0),0),[0,R0*1.1],[f0*.5,f0],40,40, "U_beforeLens0_xz")
			# plot2D_function(addStaticZ(toComplex(U_afterLens0),zLens0),[-R0*1.1,R0*1.1],[-R0*1.1,R0*1.1],800,800, "U_afterLens0_xz")

			U_beforeLens1 = expandInFreeSpace(U_afterLens0,[[-R0,R0],[-R0,R0]],zLens0,k)
			plot2D_function(addStaticY(U_beforeLens1,0), [0,R1*1.1],[zLens0 + .01*f0, zLens0 + 1.5*f1],20,20, "U_beforeLens1_xz")
			# plot2D_function(addStaticZ(U_beforeLens1,zLens1), [0,R1*1.1],[0,R1*1.1],20,20, "U_beforeLens1_xy")
			
			mappedFun_beforeLens1 = mappedFunction(addStaticZ(U_beforeLens1,zLens1), np.repeat(range_beforeLens1/2, 2), np.repeat(range_beforeLens1, 2), np.repeat(resolution_beforeLens1,2))
			U_lens1Block = passThroughLens(toRayAngle(mappedFun_beforeLens1), 0,1,R1)#just to cut the field outside of the lens
			U_beforeLens1_calculated = toComplex(U_lens1Block)(mappedFun_beforeLens1.getXY())
			plot2D(mirrorSymmetricImage(U_beforeLens1_calculated),[-range_beforeLens1/2,range_beforeLens1/2],[-range_beforeLens1/2,range_beforeLens1/2], "fieldAtObjective")
			a=0
			a[2]=3
			save_h5_image(beforLens_file_name, U_beforeLens1_calculated, zAtom = z, range = range_beforeLens1, **metadata)

		objective_file_name = f"{objective_path}camera_atomZ={z:.2e}.h5"
		if not os.path.exists(objective_file_name):
			#let's get it from the file
			U_beforeLens1_calculated, loadedMetadata = load_h5_image(beforLens_file_name, returnMetadata=True)
			# U_beforeLens1_calculated = mirrorSymmetricImage(U_beforeLens1_calculated)
			Range = loadedMetadata['range']
			loadedMetadata.pop('range', None)
			loadedMetadata.pop('resolution', None)

			U_objective = fftshift(fft2_butBetter(U_beforeLens1_calculated, np.repeat(Range,2), np.repeat(requestedRange / (lam * f1), 2), transform_n=np.repeat(resolution_objective,2)))
			
			save_h5_image(objective_file_name, U_objective, range = requestedRange, resolution = resolution_objective, **loadedMetadata)
		U_objective = load_h5_image(objective_file_name)

		contour[i] = np.abs(U_objective[int(resolution_objective/2),:])

		# objective_Ray = resolution[0] * lam * f1 / Range[0] / 2
		# plot2D(U_objective,[-requestedRange/2,requestedRange/2],[-requestedRange/2,requestedRange/2], "fieldAtObjective")


	plot2D(contour,[-requestedRange/2,requestedRange/2],[zAtom.min(),zAtom.max()], "fieldAtObjective")
	# plot2D(np.log10(contour),[-requestedRange/2,requestedRange/2],[-2e-6,2e-6], "fieldAtObjective")


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
	# 	plot2D_function(addStaticY(toComplex(U_beforeLens0),0),[0,R0*1.1],[f0*.5,f0],40,40, "U_beforeLens0_xz")
	# 	plot2D_function(addStaticZ(toComplex(U_afterLens0),zLens0),[-R0*1.1,R0*1.1],[-R0*1.1,R0*1.1],800,800, "U_beforeLens0_xz")

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

	a=0