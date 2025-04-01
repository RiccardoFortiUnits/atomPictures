# import Simulations_Libraries.trajectory_library as trajlib
import numpy as np
import matplotlib.pyplot as plt
from Camera import *
from scipy.stats import poisson
from scipy.optimize import curve_fit
# import Simulations_Libraries.general_library as genlib
import pickle
from scipy.interpolate import RegularGridInterpolator

def basicBlur(r,z, k,E,f,R):
	s = np.shape(r)
	r=r.flatten()
	z=z.flatten()
	E=E.flatten()
	def functionToIntegrate(x,r,z):
		return E[:,None]*x*j0(k*r*x/f)*np.exp(-0.5j*k*z*(x/f)**2)
	x = np.repeat(np.linspace(0,R,1000)[None,:],len(r), axis=0)
	integral = np.sum(functionToIntegrate(x, r[:,None],z[:,None]), axis=1) * R/x.shape[1]
	U=1#  k / (2 * np.pi * f) * ...
	retVal = np.abs(U * np.exp(-1j * k * z) * integral)**2
	return np.log(retVal.reshape(s))

def basicBlur_carthesian(x,y,z, k,f,R,E):
	s = np.shape(x)
	x=np.reshape(x,(-1,1,1))
	y=np.reshape(y,(-1,1,1))
	z=np.reshape(z,(-1,1,1))
	# E=np.reshape(E,(1,es[0],es[1]))
	# E=E.flatten()
	def functionToIntegrate(r,a,x,y,z):
		xi=r*np.cos(a)
		nu=r*np.sin(a)
		return E(xi,nu)*r*np.exp(-0.5j*k*z*((xi**2+nu**2)/f)**2 - 1j*k*(x*xi+y*nu)/f)
	r=np.linspace(0,R)
	a=np.linspace(0,2*np.pi,endpoint=False)
	r,a=np.meshgrid(r,a)
	r=np.reshape(r,(1,-1,1))
	a=np.reshape(a,(1,1,-1))
	integral = np.sum(functionToIntegrate(r,a,x,y,z), axis=(1,2)) * 2*np.pi * (R[-1])/np.size(E)
	U=1#  k / (2 * np.pi * f) * ...
	retVal = np.abs(U * np.exp(-1j * k * z) * integral)**2
	return np.log(retVal.reshape(s))

def onePhotonBlur(blurPosition, photonStart, photonDirection, k,f=None):
	s = np.shape(blurPosition)
	blurPosition=np.reshape(blurPosition, (-1,3))
	blurDirections = blurPosition - photonStart
	r01=np.linalg.norm(blurDirections, axis=1)
	cosTheta = np.sum(blurDirections / r01[:,None] * photonDirection,axis=1)
	U=1# k / (2 * np.pi) * ...
	retVal =  U * np.exp(1j * k * r01) / r01 * cosTheta
	return np.abs(retVal.reshape(s[:-1]))**2


# lam = 556e-9
# pow = 1
# waist = 1e-3
# f = 25.5e-3
# R = 16e-3
# rr=1e-6
# Z=1e-8
# k=2*np.pi/lam

# l= trajlib.Laser(0,0,lam,pow,waist)
# rr=np.linspace(0,R*2)

# plot2D_function(partial(basicBlur,k=k,E0=pow,f=f,w=waist,R=R), [0,R*2], [-1e-1,1e-1],100,100, "partial")

# a=0.6
# l=lambda x,y : onePhotonBlur(np.stack((np.repeat(np.repeat(f,np.shape(x)[0],axis=0)[:,None],np.shape(x)[1],axis=1)[:,:],x[:,:],y[:,:]), axis=2), np.array([0,0,0]), np.array([np.cos(a),np.sin(a),0]), k)
# plot2D_function(l, [-R,R],[-R,R],100,100, "l")

# a=0
# l=lambda r,z : basicBlur(r*rr/R,z+f,k,
# 	onePhotonBlur(np.stack((z,r,np.zeros_like(r)), axis=2), np.array([0,0,0]), np.array([np.cos(a),np.sin(a),0]), k),
# 	f,R)
# plot2D_function(l, [0,R],[-Z,Z],100,100, "l")


# E=lambda x,y : dipoleField(np.stack((x,y,f*np.ones_like(x)), axis = len(x.size)))

# a=0.6
# l=lambda x,y : onePhotonBlur(np.stack((np.repeat(np.repeat(f,np.shape(x)[0],axis=0)[:,None],np.shape(x)[1],axis=1)[:,:],x[:,:],y[:,:]), axis=2), np.array([0,0,0]), np.array([np.cos(a),np.sin(a),0]), k)
# plot2D_function(l, [-R,R],[-R,R],100,100, "l")

def toCarthesian(r,a,z):
	return (r*np.cos(a), r*np.sin(a), z)
def toCylindrical(x,y,z):
	return (np.linalg.norm(np.stack((x,y), axis=len(x.shape))), np.arctan2(y,x), z)

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

def impulseExpansionInFreeSpace(xyz1, U0, xyz0, k):
	return U0(xyz0) * h(xyz1-xyz0, k)
def quadspace(min, max, num=50, endpoint=True, retstep=False, dtype=None, axis=0):
	if min*max < 0:
		q = -min/(max-min)#min has to be negative
		negativeNum = np.maximum(1,int(np.floor(q*num)))
		positiveNum = num - negativeNum#also contains 0
		negativeLin = -(np.linspace(-np.sqrt(-min),0,negativeNum, endpoint=False, retstep=retstep, dtype=dtype, axis=axis))**2
		positiveLin =np.linspace(0, np.sqrt(max),positiveNum, endpoint, retstep, dtype, axis)**2
		return np.concatenate((negativeLin, positiveLin), axis=axis)
	return np.sqrt(np.linspace(min**2,max**2,num, endpoint, retstep, dtype, axis))
def expandInFreeSpace(U0, xy0_ranges,z0=0, k=1):
	'''
	returns the convolution U1(x1,y1,z1) of the field U0(x,y,z=z0) with h(x,y,z) to obtain the value of U1(x,y,z)
	the integral is calculated in the specified ranges
	'''
	nOfSamplesPerDimension = 20
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

# def expandInFreeSpace_cilinder(U0, r0_ranges,z0=0, k=1, ):
# 	'''
# 	returns the convolution U1(r1,a1,z1) of the field U0(r,a,z=z0) with h(r,a,z) to obtain the value of U1(r,a,z)
# 	the integral is calculated in the specified ranges
# 	'''
# 	nOfSamplesPerDimension = 20
# 	R = np.linspace(r0_ranges[0],r0_ranges[1],nOfSamplesPerDimension, axis=1)
# 	Z=z0 * np.ones_like(R)
# 	xyz0 = np.stack((R,np.zeros_like(R),Z), axis=2)
# 	def U1(rz1):
# 		s=rz1.shape
# 		rz1 = np.reshape(xyz1, (-1, 1, 3))
# 		integral = np.sum(impulseExpansionInFreeSpace(xyz1, U0, xyz0, k) * xyz0[...,0], axis=1)
# 		integral *= (r0_ranges[1]-r0_ranges[0])
# 		integral = np.reshape(integral, s[:-1])
# 		return integral
# 	return U1

def passThroughLens(U0, k,f,R):
	'''returns the function U1(x,y) after a lens of focal length f'''
	def U1(xy):
		return U0(xy) * np.exp(-.5j*k/f*(xy[...,0]**2 + xy[...,1]**2)) * (xy[...,0]**2 + xy[...,1]**2 < R**2).astype(float)
	return U1
def passThroughLens_radial(U0, k,f,R):
	'''returns the function U1(r) after a lens of focal length f'''
	def U1(r):
		return U0(r) * np.exp(-.5j*k/f*r[...,0]**2) * (r[...,0] < R).astype(float)
	return U1
def passThroughRealLens(U0, k,f,R):
	'''returns the function U1(x,y) after a lens of focal length f'''
	def U1(xy):
		r=np.linalg.norm(xy[...,0:2], axis=-1)
		return U0(xy) * np.exp(-.5j*k/(f*(1-np.cos(np.arcsin(r/f))))) * (xy[...,0]**2 + xy[...,1]**2 < R**2).astype(float)
	return U1


def photonField(photonStart, photonDirection, k):
	'''
	wave field of a single photon with a specified oringin and direction
	(I have to check if it's actually correct)
	'''
	photonDirection = np.linalg.norm(photonDirection)
	def U1(xyz):
		s = np.shape(xyz)
		xyz=np.reshape(xyz, (-1,3))
		blurDirections = xyz - photonStart
		r01=np.linalg.norm(blurDirections, axis=1)
		cosTheta = np.sum(blurDirections / r01[:,None] * photonDirection,axis=1)
		U=1# k / (2 * np.pi) * ...
		retVal =  U * np.exp(1j * k * r01) / r01 * cosTheta
		return np.abs(retVal.reshape(s[:-1]))**2
	return U1

def gridFunction(startingFunction, ranges, nOfValuesPerDimension = None, returnGrid = False, gridGeneration=np.linspace):
	if nOfValuesPerDimension is None:
		nOfValuesPerDimension = [50] * len(ranges)
	grids = [gridGeneration(ranges[i][0], ranges[i][1], nOfValuesPerDimension[i]) for i in range(len(ranges))]
	meshed_grids = np.meshgrid(*grids, indexing='ij')
	grid_points = np.stack(meshed_grids, axis=-1)

	values = startingFunction(grid_points)
	interpolator = RegularGridInterpolator([grid for grid in grids], values)
	if returnGrid:
		return interpolator, values
	return interpolator

def U0(xyz,k):
	r=np.linalg.norm(xyz, axis=-1)
	cosAlpha = xyz[...,2]/r	
	return cosAlpha / r**2 * np.exp(1j*k*r)
def U0_radial(rz,k):
	r=np.linalg.norm(rz, axis=-1)
	cosAlpha = rz[...,1]/r	
	return cosAlpha / r**2 * np.exp(1j*k*r)


def dipoleField(xyz, k):
	r=np.linalg.norm(xyz, axis=-1)
	theta = np.arcsin(xyz[...,1]/r)
	cosAlpha = xyz[...,2]/r
	return (1+np.cos(2*theta))*cosAlpha / r**2 * np.exp(1j*k*r)

	# def blur(r,z, k,E0,f,w,R):
	#     s = np.shape(r)
	#     r=r.flatten()
	#     z=z.flatten()
	#     def functionToIntegrate(x,r,z):
	#         return x*j0(k*r*x/f)*np.exp(
	#             -(x/w)**2 - 0.5j*k*z*(x/f)**2
	#         )
	#     usedR = min(R, 4*w)
	#     x = np.repeat(np.linspace(0,usedR,1000)[None,:],len(r), axis=0)
	#     integral = np.sum(functionToIntegrate(x, r[:,None],z[:,None]), axis=1) * usedR/x.shape[1]
	#     retVal = np.abs(E0 * k / (2 * np.pi * f) * np.exp(-1j * k * z) * integral)**2
	#     return retVal.reshape(s)

def impulseExpansionInFreeSpace_radial(rz1, U0, rz0, k):
	dz=rz1[...,1]-rz0[...,1]
	r0=rz0[...,0]
	r1=rz1[...,0]
	return U0(rz0) * j0(k/dz * r0 * r1) * np.exp(.5j * k / dz* (r0**2 + r1**2))
def expandInFreeSpace_radial(U0, max_r0,z0=0, k=1):
	'''
	returns the free space expansion U1(r,z1) of the field U0(r,z=z0). For cylindrical fields
	'''	
	R = np.linspace(max_r0, 0, 100, endpoint=False)[None,:]
	Z = z0*np.ones_like(R)
	rz0=np.stack((R,Z),axis=2)
	def U1(rz1):
		s=rz1.shape
		rz1 = np.reshape(rz1, (-1, 1, 2))
		allImpulseResponses = impulseExpansionInFreeSpace_radial(rz1, U0, rz0 , k)# * (xyz0[...,0]**2 + xyz0[...,1]**2 < xy0_ranges[0][1]*xy0_ranges[1][1]).astype(float)

		integral = np.sum(R * allImpulseResponses, axis=1) * max_r0 / np.size(R)
		dz=rz1[...,1]-z0
		integral = k /(2*np.pi*dz) * integral[:,None] * np.exp(1j*k*dz)
		return np.reshape(integral, s[:-1])
	return U1

# z0=1000*f#f*.25
# z1=z0+f

# U1_beforeLens=lambda xyz:impulseExpansionInFreeSpace(xyz,U0,np.zeros_like(xyz), k)
# U1_afterLens=passThroughLens(U1_beforeLens,k,f,R)

# rr=min(R,20*(z1-z0)/(R**2*k))
# print(f"using lens ray {rr}")
# U2=expandInFreeSpace(U1_afterLens, [[-rr,rr], [-rr,rr]], z0, k)

# X = np.linspace(-R,R,1001)*.01

# xyz1=np.stack((X,np.zeros_like(X), z0*np.ones_like(X)),axis=1)
# plt.plot(X, np.real(U1_beforeLens(xyz1)))

# xyz2=np.stack((X,np.zeros_like(X), z1*np.ones_like(X)),axis=1)
# plt.plot(X, np.abs(U2(xyz2)))
# plt.show()

# def for2dPlot(r,z):
# 	xyz2=np.stack((r,np.zeros_like(r), z0+z),axis=len(r.shape))
# 	return np.log(np.abs(U2(xyz2))**2)
# plot2D_function(for2dPlot, [0,R*.02], [0.5*f,1.9*f],50,100, "for2dPlot")


lam = 399e-9
k=2*np.pi/lam
pow = 1
f0 = 25.5e-3
R0 = 16e-3
f1 = 200e-3
R1 = 27e-3
atomPosition = np.array([0,0,0])
photonDirection = np.array([0,0,1])
z_lens0 = f0+50
z_lens1 = z_lens0 + f1
z_objective = z_lens1 + f1
def addStaticZ(function, z):
	def f(xy):
		return function(np.stack((xy[...,0],xy[...,1],z*np.ones(xy.shape[:-1])),axis=-1))
	return f
def addStaticYZ(function, y, z):
	def f(x):
		return function(np.stack((x,y*np.ones(x.shape),z*np.ones(x.shape)),axis=-1))
	return f
'''radial'''
effectiveR_lens0 = R0/20#min(R0,8*(z_lens1-z_lens0)/(R0**2*k))
effectiveR_lens1 = 2e-4#min(R1,200*(z_objective-z_lens1)/(R1**2*k))
U_beforeLens0 = lambda rz:U0_radial(rz, k)
# plot2D_function(U_beforeLens0, [0,effectiveR_lens0],[0,z_lens0],500,50, "U_beforeLens0")
U_afterLens0 = passThroughLens_radial(U_beforeLens0, k, f0, R0)
# plot2D_function(U_afterLens0, [0,effectiveR_lens0],[-effectiveR_lens0,effectiveR_lens0],50,50, "U_afterLens0")
#let's calculate its values once and for all, so we won't re-calculate the integrals
U_beforeLens1 = expandInFreeSpace_radial(U_afterLens0, effectiveR_lens0, z_lens0, k)
plot2D_function(U_beforeLens1, [0,effectiveR_lens1],[z_lens0*1.01,z_lens1+f0*.5],500,50, "U_beforeLens1")
U_afterLens1 = passThroughLens_radial(U_beforeLens1, k, f1, R1)
# plot2D_function(addStaticZ(U_afterLens1,z_lens1), [-effectiveR_lens1,effectiveR_lens1],[-effectiveR_lens1,effectiveR_lens1],50,50, "U_afterLens1")

# U_objective = expandInFreeSpace(U_afterLens1, [[-effectiveR_lens1,effectiveR_lens1],[-effectiveR_lens1,effectiveR_lens1]], z_lens1, k)
# plot2D_function(addStaticZ(U_objective,z_objective), [-effectiveR_lens1,effectiveR_lens1],[-effectiveR_lens1,effectiveR_lens1],50,50, "U_objective")


#for the numerical integral: if R is too big compared to the dimensions at play, the actually important 
# part of the integrals would be only on one point of the grid. Let's reduce the range of the grid, so 
# that the actually important part can contribute to the integral
# effectiveR_lens0 = R0/50#min(R0,8*(z_lens1-z_lens0)/(R0**2*k))
# effectiveR_lens1 = 2e-4#min(R1,200*(z_objective-z_lens1)/(R1**2*k))
# print(f"used radii {effectiveR_lens0}, {effectiveR_lens1}")
# # U_beforeLens0 = photonField(atomPosition, photonDirection, k)
# # U_beforeLens0 = lambda xyz:impulseExpansionInFreeSpace(xyz,U0,np.zeros_like(xyz), k)
# U_beforeLens0 = lambda xyz:dipoleField(xyz, k)
# plot2D_function(addStaticZ(U_beforeLens0,z_lens0), [-effectiveR_lens0,effectiveR_lens0],[-effectiveR_lens0,effectiveR_lens0],50,50, "U_beforeLens0")
# U_afterLens0 = passThroughLens(U_beforeLens0, k, f0, R0)
# plot2D_function(addStaticZ(U_afterLens0,z_lens0), [-effectiveR_lens0,effectiveR_lens0],[-effectiveR_lens0,effectiveR_lens0],50,50, "U_afterLens0")
# #let's calculate its values once and for all, so we won't re-calculate the integrals
# U_beforeLens1, U_beforeLens1_grid= gridFunction(expandInFreeSpace(U_afterLens0, [[-effectiveR_lens0,effectiveR_lens0],[-effectiveR_lens0,effectiveR_lens0]], z_lens0, k), 
# 							 [[-effectiveR_lens1,effectiveR_lens1],[-effectiveR_lens1,effectiveR_lens1], [z_lens1,z_lens1]],
# 							 [100,100,1], returnGrid=True, gridGeneration=np.linspace)
# plot2D_function(addStaticZ(U_beforeLens1,z_lens1), [-effectiveR_lens1,effectiveR_lens1],[-effectiveR_lens1,effectiveR_lens1],50,50, "U_beforeLens1")
# U_afterLens1 = passThroughLens(U_beforeLens1, k, f1, R1)
# plot2D_function(addStaticZ(U_afterLens1,z_lens1), [-effectiveR_lens1,effectiveR_lens1],[-effectiveR_lens1,effectiveR_lens1],50,50, "U_afterLens1")

# U_objective = expandInFreeSpace(U_afterLens1, [[-effectiveR_lens1,effectiveR_lens1],[-effectiveR_lens1,effectiveR_lens1]], z_lens1, k)
# plot2D_function(addStaticZ(U_objective,z_objective), [-effectiveR_lens1,effectiveR_lens1],[-effectiveR_lens1,effectiveR_lens1],50,50, "U_objective")
# rrr=np.linspace(0,3e-4)
# xyz2=np.stack((rrr,np.zeros_like(rrr), z_objective*np.ones_like(rrr)),axis=len(rrr.shape))
# plt.plot(rrr, np.abs(U_objective(xyz2)))
# plt.show()
# # def for2dPlot(r,z):
# # 	xyz2=np.stack((r,np.zeros_like(r), z_objective+z),axis=len(r.shape))
# # 	return np.abs(U_objective(xyz2))**2
# # plot2D_function(for2dPlot, [0,R*.002], [-0.75*f,5*f],50,100, "for2dPlot")
# # z0=f0*1.0
# # z1=z0+f0*50
# # effectiveR_lens0 = R0/100#min(R0,8*(z_lens1-z_lens0)/(R0**2*k))
# # U_beforeLens = lambda xyz: U0(xyz)
# # U_afterLens = passThroughLens(U_beforeLens, k, f0, R0)
# # U_objective = expandInFreeSpace(U_afterLens, [[-effectiveR_lens0,effectiveR_lens0],[-effectiveR_lens0,effectiveR_lens0]], z0, k)
# # plot2D_function(addStaticZ(U_beforeLens,z0), [-effectiveR_lens0,effectiveR_lens0],[-effectiveR_lens0,effectiveR_lens0],50,50, "U_beforeLens")
# # plot2D_function(addStaticZ(U_afterLens,z0), [-effectiveR_lens0,effectiveR_lens0],[-effectiveR_lens0,effectiveR_lens0],50,50, "U_afterLens")
# # plot2D_function(addStaticZ(U_objective,z1), [-effectiveR_lens0*10,effectiveR_lens0*10],[-effectiveR_lens0*10,effectiveR_lens0*10],50,50, "U_objective")

# # plot1D_function(addStaticYZ(U_beforeLens, 0, z0), [-effectiveR_lens0,effectiveR_lens0], 50, "U_beforeLens")
# # plot1D_function(addStaticYZ(U_afterLens, 0, z0), [-effectiveR_lens0,effectiveR_lens0], 50, "U_afterLens")
# # plot1D_function(addStaticYZ(U_objective, 0, z1), [-effectiveR_lens0,effectiveR_lens0], 50, "U_objective")