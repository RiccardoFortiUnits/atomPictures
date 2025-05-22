# import Simulations_Libraries.trajectory_library as trajlib
import numpy as np
import matplotlib.pyplot as plt
def img(*x):
	plt.imshow(*x)
	plt.show()
from Camera import *
from scipy.stats import poisson
from scipy.optimize import curve_fit
# import Simulations_Libraries.general_library as genlib
from scipy.interpolate import RegularGridInterpolator
from scipy.ndimage import label, minimum_filter, maximum_filter

# def f(xy):
# 	x=xy[...,0]
# 	y=xy[...,1]
# 	r1=(x+.255678)**2+(y+.355678)**2
# 	r2=(x-.155678)**2+(y-.255678)**2
# 	return 1000*((x**2-2*y)**2)*(r1+.01)*r2
# 	# return (xy[...,0]-.255678)**2 * (xy[...,1]-.03456)**2
# 	# return (xy[...,0]-.255678)**2+(xy[...,1]-.03456)**2


# L= np.array([1,1])
# center= np.array([0,0])
# N= np.array([200,200])
# lim=.2
# def getXY(center=center,L=L,N=N):
# 	y0=np.linspace(-.5,.5,N[0])*L[0]+center[0]
# 	x0=np.linspace(-.5,.5,N[1])*L[1]+center[1]
# 	x,y=np.meshgrid(x0,y0)
# 	xy=np.stack((x,y),axis=-1)
# 	return xy

# XY=getXY()
# fxy=f(XY)
# # plt.imshow(np.log(fxy))
# # plt.show()
# # dfdx,dfdy=np.gradient(fxy,axis=(0,1))
# # # plt.imshow((dfdx))
# # # plt.show()
# # # plt.imshow((dfdy))
# # # plt.show()

# # zerosY=np.concatenate((dfdx[:-1,:]*dfdx[1:,:]<=0, np.zeros((1,len(dfdx[0]))))).astype(int)
# # zerosX=np.concatenate((dfdy[:,:-1]*dfdy[:,1:]<=0, np.zeros((len(dfdy),1))),axis=1).astype(int)

# # plt.imshow(2*zerosX+zerosY)
# # plt.show()


# def get_connected_points(m, a, b):
# 	labeled_map, num_features = label(m)
# 	start_label = labeled_map[a, b]
# 	return np.argwhere(labeled_map==start_label)
# def getListOfIslands(m):
# 	labeled_map, num_features = label(m)
# 	islands = []
# 	for i in range(num_features):
# 		island = np.argwhere(labeled_map==i+1)
# 		island=island[:,::-1]
# 		islands.append(island)
# 	return islands
# def getIslandCenterAndSize(island, mapCenter, mapSize, mapResolution):
# 	minx=np.min(island[:,1])
# 	maxx=np.max(island[:,1])
# 	miny=np.min(island[:,0])
# 	maxy=np.max(island[:,0])
# 	center = np.array([(minx+maxx)*.5, (miny+maxy)*.5]) * mapSize / (mapResolution-1) + (mapCenter - mapSize * .5)
# 	size = np.array([(-minx+maxx), (-miny+maxy)]) * mapSize / (mapResolution-1)
# 	return center, size
	
# def xy_toIndexes(xy, center, size, resolution):
# 	ix = (xy[...,0] - (center[0] - size[0] * .5)) * (resolution[0]-1) / size[0]
# 	iy = (xy[...,1] - (center[1] - size[1] * .5)) * (resolution[1]-1) / size[1]
# 	return (np.stack((ix,iy),axis=len(ix.shape))+.5).astype(int)
# def notInIsland(xy,island,center, size, resolution):
# 	I = xy_toIndexes(xy,center, size, resolution)

# 	equalities = np.logical_or(I[:,:,None,1] != island[None,None,:,1], I[:,:,None,0] != island[None,None,:,0])
# 	return np.all(equalities, axis=2)

# def extendPonds(map, pondDepth):
# 	#get the local minima
# 	map = np.nan_to_num(map, nan=-np.inf)
# 	filteredMap = minimum_filter(map, size=3, mode='constant', cval=-np.inf)#let's not consider values on the borders of the map
# 	minimaCoords = np.argwhere(np.logical_and(filteredMap == map, filteredMap != -np.inf))
# 	minVals = map[minimaCoords[:,0], minimaCoords[:,1]]
# 	indexes = np.argsort(minVals)
# 	minimaCoords = minimaCoords[indexes]
# 	minVals = minVals[indexes]
# 	finalMap = np.zeros_like(map)
# 	for xy in minimaCoords:
# 		x=xy[0]
# 		y=xy[1]
# 		pondedMap = map<map[x,y]+pondDepth
# 		pond = get_connected_points(pondedMap, x,y)
# 		finalMap[pond[:,0],pond[:,1]] = 1
# 	return finalMap
	
# def getConstantSections(map, epsilon):
# 	mins = extendPonds(map,epsilon)
# 	maxs = extendPonds(-map,epsilon)
# 	res = mins+maxs
# 	res[res>1]=1
# 	return res 

# def zoomOnAllFlatSections(f,epsilon,center, ranges, resolutions):
# 	'''given an initial range and resolution, finds all the flat 
# 	sections of function f(x,y), returning a list of ranges and 
# 	centers in which each section is well "visible" with the 
# 	given resolution
# 	Of course, the initial resolution should be good enough that 
# 	each minimum is visible'''
# 	xy = getXY(center, ranges, resolutions)
# 	map = f(xy)
# 	allSections = getConstantSections(map, epsilon)
# 	#extend the island by one pixel, to avoid losing valid values of the islands when we zoom
# 	allSections = maximum_filter(allSections, size=5, mode='constant', cval=-np.inf)
	
# 	islands = getListOfIslands(allSections)
# 	if len(islands) == 1:
# 		plt.imshow(allSections)
# 		plt.show()
# 		return
		
# 	for island in islands:
# 		islandCenter, islandSize = getIslandCenterAndSize(island, center, ranges, resolutions)
# 		def f_removeNearbyIslands(xy):
# 			z=f(xy)
# 			z[notInIsland(xy, island, center, ranges, resolutions)] = np.nan
# 			return z
# 		zoomOnAllFlatSections(f_removeNearbyIslands, epsilon, islandCenter, islandSize, resolutions)
	
# zoomOnAllFlatSections(f,.01, center, L, N)
# plt.imshow(getConstantSections(fxy,.01))
# plt.show()


class mappedFunction:
	def __init__(self, f, center, size, resolution, subRange=None):
		self.f=f
		self.center=center.astype(float)
		self.size=size.astype(float)
		self.resolution=resolution.astype(int)
		self.subRange=subRange
        
	
	def __call__(self, *args, **kwds):
		map = self.f(*args, **kwds)
		if self.subRange is not None:
			map[self.subRange(*args, **kwds)] = np.nan
		return map
	
	
	def xy_toIndexes(self, xy):
		ix = (xy[...,0] - (self.center[0] - self.size[0] * .5)) * (self.resolution[0]-1) / self.size[0]
		iy = (xy[...,1] - (self.center[1] - self.size[1] * .5)) * (self.resolution[1]-1) / self.size[1]
		return (np.stack((ix,iy),axis=len(ix.shape))+.5).astype(int)
	
	def getXY(self):
		x0=np.linspace(-.5,.5,self.resolution[0])*self.size[0]+self.center[0]
		y0=np.linspace(-.5,.5,self.resolution[1])*self.size[1]+self.center[1]
		y,x=np.meshgrid(y0,x0)
		xy=np.stack((x,y),axis=-1)
		return xy
	@staticmethod
	def rangesToCenterAndSize(ranges):		
		minx=ranges[0][0]
		maxx=ranges[0][1]
		miny=ranges[1][0]
		maxy=ranges[1][1]
		center = np.array([(minx+maxx)*.5, (miny+maxy)*.5])
		size = np.array([(-minx+maxx), (-miny+maxy)])
		return center, size
	@staticmethod
	def get_connected_points(m, a, b):
		labeled_map, num_features = label(m)
		start_label = labeled_map[a, b]
		return labeled_map==start_label
	@staticmethod
	def extendPonds(map, pondDepth):
		allPonds = np.zeros_like(map, dtype=bool)
		
		#get the local minima and maxima
		grad_x, grad_y = np.gradient(map, axis=(0, 1))
		grad = np.sqrt(grad_x**2 + grad_y**2)
		for usedMap in [map,-map, grad]:
			minMap = np.nan_to_num(usedMap, nan=-np.inf)
			minFilteredMap = minimum_filter(minMap, size=3, mode='constant', cval=-np.inf)#let's not consider values on the borders of the map
			minFilteredMap = np.logical_and(minFilteredMap == usedMap, minFilteredMap != -np.inf)
			allPonds = np.logical_or(allPonds, minFilteredMap)
		
		# img(allPonds)
		map = np.nan_to_num(map, nan=-np.inf) #let's take out the nan values once and for all
	
		minimaCoords = np.argwhere(allPonds)
		for xy in minimaCoords:
			x,y=xy
			pondCenter = map[x,y]
			pondedMap = np.abs(map - pondCenter) < pondDepth
			pond = mappedFunction.get_connected_points(pondedMap, x,y)
			allPonds = np.logical_or(allPonds, pond)
		# img(allPonds)
		return allPonds
	@staticmethod
	def extendPonds_old(map, pondDepth):
		#get the local minima
		map = np.nan_to_num(map, nan=-np.inf)
		filteredMap = minimum_filter(map, size=3, mode='constant', cval=-np.inf)#let's not consider values on the borders of the map
		minimaCoords = np.argwhere(np.logical_and(filteredMap == map, filteredMap != -np.inf))
		minVals = map[minimaCoords[:,0], minimaCoords[:,1]]
		indexes = np.argsort(minVals)
		minimaCoords = minimaCoords[indexes]
		minVals = minVals[indexes]
		finalMap = np.zeros_like(map,dtype=bool)
		for xy in minimaCoords:
			x=xy[0]
			y=xy[1]
			pondedMap = map<map[x,y]+pondDepth
			pond = mappedFunction.get_connected_points(pondedMap, x,y)
			finalMap = np.logical_or(finalMap, pond)
		return finalMap
		
	@staticmethod
	def getConstantSections(map, epsilon):
		return mappedFunction.extendPonds(map,epsilon)
		# maxs = mappedFunction.extendPonds(-map,epsilon)
		res = np.logical_or(mins, maxs)
		return res 
	@staticmethod	
	def getListOfIslands(m):
		labeled_map, num_features = label(m)
		islands = []
		for i in range(num_features):
			island = np.argwhere(labeled_map==i+1)
			# island=island[:,::-1]
			islands.append(island)
		return islands
	@staticmethod	
	def getIslandCenterAndSize(islandMap, mapCenter, mapSize, mapResolution):
		island = np.argwhere(islandMap)
		minx=np.min(island[:,0])
		maxx=np.max(island[:,0])
		miny=np.min(island[:,1])
		maxy=np.max(island[:,1])
		center = np.array([(minx+maxx)*.5, (miny+maxy)*.5]) * mapSize / (mapResolution-1) + (mapCenter - mapSize * .5)
		size = np.array([(-minx+maxx), (-miny+maxy)]) * mapSize / (mapResolution-1)
		return center, size
	def notInIsland(self, xy,island):
		I = self.xy_toIndexes(xy)
		return np.logical_not(island[I[...,0],I[...,1]])
	
	def getBestDivisionForCoveredArea(self):
		mainAxis = 0 if self.size[0] > self.size[1] else 1
		leftCenter = self.center + 0
		leftCenter[mainAxis] -= self.size[mainAxis]*.25
		newSize = self.size + 0
		newSize[mainAxis] *= .5
		leftMap = mappedFunction(self.f, leftCenter, newSize, self.resolution)
		rightCenter = self.center + 0
		rightCenter[mainAxis] += self.size[mainAxis]*.25
		rightMap = mappedFunction(self.f, rightCenter, newSize, self.resolution)
		return leftMap, rightMap
			
	def getZoomedFlatSections(self, epsilon, minCoveredArea=0.35, isFirst=True):
		xy = self.getXY()
		map = self(xy)
		allSections = mappedFunction.getConstantSections(map, epsilon)
		#extend the island by 2 pixels, to avoid losing valid values of the islands when we zoom
		allSections = maximum_filter(allSections, size=5, mode='constant', cval=-np.inf)
		# self.getBestDivisionForCoveredArea()
		'''
		if isFirst:
			img(map)
			mapRange = np.max(np.nan_to_num(map, nan=-np.inf)) - np.min(np.nan_to_num(map, nan=np.inf))
			img(map - mapRange * 2 * allSections.astype(float))
		#'''
		# islands = self.getListOfIslands(allSections)
		islandMap, nOfIslands = label(allSections)		
		allSubMaps = []
		if nOfIslands == 1:
			#let's check if the island is covering enough area
			islandCenter, islandSize = self.getIslandCenterAndSize(islandMap, self.center, self.size, self.resolution)
			if np.sum(islandSize / self.size) > 1.5:#section is zoomed enough?
				coveredArea = np.sum(islandMap) / islandMap.size
				if coveredArea >= minCoveredArea:#section covers enough area?
					return [mappedFunction(self.f, islandCenter, islandSize, self.resolution, subRange = partial(self.notInIsland,island=islandMap))]
				#else, let's half the region, hoping that the 2 halfs can be better zoomed in a rectangle
				(left, right) = self.getBestDivisionForCoveredArea()
				allSubMaps += (left.getZoomedFlatSections(epsilon, minCoveredArea, isFirst=False))
				allSubMaps += (right.getZoomedFlatSections(epsilon, minCoveredArea, isFirst=False))
				if len(allSubMaps) > 0:
					return allSubMaps
				return []
		
		for islandIdx in range(nOfIslands):
			island = islandMap == islandIdx + 1
			islandCenter, islandSize = self.getIslandCenterAndSize(island, self.center, self.size, self.resolution)
			# def f_removeNearbyIslands(xy):
			# 	z=self(xy)
			# 	z[self.notInIsland(xy, island)] = np.nan
			# 	return z
			submap = mappedFunction(self.f, islandCenter, islandSize, self.resolution, subRange = partial(self.notInIsland,island=island))
			allSubMaps += (submap.getZoomedFlatSections(epsilon, minCoveredArea, isFirst=False))
		
		if len(allSubMaps) > 0:
			return allSubMaps
		return []
	
	def integral(self):
		xy=self.getXY()
		z = self(xy)
		z = np.nan_to_num(z, nan=0)
		return np.sum(z) * np.prod(self.size / self.resolution)	

if __name__ == "__main__":
	def f(xy):
		x=xy[...,0]
		y=xy[...,1]
		r1=(x)**2+(y)**2
		r2=(x-.155678)**2+(y-.255678)**2
		# return np.abs((1*r1-.15)**2)
		return 1000*r1*r2
		# return (xy[...,0]-.255678)**2 * (xy[...,1]-.03456)**2
		# return (xy[...,0]-.255678)**2+(xy[...,1]-.03456)**2
	
	L= np.array([1,1])
	center= np.array([0,0])
	N= np.array([200,200])
	def getXY(center=center,L=L,N=N):
		y0=np.linspace(-.5,.5,N[0])*L[0]+center[0]
		x0=np.linspace(-.5,.5,N[1])*L[1]+center[1]
		y,x=np.meshgrid(y0,x0)
		xy=np.stack((x,y),axis=-1)
		return xy

	XY=getXY()
	fxy=f(XY)
	img(np.log(fxy))
	# plt.imshow(np.log(fxy))
	# plt.show()
	m = mappedFunction(f,center,L,N)

	print(m.getZoomedFlatSections(.001))
