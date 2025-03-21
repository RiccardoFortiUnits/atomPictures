import Simulations_Libraries.trajectory_library as trajlib
import numpy as np
import matplotlib.pyplot as plt
from Camera import *
from scipy.stats import poisson
from scipy.optimize import curve_fit
import Simulations_Libraries.general_library as genlib
import pickle
plt.ion()

waist = 1e-3 
power = 10e-3

##test experiment and trajectory plot
exp = trajlib.experiment()

for i in range(1):
# 	exp.add_atom(trajlib.Ytterbium(*(np.random.random((3,))*1e-6-.5e-6), 0,0,0,isotope=174))
	exp.add_atom(trajlib.Ytterbium(0,0,0, 0,0,0,isotope=174))
dt = 5e-9
detuning = 0#-5.5*trajlib.MHz
freq = trajlib.c/(exp.atoms[0].transitions[0].Lambda) + detuning
lambd = trajlib.c/freq
Lambda = lambd#399e-9

b = trajlib.Laser(0,0, Lambda, power, (waist/2,waist/2), switchingTimes =      [dt, -200e-9, 400e-9, 400e-9])
b1 = trajlib.Laser(np.pi,0, Lambda, power, (waist/2,waist/2), switchingTimes = [dt, 200e-9,  400e-9, 400e-9])
exp.add_laser(b)
exp.add_laser(b1)
# result = exp.run(10e-6, dt)#2-50us
# exp.plotTrajectories()

##stuff for the blur function
tweezerLambda = 532e-9
k = 2*np.pi/tweezerLambda#Lambda
tweezerPower = 10e-3
E0 = genlib.I2E_0(tweezerPower)
effectiveFocalLength = 25.5e-3
tweezerWaist = 550e-9
objective_Ray = 15.3e-3

# startPositions, directions = exp.getScatteredPhotons()
# # G = randExtractor.distribFunFromPDF_2D(lambda x,y: gauss(x, y,1,0,1e-8), [[-1e-9,1e-9]]*2, [5e-11]*2)
#'''
M = 8
finalPixelSize = 4.6e-6
finalNOfPixels = 40
finalCameraSize = finalNOfPixels * finalPixelSize
initialCameraSize = finalCameraSize / M#if we considered all the pixels, the camera size should be == 2*lensRadius = 32e-3 m
initialPixelSize = finalPixelSize / M
lensPosition = effectiveFocalLength
lensRadius = objective_Ray#16e-3
f11=lambda r,z:                                             blur(r,z - lensPosition,k,E0,effectiveFocalLength, tweezerWaist, objective_Ray)
# plot2D_function(f11,                                                                                                                        [0,1e-2], [-1e-6+lensPosition,1e-6+lensPosition], 50, 100)
G = randExtractor.distribFunFromradiusPDF_2D_1D(lambda r,z: blur(r,z - lensPosition,k,E0,effectiveFocalLength, tweezerWaist, objective_Ray), [0,initialCameraSize/2], 
												initialPixelSize/2, 
												[-1e-6+lensPosition,1e-6+lensPosition], 1e-7)
# G = lambda x,*y:x
magnificationGrid = pixelGrid(initialCameraSize,initialCameraSize,finalNOfPixels,finalNOfPixels, G, magnification=M)
# quantumEfficiencyGrid = pixelGrid(4.6e-6,4.6e-6,100,100, randExtractor.randomLosts(0.1))
quantumEfficiencyGrid = cMosGrid(finalCameraSize,finalCameraSize,finalNOfPixels,finalNOfPixels, randExtractor.randomLosts(0.1), "Orca_testing/shots/", imageStart = (10,10), imageSizes = (-10,-10))
c = Camera(position=(0,0,lensPosition), 
		   orientation=(0,-np.pi/2,0), 
		   radius=lensRadius,
		   pixelGrids=(magnificationGrid, quantumEfficiencyGrid))
'''
magnificationGrid = pixelGrid(32e-3,32e-3, 50,50, lambda x:x)
quantumEfficiencyGrid = pixelGrid(32e-3,32e-3, 50,50, randExtractor.randomLosts(0.1))
c = Camera(position=(25.5e-3,0,0), 
		   orientation=(0,0,0), 
		   radius=16e-3,
		   pixelGrids=(magnificationGrid, quantumEfficiencyGrid))
#'''
positions = np.repeat(np.array([0,0,0])[None,:],10,axis=0)
directions = np.repeat(np.array([0,0,1])[None,:],10,axis=0)
c.takePicture(positions, directions, plot=True)
info = {
	"initial atom positions" : exp.initialAtomPositions,
	"non-magnified pixel side (m)" : initialPixelSize,
	"magnification" : M,
	"lens radius" : lensRadius,
	"lens focus" : lensPosition,
	"imaging laser power" : power,
	"imaging laser waist" : waist,
	"imaging laser wavelength" : lambd,
	"tweezer laser waist" : tweezerWaist,
	"tweezer laser power" : tweezerPower,
	"tweezer laser wavelength" : tweezerLambda,
}

durations_us = [7,10,15,20]
for duration_us in durations_us:
	def update(index, experiment : trajlib.experiment):
		startPositions, directions = experiment.getScatteredPhotons()
		c.takePicture(startPositions, directions, plot=False, saveToFile=f"D:/simulationImages/{duration_us}us/image_{index}.h5", **info)

	exp.repeatRun(duration_us * 1e-6, dt, 10, update)
# c.pixelGrids = (magnificationGrid, quantumEfficiencyGrid)
# image = c.takePicture(startPositions, directions, plot=True)
# exp.plotTrajectoriesAndCameraAcquisition(c)
