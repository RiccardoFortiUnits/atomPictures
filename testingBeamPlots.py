import Simulations_Libraries.trajectory_library as trajlib
import numpy as np
import matplotlib.pyplot as plt
from Camera import *
from scipy.stats import poisson
from scipy.optimize import curve_fit
import Simulations_Libraries.general_library as genlib
plt.ion()

waist = 1e-3 
power = 10e-3

##test sum of beams and section plot
# b = trajlib.Beam(0,0, lambda coordinates : trajlib.GaussianBeam(coordinates,556*10**-9,waist/2,waist/2,power))

# b = b + trajlib.Beam((np.pi/2, np.pi/4, 0),(0,0.00,0.007), lambda coordinates : trajlib.GaussianBeam(coordinates,556*10**-9,waist/2,waist/2,power))
# b.plotSection((0,0,0), (0,0,0), [-0.015,0.015], [-.015,0.015], 30)

##test experiment and trajectory plot
exp = trajlib.experiment()

for i in range(10):
	exp.add_atom(trajlib.Ytterbium(*((np.random.random((3,))-.5)*np.array([90e-6,90e-6,0])), 0,0,0,isotope=174))
	# exp.add_atom(trajlib.Ytterbium(0,0,0, 0,0,0,isotope=174))
dt = 5e-9
detuning = 0#-5.5*trajlib.MHz
freq = trajlib.c/(exp.atoms[0].transitions[0].Lambda) + detuning
lambd = trajlib.c/freq
Lambda = lambd#399e-9

b = trajlib.Laser(0,0, Lambda, power, (waist/2,waist/2), switchingTimes =      [dt, -200e-9, 400e-9, 400e-9])
b1 = trajlib.Laser(np.pi,0, Lambda, power, (waist/2,waist/2), switchingTimes = [dt, 200e-9,  400e-9, 400e-9])
exp.add_laser(b)
exp.add_laser(b1)
result = exp.run(10e-6, dt)#2-50us
exp.plotTrajectories()

##test camera with fake photons
# c = trajlib.Camera((1,0,0), (0,0,0), [-1,1], [-1,1], lambda direction : np.dot(direction,(1,0,0)) >= 0.5)
# positions  = np.array([[0,0,0],[0,0,0],[0,0,0],    [0,.1,.2], [0,1.3,.2]])
# directions = np.array([[1,0,0],[1,0,0],[.8,0.6,0], [1,0,0]  , [.8,-.6,0]])

# image = c.takePicture(positions, directions, plot=True)

##test camera with real photons
# exp = trajlib.experiment()

# for i in range(1):
#     exp.add_atom(trajlib.Ytterbium(.0+np.random.uniform(-0.0001,0.0001),+np.random.uniform(-0.01,0.01),+np.random.uniform(-0.01,0.01), 0,0,0,isotope=174))
# dt = 8.695652173913044e-7
# detuning = -5.5*trajlib.MHz
# freq = trajlib.c/(exp.atoms[0].transitions[1].Lambda) + detuning
# lambd = trajlib.c/freq
# b = trajlib.Laser(0,0, lambd, power, (waist/2,waist/2), switchingTimes =      [dt, -5e-5, 1e-4, 1e-4])
# b1 = trajlib.Laser(np.pi,0, lambd, power, (waist/2,waist/2), switchingTimes = [dt, 5e-5,  1e-4, 1e-4])
# exp.add_laser(b)
# exp.add_laser(b1)
# # exp.add_laser(b2)
# result = exp.run(2e-3, dt)
# exp.plotTrajectories()

# startPositions = exp.lastPositons[exp.lastHits[:-1]]
# directions = exp.lastGeneratedPhotons

# c = trajlib.Camera((0,1e-5,0), (np.pi/2,0,0), [-1e-4,1e-4], [-1e-4,1e-4], lambda direction : np.dot(direction,(1,0,0)) >= .9)


# image = c.takePicture(startPositions, directions, plot=True)

##stuff for the blur function
k = 2*np.pi/Lambda
E0 = genlib.I2E_0(power)
effectiveFocalLength = 25.5e-3
tweezerWaist = 1e-4
objective_Ray = 15.3e-3

startPositions, directions = exp.getScatteredPhotons()

G = Camera.blurFromImages("blurs/")
#'''
nPixels = 100
magnificationGrid = pixelGrid(4.6e-6*nPixels,4.6e-6*nPixels,nPixels,nPixels, lambda xy, z: G(xy, z - 1e-4))
quantumEfficiencyGrid = pixelGrid(4.6e-6*nPixels,4.6e-6*nPixels,nPixels,nPixels, randExtractor.randomLosts(0.1))
c = Camera(position=(0,0,1e-4), 
		   orientation=(0,-np.pi/2,0), 
		   radius=1e-4,
		   pixelGrids=(magnificationGrid, quantumEfficiencyGrid))
'''
magnificationGrid = pixelGrid(32e-3,32e-3, 50,50, lambda x:x)
quantumEfficiencyGrid = pixelGrid(32e-3,32e-3, 50,50, randExtractor.randomLosts(0.1))
c = Camera(position=(25.5e-3,0,0), 
		   orientation=(0,0,0), 
		   radius=16e-3,
		   pixelGrids=(magnificationGrid, quantumEfficiencyGrid))
#'''


image = c.takePicture(startPositions, directions, plot=True)
nPixels = 100
quantumEfficiencyGrid = cMosGrid(4.6e-6*nPixels,4.6e-6*nPixels,nPixels,nPixels, randExtractor.randomLosts(0.1), "Orca_testing/shots/", imageStart = (10,10), imageSizes = (-10,-10))
c.pixelGrids = (magnificationGrid, quantumEfficiencyGrid)
image = c.takePicture(startPositions, directions, plot=True)
exp.plotTrajectoriesAndCameraAcquisition(c)


# p = exp.getScatteringDistributionFromRepeatedRuns(1e-6, dt, 1000, c)
# # Define the Poisson distribution function
# def poisson_dist(k, lamb):
#     return poisson.pmf(k, lamb)

# # Calculate the histogram
# unique, counts = np.unique(p, return_counts=True)
# bin_centers = unique
# counts = counts / np.sum(counts)  # Normalize counts to get probabilities

# # Fit the histogram data to the Poisson distribution
# params, _ = curve_fit(poisson_dist, bin_centers, counts, p0=[np.mean(p)])

# # Plot the fitted Poisson distribution
# plt.plot(bin_centers, poisson_dist(bin_centers, *params), label=f'Poisson fit (mean = {params[0]})')
# plt.plot(bin_centers, counts, label='simulated data')
# plt.legend()

input("Press Enter to close the plot window and exit the script...")