import Simulations_Libraries.trajectory_library as trajlib
import numpy as np
import matplotlib.pyplot as plt
from Camera import *

plt.ion()

MOT_HorSize = 12*10**-3 
P1_G = 5.2*10**-2

##test sum of beams and section plot
# b = trajlib.Beam(0,0, lambda coordinates : trajlib.GaussianBeam(coordinates,556*10**-9,MOT_HorSize/2,MOT_HorSize/2,P1_G))

# b = b + trajlib.Beam((np.pi/2, np.pi/4, 0),(0,0.00,0.007), lambda coordinates : trajlib.GaussianBeam(coordinates,556*10**-9,MOT_HorSize/2,MOT_HorSize/2,P1_G))
# b.plotSection((0,0,0), (0,0,0), [-0.015,0.015], [-.015,0.015], 30)

##test experiment and trajectory plot
exp = trajlib.experiment()

for i in range(10):
	exp.add_atom(trajlib.Ytterbium(*(np.random.random((3,))*1e-6-.5e-6), 0,0,0,isotope=174))
dt = 5e-9
detuning_G_HOR = -5.5*trajlib.MHz
G_freq_HOR = trajlib.c/(exp.atoms[0].transitions[1].Lambda) + detuning_G_HOR
G_lambd_HOR = trajlib.c/G_freq_HOR

b = trajlib.Laser(0,0, G_lambd_HOR, P1_G, (MOT_HorSize/2,MOT_HorSize/2), switchingTimes =      [dt, -5e-7, 1e-6, 1e-6])
b1 = trajlib.Laser(np.pi,0, G_lambd_HOR, P1_G, (MOT_HorSize/2,MOT_HorSize/2), switchingTimes = [dt, 5e-7,  1e-6, 1e-6])
exp.add_laser(b)
exp.add_laser(b1)
result = exp.run(2e-5, dt)#2-50us
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
# detuning_G_HOR = -5.5*trajlib.MHz
# G_freq_HOR = trajlib.c/(exp.atoms[0].transitions[1].Lambda) + detuning_G_HOR
# G_lambd_HOR = trajlib.c/G_freq_HOR
# b = trajlib.Laser(0,0, G_lambd_HOR, P1_G, (MOT_HorSize/2,MOT_HorSize/2), switchingTimes =      [dt, -5e-5, 1e-4, 1e-4])
# b1 = trajlib.Laser(np.pi,0, G_lambd_HOR, P1_G, (MOT_HorSize/2,MOT_HorSize/2), switchingTimes = [dt, 5e-5,  1e-4, 1e-4])
# exp.add_laser(b)
# exp.add_laser(b1)
# # exp.add_laser(b2)
# result = exp.run(2e-3, dt)
# exp.plotTrajectories()

# startPositions = exp.lastPositons[exp.lastHits[:-1]]
# directions = exp.lastGeneratedPhotons

# c = trajlib.Camera((0,1e-5,0), (np.pi/2,0,0), [-1e-4,1e-4], [-1e-4,1e-4], lambda direction : np.dot(direction,(1,0,0)) >= .9)


# image = c.takePicture(startPositions, directions, plot=True)

startPositions = exp.lastPositons[exp.lastHits[:-1]]
directions = exp.lastGeneratedPhotons
# ainy_normalized = lambda x,y : ainy(x*1e10, y*1e10)
# Ainy = randExtractor.distribFunFromPDF_2D(ainy_normalized, [[-1e-4,1e-4]]*2, [5e-5]*2)

def gauss (x,y, A, x0, sigma):
	rho = np.sqrt(x**2 + y**2)
	return A * np.exp(-(rho-x0)**2/(2*sigma**2))
# G = randExtractor.distribFunFromPDF_2D(lambda x,y: gauss(np.linalg.norm(np.column_stack((x[0],y[0])), axis=1).reshape((1,-1)),1,0,5e-8**2), [[-1e-6,1e-6]]*2, [5e-8]*2)
G = randExtractor.distribFunFromradiusPDF_2D_1D(lambda r,z: blur(r,z - 1e-4,1,1,1,1,1), [-1e-6,1e-6], 1e-7, [0.9e-4,1.1e-4], 1e-7)
#'''
magnificationGrid = pixelGrid(5e-6,5e-6,200,200, G)
quantumEfficiencyGrid = pixelGrid(5e-6,5e-6,200,200, randExtractor.randomLosts(0.1))
c = Camera(position=(1e-4,0,0), 
		   orientation=(0,0,0), 
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

quantumEfficiencyGrid = cMosGrid(5e-6,5e-6,200,200, randExtractor.randomLosts(0.1), "Orca_testing/shots/")
c.pixelGrids = (magnificationGrid, quantumEfficiencyGrid)
image = c.takePicture(startPositions, directions, plot=True)
exp.plotTrajectoriesAndCameraAcquisition(c)


input("Press Enter to close the plot window and exit the script...")