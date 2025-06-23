import Simulations_Libraries.trajectory_library as trajlib
import numpy as np
import matplotlib.pyplot as plt
from Camera import *
from scipy.stats import poisson
from scipy.optimize import curve_fit
import Simulations_Libraries.general_library as genlib
plt.ion()

showPlots = False

normal = ""
fixedAtom = "_atomUnmovable"
zLattice = "_z_lattice"
usedCase = normal#
'''-------------------------------atom------------------------------'''
nOfAtoms = 10
useFixedTweezers = True
isotope = 171
baseAtom = trajlib.Ytterbium(0,0,0, 0,0,0,isotope=isotope)
detuning = 0#-5.5*trajlib.MHz
initialT = 15e-6 # initial temperature
saturation = 42.46
p_e = 1/2 * saturation/(1+saturation) # Probability of atom being in excited state
p_g = 1 - p_e # Probability of atom being in the ground state

'''-------------------------------imaging laser------------------------------'''
imgWaist = 1e-3 # m
imgPower = 10e-3# W
freq = trajlib.c/(baseAtom.transitions[0].Lambda) + detuning
Lambda = trajlib.c/freq

'''-------------------------------tweezer laser------------------------------'''

twzWaist = 540e-9
twzIntensity = 2.24e-3*trajlib.kB # Tweezer depth in J
twzLambda = 532e-9 # tweezer wavelength in m
trapFreq_r = 2*np.pi*140e3
baseFreq_z = 30e3 if usedCase!=zLattice else 200e3
trapFreq_z = 2*np.pi*baseFreq_z
n_0 = trajlib.n_T(trapFreq_r,initialT)
radius_rms, v_r_rms = trajlib.extract_coordinates_rms(T = 2/3*initialT,omega=trapFreq_r, m=baseAtom.m, N_atoms=1)
z_rms, v_z_rms = trajlib.extract_coordinates_rms(T = 1/3*initialT,omega=trapFreq_z, m=baseAtom.m, N_atoms=1)
if usedCase == fixedAtom:
	radius_rms = 0
	z_rms = 0
	v_r_rms = 0
	v_z_rms = 0
	trajlib.experiment.nextPointInTrajectory = lambda *x,**y: None

'''-------------------------------pixels and camera------------------------------'''
pixelScale = 1#                                                                       10.
pixelType = "" if pixelScale == 1 else f"_{int(pixelScale)}reduction"
nPixels = np.array([108,108]) * int(pixelScale)
pixelSize = 4.6e-6 / pixelScale
magnification = 8
cameraSize = pixelSize*nPixels
atomSpaceSize = cameraSize / magnification
lensDistance = 25.5e-3
lensRadius = 16e-3
quantumEfficiency = .83
objectiveTransmission = .76
dicroicReflection = .9
filterTransmission = .9**2
totalEfficiency = quantumEfficiency * objectiveTransmission * dicroicReflection * filterTransmission
fillProbability = 1

'''-------------------------------timings------------------------------'''
max_dt = 1e-5
min_dt = 1e-10
max_dx = twzWaist / 10
impulseDuration = 400e-9
freeFlightTime = 3e-4#the imaging laser beams won't be active until this time (this time is counted inside experimentDuration)
acquisitionDuration = 12e-6
maxAcquisitionDuration = acquisitionDuration#let's do a longer simulation, so in case we want to use a larger acquisition duration, we already have the simulation data
                                                #otherwise, just keep it equal to acquisitionDuration
experimentDuration = freeFlightTime + maxAcquisitionDuration #2-50us

'''-------------------------------folders and files------------------------------'''
pictureFolder = f"D:/simulationImages/Yt{isotope}_{int(experimentDuration*1e6)}us_freeFlight{freeFlightTime*1e6}us_{nOfAtoms}tweezerArray{usedCase}/{int(acquisitionDuration*1e6)}us_pictures{pixelType}/"
simulationFolder = f"D:/simulationImages/Yt{isotope}_{int(experimentDuration*1e6)}us_freeFlight{freeFlightTime*1e6}us_{nOfAtoms}tweezerArray{usedCase}/simulation/"
baseFileName = "simulation"
blurFolder = "bigBlurs/"
backgroundNoiseFolder = "Orca_testing/shots_free_space/"
if not os.path.exists(pictureFolder):
	os.makedirs(pictureFolder)
if not os.path.exists(simulationFolder):
	os.makedirs(simulationFolder)

'''-------------------------------ok, let's start------------------------------'''

G = Camera.blurFromImages(blurFolder)
#'''
magnificationGrid = pixelGrid(cameraSize[0],cameraSize[1],nPixels[0],nPixels[1], lambda xy, z: G(xy, z - lensDistance), magnification=magnification)
quantumEfficiencyGrid = fixed_cMosGrid(cameraSize[0],cameraSize[1],randExtractor.randomLosts(1-totalEfficiency), backgroundNoiseFolder)
c = Camera(position=(0,0,lensDistance), 
		orientation=(0,-np.pi/2,0), 
		radius=lensRadius,
		pixelGrids=(magnificationGrid, quantumEfficiencyGrid))

randomPosition = partial(np.random.normal, loc=0)

for repeat in range(1000):
	print(repeat)
	fileName = f"{baseFileName}_{repeat}.h5"
	imageFileName = f"{pictureFolder}/{fileName}"
	simulationFileName = f"{simulationFolder}/{fileName}"
	exp = trajlib.experiment() # no tweezer
	# exp = trajlib.experiment(elasticForceFromTweezer)
	if not os.path.exists(simulationFileName):
		filling = np.random.random(nOfAtoms) <= fillProbability
		while filling.sum() < 1:
			filling = np.random.random(nOfAtoms) <= fillProbability
		for i in range(nOfAtoms):
			if filling[i]:
				atomPosition = np.concatenate((np.random.normal(0, radius_rms, 2),np.random.normal(0, z_rms, 1)))
				atomVelocity = np.concatenate((np.random.normal(0, v_r_rms, 2),np.random.normal(0, v_z_rms, 1)))
				newAtom = trajlib.Ytterbium(*atomPosition, *atomVelocity, isotope=isotope)
				exp.add_atom(newAtom)
		b, b1 = trajlib.Laser.counterPropagatingLasers(0,	0, Lambda, imgPower, (imgWaist/2,imgWaist/2), impulseDuration, 0, True, freeFlightTime, )
		exp.add_laser(b)
		exp.add_laser(b1)

		# result = exp.run(experimentDuration, dt)#2-50us        
		result = exp.new_run(experimentDuration, max_dt, min_dt, max_dx)
		positionAfterTimeOfFlight = exp.positionsAtTime(freeFlightTime) * magnificationGrid.magnification
		pixelPositionAfterTimeOfFlight = magnificationGrid._normalizeCoordinate(-positionAfterTimeOfFlight[:,0], positionAfterTimeOfFlight[:,1], removeOutOfBoundaryValues=False)
		metadata = dict(
			max_dt = max_dt,
			min_dt = min_dt,
			max_dx = max_dx,
			freeFlightTime = freeFlightTime,
			experimentDuration = experimentDuration,
			atoms_InitialCoordinates = exp.initialAtomPositions,
			atoms_InitialSpeeds = exp.initialAtomSpeeds,
			atom_isotope = isotope,
			laser_power = imgPower,
			laser_waist = imgWaist,
			laser_lambda = Lambda,
			laser_detuning = detuning,
			laser_frequency = freq,
			laser_impulseDuration = impulseDuration,
			initial_temperature = initialT,
			saturation = saturation,
			fillProbability = fillProbability,
			atomFilling = filling,
			positionAfterTimeOfFlight = positionAfterTimeOfFlight,
			pixelPositionAfterTimeOfFlight = pixelPositionAfterTimeOfFlight,

			lens0_radius = c.radius,
			magnification = magnification,
			quantumEfficiency = quantumEfficiency,
			objectiveTransmission = objectiveTransmission,
			dicroicReflection = dicroicReflection,
			filterTransmission = filterTransmission,
			totalEfficiency = totalEfficiency,

			camera_pixelSize = pixelSize,
			camera_pixelNumber = nPixels,
			camera_blurImagesPath = blurFolder,

			tweezer_waist = twzWaist,
			tweezer_intensity = twzIntensity,
			tweezer_lambda = twzLambda,
			tweezer_trapFreq_radial = trapFreq_r,
			tweezer_trapFreq_axial = trapFreq_z,
		)        
		exp.saveAcquisition(simulationFileName, **metadata)
	else:
		metadata = exp.loadAcquisition(simulationFileName)
		metadata.update(dict(
			lens0_radius = c.radius,
			lens0_distance = lensDistance,
			quantumEfficiency = quantumEfficiency,
			objectiveTransmission = objectiveTransmission,
			dicroicReflection = dicroicReflection,
			filterTransmission = filterTransmission,
			totalEfficiency = totalEfficiency,
			camera_pixelSize = pixelSize,
			camera_pixelNumber = nPixels,
			camera_blurImagesPath = blurFolder,
		))
	if showPlots:
		exp.plotTrajectories()
	if not os.path.exists(imageFileName):
		startPositions, directions = exp.getScatteredPhotons(freeFlightTime + acquisitionDuration)
		image = c.takePicture(startPositions, directions, plot=showPlots, saveToFile=imageFileName, acquisitionDuration = freeFlightTime + acquisitionDuration, **metadata)
		# a=cameraAtomImages(pictureFolder)
		# plt.imshow(a.averageImage())
		if showPlots:
			exp.plotTrajectoriesAndCameraAcquisition(c)
			input("Press Enter to close the plot windows")
			plt.close('all')