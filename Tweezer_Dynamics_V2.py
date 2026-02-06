# Simulation of Atoms Dynamics into Tweezer + Imaging. Consider the 3 degrees of freedom instead of the axial and radial one 
import os


working_directory = "D:/PhD_Trieste/Simulations/Imaging_Simulation"

if os.getcwd()!= working_directory:
    os.chdir(working_directory)  
    
print(working_directory)

import Simulations_Libraries.trajectory_library as trajlib
import numpy as np
import matplotlib.pyplot as plt
from Camera import *
from scipy.stats import poisson
from scipy.optimize import curve_fit
import Simulations_Libraries.general_library as genlib
import scipy.constants as const
from tqdm import tqdm
#%matplotlib

plt.ion()



showPlots = False
showCameraImage = True


normal = ""
usedCase = normal




'''-----------------------------------atom----------------------------------'''
nOfAtoms = 15
useFixedTweezer = True
isotope = 171
baseAtom = trajlib.Ytterbium(0,0,0, 0,0,0,isotope=isotope)
initialT = 20e-6 # initial temperature
detuning = 0#-5.5*trajlib.MHz
saturation = 42.46




'''------------------------------imaging laser------------------------------'''
imagingOption = 'FS'  # 'FS' or 'Tweezer'
imgWaist = 1e-3 # m
imgPower = 10e-3# W
freq = trajlib.c/(baseAtom.transitions[0].Lambda) + detuning
Lambda = trajlib.c/freq



'''-------------------------------Tweezer laser-----------------------------'''

TweezerWaist = 578e-9
TweezerIntensity = 2.3e-3*trajlib.kB # Tweezer depth in J
TweezerLambda = 532e-9 # Tweezer wavelength in m
trapFreq_r = 2*np.pi*140e3
baseFreq_a = 29e3 #np.sqrt((29e3)**2 + (60e3)**2) #if we want to put light sheet
trapFreq_a = 2*np.pi*baseFreq_a
n_0Radial = trajlib.n_T(trapFreq_r,initialT)
n_0Axial = trajlib.n_T(trapFreq_a,initialT)

x_rms = np.sqrt(trajlib.hbar/(2*baseAtom.m*trapFreq_r)*(2*n_0Radial+1))
y_rms = np.sqrt(trajlib.hbar/(2*baseAtom.m*trapFreq_r)*(2*n_0Radial+1))
z_rms = np.sqrt(trajlib.hbar/(2*baseAtom.m*trapFreq_a)*(2*n_0Axial+1))
    
v_x_rms = np.sqrt(trajlib.hbar*trapFreq_r/(2*baseAtom.m)*(2*n_0Radial+1))
v_y_rms = np.sqrt(trajlib.hbar*trapFreq_r/(2*baseAtom.m)*(2*n_0Radial+1))
v_z_rms = np.sqrt(trajlib.hbar*trapFreq_a/(2*baseAtom.m)*(2*n_0Axial+1))




'''----------------------------pixels and camera----------------------------'''
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



'''-------------------------------timings------------------------------'''
max_dt = 1e-5
min_dt = 1e-10
max_dx = TweezerWaist / 10
impulseDuration = 400e-9

TweezerDuration = 0e-6
freeFlightTime = 90e-6#40e-6

acquisitionDuration = 7e-6 #7e-6


if imagingOption=='Tweezer':
    freeFlightTime = 0
    imagingStartingTime = TweezerDuration - acquisitionDuration
    experimentDuration = TweezerDuration
    
elif imagingOption =='FS':
    imagingStartingTime = TweezerDuration + freeFlightTime - acquisitionDuration
    experimentDuration = TweezerDuration + freeFlightTime 
    
    

'''-------------------------------folders and files------------------------------'''
extraWord = f'_Temperature_{initialT*1e6}uK'
pictureFolder = working_directory +  f"/simulationImages/Tweezer/Yt{isotope}_{int(experimentDuration*1e6)}us_TweezerDuration{TweezerDuration*1e6}us_freeFlight{freeFlightTime*1e6}us_imagingTime{acquisitionDuration*1e6}us_imaging{imagingOption}_{nOfAtoms}" + extraWord
simulationFolder = working_directory + f"/simulationImages/Tweezer/Yt{isotope}_{int(experimentDuration*1e6)}us_TweezerDuration{TweezerDuration*1e6}us_freeFlight{freeFlightTime*1e6}us_imagingTime{acquisitionDuration*1e6}us_imaging{imagingOption}_{nOfAtoms}" + extraWord + "/simulation/"
baseFileName = "simulation"
blurFolder = "bigBlurs/"       # PSF along the camera axis
backgroundNoiseFolder = "Orca_testing/shots_free_space/"

if not os.path.exists(pictureFolder):
	os.makedirs(pictureFolder)
if not os.path.exists(simulationFolder):
	os.makedirs(simulationFolder)
    
    

'''-------------------------------ok, let's start------------------------------'''

G = Camera.blurFromImages(blurFolder)

magnificationGrid = pixelGrid(cameraSize[0],cameraSize[1],nPixels[0],nPixels[1], lambda xy, z: G(xy, z - lensDistance), magnification=magnification)
quantumEfficiencyGrid = fixed_cMosGrid(cameraSize[0],cameraSize[1],randExtractor.randomLosts(1-totalEfficiency), backgroundNoiseFolder)
c = Camera(position=(0,0,lensDistance), 
		orientation=(0,-np.pi/2,0), 
		radius=lensRadius,
		pixelGrids=(magnificationGrid, quantumEfficiencyGrid))





repetitions = 5

for repeat in tqdm(range(repetitions), desc="Running simulation",mininterval=10,maxinterval=30):

    
	fileName = f"{baseFileName}_{repeat}.h5"
	imageFileName = f"{pictureFolder}/{fileName}"
	simulationFileName = f"{simulationFolder}/{fileName}"
	exp = trajlib.experiment() 
    
    
	if not os.path.exists(simulationFileName):
		for i in range(nOfAtoms):
                
			atomPosition = np.random.normal(0, [x_rms, y_rms, z_rms], size=(3))
			atomVelocity = np.random.normal(0, [v_x_rms, v_y_rms, v_z_rms], size=(3))
			newAtom = trajlib.Ytterbium(*atomPosition, *atomVelocity, isotope=isotope)
			exp.add_atom(newAtom)
      
		b, b1 = trajlib.Laser.counterPropagatingLasers(angle = 0, x0 = 0, wavelength=Lambda, Intensity=imgPower, w0 = (imgWaist/2,imgWaist/2), impulseDuration=impulseDuration, deadTime=0, halfFirstPeriod=True, initialTimeOff=imagingStartingTime, )
		exp.add_laser(b)
		exp.add_laser(b1)
        
        
		result = exp.new_run(time = experimentDuration, trapDuration= TweezerDuration, trapType = 'Tweezer', trapFreq_r = trapFreq_r, trapFreq_a = trapFreq_a,max_dt =  max_dt,min_dt =  min_dt, max_dx = max_dx)
		
		positionAfterTimeOfFlight = exp.positionsAtTime(experimentDuration) * magnificationGrid.magnification
		pixelPositionAfterTimeOfFlight = magnificationGrid._normalizeCoordinate(-positionAfterTimeOfFlight[:,0], positionAfterTimeOfFlight[:,1], removeOutOfBoundaryValues=False)
		
		metadata = dict(
			max_dt = max_dt,
			min_dt = min_dt,
			max_dx = max_dx,
            freeFlightTime = freeFlightTime,
            TweezerDuration = TweezerDuration,
			experimentDuration = experimentDuration,
            imagingTime = acquisitionDuration, 
            imagingOption = imagingOption,
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
            positionAfterTweezerDuration =[],
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
            
			Tweezer_waist = TweezerWaist,
			Tweezer_intensity = TweezerIntensity,
			Tweezer_lambda = TweezerLambda,
			Tweezer_trapFreq_radial = trapFreq_r,
			Tweezer_trapFreq_axial = trapFreq_a,
            
            
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
            
            atoms_InitialCoordinates = exp.initialAtomPositions,
			atoms_InitialSpeeds = exp.initialAtomSpeeds,
            
            
            
		))
        
	
        
        
        
	if not os.path.exists(imageFileName):
		startPositions, directions = exp.getScatteredPhotons(experimentDuration)
		
		if showPlots:
			if acquisitionDuration == 0:
				exp.plotTrajectories()
			else:
				exp.plotTrajectoriesAndCameraAcquisition(c)
            
		image = c.takePicture(startPositions, directions, plot=showCameraImage, saveToFile=imageFileName, acquisitionDuration = experimentDuration, **metadata)


        

	elif os.path.exists(imageFileName):
        
		if showCameraImage:
		    a=cameraAtomImages(pictureFolder)
		    a.showImage(Index = repeat, planeAtomsPosition =[exp.lastPositons[:,:,0], exp.lastPositons[:,:,1]],conversionFactor = magnification/pixelSize,showAtomsPOsitionInPlane = False)
        
		if showPlots:
			if acquisitionDuration == 0:
				exp.plotTrajectories()
			else:
				exp.plotTrajectoriesAndCameraAcquisition(c)
