# Simulation of Atoms Dynamics into ODT + Imaging
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
from tqdm import tqdm

#%matplotlib

plt.ion()



showPlots = False
showCameraImage = False
computeCameraImages = True

normal = ""
usedCase = normal




'''-------------------------------atom------------------------------'''
nOfAtoms = 100
useFixedODT = True
isotope = 171
baseAtom = trajlib.Ytterbium(0,0,0, 0,0,0,isotope=isotope)
initialT =10e-6 # initial temperature
detuning = 0#-5.5*trajlib.MHz
saturation = 42.46
p_e = 1/2 * saturation/(1+saturation) # Probability of atom being in excited state
p_g = 1 - p_e # Probability of atom being in the ground state



'''-------------------------------imaging laser------------------------------'''
imagingOption = 'ODT' # 'FS' or 'ODT'
imgWaist = 1e-3 # m
imgPower = 10e-3# W
freq = trajlib.c/(baseAtom.transitions[0].Lambda) + detuning
Lambda = trajlib.c/freq



'''-------------------------------ODT laser------------------------------'''

ODTWaist = 14e-6
ODTIntensity = 300e-6*trajlib.kB # ODT depth in J
ODTLambda = 759e-9 # ODT wavelength in m


'''
k = 4

trapFreqX =np.sqrt((1 + k**4)/2) * 2 * np.pi * 30 
trapFreqY = 2*np.pi*2.7e3
trapFreqZ = k * 2*np.pi*2.7e3
'''

k = 1
trapFreqX = 2*np.pi*200
trapFreqY = 2*np.pi*200
trapFreqZ = 2*np.pi*3e3


n_0X = trajlib.n_T(trapFreqX,initialT)
n_0Y = trajlib.n_T(trapFreqY,initialT)
n_0Z = trajlib.n_T(trapFreqZ,initialT)

x_rms = np.sqrt(trajlib.hbar/(2*baseAtom.m*trapFreqX)*(2*n_0X+1))
y_rms = np.sqrt(trajlib.hbar/(2*baseAtom.m*trapFreqY)*(2*n_0Y+1))
z_rms = np.sqrt(trajlib.hbar/(2*baseAtom.m*trapFreqZ)*(2*n_0Z+1))
    
v_x_rms = np.sqrt(trajlib.hbar*trapFreqX/(2*baseAtom.m)*(2*n_0X+1))
v_y_rms = np.sqrt(trajlib.hbar*trapFreqY/(2*baseAtom.m)*(2*n_0Y+1))
v_z_rms = np.sqrt(trajlib.hbar*trapFreqZ/(2*baseAtom.m)*(2*n_0Z+1))

minMeanPlanarInterparticleSpacing = 1/np.sqrt(nOfAtoms * trajlib.QHO_1D_Spatial_Distribution(q = 0,qRMS = x_rms) * trajlib.QHO_1D_Spatial_Distribution(q = 0,qRMS = y_rms))


trapFreqBar = (trapFreqX * trapFreqY * trapFreqZ) ** (1/3)
g_s = 1

# ---------- Fermi wave vector ----------
fermiEnergy = trajlib.hbar * trapFreqBar * (6 * nOfAtoms / g_s) ** (1/3)
#fermiTemperature = fermiEnergy/trajlib.kB 
fermiTemperature = (fermiEnergy/trajlib.kB) - 0.5 * trajlib.hbar * (trapFreqX + trapFreqY + trapFreqZ )/trajlib.kB  # Correct definition  
fermiK = np.sqrt(2 * baseAtom.m * fermiEnergy / trajlib.hbar**2)



'''-------------------------------pixels and camera------------------------------'''
pixelScale = 1#                                                                     
pixelType = "" if pixelScale == 1 else f"_{int(pixelScale)}reduction"
#nPixels = np.array([108,108]) * int(pixelScale)  # reducted ROI
#nPixels = np.array([80,300]) * int(pixelScale) # biggest possible ROI
nPixels = np.array([200,200]) * int(pixelScale) # biggest possible ROI
pixelSize = 4.6e-6 / pixelScale
magnification = 8 #8
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
max_dt = 1e-5    # Upper bound for time intervals
min_dt = 1e-10   # Lower bound for time intervals. Control parameter if max_dt is crossed to fast. 
max_dx = ODTWaist / 10 # Max distance that an atom can travel before updating its coordinates and velocities
impulseDuration = 400e-9
ODTDuration = 10e-6
freeFlightTime = 0e-6# the imaging laser beams won't be active until this time (this time is counted inside experimentDuration)

acquisitionDuration = 10e-6


if imagingOption=='ODT':
    freeFlightTime = 0
    imagingStartingTime = ODTDuration + freeFlightTime - acquisitionDuration
    experimentDuration = ODTDuration + freeFlightTime
    
elif imagingOption =='FS':
    imagingStartingTime = ODTDuration + freeFlightTime
    experimentDuration = ODTDuration + freeFlightTime + acquisitionDuration
    
    

'''-------------------------------folders and files------------------------------'''
extraWord = f'_Temperature_{initialT*1e6}uK_fx_{np.round(trapFreqX/2/np.pi)}Hz_fy_{np.round(trapFreqY/2/np.pi)}Hz_fz_{np.round(trapFreqZ/2/np.pi)}Hz'
pictureFolder = working_directory +  f"/simulationImages/ODT/Yt{isotope}_{int(experimentDuration*1e6)}us_ODTDuration{ODTDuration*1e6}us_freeFlight{freeFlightTime*1e6}us_imagingTime{acquisitionDuration*1e6}us_imaging{imagingOption}_{nOfAtoms}" + extraWord
simulationFolder = working_directory + f"/simulationImages/ODT/Yt{isotope}_{int(experimentDuration*1e6)}us_ODTDuration{ODTDuration*1e6}us_freeFlight{freeFlightTime*1e6}us_imagingTime{acquisitionDuration*1e6}us_imaging{imagingOption}_{nOfAtoms}"  + extraWord + "/simulation/"
baseFileName = "simulation"
blurFolder = "bigBlurs/"       # PSF along the camera axis
#backgroundNoiseFolder = "Orca_testing/shots_free_space_small_ROI/"
backgroundNoiseFolder = "Orca_testing/shots_free_space_big_ROI/"

if not os.path.exists(pictureFolder):
	os.makedirs(pictureFolder)
if not os.path.exists(simulationFolder):
	os.makedirs(simulationFolder)
    
    

'''-------------------------------ok, let's start------------------------------'''

G = Camera.blurFromImages(blurFolder)

magnificationGrid = pixelGrid(cameraSize[0],cameraSize[1],nPixels[0],nPixels[1], lambda xy, z: G(xy, z - lensDistance), magnification=magnification)
#quantumEfficiencyGrid = fixed_cMosGrid(cameraSize[0],cameraSize[1],randExtractor.randomLosts(1-totalEfficiency), backgroundNoiseFolder)
quantumEfficiencyGrid = fixedAdjustable_cMosGrid(cameraSize[0],cameraSize[1],nPixels[0],nPixels[1],randExtractor.randomLosts(1-totalEfficiency), backgroundNoiseFolder)
#quantumEfficiencyGrid = cMosGrid(cameraSize[0],cameraSize[1],nPixels[0],nPixels[1],randExtractor.randomLosts(1-totalEfficiency), backgroundNoiseFolder) #does not have spatial information on the noise
c = Camera(position=(0,0,lensDistance), 
		orientation=(0,-np.pi/2,0), 
		radius=lensRadius,
		pixelGrids=(magnificationGrid, quantumEfficiencyGrid))





repetitions = 2

for repeat in tqdm(range(repetitions), desc="Running simulation",mininterval=10,maxinterval=30):
    
	fileName = f"{baseFileName}_{repeat}.h5"
	imageFileName = f"{pictureFolder}/{fileName}"
	simulationFileName = f"{simulationFolder}/{fileName}"
	exp = trajlib.experiment() 
	expNoImaging = trajlib.experiment() 


	if not os.path.exists(simulationFileName):
		for i in range(nOfAtoms):
                
			atomPosition = np.random.normal(0, [x_rms, y_rms, z_rms], size=(3))
			atomVelocity = np.random.normal(0, [v_x_rms, v_y_rms, v_z_rms], size=(3))
			newAtom = trajlib.Ytterbium(*atomPosition, *atomVelocity, isotope=isotope)
			newAtomNoImaging = trajlib.Ytterbium(*atomPosition, *atomVelocity, isotope=isotope)

			exp.add_atom(newAtom)
			expNoImaging.add_atom(newAtomNoImaging)
            
      
		b, b1 = trajlib.Laser.counterPropagatingLasers(angle = 0, x0 = 0, wavelength=Lambda, Intensity=imgPower, w0 = (imgWaist/2,imgWaist/2), impulseDuration=impulseDuration, deadTime=0, halfFirstPeriod=True, initialTimeOff=imagingStartingTime, )
		exp.add_laser(b)
		exp.add_laser(b1)
        
        
		result = exp.new_run(time = experimentDuration, trapDuration= ODTDuration, trapType = 'ODT', trapFreqX = trapFreqX, trapFreqY = trapFreqY, trapFreqZ = trapFreqZ, max_dt =  0.01*max_dt,min_dt =  min_dt, max_dx = max_dx)
		resultNoImaging = expNoImaging.new_run(time = experimentDuration, trapDuration= ODTDuration, trapType = 'ODT', trapFreqX = trapFreqX, trapFreqY = trapFreqY, trapFreqZ = trapFreqZ, max_dt =  0.01*max_dt,min_dt =  min_dt, max_dx = max_dx)
		
        #positionAfterTimeOfFlight = exp.positionsAtTime(experimentDuration) * magnificationGrid.magnification
		#pixelPositionAfterTimeOfFlight = magnificationGrid._normalizeCoordinate(-positionAfterTimeOfFlight[:,0], positionAfterTimeOfFlight[:,1], removeOutOfBoundaryValues=False)
		
		metadata = dict(
			max_dt = max_dt,
			min_dt = min_dt,
			max_dx = max_dx,
            freeFlightTime = freeFlightTime,
            ODTDuration = ODTDuration,
			experimentDuration = experimentDuration,
            imagingTime = acquisitionDuration, 
            imagingOption = imagingOption,
			atoms_InitialCoordinates = exp.initialAtomPositions,
			atoms_InitialSpeeds = exp.initialAtomSpeeds,
			atom_isotope = isotope,
            atomNumber = nOfAtoms,
            laser_power = imgPower,
			laser_waist = imgWaist,
			laser_lambda = Lambda,
			laser_detuning = detuning,
			laser_frequency = freq,
			laser_impulseDuration = impulseDuration,
			initial_temperature = initialT,
            saturation = saturation,
            positionAfterODTDuration =[],
            #positionAfterTimeOfFlight = positionAfterTimeOfFlight,
			#pixelPositionAfterTimeOfFlight = pixelPositionAfterTimeOfFlight,
            
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
            
			ODT_waist = ODTWaist,
			ODT_intensity = ODTIntensity,
			ODT_lambda = ODTLambda,
			ODT_trapFreqX = trapFreqX,
            ODT_trapFreqY = trapFreqY,
            ODT_trapFreqZ = trapFreqZ,
            ellipticity = k,
            
            
            minMeanPlanarInterparticleSpacing = minMeanPlanarInterparticleSpacing,
            inverseFermiWV = fermiK**(-1),
            fermiEnergy = fermiEnergy,
            fermiTemperature = fermiTemperature,
            lastPositionsNoImaging = expNoImaging.lastPositons[-1,:,:], # correct the orientation to compare them with the camera images
            initialT = initialT
            
			
            
            
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
            magnification = magnification,
            minMeanPlanarInterparticleSpacing = minMeanPlanarInterparticleSpacing
            
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
                
    
	if computeCameraImages:
        
		startPositions, directions = exp.getScatteredPhotons(experimentDuration)      
		#atomInitialPositions = exp.lastPositons[0,:,:]*magnification/pixelSize
		atomInitialPositions = metadata['lastPositionsNoImaging']*magnification/pixelSize
		image = c.takePicture(startPositions, directions, plot=True, saveToFile=None, acquisitionDuration = experimentDuration, atomPositions = atomInitialPositions, scatterPlot = True,**metadata)
        






