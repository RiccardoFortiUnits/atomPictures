import Simulations_Libraries.trajectory_library as trajlib
import numpy as np
import matplotlib.pyplot as plt
from Camera import *
from scipy.stats import poisson
from scipy.optimize import curve_fit
import Simulations_Libraries.general_library as genlib
plt.ion()

showPlots = False

pictureFolder = "D:/simulationImages/10us_smallImages/pictures/"
simulationFolder = "D:/simulationImages/10us_smallImages/simulation/"
blurFolder = "blurs/"

waist = 1e-3 # m
# Isaturation_overArea = 1.4 #W/m^2
power = 10e-3# W               #40 * Isaturation_overArea * (waist/2)**2 * np.pi

nOfAtoms = 1
isotope = 174
baseAtom = trajlib.Ytterbium(0,0,0, 0,0,0,isotope=isotope)
detuning = 0#-5.5*trajlib.MHz
freq = trajlib.c/(baseAtom.transitions[0].Lambda) + detuning
lambd = trajlib.c/freq
Lambda = lambd#399e-9

dt = 1e-10 # excited state time ~5e-9 s
impulseDuration = 400e-9
experimentDuration = 10e-6 #2-50us

nPixels = 20
pixelSize = 4.6e-6
cameraSize = pixelSize*nPixels
lensDistance = 25.5e-3

if not os.path.exists(pictureFolder):
    os.makedirs(pictureFolder)
if not os.path.exists(simulationFolder):
    os.makedirs(simulationFolder)

# atomo in trap (add elastic force)
# (no check exit from trap)
# residual tra simulazione e dati sperimentali (sia first image after free space che second image after trap) ( chiedi a Sara)

def elasticForceFromTweezer(laserIntensity):
    k=laserIntensity
    def elasticForce(a : trajlib.Atom, *args):
        pos = a.position
        pos = -k * pos
        pos[2] -= trajlib.g
        return pos
    return elasticForce


G = Camera.blurFromImages(blurFolder)
#'''
magnificationGrid = pixelGrid(cameraSize,cameraSize,nPixels,nPixels, lambda xy, z: G(xy, z - lensDistance))
# quantumEfficiencyGrid = pixelGrid(cameraSize,cameraSize,nPixels,nPixels, randExtractor.randomLosts(0.1))
quantumEfficiencyGrid = refreshing_cMosGrid(cameraSize,cameraSize,nPixels,nPixels, randExtractor.randomLosts(0.1), "Orca_testing/shots/", imageStart = (10,10), imageSizes = (-10,-10))
c = Camera(position=(0,0,lensDistance), 
        orientation=(0,-np.pi/2,0), 
        radius=16e-3,
        pixelGrids=(magnificationGrid, quantumEfficiencyGrid))

for repeat in range(500):
    print(repeat)
    baseFileName = f"simulation_atomOnCenterPixel_{repeat}.h5"
    imageFileName = f"{pictureFolder}/{baseFileName}"
    simulationFileName = f"{simulationFolder}/{baseFileName}"
    exp = trajlib.experiment()
    if not os.path.exists(simulationFileName):

        for i in range(nOfAtoms):
            exp.add_atom(trajlib.Ytterbium(*((np.random.random((3,)))*np.array([-pixelSize,pixelSize,0])), 0,0,0,isotope=isotope))
            # exp.add_atom(trajlib.Ytterbium(0,0,0, 0,0,0,isotope=174))
        b = trajlib.Laser(0,0, Lambda, power, (waist/2,waist/2), switchingTimes =      [-impulseDuration / 2, impulseDuration, impulseDuration])
        b1 = trajlib.Laser(np.pi,0, Lambda, power, (waist/2,waist/2), switchingTimes = [ impulseDuration / 2, impulseDuration, impulseDuration])
        exp.add_laser(b)
        exp.add_laser(b1)
        # result = exp.run(experimentDuration, dt)#2-50us        
        result = exp.new_run(experimentDuration)
        
        metadata = dict(
            dt = dt,
            experimentDuration = experimentDuration,
            atoms_InitialCoordinates = exp.initialAtomPositions,
            atoms_InitialSpeeds = exp.initialAtomSpeeds,
            atom_isotope = isotope,
            laser_power = power,
            laser_waist = waist,
            laser_lambda = Lambda,
            laser_detuning = detuning,
            laser_frequency = freq,
            laser_impulseDuration = impulseDuration,
            lens0_radius = c.radius,
            lens0_distance = lensDistance,
            camera_pixelSize = pixelSize,
            camera_pixelNumber = nPixels,
            camera_blurImagesPath = blurFolder,
        )        
        exp.saveAcquisition(simulationFileName, **metadata)
    else:
        metadata = exp.loadAcquisition(simulationFileName)
    if showPlots:
        exp.plotTrajectories()
    if not os.path.exists(imageFileName):
        startPositions, directions = exp.getScatteredPhotons()
        image = c.takePicture(startPositions, directions, plot=showPlots, saveToFile=imageFileName, **metadata)
        if showPlots:
            exp.plotTrajectoriesAndCameraAcquisition(c)
            input("Press Enter to close the plot windows")
            plt.close('all')