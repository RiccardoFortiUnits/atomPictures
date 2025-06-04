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

scaling = 1#                                                                                              .5

twzWaist = 540e-9
twzIntensity = scaling * 2.24e-3*trajlib.kB # Tweezer depth in J
twzLambda = 532e-9 # tweezer wavelength in m
trapFreq_r = np.sqrt(scaling) * 2*np.pi*140e3
baseFreq_z = 30e3 if usedCase!=zLattice else 200e3
trapFreq_z = np.sqrt(scaling) * 2*np.pi*baseFreq_z
n_0 = trajlib.n_T(trapFreq_r,initialT)
radius_rms, v_r_rms = trajlib.extract_coordinates_rms(T = 2/3*initialT,omega=trapFreq_r, m=baseAtom.m, N_atoms=1)
z_rms, v_z_rms = trajlib.extract_coordinates_rms(T = 1/3*initialT,omega=trapFreq_z, m=baseAtom.m, N_atoms=1)
# radius_rms = np.sqrt(trajlib.hbar/(2*baseAtom.m*trapFreq_r)*(2*n_0+1))  # initial RMS position (radial)
# z_rms = np.sqrt(trajlib.hbar/(2*baseAtom.m*trapFreq_z)*(2*n_0+1))  # initial RMS position (axial)
# v_r_rms = np.sqrt(trajlib.hbar*trapFreq_r/(2*baseAtom.m)*(2*n_0+1))  # initial RMS velocity (radial)
# v_z_rms = np.sqrt(trajlib.hbar*trapFreq_z/(2*baseAtom.m)*(2*n_0+1))  # initial RMS velocity (axial)
if usedCase == fixedAtom:
    radius_rms = 0
    z_rms = 0
    v_r_rms = 0
    v_z_rms = 0
    trajlib.experiment.nextPointInTrajectory = lambda *x,**y: None

'''-------------------------------pixels and camera------------------------------'''
pixelScale = 1#                                                                       10.
pixelType = "" if pixelScale == 1 else f"_{int(pixelScale)}reduction"
nPixels = np.array([20,20 * ((nOfAtoms+1)//2)]) * int(pixelScale)
pixelSize = 4.6e-6 / pixelScale
magnification = 8
cameraSize = pixelSize*nPixels
atomSpaceSize = cameraSize / magnification
defocus = 0e-6
lensDistance = 25.5e-3
lensRadius = 16e-3
quantumEfficiency = .83
objectiveTransmission = .76
dicroicReflection = .9
filterTransmission = .9**2
totalEfficiency = quantumEfficiency * objectiveTransmission * dicroicReflection * filterTransmission
atomDistance = [.5e-6]
fillProbability = .6
for distance in atomDistance:
    atomSpaceSize = distance
    '''-------------------------------timings------------------------------'''
    max_dt = 1e-8
    min_dt = 1e-10
    max_dx = twzWaist / 10
    impulseDuration = 400e-9
    experimentDuration = 20e-6 #2-50us
    acquisitionDuration = 12e-6#experimentDuration * 1#if you want, you can do more than one acquisition time for the same simulation (of course the simulation has to be long enough)

    '''-------------------------------folders and files------------------------------'''
    pictureFolder = f"D:/simulationImages/CloserAtoms_{distance*1e6}um_Yt{isotope}_{int(experimentDuration*1e6)}us_{nOfAtoms}tweezerArray{usedCase}/{int(acquisitionDuration*1e6)}us_pictures{pixelType}/"
    simulationFolder = f"D:/simulationImages/CloserAtoms_{distance*1e6}um_Yt{isotope}_{int(experimentDuration*1e6)}us_{nOfAtoms}tweezerArray{usedCase}/simulation/"
    baseFileName = "simulation"
    blurFolder = "D:/simulationImages/blurs/399nm"

if not os.path.exists(pictureFolder):
    os.makedirs(pictureFolder)
if not os.path.exists(simulationFolder):
    os.makedirs(simulationFolder)
    if not os.path.exists(pictureFolder):
        os.makedirs(pictureFolder)
    if not os.path.exists(simulationFolder):
        os.makedirs(simulationFolder)

    '''-------------------------------ok, let's start------------------------------'''

    if nOfAtoms==1:
        tweezerBaseCenter = np.zeros((1,3))
    else:
        # tweezerBaseCenter = np.stack((np.linspace(atomSpaceSize[1]*.4,-atomSpaceSize[1]*.4,nOfAtoms), np.linspace(-atomSpaceSize[0]*.1,atomSpaceSize[0]*.1,nOfAtoms), np.zeros(nOfAtoms)), axis=-1)
        tweezerBaseCenter = np.stack((pixelSize/2/magnification+np.linspace(atomSpaceSize/2,-atomSpaceSize/2,nOfAtoms)*(nOfAtoms-1), np.zeros(nOfAtoms), np.zeros(nOfAtoms)), axis=-1)

    # if useFixedTweezers:
    #     tweezerBaseCenter += (np.random.random(tweezerBaseCenter.shape) * np.array([-pixelSize,pixelSize,0]) / magnification)

    def elasticForceFromTweezer(a : trajlib.Atom, *args):
        x,y,z = a.position - a.tweezerPosition

        a_x = - trapFreq_r**2 * x
        a_y = - trapFreq_r**2 * y
        a_z = - trapFreq_z**2 * z
        f = np.array([a_x,a_y,a_z - trajlib.g])

        return f


    G = Camera.blurFromImages(blurFolder)
    #'''
    magnificationGrid = pixelGrid(cameraSize[0],cameraSize[1],nPixels[0],nPixels[1], lambda xy, z: G(xy, z - lensDistance), magnification=magnification)
    quantumEfficiencyGrid = refreshing_cMosGrid(cameraSize[0],cameraSize[1],nPixels[0],nPixels[1], randExtractor.randomLosts(1-totalEfficiency), "Orca_testing/shots/", imageStart = (10,10), imageSizes = (-10,-10))
    c = Camera(position=(0,0,lensDistance + defocus), 
            orientation=(0,-np.pi/2,0), 
            radius=lensRadius,
            pixelGrids=(magnificationGrid, quantumEfficiencyGrid))

    randomPosition = partial(np.random.normal, loc=0)

    for repeat in range(1000):
        print(repeat)
        fileName = f"{baseFileName}_{repeat}.h5"
        imageFileName = f"{pictureFolder}/{fileName}"
        simulationFileName = f"{simulationFolder}/{fileName}"
        # exp = trajlib.experiment()
        exp = trajlib.experiment(elasticForceFromTweezer)
        if not os.path.exists(simulationFileName):
            filling = np.random.random(nOfAtoms) <= fillProbability
            while filling.sum() < 1:
                filling = np.random.random(nOfAtoms) <= fillProbability
            for i in range(nOfAtoms):
                if filling[i]:
                    tweezerPosition = tweezerBaseCenter[i]
                    if not useFixedTweezers:
                        tweezerPosition += np.random.random(3)*np.array([-pixelSize,pixelSize,0])# + np.array([i*pixelSize*10-cameraSize[1]/2+pixelSize*5,i*pixelSize*.5-pixelSize*5,0])
                    atomPosition = tweezerPosition + np.concatenate((np.random.normal(0, radius_rms, 2),np.random.normal(0, z_rms, 1)))
                    atomVelocity = np.concatenate((np.random.normal(0, v_r_rms, 2),np.random.normal(0, v_z_rms, 1)))
                    newAtom = trajlib.Ytterbium(*atomPosition, *atomVelocity, isotope=isotope)
                    newAtom.tweezerPosition = tweezerPosition
                    exp.add_atom(newAtom)
                # exp.add_atom(trajlib.Ytterbium(0,0,0, 0,0,0,isotope=174))
            b = trajlib.Laser(0,0, Lambda, imgPower, (imgWaist/2,imgWaist/2), switchingTimes =      [-impulseDuration / 2, impulseDuration, impulseDuration])
            b1 = trajlib.Laser(np.pi,0, Lambda, imgPower, (imgWaist/2,imgWaist/2), switchingTimes = [ impulseDuration / 2, impulseDuration, impulseDuration])
            exp.add_laser(b)
            exp.add_laser(b1)
            # result = exp.run(experimentDuration, dt)#2-50us        
            result = exp.new_run(experimentDuration, max_dt, min_dt, max_dx)
            
            metadata = dict(
                max_dt = max_dt,
                min_dt = min_dt,
                max_dx = max_dx,
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
                atomDistance = distance,
                atomFilling = filling,

                lens0_radius = c.radius,
                lens0_distance = lensDistance + defocus,
                defocus = defocus,
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
                tweezer_centers = [a.tweezerPosition for a in exp.atoms]
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
            if useFixedTweezers:
                tweezerBaseCenter = metadata["tweezer_centers"]
        if showPlots:
            exp.plotTrajectories()
        if not os.path.exists(imageFileName):
            startPositions, directions = exp.getScatteredPhotons(acquisitionDuration)
            image = c.takePicture(startPositions, directions, plot=showPlots, saveToFile=imageFileName, acquisitionDuration = acquisitionDuration, **metadata)
            # a=cameraAtomImages(pictureFolder)
            # plt.imshow(a.averageImage())
            if showPlots:
                exp.plotTrajectoriesAndCameraAcquisition(c)
                input("Press Enter to close the plot windows")
                plt.close('all')