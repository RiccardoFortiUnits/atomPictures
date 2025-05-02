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


'''-------------------------------atom------------------------------'''
nOfAtoms = 1
isotope = 174
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

exp_diff_light_shift = 20e6 # Hz/V, measured on 22/01/2025
light_shift_K = exp_diff_light_shift*trajlib.h/trajlib.kB
U_1 = twzWaist - light_shift_K*trajlib.kB*0.4 # trap depth 1P1

z_R = trajlib.Rayleigh_range(twzWaist, twzLambda)
w_r = trajlib.omega_r(twzIntensity, twzWaist, baseAtom)
w_z = trajlib.omega_z(twzIntensity, z_R, baseAtom)
n_0 = trajlib.n_T(w_r,initialT)
radius_rms = np.sqrt(trajlib.hbar/(2*baseAtom.m*w_r)*(2*n_0+1))  # initial RMS position (radial)
z_rms = np.sqrt(trajlib.hbar/(2*baseAtom.m*w_z)*(2*n_0+1))  # initial RMS position (axial)
v_r_rms = np.sqrt(trajlib.hbar*w_r/(2*baseAtom.m)*(2*n_0+1))  # initial RMS velocity (radial)
v_z_rms = np.sqrt(trajlib.hbar*w_z/(2*baseAtom.m)*(2*n_0+1))  # initial RMS velocity (axial)

U_eff = p_g*twzIntensity + p_e*U_1# redefine U0 as an effective trap depth because of different g and e polarizability
w_r = trajlib.omega_r(U_eff,twzWaist, baseAtom)
w_z = trajlib.omega_z(U_eff, z_R, baseAtom)

'''-------------------------------timings------------------------------'''
max_dt = 1e-5
min_dt = 1e-10
max_dx = twzWaist / 10
impulseDuration = 400e-9
experimentDuration = 10e-6 #2-50us

'''-------------------------------pixels------------------------------'''
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

def elasticForceFromTweezer(a : trajlib.Atom, *args):
    x,y,z = a.position - a.tweezerPosition
    # a_x = -np.exp(-a.m*w_r**2/(2*U_eff)*a.X**2) * w_r**2*a.X
    # a_y = -np.exp(-a.m*w_r**2/(2*U_eff)*a.Y**2) * w_r**2*a.Y
    # a_z = -np.exp(-a.m*w_z**2/(2*U_eff)*a.Z**2) * w_z**2*a.Z
    
    # a_x = -4*a.X* U_eff/(twzWaist**2*trajlib.zeta_dependence(a.Z,twzWaist,twzLambda)**2)*np.exp(-2*(a.X**2+a.Y**2)/(twzWaist**2*trajlib.zeta_dependence(a.Z,twzWaist,twzLambda)))/a.m
    # a_y = -4*a.Y* U_eff/(twzWaist**2*trajlib.zeta_dependence(a.Z,twzWaist,twzLambda)**2)*np.exp(-2*(a.X**2+a.Y**2)/(twzWaist**2*trajlib.zeta_dependence(a.Z,twzWaist,twzLambda)))/a.m
    # a_z = -2*(twzLambda/(np.pi*twzWaist**2))**2 * a.Z* U_eff/(trajlib.zeta_dependence(a.Z,twzWaist,twzLambda)**2)*np.exp(-2*(a.X**2+a.Y**2)/(twzWaist**2*trajlib.zeta_dependence(a.Z,twzWaist,twzLambda)))*(1+2*(a.X**2+a.Y**2)/trajlib.zeta_dependence(a.Z,twzWaist,twzLambda)/twzWaist**2)/a.m
    omega_r = 200e3
    omega_z = 150e3
    a_x = - omega_r**2 * x
    a_y = - omega_r**2 * y
    a_z = - omega_z**2 * z
    f = np.array([a_x,a_y,a_z - trajlib.g])

    return f


G = Camera.blurFromImages(blurFolder)
#'''
magnificationGrid = pixelGrid(cameraSize,cameraSize,nPixels,nPixels, lambda xy, z: G(xy, z - lensDistance))
# quantumEfficiencyGrid = pixelGrid(cameraSize,cameraSize,nPixels,nPixels, randExtractor.randomLosts(0.1))
quantumEfficiencyGrid = refreshing_cMosGrid(cameraSize,cameraSize,nPixels,nPixels, randExtractor.randomLosts(0.1), "Orca_testing/shots/", imageStart = (10,10), imageSizes = (-10,-10))
c = Camera(position=(0,0,lensDistance), 
        orientation=(0,-np.pi/2,0), 
        radius=16e-3,
        pixelGrids=(magnificationGrid, quantumEfficiencyGrid))

randomPosition = partial(np.random.normal, loc=0)

for repeat in range(500):
    print(repeat)
    baseFileName = f"xsimulation_atomOnCenterPixel_{repeat}.h5"
    imageFileName = f"{pictureFolder}/{baseFileName}"
    simulationFileName = f"{simulationFolder}/{baseFileName}"
    # exp = trajlib.experiment()
    exp = trajlib.experiment(elasticForceFromTweezer)
    if not os.path.exists(simulationFileName):

        for i in range(nOfAtoms):
            tweezerPosition = np.random.random(3)*np.array([-pixelSize,pixelSize,0])
            atomPosition = tweezerPosition + np.concatenate((np.random.normal(pixelSize/2, radius_rms, 2),np.random.normal(0, z_rms, 1)))
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

            lens0_radius = c.radius,
            lens0_distance = lensDistance,
            camera_pixelSize = pixelSize,
            camera_pixelNumber = nPixels,
            camera_blurImagesPath = blurFolder,
            tweezer_waist = twzWaist,
            tweezer_intensity = twzIntensity,
            tweezer_lambda = twzLambda,
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