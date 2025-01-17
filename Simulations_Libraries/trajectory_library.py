#################
#    ArqusLab   #
#    March 2023 #             
#################

# This library has the functions for simulating the trajectories of atoms

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.colors as mcolors
import matplotlib.pylab as pl
import seaborn as sns
import scipy
from sympy.physics.wigner import clebsch_gordan, wigner_3j,wigner_6j
import scipy.constants as const
import numba as nu
from tqdm.notebook import tqdm
import sys
sys.path.insert(0, '//ARQUS-NAS/ArQuS Shared/Simulations/MOT capture simulation/Simulations_11_2024/')

import Simulations_Libraries.general_library as genlib


# Universal Constants
c = const.physical_constants["speed of light in vacuum"][0]
h = const.physical_constants["Planck constant"] [0]
hbar = h/(2*np.pi)
kB = const.physical_constants["Boltzmann constant"][0]
eps_0 = const.physical_constants["electric constant"][0]
e = const.physical_constants["elementary charge"][0]
u = const.physical_constants["atomic mass constant"][0]
g = const.physical_constants["standard acceleration of gravity"][0]
mu_0 = const.physical_constants["vacuum mag. permeability"][0]*10**4 #(in Gauss)
Debye = 3.33564*10**-30             # C m
muB = 1.399*10 ** 6;                # Bohr's magneton [Hz/G]
mu_N = 5.050783699 * 10**-27        # Nuclear magneton J/T
mu_NG = mu_N*10**-4/hbar            # Nuclear magneton Hz/G
mu_n = mu_NG *(-1.913)              # Neutron magnetic moment Hz/G
a_0 = 5.29*10**-11
Eh = 4.4*10**-18
au = 1.64877727436* 10**-41         # atomic eletric polarizability C^2 m^2 J^-1
nm = 10**-9
MHz = 10**6
kHz = 10**3

gJ_1S0_3P1 = 1.493

TOTAL_LENGTH = 37*10**-2 # Length of the system bewteen end of AoSense and center of the glass cell


class Atom:
    def __init__(self, x,y,z,vx,vy,vz):
        self.X = x
        self.Y = y
        self.Z = z
        self.vx = vx
        self.vy = vy
        self.vz = vz
        

########################################
##### Generation of initial atomic sample ######
########################################
def CreateAtoms (N, m, Show, AtomicBeamRadius = 3 *10**-3, Tube_Diameter = 6*10**-3, Tube_Length = 106*10**-3, v_ax = 40, T_perp = 3 * 10**-3):
    '''
    CreateAtoms (N, m, Show)
    Returns N atoms with  x = (x, y, z) and velocities v = (vx, vy, vz) 
    --------------------
    Arguments: 
    - N: number of atoms to be generated
    - m: mass (float)
    - Show: show plots (bool)
    '''
    Atoms = []
    
    x = GenerateInitialPositions(N, Show, AtomicBeamRadius, Tube_Diameter)
    v = GenerateInitialVelocities(N, m, Show, Tube_Diameter, Tube_Length, v_ax, T_perp)
    
    for i in range(N):
        Atoms.append(Atom(x[i,0], x[i,1],x[i,2],v[i,0],v[i,1],v[i,2])) #(x,y,z,vx,vy,vz)
    return Atoms

def GenerateInitialPositions(N, Show, AtomicBeamRadius = 3 *10**-3 ,Tube_Diameter = 6*10**-3):
    '''
    Atomic beam radius and tube diameter are taken from AOSense CAD
    '''
    # Generate a Gaussian disribution of positions
    # Radii
    r0 = np.zeros(N)
    phi0 = np.zeros(N)
    
    i = 0
    while i in range (N):
        r0[i] = abs(np.random.normal(0,AtomicBeamRadius))
        phi0[i] = np.random.uniform(0,2*np.pi)
        if r0[i] < Tube_Diameter/2:
            i = i+1
    if Show:
        f, axs = plt.subplots(1,2,figsize=(14,6))
        axs[0].hist(r0*1000,bins=15)
        axs[0].set_xlabel('Initial distance from beam axis (mm)')
        axs[0].axvline(AtomicBeamRadius*1000,color = 'r')

        sns.kdeplot(r0*np.sin(phi0)*1000,r0*np.cos(phi0)*1000, shade=True, ax=axs[1], cmap = 'Blues')
        axs[1].set_xlabel("$y$ (mm)")
        axs[1].set_ylabel("$z$ (mm)")
        axs[1].set_xlim(-5,5)
        axs[1].set_ylim(-5,5)
        plt.show()

    # Assign positions to atoms and store them in x_initial for quicky revocer
    x_initial = np.zeros([N,3])
    for i in range(N):
        x_initial[i,0] = 0
        x_initial[i,1] = r0[i]*np.cos(phi0[i])
        x_initial[i,2] = r0[i]*np.sin(phi0[i])
    return x_initial


def GenerateInitialVelocities(N, m, Show, Tube_Diameter = 6*10**-3, Tube_Length = 106*10**-3, v_ax = 40, T_perp = 3 * 10**-3):
    '''
    Atomic beam radius and tube diameter are taken from AOSense CAD
    Typical velocities from the datasheet
    '''
    T_ax = T_givenV(v_ax, m)
    TubeCut = (Tube_Diameter/2)/Tube_Length
    v_ax_test = np.linspace(0,100,N)
    v_ax = np.zeros(N)
    v_perp = np.zeros(N)
    # Generate axial velocity sampling an atomic beam velocity distribution
    i = 0
    while (i < N):
        vel = np.random.uniform(0,100)
        test = np.random.uniform(0,1)
        if (test < AtomicBeam_VelocityDistr(vel,T_ax,m)/np.amax(AtomicBeam_VelocityDistr(v_ax_test,T_ax,m))): 
            v_ax[i] = vel
            i = i+1
    i = 0
    while (i < N):
        v = np.linspace(-2,2,100)
        vel = np.random.uniform(0,2)
        test = np.random.uniform(0,1)
        # Sample distribution
        if (test < MaxwellBoltzmann_VelocityDistr(vel,T_perp,m)/np.amax(MaxwellBoltzmann_VelocityDistr(v,T_perp,m))): 
            # Check that the tube is not cutting
            if (vel/v_ax[i] < TubeCut):
                v_perp[i] = vel
                i = i+1
    v_initial = np.zeros([N,3])
    for i in range(N):
        phi_vel = np.random.uniform(0,2*np.pi)
        v_initial[i,0] = v_ax[i]
        v_initial[i,1] = v_perp[i]*np.cos(phi_vel)
        v_initial[i,2] = v_perp[i]*np.sin(phi_vel)

    if Show:
        fig, axs = plt.subplots(ncols=2,nrows=2,figsize = (15,10))
        axs[0][0].hist(v_ax,bins=10)
        axs[0][0].plot(v_ax_test,AtomicBeam_VelocityDistr(v_ax_test,T_ax,m)/np.amax(AtomicBeam_VelocityDistr(v_ax_test,T_ax,m))*(N/5))
        axs[0][0].set_xlabel('$v_{ax}$ (m/s)')
        sns.kdeplot(np.abs(v_ax),np.abs(v_perp), shade=True, ax=axs[0][1], cmap = 'Blues')
        axs[0][1].set_xlabel("$v_{ax}$ (m/s)")
        axs[0][1].set_ylabel("$v_{perp}$ (m/s)")
        axs[0][1].plot(np.linspace(0,50),np.linspace(0,50)*TubeCut, color = 'red', linestyle = '--', label = 'Tube cut')
        axs[0][1].scatter(40,v_givenT(T_perp,m), marker = '*', color = 'red',label = 'Datasheet') 
        axs[0][1].legend(loc = 'upper right')
        axs[1][0].hist(v_initial[:,0],bins=10,label = '$v_{x}$')
        axs[1][0].set_xlabel('$v_{x}$ (m/s)')
        axs[1][1].hist(v_initial[:,1],bins=10,label = '$v_{y}$')
        axs[1][1].set_xlabel('$v_{y}$ (m/s)')
        plt.show() 
    return v_initial

def AtomicBeam_VelocityDistr (v,T,m):
    '''
    Compute the velocity distribution of an atomic beam with center velocity v, temperature T and mass m
    '''
    alpha = np.sqrt(2*kB*T/m)
    f = 4/np.sqrt(np.pi) * 1/alpha *v**3 * np.exp(-(v/alpha)**2)
    return f

def MaxwellBoltzmann_VelocityDistr (v,T,m):
    f = 4*np.pi*(m/(2*np.pi*kB*T))**(3/2)*v**2*np.exp(-m*v**2/(2*kB*T))
    return f;

def v_givenT (T,m):
    return np.sqrt(3*kB*T/m)

def T_givenV (v,m):
    return m*v**2/(3*kB)

##############################################################
################## Atom motion simulation ####################
##############################################################
# @nu.jit()
def OneStep(A,Dv,Dt):
    A.X +=  A.vx*Dt
    A.Y +=  A.vy*Dt 
    A.Z +=  A.vz*Dt - 0.5*g*Dt**2
    A.vx += Dv[0]
    A.vy += Dv[1]  
    A.vz += Dv[2] - g*Dt # g is positive!

# @nu.jit(nopython=True)
def PropagateUntilGlassCell (Atoms, Total_length = 37*10**-2):
    '''
    Propagate atoms until the glass cell, where the real simulation starts
    ---------
    Total length is given by the length of our exp. setup
    '''
    N = len(Atoms)
    x_cell = Total_length-20*10**-3-20*10**-3
    tf = 100*10**-3
    dt = 1*10**-5
    Nsteps = int(tf/dt)
    for n in tqdm(range(N)):
        for i in range(Nsteps-1):
            OneStep(Atoms[n], np.array([0,0,0]), dt)
            if Atoms[n].X > x_cell: # Atom is entering the cell
                break 
    print(f'Atoms have propagated freely up to {x_cell*10**+2} cm')        
    return 


def SavePositions (Atoms):
    '''
    SavePositions (Atoms):
    Returns arrays filled with position and velocities of Atoms
    '''
    N = len(Atoms)
    x = np.zeros([N,3])
    v = np.zeros([N,3])
    for i in range(N):
        x[i,0] = Atoms[i].X
        x[i,1] = Atoms[i].Y
        x[i,2] = Atoms[i].Z
        v[i,0] = Atoms[i].vx
        v[i,1] = Atoms[i].vy
        v[i,2] = Atoms[i].vz
    return x, v


################################################################
################# Big simulation function ######################
################################################################

def crossed_beams_scattering_prob(blue_transition,tau_blue,blue_lambda,E01_CB,E02_CB,k1_CB,k2_CB,A,pos_x,pos_y,pos_z):
    P_sc_CB = np.zeros(2)
    P_sc_CB[0] = tau_blue*ScatteringRate_2Level_Doppler3D_Bfield(blue_transition,blue_lambda,E01_CB[pos_x,pos_y,pos_z],k1_CB,[A.vx,A.vy,A.vz],0)
    P_sc_CB[1] = tau_blue*ScatteringRate_2Level_Doppler3D_Bfield(blue_transition,blue_lambda,E02_CB[pos_x,pos_y,pos_z],k2_CB,[A.vx,A.vy,A.vz],0)
    return P_sc_CB

def MOT_beams_scattering_prob(green_transition,tau_green,G_lambd_HOR,G_lambd_VER,E01_G,E02_G,E03_G,k1_G,k2_G,k3_G,newCoordinates,B_act,A,pos_x,pos_y,pos_z):
    P_sc_MOT = np.zeros(5)
    P_sc_MOT[0] = tau_green*ScatteringRate_2Level_Doppler3D_Bfield(green_transition,G_lambd_HOR,E01_G[pos_x,pos_y,pos_z],k1_G,[A.vx,A.vy,A.vz],np.sign(newCoordinates[0])*gJ_1S0_3P1*muB*B_act)
    P_sc_MOT[1] = tau_green*ScatteringRate_2Level_Doppler3D_Bfield(green_transition,G_lambd_HOR,E01_G[pos_x,pos_y,pos_z],-k1_G,[A.vx,A.vy,A.vz],-np.sign(newCoordinates[0])*gJ_1S0_3P1*muB*B_act)
    P_sc_MOT[2] = tau_green*ScatteringRate_2Level_Doppler3D_Bfield(green_transition,G_lambd_HOR,E02_G[pos_x,pos_y,pos_z],k2_G,[A.vx,A.vy,A.vz],np.sign(newCoordinates[1])*gJ_1S0_3P1*muB*B_act)
    P_sc_MOT[3] = tau_green*ScatteringRate_2Level_Doppler3D_Bfield(green_transition,G_lambd_HOR,E02_G[pos_x,pos_y,pos_z],-k2_G,[A.vx,A.vy,A.vz],-np.sign(newCoordinates[1])*gJ_1S0_3P1*muB*B_act)
    P_sc_MOT[4] = tau_green*ScatteringRate_2Level_Doppler3D_Bfield(green_transition,G_lambd_VER,E03_G[pos_x,pos_y,pos_z],k3_G,[A.vx,A.vy,A.vz],np.sign(newCoordinates[2])*gJ_1S0_3P1*muB*B_act)
    return P_sc_MOT

def Trajectory_simulation_only_5BMOT(N_steps,t_end,dt_no_scatter,x,y,z,Total_Length,B,G_lambd_HOR,G_lambd_VER,E01_G,E02_G,E03_G,k1_G,k2_G,k3_G,A,GS_Transitions,tau_excited,m):
    x_traj = []
    v_traj = []
    P_sc_MOT = np.zeros(5)    
    which = []
    Green_photons = 0
    t = 0    
    for step in range(N_steps):
        MOT_scatter = False
        MOT_try = False
        Dv = np.zeros(3)
        v_traj.append([A.vx,A.vy,A.vz])
        x_traj.append([A.X,A.Y,A.Z])
        pos_x = genlib.Find_Target(A.X,x)
        pos_y = genlib.Find_Target(A.Y,y)
        pos_z = genlib.Find_Target(A.Z,z)

        # Find the magnetic shift: consider that atoms scattering from opposite beams will see opposite magnetic shifts
        B_act = B[pos_x,pos_y,pos_z]
        # Decide from which beam the atom tries to absorb a photon
        newCoordinates = np.asarray(CoordinateChange((x[pos_x],y[pos_y],z[pos_z]),3*np.pi/4,Total_Length))
        # First: compute P_sc for all beams
        P_sc_MOT[0] = tau_excited*ScatteringRate_2Level_Doppler3D_Bfield(GS_Transitions[1],G_lambd_HOR,E01_G[pos_x,pos_y,pos_z],k1_G,[A.vx,A.vy,A.vz],np.sign(newCoordinates[0])*gJ_1S0_3P1*muB*B_act)
        P_sc_MOT[1] = tau_excited*ScatteringRate_2Level_Doppler3D_Bfield(GS_Transitions[1],G_lambd_HOR,E01_G[pos_x,pos_y,pos_z],-k1_G,[A.vx,A.vy,A.vz],-np.sign(newCoordinates[0])*gJ_1S0_3P1*muB*B_act)
        P_sc_MOT[2] = tau_excited*ScatteringRate_2Level_Doppler3D_Bfield(GS_Transitions[1],G_lambd_HOR,E02_G[pos_x,pos_y,pos_z],k2_G,[A.vx,A.vy,A.vz],np.sign(newCoordinates[1])*gJ_1S0_3P1*muB*B_act)
        P_sc_MOT[3] = tau_excited*ScatteringRate_2Level_Doppler3D_Bfield(GS_Transitions[1],G_lambd_HOR,E02_G[pos_x,pos_y,pos_z],-k2_G,[A.vx,A.vy,A.vz],-np.sign(newCoordinates[1])*gJ_1S0_3P1*muB*B_act)
        P_sc_MOT[4] = tau_excited*ScatteringRate_2Level_Doppler3D_Bfield(GS_Transitions[1],G_lambd_VER,E03_G[pos_x,pos_y,pos_z],k3_G,[A.vx,A.vy,A.vz],np.sign(newCoordinates[2])*gJ_1S0_3P1*muB*B_act)
        #----------------------------------------------------------------------------
        # Decide from which beam you want to scatter from:
        # Roll a loaded die: divide the (0,1) interval into unequal intervals
        Psc_combined = P_sc_MOT
        Weight_tot = np.sum(Psc_combined)/np.mean(Psc_combined)
        beam_guess = np.random.uniform(0,1)
        weights = Psc_combined/np.mean(Psc_combined)/Weight_tot   
        if np.max(Psc_combined) < 10**-3:
            which.append(0)
        else:
            if beam_guess < weights[0]:
                MOTBeam = 0
                k_MOT = k1_G
                MOT_try = True
            elif beam_guess > weights[0] and beam_guess < np.sum(weights[:2]):
                MOTBeam = 1
                k_MOT = -k1_G
                MOT_try = True
            elif beam_guess > np.sum(weights[:2]) and beam_guess < np.sum(weights[:3]):
                MOTBeam = 2
                k_MOT = k2_G
                MOT_try = True
            elif beam_guess > np.sum(weights[:3]) and beam_guess < np.sum(weights[:4]):
                MOTBeam = 3
                k_MOT = -k2_G
                MOT_try = True
            elif beam_guess > np.sum(weights[:4]):
                MOTBeam = 4
                k_MOT = k3_G
                MOT_try = True 
            else:
                print('Houston')
                print(beam_guess)
                print(weights)
                break
        if MOT_try == True: # Try to scatter from the MOT beams
            which.append(1)
            if np.random.uniform(0,1) < P_sc_MOT[MOTBeam]:
                Dv += hbar*k_MOT/m+SpontaneousEmission(GS_Transitions[1].Lambda,m)
                Green_photons += 1
                MOT_scatter = True # the atom has scattered from the crossed beams
        #-------------------------------------------------------------------------------
        # The atom has attempted scattering from both crossed beams and MOT: it goes forward
        if MOT_scatter == True: 
            # if the atom has scattered from the MOT, it is excited to the excited state and it spends there a time dt before being able to scatter again 
            Dt = tau_excited
        else:
            Dt = dt_no_scatter
        OneStep(A,Dv,Dt)    
        t += Dt
        #-------------------------------------------------------------------------------
        # Break conditions:  
        if  t >= t_end:
            # print(f'Atom time finished at step {step}')
            captured = 1
            break
        elif(np.sqrt(A.Y**2+(A.X-Total_Length)**2)>15*10**-3 and A.X>Total_Length):
            # print(f'Atom out horizontally at step {step}')
            captured = 0
            break
        elif np.abs(A.Z) > 10*10**-3:
            # print(f'Atom out vertically at step {step}')
            captured = 0
            break
        elif A.X < 350e-3 and A.vx < 0:
            # print(f'Atom going back at step {step}')
            captured = 0
            break
        else:
            captured = 0
    x_final = np.asarray(x_traj)[-1,:]*1e3
    z_avg = np.mean(np.asarray(x_traj)[:,2])*1e3
    z_std = np.std(np.asarray(x_traj)[:,2])*1e3
    z_excursion = (np.max(np.asarray(x_traj)[:,2])-np.min(np.asarray(x_traj)[:,2]))*1e3
    v_final = np.asarray(v_traj)[-1,:]
    abs_vz_avg = np.mean(np.abs(np.asarray(v_traj)[:,2]))
    return t, x_final, z_avg, z_std, z_excursion, z_excursion, v_final, abs_vz_avg, Green_photons, captured

def Trajectory_simulation_full(N_steps,t_end,dt_no_scatter,x,y,z,Total_Length,MOT_VerSize,MOT_HorSize,B,Blue_lambd,E01_CB,E02_CB,k1_CB,k2_CB,G_lambd_HOR,G_lambd_VER,E01_G,E02_G,E03_G,k1_G,k2_G,k3_G,A,GS_Transitions,tau_green,tau_blue,m):
    x_traj = np.zeros([N_steps, 3])
    v_traj = np.zeros([N_steps, 3])
    Green_photons = 0
    Blue_photons = 0

    t = 0    
    for step in range(N_steps):
        MOT_scatter = False
        MOT_try = False
        CB_try = False
        No_scatter = False
        Dv = np.zeros(3)
        v_traj[step] = [A.vx,A.vy,A.vz]
        x_traj[step] = [A.X,A.Y,A.Z]
        pos_x = genlib.Find_Target(A.X,x)
        pos_y = genlib.Find_Target(A.Y,y)
        pos_z = genlib.Find_Target(A.Z,z)

        # Find the magnetic shift: consider that atoms scattering from opposite beams will see opposite magnetic shifts
        B_act = B[pos_x,pos_y,pos_z]
        # Decide from which beam the atom tries to absorb a photon
        newCoordinates = np.asarray(CoordinateChange((x[pos_x],y[pos_y],z[pos_z]),3*np.pi/4,Total_Length))
        # First: compute P_sc for all beams
        # needs to have \tau_blue!
        P_sc_CB = crossed_beams_scattering_prob(GS_Transitions[0],tau_blue,Blue_lambd,E01_CB,E02_CB,k1_CB,k2_CB,A,pos_x,pos_y,pos_z)
        P_sc_MOT = MOT_beams_scattering_prob(GS_Transitions[1],tau_blue,G_lambd_HOR,G_lambd_VER,E01_G,E02_G,E03_G,k1_G,k2_G,k3_G,newCoordinates,B_act,A,pos_x,pos_y,pos_z)
        #----------------------------------------------------------------------------
        # Decide from which beam you want to scatter from:
        # Roll a loaded die: divide the (0,1) interval into unequal intervals
        Psc_combined = np.concatenate([P_sc_CB,P_sc_MOT])
        Weight_tot = np.sum(Psc_combined)/np.mean(Psc_combined)
        beam_guess = np.random.uniform(0,1)
        weights = Psc_combined/np.mean(Psc_combined)/Weight_tot   
        if np.max(Psc_combined) < 10**-3:
            No_scatter = True
        else:
            if beam_guess < weights[0]:
                CBeam = 0
                k_CB = k1_CB
                CB_try = True
            elif beam_guess > weights[0] and beam_guess < np.sum(weights[:2]):
                CBeam = 1
                k_CB = k2_CB
                CB_try = True
            elif beam_guess > np.sum(weights[:2]) and beam_guess < np.sum(weights[:3]):
                MOTBeam = 0
                k_MOT = k1_G
                MOT_try = True
            elif beam_guess > np.sum(weights[:3]) and beam_guess < np.sum(weights[:4]):
                MOTBeam = 1
                k_MOT = -k1_G
                MOT_try = True
            elif beam_guess > np.sum(weights[:4]) and beam_guess < np.sum(weights[:5]):
                MOTBeam = 2
                k_MOT = k2_G
                MOT_try = True                
            elif beam_guess > np.sum(weights[:5]) and beam_guess < np.sum(weights[:6]):
                MOTBeam = 3
                k_MOT = -k2_G
                MOT_try = True
            elif beam_guess > np.sum(weights[:6]) and beam_guess < np.sum(weights[:7]):
                MOTBeam = 4
                k_MOT = k3_G
                MOT_try = True
            elif beam_guess > np.sum(weights[:7]):
                MOTBeam = 5
                k_MOT = -k3_G
                MOT_try = True
            else:
                print('Houston')
                print(beam_guess)
                print(weights)
                break
        if CB_try == True:  # Try to scatter from the crossed beams
            if np.random.uniform(0,1) < P_sc_CB[CBeam]:
                # After absorption an atom re-hemits the photon in a random direction
                Dv += hbar*k_CB/m+SpontaneousEmission(GS_Transitions[0].Lambda,m)
                Blue_photons += 1
                CB_scatter = True # The atom has scattered from the crossed beams
        elif MOT_try == True: # Try to scatter from the MOT beams
            if np.random.uniform(0,1) < P_sc_MOT[MOTBeam]:
                Dv += hbar*k_MOT/m+SpontaneousEmission(GS_Transitions[1].Lambda,m)
                Green_photons += 1
                MOT_scatter = True # the atom has scattered from the crossed beams
        #-------------------------------------------------------------------------------
        # The atom has attempted scattering from both crossed beams and MOT: it goes forward
        if MOT_scatter == True: 
            # if the atom has scattered from the MOT, it is excited to the excited state and it spends there a time dt before being able to scatter again 
            Dt = tau_green
        elif No_scatter == True:
            Dt = dt_no_scatter
        else:
            Dt = tau_blue
        OneStep(A,Dv,Dt)    
        t += Dt
        #-------------------------------------------------------------------------------
        # Break conditions:  
        if  t >= t_end:
            last = step
            # print(f'Atom time finished at step {step}')
            break
        elif(np.sqrt(A.Y**2+(A.X-Total_Length)**2)>15*10**-3 and A.X>Total_Length):
            # print(f'Atom out horizontally at step {step}')
            last = step
            captured = 0
            break
        elif np.abs(A.Z) > 10*10**-3:
            # print(f'Atom out vertically at step {step}')
            captured = 0
            last = step
            break
        elif A.X < 350e-3 and A.vx < 0:
            # print(f'Atom going back at step {step}')
            captured = 0
            last = step
            break
    if np.sqrt(A.Y**2+(A.X-Total_Length)**2) <MOT_VerSize/2 and np.abs(A.Z) < MOT_HorSize/2: captured = 1
    else: captured = 0

    x_final = x_traj[last,:]*1e3
    z_avg = np.mean(x_traj[:last,2])*1e3
    z_std = np.std(x_traj[:last,2])*1e3
    try:
        z_excursion = (np.max(x_traj[:last,2])-np.min(x_traj[:last,2]))*1e3
    except:
        z_excursion = 0
    v_final = v_traj[last,:]
    abs_vz_avg = np.mean(np.abs(v_traj[:last,2]))
    return t, x_final, z_avg, z_std, z_excursion, v_final, abs_vz_avg, captured


def Trajectory_simulation_full_bug(N_steps,t_end,dt_no_scatter,x,y,z,Total_Length,MOT_VerSize,MOT_HorSize,B,Blue_lambd,E01_CB,E02_CB,k1_CB,k2_CB,G_lambd_HOR,G_lambd_VER,E01_G,E02_G,E03_G,k1_G,k2_G,k3_G,A,GS_Transitions,tau_green,tau_blue,m):
    x_traj = np.zeros([N_steps, 3])
    v_traj = np.zeros([N_steps, 3])
    Green_photons = 0
    Blue_photons = 0

    t = 0    
    for step in range(N_steps):
        MOT_scatter = False
        MOT_try = False
        CB_try = False
        No_scatter = False
        Dv = np.zeros(3)
        v_traj[step] = [A.vx,A.vy,A.vz]
        x_traj[step] = [A.X,A.Y,A.Z]
        pos_x = genlib.Find_Target(A.X,x)
        pos_y = genlib.Find_Target(A.Y,y)
        pos_z = genlib.Find_Target(A.Z,z)

        # Find the magnetic shift: consider that atoms scattering from opposite beams will see opposite magnetic shifts
        B_act = B[pos_x,pos_y,pos_z]
        # Decide from which beam the atom tries to absorb a photon
        newCoordinates = np.asarray(CoordinateChange((x[pos_x],y[pos_y],z[pos_z]),3*np.pi/4,Total_Length))
        # First: compute P_sc for all beams
        P_sc_CB = crossed_beams_scattering_prob(GS_Transitions[0],tau_blue,Blue_lambd,E01_CB,E02_CB,k1_CB,k2_CB,A,pos_x,pos_y,pos_z)
        P_sc_MOT = MOT_beams_scattering_prob(GS_Transitions[1],tau_green,G_lambd_HOR,G_lambd_VER,E01_G,E02_G,E03_G,k1_G,k2_G,k3_G,newCoordinates,B_act,A,pos_x,pos_y,pos_z)
        #----------------------------------------------------------------------------
        # Decide from which beam you want to scatter from:
        # Roll a loaded die: divide the (0,1) interval into unequal intervals
        Psc_combined = np.concatenate([P_sc_CB,P_sc_MOT])
        Weight_tot = np.sum(Psc_combined)/np.mean(Psc_combined)
        beam_guess = np.random.uniform(0,1)
        weights = Psc_combined/np.mean(Psc_combined)/Weight_tot   
        if np.max(Psc_combined) < 10**-3:
            No_scatter = True
        else:
            if beam_guess < weights[0]:
                CBeam = 0
                k_CB = k1_CB
                CB_try = True
            elif beam_guess > weights[0] and beam_guess < np.sum(weights[:2]):
                CBeam = 1
                k_CB = k2_CB
                CB_try = True
            elif beam_guess > np.sum(weights[:2]) and beam_guess < np.sum(weights[:3]):
                MOTBeam = 0
                k_MOT = k1_G
                MOT_try = True
            elif beam_guess > np.sum(weights[:3]) and beam_guess < np.sum(weights[:4]):
                MOTBeam = 1
                k_MOT = -k1_G
                MOT_try = True
            elif beam_guess > np.sum(weights[:4]) and beam_guess < np.sum(weights[:5]):
                MOTBeam = 2
                k_MOT = k2_G
                MOT_try = True                
            elif beam_guess > np.sum(weights[:5]) and beam_guess < np.sum(weights[:6]):
                MOTBeam = 3
                k_MOT = -k2_G
                MOT_try = True
            elif beam_guess > np.sum(weights[:6]) and beam_guess < np.sum(weights[:7]):
                MOTBeam = 4
                k_MOT = k3_G
                MOT_try = True
            elif beam_guess > np.sum(weights[:7]):
                MOTBeam = 5
                k_MOT = -k3_G
                MOT_try = True
            else:
                print('Houston')
                print(beam_guess)
                print(weights)
                break
        if CB_try == True:  # Try to scatter from the crossed beams
            if np.random.uniform(0,1) < P_sc_CB[CBeam]:
                # After absorption an atom re-hemits the photon in a random direction
                Dv += hbar*k_CB/m+SpontaneousEmission(GS_Transitions[0].Lambda,m)
                Blue_photons += 1
                CB_scatter = True # The atom has scattered from the crossed beams
        elif MOT_try == True: # Try to scatter from the MOT beams
            if np.random.uniform(0,1) < P_sc_MOT[MOTBeam]:
                Dv += hbar*k_MOT/m+SpontaneousEmission(GS_Transitions[1].Lambda,m)
                Green_photons += 1
                MOT_scatter = True # the atom has scattered from the crossed beams
        #-------------------------------------------------------------------------------
        # The atom has attempted scattering from both crossed beams and MOT: it goes forward
        if MOT_scatter == True: 
            # if the atom has scattered from the MOT, it is excited to the excited state and it spends there a time dt before being able to scatter again 
            Dt = tau_green
        elif No_scatter == True:
            Dt = dt_no_scatter
        else:
            Dt = tau_blue
        OneStep(A,Dv,Dt)    
        t += Dt
        #-------------------------------------------------------------------------------
        # Break conditions:  
        if  t >= t_end:
            last = step
            # print(f'Atom time finished at step {step}')
            break
        elif(np.sqrt(A.Y**2+(A.X-Total_Length)**2)>15*10**-3 and A.X>Total_Length):
            # print(f'Atom out horizontally at step {step}')
            last = step
            captured = 0
            break
        elif np.abs(A.Z) > 10*10**-3:
            # print(f'Atom out vertically at step {step}')
            captured = 0
            last = step
            break
        elif A.X < 350e-3 and A.vx < 0:
            # print(f'Atom going back at step {step}')
            captured = 0
            last = step
            break
    if np.sqrt(A.Y**2+(A.X-Total_Length)**2) <MOT_VerSize/2 and np.abs(A.Z) < MOT_HorSize/2: captured = 1
    else: captured = 0

    x_final = x_traj[last,:]*1e3
    z_avg = np.mean(x_traj[:last,2])*1e3
    z_std = np.std(x_traj[:last,2])*1e3
    try:
        z_excursion = (np.max(x_traj[:last,2])-np.min(x_traj[:last,2]))*1e3
    except:
        z_excursion = 0
    v_final = v_traj[last,:]
    abs_vz_avg = np.mean(np.abs(v_traj[:last,2]))
    return t, x_final, z_avg, z_std, z_excursion, v_final, abs_vz_avg, captured
##################################################################
################# Atom - photon interaction ######################
##################################################################
def Polarizability_2Level(trans, lambdas): # main source: Martin phd -> ok for atoms starting from 1S0 and 3P0
    w = 2*np.pi*c/lambdas
    w0 = trans.Omega
    MatrixElemSquared = 3*np.pi*eps_0*hbar*c**3/w0**3*trans.Gamma
    return 2*w0/(hbar*(w0**2-w**2)) * MatrixElemSquared


def GaussianBeam (data_tuple,wavelength,w0z,w0y,P0):
    # These coordinates are in the beam's system of reference:
    # Beam propagates along z!
    (xP,yP,zP) = data_tuple
    zRz = np.pi*w0z**2/wavelength
    zRy = np.pi*w0y**2/wavelength
    wz = w0z*np.sqrt(1+(zP/zRz)**2)
    wy = w0y*np.sqrt(1+(zP/zRy)**2)
    I0 = 2*P0/(np.pi*(w0z*w0y))
    I = I0*(w0z/wz)**2*(w0y/wy)**2*np.exp(-2*((zP/wz)**2+(yP/wy)**2))
    return I
@nu.jit()
def CoordinateChange(data_tuple,angle,x0):
    (x,y,z) = data_tuple
    x_Prime = (x-x0)/np.cos(angle) + y*np.sin(angle) - (x-x0)*np.tan(angle)*np.sin(angle)
    y_Prime = y*np.cos(angle) - (x-x0)*np.sin(angle)
    z_Prime = z
    return x_Prime,y_Prime,z_Prime

# @nu.jit()
def ScatteringRate_2Level_Doppler3D_Bfield(transition, wavelength, E_0,k, v, ZeemanShift): 
    '''
    Compute scattering rate of a 2 level atom, considering the 3D Doppler and Zeeman effects.
    
    ScatteringRate_2Level_Doppler3D_Bfield(transition, wavelength, E_0, k,v ,ZeemanShift):
    transition: considered transition for the 2 level approximation. dtype = atomic transition class from Yb library.
    wavelength: laser wavelength in meters. dtype = float
    E_0: amplitude of the e.m. field computed with I2E_0 function from Yb library. dtype = float
    k: 3D wavevector of the laser beam. dtype = float (3,)
    v: velocity of the atom in m/s. dtype = float (3,)
    ZeemanShift: Zeeman shift in Hz. dtype = float
    
    Matrix elements are computed as in Martin PhD thesis; scattering rate as in Muzi Falconi master thesis.
    Laser frequency (\omega) in the atomic system of reference is computed from laser wavelength and Doppler+Zeeman shifts
    Dipole matrix element is computed from trnaistion linewidth 
    Rabi frequency is computed from dipole matrix element and amplitude of e.m. field
    Scattering rate is computed from Rabi freq., detuning and transition linewidth.
    '''
 
    w = 2*np.pi*c/wavelength - np.dot(k,v) - 2*np.pi*ZeemanShift
    w0 = transition.Omega 
    Gamma = transition.Gamma 
    MatrixElemSquared = 3*np.pi*eps_0*hbar*c**3/w0**3*transition.Gamma
    OmegaRabiSquared = MatrixElemSquared*E_0**2/hbar**2
    return OmegaRabiSquared/Gamma / (1 + 2*OmegaRabiSquared/Gamma**2 + 4*(w-w0)**2 /Gamma**2)

def theta_min_backwindow (waist_x, MOT_d): # this is using the beams from the window opposite to the atom source
    window_d = 13*10**-3
    window_depth = 8*10**-3
    CellRadius = 20*10**-3
    x_start = window_d/2-2*waist_x
    tm = np.arctan((x_start-MOT_d/2)/(window_depth+CellRadius))
    dist = MOT_d/2/(np.tan(tm))
    return np.rad2deg(tm),dist

def theta_min_diag_window (size_x, MOT_d): 
    x_projection = 12*10**-3-(MOT_d/2)
    z_projection = 23*10**-3
    tm = np.arctan(x_projection/z_projection)
    CBsize = size_x/np.sin(tm)
    dist = MOT_d/2/(np.tan(tm)) + size_x/np.sin(tm)/2
    return tm,dist,CBsize

def Saturation (P,w,Is):
    return (2*P)/(np.pi*w**2)/Is

@nu.jit(nopython=True)
def SpontaneousEmission (wavelength,m):
    """
    Emit a photon of given wavelength in a random direction
    """
    kmod = 2*np.pi/(wavelength)
    k = np.zeros(3)
    theta = 2*np.arcsin(np.sqrt(np.random.uniform(0,1)))
    phi = np.random.uniform(0,2*np.pi)
    k[0] = kmod*np.cos(phi)*np.sin(theta)
    k[1] = kmod*np.sin(phi)*np.sin(theta)
    k[2] = kmod*np.cos(theta)
    return hbar*k/m

def Space_grid(Ngrid, cell_center = 37*10**-2, y_lim = 15, z_lim = 8, start = 330e-3):
    '''
    Generate 3D grid for the simulation
    Space_grid(x_lim,y_lim,z_cell,cell_center,N_plot):
    Units: m
    '''
    # Generate space grid:
    x = np.linspace(start,cell_center+15*10**-3,Ngrid)
    y = np.linspace(-y_lim,y_lim,Ngrid)*10**-3
    z = np.linspace(-z_lim,z_lim,Ngrid)*10**-3
    return x, y, z
# @nu.jit()
def AntiHelmhotz(data_tuple,I,x,y,z,R = 5*10**-2, spyres = 40, Total_Length = 37*10**-2,PlotB = False):
    (x_grid,y_grid,z_grid) = data_tuple
    coeff = -3*mu_0*I*R**2*(R/2)/(2*(R**2+(R/2)**2)**(5/2))*spyres
    x_comp = coeff*x_grid/2 
    y_comp = coeff*y_grid/2 
    z_comp = - coeff*z_grid
    B_field = np.sqrt(x_comp**2+y_comp**2+z_comp**2)  
    if PlotB: 
        grad_y = np.round(B_field[genlib.Find_Target(Total_Length,x),genlib.Find_Target(10**-2,y),genlib.Find_Target(0,z)]-B_field[genlib.Find_Target(Total_Length,x),genlib.Find_Target(0,y),genlib.Find_Target(0,z)],2)  
        grad_z = np.round(B_field[genlib.Find_Target(Total_Length,x),genlib.Find_Target(0,y),genlib.Find_Target(10**-2,z)]-B_field[genlib.Find_Target(Total_Length,x),genlib.Find_Target(0,y),genlib.Find_Target(0,z)],2)   
        print('dB/dz: ',np.round(grad_z,2),'G/cm')
        print('dB/dy: ',np.round(grad_y,2),'G/cm')
        print(f'Minimum B: {np.round(np.min(B_field),4)} G')
        Nplot = 200
        figure, axes = plt.subplots(1,3,figsize = (25,5))
        axes[0].plot(x*10**3, B_field[:,int(Nplot/2),int(Nplot/2)],color='tab:orange')
        axes[0].set_xlabel('x (mm)')
        axes[0].set_ylabel('B (G)')
        axes[1].plot(y*10**3, B_field[genlib.Find_Target(Total_Length,x),:,genlib.Find_Target(0,z)],color='tab:orange')
        axes[1].set_xlabel('y (mm)')
        axes[1].set_ylabel('B (G)')
        axes[2].plot(z*10**3, B_field[genlib.Find_Target(Total_Length,x),genlib.Find_Target(0,y),:],color='tab:orange')
        axes[2].set_xlabel('z (mm)')
        axes[2].set_ylabel('B (G)')
        plt.show()
    return B_field


def CrossedBeams (data_tuple, theta, d, diameters, Power):
    '''
    data_tuple: space grid
    d: distance between center of crossed beams and center of the MOT
    '''
    (x, y, z) = data_tuple
    CB_x_diam = diameters[0]
    CB_y_diam = diameters[1]
    x_new, y_new, z_new = CoordinateChange((x, y, z),theta,37e-2-d)
    x_new2, y_new2, z_new2 = CoordinateChange((x, y, z),-theta,37e-2-d)
    CBeam1 = GaussianBeam((x_new, y_new, z_new),399*10**-9,CB_x_diam/2,CB_y_diam/2,Power)
    CBeam2 = GaussianBeam((x_new2, y_new2, z_new2),399*10**-9,CB_x_diam/2,CB_y_diam/2,Power)
    return CBeam1, CBeam2