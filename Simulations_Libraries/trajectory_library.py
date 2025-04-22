#################
#    ArqusLab   #
#    March 2023 #             
#################

# This library has the functions for simulating the trajectories of atoms

from __future__ import annotations
from typing import List
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
import Simulations_Libraries.Yb_library as yblib
from scipy.interpolate import interp1d
from Camera import *

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
	c = const.physical_constants["speed of light in vacuum"][0]
	h = const.physical_constants["Planck constant"] [0]
	hbar = h/(2*np.pi)
	kB = const.physical_constants["Boltzmann constant"][0]
	muB = 1.399*10 ** 6;                # Bohr's magneton [Hz/G]

	nm = 10**-9
	MHz = 10**6
	kHz = 10**3
	def __init__(self, x,y,z,vx,vy,vz):
		self.X : float = float(x)
		self.Y : float = float(y)
		self.Z : float = float(z)
		self.vx : float = float(vx)
		self.vy : float = float(vy)
		self.vz : float = float(vz)
		self.transitions : List[yblib.Transition] = []
		self.m = 1.66e-24
		self.initialPosition = np.array([x,y,z])
		self.initialSpeed = np.array([vx,vy,vz])
	def reset(self):
		self.X, self.Y, self.Z = self.initialPosition
		self.vx, self.vy, self.vz = self.initialSpeed
	@property
	def position(self):
		return np.array([self.X, self.Y, self.Z])
	@position.setter
	def position(self, value):
		(self.X, self.Y, self.Z) = (value[0], value[1], value[2])
	@property
	def velocity(self):
		return np.array([self.vx, self.vy, self.vz])
	@velocity.setter
	def velocity(self, value):
		(self.vx, self.vy, self.vz) = (value[0], value[1], value[2])
	

class Ytterbium(Atom):

	# Initialise ytterbium atoms 
	Yb_171 = yblib.Yb(171,170.936323,1/2)
	Yb_173 = yblib.Yb(173,172.938208,5/2)
	Yb_174 = yblib.Yb(173,173.938859,0)

	def __init__(self, x,y,z,vx,vy,vz, isotope = 174):
		super().__init__(x, y, z, vx, vy, vz)
		Isotope = [self.Yb_171, None, self.Yb_173, self.Yb_174][isotope - 171]
		# State(Isotope, name, S, L, J)
		_1S0 = yblib.State(Isotope,"1S0",0,0,0)
		_1P1 = yblib.State(Isotope,"1P1",0,0,1)
		_3P1 = yblib.State(Isotope,"3P1",1,1,1)

		## GS (1S0) Transitions (g.s., e.s., lambda (m), 2pi Gamma (s^-1), *I_sat (mW/cm^2)):
		GS_Transitions = []
		GS_Transitions.append(yblib.Transition(_1S0,_1P1,398.911*nm,183.02*MHz,59.97)) # values of lambda and gamma from Riegger Table B.1
		GS_Transitions.append(yblib.Transition(_1S0,_3P1,555.802*nm,1.15*MHz,0.139))
		self.transitions = GS_Transitions
		self.m = Isotope.m

class Beam:
	def __init__(self, angle, x0, function):
		if isinstance(angle,list) or isinstance(angle, tuple):
			self.angleXY = angle[0]
			self.angleXZ = angle[1]
			self.angleYZ = angle[2]
		else:
			self.angleXY = angle
			self.angleXZ = 0
			self.angleYZ = 0
		if isinstance(x0, list) or isinstance(x0, tuple):
			self.x0 = x0[0]
			self.y0 = x0[1]
			self.z0 = x0[2]
		else:
			self.x0 = x0
			self.y0 = 0
			self.z0 = 0
		self._originalFuntion = function
		self.function = lambda coordinate_tuple, **kwargs : function(CoordinateChange_extended(coordinate_tuple,self.x0, self.y0, self.z0, self.angleXY, self.angleXZ, self.angleYZ), **kwargs)
	@property
	def angle(self):
		return (self.angleXY, self.angleXZ, self.angleYZ)
	@property
	def center(self):
		return (self.x0, self.y0, self.z0)
	def __call__(self, x,y = None,z = None, **kwargs):
		if y == None and z == None:
			return self.function(x, **kwargs)
		return self.function((x,y,z), **kwargs)
	def __add__(self, other: Beam):
		if type(other) == Beam:
			#if a Beam is combined with another, it should not have an intrinsic direction
			return Beam(0,0, lambda coordinate_tuple, **kwargs: self(coordinate_tuple, **kwargs)+other(coordinate_tuple, **kwargs))
		if isinstance(other, (int, float, np.number)):
			#let's give the new beam the same direction as the original one
			return Beam(self.angle, self.center, lambda coordinate_tuple, **kwargs: self._originalFuntion(coordinate_tuple, **kwargs)+other)
		raise ValueError('Sum not defined for Beam and '+str(type(other)))
	def __sub__(self, other):
		if type(other) == Beam:
			return Beam(0,0, lambda coordinate_tuple, **kwargs: self(coordinate_tuple, **kwargs)-other(coordinate_tuple, **kwargs))
		if isinstance(other, (int, float, np.number)):
			return Beam(self.angle, self.center, lambda coordinate_tuple, **kwargs: self._originalFuntion(coordinate_tuple, **kwargs)-other)
		raise ValueError('Subtraction not defined for Beam and '+str(type(other)))
	def __mul__(self, other):
		if type(other) == Beam:
			return Beam(0,0, lambda coordinate_tuple, **kwargs: self(coordinate_tuple, **kwargs)*other(coordinate_tuple, **kwargs))
		if isinstance(other, (int, float, np.number)):
			return Beam(self.angle, self.center, lambda coordinate_tuple, **kwargs: self._originalFuntion(coordinate_tuple, **kwargs)*other)
		raise ValueError('Multiplication not defined for Beam and '+str(type(other)))
	def __truediv__(self, other):
		if type(other) == Beam:
			return Beam(0,0, lambda coordinate_tuple, **kwargs: self(coordinate_tuple, **kwargs)/other(coordinate_tuple, **kwargs))
		if isinstance(other, (int, float, np.number)):
			return Beam(self.angle, self.center, lambda coordinate_tuple, **kwargs: self._originalFuntion(coordinate_tuple, **kwargs)/other)
		raise ValueError('Division not defined for Beam and '+str(type(other)))
	def __div__(self, other):
		return self.__truediv__(other)
	
	def withExtraFunction(self, extra_function):
		return Beam(self.angle, self.center, lambda coordinate_tuple, **kwargs: extra_function(self._originalFuntion(coordinate_tuple, **kwargs)))
	
	@property
	def direction(self):
		#warning: this function gives the direction of the beam, if its intensity function represents 
			#a beam travelling on the x axis. If this beam is the combination of more Beam objects, it
			#should not have an intrinsic direction, and this function will return (1,0,0)
		return np.array(CoordinateChange_extended((self.x0+1, self.y0, self.z0),self.x0, self.y0, self.z0, self.angleXY, self.angleXZ, self.angleYZ))

	
	def plotSection(self, planeCenter, planeRotation, y_range, z_range, resolution = 100):
		planeCenter = np.array(planeCenter)
		planeRotation = np.array(planeRotation)
		f = lambda y, z : self(reverseCoordinateChange((0,y,z), 
													   planeCenter[0], planeCenter[1], planeCenter[2], 
													   planeRotation[0], planeRotation[1], planeRotation[2]))
		plot_2d_function(f, y_range, z_range, resolution)
class Laser(Beam):
	def __init__(self, angle, x0, wavelength, Intensity, w0, function = None, refractive_index = 1, switchingTimes = None):#switchingTimes = [dt, startTime, activeTime, inactiveTime]
		self.wavelength = wavelength
		self.intensity = Intensity
		if not isinstance(w0, (list, tuple)):
			w0 = (w0,w0)
		self.w0 = w0
		self.refractive_index = refractive_index
		if function is None:
			function = lambda coordinate_tuple, **kwargs: GaussianBeam(coordinate_tuple, wavelength, w0[0], w0[1], Intensity)
		if switchingTimes is not None:
			self.switchPeriod = switchingTimes[1] + switchingTimes[2]
			self.switchActiveTime = switchingTimes[1]
			self.startingTime = switchingTimes[0]
			def enableLaser(t):
				t = (t - self.startingTime) % self.switchPeriod
				return t < self.switchActiveTime
			function = lambda coordinate_tuple, t = 0 : GaussianBeam(coordinate_tuple, wavelength, w0[0], w0[1], Intensity) if enableLaser(t) else 0
		elif not callable(function):
			raise ValueError(f'The function must be callable, got {type(function)} instead')
		
		super().__init__(angle, x0, function)
	def reset(self):
		if hasattr(self, 'initialTime'):
			self.time = self.initialTime
			self.repetition = 0
	def electricalField(self) -> Laser:
		#use this function to obtain the electrical field of this beam. It will still be of class Laser, but the intensity will be proportional to the electrical field instead of the laser intensity
		return self.withExtraFunction(genlib.I2E_0)
	@property
	def k(self):
		return 2*np.pi/self.wavelength * self.direction
	
	def withExtraFunction(self, extra_function):
		return Laser(self.angle, self.center, self.wavelength, self.intensity, self.w0, 
					 lambda coordinate_tuple, **kwargs: extra_function(self._originalFuntion(coordinate_tuple, **kwargs)))
	

	
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
	A.vx += Dv[0]
	A.vy += Dv[1]  
	A.vz += Dv[2] - g*Dt # g is positive!
	A.X +=  A.vx*Dt
	A.Y +=  A.vy*Dt 
	A.Z +=  A.vz*Dt - 0.5*g*Dt**2

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
	x_Prime = (x - x0) * np.cos(angle) + y * np.sin(angle)
	# x_Prime = (x-x0)/np.cos(angle) + y*np.sin(angle) - (x-x0)*np.tan(angle)*np.sin(angle)
	y_Prime = y*np.cos(angle) - (x-x0)*np.sin(angle)
	z_Prime = z
	return x_Prime,y_Prime,z_Prime
def CoordinateChange_extended(data_tuple, x0, y0, z0, angleXY, angleXZ, angleYZ):
	(x, y, z) = data_tuple
	x, y, z = x - x0, y - y0, z - z0
	
	# Rotation around the XY plane
	y_Prime = y * np.cos(angleXY) + x * np.sin(angleXY)
	x_temp = x * np.cos(angleXY) - y * np.sin(angleXY)
	
	# Rotation around the XZ plane
	x_Prime = x_temp * np.cos(angleXZ) - z * np.sin(angleXZ)
	z_temp = x_temp * np.sin(angleXZ) + z * np.cos(angleXZ)
	
	
	# Rotation around the YZ plane
	z_Prime = y_Prime * np.sin(angleYZ) + z_temp * np.cos(angleYZ)
	y_Prime = y_Prime * np.cos(angleYZ) - z_temp * np.sin(angleYZ)
	
	return x_Prime, y_Prime, z_Prime
def reverseCoordinateChange(data_tuple, x0, y0, z0, angleXY, angleXZ, angleYZ):
	#reverts the coordinate change done by CoordinateChange_extended
	(x, y, z) = data_tuple
	angleXY = - angleXY
	angleXZ = - angleXZ
	angleYZ = - angleYZ
 
	# Rotation around the YZ plane
	z_temp = y * np.sin(angleYZ) + z * np.cos(angleYZ)
	y_temp = y * np.cos(angleYZ) - z * np.sin(angleYZ)

	# Rotation around the XZ plane
	x_temp = x * np.cos(angleXZ) - z_temp * np.sin(angleXZ)
	z_Prime = x * np.sin(angleXZ) + z_temp * np.cos(angleXZ)

	# Rotation around the XY plane
	y_Prime = x_temp * np.sin(angleXY) + y_temp * np.cos(angleXY)
	x_Prime = x_temp * np.cos(angleXY) - y_temp * np.sin(angleXY)

	return x_Prime + x0, y_Prime + y0, z_Prime + z0

def plotVector(ax, vector_coordinates, startingPoint_coordinates = (0,0,0)):
	ax.quiver(startingPoint_coordinates[0], startingPoint_coordinates[1], startingPoint_coordinates[2],
			  vector_coordinates[0], vector_coordinates[1], vector_coordinates[2],
			  arrow_length_ratio=0.1)
	
def plot_2d_function(f, x_range, y_range, resolution=100):
	"""
	Plots a 2D image where each pixel has the intensity of a certain function f(x, y).
	
	Parameters:
	f (function): The function to plot. It should take two arguments (x, y) and return a single value.
	x_range (tuple): The range of x values as (x_min, x_max).
	y_range (tuple): The range of y values as (y_min, y_max).
	resolution (int): The resolution of the plot (number of pixels along each axis).
	"""
	x = np.linspace(x_range[0], x_range[1], resolution)
	y = np.linspace(y_range[0], y_range[1], resolution)
	X, Y = np.meshgrid(x, y)
	Z = np.vectorize(f)(X, Y)
	
	plt.figure(figsize=(6, 6))  # Ensure the plot is always square
	plt.imshow(Z, extent=(x_range[0], x_range[1], y_range[0], y_range[1]), origin='lower', cmap='viridis', aspect='auto')
	plt.xlim(x_range)
	plt.ylim(y_range)
	plt.colorbar(label='Intensity')
	plt.xlabel('x')
	plt.ylabel('y')
	plt.title('2D Function Plot')
	plt.show()


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

inverse_cdf_interp = None
def initializeSpontaneousEmission_qPolarization():
	global inverse_cdf_interp
	def cdf(x):
		# return x/np.pi + (1 - np.cos(2*x))/(6*np.pi)
		return x/np.pi + (np.cos(2*x) - 1)/(2*np.pi)

	# Create a lookup table for the inverse CDF
	x_values = np.linspace(0, np.pi, 1000)
	cdf_values = cdf(x_values)
	inverse_cdf_interp = interp1d(cdf_values, x_values, kind='cubic', fill_value="extrapolate")
initializeSpontaneousEmission_qPolarization()
def qPolarizationAnglePDF(theta):
		return 3+np.cos(2*theta)

qPolarizationExtractor = randExtractor.distribFunFromPDF_1D(qPolarizationAnglePDF, (-np.pi/2, np.pi/2), np.pi/200)
def SpontaneousEmission_qPolarization (wavelength,m):
	"""
	Emit a photon of given wavelength in a random direction
	"""
	# return SpontaneousEmission(wavelength, m)
	kmod = 2*np.pi/(wavelength)
	k = np.zeros(3)
	theta = qPolarizationExtractor(0)
	phi = np.random.uniform(0,2*np.pi)
	k[0] = kmod*np.cos(phi)*np.cos(theta)
	k[2] = kmod*np.sin(phi)*np.cos(theta)
	k[1] = kmod*np.sin(theta)
	# if np.random.random_integers(0,1)==0:
	#     k=np.array([-1,0,0])
	# else:
	#     k=np.array([0,1,0])
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
def AntiHelmhotz(data_tuple,I,R = 5*10**-2, spyres = 40):
	(x_grid,y_grid,z_grid) = data_tuple
	coeff = -3*mu_0*I*R**2*(R/2)/(2*(R**2+(R/2)**2)**(5/2))*spyres
	x_comp = coeff*x_grid/2 
	y_comp = coeff*y_grid/2 
	z_comp = - coeff*z_grid
	B_field = np.sqrt(x_comp**2+y_comp**2+z_comp**2)  
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


class experiment(experimentViewer):
	'''properties of this class that do not require the load of heavy libraries are defined inside experimentViewer'''
	def __init__(self, force = None):
		self.atoms : List[Atom]= []
		self.lasers : List[Laser] = []
		if force is None:
			self.force = self.standardAppliedForce
		else:
			self.force = force
	
	def add_atom(self, atom : Atom):
		self.atoms.append(atom)
	def add_laser(self, laser : Laser):
		self.lasers.append(laser)
	@staticmethod
	def standardAppliedForce(a : Atom, *args):
		return np.array([0,0,-g])
	@staticmethod
	def TweezerForce(kx,kyz):
		k=np.array([kx,kyz,kyz])
		def force(a : Atom, *args):
			return np.dot(k,a.position) +np.array([0,0,-g])
		return force
	

	def reset(self):
		for atom in self.atoms:
			atom.reset()
		for laser in self.lasers:
			laser.reset()

	def nextPointInTrajectory(self, a: Atom, dt, impulseForce_beforeDt = np.zeros(3), impulseForce_afterDt = np.zeros(3)):
		'''
		impulseForces are added to the velocities before and after the time step dt (so, impulseForce_afterDt will impact the trajectory only from the next step).
		'''
		a.velocity += impulseForce_beforeDt + self.force(a) * dt
		a.position += a.velocity * dt
		a.velocity += impulseForce_afterDt
	@property
	def initialAtomPositions(self):
		l = np.zeros((len(self.atoms), 3))
		for i in range(len(self.atoms)):
			l[i] = self.atoms[i].initialPosition
		return l
	@property
	def initialAtomSpeeds(self):
		l = np.zeros((len(self.atoms), 3))
		for i in range(len(self.atoms)):
			l[i] = self.atoms[i].initialSpeed
		return l
	def run(self, time = 1e-6, stepTime = 1e-9):
		ZeemanShift = 0#for now, let's not use this
		times = np.arange(0, time, stepTime)
		positions = np.zeros((len(times), len(self.atoms), 3))
		hits = np.empty((len(times), len(self.atoms), len(self.lasers)), dtype=object)
		probabilityList = np.zeros(len(self.lasers))
		electricalFields = [laser.electricalField() for laser in self.lasers]
		excitedStateTimer = 0
		for t_idx, tau in enumerate(times):#it would be nice to advance in time with different dt, depending on if the atom was excited or not, but then you would have different times for each atom
			for a_idx, A in enumerate(self.atoms):
				positions[t_idx, a_idx, :] = A.position
				Dv = np.zeros(3)
				if excitedStateTimer <= 0:
					for i, laser in enumerate(electricalFields):
						probabilityList[i] = stepTime * ScatteringRate_2Level_Doppler3D_Bfield(A.transitions[0],laser.wavelength,laser(A.X, A.Y, A.Z, t = tau),laser.k,[A.vx,A.vy,A.vz],ZeemanShift)
					rand = np.random.random(len(self.lasers))
					chosenLasers = [i for i in range(len(self.lasers)) if rand[i] < probabilityList[i]]
					if len(chosenLasers) > 0:
						excitedStateTimer = np.random.exponential(1/A.transitions[0].Gamma)
						i = np.random.choice(chosenLasers)
						Dv += hbar*self.lasers[i].k/A.m
						#after the time excitedStateTimer, we'll emit the photon
				else:
					excitedStateTimer -= stepTime
					if excitedStateTimer <= 0:
						generatedPhotonKick = SpontaneousEmission_qPolarization(A.transitions[0].Lambda,A.m)
						Dv -= generatedPhotonKick
						hits[t_idx, a_idx, i] = + generatedPhotonKick / np.linalg.norm(generatedPhotonKick)
				OneStep(A, Dv, stepTime)
			
		self.lastPositons = positions
		is_NotNone = np.vectorize(lambda x: x is not None)
		self.lastHits = np.where(is_NotNone(hits))
		self.lastGeneratedPhotons = np.array(list(hits[self.lastHits]))
		return positions
	def new_run(self, time = 1e-6, max_dt = 1e-8, min_dt = 1e-11, max_dx = 1e-8):
		ZeemanShift = 0#for now, let's not use this

		electricalFields = [laser.electricalField() for laser in self.lasers]
		tot_timings = []
		tot_hits = []
		tot_positions = []
		tot_generatedPhotons = []
		for a_idx, A in enumerate(self.atoms):
			t = 0
			timings = []
			hits = []
			positions = []
			generatedPhotons = []
			hittingTime = np.zeros(len(self.lasers))
			excitedTime = 0
			while t < time:
				timings.append(t)
				positions.append(A.position)
				# assume the atom would experience the same fields if it doesn't move too much and if not too much time passes (But we'll have to still consider gravity)
				timeToReachMax_dx = max_dx / np.linalg.norm(A.velocity) if np.linalg.norm(A.velocity) > 0 else np.inf
				maxT = np.maximum(np.minimum(max_dt, timeToReachMax_dx), min_dt)
				if excitedTime <= 0:#are we in the ground state?
					#let's check if the atom would be hit by a laser during the time step
					for i, laser in enumerate(electricalFields):
						#extract the random times at which each laser would hit the atom
						hitProbability = ScatteringRate_2Level_Doppler3D_Bfield(A.transitions[0],laser.wavelength,laser(A.X, A.Y, A.Z, t = t),laser.k,[A.vx,A.vy,A.vz],ZeemanShift)
						hittingTime[i] = np.random.exponential(1 / hitProbability) if hitProbability != 0 else np.inf
					hittingLaserIdx = np.argmin(hittingTime)
					if hittingTime[hittingLaserIdx] < maxT:#would it hit the atom during the considered time?
						hitTime = hittingTime[hittingLaserIdx]
						self.nextPointInTrajectory(A, hitTime, impulseForce_afterDt = hbar*self.lasers[hittingLaserIdx].k/A.m)
						t+=hitTime

						excitedTime = np.random.exponential(1/A.transitions[0].Gamma)
					else:
						#no hit, let's increment the timer
						self.nextPointInTrajectory(A, maxT)
						t += maxT
				elif maxT > excitedTime:#would we finish the excited state before the next hit?
					generatedPhotonKick = SpontaneousEmission_qPolarization(A.transitions[0].Lambda,A.m)
					self.nextPointInTrajectory(A, excitedTime, impulseForce_afterDt = -generatedPhotonKick)
					t+=excitedTime
					excitedTime = 0


					hits.append([len(timings)-1, a_idx, hittingLaserIdx])
					generatedPhotons.append(generatedPhotonKick / np.linalg.norm(generatedPhotonKick))

				else:
					#let's just increment the timer
					excitedTime -= maxT
					self.nextPointInTrajectory(A, maxT)
					t += maxT

			tot_timings.append(np.array(timings))
			tot_positions.append(np.array(positions))
			tot_hits += hits
			tot_generatedPhotons += generatedPhotons
		def getTableWithUnevenColumns(columnList, fillingValue = 0):
			longestColumn = max([len(t) for t in columnList])
			table = fillingValue * np.ones((longestColumn, len(columnList), *columnList[0].shape[1:]))
			for i in range(len(columnList)): 
				table[:len(columnList[i]),i] = columnList[i]
			return table
		self.lastTimings = getTableWithUnevenColumns(tot_timings, np.inf)
		self.lastPositons = getTableWithUnevenColumns(tot_positions, np.nan)
		self.lastHits = tuple(np.array(tot_hits).T)
		self.lastGeneratedPhotons = np.array(tot_generatedPhotons)
		return positions
	
	# lastPositons:           array[nOfTimes][nOfAtoms][3] positions of each atom at each time frame
	# lastHits:               array{timeIndex, atomIndex, laserIndex}[nOfHits] all the recorded hits, specifies the time, atom and laser involved in the hit
	# lastGeneratedPhotons:   array[nOfHits][3] the generated photons for each hit

	def getScatteredPhotons(self):
		startPositions = self.lastPositons[self.lastHits[:-1]]
		directions = self.lastGeneratedPhotons
		return startPositions, directions
	

	def getScatteringDistributionFromRepeatedRuns(self, time = 1e-6, stepTime = 1e-9, nOfRuns = 100, camera : Camera | None = None):
		receivedPhotons = np.zeros(nOfRuns)
		for i in range(nOfRuns):
			print(i)
			self.run(time, stepTime)
			positions, directions = self.getScatteredPhotons()
			if camera is None:
				receivedPhotons[i] = len(positions)
			else:
				photons = camera.hitLens(positions, directions)
				receivedPhotons[i] = len(photons)
			self.reset()
		return receivedPhotons
	def repeatRun(self, time = 1e-6, stepTime = 1e-9, nOfRuns = 100, updateFunction = None):
		'''
		updateFunction should have the input arguments index, experiment
		'''
		for i in range(nOfRuns):
			print(i)
			self.run(time, stepTime)
			if updateFunction is not None:
				updateFunction(index = i, experiment = self)
			self.reset()
		