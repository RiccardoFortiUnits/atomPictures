#!/usr/bin/env python
# coding: utf-8
# Define properties and classe for Ytterbium atoms 

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.colors as mcolors
import matplotlib.pylab as pl
import seaborn as sns
import scipy
from sympy.physics.wigner import clebsch_gordan, wigner_3j,wigner_6j
import scipy.constants as const
from time import sleep
from tqdm.notebook import tqdm
import numba as nu

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

def Find_Target (target,array): 
    '''
    Find_Target(target, array)
    Finds the index of the closest value to a target in an array
    Probably can be done with np.where()
    '''
    d = abs(array - target)
    return np.argmin(d)

def Gaussian (x, A, x0, sigma, offset):
    ''' 
    Gaussian(x, A, x_0, sigma, offset)
    1D Gaussian: A*e^(-2*((x-x_0)/sigma)^2) + offset
    x: array 
    A: amplitude. dtype = float
    x_0: center. dtype = float 
    sigma: "waist" (1/e^2 level). dtype = float
    '''
    return A*np.exp(-2*((x-x0)/sigma)**2) + offset

def TwoD_Gaussian (xdata_tuple, A, x0,y0, sigma_x, sigma_y, offset):
    ''' 
    TwoD_Gaussian(xdata_tuple, A, x0,y0, sigma_x, sigma_y, offset)
    1D Gaussian: A*e^(-2*(((x-x_0)/sigma_x))^2 + ((y-y_0)/sigma_y))^2 ) + offset
    xdata_tuple: array. dtype = tuple 
    A: amplitude. dtype = float
    x_0: center. dtype = float 
    sigma: "waist" (1/e^2 level). dtype = float
        '''
    (x,y) = xdata_tuple
    G = A*np.exp(-2*((x-x0)**2/sigma_x**2 + (y-y0)**2/sigma_y**2)) + offset
    return G.ravel()

def I2E_0 (I): # (W/m^2) as in Martin & master 
    '''
    Convert intensity to modulus of electric field:
    E_0 = sqrt(2I/(epsilon_0 * c)
    '''
    return np.sqrt(2*I/(eps_0*c))

def compute_saturation(power, diameter, I_saturation):
    '''
    Compute s = I/I_s form power and beam diameter
    '''
    I = 2*power/(np.pi*(diameter/2)**2) # in W/m^2
    return I/I_saturation

def compute_power(saturation, diameter, I_saturation):
    '''
    Compute power in W from s = I/I_s and diameter
    '''
    return saturation*I_saturation*np.pi*(diameter/2)**2 /2