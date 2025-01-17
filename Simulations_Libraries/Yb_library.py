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
import math

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
Eh = 4.4*10**-18;
au = 1.64877727436* 10**-41         # atomic eletric polarizability C^2 m^2 J^-1
nm = 10**-9
MHz = 10**6
kHz = 10**3

class Yb:
    def __init__(self, Isotope, m, I):
        self.Isotope = Isotope
        self.m = m*u
        self.I = I

class State:
    def __init__(self, Isotope, name, S, L, J):
        self.Name = name
        self.I = Isotope.I
        self.S = S
        self.L = L
        self.J = J
        self.mJ = np.linspace(-self.J,self.J,int(2*self.J+1))

class Transition:
    def __init__(self, GS, ES, Lambda, Gamma, *Is):
        self.Name = GS.Name + '->' + ES.Name
        self.GS = GS
        self.ES = ES
        self.Gamma = Gamma 
        self.tau = 1/self.Gamma
        self.Lambda = Lambda
        self.Frequency = c/Lambda
        self.Omega = 2*np.pi*self.Frequency
        self.Is = 10*np.asarray(Is) # convert mW/cm^2 in W/m^2