#!/usr/local/bin/python3
# -*- coding: utf-8 -*-
"""
Constants used in this package.
Follow the CODATA Recommended Values of Fundamental Physical Constants.
J. Phys. Chem. Ref. Data 50, 033105 (2021);
https://doi.org/10.1063/5.0064853

from nemesispy.common.constants import R_SUN,M_SUN
"""
# CODATA
C_LIGHT = 299792458       # ms-1 speed of light in vacuum
G = 6.67430e-11           # m3 kg-1 s-2 Newtonian constant of gravitation
PLANCK = 6.62607015e-34   # Js Planck constant
K_B = 1.380649e-23        # J K-1 Boltzmann constant
N_A = 6.02214076e23       # Avagadro's number
SIGMA_SB = 5.670374419e-8 # Wm-2K-4 Stefan-Boltzmann constant
AMU = 1.66053906660e-27   # kg atomic mass unit

# Astronomical units
PC = 3.0857e16            # m parsec
AU = 1.495978707e11       # m astronomical unit
R_SUN = 6.95700e8         # m solar radius
R_JUP_E = 7.1492e7        # m Jupiter equatorial Jupiter radius at 1 bar (nominal)
R_JUP_V = 6.9911e7        # m Jupiter volumetric mean radius
R_EARTH = 6.371e3         # m Mean Earth radius
M_SUN = 1.98847542e30     # kg solar mass
M_JUP = 1.898e27          # kg Jupiter mass
M_EARTH = 5.972e24        # kg Earth mass

# Others
ATM = 101325              # Pa atmospheric pressure
BAR = 100000              # Bar

# Derived
C1 = 1.1910429723971884e-12 # W cm2 2*PLANCK*C_LIGHT**2
C2 = 1.4387768775039337 # cm K-1 PLANCK*C_LIGHT/K_B