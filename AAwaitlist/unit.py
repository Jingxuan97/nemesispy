#!/usr/local/bin/python3
# -*- coding: utf-8 -*-
"""
Define constants and units used in the code.
"""

unit = {
    'pc': 3.08567e16,        # m parsec
    'ly': 9.460730e15,       # m lightyear
    'AU': 1.49598e11,        # m astronomical unit
    'R_Sun': 6.95700e8,      # m solar radius
    'R_Jup': 7.1492e7,       # m nominal equatorial Jupiter radius (1 bar pressure level)
    'R_E': 6.371e6,          # m nominal Earth radius
    'd_H2': 2.827e-10,       # m molecular diameter of H2
    'M_Sun': 1.989e30,       # kg solar mass
    'M_Jup': 1.8982e27,      # kg Jupiter mass
    'M_E': 5.972e24,         # kg Earth mass
    'amu': 1.66054e-27,      # kg atomic mass unit
    'atm': 101325,           # Pa atmospheric pressure
}

const = {
    'k_B': 1.38065e-23,         # J K-1 Boltzmann constant
    'sig_B': 5.67037e-8,        # W m-2 K-4 Stephan Boltzmann constant
    'R': 8.31446,               # J mol-1 K-1 universal gas constant
    'G': 6.67430e-11,           # m3 kg-1 s-2 universal gravitational constant
    'eps_LJ': 59.7*5.67037e-8,  # J depth of the Lennard-Jones potential well for H2
    'c_p': 14300,               # J K-1 hydrogen specific heat
    'h': 6.62607,               # Js Planck's constant
    'hbar': 1.05457,            # Js
    'N_A': 6.02214e23,          # Avagadro's number
}