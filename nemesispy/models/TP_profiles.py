#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import scipy as sp
from scipy import special
from nemesispy.common.constants import R_SUN,M_SUN


def TP_Guillot(P,g_plt,T_eq,k_IR,gamma,f,T_int=100):
    """
    TP profile from eqn. (29) in Guillot 2010.
    DOI: 10.1051/0004-6361/200913396
    Model parameters (4) : k_IR, gamma, f, T_int

    Parameters
    ----------
    P : ndarray
        Pressure grid (in Pa) on which the TP profile is to be constructed.
    g_plt : real
        Gravitational acceleration at the highest pressure in the pressure
        grid.
    T_eq : real
        Temperature corresponding to the stellar flux.
        T_eq = T_star * (R_star/(2*semi_major_axis))**0.5
    gamma : real
        gamma = k_V/k_IR, ratio of visible to thermal opacities
    k_IR : real
        Mean absorption coefficient in the thermal wavelengths.
    f : real
        f parameter (positive), See eqn. (29) in Guillot 2010.
        With f = 1 at the substellar point, f = 1/2 for a
        day-side average and f = 1/4 for whole planet surface average.
    T_int : real
        Temperature corresponding to the intrinsic heat flux of the planet.

    Returns
    -------
    TP : ndarray
        Temperature as a function of pressure.
    """
    # Derived constants
    # gamma = k_V/k_IR # ratio of visible to thermal opacities
    tau = k_IR * P / g_plt # optical depth, assuming g_plt is constant
    T_irr = 2**0.5 * T_eq
    sqrt3 = 3**0.5
    flux1 = 0.75 * T_int**4 * (2/3+tau)
    flux2 = 0.75 * T_irr**4 * f \
        * (2/3 + 1/(gamma*sqrt3) + (gamma/sqrt3 - 1/(gamma*sqrt3)) \
        * np.exp(-gamma * tau * sqrt3))
    TP = (flux1+flux2)**0.25
    return TP

import matplotlib.pyplot as plt

"""
NLAYER = 100
P_grid = np.geomspace(10e8,100,NLAYER) # pressure in pa
T_eq = 1469/(2**0.5)
k_V = 4e-4
k_IR = 1e-3
g = 25
#f = 3
for f in np.linspace(0,2,10):
    for T_int in np.linspace(10,3000,10):
        x = TP_Guillot(P=P_grid,g_plt=g,T_eq=T_eq,k_IR=k_IR,gamma=k_V/k_IR,
                f=f,T_int=T_int)
        plt.plot(x,P_grid/1e5,linewidth=0.5)

plt.semilogy()
plt.gca().invert_yaxis()
plt.xlim(0,3000)
plt.show()
"""

def TP_Line(P,g_plt,T_eq,k_IR,gamma1,gamma2,alpha,beta,T_int):
    """
    TP profile from eqn. (20) in Line et al. 2012.
    doi:10.1088/0004-637X/749/1/93
    Model parameters (5) : k_IR, gamma1, gamma2, alpha, beta, T_int

    Parameters
    ----------
    P : ndarray
        Pressure grid (in Pa) on which the TP profile is to be constructed.
    g_plt : real
        Gravitational acceleration at the highest pressure in the pressure
        grid.
    T_eq : real
        Temperature corresponding to the stellar flux.
        T_eq = T_star * (R_star/(2*semi_major_axis))**0.5
    k_IR : real
        Range [1e-5,1e3]
        Mean absorption coefficient in the thermal wavelengths.
        m^2/kg
    gamma_1 : real
        Range ~ [1e-3,1e2]
        gamma_1 = k_V1/k_IR, ratio of mean opacity of the first visible stream
        to mean opacity in the thermal stream.
    gamma_2 : real
        Range ~ [1e-3,1e2]
        gamma_2 = k_V2/k_IR, ratio of mean opacity of the second visible stream
        to mean opacity in the thermal stream.
    alpha : real
        Range [0,1]
        Percentage of the visible stream represented by opacity gamma1.
    beta : real
        Range [0,2]
        A catch all parameter for albedo, emissivity, day-night redistribution.
    T_int : real
        Temperature corresponding to the intrinsic heat flux of the planet.

    Returns
    -------
    TP : ndarray
        Temperature as a function of pressure.
    """
    T_irr = beta * T_eq
    tau = k_IR * P / g_plt # optical depth, assuming g_plt is constant
    xi1 = (2./3.) * ( 1 + (1/gamma1)*(1+(0.5*gamma1*tau-1)*np.exp(-gamma1*tau))\
            + gamma1 * (1-0.5*tau**2) * special.expn(2, gamma1*tau) )
    xi2 = (2./3.) * ( 1 + (1/gamma2)*(1+(0.5*gamma2*tau-1)*np.exp(-gamma2*tau))\
            + gamma2 * (1-0.5*tau**2) * special.expn(2, gamma2*tau) )
    flux1 = 0.75 * T_int**4 * (2./3.+tau)
    flux2 = T_irr**4 * (1-alpha) * xi1
    flux3 = T_irr**4 * alpha * xi2
    TP = (flux1+flux2+flux3)**0.25
    return TP

NLAYER = 100
P_grid = np.geomspace(10e8,100,NLAYER) # pressure in pa
T_eq = 1469/(2**0.5)
k_IR = 1e-3
gamma1 = 0.1
gamma2 = 0.1
alpha = 0.9
beta = 1
g = 25
#f = 3
plt.semilogy()
plt.gca().invert_yaxis()
plt.xlim(0,3000)

for k_IR in [1e-5,1e-2,1,10]:
    for gamma1 in [1e-3,1e-2,1,10]:
        for gamma2 in [1e-3,1e-2,1,10]:
            for alpha in [0.25,0.5,0.75,1]:
                for beta in [0,0.5,1,2]:
                    for T_int in np.linspace(10,3000,10):
                        x = TP_Line(P=P_grid,g_plt=g,T_eq=T_eq,k_IR=k_IR,gamma1=gamma1,gamma2=gamma2,
                                alpha=alpha,beta=beta,T_int=T_int)
                        plt.plot(x,P_grid/1e5,linewidth=0.5)
                        plt.show()
                        plt.semilogy()
                        plt.gca().invert_yaxis()
                        plt.xlim(0,3000)
