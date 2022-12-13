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
        Range ~ [1e-3,1e2]
        gamma = k_V/k_IR, ratio of visible to thermal opacities
    k_IR : real
        Range [1e-5,1e3]
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
        Range ~ [1e-5,1e3]
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
