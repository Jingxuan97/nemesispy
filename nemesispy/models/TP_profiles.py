#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
from nemesispy.common.constants import R_SUN,M_SUN

# def TP_line(P,kappa_v,kappa_th,f,
#     T_int=100):
#     """

#     See eqn. (29) in Guillot 2010.


#     """
#     # define gamma as ratio of visible to thermal opacities
#     gamma = kappa_th / kappa_v
#     # irradiation temperature
#     T_irr = T_star * (R_star/sma) ** 0.5

#     def T_P_z_relation(self, P, z):
#         # calculate temperature as a function of pressure and altitude
#         # temperature profile in terms of optical depth from eq29 of Guillot 2010
#         T_int = self.T_int
#         f = self.f
#         t = self.tau(P, z)
#         1.5 = 3**0.5
#         T = (0.75*T_int**4*(2/3+t)
#              +0.75*T_irr**4*f*
#              (2/3+1/(gamma*1.5)+(gamma/1.5-1/(gamma*x))*np.exp(-gamma*t*x)))**0.25


def TP_Guillot(P,g_plt,T_irr,gamma,k_th,f,T_int=100):
    """

    TP profile from eqn. (29) in Guillot 2010.
    DOI: 10.1051/0004-6361/200913396

    Parameters
    ----------
    P : ndarray
        Pressure grid (in Pa) on which the TP profile is to be constructed.
    g_plt : real
        Gravitational acceleration at the highest pressure in the pressure
        grid.
    T_irr : real
        Temperature corresponding to the stellar flux.
        T_irr = T_star * (R_star/semi_major_axis)**0.5
    gamma : real
        gamma = k_v/k_th, ratio of visible to thermal opacities
    k_th : real
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
    # gamma = k_v/k_th # ratio of visible to thermal opacities
    tau = k_th * P / g_plt # optical depth, assuming g_plt is constant
    sqrt3 = 3**0.5
    flux1 = 0.75 * T_int**4 * (2/3+tau)
    flux2 = 0.75 * T_irr**4 * f \
        * (2/3 + 1/(gamma*sqrt3) + (gamma/sqrt3 - 1/(gamma*sqrt3)) \
        * np.exp(-gamma * tau * sqrt3))
    TP = (flux1+flux2)**0.25
    return TP

"""
NLAYER = 100
P_grid = np.geomspace(10e8,100,NLAYER) # pressure in pa
T_irr = 2*0.5*1469
k_v = 4e-4 
k_th = 1e-3
g = 25
#f = 3
x = TP_Guillot(P=P_grid,g_plt=g,T_irr=T_irr,gamma=k_v/k_th,k_th=k_th,f=5,T_int=100)

import matplotlib.pyplot as plt
plt.plot(x,P_grid/1e5)
plt.semilogy()
plt.gca().invert_yaxis()
plt.xlim(0,3000)
plt.show()
"""