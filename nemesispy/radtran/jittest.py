#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  9 18:43:36 2022

@author: jingxuanyang
"""
import numba
import numpy as np
from numba import jit
from nemesispy.radtran.utils import calc_mmw,find_nearest
from nemesispy.data.constants import C_LIGHT, K_B, PLANCK, AMU, G

H = np.linspace(0,1404762.36466,20)

# Pressure in pa, note 1 atm = 101325 pa
P = np.array([2.00000000e+06, 1.18757212e+06, 7.05163779e+05, 4.18716424e+05,
       2.48627977e+05, 1.47631828e+05, 8.76617219e+04, 5.20523088e+04,
       3.09079355e+04, 1.83527014e+04, 1.08975783e+04, 6.47083012e+03,
       3.84228875e+03, 2.28149750e+03, 1.35472142e+03, 8.04414702e+02,
       4.77650239e+02, 2.83622055e+02, 1.68410823e+02, 1.00000000e+02])

# Temperature in Kelvin
T = np.array([2294.22993056, 2275.69702232, 2221.47726725, 2124.54056941,
       1996.03871629, 1854.89143353, 1718.53879797, 1599.14914582,
       1502.97122783, 1431.0218576 , 1380.55933525, 1346.97814697,
       1325.49943114, 1312.13831743, 1303.97872899, 1299.05347108,
       1296.10266693, 1294.34217288, 1293.29484759, 1292.67284408])

@jit(nopython=True)
def calc_grav_simple(h, M_plt, R_plt):
    """
    TBD
    Sets the gravitational acceleration based on selected planet and
    latitude. Gravity is calculated normal to the surface, and corrected
    for rotational effects. The input latitude is assumed planetographic.

    Parameters
    ----------

    Returns
    -------
    """
    g = G*M_plt/(R_plt+h)**2
    gravity = g
    return gravity



def adjust_hydrostatH(H, P, T, ID, VMR, M_plt, R_plt):
    """
    Adjust the input altitude profile H according to hydrostatic equilibrium
    using the input pressure, temperature and VMR profiles and planet mass
    and radius. The routine calls calc_grav_simple to calculate gravity.

    Parameters
    ----------
    H : ndarray
        Altitude profile to be adjusted
    P : ndarray
        Pressure profile
    T : ndarray
        Temperature profile
    ID : ndarray
        Gas ID.
    VMR : ndarray
        Volume mixing ration profile.
    M_plt : real
        Planetary mass
    R_plt : real
        Planetary radius

    Returns
    -------
    H : ndarray
        Adjusted altitude profile satisfying hydrostatic equlibrium.

    """

    # note number of profile points and number of gases
    NPRO,NVMR = VMR.shape

    # initialise array for scale heights
    scale_height = np.zeros(NPRO)

    # First find level closest ot zero altitude
    alt0,ialt = find_nearest(H,0.0)
    if ( (alt0>0.0) & (ialt>0)):
        ialt = ialt -1

    #Calculate the mean molecular weight at each level
    XMOLWT = np.zeros(NPRO)
    for ipro in range(NPRO):
        XMOLWT[ipro] = calc_mmw(ID,VMR[ipro,:],ISO=None)

    # iterate until hydrostatic equilibrium
    XDEPTH = 2
    adjusted_H = H
    dummy_H = np.zeros(NPRO)
    while XDEPTH > 1:

        h = np.zeros(NPRO)
        dummy_H[:] = adjusted_H
        P[:] = P

        #Calculating the atmospheric model depth
        ATDEPTH = h[-1] - h[0]

        #Calculate the gravity at each altitude level
        gravity = np.zeros(NPRO)
        gravity[:] =  calc_grav_simple(h=h[:], M_plt=M_plt, R_plt=R_plt)

        #Calculate the scale height
        scale = np.zeros(NPRO)
        scale[:] = K_B*T[:]/(XMOLWT[:]*gravity[:])

        if ialt > 0 and ialt < NPRO-1 :
            h[ialt] = 0.0

        # nupper = NPRO - ialt - 1
        for i in range(ialt+1, NPRO):
            sh = 0.5 * (scale[i-1] + scale[i])
            #self.H[i] = self.H[i-1] - sh * np.log(self.P[i]/self.P[i-1])
            h[i] = h[i-1] - sh * np.log(P[i]/P[i-1])

        for i in range(ialt-1,-1,-1):
            sh = 0.5 * (scale[i+1] + scale[i])
            #self.H[i] = self.H[i+1] - sh * np.log(self.P[i]/self.P[i+1])
            h[i] = h[i+1] - sh * np.log(P[i]/P[i+1])

        atdepth1 = h[-1] - h[0]

        XDEPTH = 100.*abs((atdepth1-ATDEPTH)/ATDEPTH)
        # print('xdepth',XDEPTH)
        H = h[:]

    return H