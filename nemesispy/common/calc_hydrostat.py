#!/usr/local/bin/python3
# -*- coding: utf-8 -*-
"""
Use hydrostatic equilibrium to find altitudes given pressures, temperatures
and mean molecular weights.
"""
import numpy as np
from numba import jit
from nemesispy.common.constants import G, K_B

@jit(nopython=True)
def calc_grav_simple(h, M_plt, R_plt):
    """
    Calculates the gravitational acceleration at altitude h on a planet.

    Parameters
    ----------
    h : real
        Altitude.
        Unit: m
    M_plt : real
        Planet mass.
        Unit: kg
    R_plt : real
        Planet radius.
        Unit: m

    Returns
    -------
    g : real
        Gravitational acceleration.
        Unit: ms^-2
    """
    g = G*M_plt/(R_plt+h)**2
    return g

@jit(nopython=True)
def calc_hydrostat(P, T, mmw, M_plt, R_plt, H=np.array([])):
    """
    Calculates an altitude profile from given pressure, temperature and
    mean molecular weight profiles assuming hydrostatic equilibrium.


    Parameters
    ----------
    P : ndarray
        Pressure profile
        Unit: Pa
    T : ndarray
        Temperature profile
        Unit: K
    mmw : ndarray
        Mean molecular weight profile
        Unit: kg
    M_plt : real
        Planetary mass
        Unit: kg
    R_plt : real
        Planetary radius
        Unit: m
    H : ndarray
        Altitude profile to be adjusted
        Unit: m

    Returns
    -------
    adjusted_H : ndarray
        Altitude profile satisfying hydrostatic equlibrium.
        Unit: m

    """
    # Note number of profile points and set up a temporary height profile
    NPRO = len(P)
    if len(H)==0:
        H = np.linspace(0,1e6,NPRO)

    # First find level closest ot zero altitude
    ialt = (np.abs(H - 0.0)).argmin()
    alt0 = H[ialt]
    if ( (alt0>0.0) & (ialt>0)):
        ialt = ialt -1

    # iterate until hydrostatic equilibrium
    xdepth = 2
    adjusted_H = H
    dummy_H = np.zeros(NPRO)
    while xdepth > 1:

        dummy_H = adjusted_H

        # Calculate the atmospheric model depth
        atdepth = dummy_H[-1] - dummy_H[0]
        # Calculate the gravity at each altitude level
        gravity =  calc_grav_simple(h=dummy_H, M_plt=M_plt, R_plt=R_plt)
        # Calculate the scale height
        scale = K_B*T/(mmw*gravity)

        if ialt > 0 and ialt < NPRO-1 :
            dummy_H[ialt] = 0.0

        # nupper = NPRO - ialt - 1
        for i in range(ialt+1, NPRO):
            sh = 0.5 * (scale[i-1] + scale[i])
            dummy_H[i] = dummy_H[i-1] - sh * np.log(P[i]/P[i-1])

        for i in range(ialt-1,-1,-1):
            sh = 0.5 * (scale[i+1] + scale[i])
            dummy_H[i] = dummy_H[i+1] - sh * np.log(P[i]/P[i+1])

        atdepth1 = dummy_H[-1] - dummy_H[0]
        xdepth = 100.*abs((atdepth1-atdepth)/atdepth)
        adjusted_H = dummy_H

    return adjusted_H