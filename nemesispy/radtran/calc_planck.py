#!/usr/local/bin/python3
# -*- coding: utf-8 -*-
"""
Calculate the Planck function.
Note that in the Fortran Nemesis code, the values of constants used are
C1 = 1.1911e-12 W cm2
C2 = 1.439 cm K-1
"""
import numpy as np
from numba import jit

@jit(nopython=True)
def calc_planck(wave_grid,T,ispace=1):
    """
    Calculates the blackbody radiance.

    Parameters
    ----------
    wave_grid(nwave) : ndarray
        Wavelength or wavenumber array
        Unit: um or cm-1
    T : real
        Temperature of the blackbody (K)
    ispace : int
        Flag indicating the spectral units
        (0) Wavenumber (cm-1)
        (1) Wavelength (um)

    Returns
    -------
	bb(nwave) : ndarray
        Spectral radiance.
        Unit: (0) W cm-2 sr-1 (cm-1)-1
              (1) W cm-2 sr-1 um-1
    """
    wave = wave_grid.astype(np.float64)
    if np.any(wave<=0) or T < 0:
        raise(Exception('error in calc_planck: negative wavelengths' \
            +' or temperature'))
    C1 = np.array([1.1910e-12]) # W cm2 2*PLANCK*C_LIGHT**2
    C2 = np.array([1.4388]) # cm K-1 PLANCK*C_LIGHT/K_B
    if ispace==0:
        y = wave
        a = C1 * (y**3.)
    elif ispace==1:
        y = 1.0e4/wave
        a = C1 * (y**5.) / 1.0e4
    else:
        raise Exception('error in calc_planck: ISPACE must be either 0 or 1')

    tmp = C2 * y / T
    b = np.exp(tmp) - 1
    bb = (a/b)

    return bb