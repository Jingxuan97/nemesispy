#!/usr/local/bin/python3
# -*- coding: utf-8 -*-
import numpy as np
from numba import jit

@jit(nopython=True)
def calc_planck(wave_grid,temp,ispace=1):
    """
    Calculates the blackbody radiation.

    Parameters
    ----------
    wave_grid(nwave) : ndarray
        Wavelength or wavenumber array
    temp : real
        Temperature of the blackbody (K)
    ispace : int
        Flag indicating the spectral units
        (0) Wavenumber (cm-1)
        (1) Wavelength (um)

    Returns
    -------
	bb(nwave) : ndarray
        Planck function
        Unit: (0) W cm-2 sr-1 (cm-1)-1
              (1) W cm-2 sr-1 um-1
    """
    c1 = np.array([1.1911e-12])
    c2 = np.array([1.439])
    if ispace==0:
        y = wave_grid
        a = c1 * (y**3.)
    elif ispace==1:
        y = 1.0e4/wave_grid
        a = c1 * (y**5.) / 1.0e4
    else:
        raise Exception('error in calc_planck: ISPACE must be either 0 or 1')

    tmp = c2 * y / temp
    b = np.exp(tmp) - 1
    bb = (a/b)

    return bb