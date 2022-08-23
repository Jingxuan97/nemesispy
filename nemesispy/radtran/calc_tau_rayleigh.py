#!/usr/local/bin/python3
# -*- coding: utf-8 -*-
"""
Calculate Rayleigh scattering optical path.
"""
from numba import jit
import numpy as np

@jit(nopython=True)
def calc_tau_rayleigh(wave_grid,U_layer,ISPACE=1):
    """
    Calculate the Rayleigh scattering optical depth for Gas Giant atmospheres
    using data from Allen (1976) Astrophysical Quantities.

    Assume H2 ratio of 0.864.

    Parameters
    ----------
    wave_grid(NWAVE) : ndarray
        Wavenumber (cm-1) or wavelength array (um)
    U_layer(NLAYER) : ndarray
        Total absober amount
    ISPACE : int
        Flag indicating the spectral units
        (0) Wavenumber in cm-1 or (1) Wavelegnth (um)

    Outputs
    -------
    tau_rayleigh(NWAVE,NLAYER) : ndarray
        Rayleigh scattering optical path at each wavlength in each layer
    """
    AH2 = 13.58E-5
    BH2 = 7.52E-3
    AHe = 3.48E-5
    BHe = 2.30E-3
    fH2 = 0.864
    k = 1.37971e-23 # JK-1
    P0 = 1.01325e5 # Pa
    T0 = 273.15 # K

    NLAYER = len(U_layer)
    NWAVE = len(wave_grid)

    if ISPACE == 0:
        LAMBDA = 1./wave_grid * 1.0e-2 # converted wavelength unit to m
        x = 1.0/(LAMBDA*1.0e6)
    else:
        LAMBDA = wave_grid * 1.0e-6 # wavelength in m
        x = 1.0/(LAMBDA*1.0e6)

    # calculate refractive index
    nH2 = AH2 * (1.0+BH2*x*x)
    nHe = AHe * (1.0+BHe*x*x)

    #calculate Jupiter air's refractive index at STP (Actually n-1)
    nAir = fH2 * nH2 + (1-fH2)*nHe

    #H2,He Seem pretty isotropic to me?...Hence delta = 0.
    #Penndorf (1957) quotes delta=0.0221 for H2 and 0.025 for He.
    #(From Amundsen's thesis. Amundsen assumes delta=0.02 for H2-He atmospheres
    delta = 0.0
    temp = 32*(np.pi**3.)*nAir**2.
    N0 = P0/(k*T0)

    x = N0*LAMBDA*LAMBDA
    faniso = (6.0+3.0*delta)/(6.0 - 7.0*delta)

    # Calculate the scattering cross sections in m2
    k_rayleighj = temp*faniso/(3.*(x**2)) #(NWAVE)

    # Calculate the Rayleigh opacities in each layer
    tau_rayleigh = np.zeros((len(wave_grid),NLAYER))

    for ilay in range(NLAYER):
        tau_rayleigh[:,ilay] = k_rayleighj[:] * U_layer[ilay] #(NWAVE,NLAYER)

    return tau_rayleigh