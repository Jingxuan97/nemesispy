#!/usr/local/bin/python3
# -*- coding: utf-8 -*-
import numpy as np
from copy import copy

from numpy.core.defchararray import array
from nemesispy.radtran.interp import new_k_overlap
from nemesispy.radtran.interp import interp_k
from nemesispy.radtran.cia import calc_tau_cia

def calc_planck(wave,temp,ispace=1):
    """
    Calculate the blackbody radiation given by the Planck function

    Parameters
    ----------
    wave(nwave) : ndarray
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
    c1 = 1.1911e-12
    c2 = 1.439
    if ispace==0:
        y = wave
        a = c1 * (y**3.)
    elif ispace==1:
        y = 1.0e4/wave
        a = c1 * (y**5.) / 1.0e4
    else:
        raise Exception('error in calc_planck: ISPACE must be either 0 or 1')

    tmp = c2 * y / temp
    b = np.exp(tmp) - 1
    bb = a/b

    return bb

def calc_tau_rayleighj(wave_grid,TOTAM,ISPACE=1):
    """
    Function to calculate the Rayleigh scattering opacity in each atmospheric layer,
    for Gas Giant atmospheres using data from Allen (1976) Astrophysical Quantities

    Parameters
    ----------
        ISPACE : int
            Flag indicating the spectral units
            (0) Wavenumber in cm-1 or (1) Wavelegnth (um)
        wave_grid(NWAVE) : ndarray
            Wavenumber (cm-1) or wavelength array (um)

    Outputs
    -------
    tau_rayleigh(NWAVE,NLAYER) : ndarray
        Rayleigh scattering opacity in each layer
    """
    AH2 = 13.58E-5
    BH2 = 7.52E-3
    AHe = 3.48E-5
    BHe = 2.30E-3
    fH2 = 0.864
    k = 1.37971e-23 # JK-1
    P0 = 1.01325e5 # Pa
    T0 = 273.15 # K

    NLAYER = len(TOTAM)
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
    tau_rayleigh = np.zeros([len(wave_grid),NLAYER])

    for ilay in range(NLAYER):
        tau_rayleigh[:,ilay] = k_rayleighj[:] * TOTAM[ilay] #(NWAVE,NLAYER)

    return tau_rayleigh

def calc_tau_gas(k_gas_w_g_p_t, P_layer, T_layer, VMR_layer, U_layer,
            P_grid, T_grid, del_g):
    """
    Calculate the optical path due to gaseous absorbers.

    Parameters
    ----------
    k_gas_w_g_p_t(NGAS,NWAVE,NG,NPRESSKTA,NTEMPKTA) : ndarray
        Raw k-coefficients.
        Has dimension: NWAVE x NG x NPRESSKTA x NTEMPKTA.
    P_layer(NLAYER) : ndarray
        Atmospheric pressure grid.
    T_layer(NLAYER) : ndarray
        Atmospheric temperature grid.
    VMR_layer(NLAYER,NGAS) : ndarray
        Array of volume mixing ratios for NGAS.
        Has dimensioin: NLAYER x NGAS
    U_layer : ndarray
        DESCRIPTION.
    P_grid(NPRESSKTA) : ndarray
        Pressure grid on which the k-coeff's are pre-computed.
    T_grid(NTEMPKTA) : ndarray
        Temperature grid on which the k-coeffs are pre-computed.
    del_g : ndarray
        DESCRIPTION.

    Returns
    -------
    tau_w_g_l(NWAVE,NG,NLAYER) : ndarray
        DESCRIPTION.
    """

    U_layer *= 1.0e-20 # absorber amounts (U_layer) is scaled by a factor 1e-20
    U_layer *= 1.0e-4 # convert from absorbers per m^2 to per cm^2

    k_gas_w_g_l = interp_k(P_grid, T_grid, P_layer, T_layer, k_gas_w_g_p_t) # NGAS,NWAVE,NG,NLAYER

    k_w_g_l,f_combined = new_k_overlap(k_gas_w_g_l, del_g, VMR_layer) # NWAVE,NG,NLAYER

    utotl = U_layer

    tau_gas = k_w_g_l * utotl * f_combined  # NWAVE, NG, NLAYER

    return tau_gas

def radtran(wave_grid, U_layer, P_layer, T_layer, VMR_layer, k_gas_w_g_p_t,
            P_grid, T_grid, del_g, ScalingFactor, RADIUS, solspec,
            k_cia, ID, NU_GRID, CIA_TEMPS, DEL_S):
    """
    Calculate emission spectrum using the correlated-k method.

    # Absorber amounts (U_layer) is scaled by a factor 1e-20 because Nemesis
    # k-tables are scaled by a factor of 1e20. Done in calc_tau_gas.

    # Need to be smart about benchmarking against NEMESIS

    Parameters
    ----------
    wave_grid(NWAVE) : ndarray
        Wavelengths (um) grid for calculating spectra.
    U_layer(NLAYER) : ndarray
        Total number of gas particles in each layer.
        We want SI unit (no. of particle/m^2) here.
    P_layer(NLAYER) : ndarray
        Atmospheric pressure grid.
        We want SI unit (Pa) here.
    T_layer(NLAYER) : ndarray
        Atmospheric temperature grid. In Kelvin.
    VMR_layer(NLAYER,NGAS) : ndarray
        Array of volume mixing ratios for NGAS.
        Has dimensioin: NLAYER x NGAS
    k_gas_w_g_p_t : ndarray
        k-coefficients. Has dimension: NWAVE x NG x NPRESSKTA x NTEMPKTA.
    P_grid(NPRESSKTA) : ndarray
        Pressure grid on which the k-coeff's are pre-computed.
        We want SI unit (Pa) here.
    T_grid(NTEMPKTA) : ndarray
        Temperature grid on which the k-coeffs are pre-computed. In Kelvin
    del_g : ndarray
        Quadrature weights of the g-ordinates.
    ScalingFactor(NLAYER) : ndarray ##
        Scale stuff to line of sight
    RADIUS : real ##
        Planetary radius
        We want SI unit (m) here.
    solspec : ndarray ##
        Stellar spectra, used when the unit of the output is in fraction
        of stellar irradiance.

        Stellar flux at planet's distance (W cm-2 um-1 or W cm-2 (cm-1)-1)

    Returns
    -------
    SPECOUT : ndarray
        Output spectrum (W cm-2 um-1 sr-1)
    """
    # Reverse layer ordering from TOP of atmoaphsere first to BOTTOM of atmosphere last
    P_layer = P_layer[::-1]
    T_layer = T_layer[::-1]
    U_layer = U_layer[::-1]
    ScalingFactor = ScalingFactor[::-1]
    VMR_layer = VMR_layer[::-1,:]

    # Record constants
    NGAS, NWAVE, NG, NPRESS, NTEMP = k_gas_w_g_p_t.shape
    NLAYER = len(P_layer)

    # Initiate arrays to record optical paths (NWAVE x NG x NLAYER)
    tau_total_w_g_l = np.zeros([NWAVE,NG,NLAYER]) # Total Optical path

    # Rayleigh scattering optical path (NWAVE x NLAYER)
    tau_rayleigh = calc_tau_rayleighj(wave_grid=wave_grid,TOTAM=U_layer)

    # Collision induced absorptioin optical path (NWAVE x NLAYER)
    tau_cia = calc_tau_cia(wave_grid=wave_grid,K_CIA=k_cia,ISPACE=1,
        ID=ID,TOTAM=U_layer,T_layer=T_layer,P_layer=P_layer,VMR_layer=VMR_layer,DELH=DEL_S,
        NU_GRID=NU_GRID,TEMPS=CIA_TEMPS,INORMAL=0,NPAIR=9)

    # Dust scattering optical path (NWAVE x NLAYER)
    tau_dust = np.zeros([NWAVE,NLAYER])
    """
    DUST opacity TO BE DONE!
    """

    # Active gas optical path (NWAVE x NG x NLAYER)
    tau_gas = calc_tau_gas(k_gas_w_g_p_t, P_layer, T_layer, VMR_layer, U_layer,
            P_grid, T_grid, del_g)

    # Merge all different opacities
    for ig in range(NG):
        tau_total_w_g_l[:,ig,:] = tau_gas[:,ig,:] + tau_cia[:,:] + tau_dust[:,:] + tau_rayleigh[:,:]

    #Scale to the line-of-sight opacities
    tau_total_w_g_l = tau_total_w_g_l * ScalingFactor


    # Thermal Emission Calculation
    # IMOD = 3
    NPATH = 1
    spec_out = np.zeros([NWAVE,NG,NPATH])

    # Defining the units of the output spectrum / divide by stellar spectrum
    # IFORM = 1
    radextra = sum(DEL_S[:-1])
    #radextra*=0
    xfac = np.pi*4.*np.pi*((RADIUS+radextra)*1e2)**2.
    xfac = xfac / solspec

    # Calculate spectrum
    for ipath in range(NPATH):

        #Calculating atmospheric contribution
        tau_cumulative_w_g = np.zeros((NWAVE,NG))
        tr_old_w_g = np.ones((NWAVE,NG))
        spec_w_g = np.zeros((NWAVE,NG))

        for ilayer in range(NLAYER):

            tau_cumulative_w_g[:,:] =  tau_total_w_g_l[:,:,ilayer] + tau_cumulative_w_g[:,:]
            tr_w_g = np.exp(-tau_cumulative_w_g) # transmission function
            bb = calc_planck(wave_grid, T_layer[ilayer]) # blackbody function
            for ig in range(NG):
                spec_w_g[:,ig] = spec_w_g[:,ig]+(tr_old_w_g[:,ig]-tr_w_g[:,ig])*bb[:]

            # tr_old_w_g = copy(tr_w_g)
            tr_old_w_g = tr_w_g

        # surface/bottom layer contribution
        p1 = P_layer[int(len(P_layer)/2-1)] #midpoint
        p2 = P_layer[-1] # lowest point in altitude/highest in pressure

        surface = None
        if p2 > p1: # i.e. if not a limb path
            if not surface:
                radground = calc_planck(wave_grid,T_layer[-1])
            for ig in range(NG):
                spec_w_g[:,ig] = spec_w_g[:,ig] + tr_old_w_g[:,ig]*radground

        spec_out[:,:,ipath] = spec_w_g[:,:]

    spec_out = np.tensordot(spec_out, del_g, axes=([1],[0])) * xfac
    spec_out = spec_out.T[0]

    return spec_out