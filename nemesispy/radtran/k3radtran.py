#!/usr/local/bin/python3
# -*- coding: utf-8 -*-
import numpy as np
from copy import copy
# from numba import jit
from nemesispy.radtran.k2interp import interp_k, new_k_overlap
from nemesispy.radtran.k5cia import calc_tau_cia
def planck(wave,temp,ispace=1):
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
        Planck function (W cm-2 sr-1 (cm-1)-1 or W cm-2 sr-1 um-1)
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
        raise Exception('error in planck: ISPACE must be either 0 or 1')
    tmp = c2 * y / temp
    b = np.exp(tmp) - 1
    bb = a/b
    return bb

def tau_gas(k_gas_w_g_p_t, P_layer, T_layer, VMR_layer, U_layer,
            P_grid, T_grid, del_g):
    """
    Calculate the optical path due to gaseous absorbers.

    Parameters
    ----------
    k_gas_w_g_p_t(Ngas,Nwave,Ng,Npress,Ntemp) : ndarray
        Raw k-coefficients.
        Has dimension: Nwave x Ng x Npress x Ntemp.
    P_layer(Nlayer) : ndarray
        Atmospheric pressure grid.
    T_layer(Nlayer) : ndarray
        Atmospheric temperature grid.
    VMR_layer(Nlayer,Ngas) : ndarray
        Array of volume mixing ratios for Ngas.
        Has dimensioin: Nlayer x Ngas
    U_layer : ndarray
        DESCRIPTION.
    P_grid(Npress) : ndarray
        Pressure grid on which the k-coeff's are pre-computed.
    T_grid(Ntemp) : ndarray
        Temperature grid on which the k-coeffs are pre-computed.
    del_g : ndarray
        DESCRIPTION.

    Returns
    -------
    tau_w_g_l(Nwave,Ng,Nlayer) : ndarray
        DESCRIPTION.
    """
    k_gas_w_g_l = interp_k(P_grid, T_grid, P_layer, T_layer, k_gas_w_g_p_t) # NGAS,NWAVE,NG,NLAYER
    # Ngas, Nwave, Ng, Nlayer = k_gas_w_g_l.shape
    # print('k_gas_w_g_l', k_gas_w_g_l)

    k_w_g_l = new_k_overlap(k_gas_w_g_l, del_g, VMR_layer.T) # NWAVE,NG,NLAYER

    utotl = U_layer * 1.0e-4 # scaling

    TAUGAS = k_w_g_l * utotl # NWAVE, NG, NLAYER

    return TAUGAS

def radtran(wave_grid, U_layer, P_layer, T_layer, VMR_layer, k_gas_w_g_p_t,
            P_grid, T_grid, del_g, ScalingFactor, RADIUS, solspec,
            k_cia,ID,NU_GRID,CIA_TEMPS):
    """
    Calculate emission spectrum using the correlated-k method.
    Absorber amounts (U_layer) is scaled by a factor 1e-20 because Nemesis
    k-tables are scaled by a factor of 1e20.

    Parameters
    ----------
    wave_grid(Nwave) : ndarray
        Wavelengths (um) grid for calculating spectra.
    U_layer(Nlayer) : ndarray
        Total number of gas particles in each layer.
    P_layer(Nlayer) : ndarray
        Atmospheric pressure grid.
    T_layer(Nlayer) : ndarray
        Atmospheric temperature grid.
    VMR_layer(Nlayer,Ngas) : ndarray
        Array of volume mixing ratios for Ngas.
        Has dimensioin: Nlayer x Ngas
    k_gas_w_g_p_t : ndarray
        k-coefficients. Has dimension: Nwave x Ng x Npress x Ntemp.
    P_grid(Npress) : ndarray
        Pressure grid on which the k-coeff's are pre-computed.
    T_grid(Ntemp) : ndarray
        Temperature grid on which the k-coeffs are pre-computed.
    del_g : ndarray
        Quadrature weights of the g-ordinates.
    ScalingFactor : ndarray ( NLAY ) ##
        Scale stuff to line of sight
    RADIUS : real ##
        Planetary radius
    solspec : ndarray ##
        Stellar spectra, used when the unit of the output is in fraction
        of stellar irradiance.

    Returns
    -------
    radiance : ndarray
        Output radiance (W cm-2 um-1 sr-1)
    """
    # Reverse layers from TOP of atmoaphsere first to BOTTOM of atmosphere last
    U_layer = U_layer[::-1]
    P_layer = P_layer[::-1]
    T_layer = T_layer[::-1]
    VMR_layer = VMR_layer[::-1]

    U_layer *= 1.0e-20 # absorber amounts (U_layer) is scaled by a factor 1e-20

    # Dimensioins
    NGAS, NWAVE, NG, NGRID = k_gas_w_g_p_t.shape[:-1]
    # print('NGAS, NWAVE, NG, NGRID',NGAS, NWAVE, NG, NGRID)
    ### Second order opacities to be continued
    # Collision Induced Absorptioin Optical Path
    NLAY = len(P_layer)
    TAUCIA = np.zeros([NWAVE,NLAY])

    """
    TO BE DONE!
    """
    TAUCIA = calc_tau_cia(WAVE_GRID=wave_grid,K_CIA=k_cia,ISPACE=1,
        ID=ID,TOTAM=U_layer,T_layer=T_layer,VMR_layer=VMR_layer,
        NU_GRID=NU_GRID,TEMPS=CIA_TEMPS,INORMAL=0,NPAIR=9,DELH=1)
    # Rayleigh Scattering Optical Path
    TAURAY = np.zeros([NWAVE,NLAY])
    # Dust Scattering Optical Path
    TAUDUST = np.zeros([NWAVE,NLAY])

    ### Gaseous Opacity
    # Calculating the k-coefficients for each gas in each layer

    """
    vmr_gas = np.zeros([NGAS, NLAY])

    utotl = np.zeros(NLAY)

    for i in range(NLAY):
        vmr_gas[i,:] = None　#Layer.PP[:,IGAS].T / Layer.PRESS #VMR of each radiatively active gas
        utotl[:] = None #utotl[:] + Layer.AMOUNT[:,IGAS].T * 1.0e-4 * 1.0e-20   #Vertical column density of the radiatively active gases

    k_gas_w_g_l = interp_k(P_grid, T_grid, P_layer, T_layer, k_gas_w_g_p_t)
    k_w_g_l = mix_multi_gas_k(k_gas_w_g_l, del_g, vmr_gas)
    TAUGAS = k_w_g_l * utotl
    """
    TAUGAS = tau_gas(k_gas_w_g_p_t, P_layer, T_layer, VMR_layer, U_layer,
            P_grid, T_grid, del_g)
    # print('TAUGAS', TAUGAS)
    TAUTOT = np.zeros(TAUGAS.shape) # NWAVE x NG x NLAYER
    # print('TAUGAS.shape',TAUGAS.shape)
    # Merge all different opacities
    for ig in range(NG): # wavebin x layer / NWAVE x NG x NLAYER
        TAUTOT[:,ig,:] = TAUGAS[:,ig,:] + TAUCIA[:,:] + TAUDUST[:,:] + TAURAY[:,:]

    #Scale to the line-of-sight opacities
    TAUTOT_LAYINC = TAUTOT * ScalingFactor

    # Thermal Emission Calculation
    # IMOD = 3

    #Defining the units of the output spectrum / divide by stellar spectrum
    xfac = np.pi*4.*np.pi*((RADIUS)*1.0e2)**2.
    xfac = xfac / solspec

    #Calculating spectrum
    taud = np.zeros((NWAVE,NG))
    trold = np.zeros((NWAVE,NG))
    specg = np.zeros((NWAVE,NG))


    # taud[:,:] = taud[:,:] + TAUTOT_LAYINC[:,:,j]
    # NEED LAYER REVERSAL
    # print('T_layer1', T_layer)
    # T_layer = T_layer[::-1]
    # print('T_layer2', T_layer)

    # TAUTOT = TAUTOT[:,:,::-1]

    # Thermal Emission from planet
    # SPECOUT = np.zeros(NWAVE, NG)

    for ilayer in range(NLAY):

        taud[:,:] = TAUTOT[:,:,-ilayer]

        tr = np.exp(-taud) # transmission function

        bb = planck(wave_grid, T_layer[-ilayer]) # blackbody function

        for ig in range(NG):
            specg[:,ig] = specg[:,ig] + (trold[:,ig]-tr[:,ig])*bb[:] * xfac

        trold = copy(tr)

    # surface/bottom layer contribution
    p1 = P_layer[int(len(P_layer)/2-1)] #midpoint
    p2 = P_layer[-1] #lowest point in altitude/highest in pressure

    surface = None
    if p2 > p1: # i.e. if not a limb path
        print(p2,p1)
        if surface is None:
            radground = planck(wave_grid,T_layer[-1])
            print('radground',radground)
        for ig in range(NG):
            specg[:,ig] = specg[:,ig] + trold[:,ig]*radground*xfac

    SPECOUT = np.tensordot(specg, del_g, axes=([1],[0])) * xfac

    print('TAUCIA',TAUCIA)
    return SPECOUT
