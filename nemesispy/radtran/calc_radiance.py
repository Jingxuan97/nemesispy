#!/usr/local/bin/python3
# -*- coding: utf-8 -*-
"""
Routines to calculate thermal emission spectra from a planetary atmosphere
using the correlated-k method to combine gaseous opacities.
We inlcude collision-induced absorption from H2-H2, H2-he, H2-N2, N2-Ch4, N2-N2,
Ch4-Ch4, H2-Ch4 pairs and Rayleigh scattering from H2 molecules and
He molecules.

As of now the routines are fully accelerated using numba.jit.
"""
import numpy as np
from numba import jit
from nemesispy.radtran.calc_planck import calc_planck
from nemesispy.radtran.calc_tau_gas import calc_tau_gas
from nemesispy.radtran.calc_tau_cia import calc_tau_cia
from nemesispy.radtran.calc_tau_rayleigh import calc_tau_rayleigh

@jit(nopython=True)
def calc_radiance(wave_grid, U_layer, P_layer, T_layer, VMR_layer, k_gas_w_g_p_t,
    P_grid, T_grid, del_g, ScalingFactor, RADIUS, solspec,
    k_cia, ID, cia_nu_grid, cia_T_grid, DEL_S):
    """
    Calculate emission spectrum using the correlated-k method.

    Parameters
    ----------
    wave_grid(NWAVE) : ndarray
        Wavelengths (um) grid for calculating spectra.
    U_layer(NLAYER) : ndarray
        Total number of gas particles in each layer.
        Unit: no. of particle/m^2
    P_layer(NLAYER) : ndarray
        Atmospheric pressure grid.
        Unit: Pa
    T_layer(NLAYER) : ndarray
        Atmospheric temperature grid.
        Unit: K
    VMR_layer(NLAYER,NGAS) : ndarray
        Array of volume mixing ratios for NGAS.
        Has dimensioin: NLAYER x NGAS
    k_gas_w_g_p_t(NGAS,NWAVE,NG,NPRESSK,NTEMPK) : ndarray
        k-coefficients.
        Has dimension: NGAS x NWAVE x NG x NPRESSK x NTEMPK.
    P_grid(NPRESSK) : ndarray
        Pressure grid on which the k-coeff's are pre-computed.
        We want SI unit (Pa) here.
    T_grid(NTEMPK) : ndarray
        Temperature grid on which the k-coeffs are pre-computed. In Kelvin
    del_g : ndarray
        Quadrature weights of the g-ordinates.
    ScalingFactor(NLAYER) : ndarray
        Scale stuff to line of sight
    RADIUS : real
        Planetary radius
        Unit: m
    solspec : ndarray
        Stellar spectra, used when the unit of the output is in fraction
        of stellar irradiance.

        Stellar flux at planet's distance (W cm-2 um-1 or W cm-2 (cm-1)-1)

    Returns
    -------
    spectrum : ndarray
        Output spectrum (W cm-2 um-1 sr-1)
    """
    # Reorder atmospheric layers from top to bottom
    ScalingFactor = ScalingFactor[::-1]
    P_layer = P_layer[::-1] # layer pressures (Pa)
    T_layer = T_layer[::-1] # layer temperatures (K)
    U_layer = U_layer[::-1] # layer absorber amounts (no./m^2)
    VMR_layer = VMR_layer[::-1,:] # layer volume mixing ratios
    DEL_S = DEL_S[::-1] # path lengths in each layer

    # Record array dimensions
    NGAS, NWAVE, NG, NPRESS, NTEMP = k_gas_w_g_p_t.shape
    NLAYER = len(P_layer)

    # Initiate arrays to record total optical paths
    tau_total_w_g_l = np.zeros((NWAVE,NG,NLAYER))

    # Collision induced absorptioin optical path (NWAVE x NLAYER)
    tau_cia = calc_tau_cia(wave_grid=wave_grid,K_CIA=k_cia,ISPACE=1,
        ID=ID,TOTAM=U_layer,T_layer=T_layer,P_layer=P_layer,VMR_layer=VMR_layer,
        DELH=DEL_S,cia_nu_grid=cia_nu_grid,TEMPS=cia_T_grid,INORMAL=1,NPAIR=9)

    # Rayleigh scattering optical path (NWAVE x NLAYER)
    tau_rayleigh = calc_tau_rayleigh(wave_grid=wave_grid,U_layer=U_layer)

    # Dust scattering optical path (NWAVE x NLAYER)
    tau_dust = np.zeros((NWAVE,NLAYER))

    # FORTRAN straight transcript
    tau_gas = calc_tau_gas(k_gas_w_g_p_t, P_layer, T_layer, VMR_layer, U_layer,
            P_grid, T_grid, del_g)

    # Merge all different opacities
    for ig in range(NG):
        tau_total_w_g_l[:,ig,:] = tau_gas[:,ig,:] + tau_cia[:,:] \
            + tau_dust[:,:] + tau_rayleigh[:,:]

    #Scale to the line-of-sight opacities
    tau_total_w_g_l = tau_total_w_g_l * ScalingFactor

    # Defining the units of the output spectrum / divide by stellar spectrum
    # IFORM = 1
    # radextra = np.sum(DEL_S[:-1])
    # radextra = 0
    # xfac = np.pi*4.*np.pi*((RADIUS+radextra)*1e2)**2./solspec[:]
    xfac = np.pi*4.*np.pi*(RADIUS*1e2)**2./solspec[:]

    # Calculating atmospheric gases contribution
    tau_cumulative_w_g = np.zeros((NWAVE,NG))
    tr_old_w_g = np.ones((NWAVE,NG))
    spec_w_g = np.zeros((NWAVE,NG))

    for ilayer in range(NLAYER):
        tau_cumulative_w_g[:,:] =  tau_total_w_g_l[:,:,ilayer] + tau_cumulative_w_g[:,:]
        tr_w_g = np.exp(-tau_cumulative_w_g[:,:]) # transmission function
        bb = calc_planck(wave_grid, T_layer[ilayer]) # blackbody function

        for iwave in range(NWAVE):
            for ig in range(NG):
                spec_w_g[iwave,ig] = spec_w_g[iwave,ig] \
                    + np.float32(tr_old_w_g[iwave,ig]-tr_w_g[iwave,ig])\
                    *bb[iwave]*xfac[iwave]

        tr_old_w_g = tr_w_g

    # surface/bottom layer contribution
    p1 = P_layer[int(len(P_layer)/2-1)] #midpoint
    p2 = P_layer[-1] # lowest point in altitude/highest in pressure

    surface = None
    if p2 > p1: # i.e. if not a limb path
        if not surface:
            radground = calc_planck(wave_grid,T_layer[-1])

        for ig in range(NG):
            spec_w_g[:,ig] = spec_w_g[:,ig] + tr_old_w_g[:,ig]*radground*xfac

    # Integrate over g-ordinates
    spectrum = np.zeros((NWAVE))
    for iwave in range(NWAVE):
        for ig in range(NG):
            spectrum[iwave] += spec_w_g[iwave,ig]*del_g[ig]

    return spectrum