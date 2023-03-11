#!/usr/bin/env python3
#-*- coding: utf-8 -*-
import numpy as np
from nemesispy.models.TP_profiles import TP_Guillot
import scipy.interpolate as interpolate

def gen(lon_grid,
    ck0,ck1,ck2,sk1,sk2,
    cg0,cg1,cg2,sg1,sg2,
    cf0,cf1,cf2,sf1,sf2,
    cT0,cT1,cT2,sT1,sT2):
    """
    Generate the parameters for 2-Stream TP profile from Fourier coefficients.

    Parameters
    ----------
        lon_grid : ndarray
            Input longitude array
            Unit: degree.
        ck0,ck1,ck2,sk1,sk2 : reals
            Fourier coefficients for log kappa
        cg0,cg1,cg2,sg1,sg2 : reals
            Fourier coefficients for log gamma
        cf0,cf1,cf2,sf1,sf2 : reals
            Fourier coefficients for log f
        cT0,cT1,cT2,sT1,sT2 : real
            Fourier coefficients for log T_int
    Returns
    -------
        log_kappa : ndarray
            log kappa defined on the input longitude array
        log_gamma : ndarray
            log gamma defined on the input longitude array
        log_f : ndarray
            log f defined on the input longitude array
        T_int_array :
            T_int defined on the input longitude array
    """
    y = lon_grid/180*np.pi
    log_kappa = ck0 + ck1 * np.cos(y) + ck2 * np.cos(2*y)\
        + sk1 * np.sin(y) + sk2 * np.sin(2*y)
    log_gamma = cg0 + cg1 * np.cos(y) + cg2 * np.cos(2*y)\
        + sg1 * np.sin(y) + sg2 * np.sin(2*y)
    log_f = cf0 + cf1 * np.cos(y) + cf2 * np.cos(2*y)\
        + sf1 * np.sin(y) + sf2 * np.sin(2*y)
    log_T_int = cT0 + cT1 * np.cos(y) + cT2 * np.cos(2*y)\
        + sT1 * np.sin(y) + sT2 * np.sin(2*y)

    return log_kappa, log_gamma, log_f, log_T_int

def tmap3(P_grid, lon_grid, lat_grid,
        g_plt, T_eq, log_kappa, log_gamma, log_f, log_T_int):
    """
    Generate 3D temperature maps using 2-stream TP profile.
    The temperature map is defined on a (longitude,latitude,pressure) grid.

    Parameters
    ----------
    P_grid : ndarray
        Pressure grid for the temperature map.

    Returns
    -------
    tp_grid : ndarray
        Temperature map.
    """

    # set up output array
    nlon = len(lon_grid)
    nlat = len(lat_grid)
    nP = len(P_grid)
    tp_grid = np.zeros((nlon,nlat,nP))

    # convert log parameters to parameters
    k_array = 10**log_kappa
    g_array = 10**log_gamma
    f_array = 10**log_f
    T_int_array = 10**log_T_int

    # calculate equatorial TP profiles
    for ilon in range(nlon):
        tp = TP_Guillot(P=P_grid,g_plt=g_plt,T_eq=T_eq,
            k_IR=k_array[ilon],gamma=g_array[ilon],
            f=f_array[ilon],T_int=T_int_array[ilon])
        tp_grid[ilon,0,:] = tp

    T_evening = np.zeros(nP)
    T_morning = np.zeros(nP)
    for iP in range(nP):
        T_evening[iP] = np.interp(90,lon_grid,tp_grid[:,0,iP])
        T_morning[iP] = np.interp(-90,lon_grid,tp_grid[:,0,iP])

    TP_mean = 0.5 * (T_morning + T_evening)

    for ilat,lat in enumerate(lat_grid):
        for ilon in range(nlon):
            tp_grid[ilon,ilat,:] = TP_mean \
                + (tp_grid[ilon,0,:]- TP_mean) * np.cos(lat/180*np.pi)**0.25
    return tp_grid

def gen_tmap3(P_grid, lon_grid, lat_grid,
    g_plt,T_eq,
    ck0,ck1,ck2,sk1,sk2,
    cg0,cg1,cg2,sg1,sg2,
    cf0,cf1,cf2,sf1,sf2,
    cT0,cT1,cT2,sT1,sT2):

    log_kappa, log_gamma, log_f, log_T_int \
        = gen(lon_grid,
            ck0,ck1,ck2,sk1,sk2,
            cg0,cg1,cg2,sg1,sg2,
            cf0,cf1,cf2,sf1,sf2,
            cT0,cT1,cT2,sT1,sT2)
    tp_grid = tmap3(P_grid, lon_grid, lat_grid,
        g_plt, T_eq, log_kappa, log_gamma, log_f, log_T_int)
    return tp_grid

def gen_vmrmap1(h2o,co2,co,ch4,nlon,nlat,npress):
    """
    Generate a 3D gas abundance map.
    The abundance map is defined on a (longitude,latitude,pressure) grid.

    Parameters
    ---------

    Returns
    -------
        vmr_grid
    """
    vmr_grid = np.ones((nlon,nlat,npress,6))
    vmr_grid[:,:,:,0] *= 10**h2o
    vmr_grid[:,:,:,1] *= 10**co2
    vmr_grid[:,:,:,2] *= 10**co
    vmr_grid[:,:,:,3] *= 10**ch4
    vmr_grid[:,:,:,4] *= 0.16 * (1-10**h2o-10**co2-10**co-10**ch4)
    vmr_grid[:,:,:,5] *= 0.84 * (1-10**h2o-10**co2-10**co-10**ch4)
    return vmr_grid