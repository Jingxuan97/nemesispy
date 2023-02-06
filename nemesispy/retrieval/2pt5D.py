#!/usr/bin/env python3
#-*- coding: utf-8 -*-
import numpy as np
from nemesispy.models.TP_profiles import TP_Guillot
def arctan(x,y):
    """
    Calculate the argument of the point (x,y) in the range [0,2pi).

    Parameters
    ----------
        x : real
            x-coordinate of the point (length of the adjacent side)
        y : real
            y-coordinate of the point (length of the opposite side)
    Returns
    -------
        ang : real
            Argument of (x,y) in radians
    """
    if(x == 0.0):
        if (y == 0.0) : ang = 0.0 # (x,y) is the origin, ill-defined
        elif (y > 0.0) : ang = 0.5*np.pi # (x,y) is on positive y-axis
        else : ang = 1.5*np.pi  # (x,y) is on negative y-axis
    else:
        ang=np.arctan(y/x)
        if (y > 0.0) :
            if (x > 0.0) : ang = ang # (x,y) is in 1st quadrant
            else : ang = ang+np.pi # (x,y) is in 2nd quadrant
        elif (y == 0.0) :
            if (x > 0.0) : ang = 0 # (x,y) is on positive x-axis
            else : ang = np.pi # (x,y) is on negative x-axis
        else:
            if (x > 0.0) : ang = ang+2*np.pi # (x,y) is in 4th quadrant
            else : ang = ang+np.pi # (x,y) is in 3rd quadrant
    return ang

def gen(x,
    ck0,ck1,ck2,sk1,sk2,
    cg0,cg1,cg2,sg1,sg2,
    cf0,cf1,cf2,sf1,sf2,
    T_int):
    """
    Generate the parameters for 2-Stream TP profile from Fourier coefficients.

    Parameters
    ----------
        x : ndarray
            Input longitude array
            Unit: degree.
        ck0,ck1,ck2,sk1,sk2 : reals
            Fourier coefficients for log kappa
        cg0,cg1,cg2,sg1,sg2 : reals
            Fourier coefficients for log gamma
        cf0,cf1,cf2,sf1,sf2 : reals
            Fourier coefficients for log f
        T_int : real
            T_int
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
    y = x/180*np.pi
    y = x/180*np.pi
    log_kappa = ck0 + ck1 * np.cos(y) + ck2 * np.cos(2*y)\
        + sk1 * np.sin(y) + sk2 * np.sin(2*y)
    log_gamma = cg0 + cg1 * np.cos(y) + cg2 * np.cos(2*y)\
        + sg1 * np.sin(y) + sg2 * np.sin(2*y)
    log_f = cf0 + cf1 * np.cos(y) + cf2 * np.cos(2*y)\
        + sf1 * np.sin(y) + sf2 * np.sin(2*y)
    T_int_array = np.ones(y.shape) * T_int
    return log_kappa, log_gamma, log_f, T_int_array

def tmap1(P_grid, lon_grid, lat_grid,
        g_plt, T_eq, log_kappa, log_gamma, log_f, T_int_array):
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
    # # start with fixed grid
    # P_grid = P_range
    # lon_grid = global_lon_grid
    # lat_grid = global_lat_grid

    # set up output array
    nlon = len(lon_grid)
    nlat = len(lat_grid)
    nP = len(P_grid)
    tp_grid = np.zeros((nlon,nlat,nP))

    # convert log parameters to parameters
    k_array = 10**log_kappa
    g_array = 10**log_gamma
    f_array = 10**log_f
    T_int_array = T_int_array

    # calculate equatorial TP profiles
    for ilon in range(nlon):
        tp = TP_Guillot(P=P_grid,g_plt=g_plt,T_eq=T_eq,
            k_IR=k_array[ilon],gamma=g_array[ilon],
            f=f_array[ilon],T_int=T_int_array[ilon])
        tp_grid[ilon,0,:] = tp

    TP_mean = 0.5 * (tp_grid[17,0,:] + tp_grid[-18,0,:])
    for ilat,lat in enumerate(lat_grid):
        for ilon in range(nlon):
            tp_grid[ilon,ilat,:] = TP_mean \
                + (tp_grid[ilon,0,:]- TP_mean) * np.cos(lat/180*np.pi)**0.25
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