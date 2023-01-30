# -*- coding: utf-8 -*-
import numpy as np
import os
import pymultinest

# nemesispy modules
from nemesispy.common.constants import G
from nemesispy.models.TP_profiles import TP_Guillot
from nemesispy.common.interpolate_gcm import interp_gcm_X
from nemesispy.data.gcm.wasp43b_vivien.process_wasp43b_gcm_vivien import (
    nlon,nlat,xlon,xlat,npv,pv,\
    tmap,h2omap,comap,co2map,ch4map,hemap,h2map,vmrmap,\
    tmap_mod,h2omap_mod,comap_mod,co2map_mod,ch4map_mod,\
    hemap_mod,h2map_mod,vmrmap_mod,phase_grid,\
    kevin_phase_by_wave,kevin_wave_by_phase,\
    pat_phase_by_wave,pat_wave_by_phase)

### Fit data
NLAYER = 20
P_range = np.geomspace(20*1e5,1e-3*1e5,20)
grid_longitudes = xlon
grid_latitudes = xlat
interped_GCM = np.zeros((nlon,nlat,NLAYER))
for ilon,longi in enumerate(xlon):
    for ilat,lati in enumerate(xlat):
        interped_GCM[ilon,ilat,:] \
            = interp_gcm_X(longi,lati,P_range,
                xlon,xlat,pv,tmap_mod)

### Reference Planet Input: WASP 43b
T_star = 4520 # star temperature in K
R_star = 463892759.99999994 # m, 0.6668 * R_SUN
SMA = 2243970000.0 # m, 0.015*AU
M_plt = 3.8951064000000004e+27 # kg
R_plt = 74065.70 * 1e3 # m
T_irr = T_star * (R_star/SMA)**0.5
T_eq = T_irr/2**0.5
g = G*M_plt/R_plt**2

def gen(x,
    ck0,ck1,ck2,sk1,sk2, # log kappa Fourier coefficients
    cg0,cg1,cg2,sg1,sg2, # log gamma Fourier coefficients
    cf0,cf1,cf2,sf1,sf2, # llog f Fourier coefficients
    T_int):
    """Generate log(2-Stream Guillot parameters) from Fourier coefficients"""
    y = x/180*np.pi
    log_kappa = ck0 + ck1 * np.cos(y) + ck2 * np.cos(2*y)\
        + sk1 * np.sin(y) + sk2 * np.sin(2*y)
    log_gamma = cg0 + cg1 * np.cos(y) + cg2 * np.cos(2*y)\
        + sg1 * np.sin(y) + sg2 * np.sin(2*y)
    log_f = cf0 + cf1 * np.cos(y) + cf2 * np.cos(2*y)\
        + sf1 * np.sin(y) + sf2 * np.sin(2*y)
    T_int_array = np.ones(y.shape) * T_int
    return log_kappa, log_gamma, log_f, T_int_array

def tmap1(g_plt, T_eq, log_kappa, log_gamma, log_f, T_int_array):
    """Generate tmap"""
    # start with fixed grid
    P_grid = np.geomspace(20*1e5,1e-3*1e5,20)
    lon_grid = grid_longitudes
    lat_grid = grid_latitudes

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

    T_morning = (tp_grid[15,0,:] + tp_grid[16,0,:])/2
    T_evening = (tp_grid[-16,0,:] + tp_grid[-17,0,:])/2
    TP_mean = 0.5 * (T_morning + T_evening)

    for ilat,lat in enumerate(lat_grid):
        for ilon in range(nlon):
            tp_grid[ilon,ilat,:] = TP_mean \
                + (tp_grid[ilon,0,:]- TP_mean) * np.cos(lat/180*np.pi)**0.25
    return tp_grid

def gen_tmap1(g_plt,T_eq,
    ck0,ck1,ck2,sk1,sk2,
    cg0,cg1,cg2,sg1,sg2,
    cf0,cf1,cf2,sf1,sf2,
    T_int):
    lon_grid = grid_longitudes
    log_kappa, log_gamma, log_f, T_int_array \
        = gen(lon_grid,
            ck0,ck1,ck2,sk1,sk2,
            cg0,cg1,cg2,sg1,sg2,
            cf0,cf1,cf2,sf1,sf2,
            T_int)
    tp_grid = tmap1(g_plt, T_eq, log_kappa, log_gamma, log_f, T_int_array)
    print('tp_grid shape',tp_grid.shape)
    return tp_grid

def Prior(cube,ndim,nparams):
    # log_kappa
    cube[0] = -4 + (2 - (-4)) * cube[0]
    cube[1] = -1 + (1 - (-1)) * cube[1]
    cube[2] = -1 + (1 - (-1)) * cube[2]
    cube[3] = -1 + (1 - (-1)) * cube[3]
    cube[4] = -1 + (1 - (-1)) * cube[4]
    # log_gamma
    cube[5] = -4 + (1 - (-4)) * cube[5]
    cube[6] = -1 + (1 - (-1)) * cube[6]
    cube[7] = -1 + (1 - (-1)) * cube[7]
    cube[8] = -1 + (1 - (-1)) * cube[8]
    cube[9] = -1 + (1 - (-1)) * cube[9]
    # log_f
    cube[10] = -3 + (1 - (-3)) * cube[10]
    cube[11] = -1 + (1 - (-1)) * cube[11]
    cube[12] = -1 + (1 - (-1)) * cube[12]
    cube[13] = -1 + (1 - (-1)) * cube[13]
    cube[14] = -1 + (1 - (-1)) * cube[14]
    # T_int
    cube[15] = 100 + (1000 - (100)) * cube[15]

def LogLikelihood(cube,ndim,nparams):
    tp_grid = gen_tmap1(g,T_eq,
        cube[0],cube[1],cube[2],cube[3],cube[4],
        cube[5],cube[6],cube[7],cube[8],cube[9],
        cube[10],cube[11],cube[12],cube[13],cube[14],
        cube[15])
    like = -0.5 * np.sum(
        (tp_grid[:,4:-4,:] - interped_GCM[:,4:-4,:])**2/20**2
        )
    print('likelihood : ',like)
    return like

n_params = 16
folder_name = 'direct_fit'
file_base = folder_name + '/fixedT-'
if not os.path.isdir(folder_name):
    os.mkdir(folder_name)
pymultinest.run(LogLikelihood,
                Prior,
                n_params,
                n_live_points=400,
                outputfiles_basename=file_base
                )