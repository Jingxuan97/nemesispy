#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import os
import pymultinest
from nemesispy.data.helper import lowres_file_paths, cia_file_path
from nemesispy.common.constants import G
from nemesispy.radtran.forward_model import ForwardModel
from nemesispy.models.TP_profiles import TP_Guillot
from nemesispy.data.gcm.process_gcm import (nlon,nlat,xlon,xlat,npv,pv,\
    tmap,h2omap,comap,co2map,ch4map,hemap,h2map,vmrmap,\
    tmap_mod,h2omap_mod,comap_mod,co2map_mod,ch4map_mod,\
    hemap_mod,h2map_mod,vmrmap_mod,phase_grid,\
    kevin_phase_by_wave,kevin_wave_by_phase,\
    pat_phase_by_wave,pat_wave_by_phase,\
    vmrmap_mod_new,tmap_hot)

NLAYER = 20
P_range = np.geomspace(20*1e5,1e-3*1e5,20)
global_lon_grid = np.array(
    [-175., -170., -165., -160., -155., -150., -145., -140., -135.,
    -130., -125., -120., -115., -110., -105., -100.,  -95.,  -90.,
    -85.,  -80.,  -75.,  -70.,  -65.,  -60.,  -55.,  -50.,  -45.,
    -40.,  -35.,  -30.,  -25.,  -20.,  -15.,  -10.,   -5.,    0.,
        5.,   10.,   15.,   20.,   25.,   30.,   35.,   40.,   45.,
        50.,   55.,   60.,   65.,   70.,   75.,   80.,   85.,   90.,
        95.,  100.,  105.,  110.,  115.,  120.,  125.,  130.,  135.,
    140.,  145.,  150.,  155.,  160.,  165.,  170.,  175.]) # 71
global_lat_grid = np.array(
    [ 5., 10., 15., 20., 25., 30., 35., 40., 45., 50., 55., 60., 65.,
    70., 75., 80., 85.]) # 17
err = np.array([[6.60e-05, 6.70e-05, 7.10e-05, 6.30e-05, 6.90e-05, 6.60e-05,
        5.90e-05, 4.50e-05, 6.10e-05, 6.50e-05, 7.10e-05, 6.60e-05,
        6.80e-05, 6.30e-05, 7.00e-05],
       [6.10e-05, 6.10e-05, 6.50e-05, 5.80e-05, 6.40e-05, 6.10e-05,
        5.30e-05, 3.90e-05, 5.50e-05, 5.90e-05, 6.50e-05, 6.10e-05,
        6.30e-05, 5.70e-05, 6.40e-05],
       [5.80e-05, 5.90e-05, 6.20e-05, 5.50e-05, 6.10e-05, 5.80e-05,
        5.20e-05, 3.80e-05, 5.30e-05, 5.70e-05, 6.30e-05, 5.80e-05,
        6.00e-05, 5.50e-05, 6.20e-05],
       [5.60e-05, 5.60e-05, 6.00e-05, 5.30e-05, 5.90e-05, 5.60e-05,
        4.90e-05, 3.60e-05, 5.10e-05, 5.50e-05, 6.10e-05, 5.60e-05,
        5.80e-05, 5.20e-05, 5.90e-05],
       [5.70e-05, 5.70e-05, 6.10e-05, 5.40e-05, 5.90e-05, 5.70e-05,
        5.00e-05, 3.70e-05, 5.20e-05, 5.50e-05, 6.10e-05, 5.70e-05,
        5.80e-05, 5.30e-05, 6.00e-05],
       [5.30e-05, 5.30e-05, 5.70e-05, 5.00e-05, 5.60e-05, 5.30e-05,
        4.70e-05, 3.30e-05, 4.80e-05, 5.20e-05, 5.70e-05, 5.30e-05,
        5.50e-05, 5.00e-05, 5.60e-05],
       [5.50e-05, 5.50e-05, 5.80e-05, 5.20e-05, 5.70e-05, 5.50e-05,
        4.80e-05, 3.40e-05, 5.00e-05, 5.30e-05, 5.90e-05, 5.50e-05,
        5.60e-05, 5.10e-05, 5.80e-05],
       [5.20e-05, 5.20e-05, 5.50e-05, 4.80e-05, 5.40e-05, 5.10e-05,
        4.40e-05, 3.00e-05, 4.60e-05, 5.00e-05, 5.60e-05, 5.10e-05,
        5.30e-05, 4.80e-05, 5.50e-05],
       [5.60e-05, 5.60e-05, 6.00e-05, 5.30e-05, 5.90e-05, 5.60e-05,
        4.90e-05, 3.60e-05, 5.10e-05, 5.50e-05, 6.00e-05, 5.60e-05,
        5.70e-05, 5.30e-05, 5.90e-05],
       [5.60e-05, 5.60e-05, 5.90e-05, 5.30e-05, 5.80e-05, 5.50e-05,
        4.90e-05, 3.60e-05, 5.10e-05, 5.40e-05, 5.90e-05, 5.50e-05,
        5.70e-05, 5.20e-05, 5.80e-05],
       [5.60e-05, 5.60e-05, 5.90e-05, 5.20e-05, 5.80e-05, 5.50e-05,
        4.80e-05, 3.30e-05, 5.00e-05, 5.40e-05, 6.00e-05, 5.50e-05,
        5.70e-05, 5.20e-05, 5.90e-05],
       [5.50e-05, 5.50e-05, 5.80e-05, 5.20e-05, 5.70e-05, 5.50e-05,
        4.80e-05, 3.50e-05, 5.00e-05, 5.30e-05, 5.90e-05, 5.50e-05,
        5.60e-05, 5.10e-05, 5.80e-05],
       [5.80e-05, 5.80e-05, 6.20e-05, 5.50e-05, 6.10e-05, 5.80e-05,
        5.10e-05, 3.60e-05, 5.20e-05, 5.60e-05, 6.20e-05, 5.80e-05,
        5.90e-05, 5.40e-05, 6.10e-05],
       [5.80e-05, 5.80e-05, 6.10e-05, 5.40e-05, 6.00e-05, 5.70e-05,
        5.10e-05, 3.70e-05, 5.20e-05, 5.60e-05, 6.20e-05, 5.70e-05,
        5.90e-05, 5.40e-05, 6.10e-05],
       [6.30e-05, 6.30e-05, 6.70e-05, 6.00e-05, 6.60e-05, 6.30e-05,
        5.60e-05, 4.20e-05, 5.80e-05, 6.10e-05, 6.70e-05, 6.30e-05,
        6.40e-05, 5.90e-05, 6.60e-05],
       [1.03e-04, 1.05e-04, 1.03e-04, 1.03e-04, 1.00e-04, 7.90e-05,
        7.70e-05, 6.00e-05, 8.00e-05, 1.21e-04, 1.03e-04, 1.03e-04,
        1.03e-04, 1.03e-04, 1.03e-04],
       [1.33e-04, 1.33e-04, 1.36e-04, 1.34e-04, 1.19e-04, 1.03e-04,
        1.03e-04, 8.40e-05, 1.03e-04, 1.09e-04, 1.33e-04, 1.34e-04,
        1.34e-04, 1.33e-04, 1.33e-04]])
def gen(x,ck0,ck1,ck2,sk1,sk2,cg0,cg1,cg2,sg1,sg2,cf0,cf1,cf2,sf1,sf2,
    ct0,ct1,ct2,st1,st2):
    """
    Generate log(2-Stream Guillot parameters) from Fourier coefficients
    """
    y = x/180*np.pi
    log_kappa = ck0 + ck1 * np.cos(y) + ck2 * np.cos(2*y)\
        + sk1 * np.sin(y) + sk2 * np.sin(2*y)
    log_gamma = cg0 + cg1 * np.cos(y) + cg2 * np.cos(2*y)\
        + sg1 * np.sin(y) + sg2 * np.sin(2*y)
    log_f = cf0 + cf1 * np.cos(y) + cf2 * np.cos(2*y)\
        + sf1 * np.sin(y) + sf2 * np.sin(2*y)
    log_T_int = ct0 + ct1 * np.cos(y) + ct2 * np.cos(2*y)\
        + st1 * np.sin(y) + st2 * np.sin(2*y)
    return log_kappa, log_gamma, log_f, log_T_int

def tmap1(g_plt, T_eq, log_kappa, log_gamma, log_f, log_T_int):
    """
    Generate tmap
    """
    # start with fixed grid
    P_grid = P_range
    lon_grid = global_lon_grid
    lat_grid = global_lat_grid

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

    TP_mean = 0.5 * (tp_grid[17,0,:] + tp_grid[-18,0,:])
    for ilat,lat in enumerate(lat_grid):
        for ilon in range(nlon):
            tp_grid[ilon,ilat,:] = TP_mean \
                + (tp_grid[ilon,0,:]- TP_mean) * np.cos(lat/180*np.pi)**0.25
    return tp_grid

def gen_tmap1(g_plt,T_eq,ck0,ck1,ck2,sk1,sk2,cg0,cg1,cg2,sg1,sg2,cf0,cf1,
    cf2,sf1,sf2,ct0,ct1,ct2,st1,st2,):
    lon_grid = global_lon_grid
    log_kappa, log_gamma, log_f, log_T_int \
        = gen(lon_grid,ck0,ck1,ck2,sk1,sk2,cg0,cg1,cg2,sg1,sg2,cf0,cf1,cf2,
            sf1,sf2,ct0,ct1,ct2,st1,st2)
    tp_grid = tmap1(g_plt, T_eq, log_kappa, log_gamma, log_f, log_T_int)
    return tp_grid

def gen_vmrmap1(h2o,co2,co,ch4,nlon,nlat,npress):
    vmr_grid = np.ones((nlon,nlat,npress,6))
    vmr_grid[:,:,:,0] *= 10**h2o
    vmr_grid[:,:,:,1] *= 10**co2
    vmr_grid[:,:,:,2] *= 10**co
    vmr_grid[:,:,:,3] *= 10**ch4
    vmr_grid[:,:,:,4] *= 0.16 * (1-10**h2o-10**co2-10**co-10**ch4)
    vmr_grid[:,:,:,5] *= 0.84 * (1-10**h2o-10**co2-10**co-10**ch4)
    return vmr_grid

### test
# log1 = np.linspace(-2,-1,71)
# log2 = np.linspace(-2,1,71)
# log3 = np.linspace(-2,-0.2,71)
# log4 = np.linspace(1,3,71)
# test = map1(P_grid=P_range,g_plt=g,T_eq=T_eq,
#     log_kappa=log1,log_gamma=log2,log_f=log3,log_T_int=log4)

### Reference Planet Input: WASP 43b
T_star = 4520 # star temperature in K
R_star = 463892759.99999994 # m, 0.6668 * R_SUN
SMA = 2243970000.0 # m, 0.015*AU
M_plt = 3.8951064000000004e+27 # kg
R_plt = 74065.70 * 1e3 # m
T_irr = T_star * (R_star/SMA)**0.5
T_eq = T_irr/2**0.5
g = G*M_plt/R_plt**2
gas_id = np.array([  1, 2,  5,  6, 40, 39])
iso_id = np.array([0, 0, 0, 0, 0, 0])
nmu = 3

### Model parameters (focus to pressure range where Transmission WF peaks)
wave_grid = np.array(
        [1.1425, 1.1775, 1.2125, 1.2475, 1.2825, 1.3175, 1.3525, 1.3875,
       1.4225, 1.4575, 1.4925, 1.5275, 1.5625, 1.5975, 1.6325, 3.6   ,
       4.5   ])
phase_grid = np.array(
        [ 22.5,  45. ,  67.5,  90. , 112.5, 135. , 157.5, 180. , 202.5,
       225. , 247.5, 270. , 292.5, 315. , 337.5])
nwave = len(wave_grid)
nphase = len(phase_grid)
wasp43_spec = np.array([3.341320e+25, 3.215455e+25, 3.101460e+25, 2.987110e+25,
       2.843440e+25, 2.738320e+25, 2.679875e+25, 2.598525e+25,
       2.505735e+25, 2.452230e+25, 2.391140e+25, 2.345905e+25,
       2.283720e+25, 2.203690e+25, 2.136015e+25, 1.234010e+24,
       4.422200e+23])

### Set up forward model
FM = ForwardModel()
FM.set_planet_model(M_plt=M_plt,R_plt=R_plt,gas_id_list=gas_id,iso_id_list=iso_id,
    NLAYER=NLAYER)
FM.set_opacity_data(kta_file_paths=lowres_file_paths, cia_file_path=cia_file_path)

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

    # log_T_int
    cube[15] = 2 + (4 - (2)) * cube[15]
    cube[16] = -1 + (1 - (-1)) * cube[16]
    cube[17] = -1 + (1 - (-1)) * cube[17]
    cube[18] = -1 + (1 - (-1)) * cube[18]
    cube[19] = -1 + (1 - (-1)) * cube[19]

    # log VMR
    cube[20] = -8 + (-2 - (-8)) * cube[20]
    cube[21] = -8 + (-2 - (-8)) * cube[21]
    cube[22] = -8 + (-2 - (-8)) * cube[22]
    cube[23] = -8 + (-2 - (-8)) * cube[23]

def LogLikelihood(cube,ndim,nparams):
    tp_grid = gen_tmap1(g,T_eq,
        cube[0],cube[1],cube[2],cube[3],cube[4],
        cube[5],cube[6],cube[7],cube[8],cube[9],
        cube[10],cube[11],cube[12],cube[13],cube[14],
        cube[15],cube[16],cube[17],cube[18],cube[19])

    vmr_grid = gen_vmrmap1(cube[20],cube[21],cube[22],cube[23],
        nlon=len(global_lon_grid), nlat=len(global_lat_grid),
        npress=len(P_range))

    chi = 0
    for iphase, phase in enumerate(phase_grid):
        one_phase =  FM.calc_disc_spectrum(phase=phase, nmu=nmu, P_model=P_range,
            global_model_P_grid=P_range, global_T_model=tp_grid,
            global_VMR_model=vmr_grid, mod_lon=global_lon_grid,
            mod_lat=global_lat_grid, solspec=wasp43_spec)

        chi += np.sum(
            (pat_phase_by_wave[iphase,:] - one_phase)**2/err[:,iphase]**2 )
    plt.plot()
    like = -0.5*chi
    print(like)
    #plt.plot(wave_grid,pat_phase_by_wave[-1,:],color='k')
    #plt.plot(wave_grid,one_phase,color='r')
    #plt.show()
    return like

n_params = 24
if not os.path.isdir('chains1'):
    os.mkdir('chains1')
pymultinest.run(LogLikelihood,
                Prior,
                n_params,
                n_live_points=400,
                outputfiles_basename='chains1/full-'
                )