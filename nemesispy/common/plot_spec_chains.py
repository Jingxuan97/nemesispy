#!/usr/local/bin/python3
# -*- coding: utf-8 -*-
"""
Plot simulation from retrieved 1D TP profiles
"""
import numpy as np
import scipy as sp
from nemesispy.data.gcm.process_gcm \
    import (nlon,nlat,xlon,xlat,npv,pv,\
    tmap,h2omap,comap,co2map,ch4map,hemap,h2map,vmrmap,\
    tmap_mod,h2omap_mod,comap_mod,co2map_mod,ch4map_mod,hemap_mod,h2map_mod,
    vmrmap_mod,phase_grid,\
    kevin_phase_by_wave,kevin_wave_by_phase,pat_phase_by_wave,pat_wave_by_phase)

from nemesispy.radtran.forward_model import ForwardModel
lowres_files = ['/Users/jingxuanyang/ktables/h2owasp43.kta',
'/Users/jingxuanyang/ktables/cowasp43.kta',
'/Users/jingxuanyang/ktables/co2wasp43.kta',
'/Users/jingxuanyang/ktables/ch4wasp43.kta']

cia_file_path = '/Users/jingxuanyang/Desktop/Workspace/' \
    + 'nemesispy2022/nemesispy/data/cia/exocia_hitran12_200-3800K.tab'

################################################################################
### Wavelengths grid and orbital phase grid
wave_grid = np.array([1.1425, 1.1775, 1.2125, 1.2475, 1.2825, 1.3175, 1.3525,
    1.3875, 1.4225, 1.4575, 1.4925, 1.5275, 1.5625, 1.5975, 1.6325, 3.6   ,
    4.5   ])
phase_grid = np.array([ 22.5,  45. ,  67.5,  90. , 112.5, 135. , 157.5, 180. , 202.5,
       225. , 247.5, 270. , 292.5, 315. , 337.5])
nwave = len(wave_grid)
nphase = len(phase_grid)
wasp43_spec = np.array([3.341320e+25, 3.215455e+25, 3.101460e+25, 2.987110e+25,
       2.843440e+25, 2.738320e+25, 2.679875e+25, 2.598525e+25,
       2.505735e+25, 2.452230e+25, 2.391140e+25, 2.345905e+25,
       2.283720e+25, 2.203690e+25, 2.136015e+25, 1.234010e+24,
       4.422200e+23])
retrieved_TP_phase_by_wave = np.zeros((nphase,nwave))
retrieved_TP_wave_by_phase = np.zeros((nwave,nphase))

### Reference Planet Input
M_plt = 3.8951064000000004e+27 # kg
R_plt = 74065.70 * 1e3 # m
gas_id = np.array([  1, 2,  5,  6, 40, 39])
iso_id = np.array([0, 0, 0, 0, 0, 0])
NLAYER = 20
nmu = 5
P_range = np.geomspace(20*1e5,1e-3*1e5,NLAYER)

### Create TP map
T_profiles = np.loadtxt('retrieved_TP.txt')
retrieved_T_map = np.zeros((nlon,nlat,npv))
for ilon in range(nlon):
    for ilat in range(nlat):
        TP = T_profiles[ilon*31+ilat,]
        f = sp.interpolate.interp1d(P_range,TP,fill_value="extrapolate")
        retrieved_T_map[ilon,ilat,:] = f(pv)

### Set up forward model
FM = ForwardModel()
FM.set_planet_model(M_plt=M_plt,R_plt=R_plt,gas_id_list=gas_id,iso_id_list=iso_id,
    NLAYER=NLAYER)
FM.set_opacity_data(kta_file_paths=lowres_files, cia_file_path=cia_file_path)

for iphase, phase in enumerate(phase_grid):
    one_phase =  FM.calc_disc_spectrum(phase=phase, nmu=nmu, P_model=P_range,
        global_model_P_grid=pv, global_T_model=retrieved_T_map,
        global_VMR_model=vmrmap_mod, mod_lon=xlon, mod_lat=xlat,
        solspec=wasp43_spec)
    retrieved_TP_phase_by_wave[iphase,:] = one_phase

for iwave in range(len(wave_grid)):
    for iphase in range(len(phase_grid)):
        retrieved_TP_wave_by_phase[iwave,iphase] \
            = retrieved_TP_phase_by_wave[iphase,iwave]
