#!/usr/bin/env python3
#-*- coding: utf-8 -*-
import numpy as np
from nemesispy.models.backup_tmaps.tmap_3tp import tmap_3tp
from nemesispy.retrieval.plot_tmap import plot_tmap_contour
from nemesispy.common.interpolate_gcm import interp_gcm_X
from nemesispy.common.utils import mkdir

# Read GCM data
from nemesispy.common.constants import G
from nemesispy.data.gcm.wasp43b_vivien.process_wasp43b_gcm_vivien import (
    nlon,nlat,xlon,xlat,npv,pv,\
    tmap,h2omap,comap,co2map,ch4map,hemap,h2map,vmrmap,\
    tmap_mod,h2omap_mod,comap_mod,co2map_mod,ch4map_mod,\
    hemap_mod,h2map_mod,vmrmap_mod,phase_grid,\
    kevin_phase_by_wave,kevin_wave_by_phase,\
    pat_phase_by_wave,pat_wave_by_phase)

folder = 'visualise_tmap'
mkdir(folder)

### Grid data
NLAYER = 20
# 20 pressures
P_range = np.geomspace(20*1e5,1e-3*1e5,20)
# 359 longitudes
global_lon_grid = np.linspace(-179,179,100)
# 90 latitudes
global_lat_grid = np.linspace(0,89,90)

log_kappa_hot = -2.0
log_gamma_hot = -0.5
log_f_hot = - 0.5
T_int_hot = 200

log_kappa_day = -2.2
log_gamma_day = -1
log_f_day = - 1
T_int_day = 200

log_kappa_night = -4
log_gamma_night = 0
log_f_night = -2
T_int_night = 200

scale = .9
phase_offset = 20

west_fraction = 0.5
east_fraction = 0.5

### Reference Planet Input: WASP 43b
T_star = 4520 # star temperature in K
R_star = 463892759.99999994 # m, 0.6668 * R_SUN
SMA = 2243970000.0 # m, 0.015*AU
M_plt = 3.8951064000000004e+27 # kg
R_plt = 74065.70 * 1e3 # m
T_irr = T_star * (R_star/SMA)**0.5
T_eq = T_irr/2**0.5
g = G*M_plt/R_plt**2


tp_grid = tmap_3tp(P_range,global_lon_grid,global_lat_grid,
    g,T_eq,
    scale, phase_offset,  west_fraction, east_fraction,
    log_kappa_day, log_gamma_day, log_f_day, T_int_day,
    log_kappa_night, log_gamma_night, log_f_night, T_int_night,
    log_kappa_hot, log_gamma_hot, log_f_hot, T_int_hot)

# # gcm interpolated to model grid
# gcm_truth = np.zeros((len(global_lon_grid),len(global_lat_grid),20))
# for ilon,mlon in enumerate(global_lon_grid):
#     for ilat,mlat in enumerate(global_lat_grid):
#         gcm_truth[ilon,ilat]\
#             = interp_gcm_X(mlon,mlat,P_range,xlon,xlat,pv,tmap)

# ### PLOTTING
# plot mean contours
for iP,P in enumerate(P_range):
    plot_tmap_contour(P,tp_grid,global_lon_grid,global_lat_grid,
        P_range,foreshorten=False, grid_points=False,
        title='{:.3e} bar'.format(P/1e5),
        figname='{}/contour_{}.png'.format(folder,iP),
        ylims=(0,90))
    # print(tp_grid[:,:,iP])

# # plot MAP difference with gcm
# diff =  tp_grid - gcm_truth
# for iP,P in enumerate(P_range):
#     plot_tmap_contour(P,diff,global_lon_grid,global_lat_grid,P_range,grid_points=False,
#         title='{:.3e} bar'.format(P/1e5),
#         figname='{}/diff_{}.png'.format(folder,iP),
#         T_range=(-500,500),
#         cmap='bwr',
#         ylims=(0,90))