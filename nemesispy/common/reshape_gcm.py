# -*- coding: utf-8 -*-
import numpy as np
import scipy.interpolate as interpolate
# Read GCM data
from nemesispy.data.gcm.process_gcm import (nlon,nlat,xlon,xlat,npv,pv,\
    tmap,h2omap,comap,co2map,ch4map,hemap,h2map,vmrmap,\
    tmap_mod,h2omap_mod,comap_mod,co2map_mod,ch4map_mod,\
    hemap_mod,h2map_mod,vmrmap_mod,phase_grid,\
    kevin_phase_by_wave,kevin_wave_by_phase,\
    pat_phase_by_wave,pat_wave_by_phase,\
    vmrmap_mod_new,tmap_hot)

NLAYER = 20
P_range = np.geomspace(20*1e5,1e-3*1e5,NLAYER)
T_reshaped = np.zeros((nlon*nlat,NLAYER))
for ilon in range(nlon):
    for ilat in range(nlat):
        f = interpolate.interp1d(pv,tmap_mod[ilon,ilat,:])
        T = f(P_range)
        T_reshaped[ilon*32+ilat,:] = T

np.savetxt('AAgcm_reshaped.txt',T_reshaped,delimiter=',')

        