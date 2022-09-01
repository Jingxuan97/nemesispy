# -*- coding: utf-8 -*-
"""
Plot T(Longitude,Pressure) 2D contour plots
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
# Read GCM data
from nemesispy.data.gcm.wasp43b_vivien.process_wasp43b_gcm_vivien import (
    nlon,nlat,xlon,xlat,npv,pv,\
    tmap,h2omap,comap,co2map,ch4map,hemap,h2map,vmrmap,\
    tmap_mod,h2omap_mod,comap_mod,co2map_mod,ch4map_mod,\
    hemap_mod,h2map_mod,vmrmap_mod,phase_grid,\
    kevin_phase_by_wave,kevin_wave_by_phase,\
    pat_phase_by_wave,pat_wave_by_phase)
from nemesispy.common.interpolate_gcm import interp_gcm_X

### T(P,Lon)
NLAYER = 20
P_grid = np.geomspace(20e5,100,NLAYER) # pressure in pa
T_lon_P = np.zeros((NLAYER,nlon))

### MultiPlot Spec
# set up figure : 6 T plots in 2 rows
fig, axs = plt.subplots(nrows=2,ncols=3,sharex=True,sharey=True,
    figsize=[8,4],dpi=600)
fig.supxlabel('Longitude [degree]')
fig.supylabel('Pressure [bar]')

irow = 0
icol = 0
for lat_index, lat in enumerate([0,15,30,45,60,75]):
    for ilon,lon in enumerate(xlon):
        iT = interp_gcm_X(lon,lat,P_grid,gcm_lon=xlon,gcm_lat=xlat,gcm_p=pv,
            X=tmap, substellar_point_longitude_shift=0)
        T_lon_P[:,ilon] = iT

    axs[irow,icol].set_title(r'lat={}'.format(lat))
    x,y = np.meshgrid(xlon,P_grid/1e5,indexing='ij')

    z = T_lon_P.T
    p = axs[irow,icol].contourf(x,y,z,levels=10,vmin=400,vmax=2600,cmap='magma')

    fig.colorbar(p, ax=axs[irow,icol])
    irow = int((lat_index+1)/3)
    if irow == 2:
        irow = 1
    icol = np.mod(icol+1,3)

axs[0,0].invert_yaxis()
axs[0,0].semilogy()
plt.savefig('figures/T_P_lon.pdf',dpi=400)