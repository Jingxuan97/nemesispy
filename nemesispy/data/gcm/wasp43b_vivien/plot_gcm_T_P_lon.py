# -*- coding: utf-8 -*-
"""
Plot T(Longitude,Pressure) 2D contour plots
"""
# Read GCM data
import numpy as np
import matplotlib.pyplot as plt
from nemesispy.data.gcm.process_gcm import (nlon,nlat,xlon,xlat,npv,pv,\
    tmap,h2omap,comap,co2map,ch4map,hemap,h2map,vmrmap,\
    tmap_mod,h2omap_mod,comap_mod,co2map_mod,ch4map_mod,\
    hemap_mod,h2map_mod,vmrmap_mod,phase_grid,\
    kevin_phase_by_wave,kevin_wave_by_phase,\
    pat_phase_by_wave,pat_wave_by_phase,\
    vmrmap_mod_new,tmap_hot)
from nemesispy.common.interpolate_gcm import interp_gcm_X


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

    # x = xlon, y = P_grid

    x,y = np.meshgrid(xlon,P_grid/1e5,indexing='ij')

    z = T_lon_P.T

    # plt.figure(figsize=(7.5,3))
    p = axs[irow,icol].contourf(x,y,z,levels=10,vmin=400,vmax=2600,cmap='magma')
    # axs[irow,icol].contourf(x,y,z,levels=10,cmap='magma')
    # axs[irow,icol].colorbar()

    # pcm =  axs[irow,icol].pcolormesh(np.random.random((20, 20)) * (icol + 1),
    #                     cmap='magma')

    fig.colorbar(p, ax=axs[irow,icol])

    axs[irow,icol].set_title(r'lat={}'.format(lat))


    irow = int((lat_index+1)/3)
    if irow == 2:
        irow = 1
    icol = np.mod(icol+1,3)


# xticks = np.array([-180, -150, -120,  -90,  -60,  -30,    0,   30,
#    60,   90,  120,   150,  180])
#axs[0,0].set_xticks[xticks]

axs[0,0].invert_yaxis()
axs[0,0].semilogy()
# plt.colorbar()
plt.savefig('T_lon_P.pdf',dpi=400)