# -*- coding: utf-8 -*-
"""
Plot T(lon) for given latitude and longitude
Plot TP profiles on latitudinal rings using WASP-43b GCM data from Irwin 2020.
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm

# Read GCM data
from nemesispy.data.gcm.wasp43b_vivien.process_wasp43b_gcm_vivien import (
    nlon,nlat,xlon,xlat,npv,pv,\
    tmap,h2omap,comap,co2map,ch4map,hemap,h2map,vmrmap,\
    tmap_mod,h2omap_mod,comap_mod,co2map_mod,ch4map_mod,\
    hemap_mod,h2map_mod,vmrmap_mod,phase_grid,\
    kevin_phase_by_wave,kevin_wave_by_phase,\
    pat_phase_by_wave,pat_wave_by_phase)
from nemesispy.common.interpolate_gcm import interp_gcm_X

# interpolate the temperature to the right pressure level
NLAYER = 20
P_grid = np.geomspace(20e5,100,NLAYER) # pressure in bar
N_P = len(P_grid)
lon_grid = np.linspace(-180,180,100)
N_lon = len(lon_grid)

lat_grid = np.linspace(0,90,18,endpoint=False)
lat_grid = np.array([0,15,30,45,60,75])

### MultiPlot Spec
# set up figure : 6 T plots in 2 rows
fig, axs = plt.subplots(nrows=2,ncols=3,sharex=True,sharey=True,
    figsize=[8,4],dpi=600)

irow = 0
icol = 0
for lat_index, lat in enumerate(lat_grid):
    T_lat = np.zeros((N_P,N_lon))
    color = iter(cm.rainbow(np.linspace(0, 1, N_P+2)))
    for lon_index, longitude in enumerate(lon_grid):
        iT = interp_gcm_X(longitude,lat,P_grid,gcm_lon=xlon,gcm_lat=xlat,
            gcm_p=pv,X=tmap, substellar_point_longitude_shift=0)
        T_lat[:,lon_index] = iT

    for index, pressure in enumerate(P_grid):
        axs[irow,icol].plot(lon_grid,T_lat[-index-1,:],
            label='{:.1e}'.format(P_grid[-index-1]/1e5),
            color=next(color),linewidth=0.5)

    axs[irow,icol].set_title(r'lat={}'.format(lat))
    irow = int((lat_index+1)/3)
    if irow == 2:
        irow = 1
    icol = np.mod(icol+1,3)
    handles, labels = axs[irow,icol].get_legend_handles_labels()

axs[1,2].set_xlim((-180,180))
axs[1,2].set_ylim((400,2500))

fig.supxlabel('Longitude [degree]')
fig.supylabel('Temperature [K]')
fig.legend(handles, labels, loc='center right',
    ncol=1,fontsize='x-small',title='Pressure\n [bar]')
plt.savefig('figures/T_lon.pdf',dpi=400)
