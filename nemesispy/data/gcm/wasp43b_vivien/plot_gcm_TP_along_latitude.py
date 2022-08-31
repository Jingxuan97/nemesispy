# -*- coding: utf-8 -*-

"""
Plot TP profiles on latitudinal rings using WASP-43b GCM data from Irwin 2020.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
# Read GCM data
from nemesispy.data.gcm.process_gcm import (nlon,nlat,xlon,xlat,npv,pv,\
    tmap,h2omap,comap,co2map,ch4map,hemap,h2map,vmrmap,\
    tmap_mod,h2omap_mod,comap_mod,co2map_mod,ch4map_mod,\
    hemap_mod,h2map_mod,vmrmap_mod,phase_grid,\
    kevin_phase_by_wave,kevin_wave_by_phase,\
    pat_phase_by_wave,pat_wave_by_phase,\
    vmrmap_mod_new,tmap_hot)

from nemesispy.common.interpolate_gcm import interp_gcm_X


# set up figure : 5 TP plots in a row
fig, axs = plt.subplots(nrows=1,ncols=5,sharex=True,sharey=True,
    figsize=[10,5],dpi=600)

# add a big axis, hide frame, hide tick and tick label of the big axis
fig.add_subplot(111,frameon=False)
plt.tick_params(labelcolor='none',which='both',
    top=False,bottom=False,left=False,right=False)

# add overall axes labels to the plot
plt.xlabel('Temperature [K]',size='large')
plt.ylabel('Pressure [bar]',size='large')

# interpolate the
NLAYER = 20
P = np.geomspace(20e5,100,NLAYER) # pressure in bar
lon_grid = np.linspace(-180,180,num=12,endpoint=False)
N_lon = len(lon_grid)

lat = 0
T_equator = np.zeros((NLAYER,N_lon))
color = iter(cm.rainbow(np.linspace(0, 1, N_lon)))
for index,longitude in enumerate(lon_grid):
    iT = interp_gcm_X(longitude,lat,P,gcm_lon=xlon,gcm_lat=xlat,gcm_p=pv,X=tmap,
            substellar_point_longitude_shift=0)
    T_equator[:,index] = iT
    axs[0].plot(iT,P/1e5, lw=0.8, label=int(longitude), color=next(color))
    axs[0].semilogy()
axs[0].set_title(r'lat={}'.format(lat))

lat = 20
T_20 = np.zeros((NLAYER,N_lon))
color = iter(cm.rainbow(np.linspace(0, 1, N_lon)))
for index,longitude in enumerate(lon_grid):
    iT = interp_gcm_X(longitude,lat,P,gcm_lon=xlon,gcm_lat=xlat,gcm_p=pv,X=tmap,
            substellar_point_longitude_shift=0)
    T_equator[:,index] = iT
    axs[1].plot(iT,P/1e5, lw=0.8, label=int(longitude), color=next(color))
axs[1].set_title(r'lat={}'.format(lat))

lat = 40
T_40 = np.zeros((NLAYER,N_lon))
color = iter(cm.rainbow(np.linspace(0, 1, N_lon)))
for index,longitude in enumerate(lon_grid):
    iT = interp_gcm_X(longitude,lat,P,gcm_lon=xlon,gcm_lat=xlat,gcm_p=pv,X=tmap,
            substellar_point_longitude_shift=0)
    T_equator[:,index] = iT
    axs[2].plot(iT,P/1e5, lw=0.8, label=int(longitude), color=next(color))
axs[2].set_title(r'lat={}'.format(lat))

lat = 60
T_60 = np.zeros((NLAYER,N_lon))
color = iter(cm.rainbow(np.linspace(0, 1, N_lon)))
for index,longitude in enumerate(lon_grid):
    iT = interp_gcm_X(longitude,lat,P,gcm_lon=xlon,gcm_lat=xlat,gcm_p=pv,X=tmap,
            substellar_point_longitude_shift=0)
    T_equator[:,index] = iT
    axs[3].plot(iT,P/1e5, lw=0.8, label=int(longitude), color=next(color))
axs[3].set_title(r'lat={}'.format(lat))

lat = 80
T_80 = np.zeros((NLAYER,N_lon))
color = iter(cm.rainbow(np.linspace(0, 1, N_lon)))
for index,longitude in enumerate(lon_grid):
    iT = interp_gcm_X(longitude,lat,P,gcm_lon=xlon,gcm_lat=xlat,gcm_p=pv,X=tmap,
            substellar_point_longitude_shift=0)
    T_equator[:,index] = iT
    axs[4].plot(iT,P/1e5, lw=0.8, label=int(longitude), color=next(color))
axs[4].set_title(r'lat={}'.format(lat))

# Since all axes are shared, edit all subplots in one go
axs[4].invert_yaxis()
axs[4].legend(loc='upper right',fontsize='xx-small',title='longitude')
axs[4].set_xlim((500,2500))

plt.savefig('plots/TPs_on_lat_band.pdf')


# plt.minorticks_on()
# plt.semilogy()
# plt.gca().invert_yaxis()
# plt.xlim((500,2500))
# plt.title('latitude={}'.format(lat))
# plt.tick_params(length=10,width=1,labelsize='x-large',which='major')
# plt.tight_layout()
# plt.show()