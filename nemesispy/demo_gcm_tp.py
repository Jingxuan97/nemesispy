"""Investigate the TP profiles from GCMs"""
#!/usr/bin/env python3
#-*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

from nemesispy.common.interpolate_gcm import interp_gcm_X
from nemesispy.data.gcm.process_gcm import (nlon,nlat,xlon,xlat,npv,pv,\
    tmap,h2omap,comap,co2map,ch4map,hemap,h2map,vmrmap,\
    tmap_mod,h2omap_mod,comap_mod,co2map_mod,ch4map_mod,\
    hemap_mod,h2map_mod,vmrmap_mod,phase_grid,\
    kevin_phase_by_wave,kevin_wave_by_phase,\
    pat_phase_by_wave,pat_wave_by_phase,\
    vmrmap_mod_new,tmap_hot)

def read_gcm(lon,lat):
    tp = interp_gcm_X(lon,lat,pv,xlon,xlat,pv,tmap)
    return tp

lon0 = 0
lat0 = 0
tp0 = read_gcm(lon0,lat0)

### Set up figure and axes
fig, ax = plt.subplots(2,1,gridspec_kw={'height_ratios': [1, 2]})
plt.subplots_adjust(bottom=0.4)
plt.subplots_adjust(hspace=0.5)
# globe
#ax[0].set_title('globe')
ax[0].set_ylabel('latitude')
ax[0].set_ylim(-90,90)
ax[0].set_xlabel('longitude')
ax[0].set_xlim(-180,180)
ax[0].grid()
point1, = ax[0].plot(lon0,lat0,marker='x',color='red')
point2, = ax[0].plot(lon0,lat0,marker='x',color='black')
# tp
#ax[1].set_title('globe')
ax[1].set_ylabel('pressure (bar)')
ax[1].set_ylim(200,2e-6)
ax[1].set_yscale('log')
ax[1].set_xlabel('temperature (K)')
ax[1].set_xlim(400,3000)
ax[1].grid()
line_tp1, = ax[1].plot(tp0,pv/1e5,color='red')
line_tp2, = ax[1].plot(tp0,pv/1e5,color='black')

### Slider Axes
axcolor = 'lightgoldenrodyellow'

axlon1 = plt.axes([0.10, 0.05, 0.30, 0.03], facecolor=axcolor)
axlat1 = plt.axes([0.10, 0.15, 0.30, 0.03], facecolor=axcolor)

axlon2 = plt.axes([0.60, 0.05, 0.30, 0.03], facecolor=axcolor)
axlat2 = plt.axes([0.60, 0.15, 0.30, 0.03], facecolor=axcolor)

### Slider values
slon1 = Slider(axlon1, 'lon1', -180, 180, valinit=lon0 , valstep=0.5)
slat1 = Slider(axlat1, 'lat1', -90, 90, valinit=lat0, valstep=0.5)

slon2 = Slider(axlon2, 'lon2', -180, 180, valinit=lon0 , valstep=0.5)
slat2 = Slider(axlat2, 'lat2', -90, 90, valinit=lat0, valstep=0.5)

def update(val):
    lon1 = slon1.val
    lat1 = slat1.val
    tp1 = read_gcm(lon1,lat1)
    point1.set_data(lon1,lat1)
    line_tp1.set_xdata(tp1)

    lon2 = slon2.val
    lat2 = slat2.val
    tp2 = read_gcm(lon2,lat2)
    point2.set_data(lon2,lat2)
    line_tp2.set_xdata(tp2)
    fig.canvas.draw_idle()

slon1.on_changed(update)
slat1.on_changed(update)
slon2.on_changed(update)
slat2.on_changed(update)
plt.show()
