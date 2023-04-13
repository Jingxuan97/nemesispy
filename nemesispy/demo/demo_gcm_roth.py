"""Investigate the TP profiles from GCMs"""
#!/usr/bin/env python3
#-*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

from nemesispy.common.interpolate_gcm import interp_gcm_X
from nemesispy.data.gcm.format_roth import roth_lon_grid, roth_lat_grid,\
    roth_p_grid, read_gcm_PT_roth, gen_file_name

Teq = 1000
LogMet = 0.0
LogG = 0.8
Mstar = 0.8

def change_map(Teq,LogMet,LogG):
    file_name = gen_file_name(Teq,LogMet,LogG,Mstar)
    gcm_tmap = read_gcm_PT_roth('./data/gcm/PTprofiles_hr_all/'+file_name)
    return gcm_tmap

gcm_tmap = change_map(Teq,LogMet,LogG)

def read_gcm(lon,lat,gcm_tmap):
    tp = interp_gcm_X(lon,lat,roth_p_grid,
        roth_lon_grid,roth_lat_grid,roth_p_grid,gcm_tmap)
    return tp

lon0 = 0
lat0 = 0
tp0 = read_gcm(lon0,lat0,gcm_tmap)

### Set up figure and axes
fig, ax = plt.subplots(2,1,gridspec_kw={'height_ratios': [1, 3]})
plt.subplots_adjust(bottom=0.4)
plt.subplots_adjust(hspace=0.5)

#ax[0].set_title('globe')
ax[0].set_ylabel('latitude')
ax[0].set_ylim(-90,90)
ax[0].set_xlabel('longitude')
ax[0].set_xlim(-180,180)
ax[0].grid()
point1, = ax[0].plot(lon0,lat0,marker='x',color='red')
point2, = ax[0].plot(lon0,lat0,marker='x',color='black')
point3, = ax[0].plot(lon0,lat0,marker='x',color='blue')

# tp
#ax[1].set_title('globe')
ax[1].set_ylabel('pressure (bar)')
ax[1].set_ylim(200,2e-6)
ax[1].set_yscale('log')
ax[1].set_xlabel('temperature (K)')
ax[1].set_xlim(400,3000)
ax[1].grid()
line_tp1, = ax[1].plot(tp0,roth_p_grid,color='red')
line_tp2, = ax[1].plot(tp0,roth_p_grid,color='black')
line_tp3, = ax[1].plot(tp0,roth_p_grid,color='blue')

### Slider Axes
axcolor = 'lightgoldenrodyellow'

axTeq = plt.axes([0.10, 0.30, 0.30, 0.03], facecolor=axcolor)
axLogG = plt.axes([0.10, 0.25, 0.30, 0.03], facecolor=axcolor)
axLogMet = plt.axes([0.10, 0.20, 0.30, 0.03], facecolor=axcolor)

axlon1 = plt.axes([0.60, 0.30, 0.30, 0.03], facecolor=axcolor)
axlat1 = plt.axes([0.60, 0.25, 0.30, 0.03], facecolor=axcolor)

axlon2 = plt.axes([0.60, 0.20, 0.30, 0.03], facecolor=axcolor)
axlat2 = plt.axes([0.60, 0.15, 0.30, 0.03], facecolor=axcolor)

axlon3 = plt.axes([0.60, 0.10, 0.30, 0.03], facecolor=axcolor)
axlat3 = plt.axes([0.60, 0.05, 0.30, 0.03], facecolor=axcolor)

### Slider values
sTeq = Slider(axTeq, 'Teq', 1000, 2400, valinit=1000, valstep=200)
sLogG = Slider(axLogG, 'LogG', 0.7, 1.9, valinit=0.8, valstep=[0.8,1.3,1.8])
sLogMet = Slider(axLogMet, 'LogMet', 0.0, 1.6, valinit=0.0, valstep=[0.0,0.7,1.5])

slon1 = Slider(axlon1, 'lon1', -180, 180, valinit=lon0 , valstep=0.5)
slat1 = Slider(axlat1, 'lat1', -90, 90, valinit=lat0, valstep=0.5)

slon2 = Slider(axlon2, 'lon2', -180, 180, valinit=lon0 , valstep=0.5)
slat2 = Slider(axlat2, 'lat2', -90, 90, valinit=lat0, valstep=0.5)

slon3 = Slider(axlon3, 'lon3', -180, 180, valinit=lon0 , valstep=0.5)
slat3 = Slider(axlat3, 'lat3', -90, 90, valinit=lat0, valstep=0.5)

class info:
    def __init__(self):
        self.start = 0
        self.Teq = None
        self.LogG = None
        self.LogMet = None
        self.gcm_tmap = None
        pass

info = info()
info.Teq = Teq
info.gcm_tmap = gcm_tmap
info.LogG = LogG
info.LogMet = LogMet

def update(val):
    if info.start == 0:
        info.start = 1
        gcm_tmap = info.gcm_tmap
    else:
        Teq = sTeq.val
        LogG = sLogG.val
        LogMet = sLogMet.val
        if info.Teq != Teq or info.LogG != LogG or info.LogMet != LogMet:
            info.gcm_tmap = change_map(Teq,LogMet,LogG)
            info.Teq = Teq
            info.LogG = LogG
            info.LogMet = LogMet
        gcm_tmap = info.gcm_tmap
    # point 1
    lon1 = slon1.val
    lat1 = slat1.val
    tp1 = read_gcm(lon1,lat1,gcm_tmap)
    point1.set_data(lon1,lat1)
    line_tp1.set_xdata(tp1)
    # point 2
    lon2 = slon2.val
    lat2 = slat2.val
    tp2 = read_gcm(lon2,lat2,gcm_tmap)
    point2.set_data(lon2,lat2)
    line_tp2.set_xdata(tp2)
    # point 3
    lon3 = slon3.val
    lat3 = slat3.val
    tp3 = read_gcm(lon3,lat3,gcm_tmap)
    point3.set_data(lon3,lat3)
    line_tp3.set_xdata(tp3)
    fig.canvas.draw_idle()

sTeq.on_changed(update)
sLogG.on_changed(update)
sLogMet.on_changed(update)

slon1.on_changed(update)
slat1.on_changed(update)
slon2.on_changed(update)
slat2.on_changed(update)
slon3.on_changed(update)
slat3.on_changed(update)

plt.show()
