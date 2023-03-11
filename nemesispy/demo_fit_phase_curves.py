#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

from nemesispy.radtran.forward_model import ForwardModel
from nemesispy.common.constants import G

from nemesispy.data.helper import lowres_file_paths, cia_file_path
from nemesispy.retrieval.tmap_deep import gen_tmap_deep, gen_vmrmap1

from nemesispy.data.gcm.process_gcm import (nlon,nlat,xlon,xlat,npv,pv,\
    tmap,h2omap,comap,co2map,ch4map,hemap,h2map,vmrmap,\
    tmap_mod,h2omap_mod,comap_mod,co2map_mod,ch4map_mod,\
    hemap_mod,h2map_mod,vmrmap_mod,phase_grid,\
    kevin_phase_by_wave,kevin_wave_by_phase,\
    pat_phase_by_wave,pat_wave_by_phase,\
    vmrmap_mod_new,tmap_hot)

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

### Define 2.5D scheme interpolation grid
# 71 longitudes
global_lon_grid = np.array(
    [-175., -170., -165., -160., -155.,
     -150., -145., -140., -135., -130.,
     -125., -120., -115., -110., -105.,
     -100.,  -95.,  -90., -85.,  -80.,
     -75.,  -70.,  -65.,  -60.,  -55.,
     -50.,  -45., -40.,  -35.,  -30.,
     -25.,  -20.,  -15.,  -10.,   -5.,
     0.,    5.,   10.,   15.,   20.,
     25.,   30.,   35.,   40.,   45.,
     50.,   55.,   60.,   65.,   70.,
     75.,   80.,   85.,   90.,   95.,
     100.,  105.,  110.,  115.,  120.,
     125.,  130.,  135.,  140.,  145.,
     150.,  155.,  160.,  165.,  170.,
     175.])

# 36 latitudes
global_lat_grid = np.array(
    [ 0. ,  2.5,  5. ,  7.5, 10. , 12.5, 15. , 17.5, 20. , 22.5, 25. ,
       27.5, 30. , 32.5, 35. , 37.5, 40. , 42.5, 45. , 47.5, 50. , 52.5,
       55. , 57.5, 60. , 62.5, 65. , 67.5, 70. , 72.5, 75. , 77.5, 80. ,
       82.5, 85. , 87.5])

### Reference Planet Input: WASP 43b
T_star = 4520 # star temperature in K
R_star = 463892759.99999994 # m, 0.6668 * R_SUN
SMA = 2243970000.0 # m, 0.015*AU
M_plt = 3.8951064000000004e+27 # kg
R_plt = 74065.70 * 1e3 # m
T_irr = T_star * (R_star/SMA)**0.5 # 2055 K
T_eq = T_irr/2**0.5 # 1453 K
g = G*M_plt/R_plt**2 # 47.39 ms-2
gas_id = np.array([  1, 2,  5,  6, 40, 39])
iso_id = np.array([0, 0, 0, 0, 0, 0])

# 20 pressures, from 20 bar to 1 milibar
nmu = 3
NLAYER = 20
P_range = np.geomspace(20*1e5,1e-3*1e5,NLAYER)

### Set up forward model
FM = ForwardModel()
FM.set_planet_model(
    M_plt=M_plt,R_plt=R_plt,
    gas_id_list=gas_id,iso_id_list=iso_id,
    NLAYER=NLAYER
    )
FM.set_opacity_data(
    kta_file_paths=lowres_file_paths,
    cia_file_path=cia_file_path
    )

# generate 2.5D temperature map
ck0 = -2
ck1 = -0.15
ck2 = 0.23
sk1 = -0.6
sk2 = -0.18

cg0 = -1.2
cg1 = 0.3
cg2 = 0
sg1 = 0
sg2 = -0.6

cf0 = -0.8
cf1 = -0.05
cf2 = -0.1
sf1 = -0.2
sf2 = 0.5

T_int = 200
T_night = 400
T_deep = 1500
T_jet = 50

tp_grid = gen_tmap_deep(P_range,global_lon_grid,global_lat_grid,
    g,T_eq,
    ck0,ck1,ck2,sk1,sk2,
    cg0,cg1,cg2,sg1,sg2,
    cf0,cf1,cf2,sf1,sf2,
    T_int,
    T_night,T_deep,T_jet)

# generate uniform abundance map
h2o = 1e-4
co2 = 1e-8
co = 1e-4
ch4 = 1e-8
vmr_grid = gen_vmrmap1(h2o,co2,co,ch4,
    nlon=len(global_lon_grid), nlat=len(global_lat_grid),
    npress=len(P_range))

retrieved_TP_phase_by_wave_20par = np.zeros((nphase,nwave))
retrieved_TP_wave_by_phase_20par = np.zeros((nwave,nphase))
# 20 par fit spec
for iphase, phase in enumerate(phase_grid):
    one_phase =  FM.calc_disc_spectrum(phase=phase, nmu=nmu, P_model=P_range,
        global_model_P_grid=P_range, global_T_model=tp_grid,
        global_VMR_model=vmr_grid,
        mod_lon=global_lon_grid,
        mod_lat=global_lat_grid,
        solspec=wasp43_spec)
    retrieved_TP_phase_by_wave_20par[iphase,:] = one_phase
for iwave in range(len(wave_grid)):
    for iphase in range(len(phase_grid)):
        retrieved_TP_wave_by_phase_20par[iwave,iphase] \
            = retrieved_TP_phase_by_wave_20par[iphase,iwave]

# Plot phase curve at each wavelength
fig, axs = plt.subplots(nrows=9,ncols=2,sharex=True,sharey=False,)
                        # figsize=[8.25,11.75],dpi=600)
plt.subplots_adjust(bottom=0.5)
xticks = np.array([0, 90, 180, 270, 360])
ix = 0
iy = 0
# linespec = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
# for iwave,wave in enumerate(wave_grid[::-1]):

#     linespec[iwave], \
#         = axs[ix,iy].plot(phase_grid, retrieved_TP_wave_by_phase_20par[16-iwave,:]*1e3,
#         marker='s',ms=0.1,mfc='r',color='r',linewidth=0.5,linestyle='-.',
#         label='Retrieval')

#     axs[ix,iy].errorbar(phase_grid, kevin_wave_by_phase[16-iwave,:,0]*1e3,
#         yerr = kevin_wave_by_phase[16-iwave,:,1]/2*1e3,
#         marker='s',ms=0.1,mfc='k',color='k',
#         linewidth=0.5,label='GCM data')

#     axs[ix,iy].set_yticklabels([])
#     handles, labels = axs[ix,iy].get_legend_handles_labels()

#     wave = np.around(wave,decimals=2)
#     axs[ix,iy].set_ylabel('{} $\mu$m '.format(wave),rotation=90,fontsize=16)

#     ix += 1
#     if ix == 9:
#         ix = 0
#         iy += 1

#########################
#########################
iwave = 0
wave = wave_grid[-1]
linespec, \
    = axs[0,0].plot(phase_grid, retrieved_TP_wave_by_phase_20par[16-iwave,:]*1e3,
    marker='s',ms=0.1,mfc='r',color='r',linewidth=0.5,linestyle='-.',
    label='Retrieval')

axs[0,0].errorbar(phase_grid, kevin_wave_by_phase[16-iwave,:,0]*1e3,
    yerr = kevin_wave_by_phase[16-iwave,:,1]/2*1e3,
    marker='s',ms=0.1,mfc='k',color='k',
    linewidth=0.5,label='Data')

axs[0,0].set_yticklabels([])
handles, labels = axs[0,0].get_legend_handles_labels()

wave = np.around(wave,decimals=2)
axs[0,0].set_ylabel('{} $\mu$m '.format(wave),rotation=90,)#fontsize=16)

ix += 1
if ix == 9:
    ix = 0
    iy += 1


axs[8,1].set_visible(False)
axs[8,0].set_xticks(xticks)
axs[8,0].tick_params(labelsize=16)
fig.legend(handles, labels, ncol=1, loc='lower right',
    fontsize=16)


### Slider Axes
axcolor = 'lightgoldenrodyellow'

axck0 = plt.axes([0.6, 0.35, 0.30, 0.03], facecolor=axcolor)
axck1 = plt.axes([0.6, 0.30, 0.30, 0.03], facecolor=axcolor)
axck2 = plt.axes([0.6, 0.25, 0.30, 0.03], facecolor=axcolor)
axsk1 = plt.axes([0.6, 0.20, 0.30, 0.03], facecolor=axcolor)
axsk2 = plt.axes([0.6, 0.15, 0.30, 0.03], facecolor=axcolor)

# axP_top = plt.axes([0.10, 0.35, 0.30, 0.03], facecolor=axcolor)
# axP_deep = plt.axes([0.10, 0.30, 0.30, 0.03], facecolor=axcolor)
# axNlayer = plt.axes([0.10, 0.25, 0.30, 0.03], facecolor=axcolor)
# axH2O = plt.axes([0.10, 0.20, 0.30, 0.03], facecolor=axcolor)
# axCO2 = plt.axes([0.10, 0.15, 0.30, 0.03], facecolor=axcolor)
# axCO = plt.axes([0.10, 0.10, 0.30, 0.03], facecolor=axcolor)
# axCH4 = plt.axes([0.10, 0.05, 0.30, 0.03], facecolor=axcolor)

### Slider values
sck0 = Slider(axck0, 'ck0', -4, 2, valinit=ck0 , valstep=0.1)
sck1 = Slider(axck1, 'ck1', -4, 2, valinit=ck1, valstep=0.1)
sck2 = Slider(axck2, 'ck2', -4, 2, valinit=ck2, valstep=0.1)
ssk1 = Slider(axsk1, 'sk1', -4, 2, valinit=sk1, valstep=0.1)
ssk2 = Slider(axsk2, 'sk2', -4, 2, valinit=sk2,valstep=0.1)

# sP_top = Slider(axP_top, 'P_top', -5 , -2, valinit=np.log10(P_top_init*1e-5),valstep=0.1)
# sP_deep = Slider(axP_deep, 'P_deep', -1 , 2, valinit=np.log10(P_deep_init*1e-5),valstep=0.1)
# sNlayer = Slider(axNlayer, 'Nlayer', 5, 200, valinit=NLAYER,valstep=1)
# sH2O = Slider(axH2O, 'H2O', -10, -1, valinit=H2O_init,valstep=0.1)
# sCO2 = Slider(axCO2, 'CO2', -10, -1, valinit=CO2_init,valstep=0.1)
# sCO = Slider(axCO, 'CO', -10, -1, valinit=CO_init,valstep=0.1)
# sCH4 = Slider(axCH4, 'CH4', -10, -1, valinit=CH4_init,valstep=0.1)

def update(val):
    ck0 = sck0.val
    ck1 = sck1.val
    ck2 = sck2.val
    sk1 = ssk1.val
    sk2 = ssk2.val
    tp_grid = gen_tmap_deep(P_range,global_lon_grid,global_lat_grid,
        g,T_eq,
        ck0,ck1,ck2,sk1,sk2,
        cg0,cg1,cg2,sg1,sg2,
        cf0,cf1,cf2,sf1,sf2,
        T_int,
        T_night,T_deep,T_jet)
    retrieved_TP_phase_by_wave_20par = np.zeros((nphase,nwave))
    retrieved_TP_wave_by_phase_20par = np.zeros((nwave,nphase))
    # 20 par fit spec
    for iphase, phase in enumerate(phase_grid):
        one_phase =  FM.calc_disc_spectrum(phase=phase, nmu=nmu, P_model=P_range,
            global_model_P_grid=P_range, global_T_model=tp_grid,
            global_VMR_model=vmr_grid,
            mod_lon=global_lon_grid,
            mod_lat=global_lat_grid,
            solspec=wasp43_spec)
        retrieved_TP_phase_by_wave_20par[iphase,:] = one_phase
    for iwave in range(len(wave_grid)):
        for iphase in range(len(phase_grid)):
            retrieved_TP_wave_by_phase_20par[iwave,iphase] \
                = retrieved_TP_phase_by_wave_20par[iphase,iwave]
    # for iwave,wave in enumerate(wave_grid[::-1]):
    #     linespec[iwave].set_ydata(
    #         axs[ix,iy].plot(phase_grid, retrieved_TP_wave_by_phase_20par[16-iwave,:]*1e3,
    #         marker='s',ms=0.1,mfc='r',color='r',linewidth=0.5,linestyle='-.',
    #         label='Retrieval'))
######
    iwave = 0
    wave = wave_grid[-1]
    linespec.set_ydata(retrieved_TP_wave_by_phase_20par[16-iwave,:]*1e3)
#######
    fig.canvas.draw_idle()

sck0.on_changed(update)
sck1.on_changed(update)
sck2.on_changed(update)
ssk1.on_changed(update)
ssk2.on_changed(update)

if __name__ == "__main__":
    plt.show()