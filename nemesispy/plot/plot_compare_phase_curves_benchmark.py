#!/usr/local/bin/python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from nemesispy.radtran.forward_model import ForwardModel
# Read GCM data
from nemesispy.data.gcm.process_gcm import (nlon,nlat,xlon,xlat,npv,pv,\
    tmap,h2omap,comap,co2map,ch4map,hemap,h2map,vmrmap,\
    tmap_mod,h2omap_mod,comap_mod,co2map_mod,ch4map_mod,\
    hemap_mod,h2map_mod,vmrmap_mod,phase_grid,\
    kevin_phase_by_wave,kevin_wave_by_phase,\
    pat_phase_by_wave,pat_wave_by_phase,\
    vmrmap_mod_new,tmap_hot)
import scipy.interpolate as interpolate

### Reference Opacity Data
lowres_files = [
'/Users/jingxuanyang/Desktop/Workspace/nemesispy2022/nemesispy/data/ktables/h2o',
'/Users/jingxuanyang/Desktop/Workspace/nemesispy2022/nemesispy/data/ktables/co2',
'/Users/jingxuanyang/Desktop/Workspace/nemesispy2022/nemesispy/data/ktables/co',
'/Users/jingxuanyang/Desktop/Workspace/nemesispy2022/nemesispy/data/ktables/ch4']
cia_file_path='/Users/jingxuanyang/Desktop/Workspace/nemesispy2022/nemesispy/data/cia/exocia_hitran12_200-3800K.tab'

### Reference Spectral Input
wasp43_spec = np.array([3.341320e+25, 3.215455e+25, 3.101460e+25, 2.987110e+25,
       2.843440e+25, 2.738320e+25, 2.679875e+25, 2.598525e+25,
       2.505735e+25, 2.452230e+25, 2.391140e+25, 2.345905e+25,
       2.283720e+25, 2.203690e+25, 2.136015e+25, 1.234010e+24,
       4.422200e+23])
wave_grid = np.array([1.1425, 1.1775, 1.2125, 1.2475, 1.2825, 1.3175, 1.3525, 1.3875,
       1.4225, 1.4575, 1.4925, 1.5275, 1.5625, 1.5975, 1.6325, 3.6   ,
       4.5   ])
phase_grid = np.array([ 22.5,  45. ,  67.5,  90. , 112.5, 135. , 157.5, 180. , 202.5,
       225. , 247.5, 270. , 292.5, 315. , 337.5])
nwave = len(wave_grid)
nphase = len(phase_grid)

### Reference Planet Input
M_plt = 3.8951064000000004e+27 # kg
R_plt = 74065.70 * 1e3 # m
gas_id = np.array([  1, 2,  5,  6, 40, 39])
iso_id = np.array([0, 0, 0, 0, 0, 0])
NLAYER = 20
P_range = np.geomspace(20*1e5,1e-3*1e5,NLAYER)
nmu = 5

### Set up forward model
FM = ForwardModel()
FM.set_planet_model(M_plt=M_plt,R_plt=R_plt,gas_id_list=gas_id,iso_id_list=iso_id,
    NLAYER=NLAYER)
FM.set_opacity_data(kta_file_paths=lowres_files, cia_file_path=cia_file_path)

# Python
gcm_phase_by_wave = np.zeros((nphase,nwave))
gcm_wave_by_phase = np.zeros((nwave,nphase))

### Generate Phase curves
# Python spec
for iphase, phase in enumerate(phase_grid):
    one_phase =  FM.calc_disc_spectrum(phase=phase, nmu=nmu, P_model=P_range,
        global_model_P_grid=pv, global_T_model=tmap_mod,
        global_VMR_model=vmrmap_mod, mod_lon=xlon, mod_lat=xlat,
        solspec=wasp43_spec)
    gcm_phase_by_wave[iphase,:] = one_phase
for iwave in range(len(wave_grid)):
    for iphase in range(len(phase_grid)):
        gcm_wave_by_phase[iwave,iphase] \
            = gcm_phase_by_wave[iphase,iwave]

### Plots
# Plot spectrum at each phase
fig, axs = plt.subplots(nrows=5,ncols=3,sharex=True,sharey=True,
                        figsize=[8.25,11.75],dpi=600)
plt.xlim(1,4.6)
plt.ylim(-5e-1,5)

fig.supxlabel(r'Wavelength [$\mu$m]')
fig.supylabel('Flux ratio (x$10^{3}$)')
ix = 0
iy = 0
for iphase,phase in enumerate(phase_grid):

    axs[ix,iy].plot(wave_grid, gcm_phase_by_wave[iphase,:]*1e3,
        marker='s',ms=0.1,mfc='g',color='g',
        linewidth=0.3,label='Python ')

    axs[ix,iy].plot(wave_grid, pat_phase_by_wave[iphase,:]*1e3,
        marker='s',ms=0.1,mfc='k',color='k',
        linewidth=0.3,label='Fortran')

    if ix == 0 and iy == 0:
        axs[ix,iy].legend(loc='upper left',fontsize='xx-small')

    axs[ix,iy].text(2.7,3.5,r'$\phi$=' + r'{}'.format(phase/360),fontsize=6)

    # populate plot
    iy += 1
    if iy == 3:
        iy = 0
        ix += 1

fig.tight_layout()
plt.savefig('benchmark_spectra.pdf')
plt.show()

# Plot phase curve at each wavelength
fig, axs = plt.subplots(nrows=9,ncols=2,sharex=True,sharey=False,
                        figsize=[8.25,11.75],dpi=600)

fig.supxlabel('phase')
fig.supylabel(r'Wavelength [$\mu$m]')

ix = 0
iy = 0
for iwave,wave in enumerate(wave_grid[::-1]):

    axs[ix,iy].plot(phase_grid, gcm_wave_by_phase[16-iwave,:]*1e3,
        marker='s',ms=0.1,mfc='g',color='g',
        linewidth=0.5,label='Python')

    axs[ix,iy].errorbar(phase_grid, pat_wave_by_phase[16-iwave,:]*1e3,
        yerr = kevin_wave_by_phase[16-iwave,:,1]/2*1e3,
        marker='s',ms=0.1,mfc='k',color='k',
        linewidth=0.5,label='Fortran')

    axs[ix,iy].set_yticklabels([])
    wave = np.around(wave,decimals=2)
    axs[ix,iy].set_ylabel(wave,rotation=0,fontsize=8)
    handles, labels = axs[ix,iy].get_legend_handles_labels()

    if ix == 6 and iy == 1:
        wave = np.around(wave,decimals=2)
        axs[ix,iy].set_ylabel(wave,rotation=0,fontsize=8)
    ix += 1
    if ix == 9:
        ix = 0
        iy += 1

axs[8,1].set_visible(False)

fig.legend(handles, labels, ncol=5, loc='lower right', fontsize=15)
fig.tight_layout()

plt.savefig('benchmark_phase_curves.pdf')
plt.show()

# # Calculate chi^2
# chi_Guillot_fixed = 0
# chi_Guillot = 0
# chi_Line = 0
# for iwave, wave in enumerate(wave_grid[::-1]):
#     chi_Guillot_fixed+=np.sum((gcm_wave_by_phase \
#      - retrieved_TP_wave_by_phase_Guillot_fixed_T_int)**2/kevin_wave_by_phase[iwave,:,1]**2)
#     chi_Guillot+=np.sum((gcm_wave_by_phase \
#      - retrieved_TP_wave_by_phase_Guillot)**2/kevin_wave_by_phase[iwave,:,1]**2)
#     chi_Line+=np.sum((gcm_wave_by_phase \
#      - retrieved_TP_wave_by_phase_Line)**2/kevin_wave_by_phase[iwave,:,1]**2)

# chi_Guillot_fixed = chi_Guillot_fixed/(nwave*nphase)
# chi_Guillot = chi_Guillot/(nwave*nphase)
# chi_Line = chi_Line/(nwave*nphase)
