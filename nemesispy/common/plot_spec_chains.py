#!/usr/local/bin/python3
# -*- coding: utf-8 -*-
"""
Plot simulation from retrieved 1D TP profiles
"""
import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
import scipy.interpolate as interpolate
from nemesispy.data.gcm.process_gcm \
    import (nlon,nlat,xlon,xlat,npv,pv,\
    tmap,h2omap,comap,co2map,ch4map,hemap,h2map,vmrmap,\
    tmap_mod,h2omap_mod,comap_mod,co2map_mod,ch4map_mod,hemap_mod,h2map_mod,
    vmrmap_mod,phase_grid,\
    kevin_phase_by_wave,kevin_wave_by_phase,pat_phase_by_wave,pat_wave_by_phase)

from nemesispy.radtran.forward_model import ForwardModel
lowres_files = ['/Users/jingxuanyang/ktables/h2owasp43.kta',
'/Users/jingxuanyang/ktables/cowasp43.kta',
'/Users/jingxuanyang/ktables/co2wasp43.kta',
'/Users/jingxuanyang/ktables/ch4wasp43.kta']

cia_file_path = '/Users/jingxuanyang/Desktop/Workspace/' \
    + 'nemesispy2022/nemesispy/data/cia/exocia_hitran12_200-3800K.tab'

################################################################################
### Wavelengths grid and orbital phase grid
wave_grid = np.array([1.1425, 1.1775, 1.2125, 1.2475, 1.2825, 1.3175, 1.3525,
    1.3875, 1.4225, 1.4575, 1.4925, 1.5275, 1.5625, 1.5975, 1.6325, 3.6   ,
    4.5   ])
phase_grid = np.array([ 22.5,  45. ,  67.5,  90. , 112.5, 135. , 157.5, 180. , 202.5,
       225. , 247.5, 270. , 292.5, 315. , 337.5])
nwave = len(wave_grid)
nphase = len(phase_grid)
wasp43_spec = np.array([3.341320e+25, 3.215455e+25, 3.101460e+25, 2.987110e+25,
       2.843440e+25, 2.738320e+25, 2.679875e+25, 2.598525e+25,
       2.505735e+25, 2.452230e+25, 2.391140e+25, 2.345905e+25,
       2.283720e+25, 2.203690e+25, 2.136015e+25, 1.234010e+24,
       4.422200e+23])
retrieved_TP_phase_by_wave = np.zeros((nphase,nwave))
retrieved_TP_wave_by_phase = np.zeros((nwave,nphase))
gcm_phase_by_wave = np.zeros((nphase,nwave))
gcm_wave_by_phase = np.zeros((nwave,nphase))

### Reference Planet Input
M_plt = 3.8951064000000004e+27 # kg
R_plt = 74065.70 * 1e3 # m
gas_id = np.array([  1, 2,  5,  6, 40, 39])
iso_id = np.array([0, 0, 0, 0, 0, 0])
NLAYER = 20
nmu = 5
P_range = np.geomspace(20*1e5,1e-3*1e5,NLAYER)

### Create TP map hack
T_profiles = np.loadtxt('AAbest_fit.txt',ndmin=2,delimiter=',')
retrieved_T_map = np.zeros((nlon,nlat,npv))
for ilon in range(nlon):
    for ilat in range(nlat):
        TP = T_profiles[ilon*32+ilat,]
        f = interpolate.interp1d(P_range,TP,fill_value=(TP[0],TP[-1]),bounds_error=False)
        retrieved_T_map[ilon,ilat,:] = f(pv)

### Create TP map by chopping gcm data
gcm_T_profiles = np.loadtxt('AAgcm_reshaped.txt',ndmin=2,delimiter=',')
gcm_T_map = np.zeros((nlon,nlat,npv))
for ilon in range(nlon):
    for ilat in range(nlat):
        TP = gcm_T_profiles[ilon*32+ilat,]
        f = interpolate.interp1d(P_range,TP,fill_value=(TP[0],TP[-1]),bounds_error=False)
        gcm_T_map[ilon,ilat,:] = f(pv)

### Set up forward model
FM = ForwardModel()
FM.set_planet_model(M_plt=M_plt,R_plt=R_plt,gas_id_list=gas_id,iso_id_list=iso_id,
    NLAYER=NLAYER)
FM.set_opacity_data(kta_file_paths=lowres_files, cia_file_path=cia_file_path)
### 1D
for iphase, phase in enumerate(phase_grid):
    one_phase =  FM.calc_disc_spectrum(phase=phase, nmu=nmu, P_model=P_range,
        global_model_P_grid=pv, global_T_model=retrieved_T_map,
        global_VMR_model=vmrmap_mod, mod_lon=xlon, mod_lat=xlat,
        solspec=wasp43_spec)
    retrieved_TP_phase_by_wave[iphase,:] = one_phase
for iwave in range(len(wave_grid)):
    for iphase in range(len(phase_grid)):
        retrieved_TP_wave_by_phase[iwave,iphase] \
            = retrieved_TP_phase_by_wave[iphase,iwave]
### GCM
for iphase, phase in enumerate(phase_grid):
    one_phase =  FM.calc_disc_spectrum(phase=phase, nmu=nmu, P_model=P_range,
        global_model_P_grid=pv, global_T_model=gcm_T_map,
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
# add a big axis, hide frame
fig.add_subplot(111,frameon=False)
# hide tick and tick label of the big axis
plt.tick_params(labelcolor='none',which='both',top=False,bottom=False,left=False,right=False)
plt.xlabel(r'Wavelength($\mu$m)')
plt.ylabel('Flux ratio x ($10^{3}$)')
ix = 0
iy = 0
for iphase,phase in enumerate(phase_grid):
    """
    axs[ix,iy].errorbar(wave_grid, kevin_phase_by_wave[iphase,:,0]*1e3,
                        yerr=kevin_phase_by_wave[iphase,:,1]*1e3,
                        marker='s',ms=0.1,ecolor='r',mfc='k',color='k',
                        linewidth=0.1,label='data')
    """
    axs[ix,iy].plot(wave_grid, retrieved_TP_phase_by_wave[iphase,:]*1e3,
                    marker='s',ms=0.1,mfc='b',color='b',
                    linewidth=0.1,label='1D')

    axs[ix,iy].plot(wave_grid, gcm_phase_by_wave[iphase,:]*1e3,
                    marker='s',ms=0.1,mfc='r',color='r',
                    linewidth=0.1,label='GCM limited')

    axs[ix,iy].plot(wave_grid, pat_phase_by_wave[iphase,:]*1e3,
                    marker='s',ms=0.1,mfc='g',color='g',
                    linewidth=0.1,label='GCM full')

    axs[ix,iy].legend(loc='upper left',fontsize='x-small')
    # axs[ix,iy].grid()
    axs[ix,iy].text(3,3.5,'{}'.format(phase),fontsize=6)
    iy += 1
    if iy == 3:
        iy = 0
        ix += 1
plt.savefig('best_fit_spectra_compare.pdf')
plt.show()

# Plot phase curve at each wavelength
fig, axs = plt.subplots(nrows=17,ncols=1,sharex=True,sharey=False,
                        figsize=[5,13],dpi=600)
# plt.xlim(0.,1.)
# add a big axis, hide frame
fig.add_subplot(111,frameon=False)
# hide tick and tick label of the big axis
plt.tick_params(labelcolor='none',which='both',top=False,bottom=False,left=False,right=False)
plt.xlabel('phase')
plt.ylabel(r'Wavelength($\mu$m)')

for iwave,wave in enumerate(wave_grid[::-1]):
    """
    axs[iwave].errorbar(phase_grid, kevin_wave_by_phase[16-iwave,:,0]*1e3,
                        yerr=kevin_wave_by_phase[16-iwave,:,1]*1e3,
                        marker='s',ms=0.1,ecolor='r',mfc='k',color='k',linewidth=0.1,
                        label='Data')
    """
    axs[iwave].plot(phase_grid, retrieved_TP_wave_by_phase[16-iwave,:]*1e3,
                        marker='s',ms=0.1,mfc='b',color='b',linewidth=0.5,linestyle='--',
                        label='1D')

    axs[iwave].plot(phase_grid, retrieved_TP_wave_by_phase[16-iwave,:]*1e3,
                        marker='s',ms=0.1,mfc='r',color='r',linewidth=0.5,linestyle=':',
                        label='GCM limited')

    axs[iwave].plot(phase_grid, pat_wave_by_phase[16-iwave,:]*1e3,
                    marker='s',ms=0.1,mfc='g',color='g',
                    linewidth=0.5,label='GCM full')

    # axs[iwave].get_yaxis().set_visible(False)
    axs[iwave].set_yticklabels([])
    wave = np.around(wave,decimals=2)
    axs[iwave].set_ylabel(wave,rotation=0,fontsize=8)
    handles, labels = axs[iwave].get_legend_handles_labels()
    # axs[iwave].legend()

fig.legend(handles, labels, loc='upper left', fontsize='x-small')
plt.tight_layout()

plt.savefig('best_fit_phase_curve_compare.pdf')
plt.show()

"""
### This is for plotting specta at all phases
for iphase in range(nphase):
    phasenumber = iphase
    fig, axs = plt.subplots(nrows=2,ncols=1,sharex=True,
        dpi=800)
    axs[0].set_title('phase = {}'.format(phasenumber))
    axs[0].plot(wave_grid,retrieved_TP_phase_by_wave[phasenumber,:],color='b',label='Python')
    axs[0].scatter(wave_grid,kevin_phase_by_wave[phasenumber,:,0],color='r',marker='+',label='Data')
    axs[0].plot(wave_grid,pat_phase_by_wave[phasenumber,:],color ='k',label='Fortran')
    axs[0].legend(loc='upper left')
    axs[0].grid()
    axs[0].set_ylabel('Flux ratio')

    diff = (one_phase - pat_phase_by_wave[phasenumber,:])/one_phase
    axs[1].scatter(wave_grid,diff,marker='.',color='b')
    axs[1].grid()
    axs[1].set_ylabel('Relative diff')
    axs[1].set_xlabel('Wavelength (Micron)')
    print(iphase,diff)
    plt.tight_layout()

    plt.show()
    # plt.savefig('good_discav_planet{}.pdf'.format(iphase),dpi=800)
"""