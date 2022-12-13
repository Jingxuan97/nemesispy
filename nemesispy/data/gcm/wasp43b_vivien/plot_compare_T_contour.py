# -*- coding: utf-8 -*-
"""Compared the retrieved TP structure to the original GCM
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import scipy.interpolate as interpolate

# Read GCM data
from nemesispy.data.gcm.wasp43b_vivien.process_wasp43b_gcm_vivien import (
    nlon,nlat,xlon,xlat,npv,pv,\
    tmap,h2omap,comap,co2map,ch4map,hemap,h2map,vmrmap,\
    tmap_mod,h2omap_mod,comap_mod,co2map_mod,ch4map_mod,\
    hemap_mod,h2map_mod,vmrmap_mod,phase_grid,\
    kevin_phase_by_wave,kevin_wave_by_phase,\
    pat_phase_by_wave,pat_wave_by_phase)

NLAYER = 20
P_range = np.geomspace(20*1e5,1e-3*1e5,NLAYER)

T_profiles_Guillot_fixed_T_int \
    = np.loadtxt('best_fit_TP_2_stream_Guillot_fixed_T_int.txt',
        ndmin=2,delimiter=',')
retrieved_T_map_Guillot_fixed_T_int = np.zeros((nlon,nlat,npv))
for ilon in range(nlon):
    for ilat in range(nlat):
        TP = T_profiles_Guillot_fixed_T_int[ilon*32+ilat,]
        f = interpolate.interp1d(P_range,TP,fill_value=(TP[0],TP[-1]),
            bounds_error=False)
        retrieved_T_map_Guillot_fixed_T_int[ilon,ilat,:] = f(pv)

T_profiles_Guillot \
    = np.loadtxt('best_fit_TP_2_stream_Guillot.txt',
        ndmin=2,delimiter=',')
retrieved_T_map_Guillot = np.zeros((nlon,nlat,npv))
for ilon in range(nlon):
    for ilat in range(nlat):
        TP = T_profiles_Guillot[ilon*32+ilat,]
        f = interpolate.interp1d(P_range,TP,fill_value=(TP[0],TP[-1]),
            bounds_error=False)
        retrieved_T_map_Guillot[ilon,ilat,:] = f(pv)

T_profiles_Line \
    = np.loadtxt('best_fit_TP_3_stream_Guillot_fixed_T.txt',
        ndmin=2,delimiter=',')
retrieved_T_map_Line = np.zeros((nlon,nlat,npv))
for ilon in range(nlon):
    for ilat in range(nlat):
        TP = T_profiles_Line[ilon*32+ilat,]
        f = interpolate.interp1d(P_range,TP,fill_value=(TP[0],TP[-1]),
            bounds_error=False)
        retrieved_T_map_Line[ilon,ilat,:] = f(pv)

T_map_list = [
    retrieved_T_map_Guillot_fixed_T_int,
    retrieved_T_map_Guillot,
    retrieved_T_map_Line
    ]
T_name = [
    r'2-Stream (Fixed $T_{int}$)',
    '2-Stream',
    r'3-Stream (Fixed $T_{int}$)'
]

# Pressure grid of the GCM
pressure_grid = pv

# ticks
xticks = np.array([-180,  -90,      0,     90,   180])
# move the y ticks to the foreshortened location
yticks_loc = np.sin(np.array([-60, -30,   0,  30,  60])/180*np.pi)*90
yticks_label = np.array([-60, -30,   0,  30,  60])
for ip in range(7,35):

    pressure = pressure_grid[ip]/1e5 # convert to unit of bar

    fig,axs = plt.subplots(nrows=3, ncols=1,figsize=[4,5],dpi=800,
        sharex=True,sharey=True,)
    fig.supxlabel('Longitude [$^\circ$]',fontsize=10)
    fig.supylabel('Latitude [$^\circ$]',fontsize=10)
    # fig.suptitle(r'$\Delta T$ at '+'P = {:.2f} bar'.format(pressure),
    #     fontsize=12)

    # set up foreshortened latitude coordinates
    fs = np.sin(xlat/180*np.pi)*90
    x,y = np.meshgrid(xlon,fs,indexing='ij')

    # read in GCM temperature map
    for imap,map in enumerate(T_map_list):
        z = map[:,:,ip] - tmap[:,:,ip]
        # plt.contourf(x,y,z,levels=10,vmin=400,vmax=2600,cmap='magma')
        im = axs[imap].contourf(x,y,z,levels=10,cmap='bwr',vmin=-200,vmax=200)
        # axs[imap].scatter(x,np.sin(y/180*np.pi)*90,s=1,marker='x',color='k')
        axs[imap].set_title('{}'.format(T_name[imap]),fontsize=12,
              fontweight="bold")
        cbar = fig.colorbar(im, ax=axs[imap])
        cbar.ax.tick_params(labelsize=5)
        axs[imap].tick_params(axis='both',which='major',labelsize=10)
        axs[imap].set_yticks(yticks_loc,yticks_label,fontsize=10)
    axs[2].set_xticks(xticks)

    fig.tight_layout()
    plt.savefig('figures/compare_T_contour_pressure_{}.pdf'.format(ip),
        dpi=800)
    plt.close()