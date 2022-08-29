 #!/usr/bin/env python3
# -*- coding: utf-8 -*-
from math import gamma
import numpy as np
import os
import pymultinest
import matplotlib.pyplot as plt
from nemesispy.models.TP_profiles import TP_Guillot
from scipy.special import voigt_profile
from nemesispy.data.gcm.process_gcm import (nlon,nlat,xlon,xlat,npv,pv,\
    tmap,h2omap,comap,co2map,ch4map,hemap,h2map,vmrmap,\
    tmap_mod,h2omap_mod,comap_mod,co2map_mod,ch4map_mod,\
    hemap_mod,h2map_mod,vmrmap_mod,phase_grid,\
    kevin_phase_by_wave,kevin_wave_by_phase,\
    pat_phase_by_wave,pat_wave_by_phase,\
    vmrmap_mod_new,tmap_hot)

# Generator function
def gen(x,
    ck0,ck1,ck2,sk1,sk2,
    cg0,cg1,cg2,sg1,sg2,
    cf0,cf1,cf2,sf1,sf2,
    ct0,ct1,ct2,st1,st2):
    """
    Generate 2-Stream Guillot profile
    """

    y = x/180*np.pi

    log_kappa = ck0 + ck1 * np.cos(y) + ck2 * np.cos(2*y)\
        + sk1 * np.sin(y) + sk2 * np.sin(2*y)

    log_gamma = cg0 + cg1 * np.cos(y) + cg2 * np.cos(2*y)\
        + sg1 * np.sin(y) + sg2 * np.sin(2*y)

    log_f = cf0 + cf1 * np.cos(y) + cf2 * np.cos(2*y)\
        + sf1 * np.sin(y) + sf2 * np.sin(2*y)

    log_T_int = ct0 + ct1 * np.cos(y) + ct2 * np.cos(2*y)\
        + st1 * np.sin(y) + st2 * np.sin(2*y)

    return log_kappa, log_gamma, log_f, log_T_int

### Reference Planet Input: WASP 43b
G = 6.67430e-11 # m3 kg-1 s-2 Newtonian constant of gravitation
T_star = 4520 # star temperature in K
R_star = 463892759.99999994 # m, 0.6668 * R_SUN
SMA = 2243970000.0 # m, 0.015*AU
M_plt = 3.8951064000000004e+27 # kg
R_plt = 74065.70 * 1e3 # m

## Derived
T_irr = T_star * (R_star/SMA)**0.5
g = G*M_plt/R_plt**2

### T map specs
NLAYER = 20
P_range = np.geomspace(20*1e5,1e-3*1e5,NLAYER)
Np = len(P_range)

# Get Fourier fits
retrieved_T_map_Fourier = np.zeros((nlon,nlat,Np))
for ilat in range(16):
    index,cube \
        = np.loadtxt('chains/20ilat-{}stats.dat'.format(ilat),
            skiprows=50,unpack=True)
    log_kappa, log_gamma, log_f, log_T_int \
        = gen(xlon,
            cube[0],cube[1],cube[2],cube[3],cube[4],
            cube[5],cube[6],cube[7],cube[8],cube[9],
            cube[10],cube[11],cube[12],cube[13],cube[14],
            cube[15],cube[16],cube[17],cube[18],cube[19])
    ka = 10**log_kappa
    ga = 10**log_gamma
    f = 10**log_f
    T_int = 10**log_T_int
    for ilon in range(nlon):
        tp = TP_Guillot(P=P_range,g_plt=g,T_eq=T_irr/2**0.5,k_IR=ka,gamma=ga,
            f=f,T_int=T_int)
        retrieved_T_map_Fourier[ilon,ilat,:] = tp
        retrieved_T_map_Fourier[ilon,-ilat-1,:] = tp

T_profiles_Guillot \
    = np.loadtxt('AAbest_fit_Guillot.txt',ndmin=2,delimiter=',')
retrieved_T_map_Guillot = np.zeros((nlon,nlat,Np))
for ilon in range(nlon):
    for ilat in range(nlat):
        TP = T_profiles_Guillot[ilon*32+ilat,]
        # f = interpolate.interp1d(P_range,TP,fill_value=(TP[0],TP[-1]),
        #     bounds_error=False)
        retrieved_T_map_Guillot[ilon,ilat,:] = TP



# T_map_list = [
#     retrieved_T_map_Guillot,
#     ]
# T_name = [
#     '2-Stream',
# ]

# # Pressure grid of the GCM
# pressure_grid = P_range

# # ticks
# xticks = np.array([-180, -150, -120,  -90,  -60,  -30,    0,   30,
#     60,   90,  120, 150,  180])
# # move the y ticks to the foreshortened location
# yticks_loc = np.sin(np.array([-60, -30,   0,  30,  60])/180*np.pi)*90
# yticks_label = np.array([-60, -30,   0,  30,  60])

# for ip in range(Np):

#     pressure = pressure_grid[ip]/1e5 # convert to unit of bar

#     fig,axs = plt.subplots(nrows=1, ncols=1,figsize=[4,4],dpi=400,
#         sharex=True,sharey=True,)
#     fig.supxlabel('Longitude [degree]',fontsize='small')
#     fig.supylabel('Latitude [degree]',fontsize='small')
#     fig.suptitle(r'$\Delta T$ at '+'Pressure = {:.1e} bar'.format(pressure),fontsize='x-small')

#     # set up foreshortened latitude coordinates
#     fs = np.sin(xlat/180*np.pi)*90
#     x,y = np.meshgrid(xlon,fs,indexing='ij')

#     # read in 2-strem Guillot GCM temperature map

#     z = retrieved_T_map_Fourier[:,:,ip] - retrieved_T_map_Guillot[:,:,ip]
#     # plt.contourf(x,y,z,levels=10,vmin=400,vmax=2600,cmap='magma')
#     im = axs[0].contourf(x,y,z,levels=10,cmap='bwr',vmin=-200,vmax=200)
#     # axs[0].scatter(x,np.sin(y/180*np.pi)*90,s=1,marker='x',color='k')
#     axs[0].set_title('{}'.format('Fourier Fit - Guillot'),fontsize='xx-small')
#     cbar = fig.colorbar(im, ax=axs[0])
#     cbar.ax.tick_params(labelsize=5)

#     axs[0].tick_params(axis='both',which='major',labelsize=5)
#     axs[2].set_xticks(xticks)
#     axs[2].set_yticks(yticks_loc,yticks_label)

#     fig.tight_layout()
#     plt.savefig('plots/compare_T_contour_pressure_{}.pdf'.format(ip))
#     plt.show()

# plots
# for ip in range(len(pressure_grid)):

#     pressure = pressure_grid[ip]/1e5 # convert to unit of bar

#     fig,axs = plt.subplots(nrows=3, ncols=1,figsize=[4,4],dpi=400,
#         sharex=True,sharey=True,)
#     fig.supxlabel('Longitude [degree]',fontsize='small')
#     fig.supylabel('Latitude [degree]',fontsize='small')
#     fig.suptitle(r'$\Delta T$ at '+'Pressure = {:.1e} bar'.format(pressure),fontsize='x-small')

#     # set up foreshortened latitude coordinates
#     fs = np.sin(xlat/180*np.pi)*90
#     x,y = np.meshgrid(xlon,fs,indexing='ij')

#     # read in GCM temperature map
#     for imap,map in enumerate(T_map_list):
#         z = map[:,:,ip] - tmap[:,:,ip]
#         # plt.contourf(x,y,z,levels=10,vmin=400,vmax=2600,cmap='magma')
#         im = axs[imap].contourf(x,y,z,levels=10,cmap='bwr',vmin=-200,vmax=200)
#         # axs[imap].scatter(x,np.sin(y/180*np.pi)*90,s=1,marker='x',color='k')
#         axs[imap].set_title('{}'.format(T_name[imap]),fontsize='xx-small')
#         cbar = fig.colorbar(im, ax=axs[imap])
#         cbar.ax.tick_params(labelsize=5)
#         axs[imap].tick_params(axis='both',which='major',labelsize=5)
#     axs[2].set_xticks(xticks)
#     axs[2].set_yticks(yticks_loc,yticks_label)

#     fig.tight_layout()
#     plt.savefig('plots/compare_T_contour_pressure_{}.pdf'.format(ip))
#     plt.show()