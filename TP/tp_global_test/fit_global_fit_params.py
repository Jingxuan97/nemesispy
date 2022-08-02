# -*- coding: utf-8 -*-

import numpy as np 
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as colors

from nemesispy.data.gcm.process_gcm import (nlon,nlat,xlon,xlat,npv,pv,\
    tmap,h2omap,comap,co2map,ch4map,hemap,h2map,vmrmap,\
    tmap_mod,h2omap_mod,comap_mod,co2map_mod,ch4map_mod,\
    hemap_mod,h2map_mod,vmrmap_mod,phase_grid,\
    kevin_phase_by_wave,kevin_wave_by_phase,\
    pat_phase_by_wave,pat_wave_by_phase,\
    vmrmap_mod_new,tmap_hot)
    

## Load the best fitting 1D model parameters
best_params = np.zeros((nlon,nlat,5))
raw_params = np.loadtxt('AAbest_params.txt',unpack=True,delimiter=',')
raw_params = raw_params.T


for ilon in range(nlon):
    for ilat in range(nlat):
        best_params[ilon,ilat,:] = raw_params[ilon*nlat+ilat,:]

log_kappa = best_params[:,:,0]
log_gamma1 = best_params[:,:,1]
log_gamma2 = best_params[:,:,2]
alpha = best_params[:,:,3]
beta = best_params[:,:,4]

## Plot the best fitting 1D parameters 
x,y = np.meshgrid(xlon,xlat,indexing='ij')
xticks = np.array([-180, -150, -120,  -90,  -60,  -30,    0,   30,   60,   90,  
            120,  150,  180])
yticks = np.array([-60, -30,   0,  30,  60])

"""
for ilat in range(15,nlat):
    plt.plot(xlon,beta[:,ilat],label=xlat[ilat])
    plt.legend(loc='upper right')
    plt.ylim(400,1800)
    plt.show()
"""

"""
for ilat in range(15,nlat):
    plt.plot(xlon,log_kappa[:,ilat],label=xlat[ilat])
    plt.legend(loc='upper right')
    # plt.ylim(400,1800)
    plt.show()
"""
"""
linear_kappa = 10**log_kappa 
for ilat in range(15,nlat):
    plt.plot(xlon,linear_kappa[:,ilat],label=xlat[ilat],marker='x')
plt.legend(loc='upper right')
    # plt.ylim(400,1800)
plt.figure(figsize=(5,15))
plt.show()
"""


### North-South Symmetry
"""
linear_kappa = 10**log_kappa 
for ilat in range(0,16):
    plt.plot(xlon,linear_kappa[:,ilat],label=xlat[ilat],marker='x')
    plt.plot(xlon,linear_kappa[:,-ilat-1],label=xlat[-ilat-1],marker='x')
    plt.legend(loc='upper right')
    plt.ylim(0,1)
    plt.grid()
    plt.show()
"""
for ilat in range(0,16):
    plt.plot(xlon,beta[:,ilat],label=xlat[ilat],marker='x')
    plt.plot(xlon,beta[:,-ilat-1],label=xlat[-ilat-1],marker='x')
    plt.legend(loc='upper right')
    plt.ylim(400,1800)
    plt.grid()
    plt.show()

