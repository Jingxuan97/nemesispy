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
        
## Plot the best fitting 1D parameters 
x,y = np.meshgrid(xlon,xlat,indexing='ij')
xticks = np.array([-180, -150, -120,  -90,  -60,  -30,    0,   30,   60,   90,  
            120,  150,  180])
yticks = np.array([-60, -30,   0,  30,  60])

"""
z_kappa = 10**best_params[:,:,0]
plt.figure(figsize=(15,5))
plt.contourf(x,y,z_kappa,
        # levels=[1e-3,1e-2,1e-1,1e0,1e1,1e2],
        levels=100,
        cmap='magma',
        norm=colors.LogNorm(
            vmin=z_kappa.min(),
            vmax=z_kappa.max())
        )
plt.colorbar()
plt.xticks(xticks)
plt.yticks(yticks)
plt.title(r'$\kappa$')
plt.savefig('dist_kappa_log.pdf')
plt.show()
"""

"""
z_kappa = 10**best_params[:,:,0]
plt.figure(figsize=(15,5))
plt.contourf(x,y,z_kappa,
        # levels=[1e-3,1e-2,1e-1,1e0,1e1,1e2],
        levels=100,
        cmap='magma',
        vmin=z_kappa.min(),
        vmax=z_kappa.max()
        )
plt.colorbar()
plt.xticks(xticks)
plt.yticks(yticks)
plt.title(r'$\kappa$')
plt.savefig('dist_kappa_linear.pdf')
plt.show()

z_kappa = best_params[:,:,0]
plt.figure(figsize=(15,5))
plt.contourf(x,y,z_kappa,
        # levels=[1e-3,1e-2,1e-1,1e0,1e1,1e2],
        levels=100,
        cmap='magma',
        )
plt.colorbar()
plt.xticks(xticks)
plt.yticks(yticks)
plt.title(r'$\kappa$')
plt.show()
"""

"""
z_kappa = 10**best_params[:,:,0]
plt.figure(figsize=(15,5))
plt.contourf(x,y,z_kappa,
        # levels=[1e-3,1e-2,1e-1,1e0,1e1,1e2],
        levels=100,
        cmap='magma',
        norm=colors.LogNorm(
            vmin=1e-3,
            vmax=1)
        )
plt.colorbar()
plt.xticks(xticks)
plt.yticks(yticks)
plt.title(r'$\kappa$')
plt.show()
"""

z_gamma1 = 10**best_params[:,:,1]
plt.figure(figsize=(15,5))
plt.contourf(x,y,z_gamma1,
             levels=10,
             cmap='magma',
             vmin=z_gamma1.min(),
             vmax=z_gamma1.max()
             )
plt.colorbar()
plt.xticks(xticks)
plt.yticks(yticks)
plt.title(r'$\gamma_1$')
plt.savefig('dist_gamma1_linear.pdf')
plt.show()

"""
z_gamma1 = 10**best_params[:,:,1]
plt.figure(figsize=(15,5))
plt.contourf(x,y,z_gamma1,
             levels=10,
             cmap='magma',
             norm=colors.LogNorm(
                 vmin=z_gamma1.min(),
                 vmax=z_gamma1.max())
             )
plt.colorbar()
plt.xticks(xticks)
plt.yticks(yticks)
plt.title(r'$\gamma_1$')
plt.savefig('dist_gamma1_linear.pdf')
plt.show()
"""

"""
z_gamma1 = best_params[:,:,1]
plt.figure(figsize=(15,5))
plt.contourf(x,y,z_gamma1,
             levels=10,
             cmap='magma',
             )
plt.colorbar()
plt.xticks(xticks)
plt.yticks(yticks)
plt.title(r'$\gamma_1$')
plt.show()


z_gamma2 = 10**best_params[:,:,2]
plt.figure(figsize=(15,5))
plt.contourf(x,y,z_gamma2,
             levels=10,
             cmap='magma',
             norm=colors.LogNorm(
                 vmin=z_gamma2.min(),
                 vmax=z_gamma2.max())
             )
plt.colorbar()
plt.xticks(xticks)
plt.yticks(yticks)
plt.title(r'$\gamma_2$')
plt.show()

z_gamma2 = best_params[:,:,2]
plt.figure(figsize=(15,5))
plt.contourf(x,y,z_gamma2,
             levels=10,
             cmap='magma',
             )
plt.colorbar()
plt.xticks(xticks)
plt.yticks(yticks)
plt.title(r'$\gamma_2$')
plt.show()


z_alpha = best_params[:,:,3]
plt.figure(figsize=(15,5))
plt.contourf(x,y,z_alpha,levels=10,cmap='magma')
plt.colorbar()
plt.xticks(xticks)
plt.yticks(yticks)
plt.show()

z_alpha = best_params[:,:,3]
plt.figure(figsize=(15,5))
plt.contourf(x,y,z_alpha,levels=10,cmap='seismic')
plt.colorbar()
plt.xticks(xticks)
plt.yticks(yticks)
plt.show()

z_beta = best_params[:,:,4]
plt.figure(figsize=(15,5))
plt.contourf(x,y,z_beta,levels=10,cmap='magma')
plt.colorbar()
plt.xticks(xticks)
plt.yticks(yticks)
plt.show()


z_diff_gamma = abs(z_gamma1-z_gamma2)
plt.figure(figsize=(15,5))
plt.contourf(x,y,z_diff_gamma,
             # levels=10,
             cmap='magma',
             norm=colors.LogNorm(
                 vmin=z_diff_gamma.min(),
                 vmax=z_diff_gamma.max())
             )
plt.colorbar(ticks=[1e-3,1e-2,1e-1,1e0,1e1,1e2])
plt.xticks(xticks)
plt.yticks(yticks)
plt.show()
"""