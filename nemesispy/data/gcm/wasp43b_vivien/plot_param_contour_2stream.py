# -*- coding: utf-8 -*-
"""
Plot the distributions of best fit parameters of the 2-Stream Guillot profile
from the fit to the WASP-43b GCM.
Parameters are {kappa,gamma,f,T_int}
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Read GCM data
from nemesispy.data.gcm.wasp43b_vivien.process_wasp43b_gcm_vivien import (
    nlon,nlat,xlon,xlat,npv,pv,\
    tmap,h2omap,comap,co2map,ch4map,hemap,h2map,vmrmap,\
    tmap_mod,h2omap_mod,comap_mod,co2map_mod,ch4map_mod,\
    hemap_mod,h2map_mod,vmrmap_mod,phase_grid,\
    kevin_phase_by_wave,kevin_wave_by_phase,\
    pat_phase_by_wave,pat_wave_by_phase)

### 2-Stream Guillot profile
param_name = [r'log $\gamma$',r'log $\kappa$',r'$f$',r'T$_{int}$']
nparam = len(param_name)
## Load the best fitting 1D model parameters
params = np.loadtxt('best_fit_params_2_stream_Guillot.txt',
    unpack=True,delimiter=',')
params = params.T

## Assign the parameters to their (longitude,latitude) grid positions
best_params = np.zeros((nlon,nlat,nparam))
for ilon in range(nlon):
    for ilat in range(nlat):
        best_params[ilon,ilat,:] = params[ilon*nlat+ilat,:]

## Plot the best fitting 1D parameters
# set up foreshortened latitude coordinates
fs = np.sin(xlat/180*np.pi)*90
x,y = np.meshgrid(xlon,fs,indexing='ij')
xticks = np.array([-180, -150, -120, -90, -60, -30, 0, 30, 60, 90, 120,
    150, 180])
# move the y ticks to the foreshortened location
yticks_loc = np.sin(np.array([-60, -30, 0, 30, 60])/180*np.pi)*90
yticks_label = np.array([-60, -30, 0, 30, 60])

## Set up multiplot
fig,axs = plt.subplots(
    nrows=5,ncols=1,
    sharex=True,sharey=True,
    figsize=(8.3,11.7),
    dpi=400
)

for iparam,name in enumerate(param_name):
    # contour plot
    z_param = best_params[:,:,iparam]
    im = axs[iparam].contourf(x,y,z_param,
            levels=20,
            cmap='magma',
            vmin=z_param.min(),
            vmax=z_param.max()
            )
    cbar = fig.colorbar(im,ax=axs[iparam])
    # axis setting
    axs[iparam].set_xticks(xticks)
    axs[iparam].set_yticks(yticks_loc,yticks_label)
    axs[iparam].set_title('{}'.format(name),#fontsize='small'
    )
axs[4].axis('off')
fig.tight_layout()
plt.savefig('figures/param_contour_2stream.pdf',dpi=400)