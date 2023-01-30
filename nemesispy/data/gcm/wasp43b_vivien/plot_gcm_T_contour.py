# -*- coding: utf-8 -*-
"""
Plot temperature contours from GCM data at all pressure levels in the GCM
pressure grid.

The plots are foreshortened in the latitudinal direction.
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

# Pressure grid of the GCM
pressure_grid = pv

for ip in range(npv):

    pressure = pressure_grid[ip]
    pressure = pressure/1e5 # convert to unit of bar

    # set up foreshortened latitude coordinates
    fs = np.sin(xlat/180*np.pi)*90
    x,y = np.meshgrid(xlon,fs,indexing='ij')

    # read in GCM temperature map
    z = tmap[:,:,ip]

    plt.figure(figsize=(11,5))
    plt.contourf(x,y,z,levels=20,vmin=400,vmax=2600,cmap='magma')
    plt.colorbar()

    xticks = np.array([-180,  -90,      0,     90,   180])

    # move the y ticks to the foreshortened location
    yticks_loc = np.sin(np.array([-60, -30,   0,  30,  60])/180*np.pi)*90
    yticks_label = np.array([-60, -30,   0,  30,  60])

    plt.xticks(xticks,fontsize=28)
    plt.yticks(yticks_loc,yticks_label,fontsize=28)
    plt.title('Temperature at $P$ = {:.2f} bar'.format(pressure),
        fontsize=28,fontweight="bold")

    plt.scatter(x,np.sin(y/180*np.pi)*90,s=1,marker='x',color='k')
    # plt.xlabel('Longitude [$^\circ$]',fontsize=14)
    # plt.ylabel('Latitude [$^\circ$]',fontsize=14)
    plt.tight_layout()
    plt.savefig('figures/gcm_contour_pressure_{}.pdf'.format(ip),
        dpi=800)
    plt.close()