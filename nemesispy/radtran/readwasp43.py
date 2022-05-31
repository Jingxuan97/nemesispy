#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 27 03:54:19 2022

@author: jingxuanyang
"""
from scipy.interpolate import interp1d
import numpy as np
import matplotlib.pyplot as plt
from nemesispy.radtran.process_gcm import (nlon,nlat,xlon,xlat,npv,pv,\
    tmap,h2omap,comap,co2map,ch4map,hemap,h2map,vmrmap,hvmap,\
    tmap_mod,h2omap_mod,comap_mod,co2map_mod,ch4map_mod,\
    hemap_mod,h2map_mod,vmrmap_mod,hvmap_mod,phase_grid,wave_grid,wasp43_spec,\
    kevin_phase_by_wave,kevin_wave_by_phase,\
    pat_phase_by_wave,pat_wave_by_phase)
    
R_wasp43 = np.loadtxt('wasp43.sol',skiprows=2,max_rows=1) * 1e3
star_wave, star_rad = np.loadtxt('wasp43.sol',skiprows=3,unpack=True)
f_wasp43b = interp1d(star_wave,star_rad)

fil_wave1, fil_weight1 = np.loadtxt('wasp43.fil',skiprows=2,max_rows=98,unpack=True)
fil_wave2, fil_weight2 = np.loadtxt('wasp43.fil',skiprows=102,unpack=True)

spec1 = 0
for iwave,wave in enumerate(fil_wave1):
    spec1 += f_wasp43b(wave)*fil_weight1[iwave]/sum(fil_weight1)

spec2 = 0
for iwave,wave in enumerate(fil_wave2):
    spec2 += f_wasp43b(wave)*fil_weight2[iwave]/sum(fil_weight2)
    

plt.plot(star_wave,star_rad,linewidth=1,label='full spectrum',color='k')
plt.scatter(wave_grid,wasp43_spec,marker='x',color='r',s=10,label='interpolated')
plt.title('WASP43 spectrum')
plt.yscale('log')
plt.xscale('log')
plt.xlabel(r'wavelength (micron)')
plt.ylabel('flux')
plt.legend()
plt.savefig('wasp43spectrum.pdf')