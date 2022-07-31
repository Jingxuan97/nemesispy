# -*- coding: utf-8 -*-
# Read GCM data

import os
import numpy as np
import matplotlib.pyplot as plt
import pymultinest
from nemesispy.common.constants import R_SUN, R_JUP_E, AMU, AU, M_JUP, R_JUP, SIGMA_SB
from nemesispy.models.models import Model2
from nemesispy.AAwaitlist.utils import calc_mmw
from nemesispy.data.gcm.process_gcm import (nlon,nlat,xlon,xlat,npv,pv,\
    tmap,h2omap,comap,co2map,ch4map,hemap,h2map,vmrmap,\
    tmap_mod,h2omap_mod,comap_mod,co2map_mod,ch4map_mod,\
    hemap_mod,h2map_mod,vmrmap_mod,phase_grid,\
    kevin_phase_by_wave,kevin_wave_by_phase,\
    pat_phase_by_wave,pat_wave_by_phase,\
    vmrmap_mod_new,tmap_hot)

# def global_TP_1():


for ip,pressure in enumerate(pv):
    
    pressure = pressure/1e5
    
    z = tmap[:,:,ip]
    
    fs = np.sin(xlat/180*np.pi)*90
    
    levels = np.array([ 700,  800,  900, 1000, 1100, 1200, 1300, 1400, 1500, 1600, 1700])
    
    x,y = np.meshgrid(xlon,fs,indexing='ij')
    
    plt.figure(figsize=(15,5))
    
    
    plt.contourf(x,y,z,levels=10) 
    plt.colorbar()
    
    
    
    xticks = np.array([-180, -150, -120,  -90,  -60,  -30,    0,   30,   60,   90,  120,
            150,  180])
    
    yticks_loc = np.sin(np.array([-60, -30,   0,  30,  60])/180*np.pi)*90
    yticks_label = np.array([-60, -30,   0,  30,  60])
    
    plt.xticks(xticks)
    plt.yticks(yticks_loc,yticks_label)
    plt.title('Pressure : {}'.format(pressure))
    plt.savefig('gcm_TP_plots/pressure_{}.pdf'.format(ip))
    
