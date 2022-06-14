# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from models import Model2

# Read GCM data
from process_gcm import (nlon,nlat,xlon,xlat,npv,pv,\
    tmap,h2omap,comap,co2map,ch4map,hemap,h2map,vmrmap,\
    tmap_mod,h2omap_mod,comap_mod,co2map_mod,ch4map_mod,\
    hemap_mod,h2map_mod,vmrmap_mod,phase_grid,\
    kevin_phase_by_wave,kevin_wave_by_phase,\
    pat_phase_by_wave,pat_wave_by_phase,\
    vmrmap_mod_new,tmap_hot)

def calc_T(cube):
    kappa = 10**cube[0]
    gamma1 = 10**cube[1]
    gamma2 = 10**cube[2]
    alpha = cube[3]
    beta = cube[4]
    T_int = 200
    
    Mod = Model2(T_star, R_star, M_plt, R_plt, SMA, P_range, mmw,
                    kappa = kappa,
                    gamma1 = gamma1,
                    gamma2 = gamma2,
                    alpha = alpha,
                    T_irr = beta,
                    T_int = T_int)
    T_model = Mod.temperature()
    # print(T_model)
    return T_model

### Reference Planet Input: WASP 43b
T_star = 4520 # star temperature in K
R_star = 463892759.99999994 # m, 0.6668 * R_SUN
SMA = 2243970000.0 # m, 0.015*AU
M_plt = 3.8951064000000004e+27 # kg
R_plt = 74065.70 * 1e3 # m
gas_id = np.array([  1, 2,  5,  6, 40, 39])
iso_id = np.array([0, 0, 0, 0, 0, 0])
# T_irr = 2055

### Model parameters (focus to pressure range where Transmission WF peaks)
# Pressure range to be fitted (focus to pressure range where Transmission WF peaks)
NLAYER = 20
P_range = np.geomspace(20*1e5,1e-3*1e5,20)

# VMR map used by Pat is smoothed (HOW)
# Hence mmw is constant with longitude, lattitude and altitude
mmw = 3.92945509119087e-27 # kg

### GCM data
ilon=11
ilat=12
T_GCM = tmap_mod[ilon,ilat,:]
T_GCM_interped = np.interp(P_range,pv[::-1],T_GCM[::-1])

index,params = np.loadtxt('chains/{}_{}-stats.dat'.format(ilon,ilat),skiprows=20,unpack=True)


### Plot 2
# plot a random selection of TP profiles from remaining live points
T = calc_T(params)
plt.plot(T,P_range/1e5,lw=1.0, alpha=1, color='#BF3EFF',label='Best fit')
plt.errorbar(T_GCM,pv/1e5,xerr=10,fmt='o',lw=0.8, ms = 2,color='k',mfc='k',label='Data')
plt.errorbar(T_GCM_interped,P_range/1e5,xerr=10,fmt='o',lw=0.8, ms = 2,color='r',mfc='r',label='Smoothed')
plt.xlabel('Temperature [K]',size='x-large')
plt.ylabel('Pressure [bar]',size='x-large')
plt.minorticks_on()
plt.semilogy()
plt.gca().invert_yaxis()
plt.tick_params(length=10,width=1,labelsize='x-large',which='major')
plt.tight_layout()
plt.savefig('BestFit_TP_{}_{}.pdf'.format(ilon,ilat), format='pdf')
plt.show()

