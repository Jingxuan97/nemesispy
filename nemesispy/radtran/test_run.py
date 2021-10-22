#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import time
import numpy as np
import matplotlib.pyplot as plt
from constants import R_SUN, R_JUP, M_JUP, AU, AMU, ATM
from models import Model2
from path import split,average
from ck import read_kls
from radtran import radiance, blackbody_um

###############################################################################
#                                                                             #
#                               MODEL input                                   #
#                                                                             #
###############################################################################
# Planet/star parameters
T_star = 6000
R_star = 1*R_SUN
M_plt = 1*M_JUP
R_plt = 1*R_JUP
SMA = 0.015*AU
# Atmospheric parameters
NProfile = 100
Nlayer = 20
P_range = np.geomspace(20,1e-3,NProfile)*1e5
mmw = 2*AMU
# PT profile parameters
kappa = 1e-3
gamma1 = 1e-1
gamma2 = 1e-1
alpha = 0.5
T_irr = 1500
# Gas parameters
ID = [1,2,5,6,40,39]
ISO = [0,0,0,0,0,0]
NVMR = len(ID)
VMR = np.zeros((NProfile,NVMR))
VMR_H2O = np.ones(NProfile)*1e-6
VMR_CO2 = np.ones(NProfile)*1e-6
VMR_CO = np.ones(NProfile)*1e-6
VMR_CH4 = np.ones(NProfile)*1e-6
VMR_He = (np.ones(NProfile)-VMR_H2O-VMR_CO2-VMR_CO-VMR_CH4)*0.15
VMR_H2 = VMR_He/0.15*0.85
VMR[:,0] = VMR_H2O
VMR[:,1] = VMR_CO2
VMR[:,2] = VMR_CO
VMR[:,3] = VMR_CH4
VMR[:,4] = VMR_He
VMR[:,5] = VMR_H2
"""
Run model
"""
atm = Model2(T_star, R_star, M_plt, R_plt, SMA, P_range, mmw,
                      kappa, gamma1, gamma2, alpha, T_irr)
H_atm = atm.height()
P_atm = atm.pressure()
T_atm = atm.temperature()
"""
SI units thus far
"""
###############################################################################
#                                                                             #
#                               Layer input                                   #
#                                                                             #
###############################################################################
# Calculate average layer properties
RADIUS = 74065.70e3
H,P,T = H_atm,P_atm,T_atm
VMR = VMR
H_base, P_base = split(H_atm=H, P_atm=P, Nlayer=Nlayer, layer_type=1)
H_layer,P_layer,T_layer,VMR_layer,U_layer,Gas_layer,scale\
    = average(radius=RADIUS, H_atm=H, P_atm=P, T_atm=T, VMR_atm=VMR, ID=ID,
              H_base=H_base, path_angle=0,integration_type=1)
"""
SI units thus far
"""
###############################################################################
#                                                                             #
#                               c-k input                                     #
#                                                                             #
###############################################################################
files1 = ['./data/ktables/h2o','./data/ktables/co2','./data/ktables/co','./data/ktables/ch4']
files2 = ['./data/ktables/H2O_Katy_R1000','./data/ktables/CO2_Katy_R1000',
          './data/ktables/CO_Katy_R1000','./data/ktables/CH4_Katy_R1000']
files3 = ['./data/ktables/H2O_Katy_ARIEL_test','./data/ktables/CO2_Katy_ARIEL_test',
          './data/ktables/CO_Katy_ARIEL_test','./data/ktables/CH4_Katy_ARIEL_test']
gas_id_list, iso_id_list, wave, g_ord, del_g,\
    P_grid, T_grid, k_gas_w_g_p_t = read_kls(files1)
"""
Switch units:
Pressure: atm
Absorber amount (layer integrated): particles/cm^2
Radiance: W cm-2 sr-1 um-1
Wavelength: um
"""
totam = U_layer*1e-4
P_atm = P_atm/ATM
vmr = VMR[:,:4]
"""
Furthermore, scale totam down by 1e-20 to as k tables are scaled by a factor of 1e20.
"""

totam = totam*1e-20
start = time.time()
for i in range(1):
    r = radiance(wave, totam, P_atm, T_atm, vmr, k_gas_w_g_p_t,
                 P_grid, T_grid, g_ord, del_g)
end = time.time()
plt.plot(wave,r,linewidth=0.5,color='k')
plt.title('{} layers'.format(Nlayer))
plt.grid()
plt.xlabel(r'wavelength($\mu$m)')
plt.ylabel(r'radiance(W sr$^{-1}$ cm$^{-2}$ $\mu$m$^{-1}$')
plt.tight_layout()
plt.grid()
plt.plot(np.linspace(0.5,10),blackbody_um(np.linspace(1,5),T_atm[0]),label='Bottom')
plt.plot(np.linspace(0.5,10),blackbody_um(np.linspace(1,5),T_atm[-1]),label='Top')
plt.plot(np.linspace(0.5,10),blackbody_um(np.linspace(1,5),T_atm[int(NProfile/2)]),label='middle')
plt.xlim(0.5,10)
plt.legend()
#plt.savefig('test_hires{}.pdf'.format(NProfile),dpi=400)

print('runtime',end-start)