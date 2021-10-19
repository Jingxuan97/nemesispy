#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import time
import numpy as np
import matplotlib.pyplot as plt
from constants import R_SUN, R_JUP, M_JUP, AU, AMU, ATM
from models import Model2
from path import split,average
from ck import read_kls, interp_k, radiance, blackbody_um, tau_gas

###############################################################################
#                                                                             #
#                               MODEL input                                   #
#                                                                             #
###############################################################################
# Planet/star parameters
T_star = 4520
R_star = 0.6668*R_SUN
M_plt = 2.052*M_JUP
R_plt = 1.036*R_JUP 
SMA = 0.015*AU
# Atmospheric parameters
NProfile = 50
Nlayer = 40
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


VMR_H2O = np.ones(NProfile)*1e-5
VMR_CO2 = np.ones(NProfile)*1e-20
VMR_CO = np.ones(NProfile)*1e-20
VMR_CH4 = np.ones(NProfile)*1e-20
VMR_He = (np.ones(NProfile)-VMR_H2O-VMR_CO2-VMR_CO-VMR_CH4)*0.15
VMR_H2 = VMR_He/0.15*0.85
VMR[:,0] = VMR_H2O
VMR[:,1] = VMR_CO2
VMR[:,2] = VMR_CO
VMR[:,3] = VMR_CH4
VMR[:,4] = VMR_He
VMR[:,5] = VMR_H2
# Run model
atm = Model2(T_star, R_star, M_plt, R_plt, SMA, P_range, mmw,
                      kappa, gamma1, gamma2, alpha, T_irr)
H_atm = atm.height()
P_atm = atm.pressure()
T_atm = atm.temperature()
## SI units thus far
###############################################################################
#                                                                             #
#                               Layer input                                   #
#                                                                             #
###############################################################################
# Calculate average layer properties
H,P,T = H_atm,P_atm,T_atm
VMR = VMR
H_base, P_base = split(H_atm=H, P_atm=P, Nlayer=Nlayer, layer_type=1)
H_layer,P_layer,T_layer,VMR_layer,U_layer,Gas_layer,scale,del_S\
    = average(radius=R_plt, H_atm=H, P_atm=P, T_atm=T, VMR_atm=VMR, ID=ID,
              H_base=H_base, path_angle=0,integration_type=1)

## SI units thus far
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

# Switch units:
# Pressure: atm
# Absorber amount (layer integrated): particles/cm^2
# Radiance: W cm-2 sr-1 um-1
# Wavelength: um
# Furthermore, scale U_layer down by 1e-20 to as k tables are scaled up by 1e20.

U_layer = U_layer*1e-4
P_layer = P_layer/ATM
vmr = VMR[:,:4]
U_layer = U_layer*1e-20


###############################################################################
#                                                                             #
#                               CIA input                                     #
#                                                                             #
###############################################################################


from cia import read_cia, interp_wn_cia, interp_T_cia, tau_cia
TEMPS, KCIA, nu_grid = read_cia('data/cia/exocia_hitran12_200-3800K.tab')
new_kcia = interp_wn_cia(wave,KCIA,nu_grid)
kcia_pair_l_w = interp_T_cia(T_layer,new_kcia,TEMPS)
length = del_S*1e2

"""
tau_cia_w_l = tau_cia(kcia_pair_l_w,U_layer,del_S,0.85,0.15)

tau_w_g_l = tau_gas(k_gas_w_g_p_t, P_layer, T_layer, VMR_layer, U_layer,
            P_grid, T_grid, g_ord, del_g)
"""

s = time.time()
for i in range(1):
    r = radiance(wave, U_layer, P_layer, T_layer, VMR_layer, k_gas_w_g_p_t,
                P_grid, T_grid, g_ord, del_g,
                del_S,kcia_pair_l_w)
e = time.time()
print('compile time + run',e-s)


s = time.time()
for i in range(1):
    r = radiance(wave, U_layer, P_layer, T_layer, VMR_layer, k_gas_w_g_p_t,
                P_grid, T_grid, g_ord, del_g,
                del_S,kcia_pair_l_w)
e = time.time()
print('run time',e-s)


def stellar(filename,wave):
    from scipy.interpolate import interp1d
    r_star = np.loadtxt(filename,skiprows=3,max_rows=4)
    wl, radiance = np.loadtxt(filename,skiprows=4,unpack=True)
    f = interp1d(wl, radiance, kind='linear', fill_value='extrapolate')
    return f(wave)


starfile = './data/stars/wasp43_stellar_newgrav.txt'
wasp = stellar(starfile,wave)


""" # emission without normalising to stellar spectra
factor = 1
emission = r*factor
plt.scatter(wave,emission,linewidth=0.5,color='k',s=10,marker='x')
plt.plot(wave,emission)
plt.title('emission, {} layers'.format(Nlayer))
plt.grid()
plt.xlabel(r'wavelength($\mu$m)')
plt.ylabel(r'radiance(W sr$^{-1}$ cm$^{-2}$ $\mu$m$^{-1}$')
plt.tight_layout()
plt.grid()
plt.plot(wave,blackbody_um(wave,T_atm[0])*factor, label='Bottom Atm')
plt.plot(wave,blackbody_um(wave,T_atm[-1])*factor, label='Top Atm')
#plt.plot(wave,blackbody_um(wave,T_atm[int(NProfile/2)])*emission, label='Mid Atm')
plt.xlim(1,4)
plt.ticklabel_format(axis="y", style="sci", scilimits=(0,0))
plt.legend()
plt.show()
#plt.savefig('a.pdf',dpi=400)
plt.close()
"""

factor = (R_plt*1e2)**2*4*np.pi**2/wasp
emission = r*factor
plt.scatter(wave,emission,linewidth=0.5,color='k',s=10,marker='x')
plt.plot(wave,emission,color='k')
#plt.title('emission, {} layers'.format(Nlayer))
plt.grid()
plt.xlabel(r'wavelength($\mu$m)')
plt.ylabel(r'total radiance(W sr$^{-1}$ $\mu$m$^{-1}$')

#plt.plot(wave,blackbody_um(wave,T_atm[0])*factor, label='Bottom Atm')
#plt.plot(wave,blackbody_um(wave,T_atm[-1])*factor, label='Top Atm')
plt.xlim(1,5)
plt.ylim(0,0.5*1e-2)
plt.ticklabel_format(axis="y", style="sci", scilimits=(0,0))
#plt.legend()
plt.title('Python')
plt.tight_layout()

fortran = [0.10996601E-02,
0.11569251E-02,
0.12590819E-02,
0.13684827E-02,
0.14563075E-02,
0.14141137E-02,
0.95489115E-03,
0.85605256E-03,
0.80826262E-03,
0.85674890E-03,
0.93831104E-03,
0.10224438E-02,
0.10894587E-02,
0.11194258E-02,
0.11034285E-02,
0.42022956E-02,
0.58084177E-02, ]
plt.scatter(wave,fortran,linewidth=0.5,color='b',s=10,marker='x')
plt.plot(wave,fortran,color='b')

plt.savefig('Python.pdf',dpi=400)
plt.show()
plt.close()


"""
factor = R_plt**2*4*np.pi**2
factor = 1
emission = r*factor
plt.scatter(1e4/wave,emission,linewidth=0.5,color='k',s=10,marker='x')
plt.plot(1e4/wave,emission)
plt.title('emission, {} layers'.format(Nlayer))
plt.grid()
plt.xlabel(r'wavenumber(cm-1)')
plt.ylabel(r'radiance(W sr$^{-1}$ cm$^{-2}$ $\mu$m$^{-1}$')
plt.tight_layout()
plt.grid()
plt.plot(1e4/wave,blackbody_um(wave,T_atm[0])*factor, label='Bottom Atm')
plt.plot(1e4/wave,blackbody_um(wave,T_atm[-1])*factor, label='Top Atm')
#plt.plot(wave,blackbody_um(wave,T_atm[int(NProfile/2)])*emission, label='Mid Atm')
#plt.xlim(0.5,7)
plt.ticklabel_format(axis="y", style="sci", scilimits=(0,0))
plt.legend()
plt.yscale('log')
plt.show()
#plt.savefig('a.pdf',dpi=400)
plt.close()
"""

"""
s = time.time()
for i in range(1000):
    r = radiance(wave, U_layer, P_layer, T_layer, VMR_layer, k_gas_w_g_p_t,
                P_grid, T_grid, g_ord, del_g)
e = time.time()
print('old',e-s)
plt.plot(wave,r,linewidth=0.5,color='k')
plt.title('{} layers'.format(Nlayer))
plt.grid()
plt.xlabel(r'wavelength($\mu$m)')
plt.ylabel(r'radiance(W sr$^{-1}$ cm$^{-2}$ $\mu$m$^{-1}$')
plt.tight_layout()
plt.grid()
#plt.plot(np.linspace(0.5,10),blackbody_um(np.linspace(1,5),T_atm[0]),label='Bottom')
#plt.plot(np.linspace(0.5,10),blackbody_um(np.linspace(1,5),T_atm[-1]),label='Top')
#plt.plot(np.linspace(0.5,10),blackbody_um(np.linspace(1,5),T_atm[int(NProfile/2)]),label='middle')
plt.xlim(0.5,10)
plt.legend()
"""
"""
s = time.time()
for i in range(1000):
    r = radiance_new(wave, U_layer, P_layer, T_layer, VMR_layer, k_gas_w_g_p_t,
                P_grid, T_grid, g_ord, del_g)
e = time.time()
print('new',e-s)
plt.scatter(wave,r,linewidth=0.5,color='b',s=10)
plt.title('{} layers'.format(Nlayer))
plt.grid()
plt.xlabel(r'wavelength($\mu$m)')
plt.ylabel(r'radiance(W sr$^{-1}$ cm$^{-2}$ $\mu$m$^{-1}$')
plt.tight_layout()
plt.grid()
#plt.plot(np.linspace(0.5,10),blackbody_um(np.linspace(1,5),T_atm[0]),label='Bottom')
#plt.plot(np.linspace(0.5,10),blackbody_um(np.linspace(1,5),T_atm[-1]),label='Top')
#plt.plot(np.linspace(0.5,10),blackbody_um(np.linspace(1,5),T_atm[int(NProfile/2)]),label='middle')
plt.xlim(0.5,10)
#plt.legend()
"""