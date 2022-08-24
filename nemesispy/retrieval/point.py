#!/usr/local/bin/python3
# -*- coding: utf-8 -*-
"""
Test retrieval of a single emmission spectrum.
"""
import pymultinest
from nemesispy.data.gcm.process_gcm import (nlon,nlat,xlon,xlat,npv,pv,\
    tmap,h2omap,comap,co2map,ch4map,hemap,h2map,vmrmap,\
    tmap_mod,h2omap_mod,comap_mod,co2map_mod,ch4map_mod,\
    hemap_mod,h2map_mod,vmrmap_mod,phase_grid,\
    kevin_phase_by_wave,kevin_wave_by_phase,\
    pat_phase_by_wave,pat_wave_by_phase,\
    vmrmap_mod_new,tmap_hot)
from nemesispy.common.interpolate_gcm import interpvivien_point
import numpy as np
import matplotlib.pyplot as plt 
import os
from nemesispy.radtran.forward_model import ForwardModel
ktable_path = "/Users/jingxuanyang/Desktop/Workspace/nemesispy2022/nemesispy/data/ktables"
lowres_file_paths = [
    'h2owasp43.kta',
    'co2wasp43.kta',
    'cowasp43.kta',
    'ch4wasp43.kta']
for ipath,path in enumerate(lowres_file_paths):
    lowres_file_paths[ipath] = os.path.join(ktable_path,path)

cia_file_path = "/Users/jingxuanyang/Desktop/Workspace/nemesispy2022/nemesispy/data/cia/"\
    +'exocia_hitran12_200-3800K.tab'

# Forward Model


# Create a single emmision model using Guillot

P_grid = np.geomspace(10*1e5,1e-3*1e5,20)
NP = len(P_grid)
NLAYER = 20
R_plt = 6.9911e7
M_plt = 1.898e27

from nemesispy.models.TP_profiles import TP_Guillot
g_plt = 25 # m s-2
T_eq = 1200 # K
k_IR = 1e-3 # m2 kg-1
gamma = 1e-1
f = 1
TP = TP_Guillot(P_grid,g_plt,T_eq,k_IR,gamma,f,T_int=100)

# Create a single emission spectrum
wave_grid = np.array([1.1425, 1.1775, 1.2125, 1.2475, 1.2825, 1.3175, 1.3525, 1.3875,
       1.4225, 1.4575, 1.4925, 1.5275, 1.5625, 1.5975, 1.6325, 3.6   ,
       4.5   ])
Nwave = len(wave_grid)
"""
wasp43_spec = np.array([3.341320e+25, 3.215455e+25, 3.101460e+25, 2.987110e+25,
       2.843440e+25, 2.738320e+25, 2.679875e+25, 2.598525e+25,
       2.505735e+25, 2.452230e+25, 2.391140e+25, 2.345905e+25,
       2.283720e+25, 2.203690e+25, 2.136015e+25, 1.234010e+24,
       4.422200e+23])
"""

gas_name = np.array(['H2O','CO2','CO','CH4','He','H2'])
gas_id = np.array([  1, 2,  5,  6, 40, 39])
iso_id = np.array([0, 0, 0, 0, 0, 0])

Ngas = len(gas_id)
VMR = np.zeros((NP,Ngas))
VMR[:,0] = np.ones(NP) * 1e-4
VMR[:,1] = np.ones(NP) * 1e-6
VMR[:,2] = np.ones(NP) * 1e-3
VMR[:,3] = np.ones(NP) * 1e-5
VMR[:,4] = (np.ones(NP) - (VMR[:,0]+VMR[:,1]+VMR[:,2]+VMR[:,3]))*0.15
VMR[:,5] = (np.ones(NP) - (VMR[:,0]+VMR[:,1]+VMR[:,2]+VMR[:,3]))*0.85

FM = ForwardModel()
FM.set_planet_model(M_plt,R_plt,gas_id,iso_id,NLAYER)
FM.set_opacity_data(kta_file_paths=lowres_file_paths,cia_file_path=cia_file_path)
spec = FM.calc_point_spectrum_hydro(P_grid,TP,VMR,path_angle=0)


"""
fig, axs = plt.subplots(nrows=1,ncols=3,figsize=(10,5))
axs[0].plot(wave_grid,spec)
axs[1].plot(TP,P_grid/1e5)
axs[1].invert_yaxis()
axs[1].semilogy()

for igas,gas in enumerate(gas_name):
    axs[2].plot(np.log10(VMR[:,igas]),P_grid/1e5,label=gas)
axs[2].invert_yaxis()
# axs[2].semilogx()
axs[2].set_xlim(-10,0.5)
axs[2].semilogy()
plt.tight_layout()
"""

err = np.array([4.28310663e+21, 4.08484701e+21, 4.60117878e+21, 5.29674503e+21,
       5.50567381e+21, 4.76738097e+21, 2.39706377e+21, 1.77036972e+21,
       1.55866912e+21, 1.61794168e+21, 1.78171846e+21, 2.08521066e+21,
       2.37674331e+21, 2.67261609e+21, 2.84343137e+21, 3.69096642e+20,
       1.89003922e+20])

def Prior(cube, ndim, nparams):
    cube[0] = -10.0 + (-1.0+10.0)*cube[0]     # log vmr H2O
    cube[1] = -10.0 + (-1.0+10.0)*cube[1]     # log vmr CO2
    cube[2] = -10.0 + (-1.0+10.0)*cube[2]     # log vmr CO
    cube[3] = -10.0 + (-1.0+10.0)*cube[3]     # log vmr CH4

    cube[4] = -4. + (1.-(-4.))*cube[4]      # log kappa
    cube[5] = -3. + (1.-(-3.))*cube[5]      # log gamma
    cube[6] = 0 + (2-0) * cube[6] # f

def LogLikelihood(cube, ndim, nparams):

    vmr_H2O = 10.0**np.array(cube[0])
    vmr_CO2 = 10.0**np.array(cube[1])
    vmr_CO = 10.0**np.array(cube[2])
    vmr_CH4 = 10.0**np.array(cube[3])

    vmr = np.ones((NP,Ngas))
    vmr[:,0] = np.ones(NP) * 1e-4
    vmr[:,1] = np.ones(NP) * 1e-6
    vmr[:,2] = np.ones(NP) * 1e-3
    vmr[:,3] = np.ones(NP) * 1e-5
    vmr[:,4] = (np.ones(NP) - (vmr[:,0]+vmr[:,1]+vmr[:,2]+vmr[:,3]))*0.15
    vmr[:,5] = (np.ones(NP) - (vmr[:,0]+vmr[:,1]+vmr[:,2]+vmr[:,3]))*0.85

    kappa = 10.0**np.array(cube[4])
    gamma = 10.0**np.array(cube[5])
    f = cube[6]
    tp = TP_Guillot(P_grid,g_plt,T_eq,k_IR,gamma,f,T_int=100)
    print('tp',tp)

    mod = FM.calc_point_spectrum_hydro(P_grid,tp,vmr,path_angle=0)
    print('mod',mod)
    print('spec',spec)
    loglikelihood= -0.5*(np.sum((spec-mod)**2/err**2))
    print('loglikelihood',loglikelihood)

    return loglikelihood


pymultinest.run(
    LogLikelihood = LogLikelihood,
    Prior = Prior,
    n_dims = 7,
    n_live_points = 400,
    sampling_efficiency = 0.8,
    outputfiles_basename = "chains/test1-",
)
