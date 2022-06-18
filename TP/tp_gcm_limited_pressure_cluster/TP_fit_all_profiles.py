# -*- coding: utf-8 -*-
import sys
import os
import time
import numpy as np
import matplotlib.pyplot as plt
import pymultinest
# sys.path.append('/Users/jingxuanyang/Desktop/Workspace/nemesispy2022/')
# from nemesispy.radtran.utils import calc_mmw
# from nemesispy.radtran.trig import interpvivien_point
# from nemesispy.data.constants import R_SUN, R_JUP_E, AMU, AU, M_JUP, R_JUP, SIGMA_SB
from models import Model2
"""
Full pressure range fit is not that great. Need to compare simulation output.
Also can chop to sensitive range (20 bar to 1e-3 bar) and see if fit is better.

This is what we do here.
"""
# Read GCM data
from process_gcm import (nlon,nlat,xlon,xlat,npv,pv,\
    tmap,h2omap,comap,co2map,ch4map,hemap,h2map,vmrmap,\
    tmap_mod,h2omap_mod,comap_mod,co2map_mod,ch4map_mod,\
    hemap_mod,h2map_mod,vmrmap_mod,phase_grid,\
    kevin_phase_by_wave,kevin_wave_by_phase,\
    pat_phase_by_wave,pat_wave_by_phase,\
    vmrmap_mod_new,tmap_hot)

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

### MultiNest retrieval set up
n_params = 5
# Range of TP profile parameters follow Feng et al. 2020
def Prior(cube, ndim, nparams):
    cube[0] = -3. + (2-(-3.))*cube[0] # kappa
    cube[1] = -3. + (2-(-3.))*cube[1] # gamma1
    cube[2] = -3. + (2-(-3.))*cube[2] # gamma2
    cube[3] = 0. + (1.-0.)*cube[3] # alpha
    cube[4] = 0. + (4000.-0.)*cube[4] # beta

if __name__ == "__main__":
    if not os.path.isdir('chains'):
        os.mkdir('chains')
    for i in range(nlon):
        for j in range(nlat):
            ilon = i
            ilat = j
            basename = 'lon{}lat{}'.format(ilon,ilat)
            T_GCM = tmap_mod[ilon,ilat,:]
            T_GCM_interped = np.interp(P_range,pv[::-1],T_GCM[::-1])
            # Convert model differences to LogLikelihood
            def LogLikelihood(cube, ndim, nparams):
                # sample the parameter space by drawing retrieved variables from prior range
                kappa = 10.0**np.array(cube[0])
                gamma1 = 10.0**np.array(cube[1])
                gamma2 = 10.0**np.array(cube[2])
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
                T_diff = T_model - T_GCM_interped
                # calculate loglikelihood, = goodness of fit
                yerr = 100
                loglikelihood= -0.5*(np.sum(T_diff**2/yerr**2))
                # print('loglikelihood',loglikelihood)
                return loglikelihood
            # print('P_range',P_range)
            # print('T_GCM',T_GCM)
            # print('T_GCM_interped',T_GCM_interped)
            start_time = time.time()
            pymultinest.run(LogLikelihood,
                            Prior,
                            n_params,
                            n_live_points=400,
                            outputfiles_basename='chains/{}_{}-'.format(ilon,ilat)
                            )
            end_time = time.time()
            runtime = end_time - start_time
            print('MultiNest runtime = ',runtime)
