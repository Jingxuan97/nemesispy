#!/usr/local/bin/python3
# -*- coding: utf-8 -*-
import numpy as np
import pymultinest
from nemesispy.models.models import Model2
from nemesispy.data.gcm.process_gcm import nlon,nlat,xlon,xlat

### Reference Planet Input: WASP 43b
T_star = 4520 # star temperature in K
R_star = 463892759.99999994 # m, 0.6668 * R_SUN
SMA = 2243970000.0 # m, 0.015*AU
M_plt = 3.8951064000000004e+27 # kg
R_plt = 74065.70 * 1e3 # m
mmw = 3.92945509119087e-27 # kg
NLAYER = 20
P_range = np.geomspace(20*1e5,1e-3*1e5,NLAYER) # Pa

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

t_retrieved_map = np.zeros((nlon,nlat,NLAYER))

n_params = 5
T_profiles = np.zeros(((nlon*nlat),NLAYER))
for ilon in range(nlon):
    for ilat in range(nlat):
        base = '{}_{}-'.format(ilon,ilat)
        a = pymultinest.Analyzer(
            n_params=n_params,
            outputfiles_basename='chains/{}_{}-'.format(ilon,ilat)
            )
        values = a.get_equal_weighted_posterior()
        samples = values[:, :n_params] # parameter values
        lnprob = values[:, -1] # loglikelihood
        Nsamp = values.shape[0] # number of samples

        NN = 1000
        draws = np.random.randint(len(samples), size=NN) # random number
        xrand = samples[draws, :]
        Tarr1 = np.array([])

        # random sample of TP parameters
        for i in range(NN):
            T = calc_T(xrand[i,:])
            Tarr1 = np.concatenate([Tarr1, T])

        Tarr1=Tarr1.reshape(NN,P_range.shape[0])
        Tmedian=np.zeros(P_range.shape[0])

        for i in range(P_range.shape[0]):
            percentiles=np.percentile(
                Tarr1[:,i],[4.55, 15.9, 50, 84.1, 95.45]
                )
            Tmedian[i]=percentiles[2]

        T_profiles[ilon*31+ilat,:] = Tmedian

np.savetxt('retrieved_TP.txt',T_profiles,delimiter=',')
