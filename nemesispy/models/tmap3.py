#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import os
import pymultinest
from nemesispy.models.TP_profiles import TP_Guillot
from scipy.special import voigt_profile
from nemesispy.data.gcm.process_gcm import (nlon,nlat,xlon,xlat,npv,pv,\
    tmap,h2omap,comap,co2map,ch4map,hemap,h2map,vmrmap,\
    tmap_mod,h2omap_mod,comap_mod,co2map_mod,ch4map_mod,\
    hemap_mod,h2map_mod,vmrmap_mod,phase_grid,\
    kevin_phase_by_wave,kevin_wave_by_phase,\
    pat_phase_by_wave,pat_wave_by_phase,\
    vmrmap_mod_new,tmap_hot)


def gen(x,
    ck0,ck1,ck2,sk1,sk2,
    cg0,cg1,cg2,sg1,sg2,
    cf0,cf1,cf2,sf1,sf2,
    ct0,ct1,ct2,st1,st2):

    y = x/180*np.pi

    log_kappa = ck0 + ck1 * np.cos(y) + ck2 * np.cos(2*y)\
        + sk1 * np.sin(y) + sk2 * np.sin(2*y)

    log_gamma = cg0 + cg1 * np.cos(y) + cg2 * np.cos(2*y)\
        + sg1 * np.sin(y) + sg2 * np.sin(2*y)

    log_f = cf0 + cf1 * np.cos(y) + cf2 * np.cos(2*y)\
        + sf1 * np.sin(y) + sf2 * np.sin(2*y)

    log_T_int = ct0 + ct1 * np.cos(y) + ct2 * np.cos(2*y)\
        + st1 * np.sin(y) + st2 * np.sin(2*y)

    return log_kappa, log_gamma, log_f, log_T_int

## Load the best fitting 1D model parameters
param_name = [r'log $\gamma$',r'log $\kappa$',r'$f$',r'T$_{int}$']
nparam = len(param_name)
params = np.loadtxt('AAbest_params_Guillot.txt',unpack=True,delimiter=',')
params = params.T

## Assign the parameters to their (longitude,latitude) grid positions
best_params = np.zeros((nlon,nlat,nparam))
for ilon in range(nlon):
    for ilat in range(nlat):
        best_params[ilon,ilat,:] = params[ilon*nlat+ilat,:]

def Prior(cube,ndim,nparams):
    cube[0] = -4 + (2 - (-4)) * cube[0]
    cube[1] = -1 + (1 - (-1)) * cube[1]
    cube[2] = -1 + (1 - (-1)) * cube[2]
    cube[3] = -1 + (1 - (-1)) * cube[3]
    cube[4] = -1 + (1 - (-1)) * cube[4]

    cube[5] = -4 + (1 - (-4)) * cube[5]
    cube[6] = -1 + (1 - (-1)) * cube[6]
    cube[7] = -1 + (1 - (-1)) * cube[7]
    cube[8] = -1 + (1 - (-1)) * cube[8]
    cube[9] = -1 + (1 - (-1)) * cube[9]

    cube[10] = -3 + (1 - (-3)) * cube[10]
    cube[11] = -1 + (1 - (-1)) * cube[11]
    cube[12] = -1 + (1 - (-1)) * cube[12]
    cube[13] = -1 + (1 - (-1)) * cube[13]
    cube[14] = -1 + (1 - (-1)) * cube[14]

    cube[15] = 2 + (4 - (2)) * cube[15]
    cube[16] = -1 + (1 - (-1)) * cube[16]
    cube[17] = -1 + (1 - (-1)) * cube[17]
    cube[18] = -1 + (1 - (-1)) * cube[18]
    cube[19] = -1 + (1 - (-1)) * cube[19]

n_params = 20
if not os.path.isdir('chains'):
    os.mkdir('chains')

def LogLikelihood(cube,ndim,nparams):
    log_kappa, log_gamma, log_f, log_T_int \
        = gen(xlon,
            cube[0],cube[1],cube[2],cube[3],cube[4],
            cube[5],cube[6],cube[7],cube[8],cube[9],
            cube[10],cube[11],cube[12],cube[13],cube[14],
            cube[15],cube[16],cube[17],cube[18],cube[19])
    # print(np.average(log_kappa),np.average(log_gamma),np.average(log_f),
    #     np.average(log_T_int))
    err = np.sum( (log_gamma-data[0])**2/0.1**2  \
        + (log_kappa-data[1])**2/0.1**2 \
        + (10**log_f-data[2])**2/0.05**2 \
        + (10**log_T_int-data[3])**2/50**2)

    loglike = -0.5*err
    print(loglike)
    return loglike

for ilat in range(16):

    ring = (best_params[:,ilat,:] + best_params[:,-ilat-1,:])/2
    data = ring.T

    pymultinest.run(LogLikelihood,
                    Prior,
                    n_params,
                    n_live_points=400,
                    outputfiles_basename='chains/20ilat-{}'.format(ilat)
                    )

    index,cube \
        = np.loadtxt('chains/20ilat-{}stats.dat'.format(ilat),
            skiprows=50,unpack=True)
    log_kappa, log_gamma, log_f, log_T_int \
        = gen(xlon,
            cube[0],cube[1],cube[2],cube[3],cube[4],
            cube[5],cube[6],cube[7],cube[8],cube[9],
            cube[10],cube[11],cube[12],cube[13],cube[14],
            cube[15],cube[16],cube[17],cube[18],cube[19])

    import matplotlib.pyplot as plt

    plt.title('lat={} kappa'.format(-xlat[ilat]))
    plt.scatter(xlon,ring[:,1],s=1,color='k',marker='x')
    plt.plot(xlon,log_kappa,color='r')
    plt.ylim(-4.5,-0.5)
    plt.savefig('20ilat_{}_kappa.pdf'.format(ilat))
    #plt.show()
    plt.close()

    plt.title('lat={} gamma'.format(-xlat[ilat]))
    plt.scatter(xlon,ring[:,0],s=1,color='k',marker='x')
    plt.plot(xlon,log_gamma,color='r')
    plt.ylim(-2,2)
    plt.savefig('20ilat_{}_gamma.pdf'.format(ilat))
    #plt.show()
    plt.close()

    plt.title('lat={} f'.format(-xlat[ilat]))
    plt.scatter(xlon,ring[:,2],s=1,color='k',marker='x')
    plt.plot(xlon,10**log_f,color='r')
    plt.ylim(0,0.43)
    plt.savefig('20ilat_{}_f.pdf'.format(ilat))
    #plt.show()
    plt.close()

    plt.title('lat={} T_int'.format(-xlat[ilat]))
    plt.scatter(xlon,ring[:,3],s=1,color='k',marker='x')
    plt.plot(xlon,10**log_T_int,color='r')
    plt.ylim(0,1500)
    plt.savefig('20ilat_{}_T_int.pdf'.format(ilat))
    #plt.show()
    plt.close()