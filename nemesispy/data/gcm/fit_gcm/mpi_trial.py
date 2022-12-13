# -*- coding: utf-8 -*-
"""
MPI accelerated routine

Use a 1D profile to fit a GCM

"""
print('importing modules')
import sys
import os
import time
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pymultinest

# Read GCM data
from nemesispy.data.gcm.wasp43b_vivien.process_wasp43b_gcm_vivien import (
    nlon,nlat,xlon,xlat,npv,pv,\
    tmap,h2omap,comap,co2map,ch4map,hemap,h2map,vmrmap,\
    tmap_mod,h2omap_mod,comap_mod,co2map_mod,ch4map_mod,\
    hemap_mod,h2map_mod,vmrmap_mod,phase_grid,\
    kevin_phase_by_wave,kevin_wave_by_phase,\
    pat_phase_by_wave,pat_wave_by_phase)
from nemesispy.models.TP_profiles import TP_Guillot
from nemesispy.common.interpolate_gcm import interp_gcm_X
from nemesispy.common.constants import G
print('modules imported')

### GCM grid information
NLON = nlon
NLAT = nlat

###  Pressure range to be fitted
# (focus to pressure range where Transmission WF peaks)
NLAYER = 20
P_range = np.geomspace(20*1e5,1e-3*1e5,20)

### Reference Planet Input: WASP 43b
T_star = 4520 # star temperature in K
R_star = 463892759.99999994 # m, 0.6668 * R_SUN
SMA = 2243970000.0 # m, 0.015*AU
M_plt = 3.8951064000000004e+27 # kg
R_plt = 74065.70 * 1e3 # m
T_eq = T_star * (R_star/SMA)**0.5 / 2**0.5 # equilibrium temperature
g = G*M_plt/R_plt**2 # gravity at reference radius

################################################################################
################################################################################
### MultiNest retrieval set up
n_params = 4
def Prior(cube, ndim, nparams):
    # Range of TP profile parameters follow Feng et al. 2020
    cube[0] = -5. + (3-(-5.))*cube[0] # k_IR
    cube[1] = -2. + (2-(-2.))*cube[1] # gamma
    cube[2] = 0. + (2-(0.))*cube[2] # f
    cube[3] = 10 + (3000-10)*cube[3] # T_int
################################################################################
################################################################################

def divide_jobs(nlon,nlat,nrank):
    """Divide jobs for multiple processors"""
    njobs = (nlon) * (nlat)
    jobs = np.zeros((njobs,2))
    for ilon in range(nlon):
        for ilat in range(nlat):
            jobs[ilon*nlat+ilat,0] = ilon
            jobs[ilon*nlat+ilat,1] = ilat
    partition = np.array_split(jobs,nrank)
    return partition

from mpi4py import MPI
if __name__ == "__main__":
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    print('Process ',rank,' of ',size,' processes')
    try:
        if not os.path.isdir('chains'):
            os.mkdir('chains')
    except FileExistsError:
        pass

    # Divvy up the jobs for the processors
    partition = divide_jobs(nlon=NLON,nlat=NLAT,nrank=size)

    # Jobs assigned to a particular process
    coord_list = partition[rank]
    print('coord_list',coord_list)

    for icoord, coord in enumerate(coord_list):
        # Read GCM and interpolate to right pressure levels
        # Note np.interp need increasing independent variables
        print('icoord',icoord)
        print('coord',coord)
        ilon = int(coord[0])
        ilat = int(coord[1])
        T_GCM = tmap_mod[ilon,ilat,:]
        T_GCM_interped = np.interp(P_range,pv[::-1],T_GCM[::-1])

        def LogLikelihood(cube, ndim, nparams):
            ########################################################
            ########################################################
            # sample the parameter space by drawing from prior range
            k_IR = 10.0**np.array(cube[0])
            gamma = 10.0**np.array(cube[1])
            f = cube[2]
            T_int = cube[3]
            T_model = TP_Guillot(
                P = P_range,
                g_plt = g,
                T_eq = T_eq,
                k_IR = k_IR,
                gamma = gamma,
                f = f,
                T_int = T_int
            )
            ########################################################
            ########################################################
            # calculate loglikelihood
            T_diff = T_model - T_GCM_interped
            err = 5 # Uncertainty in temperture
            loglikelihood= -0.5*(np.sum(T_diff**2/err**2))
            # print('TP\n', T_model)
            # print('params',gamma,k_IR,f,T_int)

            print('rank : ',rank)
            print('ilon : ', ilon, 'ilat : ', ilat)
            print('loglikelihood : ',loglikelihood)
            return loglikelihood

        start_time = time.time()
        pymultinest.run(LogLikelihood,
            Prior,
            n_params,
            n_live_points=400,
            outputfiles_basename='chains/{}_{}-'.format(ilon,ilat),
            use_MPI=False
            )
        end_time = time.time()
        runtime = end_time - start_time
        print('MultiNest runtime = ',runtime)

        iskip = 4 + n_params + 3 + n_params + 3
        index,params \
            = np.loadtxt('chains/{}_{}-stats.dat'.format(ilon,ilat),
                skiprows=iskip,unpack=True)

        k_IR = 10**params[0]
        gamma = 10**params[1]
        f = params[2]
        T_int = params[3]

        best_TP = TP_Guillot(
            P = P_range,
            g_plt = g,
            T_eq = T_eq,
            gamma = gamma,
            k_IR = k_IR,
            f = f,
            T_int = T_int
        )

        plt.plot(T_GCM_interped,P_range/1e5,label='gcm')
        plt.plot(best_TP,P_range/1e5,label='best fit')
        plt.xlabel('Temperature [K]',size='x-large')
        plt.ylabel('Pressure [bar]',size='x-large')
        plt.minorticks_on()
        plt.semilogy()
        plt.gca().invert_yaxis()
        plt.legend()
        plt.title('lon{} lat{}'.format(ilon,ilat))
        plt.tight_layout()
        plt.savefig('TP_fit_lon{}_lat{}'.format(ilon,ilat),dpi=400)
        plt.close()