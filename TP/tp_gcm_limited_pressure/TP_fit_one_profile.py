# -*- coding: utf-8 -*-
import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import pymultinest
from nemesispy.common.constants import R_SUN, R_JUP_E, AMU, AU, M_JUP, R_JUP, SIGMA_SB
from nemesispy.models.models import Model2
from nemesispy.AAwaitlist.utils import calc_mmw
from nemesispy.common.calc_trig import interpvivien_point
"""
Full pressure range fit is not that great. Need to compare simulation output.
Also can chop to sensitive range (20 bar to 1e-3 bar) and see if fit is better.

This is what we do here.
"""
# Read GCM data
from nemesispy.data.gcm.process_gcm import (nlon,nlat,xlon,xlat,npv,pv,\
    tmap,h2omap,comap,co2map,ch4map,hemap,h2map,vmrmap,\
    tmap_mod,h2omap_mod,comap_mod,co2map_mod,ch4map_mod,\
    hemap_mod,h2map_mod,vmrmap_mod,phase_grid,\
    kevin_phase_by_wave,kevin_wave_by_phase,\
    pat_phase_by_wave,pat_wave_by_phase,\
    vmrmap_mod_new,tmap_hot)

### Reference Planet Input: WASP 43b
T_star = 4520 # star temperature in K
R_star = 0.6668 * R_SUN
SMA = 0.015*AU
M_plt = 3.8951064000000004e+27 # kg
R_plt = 74065.70 * 1e3 # m
gas_id = np.array([  1, 2,  5,  6, 40, 39])
iso_id = np.array([0, 0, 0, 0, 0, 0])
# T_irr = 2055

### Model parameters (focus to pressure range where Transmission WF peaks)
P_range = np.array(
        [2.00000000e+06, 1.18757213e+06, 7.05163778e+05, 4.18716424e+05,
       2.48627977e+05, 1.47631828e+05, 8.76617219e+04, 5.20523088e+04,
       3.09079355e+04, 1.83527014e+04, 1.08975783e+04, 6.47083012e+03,
       3.84228874e+03, 2.28149751e+03, 1.35472142e+03, 8.04414701e+02,
       4.77650239e+02, 2.83622055e+02, 1.68410824e+02, 1.00000000e+02])

### GCM data
longitude = 180
lattitude = 0
T_GCM, VMR_GCM = interpvivien_point(xlon=longitude,xlat=lattitude,xp=P_range,
                    vp=pv, vt=tmap, vvmr=vmrmap, mod_lon=xlon, mod_lat=xlat)
mmw = calc_mmw(gas_id,VMR_GCM[0,:]) # VMR map is smoothed and constant
n_params = 5

# Range of TP profile parameters follow Feng et al. 2020
def Prior(cube, ndim, nparams):
    cube[0] = -3. + (2-(-3.))*cube[0] # kappa
    cube[1] = -3. + (2-(-3.))*cube[1] # gamma1
    cube[2] = -3. + (2-(-3.))*cube[2] # gamma2
    cube[3] = 0. + (1.-0.)*cube[3] # alpha
    cube[4] = 0. + (4000.-0.)*cube[4] # beta

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
    T_diff = T_model - T_GCM
    # calculate loglikelihood, = goodness of fit
    yerr = 10
    loglikelihood= -0.5*( np.sum( T_diff**2/yerr**2 ) )
    """
    print(loglikelihood)
    plt.title('Longitude = {} Lattitude = {}'.format(longitude,lattitude))
    plt.plot(T_GCM,P_range/1e5,label='GCM')
    plt.plot(T_model,P_range/1e5,label='1D fit')
    plt.plot
    plt.xlabel('Temperature [K]',size='x-large')
    plt.ylabel('Pressure [bar]',size='x-large')
    plt.semilogy()
    plt.tick_params(length=10,width=1,labelsize='large',which='major')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.legend()
    plt.savefig('TP_fit_one_profile.pdf',format='pdf',dpi=400)
    plt.show()
    """
    return loglikelihood

if __name__ == "__main__":
    if not os.path.isdir("chains"):
        os.mkdir("chains")
    pymultinest.run(LogLikelihood,
                    Prior,
                    n_params,
                    n_live_points=1000,
                    )

"""
plt.title('Longitude = {} Lattitude = {}'.format(longitude,lattitude))
plt.plot(T_GCM,P_range/1e5,label='GCM')
plt.plot(T_model,P_range/1e5,label='1D fit')
# plt.plot
plt.xlabel('Temperature [K]',size='x-large')
plt.ylabel('Pressure [bar]',size='x-large')
plt.semilogy()
plt.tick_params(length=10,width=1,labelsize='large',which='major')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.legend()
plt.savefig('TP_fit_one_profile.pdf',format='pdf',dpi=400)
plt.show()
"""