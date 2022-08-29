#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import os
import pymultinest
from nemesispy.models.TP_profiles import TP_Guillot
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
    """
    Generate log(2-Stream Guillot parameters) from Fourier coefficients
    """
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

def tmap(P, glon, glat, g_plt, T_eq,
    log_kappa, log_gamma, log_f, log_T_int):
    # convert log parameters to parameters
    ka = 10**log_kappa
    ga = 10**log_gamma
    f = 10**log_f
    T_int = 10**log_T_int
    tgrid = np.zeros(())
    for ilat


def Prior(cube,ndim,nparams):
    # log_kappa
    cube[0] = -4 + (2 - (-4)) * cube[0]
    cube[1] = -1 + (1 - (-1)) * cube[1]
    cube[2] = -1 + (1 - (-1)) * cube[2]
    cube[3] = -1 + (1 - (-1)) * cube[3]
    cube[4] = -1 + (1 - (-1)) * cube[4]

    # log_gamma
    cube[5] = -4 + (1 - (-4)) * cube[5]
    cube[6] = -1 + (1 - (-1)) * cube[6]
    cube[7] = -1 + (1 - (-1)) * cube[7]
    cube[8] = -1 + (1 - (-1)) * cube[8]
    cube[9] = -1 + (1 - (-1)) * cube[9]

    # log_f
    cube[10] = -3 + (1 - (-3)) * cube[10]
    cube[11] = -1 + (1 - (-1)) * cube[11]
    cube[12] = -1 + (1 - (-1)) * cube[12]
    cube[13] = -1 + (1 - (-1)) * cube[13]
    cube[14] = -1 + (1 - (-1)) * cube[14]

    # log_T_int
    cube[15] = 2 + (4 - (2)) * cube[15]
    cube[16] = -1 + (1 - (-1)) * cube[16]
    cube[17] = -1 + (1 - (-1)) * cube[17]
    cube[18] = -1 + (1 - (-1)) * cube[18]
    cube[19] = -1 + (1 - (-1)) * cube[19]

    # VMR
    cube[20] = -10 + (-1 - (-10)) * cube[20]
    cube[21] = -10 + (-1 - (-10)) * cube[21]
    cube[22] = -10 + (-1 - (-10)) * cube[22]
    cube[23] = -10 + (-1 - (-10)) * cube[23]
