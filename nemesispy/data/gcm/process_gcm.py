#!/usr/local/bin/python3
# -*- coding: utf-8 -*-
import numpy as np
import os
"""
Can import formatted GCM data with the command
from nemesispy.data.gcm.process_gcm import (nlon,nlat,xlon,xlat,npv,pv,pvmap,\
    tmap,h2omap,comap,co2map,ch4map,hemap,h2map,vmrmap,hvmap,\
    tmap_mod,h2omap_mod,comap_mod,co2map_mod,ch4map_mod,\
    hemap_mod,h2map_mod,vmrmap_mod,hvmap_mod,phase_grid,\
    kevin_phase_by_wave,kevin_wave_by_phase,\
    pat_phase_by_wave,pat_wave_by_phase)
"""
# Read in GCM data stored in process_vivien.txt and process_vivien_mod.txt
nlon = 64
nlat = 32
xlon = np.array([-177.19  , -171.56  , -165.94  , -160.31  , -154.69  , -149.06 ,
                -143.44  , -137.81  , -132.19  , -126.56  , -120.94  , -115.31  ,
                -109.69  , -104.06  ,  -98.438 ,  -92.812 ,  -87.188 ,  -81.562 ,
                -75.938 ,  -70.312 ,  -64.688 ,  -59.062 ,  -53.438 ,  -47.812 ,
                -42.188 ,  -36.562 ,  -30.938 ,  -25.312 ,  -19.688 ,  -14.062 ,
                -8.4375,   -2.8125,    2.8125,    8.4375,   14.062 ,   19.688 ,
                25.312 ,   30.938 ,   36.562 ,   42.188 ,   47.812 ,   53.438 ,
                59.062 ,   64.688 ,   70.312 ,   75.938 ,   81.562 ,   87.188 ,
                92.812 ,   98.438 ,  104.06  ,  109.69  ,  115.31  ,  120.94  ,
                126.56  ,  132.19  ,  137.81  ,  143.44  ,  149.06  ,  154.69  ,
                160.31  ,  165.94  ,  171.56  ,  177.19  ])
xlat = np.array([-87.188 , -81.562 , -75.938 , -70.312 , -64.688 , -59.062 ,
                -53.438 , -47.812 , -42.188 , -36.562 , -30.938 , -25.312 ,
                -19.688 , -14.062 ,  -8.4375,  -2.8125,   2.8125,   8.4375,
                14.062 ,  19.688 ,  25.312 ,  30.938 ,  36.562 ,  42.188 ,
                47.812 ,  53.438 ,  59.062 ,  64.688 ,  70.312 ,  75.938 ,
                81.562 ,  87.188 ])
npv = 53
pv = np.array([1.7064e+02, 1.2054e+02, 8.5152e+01, 6.0152e+01, 4.2492e+01,
                3.0017e+01, 2.1204e+01, 1.4979e+01, 1.0581e+01, 7.4747e+00,
                5.2802e+00, 3.7300e+00, 2.6349e+00, 1.8613e+00, 1.3148e+00,
                9.2882e-01, 6.5613e-01, 4.6350e-01, 3.2742e-01, 2.3129e-01,
                1.6339e-01, 1.1542e-01, 8.1532e-02, 5.7595e-02, 4.0686e-02,
                2.8741e-02, 2.0303e-02, 1.4342e-02, 1.0131e-02, 7.1569e-03,
                5.0557e-03, 3.5714e-03, 2.5229e-03, 1.7822e-03, 1.2589e-03,
                8.8933e-04, 6.2823e-04, 4.4379e-04, 3.1350e-04, 2.2146e-04,
                1.5644e-04, 1.1051e-04, 7.8066e-05, 5.5146e-05, 3.8956e-05,
                2.7519e-05, 1.9440e-05, 1.3732e-05, 9.7006e-06, 6.8526e-06,
                4.8408e-06, 3.4196e-06, 2.4156e-06])*1e5
iskip = 1+1+64+32+1+53

pvmap = np.zeros((nlon,nlat,npv))
for ilon in range(nlon):
    for ilat in range(nlat):
        pvmap[ilon,ilat,:] = pv

tmap = np.zeros((nlon,nlat,npv))
tmap_mod = np.zeros((nlon,nlat,npv))
co2map = np.zeros((nlon,nlat,npv))
co2map_mod = np.zeros((nlon,nlat,npv))
h2map = np.zeros((nlon,nlat,npv))
h2map_mod = np.zeros((nlon,nlat,npv))
hemap = np.zeros((nlon,nlat,npv))
hemap_mod = np.zeros((nlon,nlat,npv))
ch4map = np.zeros((nlon,nlat,npv))
ch4map_mod = np.zeros((nlon,nlat,npv))
comap = np.zeros((nlon,nlat,npv))
comap_mod = np.zeros((nlon,nlat,npv))
h2omap = np.zeros((nlon,nlat,npv))
h2omap_mod = np.zeros((nlon,nlat,npv))

__location__ = os.path.realpath(
    os.path.join(os.getcwd(), os.path.dirname(__file__)))

f = open(os.path.join(__location__,'process_wasp_43b_vivien.txt'))
vivien_gcm = f.read()
f.close()
vivien_gcm = vivien_gcm.split()
vivien_gcm = [float(i) for i in vivien_gcm]

f = open(os.path.join(__location__,'process_wasp_43b_vivien_mod.txt'))
vivien_gcm_mod = f.read()
f.close()
vivien_gcm_mod = vivien_gcm_mod.split()
vivien_gcm_mod = [float(i) for i in vivien_gcm_mod]

for ilon in range(nlon):
    for ilat in range(nlat):
        for ipv in range(npv):
            tmap[ilon,ilat,ipv] = vivien_gcm[iskip]
            h2omap[ilon,ilat,ipv] = vivien_gcm[iskip+6]
            co2map[ilon,ilat,ipv] = vivien_gcm[iskip+1]
            comap[ilon,ilat,ipv] = vivien_gcm[iskip+5]
            ch4map[ilon,ilat,ipv] = vivien_gcm[iskip+4]
            hemap[ilon,ilat,ipv] = vivien_gcm[iskip+3]
            h2map[ilon,ilat,ipv] = vivien_gcm[iskip+2]
            tmap_mod[ilon,ilat,ipv] = vivien_gcm_mod[iskip]
            h2omap_mod[ilon,ilat,ipv] = vivien_gcm_mod[iskip+6]
            co2map_mod[ilon,ilat,ipv] = vivien_gcm_mod[iskip+1]
            comap_mod[ilon,ilat,ipv] = vivien_gcm_mod[iskip+5]
            ch4map_mod[ilon,ilat,ipv] = vivien_gcm_mod[iskip+4]
            hemap_mod[ilon,ilat,ipv] = vivien_gcm_mod[iskip+3]
            h2map_mod[ilon,ilat,ipv] = vivien_gcm_mod[iskip+2]
            iskip+=7

tmap_hot = np.zeros((nlon,nlat,npv))
for ilon in range(nlon):
    for ilat in range(nlat):
        tmap_hot[ilon,ilat,:] = tmap[31,15,:]

vmrmap = np.zeros((nlon,nlat,npv,6))
for ilon in range(nlon):
    for ilat in range(nlat):
        for ipv in range(npv):
            vmrmap[ilon,ilat,ipv,0] = h2omap[ilon,ilat,ipv]
            vmrmap[ilon,ilat,ipv,1] = co2map[ilon,ilat,ipv]
            vmrmap[ilon,ilat,ipv,2] = comap[ilon,ilat,ipv]
            vmrmap[ilon,ilat,ipv,3] = ch4map[ilon,ilat,ipv]
            vmrmap[ilon,ilat,ipv,4] = hemap[ilon,ilat,ipv]
            vmrmap[ilon,ilat,ipv,5] = h2map[ilon,ilat,ipv]

vmrmap_mod = np.zeros((nlon,nlat,npv,6))
for ilon in range(nlon):
    for ilat in range(nlat):
        for ipv in range(npv):
            vmrmap_mod[ilon,ilat,ipv,0] = h2omap_mod[ilon,ilat,ipv]
            vmrmap_mod[ilon,ilat,ipv,1] = co2map_mod[ilon,ilat,ipv]
            vmrmap_mod[ilon,ilat,ipv,2] = comap_mod[ilon,ilat,ipv]
            vmrmap_mod[ilon,ilat,ipv,3] = ch4map_mod[ilon,ilat,ipv]
            vmrmap_mod[ilon,ilat,ipv,4] = hemap_mod[ilon,ilat,ipv]
            vmrmap_mod[ilon,ilat,ipv,5] = h2map_mod[ilon,ilat,ipv]

vmrmap_mod_new = np.zeros((nlon,nlat,npv,6))
for ilon in range(nlon):
    for ilat in range(nlat):
        for ipv in range(npv):
            vmrmap_mod_new[ilon,ilat,ipv,0] = 0.000479650 # h2o
            vmrmap_mod_new[ilon,ilat,ipv,1] = 7.38846e-08 # co2
            vmrmap_mod_new[ilon,ilat,ipv,2] = 0.000464342 # co
            vmrmap_mod_new[ilon,ilat,ipv,3] = 1.32733e-07 # ch4
            vmrmap_mod_new[ilon,ilat,ipv,4] = 0.162329 # He
            vmrmap_mod_new[ilon,ilat,ipv,5] = 0.836727 # H2

gas_id = np.array([  1, 2,  5,  6, 40, 39])
M_plt = 3.8951064000000004e+27 # kg
R_plt = 74065.70 * 1e3  # m

# WASP43b spectrum
wasp43_spec = np.array([3.341320e+25, 3.215455e+25, 3.101460e+25, 2.987110e+25,
       2.843440e+25, 2.738320e+25, 2.679875e+25, 2.598525e+25,
       2.505735e+25, 2.452230e+25, 2.391140e+25, 2.345905e+25,
       2.283720e+25, 2.203690e+25, 2.136015e+25, 1.234010e+24,
       4.422200e+23])

# WASP43b phase curve by (HST/WFC3 + Spitzer), Table 5 Stevenson 2017
wave_grid = np.array([1.1425, 1.1775, 1.2125, 1.2475, 1.2825, 1.3175, 1.3525, 1.3875,
       1.4225, 1.4575, 1.4925, 1.5275, 1.5625, 1.5975, 1.6325, 3.6   ,
       4.5   ])
phase_grid = np.array([ 22.5,  45. ,  67.5,  90. , 112.5, 135. , 157.5, 180. ,
    202.5, 225. , 247.5, 270. , 292.5, 315. , 337.5])
# NPHASE x NWAVE
kevin_phase_by_wave = np.array([
        [[ 6.000e-05,  6.600e-05],
        [ 5.500e-05,  6.100e-05],
        [ 6.600e-05,  5.800e-05],
        [ 8.600e-05,  5.600e-05],
        [ 5.300e-05,  5.700e-05],
        [ 9.000e-05,  5.300e-05],
        [ 2.000e-06,  5.500e-05],
        [ 2.900e-05,  5.200e-05],
        [ 3.500e-05,  5.600e-05],
        [-3.000e-06,  5.600e-05],
        [ 2.400e-05,  5.600e-05],
        [-3.000e-06,  5.500e-05],
        [ 2.200e-05,  5.800e-05],
        [ 4.800e-05,  5.800e-05],
        [ 8.700e-05,  6.300e-05],
        [-1.300e-05,  1.030e-04],
        [ 9.500e-05,  1.330e-04]],

        [[ 1.030e-04,  6.700e-05],
        [ 1.050e-04,  6.100e-05],
        [ 1.250e-04,  5.900e-05],
        [ 1.540e-04,  5.600e-05],
        [ 9.200e-05,  5.700e-05],
        [ 1.440e-04,  5.300e-05],
        [ 1.900e-05,  5.500e-05],
        [ 7.400e-05,  5.200e-05],
        [ 7.100e-05,  5.600e-05],
        [ 3.300e-05,  5.600e-05],
        [ 7.400e-05,  5.600e-05],
        [ 5.000e-05,  5.500e-05],
        [ 9.100e-05,  5.800e-05],
        [ 1.160e-04,  5.800e-05],
        [ 1.670e-04,  6.300e-05],
        [ 2.350e-04,  1.050e-04],
        [ 5.240e-04,  1.330e-04]],

        [[ 1.610e-04,  7.100e-05],
        [ 1.760e-04,  6.500e-05],
        [ 2.010e-04,  6.200e-05],
        [ 2.420e-04,  6.000e-05],
        [ 1.580e-04,  6.100e-05],
        [ 2.190e-04,  5.700e-05],
        [ 6.000e-05,  5.800e-05],
        [ 1.320e-04,  5.500e-05],
        [ 1.240e-04,  6.000e-05],
        [ 9.500e-05,  5.900e-05],
        [ 1.490e-04,  5.900e-05],
        [ 1.290e-04,  5.800e-05],
        [ 1.950e-04,  6.200e-05],
        [ 2.180e-04,  6.100e-05],
        [ 2.830e-04,  6.700e-05],
        [ 7.350e-04,  1.030e-04],
        [ 1.302e-03,  1.360e-04]],

        [[ 2.240e-04,  6.300e-05],
        [ 2.530e-04,  5.800e-05],
        [ 2.780e-04,  5.500e-05],
        [ 3.300e-04,  5.300e-05],
        [ 2.350e-04,  5.400e-05],
        [ 2.980e-04,  5.000e-05],
        [ 1.170e-04,  5.200e-05],
        [ 1.920e-04,  4.800e-05],
        [ 1.810e-04,  5.300e-05],
        [ 1.690e-04,  5.300e-05],
        [ 2.330e-04,  5.200e-05],
        [ 2.210e-04,  5.200e-05],
        [ 3.090e-04,  5.500e-05],
        [ 3.310e-04,  5.400e-05],
        [ 4.130e-04,  6.000e-05],
        [ 1.458e-03,  1.030e-04],
        [ 2.242e-03,  1.340e-04]],

        [[ 2.830e-04,  6.900e-05],
        [ 3.260e-04,  6.400e-05],
        [ 3.470e-04,  6.100e-05],
        [ 4.070e-04,  5.900e-05],
        [ 3.140e-04,  5.900e-05],
        [ 3.730e-04,  5.600e-05],
        [ 1.830e-04,  5.700e-05],
        [ 2.470e-04,  5.400e-05],
        [ 2.360e-04,  5.900e-05],
        [ 2.460e-04,  5.800e-05],
        [ 3.150e-04,  5.800e-05],
        [ 3.110e-04,  5.700e-05],
        [ 4.220e-04,  6.100e-05],
        [ 4.430e-04,  6.000e-05],
        [ 5.390e-04,  6.600e-05],
        [ 2.245e-03,  1.000e-04],
        [ 3.145e-03,  1.190e-04]],

        [[ 3.290e-04,  6.600e-05],
        [ 3.830e-04,  6.100e-05],
        [ 3.960e-04,  5.800e-05],
        [ 4.640e-04,  5.600e-05],
        [ 3.840e-04,  5.700e-05],
        [ 4.330e-04,  5.300e-05],
        [ 2.490e-04,  5.500e-05],
        [ 2.880e-04,  5.100e-05],
        [ 2.810e-04,  5.600e-05],
        [ 3.150e-04,  5.500e-05],
        [ 3.830e-04,  5.500e-05],
        [ 3.880e-04,  5.500e-05],
        [ 5.150e-04,  5.800e-05],
        [ 5.360e-04,  5.700e-05],
        [ 6.440e-04,  6.300e-05],
        [ 2.909e-03,  7.900e-05],
        [ 3.768e-03,  1.030e-04]],

        [[ 3.550e-04,  5.900e-05],
        [ 4.170e-04,  5.300e-05],
        [ 4.190e-04,  5.200e-05],
        [ 4.910e-04,  4.900e-05],
        [ 4.330e-04,  5.000e-05],
        [ 4.670e-04,  4.700e-05],
        [ 3.030e-04,  4.800e-05],
        [ 3.090e-04,  4.400e-05],
        [ 3.090e-04,  4.900e-05],
        [ 3.640e-04,  4.900e-05],
        [ 4.260e-04,  4.800e-05],
        [ 4.390e-04,  4.800e-05],
        [ 5.750e-04,  5.100e-05],
        [ 5.950e-04,  5.100e-05],
        [ 7.080e-04,  5.600e-05],
        [ 3.281e-03,  7.700e-05],
        [ 4.000e-03,  1.030e-04]],

        [[ 3.670e-04,  4.500e-05],
        [ 4.310e-04,  3.900e-05],
        [ 4.140e-04,  3.800e-05],
        [ 4.820e-04,  3.600e-05],
        [ 4.600e-04,  3.700e-05],
        [ 4.730e-04,  3.300e-05],
        [ 3.530e-04,  3.400e-05],
        [ 3.130e-04,  3.000e-05],
        [ 3.200e-04,  3.600e-05],
        [ 3.940e-04,  3.600e-05],
        [ 4.390e-04,  3.300e-05],
        [ 4.580e-04,  3.500e-05],
        [ 5.950e-04,  3.600e-05],
        [ 6.140e-04,  3.700e-05],
        [ 7.320e-04,  4.200e-05],
        [ 3.231e-03,  6.000e-05],
        [ 3.827e-03,  8.400e-05]],

        [[ 3.350e-04,  6.100e-05],
        [ 3.990e-04,  5.500e-05],
        [ 3.750e-04,  5.300e-05],
        [ 4.410e-04,  5.100e-05],
        [ 4.450e-04,  5.200e-05],
        [ 4.440e-04,  4.800e-05],
        [ 3.490e-04,  5.000e-05],
        [ 2.800e-04,  4.600e-05],
        [ 2.980e-04,  5.100e-05],
        [ 3.790e-04,  5.100e-05],
        [ 4.200e-04,  5.000e-05],
        [ 4.370e-04,  5.000e-05],
        [ 5.630e-04,  5.200e-05],
        [ 5.830e-04,  5.200e-05],
        [ 6.910e-04,  5.800e-05],
        [ 2.881e-03,  8.000e-05],
        [ 3.389e-03,  1.030e-04]],

        [[ 2.930e-04,  6.500e-05],
        [ 3.490e-04,  5.900e-05],
        [ 3.160e-04,  5.700e-05],
        [ 3.730e-04,  5.500e-05],
        [ 4.050e-04,  5.500e-05],
        [ 3.910e-04,  5.200e-05],
        [ 3.350e-04,  5.300e-05],
        [ 2.360e-04,  5.000e-05],
        [ 2.620e-04,  5.500e-05],
        [ 3.430e-04,  5.400e-05],
        [ 3.700e-04,  5.400e-05],
        [ 3.840e-04,  5.300e-05],
        [ 4.930e-04,  5.600e-05],
        [ 5.140e-04,  5.600e-05],
        [ 6.100e-04,  6.100e-05],
        [ 2.285e-03,  1.210e-04],
        [ 2.799e-03,  1.090e-04]],

        [[ 2.370e-04,  7.100e-05],
        [ 2.810e-04,  6.500e-05],
        [ 2.420e-04,  6.300e-05],
        [ 2.890e-04,  6.100e-05],
        [ 3.420e-04,  6.100e-05],
        [ 3.170e-04,  5.700e-05],
        [ 2.950e-04,  5.900e-05],
        [ 1.790e-04,  5.600e-05],
        [ 2.110e-04,  6.000e-05],
        [ 2.830e-04,  5.900e-05],
        [ 2.980e-04,  6.000e-05],
        [ 3.070e-04,  5.900e-05],
        [ 3.930e-04,  6.200e-05],
        [ 4.150e-04,  6.200e-05],
        [ 4.970e-04,  6.700e-05],
        [ 1.625e-03,  1.030e-04],
        [ 2.204e-03,  1.330e-04]],

        [[ 1.740e-04,  6.600e-05],
        [ 2.040e-04,  6.100e-05],
        [ 1.640e-04,  5.800e-05],
        [ 2.000e-04,  5.600e-05],
        [ 2.660e-04,  5.700e-05],
        [ 2.380e-04,  5.300e-05],
        [ 2.370e-04,  5.500e-05],
        [ 1.190e-04,  5.100e-05],
        [ 1.540e-04,  5.600e-05],
        [ 2.090e-04,  5.500e-05],
        [ 2.140e-04,  5.500e-05],
        [ 2.150e-04,  5.500e-05],
        [ 2.780e-04,  5.800e-05],
        [ 3.010e-04,  5.700e-05],
        [ 3.680e-04,  6.300e-05],
        [ 1.054e-03,  1.030e-04],
        [ 1.640e-03,  1.340e-04]],

        [[ 1.150e-04,  6.800e-05],
        [ 1.310e-04,  6.300e-05],
        [ 9.600e-05,  6.000e-05],
        [ 1.210e-04,  5.800e-05],
        [ 1.860e-04,  5.800e-05],
        [ 1.620e-04,  5.500e-05],
        [ 1.710e-04,  5.600e-05],
        [ 6.400e-05,  5.300e-05],
        [ 9.900e-05,  5.700e-05],
        [ 1.320e-04,  5.700e-05],
        [ 1.310e-04,  5.700e-05],
        [ 1.240e-04,  5.600e-05],
        [ 1.650e-04,  5.900e-05],
        [ 1.890e-04,  5.900e-05],
        [ 2.420e-04,  6.400e-05],
        [ 6.170e-04,  1.030e-04],
        [ 1.126e-03,  1.340e-04]],

        [[ 6.700e-05,  6.300e-05],
        [ 7.100e-05,  5.700e-05],
        [ 4.500e-05,  5.500e-05],
        [ 6.300e-05,  5.200e-05],
        [ 1.140e-04,  5.300e-05],
        [ 1.000e-04,  5.000e-05],
        [ 1.030e-04,  5.100e-05],
        [ 2.200e-05,  4.800e-05],
        [ 5.200e-05,  5.300e-05],
        [ 6.100e-05,  5.200e-05],
        [ 6.100e-05,  5.200e-05],
        [ 4.500e-05,  5.100e-05],
        [ 6.800e-05,  5.400e-05],
        [ 9.400e-05,  5.400e-05],
        [ 1.350e-04,  5.900e-05],
        [ 2.990e-04,  1.030e-04],
        [ 6.450e-04,  1.330e-04]],

        [[ 4.100e-05,  7.000e-05],
        [ 3.600e-05,  6.400e-05],
        [ 2.200e-05,  6.200e-05],
        [ 3.700e-05,  5.900e-05],
        [ 6.600e-05,  6.000e-05],
        [ 6.700e-05,  5.600e-05],
        [ 4.700e-05,  5.800e-05],
        [ 1.000e-06,  5.500e-05],
        [ 2.500e-05,  5.900e-05],
        [ 1.200e-05,  5.800e-05],
        [ 1.700e-05,  5.900e-05],
        [-5.000e-06,  5.800e-05],
        [ 9.000e-06,  6.100e-05],
        [ 3.600e-05,  6.100e-05],
        [ 7.000e-05,  6.600e-05],
        [ 8.300e-05,  1.030e-04],
        [ 2.470e-04,  1.330e-04]]])
# NWAVE x NPHASE
kevin_wave_by_phase = np.array([
        [[ 6.000e-05,  6.600e-05],
        [ 1.030e-04,  6.700e-05],
        [ 1.610e-04,  7.100e-05],
        [ 2.240e-04,  6.300e-05],
        [ 2.830e-04,  6.900e-05],
        [ 3.290e-04,  6.600e-05],
        [ 3.550e-04,  5.900e-05],
        [ 3.670e-04,  4.500e-05],
        [ 3.350e-04,  6.100e-05],
        [ 2.930e-04,  6.500e-05],
        [ 2.370e-04,  7.100e-05],
        [ 1.740e-04,  6.600e-05],
        [ 1.150e-04,  6.800e-05],
        [ 6.700e-05,  6.300e-05],
        [ 4.100e-05,  7.000e-05]],

        [[ 5.500e-05,  6.100e-05],
        [ 1.050e-04,  6.100e-05],
        [ 1.760e-04,  6.500e-05],
        [ 2.530e-04,  5.800e-05],
        [ 3.260e-04,  6.400e-05],
        [ 3.830e-04,  6.100e-05],
        [ 4.170e-04,  5.300e-05],
        [ 4.310e-04,  3.900e-05],
        [ 3.990e-04,  5.500e-05],
        [ 3.490e-04,  5.900e-05],
        [ 2.810e-04,  6.500e-05],
        [ 2.040e-04,  6.100e-05],
        [ 1.310e-04,  6.300e-05],
        [ 7.100e-05,  5.700e-05],
        [ 3.600e-05,  6.400e-05]],

        [[ 6.600e-05,  5.800e-05],
        [ 1.250e-04,  5.900e-05],
        [ 2.010e-04,  6.200e-05],
        [ 2.780e-04,  5.500e-05],
        [ 3.470e-04,  6.100e-05],
        [ 3.960e-04,  5.800e-05],
        [ 4.190e-04,  5.200e-05],
        [ 4.140e-04,  3.800e-05],
        [ 3.750e-04,  5.300e-05],
        [ 3.160e-04,  5.700e-05],
        [ 2.420e-04,  6.300e-05],
        [ 1.640e-04,  5.800e-05],
        [ 9.600e-05,  6.000e-05],
        [ 4.500e-05,  5.500e-05],
        [ 2.200e-05,  6.200e-05]],

        [[ 8.600e-05,  5.600e-05],
        [ 1.540e-04,  5.600e-05],
        [ 2.420e-04,  6.000e-05],
        [ 3.300e-04,  5.300e-05],
        [ 4.070e-04,  5.900e-05],
        [ 4.640e-04,  5.600e-05],
        [ 4.910e-04,  4.900e-05],
        [ 4.820e-04,  3.600e-05],
        [ 4.410e-04,  5.100e-05],
        [ 3.730e-04,  5.500e-05],
        [ 2.890e-04,  6.100e-05],
        [ 2.000e-04,  5.600e-05],
        [ 1.210e-04,  5.800e-05],
        [ 6.300e-05,  5.200e-05],
        [ 3.700e-05,  5.900e-05]],

        [[ 5.300e-05,  5.700e-05],
        [ 9.200e-05,  5.700e-05],
        [ 1.580e-04,  6.100e-05],
        [ 2.350e-04,  5.400e-05],
        [ 3.140e-04,  5.900e-05],
        [ 3.840e-04,  5.700e-05],
        [ 4.330e-04,  5.000e-05],
        [ 4.600e-04,  3.700e-05],
        [ 4.450e-04,  5.200e-05],
        [ 4.050e-04,  5.500e-05],
        [ 3.420e-04,  6.100e-05],
        [ 2.660e-04,  5.700e-05],
        [ 1.860e-04,  5.800e-05],
        [ 1.140e-04,  5.300e-05],
        [ 6.600e-05,  6.000e-05]],

        [[ 9.000e-05,  5.300e-05],
        [ 1.440e-04,  5.300e-05],
        [ 2.190e-04,  5.700e-05],
        [ 2.980e-04,  5.000e-05],
        [ 3.730e-04,  5.600e-05],
        [ 4.330e-04,  5.300e-05],
        [ 4.670e-04,  4.700e-05],
        [ 4.730e-04,  3.300e-05],
        [ 4.440e-04,  4.800e-05],
        [ 3.910e-04,  5.200e-05],
        [ 3.170e-04,  5.700e-05],
        [ 2.380e-04,  5.300e-05],
        [ 1.620e-04,  5.500e-05],
        [ 1.000e-04,  5.000e-05],
        [ 6.700e-05,  5.600e-05]],

        [[ 2.000e-06,  5.500e-05],
        [ 1.900e-05,  5.500e-05],
        [ 6.000e-05,  5.800e-05],
        [ 1.170e-04,  5.200e-05],
        [ 1.830e-04,  5.700e-05],
        [ 2.490e-04,  5.500e-05],
        [ 3.030e-04,  4.800e-05],
        [ 3.530e-04,  3.400e-05],
        [ 3.490e-04,  5.000e-05],
        [ 3.350e-04,  5.300e-05],
        [ 2.950e-04,  5.900e-05],
        [ 2.370e-04,  5.500e-05],
        [ 1.710e-04,  5.600e-05],
        [ 1.030e-04,  5.100e-05],
        [ 4.700e-05,  5.800e-05]],

        [[ 2.900e-05,  5.200e-05],
        [ 7.400e-05,  5.200e-05],
        [ 1.320e-04,  5.500e-05],
        [ 1.920e-04,  4.800e-05],
        [ 2.470e-04,  5.400e-05],
        [ 2.880e-04,  5.100e-05],
        [ 3.090e-04,  4.400e-05],
        [ 3.130e-04,  3.000e-05],
        [ 2.800e-04,  4.600e-05],
        [ 2.360e-04,  5.000e-05],
        [ 1.790e-04,  5.600e-05],
        [ 1.190e-04,  5.100e-05],
        [ 6.400e-05,  5.300e-05],
        [ 2.200e-05,  4.800e-05],
        [ 1.000e-06,  5.500e-05]],

        [[ 3.500e-05,  5.600e-05],
        [ 7.100e-05,  5.600e-05],
        [ 1.240e-04,  6.000e-05],
        [ 1.810e-04,  5.300e-05],
        [ 2.360e-04,  5.900e-05],
        [ 2.810e-04,  5.600e-05],
        [ 3.090e-04,  4.900e-05],
        [ 3.200e-04,  3.600e-05],
        [ 2.980e-04,  5.100e-05],
        [ 2.620e-04,  5.500e-05],
        [ 2.110e-04,  6.000e-05],
        [ 1.540e-04,  5.600e-05],
        [ 9.900e-05,  5.700e-05],
        [ 5.200e-05,  5.300e-05],
        [ 2.500e-05,  5.900e-05]],

        [[-3.000e-06,  5.600e-05],
        [ 3.300e-05,  5.600e-05],
        [ 9.500e-05,  5.900e-05],
        [ 1.690e-04,  5.300e-05],
        [ 2.460e-04,  5.800e-05],
        [ 3.150e-04,  5.500e-05],
        [ 3.640e-04,  4.900e-05],
        [ 3.940e-04,  3.600e-05],
        [ 3.790e-04,  5.100e-05],
        [ 3.430e-04,  5.400e-05],
        [ 2.830e-04,  5.900e-05],
        [ 2.090e-04,  5.500e-05],
        [ 1.320e-04,  5.700e-05],
        [ 6.100e-05,  5.200e-05],
        [ 1.200e-05,  5.800e-05]],

        [[ 2.400e-05,  5.600e-05],
        [ 7.400e-05,  5.600e-05],
        [ 1.490e-04,  5.900e-05],
        [ 2.330e-04,  5.200e-05],
        [ 3.150e-04,  5.800e-05],
        [ 3.830e-04,  5.500e-05],
        [ 4.260e-04,  4.800e-05],
        [ 4.390e-04,  3.300e-05],
        [ 4.200e-04,  5.000e-05],
        [ 3.700e-04,  5.400e-05],
        [ 2.980e-04,  6.000e-05],
        [ 2.140e-04,  5.500e-05],
        [ 1.310e-04,  5.700e-05],
        [ 6.100e-05,  5.200e-05],
        [ 1.700e-05,  5.900e-05]],

        [[-3.000e-06,  5.500e-05],
        [ 5.000e-05,  5.500e-05],
        [ 1.290e-04,  5.800e-05],
        [ 2.210e-04,  5.200e-05],
        [ 3.110e-04,  5.700e-05],
        [ 3.880e-04,  5.500e-05],
        [ 4.390e-04,  4.800e-05],
        [ 4.580e-04,  3.500e-05],
        [ 4.370e-04,  5.000e-05],
        [ 3.840e-04,  5.300e-05],
        [ 3.070e-04,  5.900e-05],
        [ 2.150e-04,  5.500e-05],
        [ 1.240e-04,  5.600e-05],
        [ 4.500e-05,  5.100e-05],
        [-5.000e-06,  5.800e-05]],

        [[ 2.200e-05,  5.800e-05],
        [ 9.100e-05,  5.800e-05],
        [ 1.950e-04,  6.200e-05],
        [ 3.090e-04,  5.500e-05],
        [ 4.220e-04,  6.100e-05],
        [ 5.150e-04,  5.800e-05],
        [ 5.750e-04,  5.100e-05],
        [ 5.950e-04,  3.600e-05],
        [ 5.630e-04,  5.200e-05],
        [ 4.930e-04,  5.600e-05],
        [ 3.930e-04,  6.200e-05],
        [ 2.780e-04,  5.800e-05],
        [ 1.650e-04,  5.900e-05],
        [ 6.800e-05,  5.400e-05],
        [ 9.000e-06,  6.100e-05]],

        [[ 4.800e-05,  5.800e-05],
        [ 1.160e-04,  5.800e-05],
        [ 2.180e-04,  6.100e-05],
        [ 3.310e-04,  5.400e-05],
        [ 4.430e-04,  6.000e-05],
        [ 5.360e-04,  5.700e-05],
        [ 5.950e-04,  5.100e-05],
        [ 6.140e-04,  3.700e-05],
        [ 5.830e-04,  5.200e-05],
        [ 5.140e-04,  5.600e-05],
        [ 4.150e-04,  6.200e-05],
        [ 3.010e-04,  5.700e-05],
        [ 1.890e-04,  5.900e-05],
        [ 9.400e-05,  5.400e-05],
        [ 3.600e-05,  6.100e-05]],

        [[ 8.700e-05,  6.300e-05],
        [ 1.670e-04,  6.300e-05],
        [ 2.830e-04,  6.700e-05],
        [ 4.130e-04,  6.000e-05],
        [ 5.390e-04,  6.600e-05],
        [ 6.440e-04,  6.300e-05],
        [ 7.080e-04,  5.600e-05],
        [ 7.320e-04,  4.200e-05],
        [ 6.910e-04,  5.800e-05],
        [ 6.100e-04,  6.100e-05],
        [ 4.970e-04,  6.700e-05],
        [ 3.680e-04,  6.300e-05],
        [ 2.420e-04,  6.400e-05],
        [ 1.350e-04,  5.900e-05],
        [ 7.000e-05,  6.600e-05]],

        [[-1.300e-05,  1.030e-04],
        [ 2.350e-04,  1.050e-04],
        [ 7.350e-04,  1.030e-04],
        [ 1.458e-03,  1.030e-04],
        [ 2.245e-03,  1.000e-04],
        [ 2.909e-03,  7.900e-05],
        [ 3.281e-03,  7.700e-05],
        [ 3.231e-03,  6.000e-05],
        [ 2.881e-03,  8.000e-05],
        [ 2.285e-03,  1.210e-04],
        [ 1.625e-03,  1.030e-04],
        [ 1.054e-03,  1.030e-04],
        [ 6.170e-04,  1.030e-04],
        [ 2.990e-04,  1.030e-04],
        [ 8.300e-05,  1.030e-04]],

        [[ 9.500e-05,  1.330e-04],
        [ 5.240e-04,  1.330e-04],
        [ 1.302e-03,  1.360e-04],
        [ 2.242e-03,  1.340e-04],
        [ 3.145e-03,  1.190e-04],
        [ 3.768e-03,  1.030e-04],
        [ 4.000e-03,  1.030e-04],
        [ 3.827e-03,  8.400e-05],
        [ 3.389e-03,  1.030e-04],
        [ 2.799e-03,  1.090e-04],
        [ 2.204e-03,  1.330e-04],
        [ 1.640e-03,  1.340e-04],
        [ 1.126e-03,  1.340e-04],
        [ 6.450e-04,  1.330e-04],
        [ 2.470e-04,  1.330e-04]]])

# Simulated WASP43b phase curve using GCM data, Fig 12 Irwin 2020
# NPHASE x NWAVE
pat_phase_by_wave = np.array([[1.25672e-04, 1.34992e-04, 1.95970e-04, 2.88426e-04, 3.37525e-04,
        2.81610e-04, 1.09969e-04, 7.62626e-05, 7.37120e-05, 8.76705e-05,
        1.14051e-04, 1.57063e-04, 2.17236e-04, 2.93693e-04, 3.60125e-04,
        1.91405e-03, 2.07116e-03],
    [1.49163e-04, 1.60559e-04, 2.21006e-04, 3.10688e-04, 3.60967e-04,
        3.11476e-04, 1.43551e-04, 1.09837e-04, 1.06979e-04, 1.22599e-04,
        1.51721e-04, 1.97449e-04, 2.59552e-04, 3.37193e-04, 4.04127e-04,
        2.13176e-03, 2.46649e-03],
    [1.85813e-04, 2.00869e-04, 2.60147e-04, 3.44923e-04, 3.96261e-04,
        3.56629e-04, 1.97996e-04, 1.65906e-04, 1.62971e-04, 1.80963e-04,
        2.13800e-04, 2.63000e-04, 3.27341e-04, 4.05451e-04, 4.71267e-04,
        2.44223e-03, 3.01733e-03],
    [2.29134e-04, 2.48922e-04, 3.06145e-04, 3.84130e-04, 4.36077e-04,
        4.08640e-04, 2.65755e-04, 2.37722e-04, 2.35328e-04, 2.56039e-04,
        2.92796e-04, 3.45179e-04, 4.11057e-04, 4.87958e-04, 5.50412e-04,
        2.80073e-03, 3.65343e-03],
    [2.67963e-04, 2.92363e-04, 3.46694e-04, 4.17297e-04, 4.69674e-04,
        4.54442e-04, 3.32080e-04, 3.10639e-04, 3.09756e-04, 3.33203e-04,
        3.73411e-04, 4.27761e-04, 4.93513e-04, 5.66249e-04, 6.22436e-04,
        3.14756e-03, 4.27115e-03],
    [2.89849e-04, 3.17008e-04, 3.68431e-04, 4.33839e-04, 4.86811e-04,
        4.79850e-04, 3.76501e-04, 3.62565e-04, 3.64057e-04, 3.89439e-04,
        4.31339e-04, 4.85190e-04, 5.48207e-04, 6.14174e-04, 6.63062e-04,
        3.39246e-03, 4.73849e-03],
    [2.88408e-04, 3.15467e-04, 3.65779e-04, 4.31420e-04, 4.85214e-04,
        4.78569e-04, 3.83556e-04, 3.74725e-04, 3.78473e-04, 4.04097e-04,
        4.44994e-04, 4.95729e-04, 5.54095e-04, 6.14053e-04, 6.58923e-04,
        3.44873e-03, 4.91866e-03],
    [2.63170e-04, 2.87418e-04, 3.39026e-04, 4.10213e-04, 4.64334e-04,
        4.49594e-04, 3.48648e-04, 3.40207e-04, 3.44986e-04, 3.68791e-04,
        4.06155e-04, 4.52554e-04, 5.06887e-04, 5.65207e-04, 6.11310e-04,
        3.27692e-03, 4.71303e-03],
    [2.21381e-04, 2.41130e-04, 2.95554e-04, 3.74965e-04, 4.28586e-04,
        4.00594e-04, 2.83632e-04, 2.71621e-04, 2.76098e-04, 2.96789e-04,
        3.29437e-04, 3.71742e-04, 4.23831e-04, 4.84027e-04, 5.34414e-04,
        2.93325e-03, 4.17340e-03],
    [1.75102e-04, 1.90086e-04, 2.47538e-04, 3.34895e-04, 3.87241e-04,
        3.45205e-04, 2.09415e-04, 1.92082e-04, 1.95451e-04, 2.12776e-04,
        2.40831e-04, 2.79999e-04, 3.31470e-04, 3.95016e-04, 4.50269e-04,
        2.52940e-03, 3.47989e-03],
    [1.36741e-04, 1.47864e-04, 2.07688e-04, 3.00975e-04, 3.51770e-04,
        2.98187e-04, 1.46239e-04, 1.23758e-04, 1.25683e-04, 1.40236e-04,
        1.64821e-04, 2.02150e-04, 2.54087e-04, 3.20991e-04, 3.80074e-04,
        2.16285e-03, 2.79562e-03],
    [1.12196e-04, 1.20876e-04, 1.82223e-04, 2.78842e-04, 3.28067e-04,
        2.66948e-04, 1.03618e-04, 7.69937e-05, 7.75259e-05, 9.01952e-05,
        1.12685e-04, 1.49395e-04, 2.02470e-04, 2.72107e-04, 3.33804e-04,
        1.88662e-03, 2.23514e-03],
    [1.01797e-04, 1.09450e-04, 1.71409e-04, 2.68866e-04, 3.16956e-04,
        2.52611e-04, 8.30065e-05, 5.34678e-05, 5.27795e-05, 6.45169e-05,
        8.63095e-05, 1.23551e-04, 1.78352e-04, 2.50347e-04, 3.13498e-04,
        1.72374e-03, 1.86316e-03],
    [1.00991e-04, 1.08450e-04, 1.70378e-04, 2.67308e-04, 3.14980e-04,
        2.50533e-04, 7.85783e-05, 4.75325e-05, 4.61990e-05, 5.79012e-05,
        7.99991e-05, 1.18052e-04, 1.73893e-04, 2.46852e-04, 3.10724e-04,
        1.67344e-03, 1.70074e-03],
    [1.05597e-04, 1.13355e-04, 1.75116e-04, 2.70937e-04, 3.18727e-04,
        2.55959e-04, 8.30074e-05, 5.07749e-05, 4.89274e-05, 6.10360e-05,
        8.41020e-05, 1.23530e-04, 1.80816e-04, 2.55081e-04, 3.19831e-04,
        1.69739e-03, 1.70232e-03]])
pat_wave_by_phase = np.array([[1.25672e-04, 1.49163e-04, 1.85813e-04, 2.29134e-04, 2.67963e-04,
        2.89849e-04, 2.88408e-04, 2.63170e-04, 2.21381e-04, 1.75102e-04,
        1.36741e-04, 1.12196e-04, 1.01797e-04, 1.00991e-04, 1.05597e-04],
    [1.34992e-04, 1.60559e-04, 2.00869e-04, 2.48922e-04, 2.92363e-04,
        3.17008e-04, 3.15467e-04, 2.87418e-04, 2.41130e-04, 1.90086e-04,
        1.47864e-04, 1.20876e-04, 1.09450e-04, 1.08450e-04, 1.13355e-04],
    [1.95970e-04, 2.21006e-04, 2.60147e-04, 3.06145e-04, 3.46694e-04,
        3.68431e-04, 3.65779e-04, 3.39026e-04, 2.95554e-04, 2.47538e-04,
        2.07688e-04, 1.82223e-04, 1.71409e-04, 1.70378e-04, 1.75116e-04],
    [2.88426e-04, 3.10688e-04, 3.44923e-04, 3.84130e-04, 4.17297e-04,
        4.33839e-04, 4.31420e-04, 4.10213e-04, 3.74965e-04, 3.34895e-04,
        3.00975e-04, 2.78842e-04, 2.68866e-04, 2.67308e-04, 2.70937e-04],
    [3.37525e-04, 3.60967e-04, 3.96261e-04, 4.36077e-04, 4.69674e-04,
        4.86811e-04, 4.85214e-04, 4.64334e-04, 4.28586e-04, 3.87241e-04,
        3.51770e-04, 3.28067e-04, 3.16956e-04, 3.14980e-04, 3.18727e-04],
    [2.81610e-04, 3.11476e-04, 3.56629e-04, 4.08640e-04, 4.54442e-04,
        4.79850e-04, 4.78569e-04, 4.49594e-04, 4.00594e-04, 3.45205e-04,
        2.98187e-04, 2.66948e-04, 2.52611e-04, 2.50533e-04, 2.55959e-04],
    [1.09969e-04, 1.43551e-04, 1.97996e-04, 2.65755e-04, 3.32080e-04,
        3.76501e-04, 3.83556e-04, 3.48648e-04, 2.83632e-04, 2.09415e-04,
        1.46239e-04, 1.03618e-04, 8.30065e-05, 7.85783e-05, 8.30074e-05],
    [7.62626e-05, 1.09837e-04, 1.65906e-04, 2.37722e-04, 3.10639e-04,
        3.62565e-04, 3.74725e-04, 3.40207e-04, 2.71621e-04, 1.92082e-04,
        1.23758e-04, 7.69937e-05, 5.34678e-05, 4.75325e-05, 5.07749e-05],
    [7.37120e-05, 1.06979e-04, 1.62971e-04, 2.35328e-04, 3.09756e-04,
        3.64057e-04, 3.78473e-04, 3.44986e-04, 2.76098e-04, 1.95451e-04,
        1.25683e-04, 7.75259e-05, 5.27795e-05, 4.61990e-05, 4.89274e-05],
    [8.76705e-05, 1.22599e-04, 1.80963e-04, 2.56039e-04, 3.33203e-04,
        3.89439e-04, 4.04097e-04, 3.68791e-04, 2.96789e-04, 2.12776e-04,
        1.40236e-04, 9.01952e-05, 6.45169e-05, 5.79012e-05, 6.10360e-05],
    [1.14051e-04, 1.51721e-04, 2.13800e-04, 2.92796e-04, 3.73411e-04,
        4.31339e-04, 4.44994e-04, 4.06155e-04, 3.29437e-04, 2.40831e-04,
        1.64821e-04, 1.12685e-04, 8.63095e-05, 7.99991e-05, 8.41020e-05],
    [1.57063e-04, 1.97449e-04, 2.63000e-04, 3.45179e-04, 4.27761e-04,
        4.85190e-04, 4.95729e-04, 4.52554e-04, 3.71742e-04, 2.79999e-04,
        2.02150e-04, 1.49395e-04, 1.23551e-04, 1.18052e-04, 1.23530e-04],
    [2.17236e-04, 2.59552e-04, 3.27341e-04, 4.11057e-04, 4.93513e-04,
        5.48207e-04, 5.54095e-04, 5.06887e-04, 4.23831e-04, 3.31470e-04,
        2.54087e-04, 2.02470e-04, 1.78352e-04, 1.73893e-04, 1.80816e-04],
    [2.93693e-04, 3.37193e-04, 4.05451e-04, 4.87958e-04, 5.66249e-04,
        6.14174e-04, 6.14053e-04, 5.65207e-04, 4.84027e-04, 3.95016e-04,
        3.20991e-04, 2.72107e-04, 2.50347e-04, 2.46852e-04, 2.55081e-04],
    [3.60125e-04, 4.04127e-04, 4.71267e-04, 5.50412e-04, 6.22436e-04,
        6.63062e-04, 6.58923e-04, 6.11310e-04, 5.34414e-04, 4.50269e-04,
        3.80074e-04, 3.33804e-04, 3.13498e-04, 3.10724e-04, 3.19831e-04],
    [1.91405e-03, 2.13176e-03, 2.44223e-03, 2.80073e-03, 3.14756e-03,
        3.39246e-03, 3.44873e-03, 3.27692e-03, 2.93325e-03, 2.52940e-03,
        2.16285e-03, 1.88662e-03, 1.72374e-03, 1.67344e-03, 1.69739e-03],
    [2.07116e-03, 2.46649e-03, 3.01733e-03, 3.65343e-03, 4.27115e-03,
        4.73849e-03, 4.91866e-03, 4.71303e-03, 4.17340e-03, 3.47989e-03,
        2.79562e-03, 2.23514e-03, 1.86316e-03, 1.70074e-03, 1.70232e-03]])
