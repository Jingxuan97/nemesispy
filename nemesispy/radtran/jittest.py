#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  9 18:43:36 2022

@author: jingxuanyang
"""
import numba
import numpy as np
from numba import jit

fov_lat_long1 = np.array([[0.0000, 18.4349],
                [35.1970, 11.8202],
                [61.8232, 333.7203],
                [50.5081, 269.6843],
                [0.0000, 251.5651],
                [0.0000, 315.0000]])

lon_coord = np.linspace(0,360,num=5)
lat_coord = np.linspace(0,90,num=10)
nlon = len(lon_coord )
nlat = len(lat_coord)
nmodel = 10
global_T_model = np.ones((nlon,nlat,10))
for ilon in range(nlon):
    if ilon<=1:
        global_T_model[ilon,:,:] *= 1000
    elif ilon == 4:
        global_T_model[ilon,:,:] *= 1000
    else:
        global_T_model[ilon,:,:] *= 0

VMR1 = np.array([[1.000e-04, 0.000e+00, 0.000e+00, 0.000e+00, 0.000e+00, 9.999e-01],
       [1.000e-04, 0.000e+00, 0.000e+00, 0.000e+00, 0.000e+00, 9.999e-01],
       [1.000e-04, 0.000e+00, 0.000e+00, 0.000e+00, 0.000e+00, 9.999e-01],
       [1.000e-04, 0.000e+00, 0.000e+00, 0.000e+00, 0.000e+00, 9.999e-01],
       [1.000e-04, 0.000e+00, 0.000e+00, 0.000e+00, 0.000e+00, 9.999e-01],
       [1.000e-04, 0.000e+00, 0.000e+00, 0.000e+00, 0.000e+00, 9.999e-01],
       [1.000e-04, 0.000e+00, 0.000e+00, 0.000e+00, 0.000e+00, 9.999e-01],
       [1.000e-04, 0.000e+00, 0.000e+00, 0.000e+00, 0.000e+00, 9.999e-01],
       [1.000e-04, 0.000e+00, 0.000e+00, 0.000e+00, 0.000e+00, 9.999e-01],
       [1.000e-04, 0.000e+00, 0.000e+00, 0.000e+00, 0.000e+00, 9.999e-01],
       [1.000e-04, 0.000e+00, 0.000e+00, 0.000e+00, 0.000e+00, 9.999e-01],
       [1.000e-04, 0.000e+00, 0.000e+00, 0.000e+00, 0.000e+00, 9.999e-01],
       [1.000e-04, 0.000e+00, 0.000e+00, 0.000e+00, 0.000e+00, 9.999e-01],
       [1.000e-04, 0.000e+00, 0.000e+00, 0.000e+00, 0.000e+00, 9.999e-01],
       [1.000e-04, 0.000e+00, 0.000e+00, 0.000e+00, 0.000e+00, 9.999e-01],
       [1.000e-04, 0.000e+00, 0.000e+00, 0.000e+00, 0.000e+00, 9.999e-01],
       [1.000e-04, 0.000e+00, 0.000e+00, 0.000e+00, 0.000e+00, 9.999e-01],
       [1.000e-04, 0.000e+00, 0.000e+00, 0.000e+00, 0.000e+00, 9.999e-01],
       [1.000e-04, 0.000e+00, 0.000e+00, 0.000e+00, 0.000e+00, 9.999e-01],
       [1.000e-04, 0.000e+00, 0.000e+00, 0.000e+00, 0.000e+00, 9.999e-01]])

VMR2 = np.array([[1.000e-04, 5.000e-04, 0.000e+00, 5.000e-04, 0.000e+00, 9.999e-01],
       [1.000e-04, 5.000e-04, 0.000e+00, 5.000e-04, 0.000e+00, 9.999e-01],
       [1.000e-04, 5.000e-04, 0.000e+00, 5.000e-04, 0.000e+00, 9.999e-01],
       [1.000e-04, 5.000e-04, 0.000e+00, 5.000e-04, 0.000e+00, 9.999e-01],
       [1.000e-04, 5.000e-04, 0.000e+00, 5.000e-04, 0.000e+00, 9.999e-01],
       [1.000e-04, 5.000e-04, 0.000e+00, 5.000e-04, 0.000e+00, 9.999e-01],
       [1.000e-04, 5.000e-04, 0.000e+00, 5.000e-04, 0.000e+00, 9.999e-01],
       [1.000e-04, 5.000e-04, 0.000e+00, 5.000e-04, 0.000e+00, 9.999e-01],
       [1.000e-04, 5.000e-04, 0.000e+00, 5.000e-04, 0.000e+00, 9.999e-01],
       [1.000e-04, 5.000e-04, 0.000e+00, 5.000e-04, 0.000e+00, 9.999e-01],
       [1.000e-04, 5.000e-04, 0.000e+00, 5.000e-04, 0.000e+00, 9.999e-01],
       [1.000e-04, 5.000e-04, 0.000e+00, 5.000e-04, 0.000e+00, 9.999e-01],
       [1.000e-04, 5.000e-04, 0.000e+00, 5.000e-04, 0.000e+00, 9.999e-01],
       [1.000e-04, 5.000e-04, 0.000e+00, 5.000e-04, 0.000e+00, 9.999e-01],
       [1.000e-04, 5.000e-04, 0.000e+00, 5.000e-04, 0.000e+00, 9.999e-01],
       [1.000e-04, 5.000e-04, 0.000e+00, 5.000e-04, 0.000e+00, 9.999e-01],
       [1.000e-04, 5.000e-04, 0.000e+00, 5.000e-04, 0.000e+00, 9.999e-01],
       [1.000e-04, 5.000e-04, 0.000e+00, 5.000e-04, 0.000e+00, 9.999e-01],
       [1.000e-04, 5.000e-04, 0.000e+00, 5.000e-04, 0.000e+00, 9.999e-01],
       [1.000e-04, 5.000e-04, 0.000e+00, 5.000e-04, 0.000e+00, 9.999e-01]])

H_model = np.array([      0.        ,  103729.52262529,  206324.36567436,
        305647.55240969,  400004.79371999,  488339.78489464,
        570330.13457951,  646803.61932618,  718436.36050467,
        785922.45264359,  851171.53914926,  914444.30061094,
        976483.72639045, 1037900.6392184 , 1099235.68378114,
       1158860.78613342, 1220924.36001225, 1280551.63539084,
       1340929.88568274, 1404644.74126812])

P_model = np.array([2.00000000e+06, 1.18757213e+06, 7.05163778e+05, 4.18716424e+05,
       2.48627977e+05, 1.47631828e+05, 8.76617219e+04, 5.20523088e+04,
       3.09079355e+04, 1.83527014e+04, 1.08975783e+04, 6.47083012e+03,
       3.84228874e+03, 2.28149751e+03, 1.35472142e+03, 8.04414701e+02,
       4.77650239e+02, 2.83622055e+02, 1.68410824e+02, 1.00000000e+02])

T_model = np.array([2294.22992596, 2275.6969865 , 2221.47715244, 2124.54034598,
       1996.03839381, 1854.89105022, 1718.53840097, 1599.14877875,
       1502.97091806, 1431.02161632, 1380.55915963, 1346.97802576,
       1325.4993502 , 1312.138265  , 1303.97869558, 1299.05345022,
       1296.10265374, 1294.34216453, 1293.29484249, 1292.67284094])

global_VMR_model_shape = (nlon,nlat) + VMR1.shape
global_VMR_model = np.ones(global_VMR_model_shape)*1e-4

global_H_model_shape = (nlon,nlat) + H_model.shape
global_H_model = np.ones(global_H_model_shape)*1e-4

global_P_model_shape = (nlon,nlat) + P_model.shape
global_P_model = np.ones(global_P_model_shape)*1e-4

global_T_model_shape = (nlon,nlat) + T_model.shape
global_T_model = np.ones(global_T_model_shape)*1e-4
# for ilon in range(nlon):
#     if ilon<=1:
#         global_VMR_model[ilon,:] = VMR1
#     elif ilon == 4:
#         global_VMR_model[ilon,:] = VMR1
#     else:
#         global_VMR_model[ilon,:] = VMR2

for ilon in range(nlon):
    global_VMR_model[ilon,:] = VMR1
    global_H_model[ilon,:] = H_model
    global_P_model[ilon,:] = P_model
    global_T_model[ilon,:] = T_model

"""
lat_ax, lon_ax = np.meshgrid(lat_coord, lon_coord, indexing='ij')
nloc = len(lat_coord)*len(lon_coord)
global_model_lat_lon = np.zeros([nloc,2])
"""

"""
iloc=0
for ilat in range(nlat):
    for ilon in range(nlon):
        global_model_lat_lon[iloc,0] = lat_ax[ilat,ilon]
        global_model_lat_lon[iloc,1] = lon_ax[ilat,ilon]
        iloc+=1
"""

def interpolate_to_lat_lon(chosen_location, global_model,
    global_model_longitudes, global_model_lattitudes):
    """
    Given a global model of some physical quantity defined at a range of
    locations specified by their longitudes and lattitudes,
    interpolate the model to the desired chosen_locations using bilinear
    interpolation.

    The model at (global_model_longitudes[i],global_model_lattitudes[j])
    is global_model[i,j,:].

    Parameters
    ----------
    chosen_location(NLOCOUT,2) : ndarray
        A array of [lattitude, longitude] at which the global
        model is to be interpolated.
    global_model(NLONIN, NLATIN, NMODEL) : ndarray
        Model defined at the global_model_locations.
        NLATIN x NlONIN x NMODEL
        NMODEL might be a tuple is the model is a 2D array.
    global_model_longitudes(NLONIN) : ndarray
        Longitude grid specifying where the model is define on the planet.
    global_model_lattitudes(NLATIN) : ndarray
        Longitude grid specifying where the model is define on the planet.

    Returns
    -------
    interp_model(NLOCOUT,NMODEL) : ndarray
        Model interpolated to the desired locations.

    """

    # NMODEL is the number of points in the MODEL
    NLONIN, NLATIN = global_model.shape[:2]
    NMODEL = global_model.shape[2:]
    print('global_model',global_model)
    print('NLONIN, NLATIN, NMODEL', NLONIN, NLATIN, NMODEL)
    # add an extra data point for the periodic longitude
    # global_model_location = np.append(global_model_location,)
    # make sure there is a point at lon = 0
    NLOCOUT = chosen_location.shape[0] # number of locations in the output
    print('NLOCOUT',NLOCOUT)

    # Interp MODEL : NLOCOUT x
    interp_model_shape = (NLOCOUT,) + NMODEL
    print('interp_model_shape',interp_model_shape)
    interp_model =  np.zeros(interp_model_shape) # output model

    lon_grid = global_model_longitudes
    lat_grid = global_model_lattitudes
    print('lon_grid',lon_grid)
    print('lat_grid',lat_grid)
    for ilocout, location in enumerate(chosen_location):
        lon = location[1]
        lat = location[0]
        print('lon,lat',lon,lat)

        if lon > np.max(lon_grid):
            lon = np.max(lon_grid)
        if lon <= np.min(lon_grid):
            lon = np.min(lon_grid) + 1e-3
        if lat > np.max(lat_grid):
            lat = np.max(lat_grid)
        if lat <= np.min(lat_grid):
            lat = np.min(lat_grid) + 1e-3

        lon_index_hi = np.where(lon_grid >= lon)[0][0]
        lon_index_low = np.where(lon_grid < lon)[0][-1]
        lat_index_hi = np.where(lat_grid >= lat)[0][0]
        lat_index_low = np.where(lat_grid < lat)[0][-1]

        lon_hi = lon_grid[lon_index_hi]
        lon_low = lon_grid[lon_index_low]
        lat_hi = lat_grid[lat_index_hi]
        lat_low = lat_grid[lat_index_low]

        Q11 = global_model[lon_index_low,lat_index_low,:]
        Q12 = global_model[lon_index_hi,lat_index_low,:]
        Q22 = global_model[lon_index_hi,lat_index_hi,:]
        Q21 = global_model[lon_index_low,lat_index_hi,:]

        fxy1 = (lat_hi-lat)/(lat_hi-lat_low)*Q11 + (lat-lat_low)/(lat_hi-lat_low)*Q21
        fxy2 = (lat_hi-lat)/(lat_hi-lat_low)*Q12 + (lat-lat_low)/(lat_hi-lat_low)*Q22
        fxy = (lon_hi-lon)/(lon_hi-lon_low)*fxy1 + (lon-lon_low)/(lon_hi-lon_low)*fxy2

        interp_model[ilocout,:] = fxy

    return interp_model

interp_T_model = interpolate_to_lat_lon(fov_lat_long1,global_T_model,lon_coord,lat_coord)
print('interp_model',interp_T_model)

interp_VMR_model = interpolate_to_lat_lon(fov_lat_long1,global_VMR_model,lon_coord,lat_coord)
print('interp_model',interp_VMR_model)


def disc_average(FOV_longitudes,FOV_lattitudes,global_models,global_model_longitudes,
    global_model_lattitudes,weights,emission_angles,
    wave_grid):
    """
    Run forward model at averaging positions.
    """
    NFOV = len(FOV_longitudes)
    NWAVE = len(wave_grid)
    disc_averaged_spec = np.zeros(NWAVE)
    for ifov in range(NFOV):
        # interp all models
        interpolate_to_lat_lon(chosen_location, global_model,
            global_model_longitudes, global_model_lattitudes)

        # calculate layer propers

        # run forward model
