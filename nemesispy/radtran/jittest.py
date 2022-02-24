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

global_VMR_model_shape = (nlon,nlat) + VMR1.shape
global_VMR_model = np.ones(global_VMR_model_shape)*1e-4
for ilon in range(nlon):
    if ilon<=1:
        global_VMR_model[ilon,:] = VMR1
    elif ilon == 4:
        global_VMR_model[ilon,:] = VMR1
    else:
        global_VMR_model[ilon,:] = VMR2

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
