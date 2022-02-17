#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  9 18:43:36 2022

@author: jingxuanyang
"""
import numba
import numpy as np
from numba import jit 

@jit(nopython=True)
def test1():
    a = np.array([1,2],[1,2])
    a = a[::-1,:]
    return a

@jit(nopython=True,
     locals={'a':numba.types.int64[:,:]})
def test2():
    a = np.array([[1,2],[1,2]])
    shape = np.shape(a)
    return shape


@jit(nopython=True)
def test3():
    a = np.array([[1,2],[1,2]])
    shape = a.shape
    return shape

fov_lat_long1 = np.array([[0.0000, 18.4349],
                [35.1970, 11.8202],
                [61.8232, 333.7203],
                [50.5081, 269.6843],
                [0.0000, 251.5651],
                [0.0000, 315.0000]])


lat_coord = np.linspace(0,90,num=10)
lon_coord = np.linspace(0,360,num=5)
nlat = len(lat_coord)
nlon = len(lon_coord )

lat_ax, lon_ax = np.meshgrid(lat_coord, lon_coord, indexing='ij')
nloc = len(lat_coord)*len(lon_coord)
global_model_lat_lon = np.zeros([nloc,2])

iloc=0
for ilat in range(nlat):
    for ilon in range(nlon):
        global_model_lat_lon[iloc,0] = lat_ax[ilat,ilon]
        global_model_lat_lon[iloc,1] = lon_ax[ilat,ilon]
        iloc+=1

def interpolate_to_lat_lon(fov_lat_lon, global_model, global_model_lat_lon):
    """
    fov_lat_lon : a array of [lattitude, longitude]

    Interpolate the input atmospheric model to the field of view averaging
    lattitudes and longitudes

    Snapped the global model for various physical quantities to the chosen
    set of locations on the planet with specified lattitudes and longitudes
    using bilinear interpolation.
    """
    # NLOC x NLAYER at desired lattitude and longitudesy
    NLOCFOV = fov_lat_lon.shape[0] # output

    # add an extra data point for the periodic longitude

    # global_model_lat_lon = np.append(global_model_lat_lon,)

    fov_model_output =  np.zeros(NLOCFOV)

    """make sure there is a point at lon = 0"""
    lat_grid = global_model_lat_lon[:,0]
    lon_grid = global_model_lat_lon[:,1]

    for iloc, location in enumerate(fov_lat_lon):
        lat = location[0]
        lon = location[1]

        if lat > np.max(lat_grid):
            lat = np.max(lat_grid)
        if lat < np.min(lat_grid):
            lat = np.min(lat_grid) + 1e-3
        if lon > np.max(lon_grid):
            lon = np.max(lon_grid)
        if lon < np.min(lon_grid):
            lon = np.min(lon_grid) + 1e-3

        lat_index_hi = np.where(lat_grid >= lat)[0][0]
        lat_index_low = np.where(lat_grid < lat)[0][-1]
        lon_index_hi = np.where(lon_grid >= lon)[0][0]
        lon_index_low = np.where(lon_grid < lon)[0][-1]

        lat_hi = lat_grid[lat_index_hi]
        lat_low = lat_grid[lat_index_low]
        lon_hi = lon_grid[lon_index_hi]
        lon_low = lon_grid[lon_index_low]

        Q11 = global_model[lat_index_low, lon_index_low]
        Q12 = global_model[lat_index_hi, lon_index_low]
        Q22 = global_model[lat_index_hi, lon_index_hi]
        Q21 = global_model[lat_index_low, lon_index_hi]

        fxy1 = (lon_hi-lon)/(lon_hi-lon_low)*Q11 + (lon-lon_low)/(lon_hi-lon_low)*Q21
        fxy2 = (lon_hi-lon)/(lon_hi-lon_low)*Q21 + (lon-lon_low)/(lon_hi-lon_low)*Q22
        fxy = (lat_hi-lat)/(lat_hi-lat_low)*fxy1 + (lat-lat_low)/(lat_hi-lat_low)*fxy2

        fov_model_output[iloc] = fxy


    return fov_model_output
