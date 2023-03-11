#!/usr/bin/env python3
#-*- coding: utf-8 -*-
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import scipy.interpolate as interpolate

def plot_averaged_TP_ilon(ilon,tmap,longitude_grid,latitude_grid,pressure_grid,
        figname='averaged_TP',dpi=400):
    """
    Plot latitudinally-averaged TP profiles from a 3D temperature model,
    using cos(latitude) as weight.
    """

    nlon = len(longitude_grid)
    nlat = len(latitude_grid)
    npress = len(pressure_grid)

    averaged_TP = np.zeros(npress)
    x = 0
    for ilat,lat in enumerate(latitude_grid):
        averaged_TP += tmap[ilon,ilat,:] * np.cos(lat/180*np.pi)
        x += np.cos(lat/180*np.pi)
    averaged_TP = averaged_TP / x
    print(averaged_TP)
    plt.plot(averaged_TP,pressure_grid)
    plt.gca().invert_yaxis()
    plt.yscale('log')
    plt.savefig(figname,dpi=dpi)
    plt.close()
