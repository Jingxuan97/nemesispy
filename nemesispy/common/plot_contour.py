#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Plot:
1)Temperature at the equator as a function of longitude and pressure.
2)Average temperature profile for each longitude, averaged over all latitudes
using cos(latitude) as a weight.
"""
import numpy as np
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from nemesispy.common.interpolate_gcm import interp_gcm_X

def plot_TP_equator(tmap, output_longitudes, output_pressures,
        longitude_grid, latitude_grid, pressure_grid,
        T_range=(400,2600),nlevels=20,cmap='magma',
        figsize=(11,5),fontsize=28, title=None,
        figname='gcm_fig.pdf',dpi=400):
    """
    Temperature at the equator as a function of longitude and pressure.
    tmap must have dimension
    len(longitude_grid) x len(latitude_grid) x pressure_grid
    """
    Npress = len(output_pressures)
    Nlon = len(output_longitudes)
    TPs = np.zeros((Npress,Nlon))
    lat = 0
    for ilon, lon in enumerate(output_longitudes):
        iT = interp_gcm_X(lon,lat,output_pressures,
            gcm_p=pressure_grid,gcm_lon=longitude_grid,gcm_lat=latitude_grid,
            X=tmap, substellar_point_longitude_shift=0)
        TPs[:,ilon] = iT

    x,y = np.meshgrid(longitude_grid,pressure_grid/1e5,indexing='ij')

    z = TPs.T
    plt.contourf(x,y,z,levels=nlevels,
        vmin=T_range[0],vmax=T_range[1],cmap=cmap)
    plt.colorbar()
    plt.gca().invert_yaxis()
    plt.semilogy()
    plt.xlabel('longitude (degree)')
    plt.ylabel('pressure (bar)')
    print(z)
    plt.tight_layout()
    plt.savefig(figname,dpi=dpi)
    plt.close()

def plot_TP_equator_weighted(tmap, output_longitudes, output_pressures,
        longitude_grid, latitude_grid, pressure_grid, cutoff,
        T_range=(400,2600),nlevels=20,cmap='magma',
        figsize=(11,5),fontsize=28, title=None,
        figname='gcm_fig.pdf',dpi=400):

    """
    Temperature at the equator as a function of longitude and pressure.
    Weighted by cos latitude.
    """
    Npress = len(output_pressures)
    Nlon = len(output_longitudes)
    TPs = np.zeros((Npress,Nlon))

    dtr = np.pi/180
    qudrature = np.linspace(0,cutoff,100)
    weight = np.cos(qudrature*dtr) * (qudrature[1]-qudrature[0]) * dtr

    sum_weight = np.sum(weight)
    for ilon, lon in enumerate(output_longitudes):
        for ilat, lat in enumerate(qudrature):
            iT = interp_gcm_X(lon,lat,output_pressures,
                gcm_p=pressure_grid,gcm_lon=longitude_grid,gcm_lat=latitude_grid,
                X=tmap, substellar_point_longitude_shift=0)
            TPs[:,ilon] += iT * weight[ilat]

    TPs = TPs/sum_weight

    x,y = np.meshgrid(longitude_grid,pressure_grid/1e5,indexing='ij')

    z = TPs.T
    plt.contourf(x,y,z,levels=nlevels,
        vmin=T_range[0],vmax=T_range[1],cmap=cmap)
    plt.colorbar()
    plt.gca().invert_yaxis()
    plt.semilogy()
    plt.xlabel('longitude (degree)')
    plt.ylabel('pressure (bar)')
    print(z)
    plt.tight_layout()
    plt.savefig(figname,dpi=dpi)
    plt.close()

def plot_TP_equator_weighted_diff(tmap1, tmap2, output_longitudes, output_pressures,
        longitude_grid, latitude_grid, pressure_grid,
        T_range=(400,2600),nlevels=20,cmap='magma',
        figsize=(11,5),fontsize=28, title=None,
        figname='gcm_fig.pdf',dpi=400):

    """
    Temperature at the equator as a function of longitude and pressure.
    Weighted by cos latitude.
    """
    Npress = len(output_pressures)
    Nlon = len(output_longitudes)
    TPs1 = np.zeros((Npress,Nlon))
    TPs2 = np.zeros((Npress,Nlon))

    dtr = np.pi/180
    qudrature = np.linspace(0,50,60)
    weight = np.cos(qudrature*dtr) * (qudrature[1]-qudrature[0]) * dtr
    sum_weight = np.sum(weight)

    ### tmap1
    for ilon, lon in enumerate(output_longitudes):
        for ilat, lat in enumerate(qudrature):
            iT = interp_gcm_X(lon,lat,output_pressures,
                gcm_p=pressure_grid,gcm_lon=longitude_grid,gcm_lat=latitude_grid,
                X=tmap1, substellar_point_longitude_shift=0)
            TPs1[:,ilon] += iT * weight[ilat]

    TPs1 = TPs1/sum_weight

    ### tmap2
    for ilon, lon in enumerate(output_longitudes):
        for ilat, lat in enumerate(qudrature):
            iT = interp_gcm_X(lon,lat,output_pressures,
                gcm_p=pressure_grid,gcm_lon=longitude_grid,gcm_lat=latitude_grid,
                X=tmap2, substellar_point_longitude_shift=0)
            TPs2[:,ilon] += iT * weight[ilat]

    TPs2 = TPs2/sum_weight

    diff = TPs1 - TPs2

    x,y = np.meshgrid(longitude_grid,pressure_grid/1e5,indexing='ij')

    z = diff.T
    plt.contourf(x,y,z,levels=nlevels,
        vmin=T_range[0],vmax=T_range[1],cmap=cmap)
    plt.colorbar()
    plt.gca().invert_yaxis()
    plt.semilogy()
    plt.xlabel('longitude (degree)')
    plt.ylabel('pressure (bar)')
    print(z)
    plt.tight_layout()
    plt.savefig(figname,dpi=dpi)
    plt.close()