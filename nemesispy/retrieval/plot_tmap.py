#!/usr/bin/env python3
#-*- coding: utf-8 -*-
import numpy as np
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import scipy.interpolate as interpolate

# from nemesispy.models.TP_profiles import TP_Guillot
from nemesispy.retrieval.tmap1 import gen_tmap1

# Read GCM data
from nemesispy.data.gcm.wasp43b_vivien.process_wasp43b_gcm_vivien import (
    nlon,nlat,xlon,xlat,npv,pv,\
    tmap,h2omap,comap,co2map,ch4map,hemap,h2map,vmrmap,\
    tmap_mod,h2omap_mod,comap_mod,co2map_mod,ch4map_mod,\
    hemap_mod,h2map_mod,vmrmap_mod,phase_grid,\
    kevin_phase_by_wave,kevin_wave_by_phase,\
    pat_phase_by_wave,pat_wave_by_phase)

### Grid data
NLAYER = 20
P_range = np.geomspace(20*1e5,1e-3*1e5,20)
global_lon_grid = np.array(
    [-175., -170., -165., -160., -155., -150., -145., -140., -135.,
    -130., -125., -120., -115., -110., -105., -100.,  -95.,  -90.,
    -85.,  -80.,  -75.,  -70.,  -65.,  -60.,  -55.,  -50.,  -45.,
    -40.,  -35.,  -30.,  -25.,  -20.,  -15.,  -10.,   -5.,    0.,
        5.,   10.,   15.,   20.,   25.,   30.,   35.,   40.,   45.,
        50.,   55.,   60.,   65.,   70.,   75.,   80.,   85.,   90.,
        95.,  100.,  105.,  110.,  115.,  120.,  125.,  130.,  135.,
    140.,  145.,  150.,  155.,  160.,  165.,  170.,  175.]) # 71
global_lat_grid = np.array(
    [ 5., 10., 15., 20., 25., 30., 35., 40., 45., 50., 55., 60., 65.,
    70., 75., 80., 85.]) # 17

# def plot_contour(pressure)

def plot_gcm_tmap_contour_ip(ip,tmap,longitude_grid,latitude_grid,pressure_grid,
        foreshorten=True,T_range=(400,2600),nlevels=20,cmap='magma',
        figsize=(11,5),fontsize=28,grid_points=True,title=None,
        figname='gcm_fig.pdf',dpi=400):
    """
    Plot temperature contours given a GCM.

    Parameters
    ----------
    ip : int
        Index of the level for plotting.
    longitude_grid : ndarray
        Unit : degree
    latitude_grid : ndarray
        Unit : degree
    pressure_grid : ndarray
        Unit : Pa
    """
    # set up longitude x latitude meshgrid for plotting
    if foreshorten:
        fs = np.sin(latitude_grid/180*np.pi)*90
        x,y = np.meshgrid(longitude_grid,fs,indexing='ij')
    else:
        x,y = np.meshgrid(longitude_grid,latitude_grid,indexing='ij')

    print(x,y)
    # read in GCM temperature map at the ip th pressure level
    z = tmap[:,:,ip]

    # plot contours
    plt.figure(figsize=figsize)
    plt.contourf(x,y,z,levels=nlevels,
        vmin=T_range[0],vmax=T_range[1],cmap=cmap)
    plt.colorbar()

    # set ticks
    xticks = np.array([-180,  -90,      0,     90,   180])
    plt.xticks(xticks,fontsize=28)
    if foreshorten:
        yticks_loc = np.sin(np.array([-60, -30,   0,  30,  60])/180*np.pi)*90
        yticks_label = np.array([-60, -30,   0,  30,  60])
        plt.yticks(yticks_loc,yticks_label,fontsize=fontsize)
    else:
        yticks = np.array([-60, -30,   0,  30,  60])
        plt.xticks(xticks,fontsize=28)

    # plot grid points
    if grid_points:
        if foreshorten:
            plt.scatter(x,np.sin(y/180*np.pi)*90,s=1,marker='x',color='k')
        else:
            plt.scatter(x,y,s=1,marker='x',color='k')

    # title
    if title:
        plt.title(title,fontsize=fontsize)

    # save plot
    plt.tight_layout()
    plt.savefig(figname,dpi=dpi)
    plt.close()

# plot_gcm_tmap_contour_ip(30,tmap,xlon,xlat,pv)

def plot_tmap_contour(P,tmap,longitude_grid,latitude_grid,pressure_grid,
        foreshorten=True,T_range=(400,2600),nlevels=20,cmap='magma',
        figsize=(11,5),fontsize=28,grid_points=True,title=None,
        figname='gcm_fig_P.pdf',dpi=400):

    nlon = len(longitude_grid)
    nlat = len(latitude_grid)

    # set up longitude x latitude meshgrid for plotting
    if foreshorten:
        fs = np.sin(latitude_grid/180*np.pi)*90
        x,y = np.meshgrid(longitude_grid,fs,indexing='ij')
    else:
        x,y = np.meshgrid(longitude_grid,latitude_grid,indexing='ij')

    # set up array for holding the temperature contour
    z = np.zeros((nlon,nlat))

    # interpolate TP profile to P at all grid point
    for ilon in range(nlon):
        for ilat in range(nlat):
            TP = tmap[ilon,ilat,:]
            f = interpolate.interp1d(pressure_grid,TP,fill_value=(TP[0],TP[-1]),
                bounds_error=False)
            z[ilon,ilat] = f(P)

    # plot contours
    plt.figure(figsize=figsize)
    plt.contourf(x,y,z,levels=nlevels,
        vmin=T_range[0],vmax=T_range[1],cmap=cmap)
    plt.colorbar()

    # set ticks
    xticks = np.array([-180,  -90,      0,     90,   180])
    plt.xticks(xticks,fontsize=28)
    if foreshorten:
        yticks_loc = np.sin(np.array([-60, -30,   0,  30,  60])/180*np.pi)*90
        yticks_label = np.array([-60, -30,   0,  30,  60])
        plt.yticks(yticks_loc,yticks_label,fontsize=fontsize)
    else:
        yticks = np.array([-60, -30,   0,  30,  60])
        plt.xticks(xticks,fontsize=28)

    # plot grid points
    if grid_points:
        if foreshorten:
            plt.scatter(x,np.sin(y/180*np.pi)*90,s=1,marker='x',color='k')
        else:
            plt.scatter(x,y,s=1,marker='x',color='k')

    # title
    if title:
        plt.title(title,fontsize=fontsize)

    # save plot
    plt.tight_layout()
    plt.savefig(figname,dpi=dpi)
    plt.close()

# plot_tmap_contour(pv[30],tmap,xlon,xlat,pv)

### Grid data
NLAYER = 20
P_range = np.geomspace(20*1e5,1e-3*1e5,20)
global_lon_grid = np.array(
    [-175., -170., -165., -160., -155., -150., -145., -140., -135.,
    -130., -125., -120., -115., -110., -105., -100.,  -95.,  -90.,
    -85.,  -80.,  -75.,  -70.,  -65.,  -60.,  -55.,  -50.,  -45.,
    -40.,  -35.,  -30.,  -25.,  -20.,  -15.,  -10.,   -5.,    0.,
        5.,   10.,   15.,   20.,   25.,   30.,   35.,   40.,   45.,
        50.,   55.,   60.,   65.,   70.,   75.,   80.,   85.,   90.,
        95.,  100.,  105.,  110.,  115.,  120.,  125.,  130.,  135.,
    140.,  145.,  150.,  155.,  160.,  165.,  170.,  175.]) # 71
global_lat_grid = np.array(
    [ 5., 10., 15., 20., 25., 30., 35., 40., 45., 50., 55., 60., 65.,
    70., 75., 80., 85.])
