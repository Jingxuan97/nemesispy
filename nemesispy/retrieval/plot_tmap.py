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

def plot_gcm_tmap_contour_ip(ip,tmap,longitude_grid,latitude_grid,pressure_grid,
        foreshorten=True,T_range=(400,2600),nlevels=20,cmap='magma',
        figsize=(11,5),fontsize=28,grid_points=True,title=None,
        figname='gcm_fig.pdf',dpi=400):
    """
    Plot temperature contour at the ip th pressure layer given a GCM.

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

    # print(x,y)
    # read in GCM temperature map at the ip th pressure level
    z = tmap[:,:,ip]

    # plot contours
    plt.figure(figsize=figsize)
    plt.contourf(x,y,z,levels=nlevels,
        vmin=T_range[0],vmax=T_range[1],cmap=cmap)
    # plt.contourf(x,y,z,levels=nlevels,
    #     cmap=cmap)
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

def plot_tmap_contour(P,tmap,longitude_grid,latitude_grid,pressure_grid,
        foreshorten=True,T_range=(400,2600),nlevels=20,cmap='magma',
        xlims=None,ylims=None,
        figsize=(11,5),fontsize=28,grid_points=True,title=None,
        figname='gcm_fig_P.pdf',dpi=400):
    """
    Plot temperature contour at any pressure given a GCM.

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

    # plot limits
    if ylims:
        plt.ylim(ylims)
    if xlims:
        plt.xlim(xlims)

    # title
    if title:
        plt.title(title,fontsize=fontsize)

    # save plot
    plt.tight_layout()
    plt.savefig(figname,dpi=dpi)
    plt.close()

def plot_tmap1(P,title,figname,
    P_grid, lon_grid, lat_grid,
    g_plt,T_eq,
    ck0,ck1,ck2,sk1,sk2,
    cg0,cg1,cg2,sg1,sg2,
    cf0,cf1,cf2,sf1,sf2,
    T_int):

    tp_grid = gen_tmap1(P_grid, lon_grid, lat_grid,
    g_plt,T_eq,
    ck0,ck1,ck2,sk1,sk2,
    cg0,cg1,cg2,sg1,sg2,
    cf0,cf1,cf2,sf1,sf2,
    T_int)

    plot_tmap_contour(P,tp_grid,lon_grid,lat_grid,P_grid,
        title=title,figname=figname)

    pass
