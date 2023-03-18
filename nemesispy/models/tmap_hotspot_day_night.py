#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
from nemesispy.models.TP_profiles import TP_Guillot

def tmap_hotspot_day_night(P_grid, lon_grid, lat_grid, g_plt, T_eq,
    hot_spot_radius, hot_spot_offset,
    log_kappa_hot, log_gamma_hot, log_f_hot, T_int_hot,
    log_kappa_day, log_gamma_day, log_f_day, T_int_day,
    log_kappa_night, log_gamma_night, log_f_night, T_int_night,
    ):
    # scale hard coded to be between 0 and 1
    assert hot_spot_radius <= 90 and hot_spot_radius >=0.
    # phase_offset hard coded to be between -90 and 90
    assert hot_spot_offset <=90 and hot_spot_offset >= -90

    # set up output array
    nlon = len(lon_grid)
    nlat = len(lat_grid)
    nP = len(P_grid)
    tp_out = np.zeros((nlon,nlat,nP))

    # convert log parameters to parameters
    kappa_hot = 10**log_kappa_hot
    gamma_hot = 10**log_gamma_hot
    f_hot = 10**log_f_hot
    kappa_day = 10**log_kappa_day
    gamma_day = 10**log_gamma_day
    f_day = 10**log_f_day
    kappa_night = 10**log_kappa_night
    gamma_night = 10**log_gamma_night
    f_night = 10**log_f_night

    # construct 1D tp profile
    tp_hot = TP_Guillot(P=P_grid,g_plt=g_plt,T_eq=T_eq,
            k_IR=kappa_hot,gamma=gamma_hot,f=f_hot,
            T_int=T_int_hot)
    tp_day = TP_Guillot(P=P_grid,g_plt=g_plt,T_eq=T_eq,
            k_IR=kappa_day,gamma=gamma_day,f=f_day,
            T_int=T_int_day)
    tp_night = TP_Guillot(P=P_grid,g_plt=g_plt,T_eq=T_eq,
            k_IR=kappa_night,gamma=gamma_night,f=f_night,
            T_int=T_int_night)

    # construct the temperature map
    for ilon,lon in enumerate(lon_grid):
        # dayside
        if lon >= -90 and lon <= 90:
            for ilat, lat in enumerate(lat_grid):
                tp_out[ilon,ilat,:] = tp_day
        # nightside
        else:
            for ilat, lat in enumerate(lat_grid):
                tp_out[ilon,ilat,:] = tp_night
        # hotspot
        if hot_spot_radius >0:
            if (lon >= (hot_spot_offset - hot_spot_radius)) and \
                (lon <= (hot_spot_offset + hot_spot_radius)):
                for ilat, lat in enumerate(lat_grid):
                    cos_lon = np.cos((lon-hot_spot_offset)*np.pi/180)
                    cos_lat = np.cos(lat*np.pi/180)
                    cos_zen = np.cos(hot_spot_radius*np.pi/180)
                    if  cos_lon*cos_lat >= cos_zen:
                        tp_out[ilon,ilat,:] = tp_hot
    return tp_out