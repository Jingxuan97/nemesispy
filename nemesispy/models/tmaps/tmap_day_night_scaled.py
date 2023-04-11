#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
from nemesispy.models.TP_profiles import TP_Guillot

def tmap_day_night_scaled(P_grid, lon_grid, lat_grid, g_plt, T_eq,
    scale, phase_offset,
    log_kappa_day, log_gamma_day, log_f_day, T_int_day,
    log_kappa_night, log_gamma_night, log_f_night, T_int_night,
    ):
    # scale hard coded to be between 0.1 and 1
    assert scale <= 1 and scale >=0.1
    # phase_offset hard coded to be between -90 and 90
    assert phase_offset <=90 and phase_offset >= -90

    # scale the exted of the hot region
    east_limit = 90 * scale
    west_limit = -90 * scale

    # set up output array
    nlon = len(lon_grid)
    nlat = len(lat_grid)
    nP = len(P_grid)
    tp_out = np.zeros((nlon,nlat,nP))

    # convert log parameters to parameters
    kappa_day = 10**log_kappa_day
    gamma_day = 10**log_gamma_day
    f_day = 10**log_f_day
    kappa_night = 10**log_kappa_night
    gamma_night = 10**log_gamma_night
    f_night = 10**log_f_night

    # construct 1D tp profile
    tp_day = TP_Guillot(P=P_grid,g_plt=g_plt,T_eq=T_eq,
            k_IR=kappa_day,gamma=gamma_day,f=f_day,
            T_int=T_int_day)
    tp_night = TP_Guillot(P=P_grid,g_plt=g_plt,T_eq=T_eq,
            k_IR=kappa_night,gamma=gamma_night,f=f_night,
            T_int=T_int_night)

    # construct the temperature map
    for ilon,lon in enumerate(lon_grid):
        # dayside
        if lon >= (phase_offset+ west_limit) and lon <= (phase_offset+east_limit):
            for ilat, lat in enumerate(lat_grid):
                tp_out[ilon,ilat,:] = tp_day
        # nightside
        else:
            for ilat, lat in enumerate(lat_grid):
                tp_out[ilon,ilat,:] = tp_night
    return tp_out

def tmap_day_night_scaled_smoothed(P_grid, lon_grid, lat_grid, g_plt, T_eq,
    scale, phase_offset,
    log_kappa_day, log_gamma_day, log_f_day, T_int_day,
    log_kappa_night, log_gamma_night, log_f_night, T_int_night,
    west_term, east_term
    ):
    # scale hard coded to be between 0.1 and 1
    assert scale <= 1 and scale >=0.1
    # phase_offset hard coded to be between -90 and 90
    assert phase_offset <=90 and phase_offset >= -90
    # assert west_term < 100
    # assert east_term < 100

    # scale the exted of the hot region
    east_limit = 90 * scale
    west_limit = 90 * scale

    # set boundaries
    day_west = phase_offset - west_limit + west_term/2
    day_east = phase_offset + east_limit - east_term/2
    term_west = phase_offset - west_limit - west_term/2
    term_east = phase_offset + east_limit + east_term/2

    print('day_west',day_west)
    print('day_east',day_east)
    print('term_west',term_west)
    print('term_east',term_east)
    # set up output array
    nlon = len(lon_grid)
    nlat = len(lat_grid)
    nP = len(P_grid)
    tp_out = np.zeros((nlon,nlat,nP))

    # convert log parameters to parameters
    kappa_day = 10**log_kappa_day
    gamma_day = 10**log_gamma_day
    f_day = 10**log_f_day
    kappa_night = 10**log_kappa_night
    gamma_night = 10**log_gamma_night
    f_night = 10**log_f_night

    # construct 1D tp profile
    tp_day = TP_Guillot(P=P_grid,g_plt=g_plt,T_eq=T_eq,
            k_IR=kappa_day,gamma=gamma_day,f=f_day,
            T_int=T_int_day)
    tp_night = TP_Guillot(P=P_grid,g_plt=g_plt,T_eq=T_eq,
            k_IR=kappa_night,gamma=gamma_night,f=f_night,
            T_int=T_int_night)

    # construct the temperature map
    for ilon,lon in enumerate(lon_grid):
        # dayside
        if lon >= term_west and lon <= term_east:
            if lon >= day_west and lon <= day_east:
                for ilat, lat in enumerate(lat_grid):
                    tp_out[ilon,ilat,:] = tp_day
            elif lon > day_east:
                for ilat, lat in enumerate(lat_grid):
                    tp_out[ilon,ilat,:] \
                        =  tp_day * (term_east-lon)/(east_term)\
                        +  tp_night * (lon-day_east)/(east_term)
                    print(tp_out[ilon,ilat,:])
            elif lon < day_west:
                for ilat, lat in enumerate(lat_grid):
                    tp_out[ilon,ilat,:] \
                        = tp_day * (lon-term_west) /(west_term) \
                        + tp_night * (day_west-lon)/(west_term)
        # nightside
        else:
            for ilat, lat in enumerate(lat_grid):
                tp_out[ilon,ilat,:] = tp_night
    return tp_out