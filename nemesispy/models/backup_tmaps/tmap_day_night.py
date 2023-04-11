#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
from nemesispy.models.TP_profiles import TP_Guillot
"""
Simplest phase curve fitting model: day side and night side with
a parameter telling the phase offset.
"""

def tmap_day_night(P_grid, lon_grid, lat_grid, g_plt, T_eq,
    phase_offset,
    log_kappa_day, log_gamma_day, log_f_day, T_int_day,
    log_kappa_night, log_gamma_night, log_f_night, T_int_night,
    ):
    """_summary_

    Parameters
    ----------
        P_grid (_type_): _description_
        lon_grid (_type_): _description_
        lat_grid (_type_): _description_
        g_plt (_type_): _description_
        T_eq (_type_): _description_
        phase_offset (_type_): _description_
        log_kappa_day (_type_): _description_
        log_gamma_day (_type_): _description_
        log_f_day (_type_): _description_
        T_int_day (_type_): _description_
        log_kappa_night (_type_): _description_
        log_gamma_night (_type_): _description_
        log_f_night (_type_): _description_
        T_int_night (_type_): _description_

    """
    # phase_offset hard coded to be between -90 and 90
    assert phase_offset <=90 and phase_offset >= -90

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
        if lon >= phase_offset-90 and lon <= phase_offset+90:
            for ilat, lat in enumerate(lat_grid):
                tp_out[ilon,ilat,:] = tp_day
        # nightside
        else:
            for ilat, lat in enumerate(lat_grid):
                tp_out[ilon,ilat,:] = tp_night
    return tp_out
