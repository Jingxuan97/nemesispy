#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
from nemesispy.models.TP_profiles import TP_Guillot

def tmap_2tp(P_grid, lon_grid, lat_grid, g_plt, T_eq,
    scale, phase_offset,
    log_kappa_day, log_gamma_day, log_f_day, T_int_day,
    log_kappa_night, log_gamma_night, log_f_night, T_int_night,
    ):
    """
    Temperature model consisting of two TP profiles.
    The atmosphere is partitioned (in longitude) into two regions: a dayside
    and a nightside. The dayside longitudinal span is allowed to vary.

    Parameters
    ----------
    P_grid : ndarray
        Pressure grid (in Pa) of the model.
    lon_grid : ndarray
        Longitude grid (in degree) of the model.
        Substellar point is assumed to be at 0.
        Range is [-180,180].
    lat_grid : ndarray
        Latitude grid (in degree) of the model.
        Range is [-90,90].
    g_plt : real
        Gravitational acceleration at the highest pressure in the pressure
        grid.
    T_eq : real
        Temperature corresponding to the stellar flux.
        T_eq = T_star * (R_star/(2*semi_major_axis))**0.5
    scale : real
        Scaling parameter for the longitudinal span of the dayside.
    phase_offset : real
        Central longitude of the dayside
    log_kappa_day : real
        Range [1e-5,1e3]
        Mean absorption coefficient in the thermal wavelengths. (dayside)
    log_gamma_day : real
        Range ~ [1e-3,1e2]
        gamma = k_V/k_IR, ratio of visible to thermal opacities (dayside)
    log_f_day : real
        f parameter (positive), See eqn. (29) in Guillot 2010.
        With f = 1 at the substellar point, f = 1/2 for a
        day-side average and f = 1/4 for whole planet surface average. (dayside)
    T_int_day : real
        Temperature corresponding to the intrinsic heat flux of the planet.
        (dayside)
    log_kappa_night : real
        Same as above definitions but for nightside.
    log_gamma_night : real
        Same as above definitions but for nightside.
    log_f_night : real
        Same as above definitions but for nightside.
    T_int_night : real
        Same as above definitions but for nightside.

    Returns
    -------
    tp_out : ndarray
        Temperature model defined on a (longitude, laitude, pressure) grid.
    """
    # scale hard coded to be between 0.1 and 0.5
    assert scale <= 1.2 and scale >=0.5
    # phase_offset hard coded to be between -45 and 45
    assert phase_offset <=45 and phase_offset >= -45

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