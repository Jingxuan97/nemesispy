#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
from nemesispy.models.TP_profiles import TP_Guillot

def tmap_cos_guillot(P_grid, lon_grid, lat_grid, g_plt, T_eq,
    phase_offset,
    log_kappa_day, log_gamma_day, log_f_day, T_int_day,
    log_kappa_night, log_gamma_night, log_f_night, T_int_night):
    """
    9 parameters
    """
    # phase_offset hard coded to be between -45 and 45
    assert phase_offset <=45 and phase_offset >= -45

    dtr = np.pi/180

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
        tp = 0.5*(tp_day+tp_night) \
            + 0.5*(tp_day-tp_night) * np.cos((lon-phase_offset)*dtr)
        for ilat,lat in enumerate(lat_grid):
            tp_out[ilon,ilat,:] = tp

    return tp_out

def tmap_cos_flat_guillot(P_grid, lon_grid, lat_grid, g_plt, T_eq,
    scale, phase_offset,
    log_kappa_day, log_gamma_day, log_f_day, T_int_day,
    log_kappa_night, log_gamma_night, log_f_night, T_int_night):
    """
    10 parameters
    """
    # phase_offset hard coded to be between -45 and 45
    assert phase_offset <=45 and phase_offset >= -45
    assert scale <=1.2 and scale >=0.5
    dtr = np.pi/180
    # boundaries
    bound_east = phase_offset + 90 * scale
    bound_west = phase_offset - 90 * scale

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

    for ilon,lon in enumerate(lon_grid):
        if lon < bound_west or lon > bound_east:
            for ilat,lat in enumerate(lat_grid):
                tp_out[ilon,ilat,:] = tp_night
        else:
            arg = (lon-phase_offset)/scale*dtr
            tp = tp_night + (tp_day-tp_night) * np.cos(arg)
            for ilat,lat in enumerate(lat_grid):
                tp_out[ilon,ilat,:] = tp

    return tp_out

def Model4(P_grid, lon_grid, lat_grid, g_plt, T_eq,
    scale, phase_offset,
    log_kappa_day, log_gamma_day, log_f_day, T_int_day,
    log_kappa_night, log_gamma_night, log_f_night, T_int_night,
    n):
    """
    2D temperature model consisting of two representative Guillot TP profiles.
    See model 4 in Yang et al. 2023. (https://doi.org/10.1093/mnras/stad2555)
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
        Set to be between 0.5 and 1.2.
    phase_offset : real
        Central longitude of the dayside
        Set to be between -45 and 45.
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
    n : real
        Parameter to control how temperature vary with longitude on the dayside.
        Should be positive.

    Returns
    -------
    tp_out : ndarray
        Temperature model defined on a (longitude, laitude, pressure) grid.
    """
    # phase_offset hard coded to be between -45 and 45
    assert phase_offset <=45 and phase_offset >= -45
    assert scale <=1.2 and scale >=0.5
    dtr = np.pi/180
    # boundaries
    bound_east = phase_offset + 90 * scale
    bound_west = phase_offset - 90 * scale

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

    for ilon,lon in enumerate(lon_grid):
        if lon < bound_west or lon > bound_east:
            for ilat,lat in enumerate(lat_grid):
                tp_out[ilon,ilat,:] = tp_night
        else:
            arg = (lon-phase_offset)/scale*dtr
            tp = tp_night + (tp_day-tp_night) * np.cos(arg) ** n
            for ilat,lat in enumerate(lat_grid):
                tp_out[ilon,ilat,:] = tp

    return tp_out

def tmap_2_guillot(P_grid, lon_grid, lat_grid, g_plt, T_eq,
    scale, phase_offset,
    log_kappa_day, log_gamma_day, log_f_day, T_int_day,
    log_kappa_night, log_gamma_night, log_f_night, T_int_night,
    ):
    """
    2D temperature model consisting of two Guillot TP profiles.
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

def tmap_3_guillot(P_grid, lon_grid, lat_grid, g_plt, T_eq,
    scale, phase_offset, west_fraction, east_fraction,
    log_kappa_hot, log_gamma_hot, log_f_hot, T_int_hot,
    log_kappa_day, log_gamma_day, log_f_day, T_int_day,
    log_kappa_night, log_gamma_night, log_f_night, T_int_night,
    ):
    """
    Temperature model for phase curve fiting consisting of three TP profiles.
    The atmosphere is partitioned (in longitude) into three regions:
    a hotspot, a dayside and a nightside.

    Parameters
    ----------
    P_grid : ndarray
        Pressure grid (in Pa) on which the model is to be constructed.
    lon_grid : ndarray
        Longitude grid (in degree) on which the model is to be constructed.
        Substellar point is at 0. Range is [-180,180].
    lat_grid : ndarray
        Latitdue grid (in degree) on which the model is to be constructed.
        Range is [-90,90].s
    g_plt : real
        Gravitational acceleration at the highest pressure in the pressure
        grid.
    T_eq : real
        Temperature corresponding to the stellar flux.
        T_eq = T_star * (R_star/(2*semi_major_axis))**0.5
    phase_offset : real
        Central longitude of the dayside
    scale : real
        Scaling parameter for the longitudinal span of the dayside.
    west_fraction : real
        Fraction of the western half of dayside spaned by the hotspot.
        Range : [0,1]
    east_fraction : real
        Fraction of the eastern half of dayside spaned by the hotspot.
        Range : [0,1]
    log_kappa_day : real
        Range ~ [1e-5,1e3]
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
    log_kappa_hot : real
        Same as above definitions but for hotspot.
    log_gamma_hot : real
        Same as above definitions but for hotspot.
    log_f_hot : real
        Same as above definitions but for hotspot.
    T_int_hot : real
        Same as above definitions but for hotspot.
    Returns
    -------
    tp_out : ndarray
        Temperature model defined on a (longitude, laitude, pressure) grid.
    """
    # scale hard coded to be between 0.1 and 0.5
    assert scale <= 1.2 and scale >=0.5
    # phase_offset hard coded to be between -45 and 45
    assert phase_offset <=45 and phase_offset >= -45
    # assert hot spot fraction is less than 1
    assert west_fraction >= 0 and  west_fraction <= 1
    assert west_fraction >= 0 and  west_fraction <= 1

    # scale the extent of the hot region
    east_day_limit = 90 * scale
    west_day_limit = -90 * scale

    # scale the extent of the hotspot
    east_hot_limit = 90 * scale * east_fraction
    west_hot_limit = 90 * scale * west_fraction

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
    kappa_hot = 10**log_kappa_hot
    gamma_hot = 10**log_gamma_hot
    f_hot = 10**log_f_hot

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
        if lon >= (phase_offset+west_day_limit) \
            and lon <= (phase_offset+east_day_limit):
            if lon >= (phase_offset-west_hot_limit) \
                and lon <= (phase_offset+east_hot_limit):
                for ilat, lat in enumerate(lat_grid):
                    tp_out[ilon,ilat,:] = tp_hot
            else:
                for ilat, lat in enumerate(lat_grid):
                    tp_out[ilon,ilat,:] = tp_day
            # print('dayside')


        # nightside
        else:
            for ilat, lat in enumerate(lat_grid):
                tp_out[ilon,ilat,:] = tp_night

    return tp_out