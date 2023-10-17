#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
from numba import jit

@jit(nopython=True)
def interp_1D(X,Y,XIN):
    """
    1D interpolation using the np.interp function.

    Parameters
    ----------
    X : ndarray
        Array of ordered independent variables.
    Y : ndarray
        Array of dependent variables.
    XIN : ndarray
        Array of independent variable values to interpolate at.

    Returns
    -------
    YOUT : ndarray
        Interpolated values.

    Important:
    np.interp needs an array of strictly increasing independent
    variables. While the routine does not throw errors when the input array is
    not strictly increasing, the results will be nonsensical. This routine
    assumes that the independent variable array is ordered and reverse the
    order if it is strictly decreasing.
    """
    if X[0]>X[-1]:
        X = X[::-1]
        Y = Y[::-1]
    YOUT = np.interp(x=XIN,xp=X,fp=Y)
    return YOUT

@jit(nopython=True)
def interp_gcm(lon, lat, p, gcm_lon, gcm_lat, gcm_p, gcm_t, gcm_vmr,
    substellar_point_longitude_shift=0):
    """
    Find the T(P) profile and VMR(P) profile at a location specified by (lon,lat)
    by interpolating a GCM.

    Parameters
    ----------
    lon : real
        Longitude of the location
    lat : real
        Latitude of the location
    p : ndarray
        Pressure grid for the output T(P) and VMR(P) profile
    gcm_lon : ndarray
        Longitude grid of the GCM, assumed to be [-180,180] and increasing. (1D)
    gcm_lat : ndarray
        Latitude grid of the GCM, assumed to be [-90,90] and increasing. (1D)
    gcm_p : ndarray
       Pressure grid of the GCM. (1D)
    gcm_t : ndarray
        Temperature model.
        Has dimension NLON x NLAT x NPRESS.
    gcm_vmr : ndarray
        VMR model.
        NLON x NLAT x NPRESS x NVMR
    substellar_point_longitude_shift : real
        The longitude shift between the longitude-lattitude coordinate system
        of the GCM model to the output coordinate system. For example, if in the
        output coordinate system the substellar point is defined at 0 E,
        whereas in the GCM coordinate system the substellar point is defined
        at 90 E, put substellar_point_longitude_shift=90.
        It should be set to 180 if you want to simulate phase curves and set
        set substellar point to 180, coinciding with secondary eclipse.

    Returns
    -------
    T : ndarray
        Temperature model interpolated to (lon,lat,p)
    VMR : ndarray
        VMR model interpolated to (lon,lat,p)

    """
    # Check input latitude make sense
    assert lat <= 90 and lat >= -90, "Input latitude out of range [-90,90]"

    # Check dimensions of gcm_t matches the given dimension of the GCM
    NLON = len(gcm_lon)
    NLAT = len(gcm_lat)
    NPRESS = len(gcm_p)
    assert NLON == gcm_t.shape[0], "GCM longitude grid dimension mismatch"
    assert NLAT == gcm_t.shape[1], "GCM latitude grid dimension mismatch"
    assert NPRESS == gcm_t.shape[2], "GCM pressure grid dimension mismatch"

    # Number of pressures in the interped profile
    NPRO = len(p)

    # Dimension parameters of the atmospheric model
    NLON,NLAT,NPRESS,NVMR = gcm_vmr.shape

    # Convert to log pressure
    log_gcm_P = np.log(gcm_p)

    # Shift to desired longitude coordinate system
    # while preserving the monotonicity of longitude
    if substellar_point_longitude_shift != 0:
        substellar_point_longitude_shift \
            = np.mod(substellar_point_longitude_shift,360)
        if substellar_point_longitude_shift <=180:
            gcm_lon = gcm_lon + substellar_point_longitude_shift
        else:
            gcm_lon = gcm_lon - (360-substellar_point_longitude_shift)

    #  Find closest point in stored array
    JLAT = -1
    for I in range(NLAT-1):
        if lat >= gcm_lat[I] and lat <= gcm_lat[I+1]:
            JLAT = I
            FLAT = (lat-gcm_lat[I])/(gcm_lat[I+1]-gcm_lat[I])
    if JLAT < 0:
        if lat < gcm_lat[0]:
            JLAT = 0
            FLAT = 0
        if lat >= gcm_lat[-1]:
            JLAT = NLAT - 2
            FLAT = 1

    JLON1 = -1
    JLON2 = -1
    for I in range(NLON-1):
        if lon >= gcm_lon[I] and lon <= gcm_lon[I+1]:
            JLON1 = I
            JLON2 = I+1
            FLON = (lon-gcm_lon[I])/(gcm_lon[I+1]-gcm_lon[I])
    if JLON1 < 0:
        if lon < gcm_lon[0]:
            # lon must be in range 0. to gcm_lon[0]
            JLON1 = NLON - 1
            JLON2 = 0
            FLON = (lon+360-gcm_lon[-1])/(gcm_lon[0]+360-gcm_lon[-1])
        if lon >= gcm_lon[-1]:
            # lon must be in range gcm_lon[-1] to 360
            JLON1 = NLON - 1
            JLON2 = 0
            FLON = (lon - gcm_lon[-1])/(gcm_lon[0]+360-gcm_lon[-1])

    # Output arrays
    interped_T = np.zeros(NPRO)
    interped_VMR = np.zeros((NPRO,NVMR))

    # Temperature interpolation array
    tempVY1 = gcm_t[JLON1,JLAT,:]
    tempVY2 = gcm_t[JLON2,JLAT,:]
    tempVY3 = gcm_t[JLON2,JLAT+1,:]
    tempVY4 = gcm_t[JLON1,JLAT+1,:]
    for IPRO in range(NPRO):
        # convert pressure to atm then to log
        LP1 = np.log(p[IPRO])
        tempY1 = interp_1D(log_gcm_P,tempVY1,LP1)
        tempY2 = interp_1D(log_gcm_P,tempVY2,LP1)
        tempY3 = interp_1D(log_gcm_P,tempVY3,LP1)
        tempY4 = interp_1D(log_gcm_P,tempVY4,LP1)
        interped_T[IPRO] = (1.0-FLON)*(1.0-FLAT)*tempY1 + FLON*(1.0-FLAT)*tempY2 \
            + FLON*FLAT*tempY3 + (1.0-FLON)*FLAT*tempY4
        for IVMR in range(NVMR):
            gasVY1 = gcm_vmr[JLON1,JLAT,:,IVMR]
            gasVY2 = gcm_vmr[JLON2,JLAT,:,IVMR]
            gasVY3 = gcm_vmr[JLON2,JLAT+1,:,IVMR]
            gasVY4 = gcm_vmr[JLON1,JLAT+1,:,IVMR]
            gasY1 = interp_1D(log_gcm_P,gasVY1,LP1)
            gasY2 = interp_1D(log_gcm_P,gasVY2,LP1)
            gasY3 = interp_1D(log_gcm_P,gasVY3,LP1)
            gasY4 = interp_1D(log_gcm_P,gasVY4,LP1)
            interped_VMR[IPRO,IVMR] = (1.0-FLON)*(1.0-FLAT)*gasY1 + FLON*(1.0-FLAT)*gasY2\
                + FLON*FLAT*gasY3 + (1.0-FLON)*FLAT*gasY4

    return interped_T, interped_VMR

@jit(nopython=True)
def interp_gcm_X(lon, lat, p, gcm_lon, gcm_lat, gcm_p, X,
    substellar_point_longitude_shift=0):
    """
    Find the profile of X as a function of pressure at a location specified by
    (lon,lat) by interpolating a GCM. X can be any scalar quantity modeled in
    the GCM, for example temperature or chemical abundance.
    Note: gcm_lon and gcm_lat must be strictly increasing.

    Parameters
    ----------
    lon : real
        Longitude of the location
    lat : real
        Latitude of the location
    p : ndarray
        Pressure grid for the output X(P) profile
    gcm_lon : ndarray
        Longitude grid of the GCM, assumed to be [-180,180] and increasing. (1D)
    gcm_lat : ndarray
        Latitude grid of the GCM, assumed to be [-90,90] and increasing. (1D)
    gcm_p : ndarrray
        Pressure grid of the GCM. (1D)
    X : ndarray
        A scalar quantity defined in the GCM, e.g., temperature or VMR. (3D)
        Has dimensioin NLON x NLAT x NP
    substellar_point_longitude_shift : real
        The longitude shift from the output coordinate system to the
        coordinate system of the GCM. For example, if in the output
        coordinate system the substellar point is defined at 0 E,
        whereas in the GCM coordinate system the substellar point is defined
        at 180 E, put substellar_point_longitude_shift=180.

    Returns
    -------
    interped_X : ndarray
        X interpolated to (lon,lat,p).

    Important:
    Longitudinal values outside of the gcm longtidinal grid are interpolated
    properly using the periodicity of longitude. However, latitudinal values
    outside of the gcm latitude grid are interpolated using the boundary
    values of the gcm grid. In practice, this reduces the accuracy of
    interpolation in the polar regions outside of the gcm grid; in particular,
    the interpolated value at the poles will be dependent on longitude.
    This is a negligible source of error in disc integrated spectroscopy since
    the contribution of radiance is weighted by cos(latitude).
    """
    # Check input longitude and latitude make sense
    assert lat <= 90 and lat >= -90, "Input latitude out of range [-90,90]"

    # Check dimensions of X matches the given dimension of the GCM
    NLON = len(gcm_lon)
    NLAT = len(gcm_lat)
    NPRESS = len(gcm_p)
    assert NLON == X.shape[0], "GCM longitude grid dimension mismatch"
    assert NLAT == X.shape[1], "GCM latitude grid dimension mismatch"
    assert NPRESS == X.shape[2], "GCM pressure grid dimension mismatch"

    # Number of pressures in the profile to be interpolated
    NPRO = len(p)

    # Work in log pressure
    log_gcm_p = np.log(gcm_p)

    # Shift to desired longitude coordinate system
    # while preserving the monotonicity of longitude
    if substellar_point_longitude_shift != 0:
        substellar_point_longitude_shift \
            = np.mod(substellar_point_longitude_shift,360)
        if substellar_point_longitude_shift <=180:
            gcm_lon = gcm_lon + substellar_point_longitude_shift
        else:
            gcm_lon = gcm_lon - (360-substellar_point_longitude_shift)

    #  Find closest point on the GCM grid to input location
    JLAT = -1
    for I in range(NLAT-1):
        if lat >= gcm_lat[I] and lat <= gcm_lat[I+1]:
            JLAT = I
            FLAT = (lat-gcm_lat[I])/(gcm_lat[I+1]-gcm_lat[I])
    if JLAT < 0:
        if lat < gcm_lat[0]:
            JLAT = 0
            FLAT = 0
        if lat >= gcm_lat[-1]:
            JLAT = NLAT - 2
            FLAT = 1

    JLON1 = -1
    JLON2 = -1
    for I in range(NLON-1):
        if lon >= gcm_lon[I] and lon <= gcm_lon[I+1]:
            JLON1 = I
            JLON2 = I+1
            FLON = (lon-gcm_lon[I])/(gcm_lon[I+1]-gcm_lon[I])
    if JLON1 < 0:
        if lon < gcm_lon[0]:
            # lon must be in range 0. to gcm_lon[0]
            JLON1 = NLON - 1
            JLON2 = 0
            FLON = (lon+360-gcm_lon[-1])/(gcm_lon[0]+360-gcm_lon[-1])
        if lon >= gcm_lon[-1]:
            # lon must be in range gcm_lon[-1] to 360
            JLON1 = NLON - 1
            JLON2 = 0
            FLON = (lon - gcm_lon[-1])/(gcm_lon[0]+360-gcm_lon[-1])

    # Output arrays
    interped_X = np.zeros(NPRO)

    # Interpolation array
    VY1 = X[JLON1,JLAT,:]
    VY2 = X[JLON2,JLAT,:]
    VY3 = X[JLON2,JLAT+1,:]
    VY4 = X[JLON1,JLAT+1,:]
    log_p = np.log(p)
    for IPRO in range(NPRO):
        # convert pressure to atm then to log
        LP1 = log_p[IPRO]
        Y1 = interp_1D(log_gcm_p,VY1,LP1)
        Y2 = interp_1D(log_gcm_p,VY2,LP1)
        Y3 = interp_1D(log_gcm_p,VY3,LP1)
        Y4 = interp_1D(log_gcm_p,VY4,LP1)
        interped_X[IPRO] = (1.0-FLON)*(1.0-FLAT)*Y1 + FLON*(1.0-FLAT)*Y2 \
            + FLON*FLAT*Y3 + (1.0-FLON)*FLAT*Y4

    return interped_X

def lat_average_gcm_X(output_lon_grid, output_lat_grid, output_p_grid,
    cutoff, gcm_lon, gcm_lat, gcm_p, X,
    substellar_point_longitude_shift=0):
    """
    Replace a gcm quantity everywhere to the latitudinal average from 0 degree
    to the cutoff, using cos(latitude as a weight).
    """
    # average weights
    dtr = np.pi/180
    qudrature = np.linspace(0,cutoff,100)
    weight = np.cos(qudrature*dtr) * (qudrature[1]-qudrature[0]) * dtr
    sum_weight = np.sum(weight)

    # output array
    Xout = np.zeros((len(output_lon_grid),
        len(output_lat_grid),
        len(output_p_grid)))

    Xequator = np.zeros((len(output_p_grid), len(output_lon_grid)))

    for ilon, lon in enumerate(output_lon_grid):
        for ilat, lat in enumerate(qudrature):
            iX = interp_gcm_X(lon,lat,output_p_grid,
                gcm_p=gcm_p,gcm_lon=gcm_lon,gcm_lat=gcm_lat,X=X,
                substellar_point_longitude_shift=substellar_point_longitude_shift)
            Xequator[:,ilon] += iX * weight[ilat]

    Xequator = Xequator/sum_weight
    for ilon, lon in enumerate(output_lon_grid):
        for ilat, lat in enumerate(output_lat_grid):
            Xout[ilon,ilat,:] = Xequator[:,ilon]

    return Xout

def lat_average_gcm_VMR(output_lon_grid, output_lat_grid, output_p_grid,
    cutoff, gcm_lon, gcm_lat, gcm_p, gcm_VMR,
    substellar_point_longitude_shift=0):
    nlon,nlat,nP,nvmr = gcm_VMR.shape

    VMR_out = np.zeros(
        (len(output_lon_grid),len(output_lat_grid),len(output_p_grid),nvmr))

    for ivmr in range(nvmr):
        VMR_out[:,:,:,ivmr] = lat_average_gcm_X(
            output_lon_grid, output_lat_grid, output_p_grid,
            cutoff, gcm_lon, gcm_lat, gcm_p, gcm_VMR[:,:,:,ivmr],
            substellar_point_longitude_shift=substellar_point_longitude_shift)

    return VMR_out