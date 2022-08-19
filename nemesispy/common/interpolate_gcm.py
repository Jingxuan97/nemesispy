#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
from numba import jit

@jit(nopython=True)
def VERINT(X,Y,XIN):
    """
    np.interp only works with strictly increasing independent variable.
    """
    if X[0]>X[-1]:
        X = X[::-1]
        Y = Y[::-1]
    YOUT = np.interp(x=XIN,xp=X,fp=Y)
    return YOUT

@jit(nopython=True)
def interpvivien_point(lon, lat, p, gcm_lon, gcm_lat, gcm_p, gcm_t, gcm_vmr,
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

    Returns
    -------
    T : ndarray
        Temperature model interpolated to (lon,lat,p)
    VMR : ndarray
        VMR model interpolated to (lon,lat,p)

    """
    # Check input latitude make sense
    assert lat<=90 and lat>=-90

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
            JLAT = NLAT - 1
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
        tempY1 = VERINT(log_gcm_P,tempVY1,LP1)
        tempY2 = VERINT(log_gcm_P,tempVY2,LP1)
        tempY3 = VERINT(log_gcm_P,tempVY3,LP1)
        tempY4 = VERINT(log_gcm_P,tempVY4,LP1)
        interped_T[IPRO] = (1.0-FLON)*(1.0-FLAT)*tempY1 + FLON*(1.0-FLAT)*tempY2 \
            + FLON*FLAT*tempY3 + (1.0-FLON)*FLAT*tempY4
        for IVMR in range(NVMR):
            gasVY1 = gcm_vmr[JLON1,JLAT,:,IVMR]
            gasVY2 = gcm_vmr[JLON2,JLAT,:,IVMR]
            gasVY3 = gcm_vmr[JLON2,JLAT+1,:,IVMR]
            gasVY4 = gcm_vmr[JLON1,JLAT+1,:,IVMR]
            gasY1 = VERINT(log_gcm_P,gasVY1,LP1)
            gasY2 = VERINT(log_gcm_P,gasVY2,LP1)
            gasY3 = VERINT(log_gcm_P,gasVY3,LP1)
            gasY4 = VERINT(log_gcm_P,gasVY4,LP1)
            interped_VMR[IPRO,IVMR] = (1.0-FLON)*(1.0-FLAT)*gasY1 + FLON*(1.0-FLAT)*gasY2\
                + FLON*FLAT*gasY3 + (1.0-FLON)*FLAT*gasY4

    return interped_T, interped_VMR

def interp_gcm_X(lon, lat, p, gcm_lon, gcm_lat, gcm_p, X,
    substellar_point_longitude_shift=0):
    """
    Find the X(P) profile at a location specified by (lon,lat) by interpolating
    a GCM. X can be any scalar quantity modeled in the GCM.

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
        The longitude shift between the longitude-lattitude coordinate system
        of the GCM model to the output coordinate system. For example, if in the
        output coordinate system the substellar point is defined at 0 E,
        whereas in the GCM coordinate system the substellar point is defined
        at 90 E, put substellar_point_longitude_shift=90.

    Returns
    -------
    interped_X : ndarray
        X interpolated to (lon,lat,p).
    """
    # Check input latitude make sense
    assert lat <= 90 and lat >= -90

    # Number of pressures in the interped profile
    NPRO = len(p)

    # Dimensions of GCM
    NLON = len(gcm_lon)
    NLAT = len(gcm_lat)
    NPRESS = len(gcm_p)

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
            JLAT = NLAT - 1
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
        Y1 = VERINT(log_gcm_p,VY1,LP1)
        Y2 = VERINT(log_gcm_p,VY2,LP1)
        Y3 = VERINT(log_gcm_p,VY3,LP1)
        Y4 = VERINT(log_gcm_p,VY4,LP1)
        interped_X[IPRO] = (1.0-FLON)*(1.0-FLAT)*Y1 + FLON*(1.0-FLAT)*Y2 \
            + FLON*FLAT*Y3 + (1.0-FLON)*FLAT*Y4

    return interped_X
