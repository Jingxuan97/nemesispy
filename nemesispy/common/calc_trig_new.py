#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from numba import jit
from nemesispy.common.calc_lobatto import disc_weights

@jit(nopython=True)
def arctan(x,y):
    """
    Calculate the argument of the point (x,y) in the range [0,2pi).

    Parameters
    ----------
        x : real
            x-coordinate of the point (length of the adjacent side)
        y : real
            y-coordinate of the point (length of the opposite side)
    Returns
    -------
        ang : real
            Argument of (x,y) in radians
    """
    if(x == 0.0):
        if (y == 0.0) : ang = 0.0 # (x,y) is the origin, ill-defined
        elif (y > 0.0) : ang = 0.5*np.pi # (x,y) is on positive y-axis
        else : ang = 1.5*np.pi  # (x,y) is on negative y-axis
    else:
        ang=np.arctan(y/x)
        if (y > 0.0) :
            if (x > 0.0) : ang = ang # (x,y) is in 1st quadrant
            else : ang = ang+np.pi # (x,y) is in 2nd quadrant
        elif (y == 0.0) :
            if (x > 0.0) : ang = 0 # (x,y) is on positive x-axis
            else : ang = np.pi # (x,y) is on negative x-axis
        else:
            if (x > 0.0) : ang = ang+2*np.pi # (x,y) is in 4th quadrant
            else : ang = ang+np.pi # (x,y) is in 3rd quadrant
    return ang

@jit(nopython=True)
def rotatey(v, phi):
    """
    Rotate a 3D vector v anticlockwisely about the y-axis by angle phi

    Parameters
    ----------
        v : ndarray
            A real 3D vector to rotate
        phi : real
            Angle to rotate the vector v by (radians)

    Returns
    -------
        v_new : ndarray
            Rotated 3D vector
    """
    a = np.zeros((3,3)) # construct the rotation matrix
    a[0,0] = np.cos(phi)
    a[0,2] = np.sin(phi)
    a[1,1] = 1.
    a[2,0] = -np.sin(phi)
    a[2,2] = np.cos(phi)
    # v_new = np.matmul(a,v) # unsupported NumPy function
    v_new = np.zeros(3)
    for i in range(3):
        for j in range(3):
            v_new[i] += a[i,j] * v[j]
    return v_new

@jit(nopython=True)
def rotatez(v, phi):
    """
    Rotate a 3D vector v anticlockwisely about the z-axis by angle phi

    Parameters
    ----------
        v : ndarray
            A real 3D vector to rotate
        phi : real
            Angle to rotate the vector by (radians)

    Returns
    -------
        v_new : ndarray
            Rotated 3D vector
    """
    a = np.zeros((3,3))
    a[0,0] = np.cos(phi)
    a[0,1] = -np.sin(phi)
    a[1,0] = np.sin(phi)
    a[1,1] = np.cos(phi)
    a[2,2] = 1
    # v_new = np.matmul(a,v) # unsupported NumPy function
    v_new = np.zeros(3)
    for i in range(3):
        for j in range(3):
            v_new[i] += a[i,j] *v[j]
    return v_new

@jit(nopython=True)
def generate_angles(phase,rho,alpha):
    """
    Finds the stellar zenith angle, stellar azimuth angle, lattitude and longitude
    of a chosen point on the visible disc of a planet under observation. The planet
    is assumed to be tidally locked, and is observed on an edge-on orbit.

    Refer to the begining of the trig.py file for geomety and conventions.

    Parameters
    ----------
    phase : real
        Orbital phase in degrees. 0 at parimary transit and 180 at secondary eclipse.
        Range: [0,360)
    rho	: real
        Fractional radius of the point on disc, must be between 0 and 1 inclusive.
        Range: [0,1]
    alpha : real
        Argument of the point on visible disc (degrees), measured
        anticlockwise from 3 o'clock.
        Range: [0,360)

    Returns
    -------
    zen : real
        Computed solar zenith angle (radians), which is the angle between the local
        normal and the stellar direction vector.
    lat	: real
        Planetocentric latitude of the point (degrees).
    lon	: real
        Planetocentric longitude of the point (degrees).
    """
    phase = np.mod(phase,360)
    dtr = np.pi/180. # degree to radiance conversion factor
    assert rho <=1, "Fractional radius should be less or equal to 1"

    # get stellar direction vector in Cartesian coordinates
    # ie unit vector in direction of star
    theta_star = np.pi/2. # star lies in planet's equitorial plane
    phi_star = 90.0 + phase # when phase angle is 0 the star lies on the y-axis
    x_star = np.sin(theta_star)*np.cos(phi_star*dtr)
    y_star = np.sin(theta_star)*np.sin(phi_star*dtr)
    z_star = np.cos(theta_star)
    v_star = np.array([x_star,y_star,z_star])

    # get Cartesian coordinates of input point
    # calculate point position vector using spherical polars (r=1,theta,phi)
    theta_point = np.arccos(rho*np.sin(alpha*dtr)) # ie planetary zenith angle of poiny
    if np.sin(theta_point) != 0.0:
        cos_phi = rho*np.cos(alpha*dtr)/abs(np.sin(theta_point))
        phi_point = (-np.arccos(cos_phi))%(2*np.pi) # azimuth angle of point (on our side)
    else:
        phi_point = 0.0 # sin(theta_point) = 0 at north polt
    x_point = np.sin(theta_point)*np.cos(phi_point)
    y_point = np.sin(theta_point)*np.sin(phi_point)
    z_point = np.cos(theta_point)
    v_point = np.array([x_point,y_point,z_point])

    # calculate angle between solar position vector and local normal
    # i.e. zen solar zenith angle
    inner_product = np.sum(v_star*v_point)
    zen = np.arccos(inner_product)
    zen = np.around(zen, 10)

    # calculate latitude and longitude of the spot
    # (sub-stellar point = 180E, anti-stellar point = 0E,
    # longtitudes in the direction of self-rotation)
    lat = np.around(90.-theta_point*180/np.pi, 10) # southern hemisphere has negative lattitude
    lon = (phi_point/dtr - (phi_star+180))%360 # substellar point is 180E

    return zen, lat, lon

@jit(nopython=True)
def disc_weights_new(phase, nmu):
    """
    Given the orbital phase, calculates the coordinates and weights of the points
    on a disc needed to compute the disc integrated radiance.

    The points are chosen on a number of rings according to Gauss-Lobatto quadrature
    scheme, and spaced on the rings according to trapezium rule.

    Refer to the begining of the trig.py file for geomety and convections.

    Parameters
    ----------
    phase : real
        Stellar phase/orbital phase in degrees.
        0=parimary transit and increase to 180 at secondary eclipse.
    nmu	: integer
        Number of zenith angle ordinates

    Output variables
    nav	: integer
        Number of FOV points
    wav	: ndarray
        FOV-averaging table:
        0th row is lattitude, 1st row is longitude, 2nd row is stellar zenith
        angle, 3rd row is emission zenith angle, 4th row is stellar azimuth angle,
        5th row is weight.
    """
    assert nmu >=2, "Need at least 2 quadrature points"
    phase = phase%360
    dtr = np.pi/180 # degree to radiance conversion factor
    delR = 1./nmu #

    # get the gauss labatto quadrature points and weights
    mu,wtmu = disc_weights(nmu)

    # set up the output arrays
    nsample = 2*nmu**2
    tablat = np.zeros(nsample) # latitudes
    tablon = np.zeros(nsample) # longitudeds
    tabzen = np.zeros(nsample) # zenith angle in quadrature scheme
    tabwt = np.zeros(nsample)  # weight of each sample

    # define FOV averaging points
    isample = 0
    for imu in range(0, nmu): # quadrature rings
        r_quad = np.sqrt(1.-mu[imu]**2) # quadrature radius (from small to large)
        half_circum = np.pi*r_quad # half the circumference

        if(half_circum > 0.0):
            nalpha = int(0.5+half_circum/delR)
            alpha_sample_list = 180*np.arange(nalpha)/(nalpha-1)
        else:
            nalpha=1

        if(nalpha > 1): # more than one sample on the quadrature ring

            for ialpha in np.arange(0,nalpha):

                alpha_sample = alpha_sample_list[ialpha]
                zen_sample, lat_sample, lon_sample \
                    = generate_angles(phase,r_quad,alpha_sample)

                # trapezium rule weights
                if (ialpha == 0):
                    wt_trap = (alpha_sample_list[ialpha+1]-alpha_sample_list[ialpha])/2.0
                elif (ialpha == nalpha-1):
                    wt_trap = (alpha_sample_list[ialpha]-alpha_sample_list[ialpha-1])/2.0
                else:
                    wt_trap = (alpha_sample_list[ialpha+1]-alpha_sample_list[ialpha-1])/2.0

                wt_azi= wt_trap/180. # sample azimuthal weight

                tablat[isample] = lat_sample # sample lattitude
                tablon[isample] = lon_sample # sample longitude
                tabzen[isample] = np.arccos(mu[imu])/dtr # sample emission zenith angle
                tabwt[isample] = 2*mu[imu]*wtmu[imu]*wt_azi # sample weight
                isample = isample+1

        else:
            alpha_sample = 0.
            zen_sample, lat_sample,lon_sample \
                = generate_angles(phase,r_quad,alpha_sample)
            if(tabzen[isample] == 0.0):
                azi_sample = 180.
            tablat[isample] = lat_sample
            tablon[isample] = lon_sample
            tabzen[isample] = np.arccos(mu[imu])/dtr
            tabwt[isample] = 2*mu[imu]*wtmu[imu]
            isample = isample+1

    nav = isample
    wav = np.zeros((4,isample))
    sum=0.
    for i in np.arange(0,isample):
        wav[0,i]=tablat[i]              # 0th array is lattitude
        wav[1,i]=tablon[i]%360          # 1st array is longitude
        wav[2,i]=tabzen[i]              # 3rd array is emission zenith angle
        wav[3,i]=tabwt[i]               # 5th array is weight
        sum = sum+tabwt[i]

    for i in range(isample):            # normalise weights so they add up to 1
        wav[3,i]=wav[3,i]/sum

    return nav, wav

# def disc_weights_2tp(phase, nmu, daymin, daymax):
#     """
#     Parameters
#     ----------
#     phase : real
#         Stellar phase/orbital phase in degrees.
#         0=parimary transit and increase to 180 at secondary eclipse.
#     nmu	: integer
#         Number of zenith angle ordinates
#     daymin : real
#         Longitude of the west boundary of dayside. Assume [-180,180] grid.
#     daymax : real
#         Longitude of the east boundary of dayside. Assume [-180,180] grid.

#     Returns
#     -------
#     new_nav : ndarray
#         Reduced number of FOV points
#     new_wav :
#         Reduced FOV-averaging table.
#     """
#     # Tmaps has sub stellar point at 0 and is [-180,180]
#     # orbital phase is [0,360]
#     daymin = daymin + 180
#     daymax = daymax + 180
#     nav, wav = disc_weights(phase, nmu)
#     lat_list = wav[0,:]
#     lon_list = wav[1,:]
#     zen_list = wav[2,:]
#     wt_list = wav[3,:]

#     # split FOV points to dayside and nightside
#     daymask = np.logical_and(lon_list>daymin,lon_list<=daymax)
#     nightmask = ~daymask

#     # dayside points
#     day_lat = lat_list[daymask]
#     day_lon = lon_list[daymask]
#     day_zen = zen_list[daymask]
#     day_wt = wt_list[daymask]

#     # nightside points
#     night_lat = lat_list[nightmask]
#     night_lon = lon_list[nightmask]
#     night_zen = zen_list[nightmask]
#     night_wt = wt_list[nightmask]

#     output_day_lat = []
#     output_day_lon = []
#     output_day_zen = []
#     output_day_wt = []

#     output_night_lat = []
#     output_night_lon = []
#     output_night_zen = []
#     output_night_wt = []

#     imu = 0
#     ilat = 0
#     while ilat < len(day_lat):
#         if ilat == 0:
#             output_day_lat.append(day_lat[0])
#             output_day_lon.append(day_lon[0])
#             output_day_zen.append(day_zen[0])
#             output_day_wt.append(day_wt[0])
#             ilat += 1
#         else:
#             # sum weight
#             if day_zen[ilat] == day_zen[ilat-1]:
#                 output_day_lat.append(day_lat[ilat])
#                 output_day_lon.append(day_lon[ilat])
#                 output_day_wt[imu] += day_wt[ilat]
#                 ilat += 1
#             # add new point
#             else:
#                 output_day_lat.append(day_lat[ilat])
#                 output_day_lon.append(day_lon[ilat])
#                 output_day_zen.append(day_zen[ilat])
#                 output_day_wt.append(day_wt[ilat])
#                 imu += 1
#                 ilat += 1

#     imu = 0
#     ilat = 0
#     while ilat < len(night_lat):
#         if ilat == 0:
#             output_night_lat.append(night_lat[0])
#             output_night_lon.append(night_lon[0])
#             output_night_zen.append(night_zen[0])
#             output_night_wt.append(night_wt[0])
#             ilat += 1
#         else:
#             if night_zen[ilat] == night_zen[ilat-1]:
#                 output_night_lat.append(night_lat[ilat])
#                 output_night_lon.append(night_lon[ilat])
#                 output_night_wt[imu] += night_wt[ilat]
#                 ilat += 1
#             else:
#                 output_night_lat.append(night_lat[ilat])
#                 output_night_lon.append(night_lon[ilat])
#                 output_night_zen.append(night_zen[ilat])
#                 output_night_wt.append(night_wt[ilat])
#                 imu += 1
#                 ilat += 1

#     output_day_wt = output_day_wt/(sum(output_day_wt)+sum(output_night_wt))
#     output_night_wt = output_night_wt/(sum(output_day_wt)+sum(output_night_wt))

#     return output_day_zen,output_day_wt,\
#         output_night_zen,output_night_wt,\
#         output_day_lat,output_day_lon,\
#         output_night_lat,output_night_lon
