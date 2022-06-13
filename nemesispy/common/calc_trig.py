#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate sample locations and corresponding weights on a planetary disc
for calculating the disc-averaged radiance of a transitting planet
at variable orbital phases.

Solar zenith and azimuth angles of the sample points
are also generated, but not currently used asreflected sunlight is not included.

Assumptions and conventions:

    1. The visible planetary disc is a circle (with dimensionless radius 1).

    2. The orbital phase is 0 at primary transit and 180 at secondary eclipse.
    Orbital phase increases in the direction of orbital motion.

    3. The planet is tidally locked, i.e. it's in synchronous rotation
    and always present the same hemisphere to the star. Therefore, We are
    observing the planet's orbit edge on.

    4. The planetocentric longitude is defined such that the antistellar point
    is on the 0E meridian, and the substellar point is on the 180E meridian.
    Longitude increases in the direction of planet's self-rotation.

    5. The north pole of the planet is defined such that when viewed directly
    above the north pole the planet is rotating anticlockwise.

Coordinate system (origin is at the centre of the target):

    x-axis points from observer's 9 o'clock to observer's 3 o'clock
    y-axis points from observer to the target
    z-axis points from south pole to north pole
    r is the radial distance
    theta is measured clockwiseky from the z-axis
    phi is measured anticlockwise from x-axis

    x = r * sin(theta) * cos(phi) = rho * cos(alpha)
    y = r * sin(theta) * sin(phi) = rho * sin(alpha)
    z = r * cos(theta)

    rho: projected distance from a point on the disc to the centre of the disc
    alpha: argument of a point on the projected disc, measured anticlockwise from x axis
"""
import numpy as np
from numba import jit

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
    is assumed to be tidally locked, and is observed on an edgy-on orbit.

    Refer to the begining of the trig.py file for geomety and convections.

    Parameters
    ----------
    phase : real
        Orbital phase in degrees. 0 at  parimary transit and 180 at secondary eclipse.
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
    azi	: real
        Computed solar azimuth angle (radians). Uses convention that
        forward scatter = 0. !!! need to define.
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
    # (sub-stellar point = 180E, anti-stellar point = 0E, longtitudes in the direction of self-rotation)
    lat = np.around(90.-theta_point*180/np.pi, 10) # southern hemisphere has negative lattitude
    lon = (phi_point/dtr - (phi_star+180))%360 # substellar point is 180E

    # calculate emission viewing angle direction vector (-y axis) (Observer direction vecto)
    x_observer = 0.
    y_observer = -1.0
    z_observer = 0.0
    v_observer = np.array([x_observer,y_observer,z_observer])

    ### calculate azimuth angle
    # Rotate frame clockwise by phi_point about z (v_point is now x-axis)
    v_star_1=rotatez(v_star,-phi_point)
    v_point_1=rotatez(v_point,-phi_point)
    v_observer_1=rotatez(v_observer,-phi_point)

    # Rotate frame clockwise by theta_point about y (v_point is now z-axis )
    v1B=rotatey(v_star_1,-theta_point)
    v2B=rotatey(v_point_1,-theta_point)
    v3B=rotatey(v_observer_1,-theta_point)

    # thetsolB=np.arccos(v1B[2])
    # thetobsB=np.arccos(v3B[2])
    phisolB=arctan(v1B[0], v1B[1])
    phiobsB=arctan(v3B[0], v3B[1])

    azi = abs(phiobsB-phisolB)
    if(azi > np.pi):
        azi=2*np.pi-azi

    # Ensure azi meets convention where azi=0 means forward-scattering
    azi = np.pi-azi

    return zen, azi, lat, lon

@jit(nopython=True)
def gauss_lobatto_weights(phase, nmu):
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
    assert nmu <=5, "Currently cannot do more than 5 quadrature rings"
    assert nmu >=2, "Need at least 2 quadrature rings"
    phase = phase%360
    dtr = np.pi/180 # degree to radiance conversion factor
    delR = 1./nmu

    # set up the output arrays
    nsample = 1000 # large array size to hold calculations
    tablat = np.zeros(nsample) # latitudes
    tablon = np.zeros(nsample) # longitudeds
    tabzen = np.zeros(nsample) # zenith angle in quadrature scheme
    tabsol = np.zeros(nsample) # solar zenith angle
    tabazi = np.zeros(nsample) # solar azimuth angle (scattering phase angle?)
    tabwt = np.zeros(nsample)  # weight of each sample

    # list the gauss labatto quadrature points and weights
    if nmu == 2:
        mu = [0.447213595499958,1.000000]                   # cos zenith angle
        wtmu = [0.8333333333333333,0.166666666666666666]    # corresponding weights
    if nmu == 3:
        mu = [0.28523151648064509,0.7650553239294646,1.0000]
        wtmu = [0.5548583770354863,0.3784749562978469,0.06666666666666666]
    if nmu == 4:
        mu = [0.2092992179024788,0.5917001814331423,0.8717401485096066,
              1.00000]
        wtmu = [0.4124587946587038,0.3411226924835043,0.2107042271435060,
                0.035714285714285]
    if nmu == 5:
        mu = [0.165278957666387,0.477924949810444,0.738773865105505,
              0.919533908166459,1.00000000000000]
        wtmu = [0.327539761183898,0.292042683679684,0.224889342063117,
                0.133305990851069,2.222222222222220E-002]


    # # define a range of angles from 0 to 360
    # """
    # thet = np.arange(361)
    # """

    # """
    # xx = np.around(np.cos(thet*dtr), 14)
    # zz = np.around(np.sin(thet*dtr), 14)
    # """

    # trace out the day/night terminator
    z_term = np.linspace(-1,1,201) # pick out some z coordinates for the terminator
    if 0<= phase <= 180:
        # terminator is to the left of z axis
        theta_term = 2*np.pi - np.arccos(z_term)
    else:
        # terminator in the right side of the disc
        theta_term = np.arccos(z_term)
    x_term = np.sin(theta_term) * np.around(np.cos(phase*dtr),14) # x coords of terminator
    r_term = np.sqrt(x_term**2+z_term**2) # radial coords of terminator
    rmin = min(r_term) # least radius (is on x axis)

    # define FOV averaging points
    isample = 0
    for imu in range(0, nmu): # quadrature rings
        r_quad = np.sqrt(1.-mu[imu]**2) # quadrature radius (from small to large)
        half_circum = np.pi*r_quad # half the circumference

        # see if the quadrature ring intersects the terminator
        # if so, find the intersection point and place a sample point there
        if r_quad > rmin: # quadrature ring intersects the terminator
            ikeep = np.where(r_term<=r_quad)
            ikeep = ikeep[0] # index of the points on the terminator with radius > r_quad
            i_intersect = np.array([ikeep[0], ikeep[-1]]) # index of two intersectionns
            x_intersect = x_term[i_intersect] # x coordinates of intersection
            z_intersect = z_term[i_intersect] # z coordinates of intersection

            # take the intersection in the upper hemisphere
            if z_intersect[1] > 0:
                alpha_intersect = arctan(x_intersect[1],z_intersect[1])/dtr
            else:
                alpha_intersect = arctan(x_intersect[0],z_intersect[0])/dtr

            # place the sample points on the quadrature rings on either side of the intersection
            nalpha1 = int(0.5+half_circum*(alpha_intersect/180.0)/delR) # round up; separation ~ R/nmu
            nalpha2 = int(0.5+half_circum*((180.-alpha_intersect)/180.0)/delR)

            # at least 1 point either side of the intersection
            if(nalpha1 < 2):
                nalpha1=2
            if(nalpha2 < 2):
                nalpha2=2

            # set the alphas of the sample points on current quadrature ring
            nalpha = nalpha1+nalpha2-1 # intersection point double counted
            alpha1 = alpha_intersect/(nalpha1-1) * np.arange(nalpha1)
            alpha2 = alpha_intersect+(180.-alpha_intersect)/(nalpha2-1) * np.arange(nalpha2)
            alpha2 = alpha2[1:(nalpha2)] # intersect was counted twice
            alpha_sample_list = np.concatenate((alpha1,alpha2))

        else: # quadrature ring does not intersect terminator
            if(half_circum > 0.0):
                nalpha = int(0.5+half_circum/delR)
                alpha_sample_list = 180*np.arange(nalpha)/(nalpha-1)
            else:
                nalpha=1

        if(nalpha > 1): # more than one sample on the quadrature ring

            for ialpha in np.arange(0,nalpha):

                alpha_sample = alpha_sample_list[ialpha]

                thetasol_sample, azi_sample, lat_sample, lon_sample \
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
                tabsol[isample] = thetasol_sample/dtr # sample stellar zenith angle
                tabazi[isample] = azi_sample/dtr # sample stellar azimuth angle
                tabwt[isample] = 2*mu[imu]*wtmu[imu]*wt_azi # sample weight
                isample = isample+1

        else:
            alpha_sample = 0.
            thetasol_sample,azi_sample, lat_sample,lon_sample \
                = generate_angles(phase,r_quad,alpha_sample)
            if(tabzen[isample] == 0.0):
                azi_sample = 180.
            tablat[isample] = lat_sample
            tablon[isample] = lon_sample
            tabzen[isample] = np.arccos(mu[imu])/dtr
            tabsol[isample] = thetasol_sample/dtr
            tabazi[isample] = azi_sample
            tabwt[isample] = 2*mu[imu]*wtmu[imu]
            isample = isample+1

    nav = isample
    wav = np.zeros((6,isample))
    sum=0.
    for i in np.arange(0,isample):
        wav[0,i]=tablat[i]              # 0th array is lattitude
        wav[1,i]=tablon[i]%360          # 1st array is longitude
        wav[2,i]=tabsol[i]              # 2nd array is stellar zenith angle
        wav[3,i]=tabzen[i]              # 3rd array is emission zenith angle
        wav[4,i]=tabazi[i]              # 4th array is stellar azimuth angle
        wav[5,i]=tabwt[i]               # 5th array is weight
        sum = sum+tabwt[i]

    for i in range(isample):            # normalise weights so they add up to 1
        wav[5,i]=wav[5,i]/sum

    # wav = np.around(wav,decimals=8)

    return nav, wav
