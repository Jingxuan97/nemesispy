import numpy as np
import arctan
import generate_angles


"""
antistellar point is at 0E, substellar point is at 180E, increase in the direction
of self rotation

orbital phase is 0 at primary transit and 180 at secondary eclipse, increases in
the direction of orbital motion

At primary transit,
x-axis points towards 3 o'clock.
y-axis points towards the star.
z-axis points towards the north pole.

Polar coordinates defined as usual:
r is the radial distance
theta is measured clockwiseky from the z-axis.
phi is measured anticlockwise from x-axis.

Further introduce 2 coordinates on the disc
rho: projected distance between the point and the centre of the disc
alpha: argument on the projected disc, measured anticlockwise from x axis

coordinate systems do not rotate with respect to the observer

x = r * sin(theta) * cos(phi) = rho * cos(alpha)
y = r * sin(theta) * sin(phi) = rho * sin(alpha)
z = r * cos(theta)
"""

def gauss_lobatto_weights(phase, nmu):
    """
    Given the orbital phase, calculates the coordinates and weights of the points
    on a disc needed to compute the disc integrated radiance.

    The points are chosen on a number of rings according to Gauss-Lobatto quadrature
    scheme, and spaced on the rings according to trapezium rule.

    Orbital phase (stellar phase angle) increases from 0 at primary transit
    to 180 at secondary eclipse.

    Planetocentric longitude is 180E at the substellar point and 0E at the antistellar
    point. East is in the direction of the planet's self-rotation.

    # We treat the visible disc as a circle with dimensionless radius 1.
    # The point on the observed planetary disc is specified by three input parameters.
    #     1) phase: orbital phase, which increases from 0 degree at primary transit to
    #     180 degree at secondary eclipse.
    #     2) rho: fraction radius, which is the projected distance between the point
    #     and the centre of the disc, Takes value between 0 and 1 inclusive.
    #     3) alpha: argument of the point, measured anticlockwise from 3 o'clock.

    # We use the following coordinate system, in a frame centred at the planetary centre.
    # Consider the primary transit. At this moment, the orbital phase angle is 0.
    # WLOG assume the orbit is anticlockwise, and we define the following:
    #     x-axis points towards 3 o'clock.
    #     y-axis points towards the star.
    #     z-axis points towards the north pole.
    #     theta is measured clockwiseky from the z-axis.
    #     phi is measured anticlockwise from x-axis.
    # As the star orbiting around the planet in our frame, we fix the axes so that
    # they are not rotating. Equivalently, to an intertial observer, the centre of
    # our frame moves with the centre of the planet, but the axes do not rotate.


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
        mu = [0.2092992179024788,0.5917001814331423,0.8717401485096066,1.00000]
        wtmu = [0.4124587946587038,0.3411226924835043,0.2107042271435060,0.035714285714285]
    if nmu == 5:
        mu = [0.165278957666387,0.477924949810444,0.738773865105505,0.919533908166459,1.00000000000000]
        wtmu = [0.327539761183898,0.292042683679684,0.224889342063117,0.133305990851069,2.222222222222220E-002]


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
            alpha_sample_list = np.concatenate([alpha1,alpha2])

        else: # quadrature ring does not intersect terminator
            if(half_circum > 0.0):
                nalpha = int(0.5+half_circum/delR)
                alpha_sample_list = 180*np.arange(nalpha)/(nalpha-1)
            else:
                nalpha=1

        if(nalpha > 1): # more than one sample on the quadrature ring

            # sum = 0.

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

                # sum = sum+wt_azi

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
    wav = np.zeros([6,isample])
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

    return nav, np.around(wav,8)

def subdiscweightsv3(xphase, nmu=3):
    """
    Python routine for setting up geometry and weights for observing a planet
    at a variable stellar phase angle xphase.

    Code splits disc into a number of rings using Gauss-Lobatto quadrature and then
    does azimuth integration using trapezium rule.

    Orbital phase (stellar phase angle) increases from 0 at primary transit
    to 180 at secondary eclipse.

    Planetary longitude is 180E at the substellar point and 0E at the antistellar
    point. East is in the direction of the planet's self-rotation.

    Parameters
    ----------
    xphase : real
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
    xphase = xphase%360
    dtr = np.pi/180
    delR = 1./nmu
    nsample = 1000             # large array size to hold calculations
    tablat = np.zeros(nsample) # latitudes
    tablon = np.zeros(nsample) # longitudeds
    tabzen = np.zeros(nsample) # zenith angle in quadrature scheme
    tabsol = np.zeros(nsample) # solar zenith angle
    tabazi = np.zeros(nsample) # solar azimuth angle (scattering phase angle?)
    tabwt = np.zeros(nsample)  # weight of each sample

    if nmu == 2:
        mu = [0.447213595499958,1.000000]                   # cos zenith angle
        wtmu = [0.8333333333333333,0.166666666666666666]    # corresponding weights
    if nmu == 3:
        mu = [0.28523151648064509,0.7650553239294646,1.0000]
        wtmu = [0.5548583770354863,0.3784749562978469,0.06666666666666666]
    if nmu == 4:
        mu = [0.2092992179024788,0.5917001814331423,0.8717401485096066,1.00000]
        wtmu = [0.4124587946587038,0.3411226924835043,0.2107042271435060,0.035714285714285]
    if nmu == 5:
        mu = [0.165278957666387,0.477924949810444,0.738773865105505,0.919533908166459,1.00000000000000]
        wtmu = [0.327539761183898,0.292042683679684,0.224889342063117,0.133305990851069,2.222222222222220E-002]

    # define limb of planet
    thet = np.arange(361)
    xx = np.around(np.cos(thet*dtr), 14)
    zz = np.around(np.sin(thet*dtr), 14)

    # define terminator
    zt = np.linspace(-1,1,201)              # r cos theta (z coordinates of the terminator)
    angle = np.arccos(zt)+np.pi/2.          # theta
    r1 = np.around(np.cos(angle),14)        # r sin theta
    xt = r1 * np.around(np.cos((xphase)*np.pi/180.), 14) # r sin theta sin xphase (x coordinates of the terminator)

    if (xphase > 180.0):
        xt = -xt # flip after phase = 180

    rr = np.sqrt(xt**2+zt**2)   # radial coordinate of the determinator
    rmin = min(rr)              # least radius (on x axis )

    isample = 0
    for imu in range(0, nmu):       # quadrature rings
        r = np.sqrt(1.-mu[imu]**2)  # quadrature radius (from small to large)
        circumh = np.pi*r	        # half the circumference
        xx = np.around(r*np.cos(thet*dtr), 14)
        zz = np.around(r*np.sin(thet*dtr), 14)

        if r > rmin:  # quadrature ring intersects terminator.
            # find the intersection and place a sample point there
            ikeep = np.where(rr<=r)
            ikeep = ikeep[0]
            ir = np.array([ikeep[0], ikeep[-1]])    # index of two intersectionns
            xr = xt[ir]                             # coordinate of intersection
            zr = zt[ir]
            if zr[1] > 0:                           # take the intersection in the upper hemisphere
                phi = arctan(xr[1],zr[1])/dtr
            else:
                phi = arctan(xr[0],zr[0])/dtr

            # split the quadrature rings with sample points
            nphi1 = int(0.5+circumh*(phi/180.0)/delR) # round up; separation ~ R/nmu
            nphi2 = int(0.5+circumh*((180.-phi)/180.0)/delR)

            # at least 1 point either side of the intersection
            if(nphi1 < 2):
                nphi1=2
            if(nphi2 < 2):
                nphi2=2

            nphi = nphi1+nphi2-1 # intersection point double counted
            phi1 = phi*np.arange(nphi1)/(nphi1-1)
            phi2 = phi+(180.-phi)*np.arange(nphi2)/(nphi2-1)
            phi2 = phi2[1:(nphi2)]
            phix = np.concatenate([phi1,phi2])

        else:   # quadrature ring does not intersect terminator
            if(circumh > 0.0):
                nphi = int(0.5+circumh/delR)
                phix = 180*np.arange(nphi)/(nphi-1)
            else:
                nphi=1

        if(nphi > 1):

            sum = 0.
            for iphi in np.arange(0,nphi):
                xphi = phix[iphi]
                xp = r*np.cos(xphi*dtr)
                yp = r*np.sin(xphi*dtr)

                thetasol, xazi, xlat, xlon = generate_angles(xphase,r,xphi)

                # trapezium rule weights
                if(iphi == 0):
                    wt = (phix[iphi+1]-phix[iphi])/2.0
                else:
                    if(iphi == nphi-1):
                        wt = (phix[iphi]-phix[iphi-1])/2.0
                    else:
                        wt = (phix[iphi+1]-phix[iphi-1])/2.0


                wtazi= wt/180.                                  # sample azimuthal weight
                sum = sum+wtazi
                tablat[isample] = xlat                          # sample lattitude
                tablon[isample] = xlon                          # sample longitude
                tabzen[isample] = np.arccos(mu[imu])/dtr        # sample emission zenith angle
                tabsol[isample] = thetasol/dtr                  # sample stellar zenith angle
                tabazi[isample] = xazi/dtr                      # sample stellar azimuth angle
                tabwt[isample] = 2*mu[imu]*wtmu[imu]*wtazi      # sample weight
                isample = isample+1

        else:
            xphi = 0.
            thetasol,xazi, xlat,xlon = generate_angles(xphase,r,xphi)
            if(tabzen[isample] == 0.0):
                xazi = 180.
            tablat[isample] = xlat
            tablon[isample] = xlon
            tabzen[isample] = np.arccos(mu[imu])/dtr
            tabsol[isample] = thetasol/dtr
            tabazi[isample] = xazi
            tabwt[isample] = 2*mu[imu]*wtmu[imu]
            isample = isample+1

    nav = isample
    wav = np.zeros([6,isample])
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

    return nav, np.around(wav,8)


"""
nav, table = subdiscweightsv3(22.5*2, 5)
np.savetxt('test.dat',table.T,fmt='%.6e')
"""