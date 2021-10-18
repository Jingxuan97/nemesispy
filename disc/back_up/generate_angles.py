import numpy as np
"""
clean up the thetasol_gen_azi_latlong routine
"""
def generate_angles(phase,rho,alpha):
    """
    Finds the stellar zenith angle, stellar azimuth angle, lattitude and longitude
    of a chosen point on the visible disc of a planet under observation. The planet
    is assumed to be tidally locked, and is observed on an edgy-on orbit.

    We treat the visible disc as a circle with dimensionless radius 1.
    The point on the observed planetary disc is specified by three input parameters.
        1) phase: orbital phase, which increases from 0 degree at primary transit to
        180 degree at secondary eclipse.
        2) rho: fraction radius, which is the projected distance between the point
        and the centre of the disc, Takes value between 0 and 1 inclusive.
        3) alpha: argument of the point, measured anticlockwise from 3 o'clock.

    We use the following coordinate system, in a frame centred at the planetary centre.
    Consider the primary transit. At this moment, the orbital phase angle is 0.
    WLOG assume the orbit is anticlockwise, and we define the following:
        x-axis points towards 3 o'clock.
        y-axis points towards the star.
        z-axis points towards the north pole.
        theta is measured clockwiseky from the z-axis.
        phi is measured anticlockwise from x-axis.
    As the star orbiting around the planet in our frame, we fix the axes so that
    they are not rotating. Equivalently, to an intertial observer, the centre of
    our frame moves with the centre of the planet, but the axes do not rotate.

    Since the planet is tidally locked, the substellar and antistellar points are
    stationary without respect to the planet. We define the planetocentric longtitude
    to be 180E at the substellar point and 0E at the antistellar point. East is in
    the direction of the planet' self rotation.

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
    phase = phase%360
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

# old version
def thetasol_gen_azi_latlong(xphase,rho,alpha):
    """
    Calculate the stellar zenith angle, stellar aziumuth angle, lattitude and
    longtitude for a point on a TRANSITING TIDALLY LOCKED planet's surface
    illuminated by its star at variable phase angle.

    Orbital phase (stellar phase angle) increases from 0 at primary transit
    to 180 at secondary eclipse.

    Planetary longitude is 180E at the substellar point and 0E at the antistellar
    point. East is in the direction of the planet's self-rotation.

    Note on the geometry used in this routine:
        Imagine a frame centred at the planet at the moment of primary transit.
        At this point, the stellar phase is 0.
        We are viewing the orbital plane edge on; WLOG assume the orbit
        is anticlockwise, then
            x-axis points towards 3 o'clock.
            y-axis points towards the star.
            z-axis points towards north.
            theta is measured from the z-axis conventionally.
            phi is measured anticlockwise from x-axis.
        Now imagine the star orbiting around the planet; our frame moves with the
        centre of the planet but is not rotating. In this routine we assume
        the planetary surface is a perfect spherical shell with a dimensionless
        radius 1.

    Parameters
    ----------
    xphase : real
        Stellar phase/orbital phase in degrees.
        0=parimary transit and increase to 180 at secondary eclipse.
    rho	: real
        Fractional radius of required position on disc.
    alpha	: real
        Position angle of point on visible disc (degrees), measured
        anticlockwise from 3 o'clock position

    Returns
    -------
    thetasol : real
        Computed solar zenith angle (radians)
    azi	: real
        Computed solar azimuth angle (radians). Uses convention that
        forward scatter = 0.
    lat	: real
        Latitude
    lon	: real
        Longitude
    """
    xphase = xphase%360
    dtr = np.pi/180.
    assert rho <=1, "Fractional radius should be less or equal to 1"

    ### calculate solar direction vector using spherical polars (r,theta,phi)
    thetas = np.pi/2.               # star lie in planet's equitorial plane
    phi_star = 90.0 + xphase
    x1 = np.sin(thetas)*np.cos(phi_star*dtr)
    y1 = np.sin(thetas)*np.sin(phi_star*dtr)
    z1 = np.cos(thetas)
    v1 = np.array([x1,y1,z1])

    ### calculate sample position vector using spherical polars (r=1,theta,phi)
    theta = np.arccos(rho*np.sin(alpha*dtr)) # planetary zenith angle of spot on surface
    if np.sin(theta) != 0.0:
        cos_phi = rho*np.cos(alpha*dtr)/abs(np.sin(theta)) # changed
        phi = (-np.arccos(cos_phi))%(2*np.pi) # azimuth angle of spot on surface / on our side
    else:
        phi = 0.0 # sin(theta) = 0 at north polt
    x2 = np.sin(theta)*np.cos(phi)
    y2 = np.sin(theta)*np.sin(phi)
    z2 = np.cos(theta)
    v2 = np.array([x2,y2,z2])

    ### calculate angle between solar position vector and local normal
    # i.e. thetasol solar zenith angle
    inner_product = np.sum(v1*v2)
    thetasol = np.arccos(inner_product)
    thetasol = np.around(thetasol, 10)

    ### calculate latitude and longitude of the spot
    # (sub-stellar point = 180E, anti-stellar point = 0E, longtitudes in the direction of self-rotation)
    lat = np.around(90.-theta*180/np.pi, 10)
    lon = (phi/dtr - (phi_star+180))%360

    ### calculate emission viewing angle direction vector (-y axis) (Observer direction vecto)
    x3 = 0.
    y3 = -1.0
    z3 = 0.0
    v3 = np.array([x3,y3,z3])

    ### calculate azimuth angle
    # Rotate frame clockwise by phi about z (v2 is now x-axis)
    v1A=rotatez(v1,-phi)
    v2A=rotatez(v2,-phi)
    v3A=rotatez(v3,-phi)

    # Rotate frame clockwise by theta about y (v2 is now z-axis )
    v1B=rotatey(v1A,-theta)
    v2B=rotatey(v2A,-theta)
    v3B=rotatey(v3A,-theta)

    # thetsolB=np.arccos(v1B[2])
    # thetobsB=np.arccos(v3B[2])
    phisolB=arctan(v1B[0], v1B[1])
    phiobsB=arctan(v3B[0], v3B[1])

    azi = abs(phiobsB-phisolB)
    if(azi > np.pi):
        azi=2*np.pi-azi

    # Ensure azi meets convention where azi=0 means forward-scattering
    azi = np.pi-azi
    return thetasol, azi, lat, lon