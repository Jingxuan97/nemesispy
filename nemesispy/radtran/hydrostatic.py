import numpy as np

G = 6.67430e-11
AMU = 1.66054e-27
K_B = 1.38065e-23

def find_nearest(input_array, target_value):
    """
    Find the closest value in an array

    Parameters
    ----------
        input_array : ndarray/list
            An array of numbers.
        target_value :
            Value to search for


    Returns
    -------
        idx : ndarray
            Index of closest_value within array
        array[idx] : ndarray
            Closest number to target_value in the input array
    """
    array = np.asarray(input_array)
    idx = (np.abs(array - target_value)).argmin()
    return array[idx], idx

def NEWGRAV(h, radius, mass, lat_in=None):
    """TBD
      Sets the gravitational acceleration based on selected planet and
      latitude. Gravity is calculated normal to the surface, and corrected
      for rotational effects. The input latitude is assumed planetographic.

    Parameters
    ----------

    Returns
    -------
    """
    G = 6.67430e-11
    g = G*mass/(radius+h)**2
    gravity = g
    return gravity

def calc_grav(M_plt,):
    G = 6.67430e-11

def XHYDROSTATH(AMFORM,IPLANET,LATITUDE,NPRO,NVMR,MOLWT,
    IDGAS,ISOGAS,H,P,T,VMR,SCALE,
    radius,mass):
    XDEPTH = 2
    while XDEPTH > 1:
        # NPRO and NVMR no need to be passed
        ATDEPTH = H[-1] - H[0] # NPRO = len(H)

        # First find level closest ot zero altitude
        DELH = 1000.0
        for I in range(NPRO):
            X = abs(H[I])
            if X < DELH:
                DELH = X
                JZERO = I

            """# alternatively...
            min_index = np.argmin(abs(H))
            if H[min_index] < DELH:
                DELH = X
            """
            # VMR = NPRO x NVMR
            XVMR = np.zeros(NVMR)
            for J in range(NVMR):
                XVMR[J] = VMR[I,J]

            """
            need to flesh out
            """
            g = NEWGRAV(H[I],radius,mass,lat_in=None)

            XMOLWT = 2.3 * AMU

            scale_height = K_B * T[I]/(XMOLWT*g)

        if JZERO > 1 and JZERO < NPRO:
            H[JZERO] = 0.0

        for K in np.range(JZERO+1,NPRO):
            """SH average scale height"""
            SH = 0.5*(scale_height[K-1]+scale_height[I+1])
            """hydrostatic equilibrium"""
            H[K] = H[K+1] - SH*np.log(P[K]/P[K+1])

        ATDEPTH1 = H[-1] - H[0]
        XDEPTH = 100 * abs((ATDEPTH1-ATDEPTH)/ATDEPTH)


def simple_grav(h, radius, mass):
    """TBD
      Sets the gravitational acceleration based on selected planet and
      latitude. Gravity is calculated normal to the surface, and corrected
      for rotational effects. The input latitude is assumed planetographic.

    Parameters
    ----------

    Returns
    -------
    """

    g = G*mass/(radius+h)**2
    gravity = g
    return gravity

def simple_hydro(H,P,T,VMR,radius,mass):
    """
    Adjust a given altitude profile H according to hydrostatic equilibrium
    using the given pressure profile P and temperature profile T.
    """

    NPRO,NVMR = VMR.shape
    scale_height = np.zeros(NPRO)

    # First find level closest ot zero altitude
    alt0,ialt = find_nearest(H,0.0)
    if ( (alt0>0.0) & (ialt>0)):
        ialt = ialt -1

    # defns
    XDEPTH = 2
    XMOLWT = 2.3 * AMU

    while XDEPTH > 1:
        h = np.zeros(NPRO)
        p = np.zeros(NPRO)
        h[:] = H
        p[:] = P

        #Calculating the atmospheric depth
        ATDEPTH = H[-1] - H[0]

        #Calculate the gravity at each altitude level
        gravity =  simple_grav(H, radius, mass)

        #Calculate the scale height
        scale = K_B*T/(XMOLWT*gravity)

        if ialt > 0 and ialt < NPRO-1 :
            h[ialt] = 0.0

        nupper = NPRO - ialt - 1

        for i in range(ialt+1,NPRO):
            sh = 0.5 * (scale[i-1] + scale[i])
            #self.H[i] = self.H[i-1] - sh * np.log(self.P[i]/self.P[i-1])
            h[i] = h[i-1] - sh * np.log(p[i]/p[i-1])

        for i in range(ialt-1,-1,-1):
            sh = 0.5 * (scale[i+1] + scale[i])
            #self.H[i] = self.H[i+1] - sh * np.log(self.P[i]/self.P[i+1])
            h[i] = h[i+1] - sh * np.log(p[i]/p[i+1])


        atdepth1 = h[-1] - h[0]

        XDEPTH = 100.*abs((atdepth1-ATDEPTH)/ATDEPTH)
        print('xdepth',XDEPTH)
        H = h[:]

    return H


        # FORTRAN TRANSCRIPTION
        # DELH = 1000.0
        # for I in range(NPRO):
        #     X = abs(H[I])
        #     if X < DELH:
        #         DELH = X
        #         JZERO = I

        #     """# alternatively...
        #     min_index = np.argmin(abs(H))
        #     if H[min_index] < DELH:
        #         DELH = X
        #     """
        #     """
        #     # VMR = NPRO x NVMR
        #     XVMR = np.zeros(NVMR)
        #     for J in range(NVMR):
        #         XVMR[J] = VMR[I,J]
        #     """

        #     """
        #     need to flesh out
        #     """
        #     g = simple_grav(H[I],radius,mass)

        #     XMOLWT = 2.3 * AMU

        #     scale_height[I] = K_B * T[I]/(XMOLWT*g)

        # if JZERO > 1 and JZERO < NPRO:
        #     H[JZERO] = 0.0

        # # for K in np.arange(JZERO+1,NPRO):
        # for K in np.arange(JZERO,NPRO-1):
        #     """SH average scale height"""
        #     SH = 0.5*(scale_height[K-1]+scale_height[I+1])
        #     """hydrostatic equilibrium"""
        #     H[K] = H[K+1] - SH*np.log(P[K]/P[K+1])

        # ATDEPTH1 = H[-1] - H[0]
        # XDEPTH = 100 * abs((ATDEPTH1-ATDEPTH)/ATDEPTH)

    return H

def adjust_hydrostatH(self):
    """
    Subroutine to rescale the heights of a H/P/T profile according to
    the hydrostatic equation above and below the level where height=0.

    Parameters
    ----------
    self.H
    self.NP
    self.P
    self.calc_grav()
    """

    #First find the level closest to the 0m altitude
    alt0,ialt = find_nearest(self.H,0.0)
    if ( (alt0>0.0) & (ialt>0)):
        ialt = ialt -1

    xdepth = 100.
    while xdepth>1:

        h = np.zeros(self.NP)
        p = np.zeros(self.NP)
        h[:] = self.H
        p[:] = self.P

        #Calculating the atmospheric depth
        atdepth = h[self.NP-1] - h[0]

        #Calculate the gravity at each altitude level
        self.calc_grav()

        #Calculate the scale height
        R = const["R"]
        scale = R * self.T / (self.MOLWT * self.GRAV)   #scale height (m)

        p[:] = self.P
        if ((ialt>0) & (ialt<self.NP-1)):
            h[ialt] = 0.0

        nupper = self.NP - ialt - 1
        for i in range(ialt+1,self.NP):
            sh = 0.5 * (scale[i-1] + scale[i])
            #self.H[i] = self.H[i-1] - sh * np.log(self.P[i]/self.P[i-1])
            h[i] = h[i-1] - sh * np.log(p[i]/p[i-1])

        for i in range(ialt-1,-1,-1):
            sh = 0.5 * (scale[i+1] + scale[i])
            #self.H[i] = self.H[i+1] - sh * np.log(self.P[i]/self.P[i+1])
            h[i] = h[i+1] - sh * np.log(p[i]/p[i+1])

        #atdepth1 = self.H[self.NP-1] - self.H[0]
        atdepth1 = h[self.NP-1] - h[0]

        xdepth = 100.*abs((atdepth1-atdepth)/atdepth)

        self.H = h[:]

        #Re-Calculate the gravity at each altitude level
        self.calc_grav()

def calc_grav(self):
    """
    Subroutine to calculate the gravity at each level following the method
    of Lindal et al., 1986, Astr. J., 90 (6), 1136-1146
    """

    #Reading data and calculating some parameters
    Grav = const["G"]
    data = planet_info[str(self.IPLANET)]
    xgm = data["mass"] * Grav * 1.0e24 * 1.0e6
    xomega = 2.*np.pi / (data["rotation"]*24.*3600.)
    xellip=1.0/(1.0-data["flatten"])
    Jcoeff = data["Jcoeff"]
    xcoeff = np.zeros(3)
    xcoeff[0] = Jcoeff[0] / 1.0e3
    xcoeff[1] = Jcoeff[1] / 1.0e6
    xcoeff[2] = Jcoeff[2] / 1.0e8
    xradius = data["radius"] * 1.0e5   #cm
    isurf = data["isurf"]
    name = data["name"]


    #Calculating some values to account for the latitude dependence
    lat = 2 * np.pi * self.LATITUDE/360.      #Latitude in rad
    latc = np.arctan(np.tan(lat)/xellip**2.)   #Converts planetographic latitude to planetocentric
    slatc = np.sin(latc)
    clatc = np.cos(latc)
    Rr = np.sqrt(clatc**2 + (xellip**2. * slatc**2.))  #ratio of radius at equator to radius at current latitude
    r = (xradius+self.H*1.0e2)/Rr    #Radial distance of each altitude point to centre of planet (cm)
    radius = (xradius/Rr)*1.0e-5     #Radius of the planet at the given distance (km)

    self.RADIUS = radius * 1.0e3

    #Calculating Legendre polynomials
    pol = np.zeros(6)
    for i in range(6):
        Pn = legendre(i+1)
        pol[i] = Pn(slatc)

    #Evaluate radial contribution from summation
    # for first three terms,
    #then subtract centrifugal effect.
    g = 1.
    for i in range(3):
        ix = i + 1
        g = g - ((2*ix+1) * Rr**(2 * ix) * xcoeff[ix-1] * pol[2*ix-1])

    gradial = (g * xgm/r**2.) - (r * xomega**2. * clatc**2.)

    #Evaluate latitudinal contribution for
    # first three terms, then add centrifugal effects

    gtheta1 = 0.
    for i in range(3):
        ix = i + 1
        gtheta1 = gtheta1 - (4. * ix**2 * Rr**(2 * ix) * xcoeff[ix-1] * (pol[2*ix-1-1] - slatc * pol[2*ix-1])/clatc)

    gtheta = (gtheta1 * xgm/r**2) + (r * xomega**2 * clatc * slatc)

    #Combine the two components and write the result
    gtot = np.sqrt(gradial**2. + gtheta**2.)*0.01   #m/s2

    self.GRAV = gtot