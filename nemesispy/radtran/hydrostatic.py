import sys
sys.path.append('/Users/jingxuanyang/Desktop/Workspace/nemesispy2022/')
import numpy as np
from nemesispy.data.constants import K_B, AMU, G, M_JUP, R_JUP
from nemesispy.radtran.utils import calc_mmw
# G = 6.67430e-11
# AMU = 1.66054e-27
# K_B = 1.38065e-23

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

def calc_grav_simple(h, M_plt, R_plt):
    """
    TBD
    Sets the gravitational acceleration based on selected planet and
    latitude. Gravity is calculated normal to the surface, and corrected
    for rotational effects. The input latitude is assumed planetographic.

    Parameters
    ----------

    Returns
    -------
    """
    g = G*M_plt/(R_plt+h)**2
    gravity = g
    return gravity

from scipy.special import legendre
def calc_grav(h, lat, M_plt, R_plt,
    J2=0, J3=0, J4=0, flatten=0, rotation=1e5):
    """


    Parameters
    ----------
    h : TYPE
        DESCRIPTION.
    lat : TYPE
        DESCRIPTION.
    M_plt : TYPE
        DESCRIPTION.
    R_plt : TYPE
        DESCRIPTION.
    J2 : TYPE, optional
        DESCRIPTION. The default is 0.
    J3 : TYPE, optional
        DESCRIPTION. The default is 0.
    J4 : TYPE, optional
        DESCRIPTION. The default is 0.
    flatten : TYPE, optional
        DESCRIPTION. The default is 0.
    rotation : TYPE, optional
        DESCRIPTION. The default is 1e5.

    Returns
    -------
    gtot : TYPE
        DESCRIPTION.

    """

    G = 6.67430e-11
    xgm = G*M_plt
    xomega = 2.*np.pi/(rotation*24*3600)
    xellip = 1/(1-flatten)
    xcoeff = np.zeros(3)
    xcoeff[0] = J2
    xcoeff[1] = J3
    xcoeff[2] = J4
    xradius = R_plt


    # account for latitude dependence
    lat = 2 * np.pi * lat/360 # convert to radiance
    ## assume for now that we use planetocentric latitude to begin with?
    # latc = np.arctan(np.tan(lat)/xellip**2.)   #Converts planetographic latitude to planetocentric
    # slatc = np.sin(latc)
    # clatc = np.cos(latc)
    slat = np.sin(lat)
    clat = np.cos(lat)
    # calculate the ratio of radius at equator to radius at current latitude
    Rr = np.sqrt(clat**2 + (xellip**2. * slat**2))
    # print('Rr',Rr)
    r = (h+xradius)/Rr
    radius = (xradius/Rr)

    # calculate legendre polynomials
    pol = np.zeros(6)
    for i in range(6):
        Pn = legendre(i+1)
        pol[i] = Pn(slat)
    # print('pol',pol)

    # Evaulate radial contribution from summation for first three terms
    # then suubtract centrifugal effects

    # note that if Jcoeffs are 0 then the following calculation can be
    # skipped, i.e. g = 1 if Jcoeffs are 0 and period is 1e5 (i.e. unknown)
    g = 1
    for i in range(3):
        ix = i + 1
        g = g - ((2*ix+1)*Rr**(2*ix)*xcoeff[ix-1]*pol[2*ix-1])

    gradial = (g*xgm/r**2) - (r*xomega**2 * clat**2.)
    # print('graial',gradial)

    # Evaluate latitudinal conttribution for first three terms,
    # then add centriifugal effects

    gtheta1 = 0.
    for i in range(3):
        ix = i+1
        gtheta1 = gtheta1 \
        -(4*ix**2*Rr**(2*ix)*xcoeff[ix-1]*(pol[2*ix-1-1]-slat*pol[2*ix-1])/clat)

    # print('xomega',xomega)
    # print('gtheta1',gtheta1)
    gtheta = (gtheta1 * xgm/r**2) + (r*xomega**2 * clat * slat)

    # combine the two components
    gtot = np.sqrt(gradial**2 + gtheta**2)

    return gtot


def adjust_hydrostatH(H, P, T, ID, VMR, M_plt, R_plt):
    """
    Adjust a given altitude profile H according to hydrostatic equilibrium
    using the given pressure profile P and temperature profile T.
    """

    # note number of profile points and number of gases
    NPRO,NVMR = VMR.shape

    # initialise array for scale heights
    scale_height = np.zeros(NPRO)

    # First find level closest ot zero altitude
    alt0,ialt = find_nearest(H,0.0)
    if ( (alt0>0.0) & (ialt>0)):
        ialt = ialt -1

    #Calculate the mean molecular weight at each level
    XMOLWT = np.zeros(NPRO)
    for ipro in range(NPRO):
        XMOLWT[ipro] = calc_mmw(ID,VMR[ipro,:],ISO=None)

    # iterate until hydrostatic equilibrium
    XDEPTH = 2
    while XDEPTH > 1:

        h = np.zeros(NPRO)
        p = np.zeros(NPRO)
        h[:] = H
        p[:] = P

        #Calculating the atmospheric model depth
        ATDEPTH = h[-1] - h[0]

        #Calculate the gravity at each altitude level
        gravity = np.zeros(NPRO)
        gravity[:] =  calc_grav_simple(h=h[:], M_plt=M_plt, R_plt=R_plt)

        #Calculate the scale height
        scale = np.zeros(NPRO)
        scale[:] = K_B*T[:]/(XMOLWT[:]*gravity[:])

        if ialt > 0 and ialt < NPRO-1 :
            h[ialt] = 0.0

        # nupper = NPRO - ialt - 1
        for i in range(ialt+1, NPRO):
            sh = 0.5 * (scale[i-1] + scale[i])
            #self.H[i] = self.H[i-1] - sh * np.log(self.P[i]/self.P[i-1])
            h[i] = h[i-1] - sh * np.log(p[i]/p[i-1])

        for i in range(ialt-1,-1,-1):
            sh = 0.5 * (scale[i+1] + scale[i])
            #self.H[i] = self.H[i+1] - sh * np.log(self.P[i]/self.P[i+1])
            h[i] = h[i+1] - sh * np.log(p[i]/p[i+1])

        atdepth1 = h[-1] - h[0]

        XDEPTH = 100.*abs((atdepth1-ATDEPTH)/ATDEPTH)
        # print('xdepth',XDEPTH)
        H = h[:]

    return H


# def XHYDROSTATH(AMFORM,IPLANET,LATITUDE,NPRO,NVMR,MOLWT,
#     IDGAS,ISOGAS,H,P,T,VMR,SCALE,
#     radius,mass):
#     XDEPTH = 2
#     while XDEPTH > 1:
#         # NPRO and NVMR no need to be passed
#         ATDEPTH = H[-1] - H[0] # NPRO = len(H)

#         # First find level closest ot zero altitude
#         DELH = 1000.0
#         for I in range(NPRO):
#             X = abs(H[I])
#             if X < DELH:
#                 DELH = X
#                 JZERO = I

#             """# alternatively...
#             min_index = np.argmin(abs(H))
#             if H[min_index] < DELH:
#                 DELH = X
#             """
#             # VMR = NPRO x NVMR
#             XVMR = np.zeros(NVMR)
#             for J in range(NVMR):
#                 XVMR[J] = VMR[I,J]

#             """
#             need to flesh out
#             """
#             g = NEWGRAV(H[I],radius,mass,lat_in=None)

#             XMOLWT = 2.3 * AMU

#             scale_height = K_B * T[I]/(XMOLWT*g)

#         if JZERO > 1 and JZERO < NPRO:
#             H[JZERO] = 0.0

#         for K in np.range(JZERO+1,NPRO):
#             """SH average scale height"""
#             SH = 0.5*(scale_height[K-1]+scale_height[I+1])
#             """hydrostatic equilibrium"""
#             H[K] = H[K+1] - SH*np.log(P[K]/P[K+1])

#         ATDEPTH1 = H[-1] - H[0]
#         XDEPTH = 100 * abs((ATDEPTH1-ATDEPTH)/ATDEPTH)


# def simple_grav(h, radius, mass):
#     """TBD
#       Sets the gravitational acceleration based on selected planet and
#       latitude. Gravity is calculated normal to the surface, and corrected
#       for rotational effects. The input latitude is assumed planetographic.

#     Parameters
#     ----------

#     Returns
#     -------
#     """

#     g = G*mass/(radius+h)**2
#     gravity = g
#     return gravity

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
