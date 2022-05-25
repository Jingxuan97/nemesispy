import numpy as np
from scipy.interpolate import interp1d
# def VERINT(X,Y,N,XIN):
#     if X[0]>X[-1]:
#         X = X[::-1]
#         Y = Y[::-1]
#     YOUT = np.interp(x=XIN,xp=X,fp=Y)
#     return YOUT

def VERINT(X,Y,N,XIN):
    f = interp1d(X,Y)
    YOUT = f(XIN)
    return YOUT


def interpvivien_point(XLON, XLAT, XP, VP, VT, VVMR,
    global_model_longitudes, global_model_lattitudes):
    """


    Parameters
    ----------
    XLON : TYPE
        Longitude of spectrum to be simulated
    XLAT : TYPE
        Latitude of spectrum to be simulated
    XP : TYPE
        Pressre grid to interpolate model into. NPRO
    VP : TYPE
        Pressure grid of the input model. NPRESS
    VT : TYPE
        Temperture model, NLON x NLAT x NP.
    VVMR : TYPE
        VMR model, NLON x NLAT x NP x NVMR
    global_model_longitudes : TYPE
        Longitudes of the input model.
    global_model_lattitudes : TYPE
        Lattitudes of the input mmodel.

    Returns
    -------
    T : TYPE
        Temperature model interpolated to (XLON,XLAT). NPRO
    VMR : TYPE
        VMR model interpolated to (XLON,XLAT). NPRO

    """
    globalT = VT
    assert XLAT<=90 and XLAT>=-90
    NPRO = len(XP)


    # dimension parameters from Vivien's GCM
    NPRESS = 53
    NLON = 64
    NLAT = 32
    NGV = 6

    VLON = global_model_longitudes
    VLAT = global_model_lattitudes

    # Convert P from bar to atm and convert to log
    logVP = np.log(VP)
    # print('logVP',logVP)

    # Vivien has 0 longitude as sub-stellar point. He we have 180
    # Hence need to add 180 to all longitudes
    VLON = VLON + 180
    # print('VLON',VLON)

    #  Find closest point in stored array
    JLAT = -1
    for I in range(NLAT-1):
        if XLAT >= VLAT[I] and XLAT <= VLAT[I+1]:
            JLAT = I
            FLAT = (XLAT-VLAT[I])/(VLAT[I+1]-VLAT[I])

    if JLAT < 0:
        if XLAT < VLAT[0]:
            JLAT = 0
            FLAT = 0
        if XLAT >= VLAT[-1]:
            JLAT = NLAT - 1
            FLAT = 1

    JLON1 = -1
    JLON2 = -1
    for I in range(NLON-1):
        if XLON >= VLON[I] and XLON <= VLON[I+1]:
            JLON1 = I
            JLON2 = I+1
            FLON = (XLON-VLON[I])/(VLON[I+1]-VLON[I])
    if JLON1 < 0:
        if XLON < VLON[0]:
            # XLON must be in range 0. to VLON[0]
            JLON1 = NLON - 1
            JLON2 = 0
            FLON = (XLON+360-VLON[-1])/(VLON[0]+360-VLON[-1])
        if XLON >= VLON[-1]:
            # XLON must be in range VLON[-1] to 360
            JLON1 = NLON - 1
            JLON2 = 0
            FLON = (XLON - VLON[-1])/(VLON[0]+360-VLON[-1])
    # print('JLAT',JLAT)
    # print('JLON',JLON1,JLON2)
    # Temperature interpolation array
    VY1 = globalT[JLON1,JLAT,:]
    VY2 = globalT[JLON2,JLAT,:]
    VY3 = globalT[JLON2,JLAT+1,:]
    VY4 = globalT[JLON1,JLAT+1,:]
    # print('VY1',VY1)
    # print('VY2',VY2)
    # print('VY3',VY3)
    # print('VY4',VY4)

    # Define VERTINT
    # SUBROUTINE VERINT(X,Y,N,YOUT,XIN)
    # interpolate T
    # print('XP',XP)
    interped_T = np.zeros(NPRO)
    for IPRO in range(NPRO):
        LP1 = np.log(XP[IPRO])
        # print('LP1',LP1)
        # LP1 = XP[IPRO]
        Y1 = VERINT(logVP,VY1,NPRESS,LP1)
        Y2 = VERINT(logVP,VY2,NPRESS,LP1)
        Y3 = VERINT(logVP,VY3,NPRESS,LP1)
        Y4 = VERINT(logVP,VY4,NPRESS,LP1)
        interped_T[IPRO] = (1.0-FLON)*(1.0-FLAT)*Y1 + FLON*(1.0-FLAT)*Y2 \
            + FLON*FLAT*Y3 + (1.0-FLON)*FLAT*Y4
        # print('LP1',LP1)
        # print('Y1',Y1)
        # print('Y2',Y2)
        # print('Y3',Y3)
        # print('Y4',Y4)
        # print('interped_T[IPRO]',interped_T[IPRO])



    NVMR = NGV
    interped_VMR = np.zeros((NPRO,NVMR))
    for IVMR in range(NVMR):
        VY1 = VVMR[JLON1,JLAT,:,IVMR]
        VY2 = VVMR[JLON2,JLAT,:,IVMR]
        VY3 = VVMR[JLON2,JLAT+1,:,IVMR]
        VY4 = VVMR[JLON1,JLAT+1,:,IVMR]
        for I in range(NPRO):
            LP1 = np.log(XP[I])
            Y1 = VERINT(logVP,VY1,NPRESS,LP1)
            Y2 = VERINT(logVP,VY2,NPRESS,LP1)
            Y3 = VERINT(logVP,VY3,NPRESS,LP1)
            Y4 = VERINT(logVP,VY4,NPRESS,LP1)
            interped_VMR[I,IVMR] = (1.0-FLON)*(1.0-FLAT)*Y1 + FLON*(1.0-FLAT)*Y2\
                + FLON*FLAT*Y3 + (1.0-FLON)*FLAT*Y4

    return interped_T, interped_VMR
