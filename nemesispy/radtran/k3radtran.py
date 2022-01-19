#!/usr/local/bin/python3
# -*- coding: utf-8 -*-
import numpy as np
from copy import copy
# from numba import jit
from nemesispy.radtran.k2interp import new_k_overlap
# from nemesispy.radtran.k2interp import mix_multi_gas_k as new_k_overlap
# from nemesispy.radtran.k2interp import cal_k as interp_k
from nemesispy.radtran.k2interp import interp_k
from nemesispy.radtran.k5cia import calc_tau_cia
"""
Check progress:
Layer Temperature routine is correct
Absorber Amount routine is correct

Need to do:
Check Radtran routine
Check Planck routine
Check ktables routine
Check CIA routine

is ktables unit atm? i think it is.

how is the filter function used?

"""
def calc_tau_rayleighj(wave_grid,TOTAM,ISPACE=1):
    """
    Function to calculate the Rayleigh scattering opacity in each atmospheric layer,
    for Gas Giant atmospheres using data from Allen (1976) Astrophysical Quantities

    Parameters
    ----------
        ISPACE : int
            Flag indicating the spectral units
            (0) Wavenumber in cm-1 or (1) Wavelegnth (um)
        wave_grid(NWAVE) : ndarray
            Wavenumber (cm-1) or wavelength array (um)

    Outputs
    -------
    TAURAY(NWAVE,NLAY) : ndarray
        Rayleigh scattering opacity in each layer
    """
    AH2 = 13.58E-5
    BH2 = 7.52E-3
    AHe = 3.48E-5
    BHe = 2.30E-3
    fH2 = 0.864
    k = 1.37971e-23 # JK-1
    P0 = 1.01325e5 # Pa
    T0 = 273.15 # K

    NLAY = len(TOTAM)
    NWAVE = len(wave_grid)

    if ISPACE == 0:
        LAMBDA = 1./wave_grid * 1.0e-2 # converted wavelength unit to m
        x = 1.0/(LAMBDA*1.0e6)
    else:
        LAMBDA = wave_grid * 1.0e-6 # wavelength in m
        x = 1.0/(LAMBDA*1.0e6)

    # calculate refractive index
    nH2 = AH2 * (1.0+BH2*x*x)
    nHe = AHe * (1.0+BHe*x*x)

    #calculate Jupiter air's refractive index at STP (Actually n-1)
    nAir = fH2 * nH2 + (1-fH2)*nHe

    #H2,He Seem pretty isotropic to me?...Hence delta = 0.
    #Penndorf (1957) quotes delta=0.0221 for H2 and 0.025 for He.
    #(From Amundsen's thesis. Amundsen assumes delta=0.02 for H2-He atmospheres
    delta = 0.0
    temp = 32*(np.pi**3.)*nAir**2.
    N0 = P0/(k*T0)

    x = N0*LAMBDA*LAMBDA
    faniso = (6.0+3.0*delta)/(6.0 - 7.0*delta)

    #Calculating the scattering cross sections in m2
    k_rayleighj = temp*faniso/(3.*(x**2)) #(NWAVE)

    #Calculating the Rayleigh opacities in each layer
    tau_ray = np.zeros([len(wave_grid),NLAY])

    for ilay in range(NLAY):
        tau_ray[:,ilay] = k_rayleighj[:] * TOTAM[ilay] #(NWAVE,NLAY)

    return tau_ray

def planck(wave,temp,ispace=1):
    """
    Calculate the blackbody radiation given by the Planck function

    Parameters
    ----------
    wave(nwave) : ndarray
        Wavelength or wavenumber array
    temp : real
        Temperature of the blackbody (K)
    ispace : int
        Flag indicating the spectral units
        (0) Wavenumber (cm-1)
        (1) Wavelength (um)

    Returns
    -------
	bb(nwave) : ndarray
        Planck function (W cm-2 sr-1 (cm-1)-1 or W cm-2 sr-1 um-1)
    """

    c1 = 1.1911e-12
    c2 = 1.439
    if ispace==0:
        y = wave
        a = c1 * (y**3.)
    elif ispace==1:
        y = 1.0e4/wave
        a = c1 * (y**5.) / 1.0e4
    else:
        raise Exception('error in planck: ISPACE must be either 0 or 1')

    tmp = c2 * y / temp
    b = np.exp(tmp) - 1
    bb = a/b

    return bb


def tau_gas(k_gas_w_g_p_t, P_layer, T_layer, VMR_layer, U_layer,
            P_grid, T_grid, del_g):
    """
    Calculate the optical path due to gaseous absorbers.

    Parameters
    ----------
    k_gas_w_g_p_t(Ngas,Nwave,Ng,Npress,Ntemp) : ndarray
        Raw k-coefficients.
        Has dimension: Nwave x Ng x Npress x Ntemp.
    P_layer(Nlayer) : ndarray
        Atmospheric pressure grid.
    T_layer(Nlayer) : ndarray
        Atmospheric temperature grid.
    VMR_layer(Nlayer,Ngas) : ndarray
        Array of volume mixing ratios for Ngas.
        Has dimensioin: Nlayer x Ngas
    U_layer : ndarray
        DESCRIPTION.
    P_grid(Npress) : ndarray
        Pressure grid on which the k-coeff's are pre-computed.
    T_grid(Ntemp) : ndarray
        Temperature grid on which the k-coeffs are pre-computed.
    del_g : ndarray
        DESCRIPTION.

    Returns
    -------
    tau_w_g_l(Nwave,Ng,Nlayer) : ndarray
        DESCRIPTION.
    """

    U_layer *= 1.0e-20 # absorber amounts (U_layer) is scaled by a factor 1e-20
    U_layer *= 1.0e-4 # convert from absorbers per m^2 to per cm^2

    k_gas_w_g_l = interp_k(P_grid, T_grid, P_layer, T_layer, k_gas_w_g_p_t) # NGAS,NWAVE,NG,NLAYER
    # Ngas, Nwave, Ng, Nlayer = k_gas_w_g_l.shape
    # print('k_gas_w_g_l', k_gas_w_g_l)

    k_w_g_l,f_combined = new_k_overlap(k_gas_w_g_l, del_g, VMR_layer.T) # NWAVE,NG,NLAYER

    utotl = U_layer

    TAUGAS = k_w_g_l * utotl * f_combined  # NWAVE, NG, NLAYER

    return TAUGAS

def tau_gas_alternative(k_gas_w_g_p_t, P_layer, T_layer, VMR_layer, U_layer,
            P_grid, T_grid, del_g):
    """
      Parameters
      ----------
      k_gas_w_g_p_t : ndarray
          DESCRIPTION.
      P_layer : ndarray
          DESCRIPTION.
      T_layer : ndarray
          DESCRIPTION.
      VMR_layer : ndarray
          DESCRIPTION.
      U_layer : ndarray
          DESCRIPTION.
      P_grid : ndarray
          DESCRIPTION.
      T_grid : ndarray
          DESCRIPTION.
      g_ord : ndarray
          DESCRIPTION.
      del_g : ndarray
          DESCRIPTION.

      Returns
      -------
      tau_w_g_l : ndarray
          DESCRIPTION.
    """
    g_ord = np.array([0.0034357 , 0.01801404, 0.04388279, 0.08044151, 0.126834  ,
    0.1819732 , 0.2445665 , 0.3131469 , 0.3861071 , 0.4617367 ,
    0.5382633 , 0.6138929 , 0.6868531 , 0.7554335 , 0.8180268 ,
    0.873166  , 0.9195585 , 0.9561172 , 0.981986  , 0.9965643 ])
    k_gas_w_g_l = interp_k(P_grid, T_grid, P_layer, T_layer, k_gas_w_g_p_t)
    Ngas, Nwave, Ng, Nlayer = k_gas_w_g_l.shape
    tau_w_g_l = np.zeros((Nwave,Ng,Nlayer))
    for iwave in range (Nwave):
        k_gas_g_l = k_gas_w_g_l[:,iwave,:,:]
        k_g_l = np.zeros((Ng,Nlayer))
        for ilayer in range(Nlayer):
            k_g_l[:,ilayer], VMR\
                = new_k_overlap(k_gas_g_l[:,:,ilayer],VMR_layer[ilayer,:],del_g)
            tau_w_g_l[iwave,:,ilayer] = k_g_l[:,ilayer]*U_layer[ilayer]*VMR
    return tau_w_g_l

def radtran(wave_grid, U_layer, P_layer, T_layer, VMR_layer, k_gas_w_g_p_t,
            P_grid, T_grid, del_g, ScalingFactor, RADIUS, solspec,
            k_cia, ID, NU_GRID, CIA_TEMPS, DEL_S):
    """
    Calculate emission spectrum using the correlated-k method.

    # Absorber amounts (U_layer) is scaled by a factor 1e-20 because Nemesis
    # k-tables are scaled by a factor of 1e20. Done in tau_gas.

    # Need to be smart about benchmarking against NEMESIS

    Parameters
    ----------
    wave_grid(Nwave) : ndarray
        Wavelengths (um) grid for calculating spectra.
    U_layer(Nlayer) : ndarray
        Total number of gas particles in each layer.
        We want SI unit (no. of particle/m^2) here.
    P_layer(Nlayer) : ndarray
        Atmospheric pressure grid.
        We want SI unit (Pa) here.
    T_layer(Nlayer) : ndarray
        Atmospheric temperature grid. In Kelvin.
    VMR_layer(Nlayer,Ngas) : ndarray
        Array of volume mixing ratios for Ngas.
        Has dimensioin: Nlayer x Ngas
    k_gas_w_g_p_t : ndarray
        k-coefficients. Has dimension: Nwave x Ng x Npress x Ntemp.
    P_grid(Npress) : ndarray
        Pressure grid on which the k-coeff's are pre-computed.
        We want SI unit (Pa) here.
    T_grid(Ntemp) : ndarray
        Temperature grid on which the k-coeffs are pre-computed. In Kelvin
    del_g : ndarray
        Quadrature weights of the g-ordinates.
    ScalingFactor : ndarray ( NLAY ) ##
        Scale stuff to line of sight
    RADIUS : real ##
        Planetary radius
        We want SI unit (m) here.
    solspec : ndarray ##
        Stellar spectra, used when the unit of the output is in fraction
        of stellar irradiance.

        Stellar flux at planet's distance (W cm-2 um-1 or W cm-2 (cm-1)-1)

    Returns
    -------
    radiance : ndarray
        Output radiance (W cm-2 um-1 sr-1)
    """
    # Reverse layers from TOP of atmoaphsere first to BOTTOM of atmosphere last
    print('T_layer first',T_layer)
    print('P_layer first',P_layer)
    print('U_layer first',U_layer)
    print('ScalingFactor first',ScalingFactor)
    print('VMR_layer first', VMR_layer)
    P_layer = P_layer[::-1]
    T_layer = T_layer[::-1]
    U_layer = U_layer[::-1]
    ScalingFactor = ScalingFactor[::-1]
    VMR_layer = VMR_layer[::-1,:]
    print('VMR_layer second', VMR_layer)
    # print('T_layer second',T_layer)


    # Get constants from the dimension of k table
    NGAS, NWAVE, NG, NPRESS, NTEMP = k_gas_w_g_p_t.shape
    NLAY = len(P_layer)

    # print('NGAS, NWAVE, NG, NGRID',NGAS, NWAVE, NG, NGRID)
    ### Second order opacities to be continued

    # Total Optical path
    TAUTOT = np.zeros([NWAVE,NG,NLAY])
    # Collision Induced Absorptioin Optical Path
    TAUCIA = np.zeros([NWAVE,NLAY])
    # Rayleigh Scattering Optical Path
    TAURAY = np.zeros([NWAVE,NLAY])
    # Dust Scattering Optical Path
    TAUDUST = np.zeros([NWAVE,NLAY])

    """
    TO BE DONE!
    """
    TAURAY = calc_tau_rayleighj(wave_grid=wave_grid,TOTAM=U_layer)
    # print('TAURAY',TAURAY)
    # TAURAY *= 0

    TAUCIA = calc_tau_cia(WAVE_GRID=wave_grid,K_CIA=k_cia,ISPACE=1,
        ID=ID,TOTAM=U_layer,T_layer=T_layer,P_layer=P_layer,VMR_layer=VMR_layer,DELH=DEL_S,
        NU_GRID=NU_GRID,TEMPS=CIA_TEMPS,INORMAL=0,NPAIR=9)
    # TAUCIA *= 0


    ### Gaseous Opacity
    # Calculating the k-coefficients for each gas in each layer
    TAUGAS = tau_gas(k_gas_w_g_p_t, P_layer, T_layer, VMR_layer, U_layer,
            P_grid, T_grid, del_g) # NWAVE x NG x NLAYER

    # print('TAUGAS', TAUGAS)


    # Merge all different opacities

    for ig in range(NG): # wavebin x layer / NWAVE x NG x NLAYER
        TAUTOT[:,ig,:] = TAUGAS[:,ig,:] + TAUCIA[:,:] + TAUDUST[:,:] + TAURAY[:,:]

    #Scale to the line-of-sight opacities
    # TAUTOT_LAYINC = TAUTOT * ScalingFactor
    TAUTOT = TAUTOT * ScalingFactor


    # Thermal Emission Calculation
    # IMOD = 3
    NPATH = 1
    SPECOUT = np.zeros([NWAVE,NG,NPATH])

    #Defining the units of the output spectrum / divide by stellar spectrum
    # IFORM = 1
    radextra = sum(DEL_S)
    xfac = np.pi*4.*np.pi*((RADIUS+radextra)*1e2)**2.
    xfac = xfac / solspec

    #Calculating spectrum
    for ipath in range(NPATH):

        #Calculating atmospheric contribution
        taud = np.zeros((NWAVE,NG))
        trold = np.zeros((NWAVE,NG))
        specg = np.zeros((NWAVE,NG))

        # Thermal Emission from planet
        # SPECOUT = np.zeros(NWAVE, NG)

        """
        ERROR 1:
            old : taud[:,:] =  TAUTOT[:,:,ilayer]
            new : taud[:,:] =  TAUTOT[:,:,ilayer] + taud[:,:]
        ERROR 2:
            old : TAUTOT[:,:,-ilayer]
            bb = planck(wave_grid, T_layer[-ilayer])
        """
        for ilayer in range(NLAY):

            taud[:,:] =  TAUTOT[:,:,ilayer] + taud[:,:]
            tr = np.exp(-taud) # transmission function
            #print('tr',tr)
            bb = planck(wave_grid, T_layer[ilayer]) # blackbody function

            for ig in range(NG):
                specg[:,ig] = specg[:,ig] + (trold[:,ig]-tr[:,ig])*bb[:] * xfac
                # print('specg[:,ig]',specg[:,ig])
            trold = copy(tr)
            #trold = tr

        # surface/bottom layer contribution
        p1 = P_layer[int(len(P_layer)/2-1)] #midpoint
        p2 = P_layer[-1] #lowest point in altitude/highest in pressure

        surface = None
        if p2 > p1: # i.e. if not a limb path
            #print(p2,p1)
            if not surface:
                radground = planck(wave_grid,T_layer[-1])
                #print('radground',radground)
            for ig in range(NG):
                specg[:,ig] = specg[:,ig] + trold[:,ig]*radground*xfac
                # print('trold[:,ig]*radground*xfac',trold[:,ig]*radground*xfac)

        SPECOUT[:,:,ipath] = specg[:,:]
        # Option 1

    SPECOUT = np.tensordot(SPECOUT, del_g, axes=([1],[0]))

    # Option 2
    # SPECOUT = np.zeros(NWAVE)
    # for iwave in range(NWAVE):
    #     SPECOUT[iwave] = np.sum(specg[iwave,:] * del_g)

    # Option 3
    # specg1 = np.zeros([NWAVE,NG,1])
    # specg1[:,:,0] = specg
    # SPECOUT = np.tensordot(specg1, del_g, axes=([1],[0]))

    # print('TAUCIA',TAUCIA) # NWAVE X NLAYER
    # print('TAUGAS',TAUGAS) # NWAVE x NG x NLAYER
    # print('TAUTOT',TAUTOT) # NWAVE x NG x NLAYER
    return SPECOUT

"""# pure H2 He case
 original name of this file: wasp43b.drv
      1.143   0.209844        17   0.000     :Vmin dV Npts FWHM-CORRK
    0.000   0.000       : Additional codes PAR1 and PAR2
  WASP43B.KLS
   24   0     1   0             : spectral model code, FLAGH2P, NCONT, FLAGC
 wasp43b.xsc                    : Dust x-section file
   20   1   6           : number of layers, paths and gases
    1                                    : identifier for gas 1
      0   0                              : isotope ID and process parameter
    2                                    : identifier for gas 2
      0   0                              : isotope ID and process parameter
    5                                    : identifier for gas 3
      0   0                              : isotope ID and process parameter
    6                                    : identifier for gas 4
      0   0                              : isotope ID and process parameter
   40                                    : identifier for gas 5
      0   0                              : isotope ID and process parameter
   39                                    : identifier for gas 6
      0   0                              : isotope ID and process parameter
format of layer data
temperature = np.array([2286.021,2253.653,2185.696,2082.271,1955.738,1822.316,
1695.965,1586.105,1497.388,1430.283,1382.350,1349.684,1328.206,1314.446,1305.782,
1300.392,1297.059,1295.010,1293.753,1292.982])
totam = np.array([0.44341E+27,0.26865E+27,0.16285E+27,0.98854E+26,0.60092E+26,0.36569E+26,
0.22268E+26,0.13563E+26,0.82610E+25,0.50316E+25,0.30653E+25,0.18685E+25,0.11397E+25,
0.69572E+24,0.42500E+24,0.25985E+24,0.15898E+24,0.97341E+23,0.59644E+23,0.36567E+23])
 layer baseH  delH   baseP      baseT   totam       pressure    temp   doppler
        absorber amounts and partial pressures
        continuum points if any
  1    0.00   86.90 0.19739E+02 2294.230 0.44341E+27 0.16191E+02 2286.021  0.0000
         0.00000E+00 0.00000E+00 0.00000E+00 0.00000E+00 0.00000E+00 0.00000E+00
 0.00000E+00 0.00000E+00 0.66512E+26 0.24287E+01 0.37690E+27 0.13763E+02
         0.00000E+00
  2   86.90   85.69 0.12030E+02 2276.392 0.26865E+27 0.97956E+01 2253.653  0.0000
         0.00000E+00 0.00000E+00 0.00000E+00 0.00000E+00 0.00000E+00 0.00000E+00
 0.00000E+00 0.00000E+00 0.40297E+26 0.14693E+01 0.22835E+27 0.83263E+01
         0.00000E+00
  3  172.59   83.08 0.73320E+01 2225.622 0.16285E+27 0.59331E+01 2185.696  0.0000
         0.00000E+00 0.00000E+00 0.00000E+00 0.00000E+00 0.00000E+00 0.00000E+00
 0.00000E+00 0.00000E+00 0.24427E+26 0.88996E+00 0.13842E+27 0.50431E+01
         0.00000E+00
  4  255.66   79.16 0.44686E+01 2135.848 0.98854E+26 0.35983E+01 2082.271  0.0000
         0.00000E+00 0.00000E+00 0.00000E+00 0.00000E+00 0.00000E+00 0.00000E+00
 0.00000E+00 0.00000E+00 0.14828E+26 0.53975E+00 0.84026E+26 0.30586E+01
         0.00000E+00
  5  334.82   74.40 0.27234E+01 2016.333 0.60092E+26 0.21851E+01 1955.738  0.0000
         0.00000E+00 0.00000E+00 0.00000E+00 0.00000E+00 0.00000E+00 0.00000E+00
 0.00000E+00 0.00000E+00 0.90138E+25 0.32776E+00 0.51078E+26 0.18573E+01
         0.00000E+00
  6  409.22   69.41 0.16598E+01 1883.171 0.36569E+26 0.13281E+01 1822.316  0.0000
         0.00000E+00 0.00000E+00 0.00000E+00 0.00000E+00 0.00000E+00 0.00000E+00
 0.00000E+00 0.00000E+00 0.54854E+25 0.19921E+00 0.31084E+26 0.11288E+01
         0.00000E+00
  7  478.64   64.70 0.10116E+01 1751.800 0.22268E+26 0.80773E+00 1695.965  0.0000
         0.00000E+00 0.00000E+00 0.00000E+00 0.00000E+00 0.00000E+00 0.00000E+00
 0.00000E+00 0.00000E+00 0.33402E+25 0.12116E+00 0.18928E+26 0.68657E+00
         0.00000E+00
  8  543.34   60.60 0.61654E+00 1633.600 0.13563E+26 0.49153E+00 1586.105  0.0000
         0.00000E+00 0.00000E+00 0.00000E+00 0.00000E+00 0.00000E+00 0.00000E+00
 0.00000E+00 0.00000E+00 0.20344E+25 0.73729E-01 0.11528E+26 0.41780E+00
         0.00000E+00
  9  603.94   57.29 0.37576E+00 1535.139 0.82610E+25 0.29926E+00 1497.388  0.0000
         0.00000E+00 0.00000E+00 0.00000E+00 0.00000E+00 0.00000E+00 0.00000E+00
 0.00000E+00 0.00000E+00 0.12391E+25 0.44889E-01 0.70218E+25 0.25437E+00
         0.00000E+00
 10  661.23   54.76 0.22901E+00 1458.478 0.50316E+25 0.18230E+00 1430.283  0.0000
         0.00000E+00 0.00000E+00 0.00000E+00 0.00000E+00 0.00000E+00 0.00000E+00
 0.00000E+00 0.00000E+00 0.75475E+24 0.27345E-01 0.42769E+25 0.15495E+00
         0.00000E+00
 11  715.99   52.95 0.13957E+00 1402.255 0.30653E+25 0.11112E+00 1382.350  0.0000
         0.00000E+00 0.00000E+00 0.00000E+00 0.00000E+00 0.00000E+00 0.00000E+00
 0.00000E+00 0.00000E+00 0.45980E+24 0.16668E-01 0.26055E+25 0.94454E-01
         0.00000E+00
 12  768.93   51.70 0.85065E-01 1363.081 0.18685E+25 0.67788E-01 1349.684  0.0000
         0.00000E+00 0.00000E+00 0.00000E+00 0.00000E+00 0.00000E+00 0.00000E+00
 0.00000E+00 0.00000E+00 0.28027E+24 0.10168E-01 0.15882E+25 0.57620E-01
         0.00000E+00
 13  820.63   50.87 0.51844E-01 1336.891 0.11397E+25 0.41383E-01 1328.206  0.0000
         0.00000E+00 0.00000E+00 0.00000E+00 0.00000E+00 0.00000E+00 0.00000E+00
 0.00000E+00 0.00000E+00 0.17095E+24 0.62074E-02 0.96872E+24 0.35175E-01
         0.00000E+00
 14  871.50   50.33 0.31597E-01 1319.922 0.69572E+24 0.25281E-01 1314.446  0.0000
         0.00000E+00 0.00000E+00 0.00000E+00 0.00000E+00 0.00000E+00 0.00000E+00
 0.00000E+00 0.00000E+00 0.10436E+24 0.37921E-02 0.59136E+24 0.21489E-01
         0.00000E+00
 15  921.83   49.98 0.19257E-01 1309.168 0.42500E+24 0.15454E-01 1305.782  0.0000
         0.00000E+00 0.00000E+00 0.00000E+00 0.00000E+00 0.00000E+00 0.00000E+00
 0.00000E+00 0.00000E+00 0.63750E+23 0.23180E-02 0.36125E+24 0.13136E-01
         0.00000E+00
 16  971.81   49.77 0.11737E-01 1302.457 0.25985E+24 0.94517E-02 1300.392  0.0000
         0.00000E+00 0.00000E+00 0.00000E+00 0.00000E+00 0.00000E+00 0.00000E+00
 0.00000E+00 0.00000E+00 0.38978E+23 0.14178E-02 0.22088E+24 0.80340E-02
         0.00000E+00
 17 1021.58   49.62 0.71530E-02 1298.308 0.15898E+24 0.57837E-02 1297.059  0.0000
         0.00000E+00 0.00000E+00 0.00000E+00 0.00000E+00 0.00000E+00 0.00000E+00
 0.00000E+00 0.00000E+00 0.23847E+23 0.86755E-03 0.13513E+24 0.49161E-02
         0.00000E+00
 18 1071.20   49.53 0.43595E-02 1295.760 0.97341E+23 0.35408E-02 1295.010  0.0000
         0.00000E+00 0.00000E+00 0.00000E+00 0.00000E+00 0.00000E+00 0.00000E+00
 0.00000E+00 0.00000E+00 0.14601E+23 0.53112E-03 0.82740E+23 0.30097E-02
         0.00000E+00
 19 1120.73   49.46 0.26570E-02 1294.201 0.59644E+23 0.21686E-02 1293.753  0.0000
         0.00000E+00 0.00000E+00 0.00000E+00 0.00000E+00 0.00000E+00 0.00000E+00
 0.00000E+00 0.00000E+00 0.89466E+22 0.32529E-03 0.50697E+23 0.18433E-02
         0.00000E+00
 20 1170.19   49.41 0.16193E-02 1293.249 0.36567E+23 0.13287E-02 1292.982  0.0000
         0.00000E+00 0.00000E+00 0.00000E+00 0.00000E+00 0.00000E+00 0.00000E+00
 0.00000E+00 0.00000E+00 0.54850E+22 0.19930E-03 0.31082E+23 0.11294E-02
         0.00000E+00
   20   3 0.10000E-01            : Nlayers, model & error limit, path  1
   1   20 1292.982  0.99998E+00  :     layer or path, emission temp, scale
   2   19 1293.753  0.10000E+01  :     layer or path, emission temp, scale
   3   18 1295.010  0.99993E+00  :     layer or path, emission temp, scale
   4   17 1297.059  0.10001E+01  :     layer or path, emission temp, scale
   5   16 1300.392  0.99999E+00  :     layer or path, emission temp, scale
   6   15 1305.782  0.10000E+01  :     layer or path, emission temp, scale
   7   14 1314.446  0.99993E+00  :     layer or path, emission temp, scale
   8   13 1328.206  0.99999E+00  :     layer or path, emission temp, scale
   9   12 1349.684  0.99996E+00  :     layer or path, emission temp, scale
  10   11 1382.350  0.10000E+01  :     layer or path, emission temp, scale
  11   10 1430.283  0.10001E+01  :     layer or path, emission temp, scale
  12    9 1497.388  0.10001E+01  :     layer or path, emission temp, scale
  13    8 1586.105  0.99997E+00  :     layer or path, emission temp, scale
  14    7 1695.965  0.10000E+01  :     layer or path, emission temp, scale
  15    6 1822.316  0.10000E+01  :     layer or path, emission temp, scale
  16    5 1955.738  0.99997E+00  :     layer or path, emission temp, scale
  17    4 2082.271  0.99998E+00  :     layer or path, emission temp, scale
  18    3 2185.696  0.10000E+01  :     layer or path, emission temp, scale
  19    2 2253.653  0.10000E+01  :     layer or path, emission temp, scale
  20    1 2286.021  0.99999E+00  :     layer or path, emission temp, scale
    1                                    : number of filter profile points
  0.00000E+00     0.000                  : filter profile point   1
 wasp43b.out
   1                                     :number of calculations
    2   2   2   0                        :type and # of parameters for calc  1
           1
           1
   0.00000000
   0.00000000
"""