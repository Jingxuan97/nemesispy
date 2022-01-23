#!/usr/local/bin/python3
# -*- coding: utf-8 -*-
import numpy as np
from copy import copy
from nemesispy.radtran.k2interp import new_k_overlap
from nemesispy.radtran.k2interp import interp_k
from nemesispy.radtran.k5cia import calc_tau_cia
"""
Check progress:
Layer Temperature routine is correct
Absorber Amount routine is correct

Need to do:
Check Radtran routine
Check Planck routine/ is good
Check ktables routine/ is good
Check CIA routine

is ktables unit atm? i think it is.

how is the filter function used?

"""
def tau_rayleighj(wave_grid,TOTAM,ISPACE=1):
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
    k_gas_w_g_p_t(NGAS,NWAVE,NG,NPRESSKTA,NTEMPKTA) : ndarray
        Raw k-coefficients.
        Has dimension: NWAVE x NG x NPRESSKTA x NTEMPKTA.
    P_layer(NLAYER) : ndarray
        Atmospheric pressure grid.
        Unit: Pa
    T_layer(NLAYER) : ndarray
        Atmospheric temperature grid.
        Unit: Kelvin
    VMR_layer(NLAYER,NGAS) : ndarray
        Array of volume mixing ratios for NGAS.
        Has dimensioin: NLAYER x NGAS
    U_layer : ndarray
        DESCRIPTION.
    P_grid(NPRESSKTA) : ndarray
        Pressure grid on which the k-coefficients are pre-computed.
        Unit: Pa
    T_grid(NTEMPKTA) : ndarray
        Temperature grid on which the k-coefficients are pre-computed.
        Unit: Kelvin
    del_g(NG) : ndarray
        Gauss quadrature weights for the g-ordinates.
        These are the widths of the bins in g-space.

    Returns
    -------
    tau_w_g_l(NWAVE,NG,NLAYER) : ndarray
        DESCRIPTION.
    """

    U_layer *= 1.0e-20 # absorber amounts (U_layer) is scaled by a factor 1e-20
    U_layer *= 1.0e-4 # convert from absorbers per m^2 to per cm^2

    k_gas_w_g_l = interp_k(P_grid, T_grid, P_layer, T_layer, k_gas_w_g_p_t) # NGAS,NWAVE,NG,NLAYER
    k_w_g_l,f_combined = new_k_overlap(k_gas_w_g_l, del_g, VMR_layer.T) # NWAVE,NG,NLAYER

    TAUGAS = k_w_g_l * U_layer * f_combined  # NWAVE, NG, NLAYER

    return TAUGAS

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
    wave_grid(NWAVE) : ndarray
        Wavelengths (um) grid for calculating spectra.
    U_layer(NLAYER) : ndarray
        Total number of gas particles in each layer.
        Unit: no. of particle/m^2
    P_layer(NLAYER) : ndarray
        Atmospheric pressure grid.
        Unit: Pa
    T_layer(NLAYER) : ndarray
        Atmospheric temperature grid.
        Unit: Kelvin.
    VMR_layer(NLAYER,NGAS) : ndarray
        Array of volume mixing ratios for NGAS.
        Has dimensioin: NLAYER x NGAS
    k_gas_w_g_p_t : ndarray
        k-coefficients. Has dimension: NWAVE x NG x NPRESSKTA x NTEMPKTA.
    P_grid(NPRESSKTA) : ndarray
        Pressure grid on which the k-coeff's are pre-computed.
        We want SI unit (Pa) here.
    T_grid(NTEMPKTA) : ndarray
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
    # Reverse layers from bottom of atmosphere first to top of atmoaphsere first
    P_layer = P_layer[::-1]
    T_layer = T_layer[::-1]
    U_layer = U_layer[::-1]
    ScalingFactor = ScalingFactor[::-1]
    VMR_layer = VMR_layer[::-1,:]

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

    TAURAY = tau_rayleighj(wave_grid=wave_grid,TOTAM=U_layer)
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
