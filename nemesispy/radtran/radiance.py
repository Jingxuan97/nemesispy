#!/usr/local/bin/python3
# -*- coding: utf-8 -*-
"""
Routines to calculate radiance from a planetary atmosphere taking account of:
    - Thermal emission
    - Collision-induced absorption from H2-H2, H2-he, H2-N2, N2-Ch4, N2-N2,
        Ch4-Ch4, H2-Ch4
    - Rayleigh scattering H2 molecules and He molecules
"""
from copy import copy
# from locale import nl_langinfo
import numpy as np
import sys
sys.path.append('/Users/jingxuanyang/Desktop/Workspace/nemesispy2022/')
# from numpy.core.defchararray import array
from nemesispy.radtran.interp import interp_k
from nemesispy.radtran.interp import noverlapg
# from nemesispy.radtran.cia import find_nearest
from scipy import interpolate
from numba import jit
import math
speed_up = False

# @jit(nopython=True)
def find_nearest(input_array, target_value):
    """
    Find the closest value in an array

    Parameters
    ----------
    input_array : ndarray/list
        An array of numbers.
    target_value : real
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

# @jit(nopython=True)
def calc_tau_cia(wave_grid, K_CIA, ISPACE,
    ID, TOTAM, T_layer, P_layer, VMR_layer, DELH,
    cia_nu_grid, TEMPS, INORMAL, NPAIR=9):
    """
    Parameters
    ----------
    wave_grid : ndarray
        Wavenumber (cm-1) or wavelength array (um) at which to compute CIA opacities.
    ID : ndarray
        Gas ID
    # ISO : ndarray
    #     Isotop ID.
    VMR_layer : TYPE
        DESCRIPTION.
    ISPACE : int
        Flag indicating whether the calculation must be performed in
        wavenumbers (0) or wavelength (1)
    K_CIA(NPAIR,NTEMP,NWAVE) : ndarray
         CIA cross sections for each pair at each temperature level and wavenumber.
    cia_nu_grid : TYPE
        DESCRIPTION.
    INORMAL : int


    Returns
    -------
    tau_cia_layer(NWAVE,NLAY) : ndarray
        CIA optical depth in each atmospheric layer.
    """

    # Need to pass NLAY from a atm profile
    NPAIR = 9

    NLAY,NVMR = VMR_layer.shape
    ISO = np.zeros((NVMR))

    # mixing ratios of the relevant gases
    qh2 = np.zeros((NLAY))
    qhe = np.zeros((NLAY))
    qn2 = np.zeros((NLAY))
    qch4 = np.zeros((NLAY))
    qco2 = np.zeros((NLAY))
    # IABSORB = np.ones(5,dtype='int32') * -1

    NWAVEC = 17

    # get mixing ratios from VMR grid
    for iVMR in range(NVMR):
        if ID[iVMR] == 39: # hydrogen
            qh2[:] = VMR_layer[:,iVMR]
            # IABSORB[0] = iVMR
        if ID[iVMR] == 40: # helium
            qhe[:] = VMR_layer[:,iVMR]
            # IABSORB[1] = iVMR
        if ID[iVMR] == 22: # nitrogen
            qn2[:] = VMR_layer[:,iVMR]
            # IABSORB[2] = iVMR
        if ID[iVMR] == 6: # methane
            qch4[:] = VMR_layer[:,iVMR]
            # IABSORB[3] = iVMR
        if ID[iVMR] == 2: # co2
            qco2[:] = VMR_layer[:,iVMR]
            # IABSORB[4] = iVMR

    # calculating the opacity
    XLEN = DELH * 1.0e2 # cm
    TOTAM = TOTAM * 1.0e-4 # cm-2

    ### back to FORTRAN ORIGINAL
    P0=101325
    T0=273.15
    AMAGAT = 2.68675E19 #mol cm-3
    KBOLTZMANN = 1.381E-23
    MODBOLTZA = 10.*KBOLTZMANN/1.013

    tau = (P_layer/P0)**2 * (T0/T_layer)**2 * DELH
    height1 = P_layer * MODBOLTZA * T_layer

    height = XLEN * 1e2
    amag1 = TOTAM /height/AMAGAT
    tau = height*amag1**2

    AMAGAT = 2.68675E19 #mol cm-3
    amag1 = TOTAM / XLEN / AMAGAT # number density
    tau = XLEN*amag1**2# optical path, why fiddle around with XLEN

    # define the calculatiion wavenumbers
    if ISPACE == 0: # input wavegrid is already in wavenumber (cm^-1)
        WAVEN = wave_grid
    elif ISPACE == 1:
        WAVEN = 1.e4/wave_grid
        isort = np.argsort(WAVEN)
        WAVEN = WAVEN[isort] # ascending wavenumbers

    # if WAVEN.min() < cia_nu_grid.min() or WAVEN.max()>cia_nu_grid.max():
    #     print('warning in CIA :: Calculation wavelengths expand a larger range than in .cia file')

    # calculate the CIA opacity at the correct temperature and wavenumber
    NWAVEC = len(wave_grid)  # Number of calculation wavelengths
    tau_cia_layer = np.zeros((NWAVEC,NLAY))

    for ilay in range(NLAY):
        # interpolating to the correct temperature
        temp1 = T_layer[ilay]
        temp0,it = find_nearest(TEMPS,temp1)

        # want to sandwich the T point
        if TEMPS[it] >= temp1:
            ithi = it
            if it==0:
                # edge case, layer T < T grid
                temp1 = TEMPS[it]
                itl = 0
                ithi = 1
            else:
                itl = it - 1

        elif TEMPS[it]<temp1:
            NT = len(TEMPS)
            itl = it
            if it == NT - 1:
                # edge case, layer T > T grid
                temp1 = TEMPS[it]
                ithi = NT - 1
                itl = NT - 2
            else:
                ithi = it + 1

        # find opacities for the chosen T
        ktlo = K_CIA[:,itl,:]
        kthi = K_CIA[:,ithi,:]

        fhl = (temp1 - TEMPS[itl])/(TEMPS[ithi]-TEMPS[itl])
        fhh = (TEMPS[ithi]-temp1)/(TEMPS[ithi]-TEMPS[itl])

        kt = ktlo * (1.-fhl) + kthi * (1.-fhh)

        # checking that interpolation can be performed to the calculation wavenumbers
        inwave = np.where( (cia_nu_grid>=WAVEN.min()) & (cia_nu_grid<=WAVEN.max()) )
        inwave = inwave[0]

        if len(inwave)>0:

            k_cia = np.zeros((NWAVEC,NPAIR))
            inwave1 = np.where( (WAVEN>=cia_nu_grid.min())&(WAVEN<=cia_nu_grid.max()) )
            inwave1 = inwave1[0]

            for ipair in range(NPAIR):

                # wavenumber interpolation
                # f = interpolate.interp1d(cia_nu_grid,kt[ipair,:])
                # k_cia[inwave1,ipair] = f(WAVEN[inwave1])

                # use numpy for numba integration
                k_cia[inwave1,ipair] = np.interp(WAVEN[inwave1],cia_nu_grid,kt[ipair,:])

            #Combining the CIA absorption of the different pairs (included in .cia file)
            sum1 = np.zeros(NWAVEC)
            if INORMAL==0: # equilibrium hydrogen (1:1)
                sum1[:] = sum1[:] + k_cia[:,0] * qh2[ilay] * qh2[ilay] \
                    + k_cia[:,1] * qhe[ilay] * qh2[ilay]

            elif INORMAL==1: # normal hydrogen (3:1)
                sum1[:] = sum1[:] + k_cia[:,2] * qh2[ilay] * qh2[ilay]\
                    + k_cia[:,3] * qhe[ilay] * qh2[ilay]

            sum1[:] = sum1[:] + k_cia[:,4] * qh2[ilay] * qn2[ilay]
            sum1[:] = sum1[:] + k_cia[:,5] * qn2[ilay] * qch4[ilay]
            sum1[:] = sum1[:] + k_cia[:,6] * qn2[ilay] * qn2[ilay]
            sum1[:] = sum1[:] + k_cia[:,7] * qch4[ilay] * qch4[ilay]
            sum1[:] = sum1[:] + k_cia[:,8] * qh2[ilay] * qch4[ilay]

            # look up CO2-CO2 CIA coefficients (external)
            """
            TO BE DONE
            """
            k_co2 = sum1*0
            # k_co2 = co2cia(WAVEN)

            sum1[:] = sum1[:] + k_co2[:] * qco2[ilay] * qco2[ilay]

            #Look up N2-N2 NIR CIA coefficients
            """
            TO BE DONE
            """
            # TO BE DONE

            #Look up N2-H2 NIR CIA coefficients
            """
            TO BE DONE
            """
            # TO BE DONE

            tau_cia_layer[:,ilay] = sum1[:] * tau[ilay]

    if ISPACE==1:
        tau_cia_layer[:,:] = tau_cia_layer[isort,:]

    return tau_cia_layer

# @jit(nopython=True)
def calc_planck(wave,temp,ispace=1):
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
        Planck function
        Unit: (0) W cm-2 sr-1 (cm-1)-1
              (1) W cm-2 sr-1 um-1
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
        raise Exception('error in calc_planck: ISPACE must be either 0 or 1')

    tmp = c2 * y / temp
    b = np.exp(tmp) - 1
    # bb = np.array((a/b),dtype=np.float32)
    bb = (a/b)
    return bb

# @jit(nopython=True)
def calc_tau_rayleighj(wave_grid,TOTAM,ISPACE=1):
    """
    Calculate the Rayleigh scattering opacity in each atmospheric layer for
    Gas Giant atmospheres using data from Allen (1976) Astrophysical Quantities

    Parameters
    ----------
        ISPACE : int
            Flag indicating the spectral units
            (0) Wavenumber in cm-1 or (1) Wavelegnth (um)
        wave_grid(NWAVE) : ndarray
            Wavenumber (cm-1) or wavelength array (um)

    Outputs
    -------
    tau_rayleigh(NWAVE,NLAYER) : ndarray
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

    NLAYER = len(TOTAM)
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

    # Calculate the scattering cross sections in m2
    k_rayleighj = temp*faniso/(3.*(x**2)) #(NWAVE)

    # Calculate the Rayleigh opacities in each layer
    tau_rayleigh = np.zeros((len(wave_grid),NLAYER))

    for ilay in range(NLAYER):
        tau_rayleigh[:,ilay] = k_rayleighj[:] * TOTAM[ilay] #(NWAVE,NLAYER)

    return tau_rayleigh*0 ### rayleigh is 0 for debug

# @jit(nopython=True)
def calc_tau_gas_fortran(k_gas_w_g_p_t, P_layer, T_layer, VMR_layer, U_layer,
    P_grid, T_grid, del_g):
    """
    Parameters
    ----------
    k_gas_w_g_p_t(NGAS,NWAVE,NG,NPRESSKTA,NTEMPKTA) : ndarray
        Raw k-coefficients.
        Has dimension: NWAVE x NG x NPRESSKTA x NTEMPKTA.
    P_layer(NLAYER) : ndarray
        Atmospheric pressure grid.
    T_layer(NLAYER) : ndarray
        Atmospheric temperature grid.
    VMR_layer(NLAYER,NGAS) : ndarray
        Array of volume mixing ratios for NGAS.
        Has dimensioin: NLAYER x NGAS
    U_layer : ndarray
        DESCRIPTION.
    P_grid(NPRESSKTA) : ndarray
        Pressure grid on which the k-coeff's are pre-computed.
    T_grid(NTEMPKTA) : ndarray
        Temperature grid on which the k-coeffs are pre-computed.
    del_g : ndarray
        DESCRIPTION.

    Returns
    -------
    tau_w_g_l(NWAVE,NG,NLAYER) : ndarray
        DESCRIPTION.
    """

    Scaled_U_layer = U_layer * 1.0e-20 # absorber amounts (U_layer) is scaled by a factor 1e-20
    Scaled_U_layer *= 1.0e-4 # convert from absorbers per m^2 to per cm^2

    k_gas_w_g_l = interp_k(P_grid, T_grid, P_layer, T_layer, k_gas_w_g_p_t)
    Ngas, Nwave, Ng, Nlayer = k_gas_w_g_l.shape

    amount_layer = np.zeros((Nlayer,Ngas))
    for ilayer in range(Nlayer):
        amount_layer[ilayer,:] = Scaled_U_layer[ilayer] * VMR_layer[ilayer,:4]

    tau_w_g_l = np.zeros((Nwave,Ng,Nlayer))
    for iwave in range (Nwave):
        k_gas_g_l = k_gas_w_g_l[:,iwave,:,:]
        k_g_l = np.zeros((Ng,Nlayer))
        for ilayer in range(Nlayer):
            k_g_l[:,ilayer]\
                = noverlapg(k_gas_g_l[:,:,ilayer],amount_layer[ilayer,:],del_g)
            tau_w_g_l[iwave,:,ilayer] = k_g_l[:,ilayer]

    return tau_w_g_l

# # @jit(nopython=True)
def calc_radiance(wave_grid, U_layer, P_layer, T_layer, VMR_layer, k_gas_w_g_p_t,
    P_grid, T_grid, del_g, ScalingFactor, RADIUS, solspec,
    k_cia, ID, cia_nu_grid, cia_T_grid, DEL_S):
    """
    Calculate emission spectrum using the correlated-k method.

    # Absorber amounts (U_layer) is scaled by a factor 1e-20 because Nemesis
    # k-tables are scaled by a factor of 1e20. Done in calc_tau_gas.

    # Need to be smart about benchmarking against NEMESIS

    Parameters
    ----------
    wave_grid(NWAVE) : ndarray
        Wavelengths (um) grid for calculating spectra.
    U_layer(NLAYER) : ndarray
        Total number of gas particles in each layer.
        We want SI unit (no. of particle/m^2) here.
    P_layer(NLAYER) : ndarray
        Atmospheric pressure grid.
        We want SI unit (Pa) here.
    T_layer(NLAYER) : ndarray
        Atmospheric temperature grid. In Kelvin.
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
    ScalingFactor(NLAYER) : ndarray ##
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
    SPECOUT : ndarray
        Output spectrum (W cm-2 um-1 sr-1)
    """
    # Reverse layer ordering from TOP of atmoaphsere first to BOTTOM of atmosphere last
    ScalingFactor = ScalingFactor[::-1]
    P_layer = P_layer[::-1]
    T_layer = T_layer[::-1]
    U_layer = U_layer[::-1]
    VMR_layer = VMR_layer[::-1,:]
    DEL_S = DEL_S[::-1]

    # Record constants
    NGAS, NWAVE, NG, NPRESS, NTEMP = k_gas_w_g_p_t.shape
    NLAYER = len(P_layer)

    # Initiate arrays to record optical paths (NWAVE x NG x NLAYER)
    tau_total_w_g_l = np.zeros((NWAVE,NG,NLAYER)) # Total Optical path

    # Collision induced absorptioin optical path (NWAVE x NLAYER)
    tau_cia = calc_tau_cia(wave_grid=wave_grid,K_CIA=k_cia,ISPACE=1,
        ID=ID,TOTAM=U_layer,T_layer=T_layer,P_layer=P_layer,VMR_layer=VMR_layer,
        DELH=DEL_S,cia_nu_grid=cia_nu_grid,TEMPS=cia_T_grid,INORMAL=1,NPAIR=9)

    # Rayleigh scattering optical path (NWAVE x NLAYER)
    tau_rayleigh = calc_tau_rayleighj(wave_grid=wave_grid,TOTAM=U_layer)

    # Dust scattering optical path (NWAVE x NLAYER)
    """To be done"""
    tau_dust = np.zeros((NWAVE,NLAYER))

    # # Active gas optical path (NWAVE x NG x NLAYER)
    # tau_gas = calc_tau_gas(k_gas_w_g_p_t, P_layer, T_layer, VMR_layer, U_layer,
    #         P_grid, T_grid, del_g)

    # FORTRAN straight transcript
    tau_gas = calc_tau_gas_fortran(k_gas_w_g_p_t, P_layer, T_layer, VMR_layer, U_layer,
            P_grid, T_grid, del_g)

    # Merge all different opacities
    for ig in range(NG):
        tau_total_w_g_l[:,ig,:] = tau_gas[:,ig,:] + tau_cia[:,:] \
            + tau_dust[:,:] + tau_rayleigh[:,:]

    #Scale to the line-of-sight opacities
    tau_total_w_g_l = tau_total_w_g_l * ScalingFactor

    # Thermal Emission Calculation, IMOD = 3
    spec_out = np.zeros((NWAVE,NG))

    # Defining the units of the output spectrum / divide by stellar spectrum
    # IFORM = 1
    # radextra = np.sum(DEL_S[:-1])
    radextra = 0

    xfac = np.pi*4.*np.pi*((RADIUS+radextra)*1e2)**2.*np.ones(NWAVE)
    xfac = xfac / solspec[:]

    # old working method
    # Calculating atmospheric gases contribution
    tau_cumulative_w_g = np.zeros((NWAVE,NG))
    tr_old_w_g = np.ones((NWAVE,NG))
    spec_w_g = np.zeros((NWAVE,NG))

    for ilayer in range(NLAYER):
        tau_cumulative_w_g[:,:] =  tau_total_w_g_l[:,:,ilayer] + tau_cumulative_w_g[:,:]
        tr_w_g = np.exp(-tau_cumulative_w_g[:,:]) # transmission function
        bb = calc_planck(wave_grid, T_layer[ilayer]) # blackbody function
        # print(bb)

        # # vectorised
        # for ig in range(NG):
        #     spec_w_g[:,ig] = spec_w_g[:,ig]+(tr_old_w_g[:,ig]-tr_w_g[:,ig])*bb[:]

        for iwave in range(NWAVE):
            for ig in range(NG):
                spec_w_g[iwave,ig] = spec_w_g[iwave,ig] \
                    + np.float32(tr_old_w_g[iwave,ig]-tr_w_g[iwave,ig])\
                    *bb[iwave]*xfac[iwave]

        tr_old_w_g = copy(tr_w_g)


    # surface/bottom layer contribution
    p1 = P_layer[int(len(P_layer)/2-1)] #midpoint
    p2 = P_layer[-1] # lowest point in altitude/highest in pressure

    surface = None
    if p2 > p1: # i.e. if not a limb path

        if not surface:
            radground = calc_planck(wave_grid,T_layer[-1])
        for ig in range(NG):
            spec_w_g[:,ig] = spec_w_g[:,ig] \
                + np.float32(tr_old_w_g[:,ig])*radground *xfac

    spectrum = np.zeros((NWAVE))
    for iwave in range(NWAVE):
        for ig in range(NG):
            spectrum[iwave] += spec_w_g[iwave,ig]*del_g[ig]

    return spectrum



"""
# Incompatible methods with numba jit

# spec_out = np.tensordot(spec_out, del_g, axes=([1],[0])) * xfac
# spec_out = spec_out.T[0]
"""

""" # Fortran straight transcription
### pray it works
bb = np.zeros((NWAVE,NLAYER))
spectrum = np.zeros((NWAVE))
spec_w_g = np.zeros((NWAVE,NG))
for iwave in range(NWAVE):
    for ig in range(NG):
        taud = 0.
        trold = 1.
        for ilayer in range(NLAYER):
            taud = taud + tau_total_w_g_l[iwave,ig,ilayer]
            tr = np.exp(-taud)
            # print('taud',taud)
            # print('tr',tr)
            if ig == 0:
                bb[iwave,ilayer] = calc_planck(wave_grid[iwave],T_layer[ilayer])
            # print('bb',bb)
            # print('np.float32((trold-tr))',np.float32((trold-tr)))
            # print('xfac',xfac)
            # print(xfac*np.float32((trold-tr)) * bb[iwave,ilayer])

            spec_w_g[iwave,ig] = spec_w_g[iwave,ig] \
                + xfac[iwave]*np.float32((trold-tr)) * bb[iwave,ilayer]
            trold = tr
        p1 = P_layer[int(len(P_layer)/2-1)] #midpoint
        p2 = P_layer[-1] # lowest point in altitude/highest in pressure
        surface = None
        if p2 > p1:
            radground = calc_planck(wave_grid[iwave],T_layer[-1])
            spec_w_g[iwave,ig] = spec_w_g[iwave,ig] \
                + xfac[iwave]*np.float32(trold)*radground

for iwave in range(NWAVE):
    for ig in range(NG):
        spectrum[iwave] += spec_w_g[iwave,ig] * del_g[ig]
"""
# Fortran nemesis
# def calc_tau_gas(k_gas_w_g_p_t, P_layer, T_layer, VMR_layer, U_layer,
#     P_grid, T_grid, del_g):
#     """
#     Calculate the optical path due to gaseous absorbers.

#     Parameters
#     ----------
#     k_gas_w_g_p_t(NGAS,NWAVE,NG,NPRESSKTA,NTEMPKTA) : ndarray
#         Raw k-coefficients.
#         Has dimension: NWAVE x NG x NPRESSKTA x NTEMPKTA.
#     P_layer(NLAYER) : ndarray
#         Atmospheric pressure grid.
#     T_layer(NLAYER) : ndarray
#         Atmospheric temperature grid.
#     VMR_layer(NLAYER,NGAS) : ndarray
#         Array of volume mixing ratios for NGAS.
#         Has dimensioin: NLAYER x NGAS
#     U_layer : ndarray
#         DESCRIPTION.
#     P_grid(NPRESSKTA) : ndarray
#         Pressure grid on which the k-coeff's are pre-computed.
#     T_grid(NTEMPKTA) : ndarray
#         Temperature grid on which the k-coeffs are pre-computed.
#     del_g : ndarray
#         DESCRIPTION.

#     Returns
#     -------
#     tau_w_g_l(NWAVE,NG,NLAYER) : ndarray
#         DESCRIPTION.
#     """

#     U_layer *= 1.0e-20 # absorber amounts (U_layer) is scaled by a factor 1e-20
#     U_layer *= 1.0e-4 # convert from absorbers per m^2 to per cm^2

#     k_gas_w_g_l = interp_k(P_grid, T_grid, P_layer, T_layer, k_gas_w_g_p_t) # NGAS,NWAVE,NG,NLAYER

#     k_w_g_l,f_combined = new_k_overlap(k_gas_w_g_l, del_g, VMR_layer) # NWAVE,NG,NLAYER

#     utotl = U_layer

#     tau_gas = k_w_g_l * utotl * f_combined  # NWAVE, NG, NLAYER

#     return tau_gas