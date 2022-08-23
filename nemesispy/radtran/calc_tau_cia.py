#!/usr/local/bin/python3
# -*- coding: utf-8 -*-
"""
Calculate collision-induced-absorption optical path.
"""
import numpy as np
from numba import jit

@jit(nopython=True)
def calc_tau_cia(wave_grid, K_CIA, ISPACE,
    ID, TOTAM, T_layer, P_layer, VMR_layer, DELH,
    cia_nu_grid, TEMPS, INORMAL, NPAIR=9):
    """
    Calculates
    Parameters
    ----------
    wave_grid : ndarray
        Wavenumber (cm-1) or wavelength array (um) at which to compute
        CIA opacities.
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
        it = (np.abs(TEMPS-temp1)).argmin()

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