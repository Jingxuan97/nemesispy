import numpy as np
import sys
from numpy.core.numeric import isfortran
from scipy.io import FortranFile

from nemesispy2022.nemesispy.radtran.k0run import T_layer

"""
@param INORMAL: int,
    Flag indicating whether the ortho/para-H2 ratio is in equilibrium (0 for 1:1)
    or normal (1 for 3:1)
@param NPAIR: int,
    Number of gaseous pairs listed
    (Default = 9 : H2-H2 (eqm), H2-He (eqm), H2-H2 (normal), H2-He (normal),
    H2-N2, H2-CH4, N2-N2, CH4-CH4, H2-CH4)
@param NT: int,
    Number of temperature levels over which the CIA data is defined
@param NWAVE: int,
    Number of spectral points over which the CIA data is defined
"""

def find_nearest(array, value):

    """
    FUNCTION NAME : find_nearest()

    DESCRIPTION : Find the closest value in an array

    INPUTS :

        array :: List of numbers
        value :: Value to search for

    OPTIONAL INPUTS: none

    OUTPUTS :

        closest_value :: Closest number to value in array
        index :: Index of closest_value within array

    CALLING SEQUENCE:

        closest_value,index = find_nearest(array,value)

    MODIFICATION HISTORY : Juan Alday (29/04/2019)

    """

    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx],idx

def read_cia(filepath,dnu=10,npara=0):
    """
    Parameters
    ----------
    filepath : str
        Filepath to the .tab file containing CIA information.
    dnu : real, optional
        Wavenumber interval. The default is 10.
    npara : int, optional
        DESCRIPTION. The default is 0.

    Returns
    -------
    NU_GRID(NWAVE) : ndarray
        Wavenumber array (NOTE: ALWAYS IN WAVENUMBER, NOT WAVELENGTH).
    TEMPS(NTEMP) : ndarray
        Temperature levels at which the CIA data is defined (K).
    K_CIA(NPAIR,NTEMP,NWAVE) : ndarray
         CIA cross sections for each pair at each temperature level and wavenumber.
    """

    if npara != 0:
        # might need sys.exit'
        raise('Routines have not been adapted yet for npara!=0')

    # Reading the actual CIA file
    if npara == 0:
        NPAIR = 9 # 9 pairs of collision induced absorption opacities

    f = FortranFile(filepath, 'r')
    TEMPS = f.read_reals( dtype='float64' )
    print(TEMPS)
    KCIA_list = f.read_reals( dtype='float32' )
    NT = len(TEMPS)
    NWAVE = int(len(KCIA_list)/NT/NPAIR)

    NU_GRID = np.linspace(0,dnu*(NWAVE-1),NWAVE)
    K_CIA = np.zeros([NPAIR,NT,NWAVE]) # NPAIR x NT x NWAVE

    index = 0
    for iwn in range(NWAVE):
        for itemp in range(NT):
            for ipair in range(NPAIR):
                K_CIA[ipair,itemp,iwn] = KCIA_list[index]
                index += 1

    return NU_GRID, TEMPS, K_CIA

from scipy import interpolate
def calc_tau_cia(WAVE_GRID,TOTAM,ID,ISO,VMR_layer,ISPACE,K_CIA,NU_GRID,TEMPS,INORMAL):
    """
    Parameters
    ----------
    WAVE_GRID : ndarray
        Wavenumber (cm-1) or wavelength array (um) at which to compute CIA opacities.
    ID : ndarray
        Gas ID
    ISO : ndarray
        DESCRIPTION.
    VMR_layer : TYPE
        DESCRIPTION.
    ISPACE : int
        Flag indicating whether the calculation must be performed in wavenumbers (0) or wavelength (1)
    K_CIA(NPAIR,NTEMP,NWAVE) : ndarray
         CIA cross sections for each pair at each temperature level and wavenumber.
    NU_GRID : TYPE
        DESCRIPTION.
    INORMAL : int


    Returns
    -------
    tau_cia_layer(NWAVE,NLAY) : ndarray
        CIA optical depth in each atmospheric layer.
    """

    # Need to pass NLAY from a atm profile
    DELH = np.ones(10)
    T_layer = np.ones(10)
    AMAGAT = 2.68675E19 #mol cm-3
    NPAIR = 9


    NLAY,NVMR = VMR_layer.shape

    # mixing ratios of the relevant gases
    qh2 = np.zeros(NLAY)
    qhe = np.zeros(NLAY)
    qn2 = np.zeros(NLAY)
    qch4 = np.zeros(NLAY)
    qco2 = np.zeros(NLAY)
    # IABSORB = np.ones(5,dtype='int32') * -1

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

        amag1 = TOTAM / XLEN / AMAGAT # number density
        tau = XLEN*amag1**2 # optical path, why fiddle around with XLEN

        # define the calculatiion wavenumbers, WAVE_GRID IS WAVEC
        if ISPACE == 0:
            WAVEN = WAVE_GRID
        elif ISPACE == 1:
            WAVEN = 1.E4/WAVE_GRID
            isort = np.argsort(WAVEN)
            WAVEN = WAVEN[isort]

        if WAVEN.min() < NU_GRID.min() or WAVEN.max()>NU_GRID.max():
            print('warning in CIA :: Calculation wavelengths expand a larger range than in .cia file')

        # calculate the CIA opacity at the correct temperature and wavenumber
        NWAVEC = len(WAVE_GRID)  # Number of calculation wavelengths
        tau_cia_layer = np.zeros([NWAVEC,NLAY])

        for ilay in range(NLAY):

            # interpolating to the correct temperature
            temp1 = T_layer[ilay]
            temp0,it = find_nearest(TEMPS,temp1)

            # want to sandwich the T point
            if TEMPS[it] >= temp1:
                ithi = it
                if it==0:
                    temp1 = TEMPS[it]
                    itl = 0
                    ithi = 1
                else:
                    itl = it - 1

            elif TEMPS[it]<temp1:
                NT = len(TEMPS)
                itl = it
                if it == NT - 1:
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
            """THERE ARE TWO SETS OF WAVE GRIDS"""

            inwave = np.where( (NU_GRID>=WAVEN.min()) & (NU_GRID<=WAVEN.max()) )
            inwave = inwave[0]

            if len(inwave)>0:

                k_cia = np.zeros([NWAVEC,NPAIR])
                inwave1 = np.where( (WAVEN>=NU_GRID.min())&(WAVEN<=NU_GRID.max()) )
                inwave1 = inwave1[0]

                for ipair in range(NPAIR):

                    f = interpolate.interp1d(NU_GRID,kt[ipair,:])
                    k_cia[inwave1,ipair] = f(WAVEN[inwave1])

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


                sum1[:] = sum1[:] + k_co2[:] * qco2[ilay] * qco2[ilay]
                #Look up N2-N2 NIR CIA coefficients


                #Look up N2-H2 NIR CIA coefficients



                tau_cia_layer[:,ilay] = sum1[:] * tau[ilay]

        if ISPACE==1:
            tau_cia_layer[:,:] = tau_cia_layer[isort,:]

    return tau_cia_layer



def co2cia(WAVEN):
    """
    Subroutine to return CIA absorption coefficients for CO2-CO2

    @param WAVEN: 1D array
        Wavenumber array (cm-1)
    """

    WAVEL = 1.0e4/WAVEN
    CO2CIA = np.zeros(len(WAVEN))

    #2.3 micron window. Assume de Bergh 1995 a = 4e-8 cm-1/amagat^2
    iin = np.where((WAVEL>=2.15) & (WAVEL<=2.55))
    iin = iin[0]
    if len(iin)>0:
        CO2CIA[iin] = 4.0e-8

    #1.73 micron window. Assume mean a = 6e-9 cm-1/amagat^2
    iin = np.where((WAVEL>=1.7) & (WAVEL<=1.76))
    iin = iin[0]
    if len(iin)>0:
        CO2CIA[iin] = 6.0e-9

    #1.28 micron window. Update from Federova et al. (2014) to
    #aco2 = 1.5e-9 cm-1/amagat^2
    iin = np.where((WAVEL>=1.25) & (WAVEL<=1.35))
    iin = iin[0]
    if len(iin)>0:
        CO2CIA[iin] = 1.5e-9

    #1.18 micron window. Assume a mean a = 1.5e-9 cm-1/amagat^2
    #if(xl.ge.1.05.and.xl.le.1.35)aco2 = 1.5e-9
    #Update from Federova et al. (2014)
    iin = np.where((WAVEL>=1.125) & (WAVEL<=1.225))
    iin = iin[0]
    if len(iin)>0:
        CO2CIA[iin] = 0.5*(0.31+0.79)*1e-9

    #1.10 micron window. Update from Federova et al. (2014)
    iin = np.where((WAVEL>=1.06) & (WAVEL<=1.125))
    iin = iin[0]
    if len(iin)>0:
        CO2CIA[iin] = 0.5*(0.29+0.67)*1e-9

    return CO2CIA