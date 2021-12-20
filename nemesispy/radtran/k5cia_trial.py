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
def calc_tau_cia(K_CIA,ISPACE,WAVE_GRID,ID,ISO,VMR_layer):

    # Need to pass NLAY from a atm profile
    DELH = None
    TOTAM = None
    TEMPS = None
    T_layer = None
    AMAGAT = 2.68675E19 #mol cm-3


    NLAY,NVMR = VMR_layer.shape

    # mixing ratios of the relevant gases
    qh2 = np.zeros(NLAY)
    qhe = np.zeros(NLAY)
    qn2 = np.zeros(NLAY)
    qch4 = np.zeros(NLAY)
    qco2 = np.zeros(NLAY)
    IABSORB = np.ones(5,dtype='int32') * -1

    for iVMR in range(NVMR):

        if ID[iVMR] == 39: # hydrogen
            qh2[:] = VMR_layer[:,iVMR]
            IABSORB[0] = iVMR

        if ID[iVMR] == 40: # helium
            qhe[:] = VMR_layer[:,iVMR]
            IABSORB[1] = iVMR

        if ID[iVMR] == 22: # nitrogen
            qn2[:] = VMR_layer[:,iVMR]
            IABSORB[2] = iVMR

        if ID[iVMR] == 6: # methane
            qch4[:] = VMR_layer[:,iVMR]
            IABSORB[3] = iVMR

        if ID[iVMR] == 2: # co2
            qco2[:] = VMR_layer[:,iVMR]
            IABSORB[4] = iVMR

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


        # if((WAVEN.min()<self.WAVEN.min()) or (WAVEN.max()>self.WAVEN.max())):
        #     print('warning in CIA :: Calculation wavelengths expand a larger range than in .cia file')

        # calculate the CIA opacity at the correct temperature and wavenumber
        NWAVEC = len(WAVE_GRID)
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
            kthi = K_CIA















"/Users/jingxuanyang/Desktop/Workspace/nemesispy2022/nemesispy/data/cia/exocia_hitran12_200-3800K.tab"
def read_cia(self,runname,raddata='/Users/aldayparejo/Documents/Projects/PlanetaryScience/NemesisPy-dist/NemesisPy/Data/cia/'):
    """
    Read the .cia file
    Parameters
    ----------
    runname: str
        Name of the NEMESIS run
    """

    #Reading .cia file
    f = open(runname+'.cia','r')
    s = f.readline().split()
    cianame = s[0]
    s = f.readline().split()
    dnu = float(s[0])
    s = f.readline().split()
    npara = int(s[0])
    f.close()

    if npara!=0:
        sys.exit('error in read_cia :: routines have not been adapted yet for npara!=0')

    #Reading the actual CIA file
    if npara==0:
        NPAIR = 9

    f = FortranFile(raddata+cianame, 'r' )
    TEMPS = f.read_reals( dtype='float64' )
    KCIA_list = f.read_reals( dtype='float32' )
    NT = len(TEMPS)
    NWAVE = int(len(KCIA_list)/NT/NPAIR)

    NU_GRID = np.linspace(0,dnu*(NWAVE-1),NWAVE)
    K_CIA = np.zeros([NPAIR, NT, NWAVE])

    index = 0
    for iwn in range(NWAVE):
        for itemp in range(NT):
            for ipair in range(NPAIR):
                K_CIA[ipair,itemp,iwn] = KCIA_list[index]
                index += 1

    self.NWAVE = NWAVE
    self.NT = NT
    self.NPAIR = NPAIR
    self.WAVEN = NU_GRID
    self.TEMP = TEMPS
    self.K_CIA = K_CIA