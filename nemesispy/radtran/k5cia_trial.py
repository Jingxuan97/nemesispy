import numpy as np
import sys
from scipy.io import FortranFile

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
def calc_tau_cia(ISPACE,WAVE_GRID):







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