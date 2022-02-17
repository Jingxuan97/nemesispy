import numpy as np
from scipy import interpolate
from numba import jit



#@jit(nopython=True)
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