from data.mol_info import mol_info
import numpy as np
from numba import jit
from constants import C_LIGHT, K_B, PLANCK
import time

def calc_mmw(ID, VMR, ISO=None):
    """
    Calculate mean molecular weight using the information in
    Reference/mol_info.py. Molecules are referenced by their Radtran ID
    specified in Reference/radtran_id.py. By default, terrestrial
    elative isotopic abundance is assumed.

    Inputs
    ------
    ID: array,
        List of gases specified by their Radtran identifiers.
    VMR: array,
        Corresponding VMR of the gases.
    ISO: array,
        If ISO = None then terrestrial relative isotopic abundance is assumed.
        If you want to specify particular isotopes, input the Radtran isotope
        identifiers here (see ref_id.py).

    Returns
    -------
    MMW: real,
        Mean molecular weight.
    """
    NGAS = len(ID)
    MMW = 0
    for i in range(NGAS):
        if ISO == None:
            mass = mol_info['{}'.format(ID[i])]['mmw']
        else:
            mass = mol_info['{}'.format(ID[i])]['isotope']\
                ['{}'.format(ISO[i])]['mass']
        MMW += VMR[i] * mass
    return MMW

@jit(nopython=True)
def blackbody_SI(wl, T):
    """
    Calculate blackbody radiance in SI units.

    Parameters
    ----------
    wl : real
        Wavelength in m.
    T : real
        Temperature in K.

    Returns
    -------
    radiance : real
        Radiance in W sr-1 m-2 m-1.
    """
    h = PLANCK
    c = C_LIGHT
    k = K_B
    radiance = (2*h*c**2)/(wl**5)*(1/(np.exp((h*c)/(wl*k*T)) - 1))
    return radiance

@jit(nopython=True)
def blackbody_um(wl, T):
    """
    Calculate blackbody radiance in W cm-2 sr-1 um-1.

    Parameters
    ----------
    wl : real
        Wavelength in um.
    T : real
        Temperature in K.

    Returns
    -------
    radiance : real
       Radiance in W cm-2 sr-1 um-1.
    """
    h = PLANCK
    c = C_LIGHT
    k = K_B
    radiance = (2*h*c**2)/((wl*1e-6)**5)*(1/(np.exp((h*c)/((wl*1e-6)*k*T))-1))*1e-10
    return radiance

def test_calc_mmw():
    ID = [1,2,3]
    ISO = [1,1,1]
    VMR = [0.1,0.1,0.8]
    mmw = calc_mmw(ID,VMR,ISO)
    ID = [1,2,3]
    ISO = None
    VMR = [0.1,0.1,0.8]
    s = time.time()
    for i in range(100):
        mmw = calc_mmw(ID,VMR,ISO)
    e = time.time()
    print(mmw)
    print('time',e-s)
# test_calc_mmw()

def test_blackbody_m():
    wl = 1e-6
    T = np.linspace(100,1000)
    s = time.time()
    for i in range(100):
        radiance = blackbody_SI(wl,T)
    e = time.time()
    print(radiance)
    print('time',e-s)
#test_blackbody_m()

def test_blackbody_um():
    wl = 1
    T = np.linspace(100,1000)
    s = time.time()
    for i in range(1000000):
        radiance = blackbody_um(wl,T)
    e = time.time()
    print(radiance)
    print('time',e-s)
#test_blackbody_um()

def planck_wave(v, T):
    """
      Inputs
      ------
      IWAVE: int
          Indicates wavenumbers (0) or wavelength(1) for units.

      v: real
          Wavenumber (cm-1) or wavelength (um)

      T: real
          Temperature in K

      Returns
      -------
      Calculated function is in units of W cm-2 sr-1 cm for IWAVE=0
          or W cm-2 sr-1 um-1 for IWAVE=1
    """
    IWAVE = 1
    c1 = 1.1911e-12
    c2 = 1.439
    if IWAVE == 0:
        y = v
        a = c1*v**3
    else:
        y = 1e4/v
        a = c1*y**5/1e4

    tmp = c2*y/T
    b = np.exp(tmp) - 1
    planck_wave = a/b
    return planck_wave

def blackbody(T,wl):
    """
    This function takes in a temperature (T) and a
    wavelength grid (wl) and returns a blackbody flux grid.
    This is used to compute the layer/slab "emission". All in MKS units.

    Parameters
    ----------
    T : float
        temperature of blackbody in kelvin
    wl : ndarray
        wavelength grid in meters

    Returns
    -------
    B : ndarray
        an array of blackbody fluxes (W/m2/m/ster) at each wavelength (size Nwavelengths)
    """
    # Define constants used in calculation
    h = 6.626E-34
    c = 3.0E8
    k = 1.38E-23

    # Calculate Blackbody Flux (B) at each wavelength point (wl)
    B = ((2.0*h*c**2.0)/(wl**5.0))*(1.0/(np.exp((h*c)/(wl*k*T)) - 1.0))

    # Return blackbody flux
    return B


def test3():
    s = time.time()
    files = ['./data/ktables/h2o','./data/ktables/co2','./data/ktables/co','./data/ktables/ch4']
    gas_id_list, iso_id_list, wave, g_ord, del_g,\
        P_grid, T_grid, k_gas_w_g_p_t = read_kls(files)
    P_grid = np.log10(P_grid)
    T_grid = np.log10(T_grid)
    P_atm = np.array([7.42239701e+05, 2.81660408e+05, 1.08371277e+05, 4.22976280e+04,
          1.67426473e+04, 6.66814550e+03, 2.69006862e+03, 1.10450126e+03,
          4.54217702e+02, 1.86590540e+02])*1e-5
    T_atm = np.array([3014.12747492, 3014.11537511, 3014.11076141, 3014.30110731,
          3036.04559052, 3204.23046439, 3739.65219224, 4334.23131605,
          4772.68340169, 4964.2941274 ])*0.5
    P_atm = np.log10(P_atm)
    T_atm = np.log10(T_atm)
    k_gas_w_g_p_t = np.log10(k_gas_w_g_p_t)
    frac = np.array([[1.e-10, 1.e-04, 1.e-04, 1.e-10],
       [1.e-10, 1.e-04, 1.e-04, 1.e-10],
       [1.e-10, 1.e-04, 1.e-04, 1.e-10],
       [1.e-10, 1.e-04, 1.e-04, 1.e-10],
       [1.e-10, 1.e-04, 1.e-04, 1.e-10],
       [1.e-10, 1.e-04, 1.e-04, 1.e-10],
       [1.e-10, 1.e-04, 1.e-04, 1.e-10],
       [1.e-10, 1.e-04, 1.e-04, 1.e-10],
       [1.e-10, 1.e-04, 1.e-04, 1.e-10],
       [1.e-10, 1.e-04, 1.e-04, 1.e-10]])
    mass_path = np.array([7.86544802e+30, 3.01842190e+30, 1.16818592e+30, 4.56144181e+29,
       1.79953458e+29, 7.23170929e+28, 2.84773251e+28, 1.16671235e+28,
       4.88662629e+27, 2.09772744e+27])*1e-4
    for i in range(100):
        k_l_w_gas_g = 10**interp_k(P_grid, T_grid, P_atm, T_atm, k_gas_w_g_p_t)
        ck = k_l_w_gas_g[:,10,:,:]
        dtau = compute_tau(ck, mass_path, frac, g_ord, del_g)
    e = time.time()
    print('time',e-s)
# test3()