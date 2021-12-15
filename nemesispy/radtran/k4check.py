#!/usr/local/bin/python3
# -*- coding: utf-8 -*-
"""Routines to read pre-tabulated correlated-k look-up tables (k-tables).
All k-tables are assumed to share the same wavelength-grid, pressure-grid,
temperature-grid, g-ordinates and quadrature weights.
"""
import numpy as np
import sys
sys.path.append('/Users/jingxuanyang/Desktop/Workspace/nemesispy2022/')

def read_kta(filename):
    # filepath is more accurate than filename
    # this function is hopefully called once in a retrieval so no need to
    # optimise time
    """
    Reads a pre-tabulated correlated-k look-up table from a Nemesis .kta file.

    Parameters
    ----------
    filename : str
        The path to the Nemesis .kta file to be read.

    Returns
    -------
    gas_id : int
        Gas identifier.
    iso_id : int
        Isotope identifier.
    wave_grid : ndarray
        Wavenumber/wavelength grid of the k-table.
    g_ord : ndarray
        g-ordinates of the k-table.
    del_g : ndarray
        Quadrature weights of the g-ordinates.
    P_grid : ndarray
        Pressure grid on which the k-coeff's are pre-computed.
    T_grid : ndarray
        Temperature grid on which the k-coeffs are pre-computed.
    k_w_g_p_t : ndarray
        k-coefficients. Has dimension: Nwave x Ng x Npress x Ntemp.
    """
    # Open file
    if filename[-3:] == 'kta':
        f = open(filename,'rb')
    else:
        f = open(filename+'.kta','rb')

    # Define bytes consumed by elements of table
    nbytes_int32 = 4
    nbytes_float32 = 4
    ioff = 0

    # Read header
    irec0 = int(np.fromfile(f,dtype='int32',count=1))
    nwavekta = int(np.fromfile(f,dtype='int32',count=1))
    vmin = float(np.fromfile(f,dtype='float32',count=1))
    delv = float(np.fromfile(f,dtype='float32',count=1))
    fwhm = float(np.fromfile(f,dtype='float32',count=1))
    npress = int(np.fromfile(f,dtype='int32',count=1))
    ntemp = int(np.fromfile(f,dtype='int32',count=1))
    ng = int(np.fromfile(f,dtype='int32',count=1))
    gas_id = int(np.fromfile(f,dtype='int32',count=1))
    iso_id = int(np.fromfile(f,dtype='int32',count=1))
    ioff = ioff + 10*nbytes_int32

    # Read g-ordinates, g weights
    g_ord = np.fromfile(f,dtype='float32',count=ng)
    del_g = np.fromfile(f,dtype='float32',count=ng)
    ioff = ioff + 2*ng*nbytes_float32
    dummy = np.fromfile(f,dtype='float32',count=1)
    dummy = np.fromfile(f,dtype='float32',count=1)
    ioff = ioff + 2*nbytes_float32

    # Read temperature/pressure grid
    P_grid = np.fromfile(f,dtype='float32',count=npress)
    T_grid = np.fromfile(f,dtype='float32',count=ntemp)
    ioff = ioff + npress*nbytes_float32+ntemp*nbytes_float32

    # Calculate wavenumber/wavelength grid
    if delv>0.0:  # uniform grid
        vmax = delv*nwavekta + vmin
        wave_grid = np.linspace(vmin,vmax,nwavekta)
    else:   # non-uniform grid
        wave_grid = np.zeros([nwavekta])
        wave_grid = np.fromfile(f,dtype='float32',count=nwavekta)
        ioff = ioff + nwavekta*nbytes_float32
    nwave = len(wave_grid)

    #Read k-coefficients
    k_w_g_p_t = np.zeros([nwave,ng,npress,ntemp])

    #Jump to the minimum wavenumber
    ioff = (irec0-1)*nbytes_float32
    f.seek(ioff,0)

    #Reading the coefficients we require
    k_out = np.fromfile(f,dtype='float32',count=ntemp*npress*ng*nwave)
    ig = 0
    for iw in range(nwave):
        for ip in range(npress):
            for it in range(ntemp):
                # nwavenumber x ng x npress x ntemp
                k_w_g_p_t[iw,:,ip,it] = k_out[ig:ig+ng]
                ig = ig + ng
    f.close()
    return gas_id, iso_id, wave_grid, g_ord, del_g, P_grid, T_grid, k_w_g_p_t

def read_kls(filenames):
    """
    Read a list of k-tables from serveral Nemesis .kta files.

    Parameters
    ----------
    filenames : list
        A list of strings containing names of the kta files to be read.

    Returns
    -------
    gas_id_list : ndarray
        Gas identifier list.
    iso_id_list : ndarray
        Isotope identifier list.
    wave_grid : ndarray
        Wavenumbers/wavelengths grid of the k-table.
    g_ord : ndarray
        g-ordinates of the k-tables.
    del_g : ndarray
        Quadrature weights of the g-ordinates.
    P_grid : ndarray
        Pressure grid on which the k coeff's are pre-computed.
    T_grid : ndarray
        Temperature grid on which the k coeffs are pre-computed.
    k_gas_w_g_p_t : ndarray
        k coefficients.
        Has dimensaion: Ngas x Nwave x Ng x Npress x Ntemp.

    Notes
    -----
    Assume all k-tables computed on the same wavenumber/wavelength, pressure
    and temperature grid and with same g-ordinates and quadrature weights.
    """
    k_gas_w_g_p_t=[]
    gas_id_list = []
    iso_id_list = []
    for filename in filenames:
        gas_id, iso_id, wave_grid, g_ord, del_g, P_grid, T_grid,\
          k_g = read_kta(filename)
        gas_id_list.append(gas_id)
        iso_id_list.append(iso_id)
        k_gas_w_g_p_t.append(k_g)
    gas_id_list = np.array(gas_id_list)
    iso_id_list = np.array(iso_id_list)
    k_gas_w_g_p_t = np.array(k_gas_w_g_p_t)
    return gas_id_list, iso_id_list, wave_grid, g_ord, del_g, P_grid, T_grid,\
        k_gas_w_g_p_t

################################################################################
################################################################################
################################################################################
#!/usr/local/bin/python3
# -*- coding: utf-8 -*-
"""Mix ktables of different gases."""
import numpy as np
# from numba import jit
# from nemesispy.data.constants import C_LIGHT, K_B, PLANCK
"""
AU = 1.49598e11        # m astronomical unit
R_SUN = 6.95700e8      # m solar radius
R_JUP = 6.9911e7       # m Jupiter radius
R_JUP_E = 7.1492e7     # m nominal equatorial Jupiter radius
M_SUN = 1.989e30       # kg solar mass
M_JUP = 1.898e27       # kg Jupiter mass
C_LIGHT = 299792458    # ms-1 speed of light
G = 6.67430e-11        # m3 kg-1 s-2 universal gravitational constant
K_B = 1.38065e-23      # J K-1 Boltzmann constant
PLANCK = 6.62607e-34   # Js Planck's constant
N_A = 6.02214e23       # Avagadro's number
AMU = 1.66054e-27      # kg atomic mass unit
ATM = 101325           # Pa atmospheric pressure
"""
# @jit(nopython=True)
def interp_k(P_grid, T_grid, P_layer, T_layer, k_gas_w_g_p_t):
    """
    Adapted from chimera https://github.com/mrline/CHIMERA.
    Interpolates the k-tables to input atmospheric P & T for each wavenumber and
    g-ordinate for each gas with a standard bi-linear interpolation scheme.

    Parameters
    ----------
    P_grid : ndarray
        Pressure grid on which the k-coeffs are pre-computed.
    T_grid : ndarray
        Temperature grid on which the k-coeffs are pre-computed.
    P_layer : ndarray
        Atmospheric pressure grid.
    T_layer : ndarray
        Atmospheric temperature grid.
    k_gas_w_g_p_t : ndarray
        k-coefficient array, size = ngas x nwavenumber x ng x npress x ntemp

    Returns
    -------
    k_gas_w_g_l : ndarray
        The interpolated-to-atmosphere k-coefficients.
        Has dimension: Ngas x Nwavenumber x Ng x Nlayer.
    Notes
    -----
    Units: bar for pressure and K for temperature.
    Code breaks if P_layer/T_layer is out of the range of P_grid/T_grid.
    Mainly need to worry about max(T_layer)>max(T_grid).
    """
    Ngas, Nwave, Ng, Npress, Ntemp = k_gas_w_g_p_t.shape
    Nlayer = len(P_layer)
    k_gas_w_g_l = np.zeros((Ngas,Nwave,Ng,Nlayer))
    for ilayer in range(Nlayer): # loop through layers
        P = P_layer[ilayer]
        T = T_layer[ilayer]
        # Workaround when max atmosphere T is out of range of k table grid
        if T > T_grid[-1]:
            T = T_grid[-1]-1
        P_index_hi = np.where(P_grid >= P)[0][0]
        P_index_low = np.where(P_grid < P)[0][-1]
        T_index_hi = np.where(T_grid >= T)[0][0]
        T_index_low = np.where(T_grid < T)[0][-1]
        P_hi = P_grid[P_index_hi]
        P_low = P_grid[P_index_low]
        T_hi = T_grid[T_index_hi]
        T_low = T_grid[T_index_low]
        for igas in range(Ngas): # looping through gases
            for iwave in range(Nwave): # looping through wavenumber
                for ig in range(Ng): # looping through g-ord
                    arr = k_gas_w_g_p_t[igas,iwave,ig,:,:]
                    Q11 = arr[P_index_low,T_index_low]
                    Q12 = arr[P_index_hi,T_index_low]
                    Q22 = arr[P_index_hi,T_index_hi]
                    Q21 = arr[P_index_low,T_index_hi]
                    fxy1 = (T_hi-T)/(T_hi-T_low)*Q11+(T-T_low)/(T_hi-T_low)*Q21
                    fxy2 = (T_hi-T)/(T_hi-T_low)*Q12+(T-T_low)/(T_hi-T_low)*Q22
                    fxy = (P_hi-P)/(P_hi-P_low)*fxy1+(P-P_low)/(P_hi-P_low)*fxy2
                    k_gas_w_g_l[igas, iwave, ig, ilayer] = fxy
    return k_gas_w_g_l

def new_k_overlap_two_gas(k_gas1_g, k_gas2_g, q1, q2, del_g):
    """
    Combines the absorption coefficient distributions of two gases with overlapping
    opacities. The overlapping is assumed to be random and the k-distributions are
    assumed to have NG-1 mean values and NG-1 weights. Correspondingly there are
    NG ordinates in total.

    Parameters
    ----------
    k_gas1_g(ng) : ndarray
        k-coefficients for gas 1 at a particular wave bin and layer (temperature/pressure).
    k_gas2_g(ng) : ndarray
        k-coefficients for gas 2 at a particular wave bin and layer (temperature/pressure).
    q1 : real
        Volume mixing ratio of gas 1.
    q2 : real
        Volume mixing ratio of gas 2.
    del_g(ng) : ndarray
        Gauss quadrature weights for the g-ordinates, assumed same for both gases.

    Returns
    -------
    k_combined_g(ng) : ndarray
        Combined k-distribution of both gases at a particular wave bin and layer.
    q_combined : real
        Combined volume mixing ratio of both gases.
    """
    ng = len(del_g)  #Number of g-ordinates
    k_combined_g = np.zeros(ng)
    q_combined = q1 + q2

    if((k_gas1_g[ng-1]<=0.0) and (k_gas2_g[ng-1]<=0.0)):
        pass
    elif( (q1<=0.0) and (q2<=0.0) ):
        pass
    elif((k_gas1_g[ng-1]==0.0) or (q1==0.0)):
        k_combined_g[:] = k_gas2_g[:] * q2/(q1+q2)
    elif((k_gas2_g[ng-1]==0.0) or (q2==0.0)):
        k_combined_g[:] = k_gas1_g[:] * q1/(q1+q2)
    else:

        nloop = ng * ng
        weight = np.zeros(nloop)
        contri = np.zeros(nloop)
        ix = 0
        for i in range(ng):
            for j in range(ng):
                weight[ix] = del_g[i] * del_g[j]
                contri[ix] = (k_gas1_g[i]*q1 + k_gas2_g[j]*q2)/(q1+q2)
                ix = ix + 1

        #getting the cumulative g ordinate
        g_ord = np.zeros(ng+1)
        g_ord[0] = 0.0
        for ig in range(ng):
            g_ord[ig+1] = g_ord[ig] + del_g[ig]

        if g_ord[ng]<1.0:
            g_ord[ng] = 1.0

        #sorting contri array
        isort = np.argsort(contri)
        contrib1 = contri[isort]
        weight1 = weight[isort]

        #creating combined g-ordinate array
        gdist = np.zeros(nloop)
        gdist[0] = weight1[0]
        for i in range(nloop-1):
            ix = i + 1
            gdist[ix] = weight1[ix] + gdist[i]

        ig = 0
        sum1 = 0.0
        for i in range(nloop):

            if( (gdist[i]<g_ord[ig+1]) & (ig<=ng-1) ):
                k_combined_g[ig] = k_combined_g[ig] + contrib1[i] * weight1[i]
                sum1 = sum1 + weight1[i]
            else:
                frac = (g_ord[ig+1]-gdist[i-1])/(gdist[i]-gdist[i-1])
                k_combined_g[ig] = k_combined_g[ig] + frac * contrib1[i] * weight1[i]
                sum1 = sum1 + weight1[i]
                k_combined_g[ig] = k_combined_g[ig] / sum1
                ig = ig + 1
                if(ig<=ng-1):
                    sum1 = (1.-frac)*weight1[i]
                    k_combined_g[ig] = k_combined_g[ig] + (1.-frac) * contrib1[i] * weight1[i]

        if ig==ng-1:
            k_combined_g[ig] = k_combined_g[ig] / sum1

    return k_combined_g, q_combined

def new_k_overlap(k_gas_w_g_l,del_g,f):
    """
    Combines the absorption coefficient distributions of several gases with overlapping
    opacities. The overlaps are implicitly assumed to be random and the k-distributions
    are assumed to have NG-1 mean values and NG-1 weights. Correspondingly there
    are NG ordinates in total.

    Parameters
    ----------
    k_gas_w_g_l(ngas, nwave, ng, nlayer) : ndarray
        k-distributions of the different gases
    del_g(ng) : ndarray
        Gauss quadrature weights for the g-ordinates, assumed same for all gases.
    f(ngas,nlayer) : ndarray
        fraction of the different gases at each of the p-T points

    Returns
    -------
    k_w_g_l(nwave,ng,nlayer) : ndarray
        Opacity at each wavelength bin, each g ordinate and each layer.
    """
    ngas,nwave,ng,nlayer = k_gas_w_g_l.shape

    k_w_g_l = np.zeros((nwave,ng,nlayer))

    if ngas<=1:  #There are not enough gases to combine
        k_w_g_l[:,:,:] = k_gas_w_g_l[:,:,:,0]
    else:
        for ip in range(nlayer): #running for each p-T case
            for igas in range(ngas-1):
                #getting first and second gases to combine
                if igas==0:
                    k_gas1_w_g = np.zeros((nwave,ng))
                    k_gas2_w_g = np.zeros((nwave,ng))
                    k_gas1_w_g[:,:] = k_gas_w_g_l[igas,:,:,ip]
                    k_gas2_w_g[:,:] = k_gas_w_g_l[igas+1,:,:,ip]
                    f1 = f[igas,ip]
                    f2 = f[igas+1,ip]

                    k_combined = np.zeros((nwave,ng))
                    f_combined = np.zeros((ngas,nlayer))

                else:
                    #k_gas1_w_g = np.zeros((nwave,ng))
                    #k_gas2_w_g = np.zeros((nwave,ng))
                    k_gas1_w_g[:,:] = k_combined[:,:]
                    k_gas2_w_g[:,:] = k_gas_w_g_l[igas+1,:,:,ip]
                    f1 = f_combined
                    f2 = f[igas+1,ip]

                    k_combined = np.zeros((nwave,ng))

                for iwave in range(nwave):

                    k_g_combined, f_combined = new_k_overlap_two_gas(k_gas1_w_g[iwave,:], k_gas2_w_g[iwave,:], f1, f2, del_g)
                    k_combined[iwave,:] = k_g_combined[:]

            k_w_g_l[:,:,ip] = k_combined[:,:]

            return k_w_g_l


################################################################################
################################################################################
################################################################################
import numpy as np
# from numba import jit
# from nemesispy.radtran.k2interp import interp_k, new_k_overlap

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
            P_grid, T_grid, g_ord, del_g):
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
          NLAYER x NGAS
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
    k_gas_w_g_l = interp_k(P_grid, T_grid, P_layer, T_layer, k_gas_w_g_p_t) # NGAS,NWAVE,NG,NLAYER
    # Ngas, Nwave, Ng, Nlayer = k_gas_w_g_l.shape
    print('k_gas_w_g_l', k_gas_w_g_l)

    k_w_g_l = new_k_overlap(k_gas_w_g_l, del_g, VMR_layer.T) # NWAVE,NG,NLAYER

    utotl = U_layer * 1.0e-4 * 1.0e-20 # scaling

    print('utotl', utotl)
    print('k_w_g_l', k_w_g_l)
    TAUGAS = k_w_g_l * utotl # NWAVE, NG, NLAYER

    return TAUGAS

def radtran(wave_grid, U_layer, P_layer, T_layer, VMR_layer, k_gas_w_g_p_t,
            P_grid, T_grid, g_ord, del_g, ScalingFactor, RADIUS, solspec):
    """
    Calculate emission spectrum using the correlated-k method.

    Parameters
    ----------
    wave_grid : ndarray
        Wavelengths (um) grid for calculating spectra.
    U_layer : ndarray
        Total number of gas particles in each layer.
    P_layer : ndarray
        Atmospheric pressure grid.
    T_layer : ndarray
        Atmospheric temperature grid.
    VMR_layer : ndarray
        Array of volume mixing ratios for Ngas.
    k_gas_w_g_p_t : ndarray
        k-coefficients. Has dimension: Nwave x Ng x Npress x Ntemp.
    P_grid : ndarray
        Pressure grid on which the k-coeff's are pre-computed.
    T_grid : ndarray
        Temperature grid on which the k-coeffs are pre-computed.
    g_ord : ndarray
        g-ordinates of the k-table.
    del_g : ndarray
        Quadrature weights of the g-ordinates.
    ScalingFactor : ndarray ( NLAY ) ##
        Scale stuff to line of sight
    RADIUS : real ##
        Planetary radius
    solspec : ndarray ##
        Stellar spectra, used when the unit of the output is in fraction
        of stellar irradiance.

    Returns
    -------
    radiance : ndarray
        Output radiance (W cm-2 um-1 sr-1)
    """
    # Dimensioins
    NGAS, NWAVE, NG, NGRID = k_gas_w_g_p_t.shape[:-1]
    print('NGAS, NWAVE, NG, NGRID',NGAS, NWAVE, NG, NGRID)
    ### Second order opacities to be continued
    # Collision Induced Absorptioin Optical Path
    NLAY = len(P_layer)
    TAUCIA = np.zeros([NWAVE,NLAY])
    # Rayleigh Scattering Optical Path
    TAURAY = np.zeros([NWAVE,NLAY])
    # Dust Scattering Optical Path
    TAUDUST = np.zeros([NWAVE,NLAY])

    ### Gaseous Opacity
    # Calculating the k-coefficients for each gas in each layer

    """
    vmr_gas = np.zeros([NGAS, NLAY])

    utotl = np.zeros(NLAY)

    for i in range(NLAY):
        vmr_gas[i,:] = None　#Layer.PP[:,IGAS].T / Layer.PRESS #VMR of each radiatively active gas
        utotl[:] = None #utotl[:] + Layer.AMOUNT[:,IGAS].T * 1.0e-4 * 1.0e-20   #Vertical column density of the radiatively active gases

    k_gas_w_g_l = interp_k(P_grid, T_grid, P_layer, T_layer, k_gas_w_g_p_t)
    k_w_g_l = mix_multi_gas_k(k_gas_w_g_l, del_g, vmr_gas)
    TAUGAS = k_w_g_l * utotl
    """
    TAUGAS = tau_gas(k_gas_w_g_p_t, P_layer, T_layer, VMR_layer, U_layer,
            P_grid, T_grid, g_ord, del_g)
    print('TAUGAS', TAUGAS)
    TAUTOT = np.zeros(TAUGAS.shape) # NWAVE x NG x NLAYER
    print('TAUGAS.shape',TAUGAS.shape)
    # Merge all different opacities
    for ig in range(NG): # wavebin x layer / NWAVE x NG x NLAYER
        TAUTOT[:,ig,:] = TAUGAS[:,ig,:] + TAUCIA[:,:] + TAUDUST[:,:] + TAURAY[:,:]

    #Scale to the line-of-sight opacities
    TAUTOT_LAYINC = TAUTOT * ScalingFactor

    # Thermal Emission Calculation
    # IMOD = 3

    #Defining the units of the output spectrum / divide by stellar spectrum
    xfac = np.pi*4.*np.pi*((RADIUS)*1.0e2)**2.
    xfac = xfac / solspec

    #Calculating spectrum
    taud = np.zeros((NWAVE,NG))
    trold = np.zeros((NWAVE,NG))
    specg = np.zeros((NWAVE,NG))


    # taud[:,:] = taud[:,:] + TAUTOT_LAYINC[:,:,j]

    for ilayer in range(NLAY):

        taud[:,:] = TAUTOT[:,:,ilayer]

        tr = np.exp(-taud) # transmission function

        bb = planck(wave_grid, T_layer[ilayer]) # blackbody function

        for ig in range(NG):
            specg[:,ig] = specg[:,ig] + (trold[:,ig]-tr[:,ig])*bb[:] * xfac


    SPECOUT = np.tensordot(specg, del_g, axes=([1],[0])) * xfac

    return SPECOUT

################################################################################
################################################################################
################################################################################
# from models import Model2
# from path import get_profiles
# AU = 1.49598e11        # m astronomical unit
# R_SUN = 6.95700e8      # m solar radius
# R_JUP = 6.9911e7       # m Jupiter radius
# R_JUP_E = 7.1492e7     # m nominal equatorial Jupiter radius
# M_SUN = 1.989e30       # kg solar mass
# M_JUP = 1.898e27       # kg Jupiter mass
# C_LIGHT = 299792458    # ms-1 speed of light
# G = 6.67430e-11        # m3 kg-1 s-2 universal gravitational constant
# K_B = 1.38065e-23      # J K-1 Boltzmann constant
# PLANCK = 6.62607e-34   # Js Planck's constant
# N_A = 6.02214e23       # Avagadro's number
# AMU = 1.66054e-27      # kg atomic mass unit
# ATM = 101325           # Pa atmospheric pressure
# from nemesispy.data.constants import R_SUN, R_JUP_E, AMU, AU, M_JUP
# ### Required Inputs
# # Planet/star parameters
# T_star = 6000

# M_plt = 1*M_JUP
# SMA = 0.015*AU
# R_star = 1*R_SUN
# planet_radius = 1*R_JUP_E
# R_plt = 1*R_JUP_E

# """
# planet_radius : real
#     Reference planetary radius where H_atm=0.  Usually at surface for
#     terrestrial planets, or at 1 bar pressure level for gas giants.
# """
# H_atm = np.array([])
# """

# """
# P_atm = np.array([])
# NProfile = 40
# Nlayer = 15
# P_range = np.geomspace(20,1e-3,NProfile)*1e5
# mmw = 2*AMU

# ### params
# kappa = 1e-3
# gamma1 = 1e-1
# gamma2 = 1e-1
# alpha = 0.5
# T_irr = 1500

# atm = Model2(T_star, R_star, M_plt, R_plt, SMA, P_range, mmw,
#                       kappa, gamma1, gamma2, alpha, T_irr)
# H_atm = atm.height()
# P_atm = atm.pressure()
# T_atm = atm.temperature()
# # print('H_atm',H_atm)
# # print('P_atm',P_atm)
# """
# H_atm : ndarray
#     Input profile heights
# P_atm : ndarray
#     Input profile pressures
# T_atm : ndarray
#     Input profile temperatures
# """
# ID = np.array([1,2,5,6,40,39])
# ISO = np.array([0,0,0,0,0,0])
# """
# ID : ndarray
#     Gas identifiers.
# """
# NVMR = len(ID)
# VMR_atm = np.zeros((NProfile,NVMR))
# VMR_H2O = np.ones(NProfile)*1e-6
# VMR_CO2 = np.ones(NProfile)*1e-6
# VMR_CO = np.ones(NProfile)*1e-6
# VMR_CH4 = np.ones(NProfile)*1e-6
# VMR_He = (np.ones(NProfile)-VMR_H2O-VMR_CO2-VMR_CO-VMR_CH4)*0.15
# VMR_H2 = VMR_He/0.15*0.85
# VMR_atm[:,0] = VMR_H2O
# VMR_atm[:,1] = VMR_CO2
# VMR_atm[:,2] = VMR_CO
# VMR_atm[:,3] = VMR_CH4
# VMR_atm[:,4] = VMR_He
# VMR_atm[:,5] = VMR_H2


# """
# VMR_atm : ndarray
#     VMR_atm[i,j] is the Volume Mixing Ratio of gas j at profile point i.
#     The jth column corresponds to the gas with RADTRANS ID ID[j].
# """
# """
# H_base : ndarray
#     Heights of the layer bases.
# """

# lowres_files = ['/Users/jingxuanyang/Desktop/Workspace/nemesispy2022/nemesispy/data/ktables/h2o',
#          '/Users/jingxuanyang/Desktop/Workspace/nemesispy2022/nemesispy/data/ktables/co2',
#          '/Users/jingxuanyang/Desktop/Workspace/nemesispy2022/nemesispy/data/ktables/co',
#          '/Users/jingxuanyang/Desktop/Workspace/nemesispy2022/nemesispy/data/ktables/ch4']

# aeriel_files = ['/Users/jingxuanyang/Desktop/Workspace/nemesispy2022/nemesispy/data/ktables/H2O_Katy_ARIEL_test',
#           '/Users/jingxuanyang/Desktop/Workspace/nemesispy2022/nemesispy/data/ktables/CO2_Katy_ARIEL_test',
#           '/Users/jingxuanyang/Desktop/Workspace/nemesispy2022/nemesispy/data/ktables/CO_Katy_ARIEL_test',
#           '/Users/jingxuanyang/Desktop/Workspace/nemesispy2022/nemesispy/data/ktables/CH4_Katy_ARIEL_test']

# hires_files = ['/Users/jingxuanyang/Desktop/Workspace/nemesispy2022/nemesispy/data/ktables/H2O_Katy_R1000',
#           '/Users/jingxuanyang/Desktop/Workspace/nemesispy2022/nemesispy/data/ktables/CO2_Katy_R1000',
#           '/Users/jingxuanyang/Desktop/Workspace/nemesispy2022/nemesispy/data/ktables/CO_Katy_R1000',
#           '/Users/jingxuanyang/Desktop/Workspace/nemesispy2022/nemesispy/data/ktables/CH4_Katy_R1000']

# filenames = lowres_files
# """
# filenames : list
#     A list of strings containing names of the kta files to be read.
# """
# # P_layer = np.array([])
# # """
# # P_layer : ndarray
# #     Atmospheric pressure grid.
# # """
# # T_layer = np.array([])
# # """
# # T_layer : ndarray
# #     Atmospheric temperature grid.
# # """
# # U_layer = np.array([])
# # """
# # U_layer : ndarray
# #     Total number of gas particles in each layer.
# # """
# # f = np.array([[],[]])
# # VMR_layer = f.T
# # """
# # f(ngas,nlayer) : ndarray
# #     fraction of the different gases at each of the p-T points
# # """

# """
# wave_grid : ndarray
#     Wavelengths (um) grid for calculating spectra.
# """
# ### Calling sequence
# # Get averaged layer properties
# """
# H_layer,P_layer,T_layer,VMR_layer,U_layer,Gas_layer,scale,del_S\
#     = average(planet_radius, H_atm, P_atm, T_atm, VMR_atm, ID, H_base)
# """
# H_layer,P_layer,T_layer,VMR_layer,U_layer,Gas_layer,scale,del_S\
#     = get_profiles(planet_radius, H_atm, P_atm, VMR_atm, T_atm, ID, Nlayer,
#     H_base=None, path_angle=0.0, layer_type=1, bottom_height=0.0, interp_type=1, P_base=None,
#     integration_type=1, Nsimps=101)

# P_layer = P_layer*1e-5
# print('H_layer', H_layer)
# print('P_layer', P_layer)
# print('T_layer', T_layer)
# print('VMR_layer', VMR_layer)
# print('U_layer', U_layer)
# print('Gas_layer', Gas_layer)
# print('scale', scale)
# print('del_S', del_S)

# # Get raw k table infos from files
# gas_id_list, iso_id_list, wave_grid, g_ord, del_g, P_grid, T_grid,\
#         k_gas_w_g_p_t = read_kls(filenames)
# print('wave_grid', wave_grid)
# print('g_ord', g_ord)
# print('del_g', del_g)
# print('P_grid', P_grid)

# """
# # Interpolate k lists to layers
# k_gas_w_g_l = interp_k(P_grid, T_grid, P_layer, T_layer, k_gas_w_g_p_t)

# # Mix gas opacities
# k_w_g_l = new_k_overlap(k_gas_w_g_l,del_g,f)
# """
# StarSpectrum = np.ones(17) # NWAVE
# # Radiative Transfer
# SPECOUT = radtran(wave_grid, U_layer, P_layer, T_layer, VMR_layer, k_gas_w_g_p_t,
#             P_grid, T_grid, g_ord, del_g, ScalingFactor=scale,
#             RADIUS=planet_radius, solspec=StarSpectrum)

# wave_grid = np.array([1.1425, 1.1775, 1.2125, 1.2475, 1.2825, 1.3175, 1.3525, 1.3875, 1.4225,
# 1.4575, 1.4925, 1.5275, 1.5625, 1.5975, 1.6325, 3.6, 4.5])

# print(SPECOUT)

# import matplotlib.pyplot as plt

# plt.plot(wave_grid,-SPECOUT)
# plt.show()
# plt.close()

import numpy as np
from nemesispy.data.constants import R_SUN, R_JUP_E, AMU, AU, M_JUP
from nemesispy.radtran.models import Model2
from nemesispy.radtran.path import get_profiles # average
from nemesispy.radtran.k1read import read_kls
# from nemesispy.radtran.k2interp import interp_k, new_k_overlap
from nemesispy.radtran.k3radtran import radtran

### Required Inputs
# Planet/star parameters
T_star = 6000

M_plt = 1*M_JUP
SMA = 0.015*AU
R_star = 1*R_SUN
planet_radius = 1*R_JUP_E
R_plt = 1*R_JUP_E

"""
planet_radius : real
    Reference planetary radius where H_atm=0.  Usually at surface for
    terrestrial planets, or at 1 bar pressure level for gas giants.
"""
H_atm = np.array([])
"""

"""
P_atm = np.array([])
NProfile = 40
Nlayer = 15
P_range = np.geomspace(20,1e-3,NProfile)*1e5
mmw = 2*AMU

### params
kappa = 1e-3
gamma1 = 1e-1
gamma2 = 1e-1
alpha = 0.5
T_irr = 1500

atm = Model2(T_star, R_star, M_plt, R_plt, SMA, P_range, mmw,
                      kappa, gamma1, gamma2, alpha, T_irr)
H_atm = atm.height()
P_atm = atm.pressure()
T_atm = atm.temperature()
# print('H_atm',H_atm)
# print('P_atm',P_atm)
"""
H_atm : ndarray
    Input profile heights
P_atm : ndarray
    Input profile pressures
T_atm : ndarray
    Input profile temperatures
"""
ID = np.array([1,2,5,6,40,39])
ISO = np.array([0,0,0,0,0,0])
"""
ID : ndarray
    Gas identifiers.
"""
NVMR = len(ID)
VMR_atm = np.zeros((NProfile,NVMR))
VMR_H2O = np.ones(NProfile)*1e-6
VMR_CO2 = np.ones(NProfile)*1e-6
VMR_CO = np.ones(NProfile)*1e-6
VMR_CH4 = np.ones(NProfile)*1e-6
VMR_He = (np.ones(NProfile)-VMR_H2O-VMR_CO2-VMR_CO-VMR_CH4)*0.15
VMR_H2 = VMR_He/0.15*0.85
VMR_atm[:,0] = VMR_H2O
VMR_atm[:,1] = VMR_CO2
VMR_atm[:,2] = VMR_CO
VMR_atm[:,3] = VMR_CH4
VMR_atm[:,4] = VMR_He
VMR_atm[:,5] = VMR_H2


"""
VMR_atm : ndarray
    VMR_atm[i,j] is the Volume Mixing Ratio of gas j at profile point i.
    The jth column corresponds to the gas with RADTRANS ID ID[j].
"""
"""
H_base : ndarray
    Heights of the layer bases.
"""

lowres_files = ['/Users/jingxuanyang/Desktop/Workspace/nemesispy2022/nemesispy/data/ktables/h2o',
         '/Users/jingxuanyang/Desktop/Workspace/nemesispy2022/nemesispy/data/ktables/co2',
         '/Users/jingxuanyang/Desktop/Workspace/nemesispy2022/nemesispy/data/ktables/co',
         '/Users/jingxuanyang/Desktop/Workspace/nemesispy2022/nemesispy/data/ktables/ch4']

aeriel_files = ['/Users/jingxuanyang/Desktop/Workspace/nemesispy2022/nemesispy/data/ktables/H2O_Katy_ARIEL_test',
          '/Users/jingxuanyang/Desktop/Workspace/nemesispy2022/nemesispy/data/ktables/CO2_Katy_ARIEL_test',
          '/Users/jingxuanyang/Desktop/Workspace/nemesispy2022/nemesispy/data/ktables/CO_Katy_ARIEL_test',
          '/Users/jingxuanyang/Desktop/Workspace/nemesispy2022/nemesispy/data/ktables/CH4_Katy_ARIEL_test']

hires_files = ['/Users/jingxuanyang/Desktop/Workspace/nemesispy2022/nemesispy/data/ktables/H2O_Katy_R1000',
          '/Users/jingxuanyang/Desktop/Workspace/nemesispy2022/nemesispy/data/ktables/CO2_Katy_R1000',
          '/Users/jingxuanyang/Desktop/Workspace/nemesispy2022/nemesispy/data/ktables/CO_Katy_R1000',
          '/Users/jingxuanyang/Desktop/Workspace/nemesispy2022/nemesispy/data/ktables/CH4_Katy_R1000']

filenames = lowres_files
"""
filenames : list
    A list of strings containing names of the kta files to be read.
"""
# P_layer = np.array([])
# """
# P_layer : ndarray
#     Atmospheric pressure grid.
# """
# T_layer = np.array([])
# """
# T_layer : ndarray
#     Atmospheric temperature grid.
# """
# U_layer = np.array([])
# """
# U_layer : ndarray
#     Total number of gas particles in each layer.
# """
# f = np.array([[],[]])
# VMR_layer = f.T
# """
# f(ngas,nlayer) : ndarray
#     fraction of the different gases at each of the p-T points
# """

"""
wave_grid : ndarray
    Wavelengths (um) grid for calculating spectra.
"""
### Calling sequence
# Get averaged layer properties
"""
H_layer,P_layer,T_layer,VMR_layer,U_layer,Gas_layer,scale,del_S\
    = average(planet_radius, H_atm, P_atm, T_atm, VMR_atm, ID, H_base)
"""
H_layer,P_layer,T_layer,VMR_layer,U_layer,Gas_layer,scale,del_S\
    = get_profiles(planet_radius, H_atm, P_atm, VMR_atm, T_atm, ID, Nlayer,
    H_base=None, path_angle=0.0, layer_type=1, bottom_height=0.0, interp_type=1, P_base=None,
    integration_type=1, Nsimps=101)

P_layer = P_layer*1e-5
print('H_layer', H_layer)
print('P_layer', P_layer)
print('T_layer', T_layer)
print('VMR_layer', VMR_layer)
print('U_layer', U_layer)
print('Gas_layer', Gas_layer)
print('scale', scale)
print('del_S', del_S)

# Get raw k table infos from files
gas_id_list, iso_id_list, wave_grid, g_ord, del_g, P_grid, T_grid,\
        k_gas_w_g_p_t = read_kls(filenames)
print('wave_grid', wave_grid)
print('g_ord', g_ord)
print('del_g', del_g)
print('P_grid', P_grid)

"""
# Interpolate k lists to layers
k_gas_w_g_l = interp_k(P_grid, T_grid, P_layer, T_layer, k_gas_w_g_p_t)

# Mix gas opacities
k_w_g_l = new_k_overlap(k_gas_w_g_l,del_g,f)
"""
StarSpectrum = np.ones(17) # NWAVE
# Radiative Transfer
SPECOUT = radtran(wave_grid, U_layer, P_layer, T_layer, VMR_layer, k_gas_w_g_p_t,
            P_grid, T_grid, g_ord, del_g, ScalingFactor=scale,
            RADIUS=planet_radius, solspec=StarSpectrum)

wave_grid = np.array([1.1425, 1.1775, 1.2125, 1.2475, 1.2825, 1.3175, 1.3525, 1.3875, 1.4225,
1.4575, 1.4925, 1.5275, 1.5625, 1.5975, 1.6325, 3.6, 4.5])

print(SPECOUT)

import matplotlib.pyplot as plt

plt.plot(wave_grid,-SPECOUT)
plt.show()
plt.close()