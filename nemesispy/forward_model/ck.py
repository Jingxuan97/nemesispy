#!/usr/local/bin/python3
# -*- coding: utf-8 -*-
"""Routines to read pre-tabulated opacity tables (k-tables) and calculate
overlapping opacities. All k-tables in a single calculation are assumed to
share the same wavelength-grid, pressure-grid, temperature-grid, g-ordinates
and quadrature weights. Furthermore, it is assumed that the k-tables have
the wavelength range desired for the final calculation result.
"""

import numpy as np
from numba import jit
from constants import C_LIGHT, K_B, PLANCK

def read_kta(filename):
    """
    Reads a pre-tabulated k-table from a Nemesis .kta file.

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
    nbytes_int32 = 4
    nbytes_float32 = 4
    ioff = 0
    if filename[-3:] == 'kta':
        f = open(filename,'rb')
    else:
        f = open(filename+'.kta','rb')
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

@jit(nopython=True)
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

@jit(nopython=True)
def mix_two_gas_k(k_g1, k_g2, VMR1, VMR2, g_ord, del_g):
    """
    Adapted from chimera https://github.com/mrline/CHIMERA.

    Mix the k-coefficients for two individual gases using the randomly
    overlapping absorption line approximation. The "resort-rebin" procedure
    is described in e.g. Goody et al. 1989, Lacis & Oinas 1991, Molliere et al.
    2015 and Amundsen et al. 2017. Each pair of gases can be treated as a
    new "hybrid" gas that can then be mixed again with another
    gas.  This is all for a *single* wavenumber bin for a single pair of gases
    at a particular pressure and temperature.

    Parameters
    ----------
    k_g1 : ndarray
        k-coeffs for gas 1 at a particular wave bin and temperature/pressure.
        Has dimension Ng.
    k_g2 : ndarray
        k-coeffs for gas 2
    VMR1 : ndarray
        Volume mixing ratio of gas 1
    VMR2 : ndarray
        Volume mixing ratio for gas 2
    g_ord : ndarray
        g-ordinates, assumed same for both gases.
    del_g : ndarray
        Gauss quadrature weights for the g-ordinates, assumed same for both gases.

    Returns
    -------
    k_g_combined
        Mixed k-coefficients for the given pair of gases
    VMR_combined
        Volume mixing ratio of "mixed gas".
    """
    VMR_combined = VMR1+VMR2
    Ng = len(g_ord)
    k_g_combined = np.zeros(Ng)
    cut_off = 1e-30
    if k_g1[-1] * VMR1 < cut_off and k_g2[-1] * VMR2 < cut_off:
        pass
    elif k_g1[-1] * VMR1 < cut_off:
        k_g_combined = k_g2*VMR2/VMR_combined
    elif k_g2[-1] * VMR2 < cut_off:
        k_g_combined = k_g1*VMR1/VMR_combined
    else:
        # Overlap Ng k-coeffs with Ng k-coeffs: Ng x Ng possible pairs
        k_g_mix = np.zeros(Ng**2)
        weight_mix = np.zeros(Ng**2)
        # Mix k-coeffs of gases weighted by their relative VMR.
        for i in range(Ng):
            for j in range(Ng):
                #equation 9 Amundsen 2017 (equation 20 Mollier 2015)
                k_g_mix[i*Ng+j] = (k_g1[i]*VMR1+k_g2[j]*VMR2)/VMR_combined
                 #equation 10 Amundsen 2017
                weight_mix[i*Ng+j] = del_g[i]*del_g[j]

        # Resort-rebin procedure: Sort new "mixed" k-coeff's from low to high
        # see Amundsen et al. 2016 or section B.2.1 in Molliere et al. 2015
        ascending_index = np.argsort(k_g_mix)
        k_g_mix_sorted = k_g_mix[ascending_index]
        weight_mix_sorted = weight_mix[ascending_index]
        #combining w/weights--see description on Molliere et al. 2015
        sum_weight = np.cumsum(weight_mix_sorted)
        x = sum_weight/np.max(sum_weight)*2.-1
        for i in range(Ng):
            loc = np.where(x >=  g_ord[i])[0][0]
            k_g_combined[i] = k_g_mix_sorted[loc]

    return k_g_combined, VMR_combined

@jit(nopython=True)
def mix_multi_gas_k(k_gas_g, VMR, g_ord, del_g):
    """
      Adapted from chimera https://github.com/mrline/CHIMERA.

      Key function that properly mixes the k-coefficients
      for multiple gases by treating a pair of gases at a time.
      Each pair becomes a "hybrid" gas that can be mixed in a pair
      with another gas, succesively. This is performed at a given
      wavenumber and atmospheric layer.

      Parameters
      ----------
      k_gas_g : ndarray
          array of k-coeffs for each gas at a given wavenumber and pressure level.
          Has dimension: Ngas x Ng.
      VMR : ndarray
          array of volume mixing ratios for Ngas.
      g_ord : ndarray
          g-ordinates, assumed same for all gases.
      del_g : ndarray
          Gauss quadrature weights for the g-ordinates, assumed same for all gases.

      Returns
      -------
      k_g_combined : ndarray
          mixed k_gas_g coefficients for the given gases.
      VMR_combined : ndarray
          Volume mixing ratio of "mixed gas".
    """
    ngas = k_gas_g.shape[0]
    k_g_combined,VMR_combined = k_gas_g[0,:],VMR[0]
    #mixing in rest of gases inside a loop
    for j in range(1,ngas):
        k_g_combined,VMR_combined\
            = mix_two_gas_k(k_g_combined,k_gas_g[j,:],VMR_combined,VMR[j],g_ord,del_g)
    return k_g_combined, VMR_combined

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

      Notes
      -----
      PLANCK = 6.62607e-34 Js
      C_LIGHT = 299792458 ms-1
      K_B = 1.38065e-23 J K-1
    """
    h = PLANCK
    c = C_LIGHT
    k = K_B
    radiance = (2*h*c**2)/((wl*1e-6)**5)*(1/(np.exp((h*c)/((wl*1e-6)*k*T))-1))*1e-10
    return radiance

@jit(nopython=True)
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
    k_gas_w_g_l = interp_k(P_grid, T_grid, P_layer, T_layer, k_gas_w_g_p_t)
    Ngas, Nwave, Ng, Nlayer = k_gas_w_g_l.shape
    tau_w_g_l = np.zeros((Nwave,Ng,Nlayer))
    for iwave in range (Nwave):
        k_gas_g_l = k_gas_w_g_l[:,iwave,:,:]
        k_g_l = np.zeros((Ng,Nlayer))
        for ilayer in range(Nlayer):
            k_g_l[:,ilayer], VMR\
                = mix_multi_gas_k(k_gas_g_l[:,:,ilayer],VMR_layer[ilayer,:],g_ord,del_g)
            tau_w_g_l[iwave,:,ilayer] = k_g_l[:,ilayer]*U_layer[ilayer]*VMR
    return tau_w_g_l

@jit(nopython=True)
def tau_rayleigh(wave, U_layer, VMR):

    AH2 = 13.58E-5
    BH2 = 7.52E-3
    AHe = 3.48E-5
    BHe = 2.30E-3
    fH2 = 0.864
    k = 1.37971e-23
    P0 = 1.013e5
    T0 = 273.15

    lamb = wave*1e-6
    x = 1.0/wave

    nH2 = AH2*(1.0+BH2*x*x)
    nHe = AHe*(1.0+BHe*x*x)
    nAir = fH2*nH2 + (1-fH2)*nHe

    delta = 0.0
    faniso = (6.0+3.0*delta)/(6.0 - 7.0*delta)

    temp = 32*(np.pi**3)*nAir**2
    N0 = P0/(k*T0)

    x = N0*lamb*lamb

    rayleighj = temp*1e4*faniso/(3*(x**2))
    # in cm2
    rayleighj = rayleighj*1e20*VMR*U_layer
    return rayleighj

from cia import tau_cia

#@jit(nopython=True)
def radiance(wave, U_layer, P_layer, T_layer, VMR_layer, k_gas_w_g_p_t,
            P_grid, T_grid, g_ord, del_g,
            path_length, kcia_pair_l_w):
    """
      Calculate emission spectrum using the correlated-k method.

      Parameters
      ----------
      wave : ndarray
          Wavelengths (um).
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

      Returns
      -------
      radiance : ndarray
          Output radiance (W cm-2 um-1 sr-1)
    """
    tau_w_g_l = tau_gas(k_gas_w_g_p_t, P_layer, T_layer, VMR_layer, U_layer,
                            P_grid, T_grid, g_ord, del_g)
    Ng = len(g_ord)
    output = np.zeros(len(wave))

    # CIA
    # tau_cia_w_l =
    tau_cia_w_l \
        = tau_cia(kcia_pair_l_w,U_layer,path_length,VMR_layer[0][-1],VMR_layer[0][-2])



    for iwave,wl in enumerate(wave):
        tau_g_l = tau_w_g_l[iwave,:,:]
        bb = blackbody_um(wl,T_layer)
        # top layer first, bottom layer last
        bb = bb[::-1]
        radiance_g = np.zeros(Ng)
        for ig in range (Ng):
            tau_l = tau_g_l[ig,:]

            # CIA
            # tau_l += tau_cia_w_l[iwave,:]


            # top layer first, bottom layer last
            tau_l = tau_l[::-1]

            """
            # Rayleigh, cause jit to break
            tau_ray_l = tau_rayleigh(wl,U_layer,sum(VMR_layer[0][-2:]))
            #print(tau_ray_l)
            tau_l += tau_ray_l
            """



            # transmission
            tr = np.exp(-np.cumsum(tau_l))
            tr = np.concatenate((np.array([1]),tr))
            del_tr = tr[:-1] - tr[1:]
            radiance_g[ig] = np.sum(bb*del_tr) + bb[-1]*tr[-1]
            #print(del_tr)
        output[iwave] = np.sum(radiance_g*del_g)

    return output
