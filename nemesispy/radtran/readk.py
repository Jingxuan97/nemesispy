#!/usr/local/bin/python3
# -*- coding: utf-8 -*-
"""Routines to read pre-tabulated correlated-k look-up tables (k-tables).
All k-tables are assumed to share the same wavelength-grid, pressure-grid,
temperature-grid, g-ordinates and quadrature weights.
"""
import numpy as np

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