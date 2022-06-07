#!/usr/local/bin/python3
# -*- coding: utf-8 -*-
"""Routines to read pre-tabulated correlated-k look-up tables (k-tables).
All k-tables are assumed to share the same wavelength grid, pressure grid,
temperature grid, g ordinates and quadrature weights.

Note that the opacities in Nemesis k-tables are multiplied by a factor of 1e20,
which need to be rescaled in the radiative transfer routine by multiplying
absorber amounts by 1e-20.
"""
import numpy as np

def read_kta(filepath):
    # this function is hopefully called once in a retrieval so no need to
    # optimise time
    """
    Reads a pre-tabulated correlated-k look-up table from a Nemesis .kta file.

    Parameters
    ----------
    filepath : str
        The filepath to the Nemesis .kta file to be read.

    Returns
    -------
    gas_id : int
        Gas identifier.
    iso_id : int
        Isotopologue identifier.
    wave_grid(NWAVEKTA) : ndarray
        Wavenumber/wavelength grid of the k-table.
    g_ord(NG) : ndarray
        g-ordinates of the k-table.
    del_g(NG) : ndarray
        Quadrature weights of the g-ordinates.
    P_grid(NPRESSKTA) : ndarray
        Pressure grid on which the k-coeff's are pre-computed.
        Unit: atm
    T_grid(NTEMPKTA) : ndarray
        Temperature grid on which the k-coeffs are pre-computed.
        Unit: kelvin
    k_w_g_p_t(NWAVEKTA,NG,NPRESSKTA,NTEMPKTA) : ndarray
        Array storing the k-coefficients.
    """
    # Open file for reading (r) in binary (b) mode; will return byte strings
    if filepath[-3:] == 'kta':
        f = open(filepath,'rb')
    else:
        f = open(filepath+'.kta','rb')

    # Define bytes consumed by elements of table
    nbytes_int32 = 4
    nbytes_float32 = 4
    ioff = 0

    # Read headers
    irec0 = int(np.fromfile(f,dtype='int32',count=1)) # where ktable data starts
    NWAVEKTA = int(np.fromfile(f,dtype='int32',count=1))
    vmin = float(np.fromfile(f,dtype='float32',count=1))
    delv = float(np.fromfile(f,dtype='float32',count=1))
    fwhm = float(np.fromfile(f,dtype='float32',count=1))
    NPRESSKTA = int(np.fromfile(f,dtype='int32',count=1))
    NTEMPKTA = int(np.fromfile(f,dtype='int32',count=1))
    NG = int(np.fromfile(f,dtype='int32',count=1))
    gas_id = int(np.fromfile(f,dtype='int32',count=1))
    iso_id = int(np.fromfile(f,dtype='int32',count=1))
    ioff = ioff + 10*nbytes_int32
    print('irec0',irec0)
    print('NWAVEKTA',NWAVEKTA)
    print('vmin',vmin)
    print('delv',delv)
    print('fwhm',fwhm)
    print('NPRESSKTA',NPRESSKTA)
    print('NTEMPKTA',NTEMPKTA)
    print('NG',NG)
    print('gas_id',gas_id)
    print('iso_id',iso_id)

    # Read g-ordinates and quadrature weights
    g_ord = np.fromfile(f,dtype='float32',count=NG)
    del_g = np.fromfile(f,dtype='float32',count=NG)
    ioff = ioff + 2*NG*nbytes_float32
    dummy1 = np.fromfile(f,dtype='float32',count=1)
    dummy2 = np.fromfile(f,dtype='float32',count=1)
    ioff = ioff + 2*nbytes_float32
    print('g_ord',g_ord,len(g_ord))
    print('del_g',del_g,len(del_g))
    print('dummy1',dummy1)
    print('dummy2',dummy2)


    # Read temperature/pressure grid
    P_grid = np.fromfile(f,dtype='float32',count=NPRESSKTA)
    T_grid = np.fromfile(f,dtype='float32',count=NTEMPKTA)
    ioff = ioff + NPRESSKTA*nbytes_float32+NTEMPKTA*nbytes_float32

    # Read wavenumber/wavelength grid
    if delv>0.0:  # uniform grid
        vmax = delv*NWAVEKTA + vmin
        wave_grid = np.linspace(vmin,vmax,NWAVEKTA)
    else:   # non-uniform grid
        wave_grid = np.zeros([NWAVEKTA])
        wave_grid = np.fromfile(f,dtype='float32',count=NWAVEKTA)
        ioff = ioff + NWAVEKTA*nbytes_float32
    NWAVEKTA = len(wave_grid)
    print('wave_grid',wave_grid)

    # print('ioff',ioff)
    # print('irec0',irec0)
    # for i in range(NWAVEKTA*NPRESSKTA*NTEMPKTA*NG+irec0-1-ioff+100):
    #     data = np.fromfile(f,dtype='float32',count=1)
    #     print(data)

    # Jump to the minimum wavenumber
    ioff = (irec0-1)*nbytes_float32 # Python index starts at 0
    f.seek(ioff,0)

    # Write the k-coefficients into array form
    k_w_g_p_t = np.zeros([NWAVEKTA,NG,NPRESSKTA,NTEMPKTA])
    k_list = np.fromfile(f,dtype='float32',count=NTEMPKTA*NPRESSKTA*NG*NWAVEKTA)
    ig = 0
    for iwave in range(NWAVEKTA):
        for ipress in range(NPRESSKTA):
            for itemp in range(NTEMPKTA):
                k_w_g_p_t[iwave,:,ipress,itemp] = k_list[ig:ig+NG]
                ig = ig + NG
    f.close()
    return gas_id, iso_id, wave_grid, g_ord, del_g, P_grid, T_grid, k_w_g_p_t


def read_kls(filepaths):
    """
    Read a list of k-tables from serveral Nemesis .kta files.

    Parameters
    ----------
    filepaths : list
        A list of strings containing filepaths to the .kta files to be read.

    Returns
    -------
    gas_id_list(NGAS) : ndarray
        Gas identifier list.
    iso_id_list(NGAS) : ndarray
        Isotopologue identifier list.
    wave_grid(NWAVEKTA) : ndarray
        Wavenumbers/wavelengths grid of the k-table.
    g_ord(NG) : ndarray
        g-ordinates of the k-tables.
    del_g(NG) : ndarray
        Quadrature weights of the g-ordinates.
    P_grid(NPRESSKTA) : ndarray
        Pressure grid on which the k-coeff's are pre-computed.
        Unit: atm
    T_grid(NTEMPKTA) : ndarray
        Temperature grid on which the k-coeffs are pre-computed.
        Unit: kelvin
    k_w_g_p_t(NWAVEKTA,NG,NPRESSKTA,NTEMPKTA) : ndarray
        Array storing the k-coefficients.

    Notes
    -----
    Assume the k-tables in all the kta files are computed on the same
    wavenumber/wavelength, pressure and temperature grid and with same g-ordinates
    and quadrature weights.
    """
    k_gas_w_g_p_t=[]
    gas_id_list = []
    iso_id_list = []
    for filepath in filepaths:
        gas_id, iso_id, wave_grid, g_ord, del_g, P_grid, T_grid,\
          k_g = read_kta(filepath)
        gas_id_list.append(gas_id)
        iso_id_list.append(iso_id)
        k_gas_w_g_p_t.append(k_g)
    gas_id_list = np.array(gas_id_list)
    iso_id_list = np.array(iso_id_list)
    k_gas_w_g_p_t = np.array(k_gas_w_g_p_t)
    return gas_id_list, iso_id_list, wave_grid, g_ord, del_g, P_grid, T_grid,\
        k_gas_w_g_p_t





"""Test File Paths
'/Users/jingxuanyang/Desktop/Workspace/nemesispy2022/nemesispy/data/ktables/h2o'
'/Users/jingxuanyang/Desktop/Workspace/nemesispy2022/nemesispy/data/ktables/co2'
'/Users/jingxuanyang/Desktop/Workspace/nemesispy2022/nemesispy/data/ktables/co'
'/Users/jingxuanyang/Desktop/Workspace/nemesispy2022/nemesispy/data/ktables/ch4'

kfiles = ['/Users/jingxuanyang/Desktop/Workspace/nemesispy2022/nemesispy/data/ktables/h2o',
'/Users/jingxuanyang/Desktop/Workspace/nemesispy2022/nemesispy/data/ktables/co2',
'/Users/jingxuanyang/Desktop/Workspace/nemesispy2022/nemesispy/data/ktables/co',
'/Users/jingxuanyang/Desktop/Workspace/nemesispy2022/nemesispy/data/ktables/ch4']
"""
gas_id, iso_id, wave_grid, g_ord, del_g, P_grid, T_grid, k_w_g_p_t\
    = read_kta('/Users/jingxuanyang/Desktop/Workspace/nemesispy2022/nemesispy/data/ktables/h2o')

kfiles = ['/Users/jingxuanyang/Desktop/Workspace/nemesispy2022/nemesispy/data/ktables/h2o',
'/Users/jingxuanyang/Desktop/Workspace/nemesispy2022/nemesispy/data/ktables/co2',
'/Users/jingxuanyang/Desktop/Workspace/nemesispy2022/nemesispy/data/ktables/co',
'/Users/jingxuanyang/Desktop/Workspace/nemesispy2022/nemesispy/data/ktables/ch4']
gas_id_list, iso_id_list, wave_grid, g_ord, del_g, P_grid, T_grid, k_gas_w_g_p_t\
    = read_kls(kfiles)
"""
print('CO2\n')
gas_id, iso_id, wave_grid, g_ord, del_g, P_grid, T_grid, k_w_g_p_t\
    = read_kta('/Users/jingxuanyang/Desktop/Workspace/nemesispy2022/nemesispy/data/ktables/co2')
print('CO\n')
gas_id, iso_id, wave_grid, g_ord, del_g, P_grid, T_grid, k_w_g_p_t\
    = read_kta('/Users/jingxuanyang/Desktop/Workspace/nemesispy2022/nemesispy/data/ktables/co')
print('CH4\n')
gas_id, iso_id, wave_grid, g_ord, del_g, P_grid, T_grid, k_w_g_p_t\
    = read_kta('/Users/jingxuanyang/Desktop/Workspace/nemesispy2022/nemesispy/data/ktables/ch4')
"""