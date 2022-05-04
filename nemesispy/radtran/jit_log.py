#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 28 15:31:19 2022

@author: jingxuanyang
"""

from numba import jit 
import numpy as np

@jit(nopython=True)
def get_log(x):
    y = np.log(x)
    return y

@jit(nopython=True)
def interp_k(P_grid, T_grid, P_layer, T_layer, k_gas_w_g_p_t, wavecalc=None):
    """
    Follows normal sequence
    Calculate the k coeffcients of gases at given presures and temperatures
    using pre-tabulated k-tables.

    Parameters
    ----------
    P_grid(NPRESSKTA) : ndarray
        Pressure grid on which the k-coefficients are pre-computed.
        Unit: Pa
    T_grid(NTEMPKTA) : ndarray
        Temperature grid on which the k-coefficients are pre-computed.
        Unit: Kelvin
    P_layer(NLAYER) : ndarray
        Atmospheric pressure grid.
        Unit: Pa
    T_layer(NLAYER) : ndarray
        Atmospheric temperature grid.
        Unit: Kelvin
    k_gas_w_g_p_t(NGAS,NWAVEKTA,NG,NPRESSKTA,NTEMPKTA) : ndarray
        Array storing the k-coefficients.

    Returns
    -------
    k_gas_w_g_l(NGAS,NWAVE,NG,NLAYER) : ndarray
        The interpolated-to-atmosphere k-coefficients.
        Has dimension: NGAS x NWAVE x NG x NLAYER.
    Notes
    -----
    Units: bar for pressure and K for temperature.
    Code breaks if P_layer/T_layer is out of the range of P_grid/T_grid.
    Mainly need to worry about max(T_layer)>max(T_grid).
    No extrapolation outside of the TP grid of ktable.
    """
    NGAS, NWAVE, NG, NPRESS, NTEMP = k_gas_w_g_p_t.shape
    NLAYER = len(P_layer)
    # print('P_layer',P_layer)
    k_gas_w_g_l = np.zeros((NGAS,NWAVE,NG,NLAYER))

    # kgood (NGAS, NWAVE, NG, NLAYER)
    kgood = np.zeros((NGAS,NWAVE,NG,NLAYER))
    for ilayer in range(NLAYER):
        press1 = P_layer[ilayer]
        temp1 = T_layer[ilayer]

        # find the Pressure levels just above and below that of current layer
        lpress = np.log(press1)
        # press0, ip = find_nearest(P_grid,press1)
        ip = abs(P_grid-press1).argmin()
        press0 = P_grid[ip]

        if P_grid[ip] >= press1:
            iphi = ip
            if ip == 0:
                lpress = np.log(P_grid[0])
                ipl = 0
                iphi = 1
            else:
                ipl = ip - 1
        elif P_grid[ip]<press1:
            ipl = ip
            if ip == NPRESS -1:
                lpress = np.log(P_grid[NPRESS-1])
                iphi = NPRESS - 1
                ipl = NPRESS - 2
            else:
                iphi = ip + 1

        # find the Temperature levels just above and below that of current layer
        # temp0, it = find_nearest(T_grid, temp1)
        it = abs(T_grid-temp1).argmin()
        temp0 = T_grid[it]

        if T_grid[it]>=temp1:
            ithi = it
            if it == 0:
                temp1 = T_grid[0]
                itl = 0
                ithi = 1
            else:
                itl = it -1
        elif T_grid[it]<temp1:
            itl = it
            if it == NTEMP-1:
                temp1 = T_grid[-1]
                ithi = NTEMP - 1
                itl = NTEMP -2
            else:
                ithi = it + 1

        # interpolation
        plo = np.log(P_grid[ipl])
        phi = np.log(P_grid[iphi])
        tlo = T_grid[itl]
        thi = T_grid[ithi]

        klo1 = np.zeros((NGAS,NWAVE,NG))
        klo2 = np.zeros((NGAS,NWAVE,NG))
        khi1 = np.zeros((NGAS,NWAVE,NG))
        khi2 = np.zeros((NGAS,NWAVE,NG))

        # klo1 = np.zeros([self.NWAVE,self.NG,self.NGAS])

        klo1[:] = k_gas_w_g_p_t[:,:,:,ipl,itl]
        klo2[:] = k_gas_w_g_p_t[:,:,:,ipl,ithi]
        khi2[:] = k_gas_w_g_p_t[:,:,:,iphi,ithi]
        khi1[:] = k_gas_w_g_p_t[:,:,:,iphi,itl]

        # bilinear interpolation
        v = (lpress-plo)/(phi-plo)
        u = (temp1-tlo)/(thi-tlo)
        dudt = 1./(thi-tlo)

        igood = np.where( (klo1>0.0) & (klo2>0.0) & (khi1>0.0) & (khi2>0.0) )
        # NGAS x NWAVE x NG
        # # print('kgood',kgood)
        # print('NGAS',NGAS)
        # print('igood[0]',igood[0])
        # print('igood[1]',igood[1])
        # print('igood[2]',igood[2])
        # print('ilayer',ilayer)
        # kgood = np.zeros([self.NWAVE,self.NG,npoints,self.NGAS]) juan
        # k_gas_w_g_l = np.zeros([NGAS,NWAVE,NG,NLAYER]) mine

        kgood[igood[0],igood[1],igood[2],ilayer] \
            = (1.0-v)*(1.0-u)*np.log(klo1[igood[0],igood[1],igood[2]]) \
            + v*(1.0-u)*np.log(khi1[igood[0],igood[1],igood[2]]) \
            + v*u*np.log(khi2[igood[0],igood[1],igood[2]]) \
            + (1.0-v)*u*np.log(klo2[igood[0],igood[1],igood[2]])

        kgood[igood[0],igood[1],igood[2],ilayer] \
            = np.exp(kgood[igood[0],igood[1],igood[2],ilayer])

        ibad = np.where( (klo1<=0.0) & (klo2<=0.0) & (khi1<=0.0) & (khi2<=0.0) )
        kgood[ibad[0],ibad[1],ibad[2],ilayer] \
            = (1.0-v)*(1.0-u)*klo1[ ibad[0],ibad[1],ibad[2]] \
            + v*(1.0-u)*khi1[ ibad[0],ibad[1],ibad[2]] \
            + v*u*khi2[ ibad[0],ibad[1],ibad[2]] \
            + (1.0-v)*u*klo2[ ibad[0],ibad[1],ibad[2]]

    k_gas_w_g_l = kgood
    return k_gas_w_g_l