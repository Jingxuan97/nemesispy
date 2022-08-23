#!/usr/local/bin/python3
# -*- coding: utf-8 -*-
"""
Calculate the optical path due to atomic and molecular transitions.
The opacity of a mixture of gases is calculated by the correlated-k method and
using pre-tabulated ktables, assuming random overlap.
"""
import numpy as np
from numba import jit

@jit(nopython=True)
def interp_k(P_grid, T_grid, P_layer, T_layer, k_gas_w_g_p_t):
    """
    Interpolate the k coeffcients to given presures and temperatures
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
    k_gas_w_g_l(NGAS,NWAVEKTA,NG,NLAYER) : ndarray
        The interpolated-to-atmosphere k-coefficients.
        Has dimension: NGAS x NWAVE x NG x NLAYER.
    Notes
    -----
    Code breaks if P_layer/T_layer is out of the range of P_grid/T_grid.
    Mainly need to worry about max(T_layer)>max(T_grid).
    No extrapolation outside of the TP grid of ktable.
    """
    NGAS, NWAVE, NG, NPRESS, NTEMP = k_gas_w_g_p_t.shape
    NLAYER = len(P_layer)
    k_gas_w_g_l = np.zeros((NGAS,NWAVE,NG,NLAYER))
    kgood = np.zeros((NGAS,NWAVE,NG,NLAYER))

    for ilayer in range(NLAYER):
        p = P_layer[ilayer]
        t = T_layer[ilayer]

        # Find pressure grid points above and below current layer pressure
        ip = np.abs(P_grid-p).argmin()
        if P_grid[ip] >= p:
            ip_high = ip
            if ip == 0:
                p = P_grid[0]
                ip_low = 0
                ip_high = 1
            else:
                ip_low = ip-1
        elif P_grid[ip]<p:
            ip_low = ip
            if ip == NPRESS-1:
                p = P_grid[NPRESS-1]
                ip_high = NPRESS-1
                ip_low = NPRESS-2
            else:
                ip_high = ip + 1

        # Find temperature grid points above and below current layer temperature
        it = np.abs(T_grid-t).argmin()
        if T_grid[it] >= t:
            it_high = it
            if it == 0:
                t = T_grid[0]
                it_low = 0
                it_high = 1
            else:
                it_low = it -1
        elif T_grid[it] < t:
            it_low = it
            if it == NTEMP-1:
                t = T_grid[-1]
                it_high = NTEMP - 1
                it_low = NTEMP -2
            else:
                it_high = it + 1

        # Set up arrays for interpolation
        lnp = np.log(p)
        lnp_low = np.log(P_grid[ip_low])
        lnp_high = np.log(P_grid[ip_high])
        t_low = T_grid[it_low]
        t_high = T_grid[it_high]

        f11 = np.zeros((NGAS,NWAVE,NG))
        f12 = np.zeros((NGAS,NWAVE,NG))
        f22 = np.zeros((NGAS,NWAVE,NG))
        f21 = np.zeros((NGAS,NWAVE,NG))

        f11[:,:,:] = k_gas_w_g_p_t[:,:,:,ip_low,it_low]
        f12[:,:,:] = k_gas_w_g_p_t[:,:,:,ip_low,it_high]
        f21[:,:,:] = k_gas_w_g_p_t[:,:,:,ip_high,it_high]
        f22[:,:,:] = k_gas_w_g_p_t[:,:,:,ip_high,it_low]

        # Bilinear interpolation
        v = (lnp-lnp_low)/(lnp_high-lnp_low)
        u = (t-t_low)/(t_high-t_low)

        igood = np.where( (f11>0.0) & (f12>0.0) & (f22>0.0) & (f21>0.0) )
        ibad = np.where( (f11<=0.0) & (f12<=0.0) & (f22<=0.0) & (f21<=0.0) )

        for index in range(len(igood[0])):
            kgood[igood[0][index],igood[1][index],igood[2][index],ilayer] \
                = (1.0-v)*(1.0-u)*np.log(f11[igood[0][index],igood[1][index],igood[2][index]]) \
                + v*(1.0-u)*np.log(f22[igood[0][index],igood[1][index],igood[2][index]]) \
                + v*u*np.log(f21[igood[0][index],igood[1][index],igood[2][index]]) \
                + (1.0-v)*u*np.log(f12[igood[0][index],igood[1][index],igood[2][index]])
            kgood[igood[0][index],igood[1][index],igood[2][index],ilayer] \
                = np.exp(kgood[igood[0][index],igood[1][index],igood[2][index],ilayer])

        for index in range(len(ibad[0])):
            kgood[ibad[0][index],ibad[1][index],ibad[2][index],ilayer] \
                = (1.0-v)*(1.0-u)*f11[ibad[0][index],ibad[1][index],ibad[2][index]] \
                + v*(1.0-u)*f22[ibad[0][index],ibad[1][index],ibad[2][index]] \
                + v*u*f21[ibad[0][index],ibad[1][index],ibad[2][index]] \
                + (1.0-v)*u*f12[ibad[0][index],ibad[1][index],ibad[2][index]]

    k_gas_w_g_l = kgood
    return k_gas_w_g_l

@jit(nopython=True)
def rankg(weight, cont, del_g):
    """
    Combine the randomly overlapped k distributions of two gases into a single
    k distribution.

    Parameters
    ----------
    weight(NG) : ndarray
        Weights of points in the random k-dist
    cont(NG) : ndarray
        Random k-coeffs in the k-dist.
    del_g(NG) : ndarray
        Required weights of final k-dist.

    Returns
    -------
    k_g(NG) : ndarray
        Combined k-dist.
        Unit: cm^2 (per particle)
    """
    ng = len(del_g)
    nloop = ng*ng

    # sum delta gs to get cumulative g ordinate
    g_ord = np.zeros(ng+1)
    g_ord[1:] = np.cumsum(del_g)
    g_ord[ng] = 1

    # Sort random k-coeffs into ascending order. Integer array ico records
    # which swaps have been made so that we can also re-order the weights.
    ico = np.argsort(cont)
    cont = cont[ico]
    weight = weight[ico] # sort weights accordingly
    gdist = np.cumsum(weight)
    k_g = np.zeros(ng)

    ig = 0
    sum1 = 0.0
    cont_weight = cont * weight
    for iloop in range(nloop):
        if gdist[iloop] < g_ord[ig+1]:
            k_g[ig] = k_g[ig] + cont_weight[iloop]
            sum1 = sum1 + weight[iloop]
        else:
            frac = (g_ord[ig+1] - gdist[iloop-1])/(gdist[iloop]-gdist[iloop-1])
            k_g[ig] = k_g[ig] + np.float32(frac)*cont_weight[iloop]

            sum1 = sum1 + frac * weight[iloop]
            k_g[ig] = k_g[ig]/np.float32(sum1)

            ig = ig +1
            sum1 = (1.0-frac)*weight[iloop]
            k_g[ig] = np.float32(1.0-frac)*cont_weight[iloop]

    if ig == ng-1:
        k_g[ig] = k_g[ig]/np.float32(sum1)

    return k_g

@jit(nopython=True)
def noverlapg(k_gas_g, amount, del_g):
    """
    Combine k distributions of multiple gases given their number densities.

    Parameters
    ----------
    k_gas_g(NGAS,NG) : ndarray
        K-distributions of the different gases.
        Each row contains a k-distribution defined at NG g-ordinates.
        Unit: cm^2 (per particle)
    amount(NGAS) : ndarray
        Absorber amount of each gas.
        Unit: (no. of partiicles) cm^-2
    del_g(NG) : ndarray
        Gauss quadrature weights for the g-ordinates.
        These are the widths of the bins in g-space.

    Returns
    -------
    k_g(NG) : ndarray
        Combined k-distribution.
        Unit: cm^2 (per particle)
    """
    # amount = VMR_layer x U_layer
    NGAS = len(amount)
    NG = len(del_g)
    k_g = np.zeros(NG)
    weight = np.zeros(NG*NG)
    contri = np.zeros(NG*NG)
    for igas in range(NGAS-1):
        # first pair of gases
        if igas == 0:
            a1 = amount[igas]
            a2 = amount[igas+1]

            k_g1 = k_gas_g[igas,:]
            k_g2 = k_gas_g[igas+1,:]

            # skip if first k-distribution = 0.0
            if k_g1[NG-1]*a1 == 0.0:
                k_g = k_g2*a2
            # skip if second k-distribution = 0.0
            elif k_g2[NG-1]*a2 == 0.0:
                k_g = k_g1*a1
            else:
                nloop = 0
                for ig in range(NG):
                    for jg in range(NG):
                        weight[nloop] = del_g[ig]*del_g[jg]
                        contri[nloop] = k_g1[ig]*a1 + k_g2[jg]*a2
                        nloop = nloop + 1
                k_g = rankg(weight,contri,del_g)
        # subsequuent gases .. add amount*k to previous summed k
        else:
            a2 = amount[igas+1]
            # print('a2',a2)
            k_g1 = k_g
            k_g2 = k_gas_g[igas+1,:]

            # skip if first k-distribution = 0.0
            if k_g1[NG-1] == 0:
                k_g = k_g2*a2
            # Skip if second k-distribution = 0.0
            elif k_g2[NG-1]*a2== 0:
                k_g = k_g1
            else:
                nloop = 0
                for ig in range(NG):
                    for jg in range(NG):
                        weight[nloop] = del_g[ig]*del_g[jg]
                        contri[nloop] = k_g1[ig]+k_g2[jg]*a2
                        nloop = nloop + 1
                k_g = rankg(weight,contri,del_g)

    return k_g

@jit(nopython=True)
def calc_tau_gas(k_gas_w_g_p_t, P_layer, T_layer, VMR_layer, U_layer,
    P_grid, T_grid, del_g):
    """
    Parameters
    ----------
    k_gas_w_g_p_t(NGAS,NWAVE,NG,NPRESSK,NTEMPK) : ndarray
        k-coefficients.
        Unit: cm^2 (per particle)
        Has dimension: NWAVE x NG x NPRESSK x NTEMPK.
    P_layer(NLAYER) : ndarray
        Atmospheric pressure grid.
        Unit: Pa
    T_layer(NLAYER) : ndarray
        Atmospheric temperature grid.
        Unit: K
    VMR_layer(NLAYER,NGAS) : ndarray
        Array of volume mixing ratios for NGAS.
        Has dimensioin: NLAYER x NGAS
    U_layer(NLAYER) : ndarray
        Total number of gas particles in each layer.
        Unit: (no. of particle) m^-2
    P_grid(NPRESSK) : ndarray
        Pressure grid on which the k-coeff's are pre-computed.
    T_grid(NTEMPK) : ndarray
        Temperature grid on which the k-coeffs are pre-computed.
    del_g(NG) : ndarray
        Gauss quadrature weights for the g-ordinates.
        These are the widths of the bins in g-space.

    Returns
    -------
    tau_w_g_l(NWAVE,NG,NLAYER) : ndarray
        Optical path due to spectral line absorptions.

    Notes
    -----
    Absorber amounts (U_layer) is scaled down by a factor 1e-20 because Nemesis
    k-tables are scaled up by a factor of 1e20.
    """

    Scaled_U_layer = U_layer * 1.0e-20
    Scaled_U_layer *= 1.0e-4 # convert from per m^2 to per cm^2

    k_gas_w_g_l = interp_k(P_grid, T_grid, P_layer, T_layer, k_gas_w_g_p_t)
    Ngas, Nwave, Ng, Nlayer = k_gas_w_g_l.shape

    amount_layer = np.zeros((Nlayer,Ngas))
    for ilayer in range(Nlayer):
        amount_layer[ilayer,:] = Scaled_U_layer[ilayer] * VMR_layer[ilayer,:4]

    tau_w_g_l = np.zeros((Nwave,Ng,Nlayer))
    for iwave in range (Nwave):
        k_gas_g_l = k_gas_w_g_l[:,iwave,:,:]
        k_g_l = np.zeros((Ng,Nlayer))
        for ilayer in range(Nlayer):
            k_g_l[:,ilayer]\
                = noverlapg(k_gas_g_l[:,:,ilayer],amount_layer[ilayer,:],del_g)
            tau_w_g_l[iwave,:,ilayer] = k_g_l[:,ilayer]

    return tau_w_g_l