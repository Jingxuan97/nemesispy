#!/usr/local/bin/python3
# -*- coding: utf-8 -*-
"""Calculate the opacity of a mixture of gases using ktables."""
import numpy as np
from numba import jit
from math import log

def find_nearest(input_array, target_value):
    """
    Find the closest value in an array

    Parameters
    ----------
    input_array : ndarray/list
        An array of numbers.
    target_value : real
        Value to search for


    Returns
    -------
    idx : ndarray
        Index of closest_value within array
    array[idx] : ndarray
        Closest number to target_value in the input array
    """
    array = np.asarray(input_array)
    idx = (np.abs(array - target_value)).argmin()
    return array[idx], idx

# @jit(nopython=True)
def interp_k1(P_grid, T_grid, P_layer, T_layer, k_gas_w_g_p_t, wavecalc=None):
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
        lpress = log(press1)
        # press0, ip = find_nearest(P_grid,press1)
        ip = abs(P_grid-press1).argmin()
        press0 = P_grid[ip]

        if P_grid[ip] >= press1:
            iphi = ip
            if ip == 0:
                lpress = log(P_grid[0])
                ipl = 0
                iphi = 1
            else:
                ipl = ip - 1
        elif P_grid[ip]<press1:
            ipl = ip
            if ip == NPRESS -1:
                lpress = log(P_grid[NPRESS-1])
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
        plo = log(P_grid[ipl])
        phi = log(P_grid[iphi])
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

# @jit(nopython=True)
def interp_k(P_grid, T_grid, P_layer, T_layer, k_gas_w_g_p_t):
    """
    Adapted from chimera https://github.com/mrline/CHIMERA.
    Interpolates the k-tables to input atmospheric P & T for each wavenumber and
    g-ordinate for each gas with a standard bi-linear interpolation scheme.

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
    NGAS, NWAVE, NG, Npress, Ntemp = k_gas_w_g_p_t.shape
    NLAYER = len(P_layer)
    k_gas_w_g_l = np.zeros((NGAS,NWAVE,NG,NLAYER))
    for ilayer in range(NLAYER): # loop through layers
        P = P_layer[ilayer]
        T = T_layer[ilayer]
        # Workaround when atmospheric layer pressure or temperature is out of
        # range of the ktable TP grid
        if T > T_grid[-1]:
            T = T_grid[-1]#-1
        if T <= T_grid[0]:
            T = T_grid[0]+1e-6
        if P > P_grid[-1]:
            P = P_grid[-1]#-1
        if P <= P_grid[0]:
            P = P_grid[0]+1e-6
        # find the points on the k table TP grid that sandwich the
        # atmospheric layer TP
        P_index_hi = np.where(P_grid >= P)[0][0]
        P_index_low = np.where(P_grid < P)[0][-1]
        T_index_hi = np.where(T_grid >= T)[0][0]
        T_index_low = np.where(T_grid < T)[0][-1]
        P_hi = P_grid[P_index_hi]
        P_low = P_grid[P_index_low]
        T_hi = T_grid[T_index_hi]
        T_low = T_grid[T_index_low]
        # interpolate
        for igas in range(NGAS): # looping through gases
            for iwave in range(NWAVE): # looping through wavenumber
                for ig in range(NG): # looping through g-ord
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

# @jit(nopython=True)
def rankg(weight, cont, del_g):
    """
    Parameters
    ----------
    gw(maxg)            REAL    Required weights of final k-dist.
    ng                  INTEGER Number of weights.
    weight(maxrank)     REAL    Weights of points in random k-dist
    cont(maxrank)       REAL    Random k-coeffs.

    Returns
    -------
    k_g(maxg)       REAL    Mean k-dist.
    """

    ng = len(del_g)
    nloop = ng*ng

    """
    gw = del_g
    g_ord = np.zeros(ng+1)
    g_ord[0] = 0.0
    # sum delta gs to get cumulative g ordinate
    for ig in range(ng):
        g_ord[ig+1] = g_ord[ig] + gw[ig]

    if g_ord[ng] < 1.0:
        g_ord[ng] = 1
    """
    """
    g_ord = np.cumsum(del_g)
    g_ord = np.insert(g_ord,0,0)
    g_ord[ng] = 1
    # if g_ord[ng] < 1.0:
    #     g_ord[ng] = 1
    """
    g_ord = np.zeros(ng+1)
    g_ord[1:] = np.cumsum(del_g)
    g_ord[ng] = 1

    # Sort random k-coeffs into ascending order. Integer array ico records
    # which swaps have been made so that we can also re-order the weights.
    # cont, ico = sort2g(cont)
    ico = np.argsort(cont)
    cont = cont[ico]

    # sort weights accordingly
    weight = weight[ico]

    """
    gdist = np.zeros(nloop)
    gdist[0] = weight[0]
    for iloop in range(1,nloop):
        gdist[iloop] = weight[iloop] + gdist[iloop-1]
    """
    gdist = np.cumsum(weight)
    k_g = np.zeros(ng)


    """
    ig = 0
    sum1 = 0.0
    for iloop in range(nloop):
        if gdist[iloop] < g_ord[ig+1]:
            k_g[ig] = k_g[ig] + cont[iloop]*weight[iloop]
            sum1 = sum1 + weight[iloop]
        else:
            frac = (g_ord[ig+1] - gdist[iloop-1])/(gdist[iloop]-gdist[iloop-1])
            k_g[ig] = k_g[ig] + np.float32(frac)*cont[iloop]*weight[iloop]

            sum1 = sum1 + frac * weight[iloop]
            k_g[ig] = k_g[ig]/np.float32(sum1)

            ig = ig +1
            sum1 = (1.0-frac)*weight[iloop]
            k_g[ig] = np.float32(1.0-frac)*cont[iloop]*weight[iloop]
    """

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


    """
    icut_old = 0
    sum1 = np.zeros(ng)
    cont_weight = cont * weight
    for ig in range(ng-1):
        # array index just before a g_ord
        icut = np.max(np.where(gdist < g_ord[ig+1])[0])
        k_g[ig] = k_g[ig] + np.sum(cont_weight[icut_old:(icut+1)])
        sum1[ig] = np.sum(weight[icut_old:(icut+1)])
        frac = (g_ord[ig+1] - gdist[icut]) / (gdist[icut+1]-gdist[icut])
        k_g[ig] = k_g[ig] + np.float32(frac)*cont_weight[icut+1]
        sum1[ig] = sum1[ig] + frac * weight[icut+1]
        k_g[ig] = k_g[ig]/np.float32(sum1[ig])
        sum1[ig+1] = (1.0-frac)*weight[icut+1]

    k_g[-1] = k_g[-1] + np.sum(cont_weight[icut+1:])
    sum1[-1] = sum1[-1] + np.sum(weight[icut+1:])
    k_g[-1] = k_g[-1]/np.float32(sum1[-1])
    """
    return k_g

# @jit(nopython=True)
def noverlapg(k_gas_g, amount, del_g):
    """
C   delg(ng)        REAL    Widths of bins in g-space.
C   ng              INTEGER Number of ordinates.
C   ngas            INTEGER Number of k-tables to overlap.
C   amount(maxgas)      REAL    Absorber amount of each gas.
C   k_gn(maxg,maxgas)   REAL    K-distributions of the different
C                               gases.
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
                # print('2nd')
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
            # skip if second k-distribution = 0.0
            elif k_g2[NG-1]*a2== 0:
                # print('elif',k_g)
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
