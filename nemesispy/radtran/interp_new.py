#!/usr/local/bin/python3
# -*- coding: utf-8 -*-
"""Calculate the opacity of a mixture of gases using ktables."""
import numpy as np
from numba import jit

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


def interp_k(P_grid, T_grid, P_layer, T_layer, k_gas_w_g_p_t, wavecalc=None):
    """
    Follows normal sequence
    Calculate the k coeffcients of gases at given presures and temperatures
    using pre-tabulated k-tables.

    Parameters
    ----------

    Returns
    -------
    """
    NGAS, NWAVE, NG, NPRESS, NTEMP = k_gas_w_g_p_t.shape
    NLAYER = len(P_layer)
    # print('P_layer',P_layer)
    k_gas_w_g_l = np.zeros([NGAS,NWAVE,NG,NLAYER])

    # kgood (NGAS, NWAVE, NG, NLAYER)
    kgood = np.zeros([NGAS,NWAVE,NG,NLAYER])
    for ilayer in range(NLAYER):
        press1 = P_layer[ilayer]
        temp1 = T_layer[ilayer]

        # find the Pressure levels just above and below that of current layer
        lpress = np.log(press1)
        press0, ip = find_nearest(P_grid,press1)

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
        temp0, it = find_nearest(T_grid, temp1)

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

        klo1 = np.zeros([NGAS,NWAVE,NG])
        klo2 = np.zeros([NGAS,NWAVE,NG])
        khi1 = np.zeros([NGAS,NWAVE,NG])
        khi2 = np.zeros([NGAS,NWAVE,NG])

        # klo1 = np.zeros([self.NWAVE,self.NG,self.NGAS])

        klo1[:] = k_gas_w_g_p_t[:,:,:,ipl,itl]
        klo2[:] = k_gas_w_g_p_t[:,:,:,ipl,ithi]
        khi2[:] = k_gas_w_g_p_t[:,:,:,iphi,ithi]
        khi1[:] = k_gas_w_g_p_t[:,:,:,iphi,itl]

        # print('klo1',klo1)
        # print(klo1.shape)

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

def index(N):
    return N-1

def sort2g(RA):
    """
    Modified numerical recipes routine to sort a vector RA of length N
    into ascending order. Integer vector IB is initially set to
    1,2,3,... and on output keeps a record of how RA has been sorted.
    """
    # print('RA=',RA)
    N = len(RA)
    # print('N=',N)
    IB = np.arange(1,N+1)
    # print('IB',IB)
    L = int(N/2)+1
    # print('L',L)
    IR = N
    # print('IR',IR)

    while True:
        # print('list',RA)
        # at least two elements
        if L > 1:
            # print('L>=1, L=', L)
            L = L-1
            # print('L=',L)
            RRA = RA[index(L)]
            # print('RRA=',RRA)
            IRB = IB[index(L)]
            # print('IRB=',IRB)
        else:
            # only one element
            # print('else')
            # print('IR=',IR)
            RRA = RA[index(IR)]
            IRB = IB[index(IR)]
            RA[index(IR)] = RA[index(1)]
            IB[index(IR)] = IB[index(1)]
            IR = IR - 1
            # print('IR=',IR)
            if IR == 1:
                RA[index(1)] = RRA
                IB[index(1)] = IRB
                # print('return')
                return RA,IB-1
            # end if
        # end if
        I = L
        # print('I=',I)
        J = L+L
        # print('J=',J)

        while J<=IR:
            if J<IR:
                if RA[index(J)]<=RA[index(J+1)]:
                    J = J+1
            if RRA < RA[index(J)]:
                RA[index(I)] = RA[index(J)]
                IB[index(I)] = IB[index(J)]
                I = J
                J = J+J
            else:
                J = IR+1
            # end if
        RA[index(I)] = RRA
        IB[index(I)] = IRB

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
    # print('rankg called')
    # check if this is actually the case
    ng = len(del_g)
    nloop = ng*ng
    gw = del_g

    g_ord = np.zeros(ng+1)
    g_ord[0] = 0.0
    # sum delta gs to get cumulative g ordinate
    for ig in range(ng):
        g_ord[ig+1] = g_ord[ig] + gw[ig]

    if g_ord[ng] < 1.0:
        g_ord[ng] = 1
    # print('g_ord',g_ord)

    # Sort random k-coeffs into ascending order. Integer array ico records
    # which swaps have been made so that we can also re-order the weights.
    cont, ico = sort2g(cont)
    # sort weights accordingly
    weight = weight[ico]
    # print('cont')
    # print(cont)
    # print('weight')
    # print(weight)
    # now form new g(k) by summing over weight
    gdist = np.zeros(nloop)
    gdist[0] = weight[0]
    for iloop in range(1,nloop):
        gdist[iloop] = weight[iloop] + gdist[iloop-1]

    k_g = np.zeros(ng)

    ig = 0
    sum1 = 0.0
    # print('g_ord',g_ord)
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
            # print('frac',frac)
    # print('ig',ig)
    # print('ng',ng)
    if ig == ng-1:
        k_g[ig] = k_g[ig]/np.float32(sum1)

    return k_g

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
