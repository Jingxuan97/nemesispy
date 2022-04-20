#!/usr/local/bin/python3
# -*- coding: utf-8 -*-
"""Calculate the opacity of a mixture of gases using ktables."""
import numpy as np
from numba import jit

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
