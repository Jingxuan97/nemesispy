#!/usr/local/bin/python3
# -*- coding: utf-8 -*-
"""Calculate the opacity of a mixture of gases using ktables."""
import numpy as np
from numba import jit

# @jit(nopython=True)
def new_k_overlap_two_gas(k_gas1_g, k_gas2_g, q1, q2, del_g):
    """
    Combines the absorption coefficient distributions of two gases with overlapping
    opacities. The overlapping is assumed to be random and the k-distributions are
    assumed to have NG mean values and NG weights. Correspondingly there are
    NG+1 ordinates in total.

    Parameters
    ----------
    k_gas1_g(NG) : ndarray
        k-coefficients for gas 1 at a particular wave bin, layer temperature and
        layer pressure.
    k_gas2_g(NG) : ndarray
        k-coefficients for gas 2 at a particular wave bin and layer, with
        particular layer temperature and pressure.
    q1 : real
        Volume mixing ratio of gas 1.
    q2 : real
        Volume mixing ratio of gas 2.
    del_g(NG) : ndarray
        Gauss quadrature weights for the g-ordinates.
        These are the widths of the bins in g-space.
        Assumed same for both gases.

    Returns
    -------
    k_combined_g(NG) : ndarray
        Combined k-distribution of both gases at a particular wave bin and layer.
    q_combined : real
        Combined volume mixing ratio of both gases.
    """
    NG = len(del_g)  # Number of g-ordinates
    k_combined_g = np.zeros(NG)
    q_combined = q1 + q2

    # If one or both gases have neglible opacities
    if((k_gas1_g[NG-1]<=0.0) and (k_gas2_g[NG-1]<=0.0)):
        pass
    elif( (q1<=0.0) and (q2<=0.0) ):
        pass
    elif((k_gas1_g[NG-1]==0.0) or (q1==0.0)):
        k_combined_g[:] = k_gas2_g[:] * q2/(q1+q2)
    elif((k_gas2_g[NG-1]==0.0) or (q2==0.0)):
        k_combined_g[:] = k_gas1_g[:] * q1/(q1+q2)

    # If both gases are significant, assume random overlapping of opacities
    else:
        nloop = NG * NG
        weight = np.zeros(nloop)
        k_list = np.zeros(nloop)
        ix = 0
        for i in range(NG):
            for j in range(NG):
                # construct weight of 2D opacity distribution
                weight[ix] = del_g[i] * del_g[j]
                # opacity is weighted by relative vmr
                k_list[ix] = (k_gas1_g[i]*q1 + k_gas2_g[j]*q2)/(q1+q2)
                ix = ix + 1

        # Get the cumulative g ordinate upper bound
        g_ord_upper = np.zeros(NG+1)
        for ig in range(NG):
            g_ord_upper[ig+1] = g_ord_upper[ig] + del_g[ig]
        g_ord_upper[0] = 0.0
        g_ord_upper[NG] = 1.0

        # Sort k_list array
        isort = np.argsort(k_list)
        k_list_ascend = k_list[isort]
        weight_ascend = weight[isort]

        # Create combined g-ordinate array
        g_ord_list = np.zeros(nloop)
        g_ord_list[0] = weight_ascend[0]
        for i in range(nloop-1):
            ix = i + 1
            g_ord_list[ix] = weight_ascend[ix] + g_ord_list[i]

        ig = 0
        weight_sum = 0.0

        # Need to understand what is going on here
        for iloop in range(nloop):

            if (g_ord_list[iloop]<g_ord_upper[ig+1]) & (ig<=NG-1):
                k_combined_g[ig] = k_combined_g[ig] \
                    + k_list_ascend[iloop] * weight_ascend[iloop]
                weight_sum = weight_sum + weight_ascend[iloop]
            else:
                frac = (g_ord_upper[ig+1]-g_ord_list[iloop-1])\
                    /(g_ord_list[iloop]-g_ord_list[iloop-1])
                k_combined_g[ig] = k_combined_g[ig] \
                    + k_list_ascend[iloop] * frac * weight_ascend[iloop]
                weight_sum = weight_sum + weight_ascend[iloop]
                k_combined_g[ig] = k_combined_g[ig] / weight_sum
                ig = ig + 1
                if(ig<=NG-1):
                    weight_sum = (1.-frac)*weight_ascend[iloop]
                    k_combined_g[ig] = k_combined_g[ig]\
                        + (1.-frac) * k_list_ascend[i] * weight_ascend[i]

        if ig==NG-1:
            k_combined_g[ig] = k_combined_g[ig] / weight_sum

    return k_combined_g, q_combined

# @jit(nopython=True)
def new_k_overlap(k_gas_w_g_l,del_g,VMR):
    """
    Combines the absorption coefficient distributions of several gases with overlapping
    opacities. The overlaps are implicitly assumed to be random and the k-distributions
    are assumed to have NG-1 mean values and NG-1 weights. Correspondingly there
    are NG ordinates in total.

    Parameters
    ----------
    k_gas_w_g_l(NGAS, NWAVE, NG, NLAYER) : ndarray
        k-distributions of gases.
    del_g(NG) : ndarray
        Gauss quadrature weights for the g-ordinates.
        These are the widths of the bins in g-space.
        Assumed same for all gases.
    VMR(NLAYER,NGAS) : ndarray
        Volume mixing ratios of gases at each layer.

    Returns
    -------
    k_w_g_l(NWAVE,NG,NLAYER) : ndarray
        Opacity at each wavelength bin, each g ordinate and each layer.
    """
    NGAS,NWAVE,NG,NLAYER = k_gas_w_g_l.shape
    k_w_g_l = np.zeros((NWAVE,NG,NLAYER))
    if NGAS<=1:
        # only one gas is under consideration
        k_w_g_l[:,:,:] = k_gas_w_g_l[0,:,:,:]
    else:
        for ilayer in range(NLAYER): #running for each p-T case
            for igas in range(NGAS-1):
                if igas==0:
                    # set up first and second gases to combine
                    kgas1_w_g = np.zeros((NWAVE,NG))
                    kgas2_w_g = np.zeros((NWAVE,NG))
                    kgas1_w_g[:,:] = k_gas_w_g_l[igas,:,:,ilayer] # NWAVE x NG
                    kgas2_w_g[:,:] = k_gas_w_g_l[igas+1,:,:,ilayer]
                    f1 = VMR[ilayer,igas]
                    f2 = VMR[ilayer,igas+1]
                    k_combined = np.zeros((NWAVE,NG))

                else:
                    kgas1_w_g[:,:] = k_combined[:,:]
                    kgas2_w_g[:,:] = k_gas_w_g_l[igas+1,:,:,ilayer]
                    f1 = f_combined
                    f2 = VMR[ilayer,igas+1]
                    k_combined = np.zeros((NWAVE,NG))

                for iwave in range(NWAVE):

                    k_g_combined, f_combined \
                        = new_k_overlap_two_gas(kgas1_w_g[iwave,:],
                        kgas2_w_g[iwave,:], f1, f2, del_g)
                    k_combined[iwave,:] = k_g_combined[:]

            k_w_g_l[:,:,ilayer] = k_combined[:,:]

    return k_w_g_l, f_combined


# @jit(nopython=True)
def new_k_overlap_two_gas(k_gas1_g, k_gas2_g, q1, q2, del_g):
    """
    Combines the absorption coefficient distributions of two gases with overlapping
    opacities. The overlapping is assumed to be random and the k-distributions are
    assumed to have NG mean values and NG weights. Correspondingly there are
    NG+1 ordinates in total.

    Parameters
    ----------
    k_gas1_g(NG) : ndarray
        k-coefficients for gas 1 at a particular wave bin, layer temperature and
        layer pressure.
    k_gas2_g(NG) : ndarray
        k-coefficients for gas 2 at a particular wave bin and layer, with
        particular layer temperature and pressure.
    q1 : real
        Volume mixing ratio of gas 1.
    q2 : real
        Volume mixing ratio of gas 2.
    del_g(NG) : ndarray
        Gauss quadrature weights for the g-ordinates.
        These are the widths of the bins in g-space.
        Assumed same for both gases.

    Returns
    -------
    k_combined_g(NG) : ndarray
        Combined k-distribution of both gases at a particular wave bin and layer.
    q_combined : real
        Combined volume mixing ratio of both gases.
    """
    NG = len(del_g)  # Number of g-ordinates
    k_combined_g = np.zeros(NG)
    q_combined = q1 + q2

    # If one or both gases have neglible opacities
    if((k_gas1_g[NG-1]<=0.0) and (k_gas2_g[NG-1]<=0.0)):
        pass
    elif( (q1<=0.0) and (q2<=0.0) ):
        pass
    elif((k_gas1_g[NG-1]==0.0) or (q1==0.0)):
        k_combined_g[:] = k_gas2_g[:] * q2/(q1+q2)
    elif((k_gas2_g[NG-1]==0.0) or (q2==0.0)):
        k_combined_g[:] = k_gas1_g[:] * q1/(q1+q2)

    # If both gases are significant, assume random overlapping of opacities
    else:
        nloop = NG * NG
        weight = np.zeros(nloop)
        k_list = np.zeros(nloop)
        ix = 0
        for i in range(NG):
            for j in range(NG):
                # construct weight of 2D opacity distribution
                weight[ix] = del_g[i] * del_g[j]
                # opacity is weighted by relative vmr
                k_list[ix] = (k_gas1_g[i]*q1 + k_gas2_g[j]*q2)/(q1+q2)
                ix = ix + 1

        # Get the cumulative g ordinate upper bound
        g_ord_upper = np.zeros(NG+1)
        for ig in range(NG):
            g_ord_upper[ig+1] = g_ord_upper[ig] + del_g[ig]
        g_ord_upper[0] = 0.0
        g_ord_upper[NG] = 1.0

        # Sort k_list array
        isort = np.argsort(k_list)
        k_list_ascend = k_list[isort]
        weight_ascend = weight[isort]

        # Create combined g-ordinate array
        g_ord_list = np.zeros(nloop)
        g_ord_list[0] = weight_ascend[0]
        for i in range(nloop-1):
            ix = i + 1
            g_ord_list[ix] = weight_ascend[ix] + g_ord_list[i]

        ig = 0
        weight_sum = 0.0

        # Need to understand what is going on here
        for iloop in range(nloop):

            if (g_ord_list[iloop]<g_ord_upper[ig+1]) & (ig<=NG-1):
                k_combined_g[ig] = k_combined_g[ig] \
                    + k_list_ascend[iloop] * weight_ascend[iloop]
                weight_sum = weight_sum + weight_ascend[iloop]
            else:
                frac = (g_ord_upper[ig+1]-g_ord_list[iloop-1])\
                    /(g_ord_list[iloop]-g_ord_list[iloop-1])
                k_combined_g[ig] = k_combined_g[ig] \
                    + k_list_ascend[iloop] * frac * weight_ascend[iloop]
                weight_sum = weight_sum + weight_ascend[iloop]
                k_combined_g[ig] = k_combined_g[ig] / weight_sum
                ig = ig + 1
                if(ig<=NG-1):
                    weight_sum = (1.-frac)*weight_ascend[iloop]
                    k_combined_g[ig] = k_combined_g[ig]\
                        + (1.-frac) * k_list_ascend[i] * weight_ascend[i]

        if ig==NG-1:
            k_combined_g[ig] = k_combined_g[ig] / weight_sum

    return k_combined_g, q_combined

# @jit(nopython=True)
def new_k_overlap(k_gas_w_g_l,del_g,VMR):
    """
    Combines the absorption coefficient distributions of several gases with overlapping
    opacities. The overlaps are implicitly assumed to be random and the k-distributions
    are assumed to have NG-1 mean values and NG-1 weights. Correspondingly there
    are NG ordinates in total.

    Parameters
    ----------
    k_gas_w_g_l(NGAS, NWAVE, NG, NLAYER) : ndarray
        k-distributions of gases.
    del_g(NG) : ndarray
        Gauss quadrature weights for the g-ordinates.
        These are the widths of the bins in g-space.
        Assumed same for all gases.
    VMR(NLAYER,NGAS) : ndarray
        Volume mixing ratios of gases at each layer.

    Returns
    -------
    k_w_g_l(NWAVE,NG,NLAYER) : ndarray
        Opacity at each wavelength bin, each g ordinate and each layer.
    """
    NGAS,NWAVE,NG,NLAYER = k_gas_w_g_l.shape
    k_w_g_l = np.zeros((NWAVE,NG,NLAYER))
    if NGAS<=1:
        # only one gas is under consideration
        k_w_g_l[:,:,:] = k_gas_w_g_l[0,:,:,:]
    else:
        for ilayer in range(NLAYER): #running for each p-T case
            for igas in range(NGAS-1):
                if igas==0:
                    # set up first and second gases to combine
                    kgas1_w_g = np.zeros((NWAVE,NG))
                    kgas2_w_g = np.zeros((NWAVE,NG))
                    kgas1_w_g[:,:] = k_gas_w_g_l[igas,:,:,ilayer] # NWAVE x NG
                    kgas2_w_g[:,:] = k_gas_w_g_l[igas+1,:,:,ilayer]
                    f1 = VMR[ilayer,igas]
                    f2 = VMR[ilayer,igas+1]
                    k_combined = np.zeros((NWAVE,NG))

                else:
                    kgas1_w_g[:,:] = k_combined[:,:]
                    kgas2_w_g[:,:] = k_gas_w_g_l[igas+1,:,:,ilayer]
                    f1 = f_combined
                    f2 = VMR[ilayer,igas+1]
                    k_combined = np.zeros((NWAVE,NG))

                for iwave in range(NWAVE):

                    k_g_combined, f_combined \
                        = new_k_overlap_two_gas(kgas1_w_g[iwave,:],
                        kgas2_w_g[iwave,:], f1, f2, del_g)
                    k_combined[iwave,:] = k_g_combined[:]

            k_w_g_l[:,:,ilayer] = k_combined[:,:]

    return k_w_g_l, f_combined