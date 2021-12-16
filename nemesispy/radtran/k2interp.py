#!/usr/local/bin/python3
# -*- coding: utf-8 -*-
"""Mix ktables of different gases."""
import numpy as np
# from numba import jit

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
    k_gas_w_g_p_t(Ngas,Nwave,Ng,Npress,Ntemp) : ndarray
        k-coefficient array,
        Has dimensiion: Ngas x Nwave x Ng x Npress x Ntemp

    Returns
    -------
    k_gas_w_g_l(Ngas,Nwave,Ng,Nlayer) : ndarray
        The interpolated-to-atmosphere k-coefficients.
        Has dimension: Ngas x Nwave x Ng x Nlayer.
    Notes
    -----
    Units: bar for pressure and K for temperature.
    Code breaks if P_layer/T_layer is out of the range of P_grid/T_grid.
    Mainly need to worry about max(T_layer)>max(T_grid).
    No extrapolation outside of the TP grid of ktable.
    """
    Ngas, Nwave, Ng, Npress, Ntemp = k_gas_w_g_p_t.shape
    Nlayer = len(P_layer)
    k_gas_w_g_l = np.zeros((Ngas,Nwave,Ng,Nlayer))
    for ilayer in range(Nlayer): # loop through layers
        P = P_layer[ilayer]
        T = T_layer[ilayer]
        # Workaround when atmospheric layer pressure or temperature is out of
        # range of the ktable TP grid
        if T > T_grid[-1]:
            T = T_grid[-1]-1
        if T < T_grid[0]:
            T = T_grid[0]+1
        if P > P_grid[-1]:
            P = P_grid[-1]-1
        if P < P_grid[0]:
            P = P_grid[0]+1
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
        k-coefficients for gas 1 at a particular wave bin and layer, with
        particular layer temperature and pressure.
    k_gas2_g(ng) : ndarray
        k-coefficients for gas 2 at a particular wave bin and layer, with
        particular layer temperature and pressure.
    q1 : real
        Volume mixing ratio of gas 1.
    q2 : real
        Volume mixing ratio of gas 2.
    del_g(ng) : ndarray
        Gauss quadrature weights for the g-ordinates, assumed same for both gases.
        These are the widths of the bins in g-space.

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
