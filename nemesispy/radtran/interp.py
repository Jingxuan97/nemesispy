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


# @jit(nopython=True)
def interp_k_old(P_grid, T_grid, P_layer, T_layer, k_gas_w_g_p_t):
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
"""
def interp_k(P_grid, T_grid, P_layer, T_layer, k_gas_w_g_p_t):

    NGAS, NWAVE, NG, Npress, Ntemp = k_gas_w_g_p_t.shape
    NLAYER = len(P_layer)
    k_gas_w_g_l = np.zeros((NGAS,NWAVE,NG,NLAYER))

    for ilayer in range(NLAYER):
        press1 = P_layer[ilayer]
        temp1 = T_layer[ilayer]

        #Getting the levels just above and below the desired points
        lpress  = np.log(press1)
        press0,ip = find_nearest(P_grid,press1)

        if P_grid[ip]>=press1:
            iphi = ip
            if ip==0:
                lpress = np.log(P_grid[0])
                ipl = 0
                iphi = 1
            else:
                ipl = ip - 1
        elif P_grid[ip]<press1:
            ipl = ip
            if ip==Npress-1:
                lpress = np.log(P_grid[Npress-1])
                iphi = Npress - 1
                ipl = Npress - 2
            else:
                iphi = ip + 1

        temp0,it = find_nearest(T_grid,temp1)

        if T_grid[it]>=temp1:
            ithi = it
            if it==0:
                temp1 = T_grid[0]
                itl = 0
                ithi = 1
            else:
                itl = it - 1
        elif T_grid[it]<temp1:
            itl = it
            if it==Ntemp-1:
                temp1 = T_grid[Ntemp-1]
                ithi = Ntemp - 1
                itl = Ntemp - 2
            else:
                ithi = it + 1

        plo = np.log(P_grid[ipl])
        phi = np.log(P_grid[iphi])
        tlo = T_grid[itl]
        thi = T_grid[ithi]
        klo1 = np.zeros([NGAS,NWAVE,NG])
        klo2 = np.zeros([NGAS,NWAVE,NG])
        khi1 = np.zeros([NGAS,NWAVE,NG])
        khi2 = np.zeros([NGAS,NWAVE,NG])
        klo1[:] = k_gas_w_g_p_t[:,:,ipl,itl]
        klo2[:] = k_gas_w_g_p_t[:,:,ipl,ithi]
        khi2[:] = k_gas_w_g_p_t[:,:,iphi,ithi]
        khi1[:] = k_gas_w_g_p_t[:,:,iphi,itl]

        #Interpolating to get the k-coefficients at desired p-T
        v = (lpress-plo)/(phi-plo)
        u = (temp1-tlo)/(thi-tlo)
        dudt = 1./(thi-tlo)

        igood = np.where( (klo1>0.0) & (klo2>0.0) & (khi1>0.0) & (khi2>0.0) )
        k_gas_w_g_l[igood[2],igood[0],igood[1],ilayer] \
            = (1.0-v)*(1.0-u)*np.log(klo1[igood[2],igood[0],igood[1]]) \
            + v*(1.0-u)*np.log(khi1[igood[2],igood[0],igood[1]]) \
            + v*u*np.log(khi2[igood[2],igood[0],igood[1]]) \
            + (1.0-v)*u*np.log(klo2[igood[2],igood[0],igood[1]])
        k_gas_w_g_l[igood[2],igood[0],igood[1],ilayer] \
            = np.exp(k_gas_w_g_l[igood[2],igood[0],igood[1],ilayer])

        ibad = np.where( (klo1<=0.0) & (klo2<=0.0) & (khi1<=0.0) & (khi2<=0.0) )
        k_gas_w_g_l[ibad[2],ibad[0],ibad[1],ilayer] \
            = (1.0-v)*(1.0-u)*klo1[ibad[2],ibad[0],ibad[1]] \
            + v*(1.0-u)*khi1[ibad[2],ibad[0],ibad[1]] \
            + v*u*khi2[ibad[2],ibad[0],ibad[1]] \
            + (1.0-v)*u*klo2[ibad[2],ibad[0],ibad[1]]

    return k_gas_w_g_l
"""
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

# @jit(nopython=True)
def mix_two_gas_k(k_g1, k_g2, VMR1, VMR2, del_g):
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
        Gauss quadrature weights for the g-ordinates,
        assumed same for both gases.

    Returns
    -------
    k_g_combined
        Combined k-coefficients for the 'mixed gas'.
    VMR_combined
        Volume mixing ratio of "mixed gas".
    """

    # Introduce a minimum cut off for optical path
    cut_off = 0

    # Combine two optically active gases into a 'new' gas
    Ng = len(del_g)
    k_g_combined = np.zeros(Ng)
    VMR_combined = VMR1+VMR2

    if k_g1[-1] * VMR1 <= cut_off and k_g2[-1] * VMR2 <= cut_off:
        pass
    elif k_g1[-1] * VMR1 <= cut_off:
        k_g_combined = k_g2*VMR2/VMR_combined
    elif k_g2[-1] * VMR2 <= cut_off:
        k_g_combined = k_g1*VMR1/VMR_combined
    else:
        # Overlap Ng k-coeffs with Ng k-coeffs randomly: Ng x Ng possible pairs
        nloop = Ng**2
        weight_mix = np.zeros(nloop)
        k_g_mix = np.zeros(nloop)
        # Mix k-coeffs of gases weighted by their relative VMR.
        for i in range(Ng):
            for j in range(Ng):
                # equation 9 Amundsen 2017 (equation 20 Mollier 2015)
                k_g_mix[i*Ng+j] = (k_g1[i]*VMR1+k_g2[j]*VMR2)/VMR_combined
                # equation 10 Amundsen 2017
                weight_mix[i*Ng+j] = del_g[i]*del_g[j]

        # getting the cumulative g ordinate
        g_ord = np.zeros(Ng+1)
        g_ord[0] = 0.0
        for ig in range(Ng):
            g_ord[ig+1] = g_ord[ig]+del_g[ig]
        # print('g_ord1',g_ord)
        # g_ord = np.cumsum(del_g)
        # g_ord = np.append([0],g_ord)
        # print('g_ord2',g_ord)
        if g_ord[Ng]<1.0:
            g_ord[Ng] = 1.0

        # Resort-rebin procedure: Sort new "mixed" k-coeff's from low to high
        # see Amundsen et al. 2016 or section B.2.1 in Molliere et al. 2015
        ascending_index = np.argsort(k_g_mix)
        k_g_mix_sorted = k_g_mix[ascending_index]
        weight_mix_sorted = weight_mix[ascending_index]

        gdist = np.zeros(nloop)
        gdist[0] = weight_mix_sorted[0]
        for iloop in range(1,nloop):
            gdist[iloop] = weight_mix_sorted[iloop]+ gdist[iloop-1]

        ig = 0
        sum1 = 0.0
        for iloop in range(nloop):

            if gdist[iloop]<g_ord[ig+1] and (ig <= (Ng-1)):
                k_g_combined[ig] += k_g_mix_sorted[iloop]*weight_mix[iloop]
                sum1 = sum1 + weight_mix_sorted[iloop]
            else:
                frac = (g_ord[ig+1]-gdist[iloop-1])/(gdist[iloop]-gdist[iloop-1])
                k_g_combined[ig] += frac*k_g_mix_sorted[iloop]*weight_mix[iloop]
                sum1 = sum1 + weight_mix_sorted[iloop]
                k_g_combined[ig] = k_g_combined[ig] / sum1
                ig += 1
                if ig<=Ng-1:
                    sum1 = (1.-frac)*weight_mix_sorted[iloop]
                    k_g_combined[ig] += (1-frac)*k_g_mix_sorted[iloop]*weight_mix[iloop]

        if ig == Ng-1:
            k_g_combined[ig] = k_g_combined[ig]/sum1

        """# Chimera
        #combining w/weights--see description on Molliere et al. 2015
        sum_weight = np.cumsum(weight_mix_sorted)
        x = sum_weight/np.max(sum_weight)*2.-1

        # log_k_g_mix = np.log10(k_g_mix_sorted)

        # Get the cumulative g ordinate upper bound
        # g_ord[ig+1] = g_ord[ig] + del_g[ig]
        # IndexError: index 20 is out of bounds for axis 0 with size 20
        g_ord = np.zeros(Ng+1)
        # g_ord = np.zeros(Ng)
        for ig in range(Ng):
            g_ord[ig+1] = g_ord[ig] + del_g[ig]

        # g_ord = g_ord - del_g
        # g_ord = np.append([0],g_ord)
        # g_ord[-1] = 1
        # print(g_ord)
        for i in range(Ng):
            loc = np.where(x >=  g_ord[i])[0][0]
            k_g_combined[i] = k_g_mix_sorted[loc]
            # k_g_combined[i]=10**log_k_g_mix[loc]
        """

    return k_g_combined, VMR_combined

# @jit(nopython=True)
def mix_multi_gas_k(k_gas_g, VMR, del_g):
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
    """
    g_ord = np.array([0.0034357 , 0.01801404, 0.04388279, 0.08044151, 0.126834  ,
    0.1819732 , 0.2445665 , 0.3131469 , 0.3861071 , 0.4617367 ,
    0.5382633 , 0.6138929 , 0.6868531 , 0.7554335 , 0.8180268 ,
    0.873166  , 0.9195585 , 0.9561172 , 0.981986  , 0.9965643 ])
    """
    ngas = k_gas_g.shape[0]
    k_g_combined,VMR_combined = k_gas_g[0,:],VMR[0]
    #mixing in rest of gases inside a loop
    for j in range(1,ngas):
        k_g_combined,VMR_combined\
            = mix_two_gas_k(k_g_combined,k_gas_g[j,:],VMR_combined,VMR[j],del_g)
    return k_g_combined, VMR_combined
