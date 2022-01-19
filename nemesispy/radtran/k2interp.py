#!/usr/local/bin/python3
# -*- coding: utf-8 -*-
"""Calculate the opacity of a mixture of gases using ktables."""
import numpy as np
# from numba import jit
"""
DO a transmission spectrum
"""
def find_nearest(array, value):

    """
    FUNCTION NAME : find_nearest()

    DESCRIPTION : Find the closest value in an array

    INPUTS :

        array :: List of numbers
        value :: Value to search for

    OPTIONAL INPUTS: none

    OUTPUTS :

        closest_value :: Closest number to value in array
        index :: Index of closest_value within array

    CALLING SEQUENCE:

        closest_value,index = find_nearest(array,value)

    MODIFICATION HISTORY : Juan Alday (29/04/2019)

    """
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx],idx


def cal_k(P_grid, T_grid, P_layer, T_layer, k_gas_w_g_p_t, wavecalc=None):
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
    print('P_layer',P_layer)
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

        print('klo1',klo1)
        print(klo1.shape)

        # bilinear interpolation
        v = (lpress-plo)/(phi-plo)
        u = (temp1-tlo)/(thi-tlo)
        dudt = 1./(thi-tlo)

        igood = np.where( (klo1>0.0) & (klo2>0.0) & (khi1>0.0) & (khi2>0.0) )
        # NGAS x NWAVE x NG
        # print('kgood',kgood)
        print('NGAS',NGAS)
        print('igood[0]',igood[0])
        print('igood[1]',igood[1])
        print('igood[2]',igood[2])
        print('ilayer',ilayer)
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
    k_gas1_g(NG) : ndarray
        k-coefficients for gas 1 at a particular wave bin and layer, with
        particular layer temperature and pressure.
    k_gas2_g(NG) : ndarray
        k-coefficients for gas 2 at a particular wave bin and layer, with
        particular layer temperature and pressure.
    q1 : real
        Volume mixing ratio of gas 1.
    q2 : real
        Volume mixing ratio of gas 2.
    del_g(NG) : ndarray
        Gauss quadrature weights for the g-ordinates, assumed same for both gases.
        These are the widths of the bins in g-space.

    Returns
    -------
    k_combined_g(NG) : ndarray
        Combined k-distribution of both gases at a particular wave bin and layer.
    q_combined : real
        Combined volume mixing ratio of both gases.
    """
    NG = len(del_g)  #Number of g-ordinates
    k_combined_g = np.zeros(NG)
    q_combined = q1 + q2
    # print('q1,q2',q1,q2)

    if((k_gas1_g[NG-1]<=0.0) and (k_gas2_g[NG-1]<=0.0)):
        # both gases have neglible opaciteis
        pass
    elif( (q1<=0.0) and (q2<=0.0) ):
        # both gases have neglible VMR
        pass
    elif((k_gas1_g[NG-1]==0.0) or (q1==0.0)):
        # gas 1 is neglible
        k_combined_g[:] = k_gas2_g[:] * q2/(q1+q2)
    elif((k_gas2_g[NG-1]==0.0) or (q2==0.0)):
        # gas 2 is neglible
        k_combined_g[:] = k_gas1_g[:] * q1/(q1+q2)
    else:
        nloop = NG * NG
        weight = np.zeros(nloop)
        contri = np.zeros(nloop)
        ix = 0
        for i in range(NG):
            for j in range(NG):
                weight[ix] = del_g[i] * del_g[j]
                contri[ix] = (k_gas1_g[i]*q1 + k_gas2_g[j]*q2)/(q1+q2)
                ix = ix + 1

        #getting the cumulative g ordinate
        g_ord = np.zeros(NG+1)
        g_ord[0] = 0.0
        for ig in range(NG):
            g_ord[ig+1] = g_ord[ig] + del_g[ig]

        if g_ord[NG]<1.0:
            g_ord[NG] = 1.0

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

            if( (gdist[i]<g_ord[ig+1]) & (ig<=NG-1) ):
                k_combined_g[ig] = k_combined_g[ig] + contrib1[i] * weight1[i]
                sum1 = sum1 + weight1[i]
            else:
                frac = (g_ord[ig+1]-gdist[i-1])/(gdist[i]-gdist[i-1])
                k_combined_g[ig] = k_combined_g[ig] + frac * contrib1[i] * weight1[i]
                sum1 = sum1 + weight1[i]
                k_combined_g[ig] = k_combined_g[ig] / sum1
                ig = ig + 1
                if(ig<=NG-1):
                    sum1 = (1.-frac)*weight1[i]
                    k_combined_g[ig] = k_combined_g[ig] + (1.-frac) * contrib1[i] * weight1[i]

        if ig==NG-1:
            k_combined_g[ig] = k_combined_g[ig] / sum1

    # print('k_combined_g, q_combined',k_combined_g, q_combined)
    return k_combined_g, q_combined

def new_k_overlap(k_gas_w_g_l,del_g,f):
    """
    Combines the absorption coefficient distributions of several gases with overlapping
    opacities. The overlaps are implicitly assumed to be random and the k-distributions
    are assumed to have NG-1 mean values and NG-1 weights. Correspondingly there
    are NG ordinates in total.

    Parameters
    ----------
    k_gas_w_g_l(NGAS, NWAVE, NG, NLAYER) : ndarray
        k-distributions of the different gases
    del_g(NG) : ndarray
        Gauss quadrature weights for the g-ordinates, assumed same for all gases.
    f(NGAS,NLAYER) : ndarray
        fraction of the different gases at each of the p-T points

    Returns
    -------
    k_w_g_l(NWAVE,NG,NLAYER) : ndarray
        Opacity at each wavelength bin, each g ordinate and each layer.
    """
    NGAS,NWAVE,NG,NLAYER = k_gas_w_g_l.shape

    k_w_g_l = np.zeros((NWAVE,NG,NLAYER))
    """
    Error spotted:
    old : k_w_g_l[:,:,:] = k_gas_w_g_l[:,:,:,0]
    new : k_w_g_l[:,:,:] = k_gas_w_g_l[0,:,:,:]
    """
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
                    kgas1_w_g[:,:] = k_gas_w_g_l[igas,:,:,ilayer]
                    kgas2_w_g[:,:] = k_gas_w_g_l[igas+1,:,:,ilayer]
                    f1 = f[igas,ilayer]
                    f2 = f[igas+1,ilayer]

                    k_combined = np.zeros((NWAVE,NG))
                    # f_combined = np.zeros((NGAS,NLAYER))

                else:
                    #kgas1_w_g = np.zeros((NWAVE,NG))
                    #kgas2_w_g = np.zeros((NWAVE,NG))
                    kgas1_w_g[:,:] = k_combined[:,:]
                    kgas2_w_g[:,:] = k_gas_w_g_l[igas+1,:,:,ilayer]
                    f1 = f_combined
                    f2 = f[igas+1,ilayer]

                    k_combined = np.zeros((NWAVE,NG))

                for iwave in range(NWAVE):

                    k_g_combined, f_combined = new_k_overlap_two_gas(kgas1_w_g[iwave,:], kgas2_w_g[iwave,:], f1, f2, del_g)
                    k_combined[iwave,:] = k_g_combined[:]

            k_w_g_l[:,:,ilayer] = k_combined[:,:]

    return k_w_g_l, f_combined


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
        Gauss quadrature weights for the g-ordinates, assumed same for both gases.

    Returns
    -------
    k_g_combined
        Mixed k-coefficients for the given pair of gases
    VMR_combined
        Volume mixing ratio of "mixed gas".
    """
    g_ord = np.array([0.0034357 , 0.01801404, 0.04388279, 0.08044151, 0.126834  ,
       0.1819732 , 0.2445665 , 0.3131469 , 0.3861071 , 0.4617367 ,
       0.5382633 , 0.6138929 , 0.6868531 , 0.7554335 , 0.8180268 ,
       0.873166  , 0.9195585 , 0.9561172 , 0.981986  , 0.9965643 ])

    VMR_combined = VMR1+VMR2
    Ng = len(g_ord)
    k_g_combined = np.zeros(Ng)
    cut_off = 1e-40
    print('k_g1',k_g1)
    if k_g1[-1] * VMR1 < cut_off and k_g2[-1] * VMR2 < cut_off:
        pass
    elif k_g1[-1] * VMR1 < cut_off:
        k_g_combined = k_g2*VMR2/VMR_combined
    elif k_g2[-1] * VMR2 < cut_off:
        k_g_combined = k_g1*VMR1/VMR_combined
    else:
        # Overlap Ng k-coeffs with Ng k-coeffs: Ng x Ng possible pairs
        k_g_mix = np.zeros(Ng**2)
        weight_mix = np.zeros(Ng**2)
        # Mix k-coeffs of gases weighted by their relative VMR.
        for i in range(Ng):
            for j in range(Ng):
                #equation 9 Amundsen 2017 (equation 20 Mollier 2015)
                k_g_mix[i*Ng+j] = (k_g1[i]*VMR1+k_g2[j]*VMR2)/VMR_combined
                 #equation 10 Amundsen 2017
                weight_mix[i*Ng+j] = del_g[i]*del_g[j]

        # Resort-rebin procedure: Sort new "mixed" k-coeff's from low to high
        # see Amundsen et al. 2016 or section B.2.1 in Molliere et al. 2015
        ascending_index = np.argsort(k_g_mix)
        k_g_mix_sorted = k_g_mix[ascending_index]
        weight_mix_sorted = weight_mix[ascending_index]
        #combining w/weights--see description on Molliere et al. 2015
        sum_weight = np.cumsum(weight_mix_sorted)
        x = sum_weight/np.max(sum_weight)*2.-1
        for i in range(Ng):
            loc = np.where(x >=  g_ord[i])[0][0]
            k_g_combined[i] = k_g_mix_sorted[loc]

    return k_g_combined, VMR_combined

def mix_multi_gas_k(k_gas_g, del_g, VMR):
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
    g_ord = np.array([0.0034357 , 0.01801404, 0.04388279, 0.08044151, 0.126834  ,
    0.1819732 , 0.2445665 , 0.3131469 , 0.3861071 , 0.4617367 ,
    0.5382633 , 0.6138929 , 0.6868531 , 0.7554335 , 0.8180268 ,
    0.873166  , 0.9195585 , 0.9561172 , 0.981986  , 0.9965643 ])
    ngas = k_gas_g.shape[0]
    k_g_combined,VMR_combined = k_gas_g[0,:],VMR[0]
    #mixing in rest of gases inside a loop
    for j in range(1,ngas):
        k_g_combined,VMR_combined\
            = mix_two_gas_k(k_g_combined,k_gas_g[j,:],VMR_combined,VMR[j],del_g)
    return k_g_combined, VMR_combined