import numpy as np
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
        k_g_combined[:] = k_g2[:]*VMR2/VMR_combined
    elif k_g2[-1] * VMR2 <= cut_off:
        k_g_combined[:] = k_g1[:]*VMR1/VMR_combined
    else:
        # Overlap Ng k-coeffs with Ng k-coeffs randomly: Ng x Ng possible pairs
        nloop = Ng**2
        weight_mix = np.zeros((nloop))
        k_g_mix = np.zeros((nloop))
        ix = 0
        # Mix k-coeffs of gases weighted by their relative VMR.
        for i in range(Ng):
            for j in range(Ng):
                # # equation 9 Amundsen 2017 (equation 20 Mollier 2015)
                # k_g_mix[i*Ng+j] = (k_g1[i]*VMR1+k_g2[j]*VMR2)/VMR_combined
                # # equation 10 Amundsen 2017
                # weight_mix[i*Ng+j] = del_g[i]*del_g[j]

                weight_mix[ix] = del_g[i] * del_g[j]
                k_g_mix[ix] = (k_g1[i]*VMR1 + k_g2[j]*VMR2)/VMR_combined
                ix += 1

        # getting the cumulative g ordinate
        g_ord = np.zeros(Ng+1)
        g_ord[0] = 0.0
        for ig in range(Ng):
            g_ord[ig+1] = g_ord[ig]+del_g[ig]
        # print('g_ord1',g_ord)
        # g_ord = np.cumsum(del_g)
        # g_ord = np.append([0],g_ord)
        # print('g_ord2',g_ord)
        if g_ord[Ng] < 1.0:
            g_ord[Ng] = 1.0

        # Resort-rebin procedure: Sort new "mixed" k-coeff's from low to high
        # see Amundsen et al. 2016 or section B.2.1 in Molliere et al. 2015
        # check sorting
        ascending_index = np.argsort(k_g_mix)
        k_g_mix_sorted = k_g_mix[ascending_index]

        # k_g_mix_sorted, ascending_index = sort2g(k_g_mix)
        weight_mix_sorted = weight_mix[ascending_index]

        gdist = np.zeros(nloop)
        gdist[0] = weight_mix_sorted[0]
        for iloop in range(nloop-1):
            ix = iloop+1
            gdist[ix] = weight_mix_sorted[ix]+ gdist[iloop]

        ig = 0
        sum1 = 0.0
        for iloop in range(nloop):

            if gdist[iloop]<g_ord[ig+1] and (ig <= (Ng-1)):
                k_g_combined[ig] += k_g_mix_sorted[iloop]*weight_mix[iloop]
                sum1 = sum1 + weight_mix_sorted[iloop]
            else:
                frac = (g_ord[ig+1]-gdist[iloop-1])/(gdist[iloop]-gdist[iloop-1])
                k_g_combined[ig] += np.float32(frac)*k_g_mix_sorted[iloop]*weight_mix[iloop]
                sum1 = sum1 + weight_mix_sorted[iloop]
                k_g_combined[ig] = k_g_combined[ig] / np.float32(sum1)
                ig += 1
                if ig<=Ng-1:
                    sum1 = np.float32((1.-frac))*weight_mix_sorted[iloop]
                    k_g_combined[ig] \
                        += np.float32((1-frac))*k_g_mix_sorted[iloop]*weight_mix[iloop]
        print('ig')
        print(ig)
        print('Ng')
        print(Ng)
        if ig == Ng-1:
            k_g_combined[ig] = k_g_combined[ig]/np.float32(sum1)

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
      k_gas_g[NGAS,NG] : ndarray
          array of k-coeffs for each gas at a given wavenumber and pressure level.
          Has dimension: Ngas x Ng.
      VMR[NGAS] : ndarray
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
