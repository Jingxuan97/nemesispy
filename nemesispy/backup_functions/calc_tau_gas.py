# @jit(nopython=True)
def calc_tau_gas(k_gas_w_g_p_t, P_layer, T_layer, VMR_layer, U_layer,
    P_grid, T_grid, del_g):
    """
      Parameters
      ----------
    k_gas_w_g_p_t(NGAS,NWAVE,NG,NPRESSKTA,NTEMPKTA) : ndarray
        Raw k-coefficients.
        Has dimension: NWAVE x NG x NPRESSKTA x NTEMPKTA.
    P_layer(NLAYER) : ndarray
        Atmospheric pressure grid.
    T_layer(NLAYER) : ndarray
        Atmospheric temperature grid.
    VMR_layer(NLAYER,NGAS) : ndarray
        Array of volume mixing ratios for NGAS.
        Has dimensioin: NLAYER x NGAS
    U_layer : ndarray
        DESCRIPTION.
    P_grid(NPRESSKTA) : ndarray
        Pressure grid on which the k-coeff's are pre-computed.
    T_grid(NTEMPKTA) : ndarray
        Temperature grid on which the k-coeffs are pre-computed.
    del_g : ndarray
        DESCRIPTION.

    Returns
    -------
    tau_w_g_l(NWAVE,NG,NLAYER) : ndarray
        DESCRIPTION.
    """

    Scaled_U_layer = U_layer *1.0e-20 # absorber amounts (U_layer) is scaled by a factor 1e-20
    Scaled_U_layer *= 1.0e-4 # convert from absorbers per m^2 to per cm^2


    k_gas_w_g_l = interp_k(P_grid, T_grid, P_layer, T_layer, k_gas_w_g_p_t)
    Ngas, Nwave, Ng, Nlayer = k_gas_w_g_l.shape

    # Method 1
    tau_w_g_l = np.zeros((Nwave,Ng,Nlayer))
    for iwave in range (Nwave):
        k_gas_g_l = k_gas_w_g_l[:,iwave,:,:]
        k_g_l = np.zeros((Ng,Nlayer))
        for ilayer in range(Nlayer):
            k_g_l[:,ilayer], VMR\
                = mix_multi_gas_k(k_gas_g_l[:,:,ilayer],VMR_layer[ilayer,:],del_g)
            tau_w_g_l[iwave,:,ilayer] = k_g_l[:,ilayer]*Scaled_U_layer[ilayer]*VMR
    # print('VMR',VMR)

    # # Method 2
    # tau_w_g_l,f_combined = new_k_overlap(k_gas_w_g_l,del_g,VMR_layer)

    return tau_w_g_l