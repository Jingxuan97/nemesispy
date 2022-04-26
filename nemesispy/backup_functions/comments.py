
"""
# Incompatible methods with numba jit

# spec_out = np.tensordot(spec_out, del_g, axes=([1],[0])) * xfac
# spec_out = spec_out.T[0]
"""

""" # Fortran straight transcription
### pray it works
bb = np.zeros((NWAVE,NLAYER))
spectrum = np.zeros((NWAVE))
spec_w_g = np.zeros((NWAVE,NG))
for iwave in range(NWAVE):
    for ig in range(NG):
        taud = 0.
        trold = 1.
        for ilayer in range(NLAYER):
            taud = taud + tau_total_w_g_l[iwave,ig,ilayer]
            tr = np.exp(-taud)
            # print('taud',taud)
            # print('tr',tr)
            if ig == 0:
                bb[iwave,ilayer] = calc_planck(wave_grid[iwave],T_layer[ilayer])
            # print('bb',bb)
            # print('np.float32((trold-tr))',np.float32((trold-tr)))
            # print('xfac',xfac)
            # print(xfac*np.float32((trold-tr)) * bb[iwave,ilayer])

            spec_w_g[iwave,ig] = spec_w_g[iwave,ig] \
                + xfac[iwave]*np.float32((trold-tr)) * bb[iwave,ilayer]
            trold = tr
        p1 = P_layer[int(len(P_layer)/2-1)] #midpoint
        p2 = P_layer[-1] # lowest point in altitude/highest in pressure
        surface = None
        if p2 > p1:
            radground = calc_planck(wave_grid[iwave],T_layer[-1])
            spec_w_g[iwave,ig] = spec_w_g[iwave,ig] \
                + xfac[iwave]*np.float32(trold)*radground

for iwave in range(NWAVE):
    for ig in range(NG):
        spectrum[iwave] += spec_w_g[iwave,ig] * del_g[ig]
"""
# Fortran nemesis
# def calc_tau_gas(k_gas_w_g_p_t, P_layer, T_layer, VMR_layer, U_layer,
#     P_grid, T_grid, del_g):
#     """
#     Calculate the optical path due to gaseous absorbers.

#     Parameters
#     ----------
#     k_gas_w_g_p_t(NGAS,NWAVE,NG,NPRESSKTA,NTEMPKTA) : ndarray
#         Raw k-coefficients.
#         Has dimension: NWAVE x NG x NPRESSKTA x NTEMPKTA.
#     P_layer(NLAYER) : ndarray
#         Atmospheric pressure grid.
#     T_layer(NLAYER) : ndarray
#         Atmospheric temperature grid.
#     VMR_layer(NLAYER,NGAS) : ndarray
#         Array of volume mixing ratios for NGAS.
#         Has dimensioin: NLAYER x NGAS
#     U_layer : ndarray
#         DESCRIPTION.
#     P_grid(NPRESSKTA) : ndarray
#         Pressure grid on which the k-coeff's are pre-computed.
#     T_grid(NTEMPKTA) : ndarray
#         Temperature grid on which the k-coeffs are pre-computed.
#     del_g : ndarray
#         DESCRIPTION.

#     Returns
#     -------
#     tau_w_g_l(NWAVE,NG,NLAYER) : ndarray
#         DESCRIPTION.
#     """

#     U_layer *= 1.0e-20 # absorber amounts (U_layer) is scaled by a factor 1e-20
#     U_layer *= 1.0e-4 # convert from absorbers per m^2 to per cm^2

#     k_gas_w_g_l = interp_k(P_grid, T_grid, P_layer, T_layer, k_gas_w_g_p_t) # NGAS,NWAVE,NG,NLAYER

#     k_w_g_l,f_combined = new_k_overlap(k_gas_w_g_l, del_g, VMR_layer) # NWAVE,NG,NLAYER

#     utotl = U_layer

#     tau_gas = k_w_g_l * utotl * f_combined  # NWAVE, NG, NLAYER

#     return tau_gas