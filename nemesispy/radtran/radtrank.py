import numpy as np
from numba import jit
from nemesispy.radtran.interpk import interp_k, new_k_overlap

def planck(wave,temp,ispace=1):
    """
    Calculate the blackbody radiation given by the Planck function

    Parameters
    ----------
    wave(nwave) : ndarray
        Wavelength or wavenumber array
    temp : real
        Temperature of the blackbody (K)
    ispace : int
        Flag indicating the spectral units
        (0) Wavenumber (cm-1)
        (1) Wavelength (um)

    Returns
    -------
	bb(nwave) : ndarray
        Planck function (W cm-2 sr-1 (cm-1)-1 or W cm-2 sr-1 um-1)
    """

    c1 = 1.1911e-12
    c2 = 1.439
    if ispace==0:
        y = wave
        a = c1 * (y**3.)
    elif ispace==1:
        y = 1.0e4/wave
        a = c1 * (y**5.) / 1.0e4
    else:
        raise Exception('error in planck: ISPACE must be either 0 or 1')
    tmp = c2 * y / temp
    b = np.exp(tmp) - 1
    bb = a/b
    return bb

def radtran(P_layer, T_layer):
    """
    Computes the spectrum
    """
    # Dimensioins
    NWAVE = None
    NLAY = None
    # Collision Induced Absorptioin Optical Path
    TAUCIA = np.zeros([NWAVE,NLAY])
    # Rayleigh Scattering Optical Path
    TAURAY = np.zeros([NWAVE,NLAY])
    # Dust Scattering Optical Path
    TAUDUST = np.zeros([NWAVE,NLAY])

    # Gaseous Opacity
    # Calculating the k-coefficients for each gas in each layer
    vmr_gas = np.zeros([NGAS, NLAY])

    utotl = np.zeros(NLAY)

    for i in range(NLAY):
        vmr_gas[i,:] = None　#Layer.PP[:,IGAS].T / Layer.PRESS #VMR of each radiatively active gas
        utotl[:] = None #utotl[:] + Layer.AMOUNT[:,IGAS].T * 1.0e-4 * 1.0e-20   #Vertical column density of the radiatively active gases

    k_gas_w_g_l = interp_k(P_grid, T_grid, P_layer, T_layer, k_gas_w_g_p_t)
    k_w_g_l = mix_multi_gas_k(k_gas_w_g_l, del_g, vmr_gas)
    TAUGAS = k_w_g_l * utotl

    TAUTOT = np.zeros(TAUGAS.shape)
    for ig in range(NG): # wavebin x layer
        TAUTOT[:,ig,:] = TAUGAS[:,ig,:] + TAUCIA[:,:] + TAUDUST[:,:] + TAURAY[:,:]

    #Scale to teh line-of-sight opacities
    ScalingFactor = None
    TAUTOT_LAYINC = TAUTOT * ScalingFactor

    # Thermal Emission Calculation




# """
# @jit(nopython=True)
# def tau_gas(k_gas_w_g_p_t, P_layer, T_layer, VMR_layer, U_layer,
#             P_grid, T_grid, g_ord, del_g):
#     """
#       Parameters
#       ----------
#       k_gas_w_g_p_t : ndarray
#           DESCRIPTION.
#       P_layer : ndarray
#           DESCRIPTION.
#       T_layer : ndarray
#           DESCRIPTION.
#       VMR_layer : ndarray
#           DESCRIPTION.
#       U_layer : ndarray
#           DESCRIPTION.
#       P_grid : ndarray
#           DESCRIPTION.
#       T_grid : ndarray
#           DESCRIPTION.
#       g_ord : ndarray
#           DESCRIPTION.
#       del_g : ndarray
#           DESCRIPTION.

#       Returns
#       -------
#       tau_w_g_l : ndarray
#           DESCRIPTION.
#     """
#     k_gas_w_g_l = interp_k(P_grid, T_grid, P_layer, T_layer, k_gas_w_g_p_t)
#     Ngas, Nwave, Ng, Nlayer = k_gas_w_g_l.shape
#     tau_w_g_l = np.zeros((Nwave,Ng,Nlayer))
#     for iwave in range (Nwave):
#         k_gas_g_l = k_gas_w_g_l[:,iwave,:,:]
#         k_g_l = np.zeros((Ng,Nlayer))
#         for ilayer in range(Nlayer):
#             k_g_l[:,ilayer], VMR\
#                 = new_k_overlap(k_gas_g_l[:,:,ilayer],VMR_layer[ilayer,:],g_ord,del_g)
#             tau_w_g_l[iwave,:,ilayer] = k_g_l[:,ilayer]*U_layer[ilayer]*VMR
#     return tau_w_g_l
# """
