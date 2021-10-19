import numpy as np
from numba import jit
from constants import C_LIGHT, K_B, PLANCK
from ck import interp_k, mix_multi_gas_k
nopython = True

@jit(nopython=nopython)
def tau_gas(k_gas_g_l, U_layer, VMR_layer, g_ord, del_g):
    """
    Calculate the optical path at a particular wavelength.

    Parameters
    ----------
    k_gas_g_l : ndarray
        Array of k-coeffs for each gas at a given wavenumber for each layer.
        Has dimension: Ngas x Ng x Nlayer
    U_layer : ndarray
        Total number of gas particles in each layer.
    VMR_layer : ndarray
        Array of volume mixing ratios for Ngas.
    g_ord : ndarray
        g-ordinates, assumed same for all gases.
    del_g : ndarray
        Gauss quadrature weights for the g-ordinates, assumed same for all gases.

    Returns
    -------
    tau_g_l : ndarray
        Optical path. Has dimension: Ng x Nlayer.
    """
    Ngas, Ng, Nlayer = k_gas_g_l.shape
    k_g_l = np.zeros((Ng, Nlayer))
    tau_g_l = np.zeros((Ng, Nlayer))
    for ilayer in range(Nlayer):
        k_g_l[:,ilayer], vmr_mix\
            = mix_multi_gas_k(k_gas_g_l[:,:,ilayer],VMR_layer[ilayer,:],g_ord,del_g)
        tau_g_l[:,ilayer] = k_g_l[:,ilayer]*U_layer[ilayer]*vmr_mix
    #print(tau_g_l)
    return tau_g_l

@jit(nopython=nopython)
def blackbody_um(wl, T):
    """
    Calculate blackbody radiance in W cm-2 sr-1 um-1.

    Parameters
    ----------
    wl : real
        Wavelength in um.
    T : real
        Temperature in K.

    Returns
    -------
    radiance : real
        Radiance in W cm-2 sr-1 um-1.

    Notes
    -----
    PLANCK = 6.62607e-34 Js
    C_LIGHT = 299792458 ms-1
    K_B = 1.38065e-23 J K-1
    """
    h = PLANCK
    c = C_LIGHT
    k = K_B
    radiance = (2*h*c**2)/((wl*1e-6)**5)*(1/(np.exp((h*c)/((wl*1e-6)*k*T))-1))*1e-10
    return radiance

@jit(nopython=nopython)
def radiance(wave, U_layer, P_layer, T_layer, VMR_layer, k_gas_w_g_p_t,
            P_grid, T_grid, g_ord, del_g):
    """
    Calculate emission spectrum using the correlated-k method.

    Parameters
    ----------
    wave : ndarray
        Wavelengths (um).
    U_layer : ndarray
        Total number of gas particles in each layer.
    P_layer : ndarray
        Atmospheric pressure grid.
    T_layer : ndarray
        Atmospheric temperature grid.
    VMR_layer : ndarray
        Array of volume mixing ratios for Ngas.
    k_gas_w_g_p_t : ndarray
        k-coefficients. Has dimension: Nwave x Ng x Npress x Ntemp.
    P_grid : ndarray
        Pressure grid on which the k-coeff's are pre-computed.
    T_grid : ndarray
        Temperature grid on which the k-coeffs are pre-computed.
    g_ord : ndarray
        g-ordinates of the k-table.
    del_g : ndarray
        Quadrature weights of the g-ordinates.

    Returns
    -------
    radiance : ndarray
        Output radiance (W cm-2 um-1 sr-1)
    """
    k_gas_w_g_l = interp_k(P_grid, T_grid, P_layer, T_layer, k_gas_w_g_p_t)
    Ng = len(g_ord)
    radiance = np.zeros(len(wave))
    for iwave,wl in enumerate(wave):
        k_gas_g_l = k_gas_w_g_l[:,iwave,:,:]
        tau_g_l = tau_gas(k_gas_g_l, U_layer, VMR_layer, g_ord, del_g)
        bb = blackbody_um(wl,T_layer)
        # top layer first, bottom layer last
        bb = bb[::-1]
        # print(bb)
        radiance_g = np.zeros(Ng)
        for ig in range (Ng):
            """
            tr = np.exp(-np.cumsum(tau_g_l[ig,:]))
            tr = np.concatenate((np.array([1]),tr))
            # print(tr)
            del_tr = tr[:-1] - tr[1:]
            del_tr[0] = 0
            radiance_g[ig] = np.sum(bb*del_tr)
            """
            t_g = tau_g_l[ig,:]
            # top layer first, bottom layer last
            t_g = t_g[::-1]
            # transmission
            tr = np.exp(-np.cumsum(t_g))
            tr = np.concatenate((np.array([1]),tr))
            del_tr = tr[:-1] - tr[1:]
            radiance_g[ig] = np.sum(bb*del_tr)
            #print(del_tr)
            """
            tr = np.exp(-np.cumsum(tau_g_l[ig,::-1]))
            tr = np.concatenate((np.array([1]),tr))
            del_tr = tr[:-1] - tr[1:]
            radiance_g[ig] = np.sum(bb[::-1]*del_tr)
            """
            """
            if del_tr[-1]>1e-5:
                print('-1',wl,ig,del_tr[-1])
            if del_tr[0]>1e-5:
                print('0',wl,ig,del_tr[0])
            """
        radiance[iwave] = np.sum(radiance_g*del_g)
    return radiance
