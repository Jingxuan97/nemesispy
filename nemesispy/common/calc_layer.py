#!/usr/local/bin/python3
# -*- coding: utf-8 -*-
"""
Routines to split an atmosphere into layers and calculate the
average layer properties along an observing path.
"""
import numpy as np
from scipy.integrate import simps
# from numba import jit
from nemesispy.common.constants import K_B
from nemesispy.radtran.calc_mmw import calc_mmw

# @jit(nopython=True)
def split(H_model, P_model, NLAYER, layer_type=1, H_0=0.0,
    planet_radius=None, custom_path_angle=0.0,
    custom_H_base=np.array([0,0]), custom_P_base=np.array([0,0])):
    """
    Splits atmospheric models into layers. Returns layer base altitudes
    and layer base pressures.

    Parameters
    ----------
    H_model(NMODEL) : ndarray
        Altitudes at which the atmospheric models are defined.
        Unit: m
    P_model(NMODEL) : ndarray
        Pressures at which the atmospheric models arer defined.
        (At altitude H_model[i] the pressure is  P_model[i].)
        Unit: Pa
    NLAYER : int
        Number of layers to split the atmospheric models into.
    layer_type : int, optional
        Integer specifying how to split up the layers.
        0 = split by equal changes in pressure
        1 = split by equal changes in log pressure
        2 = split by equal changes in height
        3 = split by equal changes in path length
        4 = split by layer base pressure levels specified in P_base
        5 = split by layer base height levels specified in H_base
        Note 4 and 5 force NLAYER = len(P_base) or len(H_base).
        The default is 1.
    H_0 : real, optional
        Altitude of the lowest point in the atmospheric model.
        This is defined with respect to the reference planetary radius, i.e.
        the altitude at planet_radius is 0.
        The default is 0.0.
    planet_radius : real, optional
        Reference planetary planet_radius where H_model is set to be 0.  Usually
        set at surface for terrestrial planets, or at 1 bar pressure level for
        gas giants.
        Required only for layer type 3.
        The default is None.
    custom_path_angle : real, optional
        Required only for layer type 3.
        Zenith angle in degrees defined at the base of the lowest layer.
        The default is 0.0.
    custom_H_base(NLAYER) : ndarray, optional
        Required only for layer type 5.
        Altitudes of the layer bases defined by user.
        The default is None.
    custom_P_base(NLAYER) : ndarray, optional
        Required only for layer type 4.
        Pressures of the layer bases defined by user.
        The default is None.

    Returns
    -------
    H_base(NLAYER) : ndarray
        Heights of the layer bases.
    P_base(NLAYER) : ndarray
        Pressures of the layer bases.
    """
    assert (H_0>=H_model[0]) and (H_0<H_model[-1]) , \
        'Lowest layer base altitude not contained in atmospheric model'
    if layer_type == 0: # split by equal pressure intervals
        # interpolate for the pressure at base of lowest layer
        bottom_pressure = np.interp(H_0, H_model,P_model)
        P_base = np.linspace(bottom_pressure,P_model[-1],NLAYER+1)[:-1]
        # np.interp need x-coor to be increasing
        P_model = P_model[::-1]
        H_model = H_model[::-1]
        H_base = np.interp(P_base,P_model,H_model)

    elif layer_type == 1: # split by equal log pressure intervals
        # interpolate for the pressure at base of lowest layer
        bottom_pressure = np.interp(H_0,H_model,P_model)
        # P_base = np.logspace(np.log10(bottom_pressure),np.log10(P_model[-1]),\
        #     NLAYER+1)[:-1]

        P_base = 10**np.linspace(np.log10(bottom_pressure),np.log10(P_model[-1]),\
            NLAYER+1)[:-1]

        # np.interp need ascending x-coord
        P_model = P_model[::-1]
        H_model = H_model[::-1]
        H_base = np.interp(P_base,P_model,H_model)

    elif layer_type == 2: # split by equal height intervals
        H_base = np.linspace(H_model[0]+H_0, H_model[-1], NLAYER+1)[:-1]
        P_base = np.interp(H_base,H_model,P_model)

    elif layer_type == 3: # split by equal line-of-sight path intervals
        assert custom_path_angle<=90 and custom_path_angle>=0,\
            'Zennith angle should be in range [0,90] degree'
        sin = np.sin(custom_path_angle*np.pi/180) # sin(custom_path_angle angle)
        cos = np.cos(custom_path_angle*np.pi/180) # cos(custom_path_angle angle)
        r0 = planet_radius+H_0 # radial distance to lowest layer's base
        rmax = planet_radius+H_model[-1] # radial distance maximum height
        S_max = np.sqrt(rmax**2-(r0*sin)**2)-r0*cos # total path length
        S_base = np.linspace(0, S_max, NLAYER+1)[:-1]
        H_base = np.sqrt(S_base**2+r0**2+2*S_base*r0*cos)-planet_radius
        logP_base = np.interp(H_base,H_model,np.log(P_model))
        P_base = np.exp(logP_base)

    elif layer_type == 4: # split by specifying input base pressures
        # assert np.all(custom_P_base!=None),'Need input layer base pressures'
        assert  (custom_P_base[-1] > P_model[-1]) \
            and (custom_P_base[0] <= P_model[0]), \
            'Input layer base pressures out of range of atmosphere profile'
        NLAYER = len(custom_P_base)
        # np.interp need ascending x-coord
        P_model = P_model[::-1]
        H_model = H_model[::-1]
        H_base = np.interp(custom_P_base,P_model,H_model)

    elif layer_type == 5: # split by specifying input base heights
        # assert np.all(custom_H_base!=None), 'Need input layer base heighs'
        assert (custom_H_base[-1] < H_model[-1]) \
            and (custom_H_base[0] >= H_model[0]), \
            'Input layer base heights out of range of atmosphere profile'
        NLAYER = len(custom_H_base)
        P_base = np.interp(custom_H_base,H_model,P_model)
    else:
        raise Exception('Layering scheme not defined')
    return H_base, P_base

# @jit(nopython=True)
def average(planet_radius, H_model, P_model, T_model, VMR_model, ID, H_base,
        path_angle, H_0=0.0, NSIMPS=101):
    """
    Calculates absorber-amount-weighted average layer properties of an
    atmosphere.

    Inputs
    ------
    planet_radius : real
        Reference planetary planet_radius where H_model is set to be 0.  Usually
        set at surface for terrestrial planets, or at 1 bar pressure level for
        gas giants.
    H_model(NMODEL) : ndarray
        Altitudes at which the atmospheric model is defined.
        (At altitude H_model[i] the pressure is P_model[i].)
        Unit: m
    P_model(NMODEL) : ndarray
        Pressures at which the atmospheric model is defined.
        (At altitude P_model[i] the pressure is H_model[i].)
        Unit: Pa
    T_mode(NMODEL) : ndarray
        Temperature profile defined in the atmospheric model.
    ID : ndarray
        Gas identifiers.
    VMR_model(NMODEL,NGAS) : ndarray
        Volume mixing ratios of gases defined in the atmospheric model.
        VMR_model[i,j] is Volume Mixing Ratio of jth gas at ith profile point.
        The jth column corresponds to the gas with RADTRANS ID ID[j].
    H_base(NLAYER) : ndarray
        Heights of the layer bases.
    path_angle : real
        Zenith angle in degrees defined at H_0.
    H_0 : real, default 0.0
        Altitude of the lowest point in the atmospheric model.
        This is defined with respect to the reference planetary radius, i.e.
        the altitude at planet_radius is 0.
    NSIMPS : int, optional
        Number of Simpson's integration points to be used if integration_type=1.
        The default is 101.

    Returns
    -------
    H_layer(NLAYER) : ndarray
        Representative height for each layer
        Unit: m
    P_layer(NLAYER) : ndarray
        Representative pressure for each layer
        Unit: Pa
    T_layer(NLAYER) : ndarray
        Representative pressure for each layer
        Unit: Kelven
    U_layer(NLAYER) : ndarray
        Total gaseous absorber amounts along the line-of-sight path, i.e.
        total number of gas molecules per unit area.
        Unit: no of absorber per m^2
    VMR_layer(NLAYER, NGAS) : ndarray
        Representative partial pressure for each gas at each layer.
        VMR_layer[i,j] is representative partial pressure of gas j in layer i.
    Gas_layer(NLAYER, NGAS) : ndarray
        Representative absorber amounts of each gas at each layer.
        Gas_layer[i,j] is the representative number of gas j molecules
        in layer i in the form of number of molecules per unit area.
        Unit: no of absorber per m^2
    scale(NLAYER) : ndarray
        Layer scaling factor, i.e. ratio of path length through each layer
        to the layer thickness.
    del_S(NLAYER) : ndarray
        Path lengths.
        Unit: m

    Notes
    -----
    Assume SI units.
    """
    # Calculate layer geometric properties
    NLAYER = len(H_base)
    del_H = np.concatenate(((H_base[1:]-H_base[:-1]),[H_model[-1]-H_base[-1]]))
    sin = np.sin(path_angle*np.pi/180) # sin(viewing angle)
    cos = np.cos(path_angle*np.pi/180) # cos(viewing angle)
    r0 = planet_radius+H_0 # minimum radial distance
    rmax = planet_radius+H_model[-1] # maximum radial distance
    S_max = np.sqrt(rmax**2-(r0*sin)**2)-r0*cos # total path length
    S_base = np.sqrt((planet_radius+H_base)**2-(r0*sin)**2)-r0*cos # path lengths at base of layer
    del_S = np.concatenate(((S_base[1:]-S_base[:-1]),[S_max-S_base[-1]]))
    scale = del_S/del_H # Layer Scaling Factor

    # initiate output arrays
    Ngas = len(VMR_model[0])
    H_layer = np.zeros(NLAYER) # average layer height
    P_layer = np.zeros(NLAYER) # average layer pressure
    T_layer = np.zeros(NLAYER) # average layer temperature
    U_layer = np.zeros(NLAYER) # total no. of gas molecules per unit aera
    dU_dS = np.zeros(NLAYER) # no. of gas molecules per area per distance
    Gas_layer = np.zeros((NLAYER, Ngas)) # no. of molecules per aera
    VMR_layer = np.zeros((NLAYER, Ngas)) # partial pressures
    MMW_layer = np.zeros(NLAYER) # mean molecular weight

    # Calculate average properties depending on intergration type
    # use absorber-amount-weighted averages calculated with Simpsons rule
    for ilayer in range(NLAYER):
        S0 = S_base[ilayer]
        if ilayer < NLAYER-1:
            S1 = S_base[ilayer+1]
        else:
            S1 = S_max
        # sub-divide each layer into NSIMPS layers for integration
        S_int = np.linspace(S0, S1, NSIMPS)
        H_int = np.sqrt(S_int**2+r0**2+2*S_int*r0*cos)-planet_radius
        P_int = np.interp(H_int,H_model,P_model)
        T_int = np.interp(H_int,H_model,T_model)
        dU_dS_int = P_int/(K_B*T_int)
        VMR_int = np.zeros((NSIMPS, Ngas))
        MMW_int = np.zeros(NSIMPS)

        # absorber amount weighted integrals
        U_layer[ilayer] = simps(dU_dS_int,S_int)
        H_layer[ilayer] = simps(H_int*dU_dS_int,S_int)/U_layer[ilayer]
        P_layer[ilayer] = simps(P_int*dU_dS_int,S_int)/U_layer[ilayer]
        T_layer[ilayer] = simps(T_int*dU_dS_int,S_int)/U_layer[ilayer]
        for J in range(Ngas):
            VMR_int[:,J] = np.interp(H_int,H_model,VMR_model[:,J])
            Gas_layer[ilayer,J] = simps(VMR_int[:,J]*dU_dS_int,S_int)
        # VMR_int = (Gas_int.T * P_int).T # gas partial pressures
        for J in range(Ngas):
            VMR_layer[ilayer, J] \
                = simps(VMR_int[:,J]*dU_dS_int,S_int)/U_layer[ilayer]
        for K in range(NSIMPS):
            MMW_int[K] = calc_mmw(ID, VMR_int[K,:])
        MMW_layer[ilayer] = simps(MMW_int*dU_dS_int,S_int)/U_layer[ilayer]

    # Scale back to vertical layers
    U_layer = U_layer / scale
    Gas_layer = (Gas_layer.T * scale**-1 ).T

    return H_layer,P_layer,T_layer,VMR_layer,U_layer,Gas_layer,MMW_layer,\
        scale,del_S,del_H

# @jit(nopython=True)
def calc_layer(planet_radius, H_model, P_model, T_model, VMR_model, ID, NLAYER,
    path_angle, H_0=0.0, NSIMPS=101,
    layer_type=1, custom_path_angle=0.0,
    custom_H_base=None, custom_P_base=None):
    """
    Top level routine that calculates the layer properties from an atmospehric
    model.

    Parameters
    ----------
    planet_radius : real
        Reference planetary planet_radius where H_model is set to be 0.  Usually
        set at surface for terrestrial planets, or at 1 bar pressure level for
        gas giants.
    H_model(NMODEL) : ndarray
        Altitudes at which the atmospheric model is defined.
        (At altitude H_model[i] the pressure is P_model[i].)
        Unit: m
    P_model(NMODEL) : ndarray
        Pressures at which the atmospheric model is defined.
        (At altitude P_model[i] the pressure is H_model[i].)
        Unit: Pa
    T_mode(NMODEL) : ndarray
        Temperature profile defined in the atmospheric model.
    VMR_model : ndarray
        Volume mixing ratios of gases defined in the atmospheric model.
        VMR_model[i,j] is the Volume Mixing Ratio of jth gas at ith profile point.
        The jth column corresponds to the gas with RADTRANS ID ID[j].
    ID : ndarray
        Gas identifiers.
    NLAYER : int
        Number of layers to split the atmospheric model into.
    path_angle : real
        Zenith angle in degrees defined at H_0.
    H_0 : real, default 0.0
        Altitude of the lowest point in the atmospheric model.
        This is defined with respect to the reference planetary radius, i.e.
        the altitude at planet_radius is 0.
    layer_type : int, default 1
        Integer specifying how to split up the layers.
        0 = split by equal changes in pressure
        1 = split by equal changes in log pressure
        2 = split by equal changes in height
        3 = split by equal changes in path length
        4 = split by layer base pressure levels specified in P_base
        5 = split by layer base height levels specified in H_base
        Note 4 and 5 force NLAYER = len(P_base) or len(H_base).
    NSIMPS : int, optional
        Number of Simpson's integration points to be used if integration_type=1.
        The default is 101.
    custom_path_angle : real, optional
        Required only for layer type 3.
        Zenith angle in degrees defined at the base of the lowest layer.
        The default is 0.0.
    custom_H_base(NLAYER) : ndarray, optional
        Required only for layer type 5.
        Altitudes of the layer bases defined by user.
        The default is None.
    custom_P_base(NLAYER) : ndarray, optional
        Required only for layer type 4.
        Pressures of the layer bases defined by user.
        The default is None.


    Returns
    -------
    H_layer(NLAYER) : ndarray
        Representative height for each layer
        Unit: m
    P_layer(NLAYER) : ndarray
        Representative pressure for each layer
        Unit: Pa
    T_layer(NLAYER) : ndarray
        Representative pressure for each layer
        Unit: Kelven
    U_layer(NLAYER) : ndarray
        Total gaseous absorber amounts along the line-of-sight path, i.e.
        total number of gas molecules per unit area.
        Unit: no of absorber per m^2
    VMR_layer(NLAYER, NGAS) : ndarray
        Representative partial pressure for each gas at each layer.
        VMR_layer[i,j] is representative partial pressure of gas j in layer i.
    Gas_layer(NLAYER, NGAS) : ndarray
        Representative absorber amounts of each gas at each layer.
        Gas_layer[i,j] is the representative number of gas j molecules
        in layer i in the form of number of molecules per unit area.
        Unit: no of absorber per m^2
    scale(NLAYER) : ndarray
        Layer scaling factor, i.e. ratio of path length through each layer
        to the layer thickness.
    del_S(NLAYER) : ndarray
        Path lengths.
        Unit: m
    """
    H_base, P_base = split(H_model, P_model, NLAYER, layer_type=layer_type,
        H_0=H_0, planet_radius=planet_radius,
        custom_path_angle=custom_path_angle, custom_H_base=custom_H_base,
        custom_P_base=custom_P_base)

    H_layer,P_layer,T_layer,VMR_layer,U_layer,Gas_layer,MMW_layer,\
        scale,del_S,del_H = average(planet_radius, H_model, P_model, T_model,
            VMR_model, ID, H_base, path_angle=path_angle,
            H_0=H_0, NSIMPS=NSIMPS)

    return H_layer,P_layer,T_layer,VMR_layer,U_layer,Gas_layer,MMW_layer,\
        scale,del_S,del_H