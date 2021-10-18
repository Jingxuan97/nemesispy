#!/usr/local/bin/python3
# -*- coding: utf-8 -*-
"""Routines to split an atmosphere into layers and calculate the
average layer properties along the observing path.
"""
import numpy as np
from scipy.integrate import simps
from scipy.interpolate import interp1d
from constants import K_B
from utils import calc_mmw

def interp(x_data, y_data, x_input, interp_type=1):
    """
    1D interpolation using scipy.interpolate.interp1d.

    Parameters
    ----------
    x_data : ndarray
        Independent variable data.

    y_data : ndarray
        Dependent variable data.

    x_input : real
        Input independent variable.

    interp_type : int
        1=linear interpolation
        2=quadratic spline interpolation
        3=cubic spline interpolation

    Returns
    -------
    y_output : real
        Output dependent variable.
    """
    if interp_type == 1:
        f = interp1d(x_data, y_data, kind='linear', fill_value='extrapolate')
        y_output = f(x_input)
    elif interp_type == 2:
        f = interp1d(x_data, y_data, kind='quadratic', fill_value='extrapolate')
        y_output = f(x_input)
    elif interp_type == 3:
        f = interp1d(x_data, y_data, kind='cubic', fill_value='extrapolate')
        y_output = f(x_input)
    return y_output

def split(H_atm, P_atm, Nlayer, layer_type=1, bottom_height=0.0, interp_type=1,
          path_angle=0.0, radius=None, H_base=None, P_base=None):
    """
    Splits an atmospheric model into layers by specifying layer base altitudes.

    Parameters
    ----------
    H_atm : ndarray
        Altitudes at which the atmospheric model is specified.
        (At altitude H_atm[i] the pressure is P_atm[i].)
    P_atm : ndarray
        Pressures at which the atmospheric model is specified.
    Nlayer : int
        Number of layers into which the atmosphere is split.
    layer_type : int, default 1
        Integer specifying how to split up the layers.
        0 = by equal changes in pressure
        1 = by equal changes in log pressure
        2 = by equal changes in height
        3 = by equal changes in path length at zenith
        4 = layer base pressure levels specified by P_base
        5 = layer base height levels specified by H_base
        Note 4 and 5 force Nlayer = len(P_base) or len(H_base).
    bottom_height : real, default 0.0
        Altitude of the base of the lowest layer.
    interp_type : int, default 1
        Interger specifying interpolation scheme.
        1=linear, 2=quadratic spline, 3=cubic spline.
    path_angle : real, default None
        Required only for layer type 3.
        Zenith angle in degrees defined at the base of the lowest layer.
    radius : real
        Required only for layer type 3.
        Reference planetary radius in m where H=0.  Usually at surface for
        terrestrial planets, or at 1 bar pressure level for gas giants.
    H_base : ndarray, default None
        Required only for layer type 5.
        Altitudes of the layer bases defined by user.
    P_base : ndarray, default None
        Required only for layer type 4.
        Pressures of the layer bases defined by user.

    Returns
    -------
    H_base : ndarray
        Heights of the layer bases.
    P_base : ndarray
        Pressures of the layer bases.

    Notes
    -----
    When used by itself, pressure and length units can be arbitrarily chosen.
    To ensure smooth integration with other functions, use SI units.
    """
    assert (bottom_height>=H_atm[0]) and (bottom_height<H_atm[-1]) , \
        'Lowest layer base altitude not contained in atmospheric profile'
    if layer_type == 0: # split by equal pressure intervals
        # interpolate for the pressure at base of lowest layer
        bottom_pressure = interp(H_atm,P_atm,bottom_height,interp_type)
        P_base = np.linspace(bottom_pressure,P_atm[-1],Nlayer+1)[:-1]
        H_base = interp(P_atm,H_atm,P_base,interp_type)

    elif layer_type == 1: # split by equal log pressure intervals
        # interpolate for the pressure at base of lowest layer
        bottom_pressure = interp(H_atm,P_atm,bottom_height,interp_type)
        P_base = np.logspace(np.log10(bottom_pressure),np.log10(P_atm[-1]),Nlayer+1)[:-1]
        H_base = interp(P_atm,H_atm,P_base,interp_type)

    elif layer_type == 2: # split by equal height intervals
        H_base = np.linspace(H_atm[0]+bottom_height, H_atm[-1], Nlayer+1)[:-1]
        P_base = interp(H_atm,P_atm,H_base,interp_type)

    elif layer_type == 3: # split by equal line-of-sight path intervals
        assert path_angle<=90 and path_angle>=0,\
            'Zennith angle should be in range [0,90] degree'
        sin = np.sin(path_angle*np.pi/180) # sin(path_angle angle)
        cos = np.cos(path_angle*np.pi/180) # cos(path_angle angle)
        r0 = radius+bottom_height # radial distance to lowest layer's base
        rmax = radius+H_atm[-1] # radial distance maximum height
        S_max = np.sqrt(rmax**2-(r0*sin)**2)-r0*cos # total path length
        S_base = np.linspace(0, S_max, Nlayer+1)[:-1]
        H_base = np.sqrt(S_base**2+r0**2+2*S_base*r0*cos)-radius
        logP_base = interp(H_atm,np.log(P_atm),H_base,interp_type)
        P_base = np.exp(logP_base)

    elif layer_type == 4: # split by specifying input base pressures
        assert np.all(P_base!=None),'Need input layer base pressures'
        assert  (P_base[-1] > P_atm[-1]) and (P_base[0] <= P_atm[0]), \
            'Input layer base pressures out of range of atmosphere profile'
        Nlayer = len(P_base)
        H_base = interp(P_atm,H_atm,P_base,interp_type)

    elif layer_type == 5: # split by specifying input base heights
        assert np.all(H_base!=None), 'Need input layer base heighs'
        assert (H_base[-1] < H_atm[-1]) and (H_base[0] >= H_atm[0]), \
            'Input layer base heights out of range of atmosphere profile'
        Nlayer = len(H_base)
        P_base = interp(H_atm,P_atm,H_base,interp_type)
    else:
        raise Exception('Layering scheme not defined')
    return H_base, P_base

def average(radius, H_atm, P_atm, T_atm, VMR_atm, ID, H_base, path_angle=0.0,
            integration_type=1, bottom_height=0.0, Nsimps=101):
    """
    Calculates average layer properties.

    Inputs
    ------
    radius : real
        Reference planetary radius where H_atm=0.  Usually at surface for
        terrestrial planets, or at 1 bar pressure level for gas giants.
    H_atm : ndarray
        Input profile heights
    P_atm : ndarray
        Input profile pressures
    T_atm : ndarray
        Input profile temperatures
    ID : ndarray
        Gas identifiers.
    VMR_atm : ndarray
        VMR_atm[i,j] is the Volume Mixing Ratio of gas j at profile point i.
        The jth column corresponds to the gas with RADTRANS ID ID[j].
    H_base : ndarray
        Heights of the layer bases.
    path_angle : real
        Zenith angle in degrees defined at bottom_height.
    integration_type : int
        Layer integration scheme
        0 = use properties at mid-path at each layer
        1 = use absorber amount weighted average values
    bottom_height : real
        Height of the base of the lowest layer. Default 0.0.
    Nsimps : int
        Number of Simpson's integration points to be used if integration_type=1.

    Returns
    -------
    H_layer : ndarray
        Representative height for each layer
    P_layer : ndarray
        Representative pressure for each layer
    T_layer : ndarray
        Representative pressure for each layer
    VMR_layer : ndarray
        Representative partial pressure for each gas at each layer.
        VMR_layer[i,j] is the representative partial pressure of gas j in layer i.
    U_layer : ndarray
        Total gaseous absorber amounts along the line-of-sight path, i.e.
        total number of gas molecules per unit area.
    Gas_layer : ndarray
        Representative absorber amounts of each gas at each layer.
        Gas_layer[i,j] is the representative number of gas j molecules
        in layer i in the form of number of molecules per unit area.
    scale : ndarray
        Layer scaling factor, i.e. ratio of path length through each layer
        to the layer thickness.

    Notes
    -----
    Assume SI units.
    Need VMR to be two dimensional.
    """
    # Calculate layer geometric properties
    Nlayer = len(H_base)
    del_H = np.concatenate(((H_base[1:]-H_base[:-1]),[H_atm[-1]-H_base[-1]]))
    sin = np.sin(path_angle*np.pi/180) # sin(viewing angle)
    cos = np.cos(path_angle*np.pi/180) # cos(viewing angle)
    r0 = radius+bottom_height # minimum radial distance
    rmax = radius+H_atm[-1] # maximum radial distance
    S_max = np.sqrt(rmax**2-(r0*sin)**2)-r0*cos # total path length
    S_base = np.sqrt((radius+H_base)**2-(r0*sin)**2)-r0*cos # path lengths at base of layer
    del_S = np.concatenate(((S_base[1:]-S_base[:-1]),[S_max-S_base[-1]]))
    scale = del_S/del_H # Layer Scaling Factor

    # initiate output arrays
    Ngas = len(VMR_atm[0])
    H_layer = np.zeros(Nlayer) # average layer height
    P_layer = np.zeros(Nlayer) # average layer pressure
    T_layer = np.zeros(Nlayer) # average layer temperature
    U_layer = np.zeros(Nlayer) # total no. of gas molecules per unit aera
    dU_dS = np.zeros(Nlayer) # no. of gas molecules per area per distance
    Gas_layer = np.zeros((Nlayer, Ngas)) # no. of molecules per aera
    VMR_layer = np.zeros((Nlayer, Ngas)) # partial pressures
    MMW_layer = np.zeros(Nlayer) # mean molecular weight

    # Calculate average properties depending on intergration type
    if integration_type == 0:
        # use layer properties at half path length in each layer
        S = np.zeros(Nlayer)
        S[:-1] = (S_base[:-1]+S_base[1:])/2
        S[-1] = (S_base[-1]+S_max)/2
        # Derive other properties from path length S
        H_layer = np.sqrt(S**2+r0**2+2*S*r0*cos) - radius
        P_layer = interp(H_atm,P_atm,H_layer)
        T_layer = interp(H_atm,T_atm,H_layer)
        # Ideal gas law: Number/(Area*Path_length) = P_atm/(K_B*T_atm)
        dU_dS = P_layer/(K_B*T_layer)
        U_layer = dU_dS*del_S
        # Use the volume mixing ratio information
        VMR_layer = np.zeros((Nlayer, Ngas))
        for igas in range(Ngas):
            VMR_layer[:,igas] = interp(H_atm, VMR_atm[:,igas], H_layer)
        Gas_layer = (VMR_layer.T * U_layer).T
        for ilayer in range(Nlayer):
            MMW_layer[ilayer] = calc_mmw(ID, VMR_layer[ilayer])

    elif integration_type == 1:
        # use absorber-amount-weighted averages calculated with Simpsons rule
        for ilayer in range(Nlayer):
            S0 = S_base[ilayer]
            if ilayer < Nlayer-1:
                S1 = S_base[ilayer+1]
            else:
                S1 = S_max
            # sub-divide each layer into Nsimps layers for integration
            S_int = np.linspace(S0, S1, Nsimps)
            H_int = np.sqrt(S_int**2+r0**2+2*S_int*r0*cos)-radius
            P_int = interp(H_atm,P_atm,H_int)
            T_int = interp(H_atm,T_atm,H_int)
            dU_dS_int = P_int/(K_B*T_int)
            VMR_int = np.zeros((Nsimps, Ngas))
            MMW_int = np.zeros(Nsimps)

            # absorber amount weighted integrals
            U_layer[ilayer] = simps(dU_dS_int,S_int)
            H_layer[ilayer] = simps(H_int*dU_dS_int,S_int)/U_layer[ilayer]
            P_layer[ilayer] = simps(P_int*dU_dS_int,S_int)/U_layer[ilayer]
            T_layer[ilayer] = simps(T_int*dU_dS_int,S_int)/U_layer[ilayer]
            for J in range(Ngas):
                VMR_int[:,J] = interp(H_atm, VMR_atm[:,J], H_int)
                Gas_layer[ilayer,J] = simps(VMR_int[:,J]*dU_dS_int,S_int)
            # VMR_int = (Gas_int.T * P_int).T # gas partial pressures
            for J in range(Ngas):
                VMR_layer[ilayer, J] \
                    = simps(VMR_int[:,J]*dU_dS_int,S_int)/U_layer[ilayer]
            for K in range(Nsimps):
                MMW_int[K] = calc_mmw(ID, VMR_int[K,:])
            MMW_layer[ilayer] = simps(MMW_int*dU_dS_int,S_int)/U_layer[ilayer]

    # Scale back to vertical layers
    U_layer = U_layer / scale
    Gas_layer = (Gas_layer.T * scale**-1 ).T

    return H_layer,P_layer,T_layer,VMR_layer,U_layer,Gas_layer,scale,del_S
