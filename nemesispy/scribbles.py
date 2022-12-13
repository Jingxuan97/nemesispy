    # # surface/bottom layer contribution
    # p1 = P_layer[int(len(P_layer)/2-1)] #midpoint
    # p2 = P_layer[-1] # lowest point in altitude/highest in pressure

    # surface = None
    # if p2 > p1: # i.e. if not a limb path
    #     if not surface:
    #         radground = calc_planck(wave_grid,T_layer[-1])

    #     for ig in range(NG):
    #         spec_w_g[:,ig] = spec_w_g[:,ig] + tr_old_w_g[:,ig]*radground*xfac

# @jit(nopython=True)
# def interp_k_old(P_grid, T_grid, P_layer, T_layer, k_gas_w_g_p_t, n_active=4):
#     """
#     Interpolate the k coeffcients at given atmospheric presures and temperatures
#     using k-tables.

#     Parameters
#     ----------
#     P_grid(NPRESSKTA) : ndarray
#         Pressure grid of the k-tables.
#         Unit: Pa
#     T_grid(NTEMPKTA) : ndarray
#         Temperature grid of the ktables.
#         Unit: Kelvin
#     P_layer(NLAYER) : ndarray
#         Atmospheric pressure grid.
#         Unit: Pa
#     T_layer(NLAYER) : ndarray
#         Atmospheric temperature grid.
#         Unit: Kelvin
#     k_gas_w_g_p_t(NGAS,NWAVEKTA,NG,NPRESSKTA,NTEMPKTA) : ndarray
#         Array storing the k-coefficients.

#     Returns
#     -------
#     k_gas_w_g_l(NGAS,NWAVEKTA,NG,NLAYER) : ndarray
#         The interpolated-to-atmosphere k-coefficients.
#         Has dimension: NGAS x NWAVE x NG x NLAYER.
#     Notes
#     -----
#     Code breaks if P_layer/T_layer is out of the range of P_grid/T_grid.
#     Mainly need to worry about max(T_layer)>max(T_grid).
#     No extrapolation outside of the TP grid of ktable.
#     """
#     NGAS, NWAVE, NG, NPRESS, NTEMP = k_gas_w_g_p_t.shape
#     NLAYER = len(P_layer)
#     k_gas_w_g_l = np.zeros((NGAS,NWAVE,NG,NLAYER))

#     # Interpolate the k values at the layer temperature and pressure
#     for ilayer in range(NLAYER):
#         p = P_layer[ilayer]
#         t = T_layer[ilayer]

#         # Find pressure grid points above and below current layer pressure
#         ip = np.abs(P_grid-p).argmin()
#         if P_grid[ip] >= p:
#             ip_high = ip
#             if ip == 0:
#                 p = P_grid[0]
#                 ip_low = 0
#                 ip_high = 1
#             else:
#                 ip_low = ip-1
#         elif P_grid[ip]<p:
#             ip_low = ip
#             if ip == NPRESS-1:
#                 p = P_grid[NPRESS-1]
#                 ip_high = NPRESS-1
#                 ip_low = NPRESS-2
#             else:
#                 ip_high = ip + 1

#         # Find temperature grid points above and below current layer temperature
#         it = np.abs(T_grid-t).argmin()
#         if T_grid[it] >= t:
#             it_high = it
#             if it == 0:
#                 t = T_grid[0]
#                 it_low = 0
#                 it_high = 1
#             else:
#                 it_low = it -1
#         elif T_grid[it] < t:
#             it_low = it
#             if it == NTEMP-1:
#                 t = T_grid[-1]
#                 it_high = NTEMP - 1
#                 it_low = NTEMP -2
#             else:
#                 it_high = it + 1

#         # Set up arrays for interpolation
#         lnp = np.log(p)
#         lnp_low = np.log(P_grid[ip_low])
#         lnp_high = np.log(P_grid[ip_high])
#         t_low = T_grid[it_low]
#         t_high = T_grid[it_high]

#         # Bilinear interpolation
#         f11 = k_gas_w_g_p_t[:,:,:,ip_low,it_low]
#         f12 = k_gas_w_g_p_t[:,:,:,ip_low,it_high]
#         f21 = k_gas_w_g_p_t[:,:,:,ip_high,it_high]
#         f22 = k_gas_w_g_p_t[:,:,:,ip_high,it_low]
#         v = (lnp-lnp_low)/(lnp_high-lnp_low)
#         u = (t-t_low)/(t_high-t_low)
#         for igas in range(NGAS):
#             for iwave in range(NWAVE):
#                 for ig in range(NG):
#                     k_gas_w_g_l[igas,iwave,ig,ilayer] \
#                         = (1.0-v)*(1.0-u)*f11[igas,iwave,ig] \
#                         + v*(1.0-u)*f22[igas,iwave,ig] \
#                         + v*u*f21[igas,iwave,ig] \
#                         + (1.0-v)*u*f12[igas,iwave,ig]

#     return k_gas_w_g_l

@jit(nopython=True)
def interp_k(P_grid, T_grid, P_layer, T_layer, k_gas_w_g_p_t):
    """
    Interpolate the k coeffcients at given atmospheric presures and temperatures
    using k-tables.

    Parameters
    ----------
    P_grid(NPRESSKTA) : ndarray
        Pressure grid of the k-tables.
        Unit: Pa
    T_grid(NTEMPKTA) : ndarray
        Temperature grid of the ktables.
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
    k_gas_w_g_l(NGAS,NWAVEKTA,NG,NLAYER) : ndarray
        The interpolated-to-atmosphere k-coefficients.
        Has dimension: NGAS x NWAVE x NG x NLAYER.
    Notes
    -----
    Code breaks if P_layer/T_layer is out of the range of P_grid/T_grid.
    Mainly need to worry about max(T_layer)>max(T_grid).
    No extrapolation outside of the TP grid of ktable.
    """
    NGAS, NWAVE, NG, NPRESS, NTEMP = k_gas_w_g_p_t.shape
    NLAYER = len(P_layer)
    k_gas_w_g_l = np.zeros((NGAS,NWAVE,NG,NLAYER))
    kgood = np.zeros((NGAS,NWAVE,NG,NLAYER))

    # Interpolate the k values at the layer temperature and pressure
    for ilayer in range(NLAYER):
        p = P_layer[ilayer]
        t = T_layer[ilayer]

        # Find pressure grid points above and below current layer pressure
        ip = np.abs(P_grid-p).argmin()
        if P_grid[ip] >= p:
            ip_high = ip
            if ip == 0:
                p = P_grid[0]
                ip_low = 0
                ip_high = 1
            else:
                ip_low = ip-1
        elif P_grid[ip]<p:
            ip_low = ip
            if ip == NPRESS-1:
                p = P_grid[NPRESS-1]
                ip_high = NPRESS-1
                ip_low = NPRESS-2
            else:
                ip_high = ip + 1

        # Find temperature grid points above and below current layer temperature
        it = np.abs(T_grid-t).argmin()
        if T_grid[it] >= t:
            it_high = it
            if it == 0:
                t = T_grid[0]
                it_low = 0
                it_high = 1
            else:
                it_low = it -1
        elif T_grid[it] < t:
            it_low = it
            if it == NTEMP-1:
                t = T_grid[-1]
                it_high = NTEMP - 1
                it_low = NTEMP -2
            else:
                it_high = it + 1

        # Set up arrays for interpolation
        lnp = np.log(p)
        lnp_low = np.log(P_grid[ip_low])
        lnp_high = np.log(P_grid[ip_high])
        t_low = T_grid[it_low]
        t_high = T_grid[it_high]

        # NGAS,NWAVE,NG
        f11 = k_gas_w_g_p_t[:,:,:,ip_low,it_low]
        f12 = k_gas_w_g_p_t[:,:,:,ip_low,it_high]
        f21 = k_gas_w_g_p_t[:,:,:,ip_high,it_high]
        f22 = k_gas_w_g_p_t[:,:,:,ip_high,it_low]

        # Bilinear interpolation
        v = (lnp-lnp_low)/(lnp_high-lnp_low)
        u = (t-t_low)/(t_high-t_low)

        # igood = np.where( (f11>0.0) & (f12>0.0) & (f22>0.0) & (f21>0.0) )
        # ibad = np.where( (f11<=0.0) & (f12<=0.0) & (f22<=0.0) & (f21<=0.0) )

        # # print(f11)
        # print('ibad',ibad)
        # print('f11',f11[ibad])
        # print('f11[ibad[0]]',f11[ibad[0]])
        # print('ibad[0]',ibad[0])
        # for i in range(len(igood[0])):
        #     kgood[igood[0][i],igood[1][i],igood[2][i],ilayer] \
        #         = (1.0-v)*(1.0-u)*np.log(f11[igood[0][i],igood[1][i],igood[2][i]]) \
        #         + v*(1.0-u)*np.log(f22[igood[0][i],igood[1][i],igood[2][i]]) \
        #         + v*u*np.log(f21[igood[0][i],igood[1][i],igood[2][i]]) \
        #         + (1.0-v)*u*np.log(f12[igood[0][i],igood[1][i],igood[2][i]])
        #     kgood[igood[0][i],igood[1][i],igood[2][i],ilayer] \
        #         = np.exp(kgood[igood[0][i],igood[1][i],igood[2][i],ilayer])

        # for i in range(len(ibad[0])):
        #     kgood[ibad[0][i],ibad[1][i],ibad[2][i],ilayer] \
        #         = (1.0-v)*(1.0-u)*f11[ibad[0][i],ibad[1][i],ibad[2][i]] \
        #         + v*(1.0-u)*f22[ibad[0][i],ibad[1][i],ibad[2][i]] \
        #         + v*u*f21[ibad[0][i],ibad[1][i],ibad[2][i]] \
        #         + (1.0-v)*u*f12[ibad[0][i],ibad[1][i],ibad[2][i]]

        for igas in range(NGAS):
            for iwave in range(NWAVE):
                for ig in range(NG):
                    kgood[igas,iwave,ig,ilayer] \
                        = (1.0-v)*(1.0-u)*f11[igas,iwave,ig] \
                        + v*(1.0-u)*f22[igas,iwave,ig] \
                        + v*u*f21[igas,iwave,ig] \
                        + (1.0-v)*u*f12[igas,iwave,ig]

    k_gas_w_g_l = kgood
    return k_gas_w_g_l

"""
import time
start = time.time()
for i in range(2000000):
    mmw = calc_mmw([1,2],VMR=[0.5,0.5],ISO=[1,1])
    # print('mmw',mmw)
end = time.time()
print('runtime1',end-start)

start = time.time()
for i in range(2000000):
    mmw = calc_mmw([1,2],VMR=[0.5,0.5],ISO=[0,0])
    # print('mmw',mmw)
end = time.time()
print('runtime2',end-start)

start = time.time()
for i in range(2000000):
    mmw = calc_mmw([1,2],VMR=[0.5,0.5])
    # print('mmw',mmw)
end = time.time()
print('runtime3',end-start)
"""

"""
import numpy as np
mmw = calc_mmw([1,2],VMR=[0.5,0.5],ISO=[1,1])
print('mmw',mmw)
mmw = calc_mmw([1,2],VMR=[0.5,0.5],ISO=[0,0])
print('mmw',mmw)
mmw = calc_mmw([1,2],VMR=[0.5,0.5])
print('mmw',mmw)

mmw = calc_mmw(np.array([1,2]),VMR=np.array([0.5,0.5]),ISO=np.array([1,1]))
print('mmw',mmw)
mmw = calc_mmw(np.array([1,2]),VMR=np.array([0.5,0.5]),ISO=np.array([0,0]))
print('mmw',mmw)
mmw = calc_mmw(np.array([1,2]),VMR=np.array([0.5,0.5]))
print('mmw',mmw)
"""

"""Test
# ID = [1,2,3]
# ISO = None
# VMR = [0.1,0.1,0.8]
# mmw = calc_mmw(ID,VMR,ISO)
ID = [1,2,3]
ISO = [1,1,1]
VMR = [0.1,0.1,0.8]
mmw = calc_mmw(ID,VMR,ISO)
ID = [1,2,3]
ISO = None
VMR = [0.1,0.1,0.8]
mmw = calc_mmw(ID,VMR,ISO)
"""

@jit(nopython=True)
def noverlapg(k_gas_g, amount, del_g):
    """
    Combine k distributions of multiple gases given their number densities.

    Parameters
    ----------
    k_gas_g(NGAS,NG) : ndarray
        K-distributions of the different gases.
        Each row contains a k-distribution defined at NG g-ordinates.
        Unit: cm^2 (per particle)
    amount(NGAS) : ndarray
        Absorber amount of each gas.
        Unit: (no. of partiicles) cm^-2
    del_g(NG) : ndarray
        Gauss quadrature weights for the g-ordinates.
        These are the widths of the bins in g-space.

    Returns
    -------
    k_g(NG) : ndarray
        Combined k-distribution.
        Unit: cm^2 (per particle)
    """

# @jit(nopython=True)
def split(H_model, P_model, NLAYER, H_0=0.0):

    # split by equal log pressure intervals
    # interpolate for the pressure at base of lowest layer
    bottom_pressure = np.interp(H_0,H_model,P_model)
    P_base = 10**np.linspace(np.log10(bottom_pressure),np.log10(P_model[-1]),\
        NLAYER+1)[:-1]

    # np.interp need ascending x-coord
    P_model = P_model[::-1]
    H_model = H_model[::-1]
    H_base = np.interp(P_base,P_model,H_model)
    return H_base, P_base


# def split(H_model, P_model, NLAYER, layer_type=1, H_0=0.0,
#     planet_radius=None, custom_path_angle=0.0,
#     custom_H_base=np.array([0,0]), custom_P_base=np.array([0,0])):
#     """
#     Splits atmospheric models into layers. Returns layer base altitudes
#     and layer base pressures.

#     Parameters
#     ----------
#     H_model(NMODEL) : ndarray
#         Altitudes at which the atmospheric models are defined.
#         Unit: m
#     P_model(NMODEL) : ndarray
#         Pressures at which the atmospheric models arer defined.
#         (At altitude H_model[i] the pressure is  P_model[i].)
#         Unit: Pa
#     NLAYER : int
#         Number of layers to split the atmospheric models into.
#     layer_type : int, optional
#         Integer specifying how to split up the layers.
#         0 = split by equal changes in pressure
#         1 = split by equal changes in log pressure
#         2 = split by equal changes in height
#         3 = split by equal changes in path length
#         4 = split by layer base pressure levels specified in P_base
#         5 = split by layer base height levels specified in H_base
#         Note 4 and 5 force NLAYER = len(P_base) or len(H_base).
#         The default is 1.
#     H_0 : real, optional
#         Altitude of the lowest point in the atmospheric model.
#         This is defined with respect to the reference planetary radius, i.e.
#         the altitude at planet_radius is 0.
#         The default is 0.0.
#     planet_radius : real, optional
#         Reference planetary planet_radius where H_model is set to be 0.  Usually
#         set at surface for terrestrial planets, or at 1 bar pressure level for
#         gas giants.
#         Required only for layer type 3.
#         The default is None.
#     custom_path_angle : real, optional
#         Required only for layer type 3.
#         Zenith angle in degrees defined at the base of the lowest layer.
#         The default is 0.0.
#     custom_H_base(NLAYER) : ndarray, optional
#         Required only for layer type 5.
#         Altitudes of the layer bases defined by user.
#         The default is None.
#     custom_P_base(NLAYER) : ndarray, optional
#         Required only for layer type 4.
#         Pressures of the layer bases defined by user.
#         The default is None.

#     Returns
#     -------
#     H_base(NLAYER) : ndarray
#         Heights of the layer bases.
#     P_base(NLAYER) : ndarray
#         Pressures of the layer bases.
#     """
#     assert (H_0>=H_model[0]) and (H_0<H_model[-1]) , \
#         'Lowest layer base altitude not contained in atmospheric model'
#     if layer_type == 0: # split by equal pressure intervals
#         # interpolate for the pressure at base of lowest layer
#         bottom_pressure = np.interp(H_0, H_model,P_model)
#         P_base = np.linspace(bottom_pressure,P_model[-1],NLAYER+1)[:-1]
#         # np.interp need x-coor to be increasing
#         P_model = P_model[::-1]
#         H_model = H_model[::-1]
#         H_base = np.interp(P_base,P_model,H_model)

#     elif layer_type == 1: # split by equal log pressure intervals
#         # interpolate for the pressure at base of lowest layer
#         bottom_pressure = np.interp(H_0,H_model,P_model)
#         # P_base = np.logspace(np.log10(bottom_pressure),np.log10(P_model[-1]),\
#         #     NLAYER+1)[:-1]

#         P_base = 10**np.linspace(np.log10(bottom_pressure),np.log10(P_model[-1]),\
#             NLAYER+1)[:-1]

#         # np.interp need ascending x-coord
#         P_model = P_model[::-1]
#         H_model = H_model[::-1]
#         H_base = np.interp(P_base,P_model,H_model)

#     elif layer_type == 2: # split by equal height intervals
#         H_base = np.linspace(H_model[0]+H_0, H_model[-1], NLAYER+1)[:-1]
#         P_base = np.interp(H_base,H_model,P_model)

#     elif layer_type == 3: # split by equal line-of-sight path intervals
#         assert custom_path_angle<=90 and custom_path_angle>=0,\
#             'Zennith angle should be in range [0,90] degree'
#         sin = np.sin(custom_path_angle*np.pi/180) # sin(custom_path_angle angle)
#         cos = np.cos(custom_path_angle*np.pi/180) # cos(custom_path_angle angle)
#         r0 = planet_radius+H_0 # radial distance to lowest layer's base
#         rmax = planet_radius+H_model[-1] # radial distance maximum height
#         S_max = np.sqrt(rmax**2-(r0*sin)**2)-r0*cos # total path length
#         S_base = np.linspace(0, S_max, NLAYER+1)[:-1]
#         H_base = np.sqrt(S_base**2+r0**2+2*S_base*r0*cos)-planet_radius
#         logP_base = np.interp(H_base,H_model,np.log(P_model))
#         P_base = np.exp(logP_base)

#     elif layer_type == 4: # split by specifying input base pressures
#         # assert np.all(custom_P_base!=None),'Need input layer base pressures'
#         assert  (custom_P_base[-1] > P_model[-1]) \
#             and (custom_P_base[0] <= P_model[0]), \
#             'Input layer base pressures out of range of atmosphere profile'
#         NLAYER = len(custom_P_base)
#         # np.interp need ascending x-coord
#         P_model = P_model[::-1]
#         H_model = H_model[::-1]
#         H_base = np.interp(custom_P_base,P_model,H_model)

#     elif layer_type == 5: # split by specifying input base heights
#         # assert np.all(custom_H_base!=None), 'Need input layer base heighs'
#         assert (custom_H_base[-1] < H_model[-1]) \
#             and (custom_H_base[0] >= H_model[0]), \
#             'Input layer base heights out of range of atmosphere profile'
#         NLAYER = len(custom_H_base)
#         P_base = np.interp(custom_H_base,H_model,P_model)
#     else:
#         raise Exception('Layering scheme not defined')
#     return H_base, P_base