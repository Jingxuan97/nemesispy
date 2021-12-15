import sys
sys.path.append('/Users/jingxuanyang/Desktop/Workspace/nemesispy2022/')
import numpy as np
from nemesispy.data.constants import R_SUN, R_JUP_E, AMU, AU, M_JUP
from nemesispy.radtran.models import Model2
from nemesispy.radtran.path import get_profiles # average
from nemesispy.radtran.k1read import read_kls
# from nemesispy.radtran.k2interp import interp_k, new_k_overlap
from nemesispy.radtran.k3radtran import radtran

### Required Inputs
# Planet/star parameters
T_star = 6000

M_plt = 1*M_JUP
SMA = 0.015*AU
R_star = 1*R_SUN
planet_radius = 1*R_JUP_E
R_plt = 1*R_JUP_E

"""
planet_radius : real
    Reference planetary radius where H_atm=0.  Usually at surface for
    terrestrial planets, or at 1 bar pressure level for gas giants.
"""
H_atm = np.array([])
"""

"""
P_atm = np.array([])
NProfile = 40
Nlayer = 15
P_range = np.geomspace(20,1e-3,NProfile)*1e5
mmw = 2*AMU

### params
kappa = 1e-3
gamma1 = 1e-1
gamma2 = 1e-1
alpha = 0.5
T_irr = 1000

atm = Model2(T_star, R_star, M_plt, R_plt, SMA, P_range, mmw,
                      kappa, gamma1, gamma2, alpha, T_irr)
H_atm = atm.height()
P_atm = atm.pressure()
T_atm = atm.temperature()
# print('H_atm',H_atm)
# print('P_atm',P_atm)
"""
H_atm : ndarray
    Input profile heights
P_atm : ndarray
    Input profile pressures
T_atm : ndarray
    Input profile temperatures
"""
ID = np.array([1,2,5,6,40,39])
ISO = np.array([0,0,0,0,0,0])
"""
ID : ndarray
    Gas identifiers.
"""
NVMR = len(ID)
VMR_atm = np.zeros((NProfile,NVMR))
VMR_H2O = np.ones(NProfile)*1e-6
VMR_CO2 = np.ones(NProfile)*1e-6
VMR_CO = np.ones(NProfile)*1e-6
VMR_CH4 = np.ones(NProfile)*1e-6
VMR_He = (np.ones(NProfile)-VMR_H2O-VMR_CO2-VMR_CO-VMR_CH4)*0.15
VMR_H2 = VMR_He/0.15*0.85
VMR_atm[:,0] = VMR_H2O
VMR_atm[:,1] = VMR_CO2
VMR_atm[:,2] = VMR_CO
VMR_atm[:,3] = VMR_CH4
VMR_atm[:,4] = VMR_He
VMR_atm[:,5] = VMR_H2


"""
VMR_atm : ndarray
    VMR_atm[i,j] is the Volume Mixing Ratio of gas j at profile point i.
    The jth column corresponds to the gas with RADTRANS ID ID[j].
"""
"""
H_base : ndarray
    Heights of the layer bases.
"""

lowres_files = ['/Users/jingxuanyang/Desktop/Workspace/nemesispy2022/nemesispy/data/ktables/h2o',
         '/Users/jingxuanyang/Desktop/Workspace/nemesispy2022/nemesispy/data/ktables/co2',
         '/Users/jingxuanyang/Desktop/Workspace/nemesispy2022/nemesispy/data/ktables/co',
         '/Users/jingxuanyang/Desktop/Workspace/nemesispy2022/nemesispy/data/ktables/ch4']

aeriel_files = ['/Users/jingxuanyang/Desktop/Workspace/nemesispy2022/nemesispy/data/ktables/H2O_Katy_ARIEL_test',
          '/Users/jingxuanyang/Desktop/Workspace/nemesispy2022/nemesispy/data/ktables/CO2_Katy_ARIEL_test',
          '/Users/jingxuanyang/Desktop/Workspace/nemesispy2022/nemesispy/data/ktables/CO_Katy_ARIEL_test',
          '/Users/jingxuanyang/Desktop/Workspace/nemesispy2022/nemesispy/data/ktables/CH4_Katy_ARIEL_test']

hires_files = ['/Users/jingxuanyang/Desktop/Workspace/nemesispy2022/nemesispy/data/ktables/H2O_Katy_R1000',
          '/Users/jingxuanyang/Desktop/Workspace/nemesispy2022/nemesispy/data/ktables/CO2_Katy_R1000',
          '/Users/jingxuanyang/Desktop/Workspace/nemesispy2022/nemesispy/data/ktables/CO_Katy_R1000',
          '/Users/jingxuanyang/Desktop/Workspace/nemesispy2022/nemesispy/data/ktables/CH4_Katy_R1000']

filenames = aeriel_files
"""
filenames : list
    A list of strings containing names of the kta files to be read.
"""
# P_layer = np.array([])
# """
# P_layer : ndarray
#     Atmospheric pressure grid.
# """
# T_layer = np.array([])
# """
# T_layer : ndarray
#     Atmospheric temperature grid.
# """
# U_layer = np.array([])
# """
# U_layer : ndarray
#     Total number of gas particles in each layer.
# """
# f = np.array([[],[]])
# VMR_layer = f.T
# """
# f(ngas,nlayer) : ndarray
#     fraction of the different gases at each of the p-T points
# """

"""
wave_grid : ndarray
    Wavelengths (um) grid for calculating spectra.
"""
### Calling sequence
# Get averaged layer properties
"""
H_layer,P_layer,T_layer,VMR_layer,U_layer,Gas_layer,scale,del_S\
    = average(planet_radius, H_atm, P_atm, T_atm, VMR_atm, ID, H_base)
"""
H_layer,P_layer,T_layer,VMR_layer,U_layer,Gas_layer,scale,del_S\
    = get_profiles(planet_radius, H_atm, P_atm, VMR_atm, T_atm, ID, Nlayer,
    H_base=None, path_angle=0.0, layer_type=1, bottom_height=0.0, interp_type=1, P_base=None,
    integration_type=1, Nsimps=101)

P_layer = P_layer*1e-5
print('H_layer', H_layer)
print('P_layer', P_layer)
print('T_layer', T_layer)
print('VMR_layer', VMR_layer)
print('U_layer', U_layer)
print('Gas_layer', Gas_layer)
print('scale', scale)
print('del_S', del_S)

# Get raw k table infos from files
gas_id_list, iso_id_list, wave_grid, g_ord, del_g, P_grid, T_grid,\
        k_gas_w_g_p_t = read_kls(filenames)
print('wave_grid', wave_grid)
print('g_ord', g_ord)
print('del_g', del_g)
print('P_grid', P_grid)

"""
# Interpolate k lists to layers
k_gas_w_g_l = interp_k(P_grid, T_grid, P_layer, T_layer, k_gas_w_g_p_t)

# Mix gas opacities
k_w_g_l = new_k_overlap(k_gas_w_g_l,del_g,f)
"""
StarSpectrum = np.ones(len(wave_grid)) # NWAVE
# Radiative Transfer
SPECOUT = radtran(wave_grid, U_layer, P_layer, T_layer, VMR_layer, k_gas_w_g_p_t,
            P_grid, T_grid, g_ord, del_g, ScalingFactor=scale,
            RADIUS=planet_radius, solspec=StarSpectrum)

"""
wave_grid = np.array([1.1425, 1.1775, 1.2125, 1.2475, 1.2825, 1.3175, 1.3525, 1.3875, 1.4225,
1.4575, 1.4925, 1.5275, 1.5625, 1.5975, 1.6325, 3.6, 4.5])
"""
print(SPECOUT)

import matplotlib.pyplot as plt

plt.plot(wave_grid,-SPECOUT)
plt.show()
plt.close()