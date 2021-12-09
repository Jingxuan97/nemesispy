import numpy as np
from nemesispy.data.constants import R_SUN, R_JUP_E
from nemesispy.radtran.models import Model2
from nemesispy.radtran.path import average
from nemesispy.radtran.k1read import read_kls
# from nemesispy.radtran.k2interp import interp_k, new_k_overlap
from nemesispy.radtran.k3radtran import radtran


### Required Inputs
R_star = 1*R_SUN
planet_radius = R_JUP_E
"""
planet_radius : real
    Reference planetary radius where H_atm=0.  Usually at surface for
    terrestrial planets, or at 1 bar pressure level for gas giants.
"""
H_atm = np.array([])
"""
H_atm : ndarray
    Input profile heights
"""
P_atm = np.array([])
"""
P_atm : ndarray
    Input profile pressures
"""
T_atm = np.array([])
"""
T_atm : ndarray
    Input profile temperatures
"""
ID = np.array([])
"""
ID : ndarray
    Gas identifiers.
"""
VMR_atm = np.array([[]])
"""
VMR_atm : ndarray
    VMR_atm[i,j] is the Volume Mixing Ratio of gas j at profile point i.
    The jth column corresponds to the gas with RADTRANS ID ID[j].
"""
H_base = np.array([])
"""
H_base : ndarray
    Heights of the layer bases.
"""
filenames = []
lowres_files = ['/Users/jingxuanyang/Desktop/Workspace/nemesispy2022/data/ktables/h2o',
         '/Users/jingxuanyang/Desktop/Workspace/nemesispy2022/data/ktables/co2',
         '/Users/jingxuanyang/Desktop/Workspace/nemesispy2022/data/ktables/co',
         '/Users/jingxuanyang/Desktop/Workspace/nemesispy2022/data/ktables/ch4']

aeriel_files = ['/Users/jingxuanyang/Desktop/Workspace/nemesispy2022/data/ktables/H2O_Katy_ARIEL_test',
          '/Users/jingxuanyang/Desktop/Workspace/nemesispy2022/data/ktables/CO2_Katy_ARIEL_test',
          '/Users/jingxuanyang/Desktop/Workspace/nemesispy2022/data/ktables/CO_Katy_ARIEL_test',
          '/Users/jingxuanyang/Desktop/Workspace/nemesispy2022/data/ktables/CH4_Katy_ARIEL_test']

hires_files = ['/Users/jingxuanyang/Desktop/Workspace/nemesispy2022/data/ktables/H2O_Katy_R1000',
         '/Users/jingxuanyang/Desktop/Workspace/nemesispy2022/data/ktables/CO2_Katy_R1000',
          '/Users/jingxuanyang/Desktop/Workspace/nemesispy2022/data/ktables/CO_Katy_R1000',
          '/Users/jingxuanyang/Desktop/Workspace/nemesispy2022/data/ktables/CH4_Katy_R1000']
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
wave_grid = np.array([])
"""
wave_grid : ndarray
    Wavelengths (um) grid for calculating spectra.
"""
### Calling sequence
# Get averaged layer properties
H_layer,P_layer,T_layer,VMR_layer,U_layer,Gas_layer,scale,del_S\
    = average(planet_radius, H_atm, P_atm, T_atm, VMR_atm, ID, H_base)

# Get raw k table infos from files
gas_id_list, iso_id_list, wave_grid, g_ord, del_g, P_grid, T_grid,\
        k_gas_w_g_p_t = read_kls(filenames)

"""
# Interpolate k lists to layers
k_gas_w_g_l = interp_k(P_grid, T_grid, P_layer, T_layer, k_gas_w_g_p_t)

# Mix gas opacities
k_w_g_l = new_k_overlap(k_gas_w_g_l,del_g,f)
"""

# Radiative Transfer
SPECOUT = radtran(wave_grid, U_layer, P_layer, T_layer, VMR_layer, k_gas_w_g_p_t,
            P_grid, T_grid, g_ord, del_g)