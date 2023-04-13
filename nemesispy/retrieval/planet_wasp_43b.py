import numpy as np
from nemesispy.common.constants import G
from nemesispy.data.helper import lowres_file_paths, cia_file_path

planet = {
    'NLAYER' : None,
    'P_range' : None,
    'T_star' : None,
    'R_star' : None,
    'SMA' : None,
    'M_plt' : None,
    'R_plt' : None,
    'T_irr' : None,
    'T_eq' : None,
    'g_plt' : None,
    'gas_id' : None,
    'iso_id' : None,
    'wave_grid' : None,
    'phase_grid' : None,
    'stellar_spec' : None,
    'nwave' : None,
    'nphase' : None,
    'kta_file_paths' : None,
    'cia_file_path' : None
}

# number of layers in the atmospheric mdoel
NLAYER = 20
planet['NLAYER'] = NLAYER

# pressure grid in Pa
P_highest = 20 * 1e5
P_lowest = 1e-3 * 1e5
P_range = np.geomspace(P_highest,P_lowest,NLAYER)
planet['P_range'] = P_range

# star temperature in K
T_star = 4520
planet['T_star'] = T_star

# star radius in m
R_star = 463892759.99999994 # m, 0.6668 * R_SUN
planet['R_star'] = R_star

# orbital semi-major axis in m
SMA = 2243970000.0 # m, 0.015*AU
planet['SMA'] = SMA

# planet mass in kg
M_plt = 3.8951064000000004e+27 # kg
planet['M_plt'] = M_plt

# planet radius in m
R_plt = 74065.70 * 1e3 # m
planet['R_plt'] = R_plt

# irradiation temperature in K
T_irr = T_star * (R_star/SMA)**0.5 # 2055 K
planet['T_irr'] = T_irr

# planet equilibrium temperature in K
T_eq = T_irr/2**0.5 # 1453 K
planet['T_eq'] = T_eq

# planet gravity
g_plt = G*M_plt/R_plt**2 # 47.39 ms-2
planet['g_plt'] = g_plt

# gas id
gas_id = np.array([  1, 2,  5,  6, 40, 39])
planet['gas_id'] = gas_id

# isotope id
iso_id = np.array([0, 0, 0, 0, 0, 0])
planet['iso_id'] = iso_id

# wavelength grid
wave_grid = np.array(
    [1.1425, 1.1775, 1.2125, 1.2475, 1.2825, 1.3175, 1.3525, 1.3875,
    1.4225, 1.4575, 1.4925, 1.5275, 1.5625, 1.5975, 1.6325, 3.6   ,
    4.5   ])
planet['wave_grid'] = wave_grid

# orbital phase grid
phase_grid = np.array(
    [ 22.5,  45. ,  67.5,  90. , 112.5, 135. , 157.5, 180. , 202.5,
    225. , 247.5, 270. , 292.5, 315. , 337.5])
planet['phase_grid'] = phase_grid

# stellar spectrum
stellar_spec = np.array(
    [3.341320e+25, 3.215455e+25, 3.101460e+25, 2.987110e+25,
    2.843440e+25, 2.738320e+25, 2.679875e+25, 2.598525e+25,
    2.505735e+25, 2.452230e+25, 2.391140e+25, 2.345905e+25,
    2.283720e+25, 2.203690e+25, 2.136015e+25, 1.234010e+24,
    4.422200e+23])
planet['stellar_spec'] = stellar_spec

# number of wavelengths
nwave = len(planet['wave_grid'])
planet['nwave'] = nwave

# number of orbital phases
nphase = len(planet['phase_grid'])
planet['nphase'] = nphase

# opacity file paths
planet['kta_file_paths'] = lowres_file_paths
planet['cia_file_path'] = cia_file_path