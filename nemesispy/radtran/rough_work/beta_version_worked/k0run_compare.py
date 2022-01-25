import numpy as np
import matplotlib.pyplot as plt
from nemesispy.radtran.path import get_profiles # average
from nemesispy.radtran.k5cia import read_cia
from nemesispy.radtran.k1read import read_kls
"""
First layer height should be 0???
"""
"""Set up a direct comparison between FORTRAN and Python"""

########################################################################
"""Atmosphere Physical Conditions"""
########################################################################
"""planet radius(km)"""
R_plt = 74065.70

"""height(km)"""
H_layer = np.array([0.000, 90.390, 179.552, 265.916, 347.980, 424.832, 496.285,
562.745, 624.995, 683.970, 740.575, 795.573, 849.542, 902.885, 955.869, 1008.664,
1061.373, 1114.062, 1166.766, 1219.512])

"""pressure(atm)"""
P_layer = np.array([0.19739E+02,0.11720E+02,0.69594E+01,0.41324E+01,0.24538E+01,
0.14570E+01,0.86515E+00,0.51372E+00,0.30504E+00,0.18113E+00,0.10755E+00,
0.63862E-01,0.37920E-01,0.22517E-01,0.13370E-01,0.79390E-02,0.47140E-02,
0.27991E-02,0.16621E-02,0.98692E-03])

"""temperature(K)"""
T_layer = np.array([2294.2300,2275.6741,2221.3721,2124.3049,1995.6700,1854.4310,
1718.0520,1598.6730,1502.5710,1430.7090,1380.3280,1346.8170,1325.3910,
1312.0670,1303.9330,1299.0250,1296.0840,1294.3311,1293.2880,1292.6680])

NProfile = len(T_layer)

########################################################################
"""Atmosphere Chemical Compositions"""
########################################################################
"""gas ID list"""
ID = np.array([1,2,5,6,40,39])
ISO = np.array([0,0,0,0,0,0])
NVMR = len(ID)

"""gas VMR"""
VMR_atm = np.zeros((NProfile,NVMR))
VMR_H2O = np.ones(NProfile)*1e-5
VMR_CO2 = np.ones(NProfile)*1e-20
VMR_CO = np.ones(NProfile)*1e-20
VMR_CH4 = np.ones(NProfile)*1e-20
VMR_He = np.ones(NProfile)*1.49999E-01
VMR_H2 = np.ones(NProfile)*8.49992E-01
VMR_layer = np.vstack([VMR_H2O,VMR_CO2,VMR_CO,VMR_CH4,VMR_He,VMR_H2]).T

########################################################################
"""Spectroscopy Data"""
########################################################################
"""k table filed"""
lowres_files\
=['/Users/jingxuanyang/Desktop/Workspace/nemesispy2022/nemesispy/data/ktables/h2o',
'/Users/jingxuanyang/Desktop/Workspace/nemesispy2022/nemesispy/data/ktables/co2',
'/Users/jingxuanyang/Desktop/Workspace/nemesispy2022/nemesispy/data/ktables/co',
'/Users/jingxuanyang/Desktop/Workspace/nemesispy2022/nemesispy/data/ktables/ch4']
# Get raw k table infos from files
gas_id_list, iso_id_list, wave_grid, g_ord, del_g, P_grid, T_grid,\
        k_gas_w_g_p_t = read_kls(filepaths=lowres_files)

cia_file_path\
='/Users/jingxuanyang/Desktop/Workspace/nemesispy2022/nemesispy/data/cia/exocia_hitran12_200-3800K.tab'
CIA_NU_GRID,CIA_TEMPS,K_CIA = read_cia(cia_file_path)