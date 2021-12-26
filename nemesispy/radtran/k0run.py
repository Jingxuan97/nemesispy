import sys
sys.path.append('/Users/jingxuanyang/Desktop/Workspace/nemesispy2022/')
import numpy as np
from nemesispy.data.constants import R_SUN, R_JUP_E, AMU, AU, M_JUP, R_JUP
from nemesispy.radtran.models import Model2
from nemesispy.radtran.path import get_profiles # average
from nemesispy.radtran.k1read import read_kls
# from nemesispy.radtran.k2interp import interp_k, new_k_overlap
from nemesispy.radtran.k3radtran import radtran
from nemesispy.radtran.k5cia import read_cia




"""
         0.000  0.19739E+02    2294.2300  0.10000E-04  0.10000E-19  0.10000E-19  0.10000E-19  0.15000E+00  0.84999E+00
        90.390  0.11720E+02    2275.6741  0.10000E-04  0.10000E-19  0.10000E-19  0.10000E-19  0.15000E+00  0.84999E+00
       179.552  0.69594E+01    2221.3721  0.10000E-04  0.10000E-19  0.10000E-19  0.10000E-19  0.15000E+00  0.84999E+00
       265.916  0.41324E+01    2124.3049  0.10000E-04  0.10000E-19  0.10000E-19  0.10000E-19  0.15000E+00  0.84999E+00
       347.980  0.24538E+01    1995.6700  0.10000E-04  0.10000E-19  0.10000E-19  0.10000E-19  0.15000E+00  0.84999E+00
       424.832  0.14570E+01    1854.4310  0.10000E-04  0.10000E-19  0.10000E-19  0.10000E-19  0.15000E+00  0.84999E+00
       496.285  0.86515E+00    1718.0520  0.10000E-04  0.10000E-19  0.10000E-19  0.10000E-19  0.15000E+00  0.84999E+00
       562.745  0.51372E+00    1598.6730  0.10000E-04  0.10000E-19  0.10000E-19  0.10000E-19  0.15000E+00  0.84999E+00
       624.995  0.30504E+00    1502.5710  0.10000E-04  0.10000E-19  0.10000E-19  0.10000E-19  0.15000E+00  0.84999E+00
       683.970  0.18113E+00    1430.7090  0.10000E-04  0.10000E-19  0.10000E-19  0.10000E-19  0.15000E+00  0.84999E+00
       740.575  0.10755E+00    1380.3280  0.10000E-04  0.10000E-19  0.10000E-19  0.10000E-19  0.15000E+00  0.84999E+00
       795.573  0.63862E-01    1346.8170  0.10000E-04  0.10000E-19  0.10000E-19  0.10000E-19  0.15000E+00  0.84999E+00
       849.542  0.37920E-01    1325.3910  0.10000E-04  0.10000E-19  0.10000E-19  0.10000E-19  0.15000E+00  0.84999E+00
       902.885  0.22517E-01    1312.0670  0.10000E-04  0.10000E-19  0.10000E-19  0.10000E-19  0.15000E+00  0.84999E+00
       955.869  0.13370E-01    1303.9330  0.10000E-04  0.10000E-19  0.10000E-19  0.10000E-19  0.15000E+00  0.84999E+00
      1008.664  0.79390E-02    1299.0250  0.10000E-04  0.10000E-19  0.10000E-19  0.10000E-19  0.15000E+00  0.84999E+00
      1061.373  0.47140E-02    1296.0840  0.10000E-04  0.10000E-19  0.10000E-19  0.10000E-19  0.15000E+00  0.84999E+00
      1114.062  0.27991E-02    1294.3311  0.10000E-04  0.10000E-19  0.10000E-19  0.10000E-19  0.15000E+00  0.84999E+00
      1166.766  0.16621E-02    1293.2880  0.10000E-04  0.10000E-19  0.10000E-19  0.10000E-19  0.15000E+00  0.84999E+00
      1219.512  0.98692E-03    1292.6680  0.10000E-04  0.10000E-19  0.10000E-19  0.10000E-19  0.15000E+00  0.84999E+00
"""

"""
ID : ndarray
    Gas identifiers.
VMR_atm : ndarray
    VMR_atm[i,j] is the Volume Mixing Ratio of gas j at profile point i.
    The jth column corresponds to the gas with RADTRANS ID ID[j].
"""

ID = np.array([1,2,5,6,40,39])
ISO = np.array([0,0,0,0,0,0])
NProfile = 20
NVMR = len(ID)

VMR_atm = np.zeros((NProfile,NVMR))
VMR_H2O = np.ones(NProfile)*1e-5
VMR_CO2 = np.ones(NProfile)*1e-20
VMR_CO = np.ones(NProfile)*1e-20
VMR_CH4 = np.ones(NProfile)*1e-20
#VMR_He = 0.03
#VMR_H2 = 0
VMR_He = (np.ones(NProfile)-VMR_H2O-VMR_CO2-VMR_CO-VMR_CH4)*0.15
VMR_H2 = VMR_He/0.15*0.85
VMR_atm[:,0] = VMR_H2O
VMR_atm[:,1] = VMR_CO2
VMR_atm[:,2] = VMR_CO
VMR_atm[:,3] = VMR_CH4
VMR_atm[:,4] = VMR_He
VMR_atm[:,5] = VMR_H2

### Required Inputs
# Planet/star parameters
T_star = 4520
M_plt = 2.052*M_JUP
SMA = 0.015*AU
R_star = 0.6668*R_SUN
planet_radius = 1*R_JUP
R_plt = 1.036*R_JUP_E

"""
planet_radius : real
    Reference planetary radius where H_atm=0.  Usually at surface for
    terrestrial planets, or at 1 bar pressure level for gas giants.
"""
H_atm = np.array([])
"""

"""
P_atm = np.array([])
NProfile = 20
Nlayer = 25
P_range = np.geomspace(20,1e-3,NProfile)*1e5
mmw = 2*AMU

### params
kappa = 1e-3
gamma1 = 1e-1
gamma2 = 1e-1
alpha = 0.5
T_irr = 1500

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

filenames = lowres_files
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

# Get raw CIA info
cia_file_path='/Users/jingxuanyang/Desktop/Workspace/nemesispy2022/nemesispy/data/cia/exocia_hitran12_200-3800K.tab'
CIA_NU_GRID,CIA_TEMPS,K_CIA = read_cia(cia_file_path)


StarSpectrum = np.ones(len(wave_grid))*(R_star)**2*np.pi # NWAVE
# Radiative Transfer
SPECOUT = radtran(wave_grid, U_layer, P_layer, T_layer, VMR_layer, k_gas_w_g_p_t,
            P_grid, T_grid, del_g, ScalingFactor=scale,
            RADIUS=planet_radius, solspec=StarSpectrum,
            k_cia=K_CIA,ID=ID,NU_GRID=CIA_NU_GRID,CIA_TEMPS=CIA_TEMPS)

"""
wave_grid = np.array([1.1425, 1.1775, 1.2125, 1.2475, 1.2825, 1.3175, 1.3525, 1.3875, 1.4225,
1.4575, 1.4925, 1.5275, 1.5625, 1.5975, 1.6325, 3.6, 4.5])
"""
print(SPECOUT)

import matplotlib.pyplot as plt

plt.title('debug')
plt.plot(wave_grid,SPECOUT)
plt.scatter(wave_grid,SPECOUT,marker='o',color='k',linewidth=0.5,s=10)
plt.tight_layout()
plt.grid()
plt.show()
plt.close()