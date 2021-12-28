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


# Gas identifiers.
ID = np.array([1,2,5,6,40,39])
ISO = np.array([0,0,0,0,0,0])
NProfile = 20
NVMR = len(ID)

# Volume Mixing Ratio
# VMR_atm[i,j] is the Volume Mixing Ratio of gas j at profile point i.
VMR_atm = np.zeros((NProfile,NVMR))
VMR_H2O = np.ones(NProfile)*1e-5
VMR_CO2 = np.ones(NProfile)*1e-20
VMR_CO = np.ones(NProfile)*1e-20
VMR_CH4 = np.ones(NProfile)*1e-20
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
R_star = 463899e3 #km
R_plt = 74065712.0 #km

# Atmosphere layout
NProfile = 20
Nlayer = 20
P_range = np.geomspace(20,1e-3,NProfile)*101325
mmw = 2*AMU

# Atmospheric model params
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

lowres_files = ['/Users/jingxuanyang/Desktop/Workspace/nemesispy2022/nemesispy/data/ktables/h2o',
         '/Users/jingxuanyang/Desktop/Workspace/nemesispy2022/nemesispy/data/ktables/co2',
         '/Users/jingxuanyang/Desktop/Workspace/nemesispy2022/nemesispy/data/ktables/co',
         '/Users/jingxuanyang/Desktop/Workspace/nemesispy2022/nemesispy/data/ktables/ch4']
filenames = lowres_files

H_layer,P_layer,T_layer,VMR_layer,U_layer,Gas_layer,scale,del_S\
    = get_profiles(R_plt, H_atm, P_atm, VMR_atm, T_atm, ID, Nlayer,
    H_base=None, path_angle=0.0, layer_type=1, bottom_height=0.0, interp_type=1, P_base=None,
    integration_type=1, Nsimps=101)

print('H_layer', H_layer)
print('P_layer', P_layer)
print('T_layer', T_layer)
# print('VMR_layer', VMR_layer)
print('U_layer', U_layer)
print('Gas_layer', Gas_layer)
print('scale', scale)
print('del_S', del_S)

# Get raw k table infos from files
gas_id_list, iso_id_list, wave_grid, g_ord, del_g, P_grid, T_grid,\
        k_gas_w_g_p_t = read_kls(filenames)
P_grid *= 101325
print('wave_grid', wave_grid)
print('g_ord', g_ord)
print('del_g', del_g)
print('P_grid', P_grid)

# Get raw CIA info
cia_file_path='/Users/jingxuanyang/Desktop/Workspace/nemesispy2022/nemesispy/data/cia/exocia_hitran12_200-3800K.tab'
CIA_NU_GRID,CIA_TEMPS,K_CIA = read_cia(cia_file_path)

# Get raw stellar spectrum
StarSpectrum = np.ones(len(wave_grid))# *4*(R_star)**2*np.pi # NWAVE

# Radiative Transfer
SPECOUT = radtran(wave_grid, U_layer, P_layer, T_layer, VMR_layer, k_gas_w_g_p_t,
            P_grid, T_grid, del_g, ScalingFactor=scale,
            RADIUS=R_plt, solspec=StarSpectrum,
            k_cia=K_CIA,ID=ID,NU_GRID=CIA_NU_GRID,CIA_TEMPS=CIA_TEMPS)

"""
wave_grid = np.array([1.1425, 1.1775, 1.2125, 1.2475, 1.2825, 1.3175, 1.3525, 1.3875, 1.4225,
1.4575, 1.4925, 1.5275, 1.5625, 1.5975, 1.6325, 3.6, 4.5])
"""
fortran_model = [3.6743158e+22, 3.7200399e+22, 3.9049920e+22, 4.0878084e+22, 4.1409220e+22,
 3.8722958e+22, 2.5589892e+22, 2.2244738e+22, 2.0252917e+22, 2.1009452e+22,
 2.2436328e+22, 2.3985561e+22, 2.4880188e+22, 2.4668674e+22, 2.3569399e+22,
 5.1856743e+21, 2.5685985e+21]
# print(SPECOUT)

import matplotlib.pyplot as plt

plt.title('debug')
plt.plot(wave_grid,SPECOUT)
plt.scatter(wave_grid,SPECOUT,marker='o',color='k',linewidth=0.5,s=10)
plt.plot(wave_grid,fortran_model,label='fortran')
plt.legend()
plt.tight_layout()
plt.plot()
plt.grid()
plt.show()
plt.close()



#