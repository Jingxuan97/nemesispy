import sys
sys.path.append('/Users/jingxuanyang/Desktop/Workspace/nemesispy2022/')
import matplotlib.pyplot as plt
import numpy as np
from nemesispy.data.constants import R_SUN, R_JUP_E, AMU, AU, M_JUP, R_JUP
from nemesispy.radtran.utils import calc_mmw
from nemesispy.radtran.models import Model2
from nemesispy.radtran.path import calc_layer # average
from nemesispy.radtran.read import read_kls
from nemesispy.radtran.radiance import calc_radiance, calc_planck
from nemesispy.radtran.read import read_cia

lowres_files = ['/Users/jingxuanyang/Desktop/Workspace/nemesispy2022/nemesispy/data/ktables/h2o',
         '/Users/jingxuanyang/Desktop/Workspace/nemesispy2022/nemesispy/data/ktables/co2',
         '/Users/jingxuanyang/Desktop/Workspace/nemesispy2022/nemesispy/data/ktables/co',
         '/Users/jingxuanyang/Desktop/Workspace/nemesispy2022/nemesispy/data/ktables/ch4']
filenames = lowres_files

lowres_files = ['/Users/jingxuanyang/ktables/h2owasp43.kta',
'/Users/jingxuanyang/ktables/cowasp43.kta',
'/Users/jingxuanyang/ktables/co2wasp43.kta',
'/Users/jingxuanyang/ktables/ch4wasp43.kta']

cia_file_path='/Users/jingxuanyang/Desktop/Workspace/nemesispy2022/nemesispy/data/cia/exocia_hitran12_200-3800K.tab'
# Gas identifiers.
ID = np.array([1,2,5,6,40,39])
ISO = np.array([0,0,0,0,0,0])
NProfile = 20
NVMR = len(ID)

# Volume Mixing Ratio
# VMR_atm[i,j] is the Volume Mixing Ratio of jth gas at ith layer.
H2ratio = 1
VMR_H2O = np.ones(NProfile)*1e-4
VMR_CO2 = np.ones(NProfile)*1e-4 *0
VMR_CO = np.ones(NProfile)*1e-4 *0
VMR_CH4 = np.ones(NProfile)*1e-4 *0
VMR_He = (np.ones(NProfile)-VMR_H2O-VMR_CO2-VMR_CO-VMR_CH4)*(1-H2ratio)
VMR_H2 = (np.ones(NProfile)-VMR_H2O-VMR_CO2-VMR_CO-VMR_CH4)*H2ratio
VMR_atm = np.zeros((NProfile,NVMR))
VMR_atm[:,0] = VMR_H2O
VMR_atm[:,1] = VMR_CO2
VMR_atm[:,2] = VMR_CO
VMR_atm[:,3] = VMR_CH4
VMR_atm[:,4] = VMR_He
VMR_atm[:,5] = VMR_H2
mmw = calc_mmw(ID,VMR_atm[0,:])

# Planet/star parameters
T_star = 4520
M_plt = 3.8951064000000004e+27 # kg
SMA = 0.015*AU
R_star = 463892759.99999994 #km
R_plt = 74065.70 * 1e3 #km

# Atmosphere layout
NProfile = 20
Nlayer = 20
P_range = np.logspace(np.log10(20/1.01325),np.log10(1e-3/1.01325),NProfile)*1e5

# Atmospheric model params
kappa = 1e-3
gamma1 = 1e-1
gamma2 = 1e-1
alpha = 0.5
T_irr =  1500
atm = Model2(T_star, R_star, M_plt, R_plt, SMA, P_range, mmw,
                      kappa, gamma1, gamma2, alpha, T_irr)
H_atm = atm.height()
P_atm = atm.pressure()
T_atm = atm.temperature()

# Get raw k table infos from files
gas_id_list, iso_id_list, wave_grid, g_ord, del_g, P_grid, T_grid,\
        k_gas_w_g_p_t = read_kls(filenames)
CIA_NU_GRID,CIA_TEMPS,K_CIA = read_cia(cia_file_path)
wave_grid = np.array([1.1425, 1.1775, 1.2125, 1.2475, 1.2825, 1.3175, 1.3525,
1.3875, 1.4225, 1.4575, 1.4925, 1.5275, 1.5625, 1.5975, 1.6325, 3.6, 4.5])

# Get raw stellar spectrum
StarSpectrum = np.ones(len(wave_grid)) # *4*(R_star)**2*np.pi # NWAVE

# DO Gauss Labatto quadrature averaging
# angles = np.array([80.4866,61.4500,42.3729,23.1420,0.00000])
H_layer,P_layer,T_layer,VMR_layer,U_layer,Gas_layer,scale,del_S\
    = calc_layer(R_plt, H_atm, P_atm,  T_atm, VMR_atm, ID, Nlayer,
    path_angle=0, layer_type=1, H_0=0.0, interp_type=1,
    integration_type=1, NSIMPS=101)
print('scale',scale)

"""
before jit
1000 nemesis ver:381.06474781036377
1000 chimera ver:341.1965470314026

after jit

1. jit just for interp
1000 nemesis ver : f_combined is fucking things up
1000 chimera ver : 32.83659029006958

2. jit for interp and radiance


FULLY JIT
runtime 1000 = 0.004322124004364014 per
runtime 10000 = 0.002207125186920166
"""

SPECOUT = calc_radiance(wave_grid, U_layer, P_layer, T_layer, VMR_layer, k_gas_w_g_p_t,
        P_grid, T_grid, del_g, ScalingFactor=scale,
        RADIUS=R_plt, solspec=StarSpectrum,
        k_cia=K_CIA,ID=ID,NU_GRID=CIA_NU_GRID,CIA_TEMPS=CIA_TEMPS, DEL_S=del_S)


import time
run_number = 10000
start = time.time()
for i in range(run_number):
# Radiative Transfer
    SPECOUT = calc_radiance(wave_grid, U_layer, P_layer, T_layer, VMR_layer, k_gas_w_g_p_t,
            P_grid, T_grid, del_g, ScalingFactor=scale,
            RADIUS=R_plt, solspec=StarSpectrum,
            k_cia=K_CIA,ID=ID,NU_GRID=CIA_NU_GRID,CIA_TEMPS=CIA_TEMPS, DEL_S=del_S)

end = time.time()
print('run time = ', (end-start)/run_number)






# 1e-4 h2o, pure H2, fixed ground radiation at 45 zenith angle
fortran_model = [2.0095757e+22, 1.9855963e+22, 2.4384866e+22, 3.0096336e+22, 3.1821019e+22,
 2.6848518e+22, 1.1246998e+22, 7.8110916e+21, 7.0568513e+21, 7.7212110e+21,
 9.0706302e+21, 1.1298792e+22, 1.4093987e+22, 1.6893329e+22, 1.8325981e+22,
 4.0654737e+21, 2.0768640e+21]


# 1e-4 h2o, pure H2, fixed ground radiation at 0 zenith angle
fortran_model = [2.3386727e+22, 2.3223329e+22, 2.7884951e+22, 3.3525034e+22, 3.5133009e+22,
 3.0026636e+22, 1.3094568e+22, 9.2723774e+21, 8.3360171e+21, 9.0995923e+21,
 1.0637932e+22, 1.3140588e+22, 1.6189699e+22, 1.9094999e+22, 2.0491682e+22,
 4.3127540e+21, 2.1802355e+21]

diff = (SPECOUT-fortran_model)
plt.scatter(wave_grid, diff,label='diff',marker='x',color='r',s=20)
diff = (SPECOUT-fortran_model)/SPECOUT

# start plot
plt.title('debug')
# lt.plot(wave_grid,SPECOUT)

plt.scatter(wave_grid,SPECOUT,marker='o',color='b',linewidth=0.5,s=10, label='python')
plt.scatter(wave_grid,fortran_model,label='fortran',marker='x',color='k',s=20)

# BB = calc_planck(wave_grid,2285.991)*np.pi*4.*np.pi*(74065.70*1e5)**2
# plt.plot(wave_grid,BB,label='black body',marker='*')

plt.xlabel(r'wavelength($\mu$m)')
plt.ylabel(r'total radiance(W sr$^{-1}$ $\mu$m$^{-1})$')
plt.legend()
plt.tight_layout()
plt.plot()
plt.grid()
# plt.savefig('comparison.pdf',dpi=400)
plt.show()
# plt.close()


print(diff)
print(max(diff))
