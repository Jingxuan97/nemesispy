import sys
sys.path.append('/Users/jingxuanyang/Desktop/Workspace/nemesispy2022/')
import numpy as np
from nemesispy.data.constants import R_SUN, R_JUP_E, AMU, AU, M_JUP, R_JUP
from nemesispy.radtran.utils import calc_mmw
from nemesispy.radtran.models import Model2
from nemesispy.radtran.path import get_profiles # average
from nemesispy.radtran.k1read import read_kls
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
VMR_H2O = np.ones(NProfile)*1e-4
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
mmw = calc_mmw(ID,VMR_atm[0,:])
# print('mmw',mmw/AMU)

### Required Inputs
# Planet/star parameters
T_star = 4520
M_plt = 3.8951064000000004e+27 # kg
SMA = 0.015*AU
R_star = 463892759.99999994 #km
R_plt = 74065712.0 #km

# Atmosphere layout
NProfile = 20
Nlayer = 20
P_range = np.logspace(np.log10(20),np.log10(1e-3),NProfile)*1e5

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
print('T_atm',T_atm)
lowres_files = ['/Users/jingxuanyang/Desktop/Workspace/nemesispy2022/nemesispy/data/ktables/h2o',
         '/Users/jingxuanyang/Desktop/Workspace/nemesispy2022/nemesispy/data/ktables/co2',
         '/Users/jingxuanyang/Desktop/Workspace/nemesispy2022/nemesispy/data/ktables/co',
         '/Users/jingxuanyang/Desktop/Workspace/nemesispy2022/nemesispy/data/ktables/ch4']
filenames = lowres_files
"""
U_layer = np.array( [0.44339E+27, 0.26863E+27, 0.16284E+27, 0.98846E+26,
0.60088E+26, 0.36566E+26, 0.22266E+26, 0.13562E+26,
0.82604E+25, 0.50314E+25, 0.30652E+25, 0.18683E+25,
0.11396E+25, 0.69568E+24, 0.42496E+24, 0.25983E+24,
0.15897E+24, 0.97338E+23, 0.59639E+23, 0.36563E+23])*1e4
"""


# print('H_layer', H_layer)
# print('P_layer', P_layer)
# print('T_layer', T_layer)
# # print('VMR_layer', VMR_layer)
# print('U_layer', U_layer)
# print('Gas_layer', Gas_layer)
# print('scale', scale)
# print('del_S', del_S)

# Get raw k table infos from files
gas_id_list, iso_id_list, wave_grid, g_ord, del_g, P_grid, T_grid,\
        k_gas_w_g_p_t = read_kls(filenames)
# P_grid is in Pa?
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

# DO Gauss Labatto quadrature averaging
# angles = np.array([80.4866,61.4500,42.3729,23.1420,0.00000])
H_layer,P_layer,T_layer,VMR_layer,U_layer,Gas_layer,scale,del_S\
    = get_profiles(R_plt, H_atm, P_atm, VMR_atm, T_atm, ID, Nlayer,
    H_base=None, path_angle=0.0, layer_type=1, bottom_height=0.0, interp_type=1, P_base=None,
    integration_type=1, Nsimps=101)

# Radiative Transfer
SPECOUT = radtran(wave_grid, U_layer, P_layer, T_layer, VMR_layer, k_gas_w_g_p_t,
            P_grid, T_grid, del_g, ScalingFactor=scale,
            RADIUS=R_plt, solspec=StarSpectrum,
            k_cia=K_CIA,ID=ID,NU_GRID=CIA_NU_GRID,CIA_TEMPS=CIA_TEMPS, DEL_S=del_S)

"""
wave_grid = np.array([1.1425, 1.1775, 1.2125, 1.2475, 1.2825, 1.3175, 1.3525, 1.3875, 1.4225,
1.4575, 1.4925, 1.5275, 1.5625, 1.5975, 1.6325, 3.6, 4.5])
"""
fortran_model = [3.6743158e+22, 3.7200399e+22, 3.9049920e+22, 4.0878084e+22, 4.1409220e+22,
 3.8722958e+22, 2.5589892e+22, 2.2244738e+22, 2.0252917e+22, 2.1009452e+22,
 2.2436328e+22, 2.3985561e+22, 2.4880188e+22, 2.4668674e+22, 2.3569399e+22,
 5.1856743e+21, 2.5685985e+21]
# print(SPECOUT)
fortran_model = [2.5029930e+22, 2.4856582e+22, 2.9633066e+22, 3.5345970e+22, 3.6964785e+22,
 3.1785492e+22, 1.4004005e+22, 9.9735074e+21, 8.9512482e+21, 9.7668563e+21,
 1.1407971e+22, 1.4087847e+22, 1.7370373e+22, 2.0564625e+22, 2.2217288e+22,
 4.4644814e+21, 2.2432798e+21]
import matplotlib.pyplot as plt

plt.title('debug')
plt.plot(wave_grid,SPECOUT)
plt.scatter(wave_grid,SPECOUT,marker='o',color='b',linewidth=0.5,s=10, label='python')
plt.scatter(wave_grid,fortran_model,label='fortran',marker='x',color='k')
plt.plot(wave_grid,fortran_model,color='k')
plt.legend()
plt.tight_layout()
plt.plot()
plt.grid()
plt.savefig('comparison.pdf',dpi=400)
plt.show()
plt.close()



#
"""
H_layer [  39982.28105673  126553.90264747  211330.5394781   292777.81440577
  369793.59120799  442089.22797376  508754.55584789  570932.04953472
  630725.61481987  687408.69045286  741319.92330427  793568.25612261
  844736.77123865  895180.02329542  945339.53467767  995381.97058124
 1045153.6677233  1094756.07622753 1143868.78581049 1193532.81570359]

nemesis_baseH [0.00, 86.89, 172.57, 255.65, 334.80, 409.19, 478.60, 543.30,
603.90, 661.18, 715.94, 768.88, 820.57, 871.44, 921.77, 971.75, 1021.51, 1071.13,
1120.65, 1170.11]


P_layer [1.64055416e+06 9.92510354e+05 6.01167497e+05 3.64602192e+05
 2.21399676e+05 1.34504294e+05 8.20421191e+04 4.96052535e+04
 3.03623614e+04 1.84842789e+04 1.12591955e+04 6.86884432e+03
 4.19421833e+03 2.56040438e+03 1.56561013e+03 9.57956209e+02
 5.85902035e+02 3.59001036e+02 2.19586076e+02 1.34628298e+02]




T_layer [2286.02002349 2253.64646524 2185.69463006 2082.27160505 1955.73750894
 1822.19091572 1696.54582362 1585.30568524 1497.59337858 1430.36293213
 1382.34643351 1349.68416933 1328.21482221 1314.43545412 1305.78072334
 1300.39345409 1297.05833914 1295.01206525 1293.75107441 1292.98190935]

U_layer [4.43039336e+30 2.68975412e+30 1.62933025e+30 9.88373724e+29
 6.00420618e+29 3.67804576e+29 2.17388668e+29 1.36487322e+29
 8.41769794e+28 5.04754119e+28 3.05967533e+28 1.86561540e+28
 1.13591355e+28 6.93260355e+27 4.26147907e+27 2.60621767e+27
 1.58731188e+27 9.74298555e+26 5.84312548e+26 3.76228180e+26]

nemesis_totam [0.44339E+27, 0.26863E+27, 0.16284E+27, 0.98846E+26,
0.60088E+26, 0.36566E+26, 0.22266E+26, 0.13562E+26,
0.82604E+25, 0.50314E+25, 0.30652E+25, 0.18683E+25,
0.11396E+25, 0.69568E+24, 0.42496E+24, 0.25983E+24,
0.15897E+24, 0.97338E+23, 0.59639E+23, 0.36563E+23]


Gas_layer [[4.43039336e+26 4.43039336e+10 4.43039336e+10 4.43039336e+10
  6.64492547e+29 3.76545777e+30]
 [2.68975412e+26 2.68975412e+10 2.68975412e+10 2.68975412e+10
  4.03422772e+29 2.28606237e+30]
 [1.62933025e+26 1.62933025e+10 1.62933025e+10 1.62933025e+10
  2.44375098e+29 1.38479222e+30]
 [9.88373724e+25 9.88373724e+09 9.88373724e+09 9.88373724e+09
  1.48241233e+29 8.40033654e+29]
 [6.00420618e+25 6.00420618e+09 6.00420618e+09 6.00420618e+09
  9.00540864e+28 5.10306489e+29]
 [3.67804576e+25 3.67804576e+09 3.67804576e+09 3.67804576e+09
  5.51651693e+28 3.12602626e+29]
 [2.17388668e+25 2.17388668e+09 2.17388668e+09 2.17388668e+09
  3.26050394e+28 1.84761890e+29]
 [1.36487322e+25 1.36487322e+09 1.36487322e+09 1.36487322e+09
  2.04710510e+28 1.16002622e+29]
 [8.41769794e+24 8.41769794e+08 8.41769794e+08 8.41769794e+08
  1.26252843e+28 7.15432775e+28]
 [5.04754119e+24 5.04754119e+08 5.04754119e+08 5.04754119e+08
  7.57055465e+27 4.28998097e+28]
 [3.05967533e+24 3.05967533e+08 3.05967533e+08 3.05967533e+08
  4.58905404e+27 2.60046396e+28]
 [1.86561540e+24 1.86561540e+08 1.86561540e+08 1.86561540e+08
  2.79814325e+27 1.58561451e+28]
 [1.13591355e+24 1.13591355e+08 1.13591355e+08 1.13591355e+08
  1.70369994e+27 9.65429967e+27]
 [6.93260355e+23 6.93260355e+07 6.93260355e+07 6.93260355e+07
  1.03978654e+27 5.89212375e+27]
 [4.26147907e+23 4.26147907e+07 4.26147907e+07 4.26147907e+07
  6.39157938e+26 3.62189498e+27]
 [2.60621767e+23 2.60621767e+07 2.60621767e+07 2.60621767e+07
  3.90893557e+26 2.21506349e+27]
 [1.58731188e+23 1.58731188e+07 1.58731188e+07 1.58731188e+07
  2.38072973e+26 1.34908018e+27]
 [9.74298555e+22 9.74298555e+06 9.74298555e+06 9.74298555e+06
  1.46130169e+26 8.28070957e+26]
 [5.84312548e+22 5.84312548e+06 5.84312548e+06 5.84312548e+06
  8.76381176e+25 4.96616000e+26]
 [3.76228180e+22 3.76228180e+06 3.76228180e+06 3.76228180e+06
  5.64285836e+25 3.19761974e+26]]
"""