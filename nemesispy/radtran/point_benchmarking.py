import numpy as np
import matplotlib.pyplot as plt
import os
import sys
sys.path.append('/Users/jingxuanyang/Desktop/Workspace/nemesispy2022/')
from nemesispy.radtran.forward_model import ForwardModel
from nemesispy.radtran.point_benchmarking_fortran import Nemesis_api
import time

### Reference Opacity Data
lowres_files = ['/Users/jingxuanyang/Desktop/Workspace/nemesispy2022/nemesispy/data/ktables/h2o',
         '/Users/jingxuanyang/Desktop/Workspace/nemesispy2022/nemesispy/data/ktables/co2',
         '/Users/jingxuanyang/Desktop/Workspace/nemesispy2022/nemesispy/data/ktables/co',
         '/Users/jingxuanyang/Desktop/Workspace/nemesispy2022/nemesispy/data/ktables/ch4']
cia_file_path='/Users/jingxuanyang/Desktop/Workspace/nemesispy2022/nemesispy/data/cia/exocia_hitran12_200-3800K.tab'
folder_name = 'testing'


lowres_files = ['/Users/jingxuanyang/Desktop/Workspace/nemesispy2022/nemesispy/data/ktables/h2o',
         '/Users/jingxuanyang/Desktop/Workspace/nemesispy2022/nemesispy/data/ktables/co2',
         '/Users/jingxuanyang/Desktop/Workspace/nemesispy2022/nemesispy/data/ktables/co',
         '/Users/jingxuanyang/Desktop/Workspace/nemesispy2022/nemesispy/data/ktables/ch4']

"""
lowres_files = [ '/Users/jingxuanyang/Desktop/Workspace/nemesispy2022/nemesispy/data/ktables/co',
         '/Users/jingxuanyang/Desktop/Workspace/nemesispy2022/nemesispy/data/ktables/co2',
         '/Users/jingxuanyang/Desktop/Workspace/nemesispy2022/nemesispy/data/ktables/h2o',
         '/Users/jingxuanyang/Desktop/Workspace/nemesispy2022/nemesispy/data/ktables/ch4']

lowres_files = [ '/Users/jingxuanyang/Desktop/Workspace/nemesispy2022/nemesispy/data/ktables/co2',
                '/Users/jingxuanyang/Desktop/Workspace/nemesispy2022/nemesispy/data/ktables/co',
         '/Users/jingxuanyang/Desktop/Workspace/nemesispy2022/nemesispy/data/ktables/h2o',
         '/Users/jingxuanyang/Desktop/Workspace/nemesispy2022/nemesispy/data/ktables/ch4']
"""
### Reference Constants
pi = np.pi
const = {
    'R_SUN': 6.95700e8,      # m solar radius
    'R_JUP': 7.1492e7,       # m nominal equatorial Jupiter radius (1 bar pressure level)
    'AU': 1.49598e11,        # m astronomical unit
    'k_B': 1.38065e-23,      # J K-1 Boltzmann constant
    'R': 8.31446,            # J mol-1 K-1 universal gas constant
    'G': 6.67430e-11,        # m3 kg-1 s-2 universal gravitational constant
    'N_A': 6.02214e23,       # Avagadro's number
    'AMU': 1.66054e-27,      # kg atomic mass unit
    'ATM': 101325,           # Pa atmospheric pressure
}

### Reference Planet Input
M_plt = 3.8951064000000004e+27 # kg
R_plt = 74065.70 * 1e3 # m
#R_star = 463892759.99999994 # m

### Reference Spectral Input
# Stellar spectrum
stellar_spec = np.array([3.341320e+25, 3.215455e+25, 3.101460e+25, 2.987110e+25,
       2.843440e+25, 2.738320e+25, 2.679875e+25, 2.598525e+25,
       2.505735e+25, 2.452230e+25, 2.391140e+25, 2.345905e+25,
       2.283720e+25, 2.203690e+25, 2.136015e+25, 1.234010e+24,
       4.422200e+23])
stellar_spec  = np.ones(len(stellar_spec))

# Spectral output wavelengths in micron
wave_grid = np.array([1.1425, 1.1775, 1.2125, 1.2475, 1.2825, 1.3175, 1.3525, 1.3875,
       1.4225, 1.4575, 1.4925, 1.5275, 1.5625, 1.5975, 1.6325, 3.6   ,
       4.5   ],dtype=np.float32)


### Reference Atmospheric Model Input
# Height in m
H = np.array([      0.     ,  103738.07012,  206341.39335,  305672.8162 ,
        400037.91149,  488380.27388,  570377.57036,  646857.33242,
        718496.09845,  785987.95083,  851242.50591,  914520.46249,
        976565.39549, 1037987.38369, 1099327.5361 , 1158956.80091,
       1221026.73382, 1280661.28989, 1341043.14058, 1404762.36466])

# Pressure in pa, note 1 atm = 101325 pa
P = np.array([2.00000000e+06, 1.18757212e+06, 7.05163779e+05, 4.18716424e+05,
       2.48627977e+05, 1.47631828e+05, 8.76617219e+04, 5.20523088e+04,
       3.09079355e+04, 1.83527014e+04, 1.08975783e+04, 6.47083012e+03,
       3.84228875e+03, 2.28149750e+03, 1.35472142e+03, 8.04414702e+02,
       4.77650239e+02, 2.83622055e+02, 1.68410823e+02, 1.00000000e+02])

# Temperature in Kelvin
T = np.array([2294.22993056, 2275.69702232, 2221.47726725, 2124.54056941,
       1996.03871629, 1854.89143353, 1718.53879797, 1599.14914582,
       1502.97122783, 1431.0218576 , 1380.55933525, 1346.97814697,
       1325.49943114, 1312.13831743, 1303.97872899, 1299.05347108,
       1296.10266693, 1294.34217288, 1293.29484759, 1292.67284408])

NMODEL = len(H)
NLAYER = 20

"""
A = 100
H = np.linspace(     0.     , 1404762.36466,num=A)
P = np.linspace(2.00000000e+06, 1.00000000e+02,num=A)
T = np.linspace(2294, 1292,num=A)

NMODEL = len(H)
NLAYER = NMODEL
"""

"""
### Reference Atmospheric Model Input
# Height in m
H = np.array([      0.     ,  10000, 15000])

# Pressure in pa, note 1 atm = 101325 pa
P = np.array([2.00000000e+06, 5e+05, 1e5])

# Temperature in Kelvin
T = np.array([2000, 1800, 1500])

NMODEL = len(H)
NLAYER = 3

### Reference Atmospheric Model Input
# Height in m
H = np.array([      0.     ,  10000])

# Pressure in pa, note 1 atm = 101325 pa
P = np.array([2.00000000e+06, 5e+05])

# Temperature in Kelvin
T = np.array([2000, 1500 ])

NMODEL = len(H)
NLAYER = 2
"""
# Ground temperature in Kelvin and path angle
T_ground = 0
path_angle = 45

# Gas Volume Mixing Ratio, constant with height
gas_id = np.array([  1, 2,  5,  6, 40, 39])
iso_id = np.array([0, 0, 0, 0, 0, 0])
H2_ratio = 1
VMR_H2O = 1.0E-4 # volume mixing ratio of H2O
VMR_CO2 = 1.0E-4  # volume mixing ratio of CO2
VMR_CO = 1.0E-4 # volume mixing ratio of CO
VMR_CH4 = 1.0E-4 # volume mixing ratio of CH4
VMR_He = (np.ones(NMODEL)-VMR_H2O-VMR_CO2-VMR_CO-VMR_CH4)*(1-H2_ratio)
VMR_H2 = (np.ones(NMODEL)-VMR_H2O-VMR_CO2-VMR_CO-VMR_CH4)*H2_ratio
NVMR = 6
VMR = np.zeros((NMODEL,NVMR))
VMR[:,0] = VMR_H2O
VMR[:,1] = VMR_CO2
VMR[:,2] = VMR_CO
VMR[:,3] = VMR_CH4
VMR[:,4] = VMR_He
VMR[:,5] = VMR_H2

### Benchmark Fortran forward model
folder_name = 'testing'
if not os.path.isdir(folder_name):
    os.mkdir(folder_name)
file_path = os.path.dirname(os.path.realpath(__file__))
os.chdir(file_path+'/'+folder_name) # move to designated process folder

API = Nemesis_api(name=folder_name, NLAYER=NLAYER, gas_id_list=gas_id,
    iso_id_list=iso_id, wave_grid=wave_grid)
API.write_files(path_angle=path_angle, H_model=H, P_model=P, T_model=T,
    VMR_model=VMR)
API.run_forward_model()
wave, yerr, point_spectrum_fo = API.read_output()
F_delH,F_totam,F_pres,F_temp,scaling = API.read_drv_file()

H_prf, P_prf, T_prf = API.read_prf_file()

### Benchmark Python forward model
H_hydro = np.loadtxt('{}.prf'.format(folder_name),skiprows=9,unpack=True,
    usecols=(0))
H_hydro*=1e3
FM = ForwardModel()
FM.set_planet_model(M_plt=M_plt, R_plt=R_plt, gas_id_list=gas_id,
    iso_id_list=iso_id, NLAYER=NLAYER)
FM.set_opacity_data(kta_file_paths=lowres_files, cia_file_path=cia_file_path)

"""
# use .ref file directly, might not be hydro adjusted
point_spectrum_py_old = FM.run_point_spectrum(H_model=H_hydro, P_model=P, T_model=T,\
            VMR_model=VMR, path_angle=path_angle, solspec=stellar_spec)
"""
# use .prf file, hydro adjusted
point_spectrum_py_old = FM.run_point_spectrum(H_model=H_prf, P_model=P_prf,
            T_model=T_prf, VMR_model=VMR, path_angle=path_angle, solspec=stellar_spec)

point_spectrum_py = FM.test_point_spectrum(U_layer=F_totam,P_layer=F_pres,
                        T_layer=F_temp, VMR_layer=VMR, del_S=F_delH,
                        scale=scaling, solspec=stellar_spec)
start = time.time()
for i in range(1000):
    point_spectrum_py = FM.test_point_spectrum(U_layer=F_totam,P_layer=F_pres,
                            T_layer=F_temp, VMR_layer=VMR, del_S=F_delH,
                            scale=scaling, solspec=stellar_spec)
end = time.time()

### Benchmark Juan's forward model
# os.system("python3 /Users/jingxuanyang/Desktop/uptodate/NemesisPy-dist/NemesisPy/Programs/nemesisPY.py < testing.nam")
# wave, yerr, juan_version = API.read_output()

### Compare output
# Fortran model plot

fig, axs = plt.subplots(nrows=2,ncols=1,sharex=True,
    dpi=800)
# figsize=[8.25,11.75]


axs[0].scatter(wave,point_spectrum_fo,marker='x',color='k',linewidth=1,s=10,
    label='fortran')
axs[0].plot(wave,point_spectrum_fo,color='k',linewidth=0.5)

# Python model plot
"""
axs[0].scatter(wave_grid, point_spectrum_py_old, marker='o', color='b',
    linewidth=1, s=10, label='python')
axs[0].plot(wave_grid, point_spectrum_py_old, color='b', linewidth=0.5)
"""

axs[0].scatter(wave_grid, point_spectrum_py, marker='.', color='y',
    linewidth=1, s=10, label='python')
axs[0].plot(wave_grid, point_spectrum_py, color='y', linewidth=0.5)
axs[0].scatter(wave_grid, point_spectrum_py_old, marker='.', color='r',
    linewidth=1, s=10, label='layer')
axs[0].plot(wave_grid, point_spectrum_py_old, color='r', linewidth=0.5)



# axs[0].scatter(wave_grid, juan_version, marker='.', color='r',
#     linewidth=1, s=10, label='juan_version')
# axs[0].plot(wave_grid, juan_version, color='r', linewidth=0.5)

axs[0].legend(loc='upper right')
axs[0].grid()
axs[0].set_ylabel(r'total radiance(W sr$^{-1}$ $\mu$m$^{-1})$')
axs[0].set_title('{:.0e}H2O {:.0e}CO2 {:.0e}CO {:.0e}CH4 {:.0e}He {:.0e}H2'.format(
    VMR_H2O,VMR_CO2,VMR_CO, VMR_CH4, VMR_He[0], VMR_H2[0]),fontsize=8)

axs[0].ticklabel_format(axis="y", style="sci", scilimits=(0,0))
# axs[0].set_yscale('log')
# axs[0].set_ylim(1e21*0.5,6e23)
# Plot diff
diff_1 = (point_spectrum_fo-point_spectrum_py)/point_spectrum_fo
# diff_2 = (point_spectrum_fo-juan_version)/point_spectrum_fo
axs[1].scatter(wave_grid,(point_spectrum_fo-point_spectrum_py)/point_spectrum_fo,
    marker='.',color='y',label='diff (mine)')

# axs[1].scatter(wave_grid,(point_spectrum_fo-juan_version)/point_spectrum_fo,
#     marker='.',color='r',label='diff (juan)')

print('diff between python and fortran with same inputs',diff_1,np.amax(abs(diff_1)))
# print('diff between juan and Fortran',diff_2,np.amax(abs(diff_2)))
print('fortran',point_spectrum_fo)
print('python',point_spectrum_py)

axs[1].legend(loc='lower left')
axs[1].grid()
plt.grid()

diff_3 = (point_spectrum_fo-point_spectrum_py_old)/point_spectrum_fo
print('diff between python and fortran (own layering)',diff_3,np.amax(abs(diff_3)))
axs[1].scatter(wave_grid,(point_spectrum_fo-point_spectrum_py_old)/point_spectrum_fo,
    marker='.',color='r',label='diff (layer)')

# Plot config
plt.xlabel(r'wavelength($\mu$m)')
plt.ticklabel_format(axis="y", style="sci", scilimits=(0,0))
plt.tight_layout()

plt.legend()
plt.grid()

plt.tight_layout()
plt.savefig('{:.0e}H2O_{:.0e}CO2_{:.0e}CO_{:.0e}CH4_{:.0e}He_{:.0e}H2.pdf'.format(
    VMR_H2O,VMR_CO2,VMR_CO, VMR_CH4, VMR_He[0], VMR_H2[0]),dpi=400)
plt.savefig('comparison.pdf',dpi=400)
plt.show()

print('run time = ', end - start)


# Pure H2 He atm, H2 ratio 0.85
"""
driver file fix

Before unifying input amounts, (fortran - python)/fortran
diff1 = np.array([0.03534349, 0.03595834, 0.03827849, 0.03773726, 0.03565747,
       0.03547866, 0.03719903, 0.04009602, 0.0433155 , 0.04650146,
       0.04919436, 0.05115559, 0.05247834, 0.05311147, 0.05290781,
       0.00943529, 0.00149393])

After unifying input
diff2 = np.array([0.03551326, 0.03613319, 0.03846284, 0.03792167, 0.03583657,
       0.03565834, 0.03738532, 0.04029268, 0.04352485, 0.04672423,
       0.04943028, 0.05140472, 0.05273983, 0.05338396, 0.05318995,
       0.00959471, 0.00164575])
"""
"""
T_ground fix
"""
"""
INORMAL fix
Python had inormal set to 0, whereas fortran had inormal set to 1

inormal = 0
diff
array([0.02823954, 0.02992544, 0.03319957, 0.0333664 , 0.03183606,
       0.03217831, 0.03439334, 0.03774233, 0.04135752, 0.04488327,
       0.04786373, 0.05006302, 0.05158092, 0.0523718 , 0.0522957 ,
       0.00942252, 0.00149046])

inormal = 1
array([0.02823954, 0.02992544, 0.03319957, 0.0333664 , 0.03183606,
       0.03217831, 0.03439334, 0.03774233, 0.04135752, 0.04488327,
       0.04786373, 0.05006302, 0.05158092, 0.0523718 , 0.0522957 ,
       0.00942252, 0.00149046])
"""

"""
Change IRAY fix
Pytho IRAY is on
Fortran IRAY = 1 (rayleigh is on)
np.array([0.02823954, 0.02992544, 0.03319957, 0.0333664 , 0.03183606,
       0.03217831, 0.03439334, 0.03774233, 0.04135752, 0.04488327,
       0.04786373, 0.05006302, 0.05158092, 0.0523718 , 0.0522957 ,
       0.00942252, 0.00149046])
Fortran IRAY = 0 (rayleigh is off)
np.array([0.03534349, 0.03595834, 0.03827849, 0.03773726, 0.03565747,
       0.03547866, 0.03719903, 0.04009602, 0.0433155 , 0.04650146,
       0.04919436, 0.05115559, 0.05247834, 0.05311147, 0.05290781,
       0.00943529, 0.00149393])
Behaved as expected. Rayleight scattering significant. Rayleigh scattering
reduces emission.
"""

"""
star spectrum fix

real spec:
diff1 = np.array([0.03534349, 0.03595834, 0.03827849, 0.03773726, 0.03565747,
       0.03547866, 0.03719903, 0.04009602, 0.0433155 , 0.04650146,
       0.04919436, 0.05115559, 0.05247834, 0.05311147, 0.05290781,
       0.00943529, 0.00149393])
fake spec (ones):
diff2 = np.array([0.03534342, 0.03595814, 0.03827849, 0.03773715, 0.03565704,
       0.03547839, 0.0371991 , 0.04009606, 0.04331519, 0.04650137,
       0.04919452, 0.05115556, 0.05247831, 0.05311143, 0.05290782,
       0.00943512, 0.00149404])
"""






"""
[ 5.25967581e-06 -2.76723438e-04 -4.14992082e-04 -8.25601241e-05
 -6.96366419e-05 -8.37985996e-05 -3.53954143e-04 -4.90156681e-04
  4.78637269e-05  1.59335672e-03  2.98267603e-05 -3.07812821e-04
  1.91729393e-04 -1.25652374e-03 -1.50401884e-03  1.68878016e-04
 -2.05801711e-02]  0.020580171078122943 # np.argsort

[ 5.25967581e-06 -4.44398519e-04 -9.22294498e-05 -8.25601241e-05
 -6.96366419e-05 -8.37985996e-05 -3.53954143e-04 -4.90156681e-04
  4.78637269e-05  1.59335672e-03  2.98267603e-05 -3.07812821e-04
  1.91729393e-04 -1.25652374e-03 -1.50401884e-03  1.68878016e-04
 -2.05801711e-02] 0.020580171078122943 # fortran

"""
