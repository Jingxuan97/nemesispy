import matplotlib.pyplot as plt
import numpy as np
import time
from nemesispy.radtran.forward_model import ForwardModel
# Read GCM data
from nemesispy.data.gcm.process_gcm import (nlon,nlat,xlon,xlat,npv,pv,\
    tmap,h2omap,comap,co2map,ch4map,hemap,h2map,vmrmap,\
    tmap_mod,h2omap_mod,comap_mod,co2map_mod,ch4map_mod,\
    hemap_mod,h2map_mod,vmrmap_mod,phase_grid,\
    kevin_phase_by_wave,kevin_wave_by_phase,\
    pat_phase_by_wave,pat_wave_by_phase,\
    vmrmap_mod_new,tmap_hot)

import os
from nemesispy.disc_benchmarking_fortran import Nemesis_api
import time

from nemesispy.common.helper import lowres_file_paths, cia_file_path

### Wavelengths grid and orbital phase grid
wave_grid = np.array([1.1425, 1.1775, 1.2125, 1.2475, 1.2825, 1.3175, 1.3525, 1.3875,
       1.4225, 1.4575, 1.4925, 1.5275, 1.5625, 1.5975, 1.6325, 3.6   ,
       4.5   ])
phase_grid = np.array([ 22.5,  45. ,  67.5,  90. , 112.5, 135. , 157.5, 180. , 202.5,
       225. , 247.5, 270. , 292.5, 315. , 337.5])
nwave = len(wave_grid)
nphase = len(phase_grid)
wasp43_spec = np.array([3.341320e+25, 3.215455e+25, 3.101460e+25, 2.987110e+25,
       2.843440e+25, 2.738320e+25, 2.679875e+25, 2.598525e+25,
       2.505735e+25, 2.452230e+25, 2.391140e+25, 2.345905e+25,
       2.283720e+25, 2.203690e+25, 2.136015e+25, 1.234010e+24,
       4.422200e+23])

### Reference Planet Input
M_plt = 3.8951064000000004e+27 # kg
R_plt = 74065.70 * 1e3 # m
gas_id = np.array([  1, 2,  5,  6, 40, 39])
iso_id = np.array([0, 0, 0, 0, 0, 0])
NLAYER = 20
phasenumber = 3
nmu = 5
phase = phase_grid[phasenumber]
P_model = np.geomspace(20e5,100,NLAYER)

f = open('runtime.txt','a')
f.write('Nrun, Ptime, Ftime\n')
f.close()

py_start_time = time.time()
### Set up Python forward model
FM_py = ForwardModel()
FM_py.set_planet_model(M_plt=M_plt,R_plt=R_plt,gas_id_list=gas_id,iso_id_list=iso_id,
    NLAYER=NLAYER)
FM_py.set_opacity_data(kta_file_paths=lowres_file_paths, cia_file_path=cia_file_path)
### Testing one particular orbital phase (inhomogeneouus disc averaging)
one_phase =  FM_py.calc_disc_spectrum(phase=phase, nmu=nmu, P_model = P_model,
    global_model_P_grid=pv,
    global_T_model=tmap_mod, global_VMR_model=vmrmap_mod,
    mod_lon=xlon,
    mod_lat=xlat,
    solspec=wasp43_spec)
py_end_time = time.time()
py_run_time = py_end_time - py_start_time
f = open('runtime.txt','a')
f.write('{}, {}, '.format(1,py_run_time))
f.close()

### Set up Fortran forward model
folder_name = 'runtime'
FM_fo = Nemesis_api(name=folder_name, NLAYER=NLAYER, gas_id_list=gas_id,
    iso_id_list=iso_id, wave_grid=wave_grid)
if not os.path.isdir(folder_name):
    os.mkdir(folder_name)
file_path = os.path.dirname(os.path.realpath(__file__))
os.chdir(file_path+'/'+folder_name) # move to designated process folder

f_start_time = time.time()
one_phase =  FM_py.calc_disc_spectrum(phase=phase, nmu=nmu, P_model = P_model,
    global_model_P_grid=pv,
    global_T_model=tmap_mod, global_VMR_model=vmrmap_mod,
    mod_lon=xlon,
    mod_lat=xlat,
    solspec=wasp43_spec)
f_end_time = time.time()
f_run_time = f_end_time-f_start_time
f = open('runtime.txt','a')
f.write('{}\n'.format(f_run_time))
f.close()


H = np.linspace(0,1e4,NLAYER)
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

# Gas Volume Mixing Ratio, constant with height
gas_id = np.array([  1, 2,  5,  6, 40, 39])
iso_id = np.array([0, 0, 0, 0, 0, 0])
H2_ratio = 0.864
VMR_H2O = 1.0E-4
VMR_CO2 = 1.0E-4
VMR_CO = 1.0E-4
VMR_CH4 = 1.0E-4
VMR_He = (np.ones(NLAYER)-VMR_H2O-VMR_CO2-VMR_CO-VMR_CH4)*(1-H2_ratio)
VMR_H2 = (np.ones(NLAYER)-VMR_H2O-VMR_CO2-VMR_CO-VMR_CH4)*H2_ratio
NVMR = 6
VMR = np.zeros((NLAYER,NVMR))
VMR[:,0] = VMR_H2O
VMR[:,1] = VMR_CO2
VMR[:,2] = VMR_CO
VMR[:,3] = VMR_CH4
VMR[:,4] = VMR_He
VMR[:,5] = VMR_H2

run_list = [2,4,8,16,32,64,128]
for Nrun in run_list:
    py_start_time = time.time()
    for Niter in range(int(Nrun)):
        one_phase =  FM_py.calc_disc_spectrum(phase=phase, nmu=nmu, P_model = P_model,
            global_model_P_grid=pv,
            global_T_model=tmap_mod, global_VMR_model=vmrmap_mod,
            mod_lon=xlon,
            mod_lat=xlat,
            solspec=wasp43_spec)
    py_end_time = time.time()
    py_run_time = py_end_time - py_start_time

    f_start_time = time.time()
    for Niter in range(int(Nrun)):
        FM_fo.write_files(NRING=nmu, H_model=H, P_model=P, T_model=T,
            VMR_model=VMR)
        FM_fo.run_forward_model()
    f_end_time = time.time()
    f_run_time = f_end_time - f_start_time

    f = open('runtime.txt','a')
    f.write('{}, {}, {}\n'.format(Nrun,py_run_time,f_run_time))
    f.close()



# fig, axs = plt.subplots(nrows=2,ncols=1,sharex=True,
#     dpi=100)
# axs[0].set_title('phase = {}'.format(phase))
# axs[0].plot(wave_grid,one_phase,color='b',
#     linewidth=0.5,linestyle='--',
#     marker='x',markersize=2,label='Python')
# axs[0].scatter(wave_grid,kevin_phase_by_wave[phasenumber,:,0],color='r',marker='+',label='Data')
# axs[0].plot(wave_grid,pat_phase_by_wave[phasenumber],color='k',
#     linewidth=0.5,linestyle='-',
#     marker='x',markersize=2,label='Fortran')
# axs[0].legend()
# axs[0].grid()

# diff = (one_phase - pat_phase_by_wave[phasenumber,:])/one_phase
# axs[1].scatter(wave_grid,diff,marker='.',color='b')
# axs[1].grid()
# print(diff)

# plt.savefig('eg.pdf')
