import numpy as np
import os
import matplotlib.pyplot as plt

from nemesispy.radtran.forward_model import ForwardModel
from disc_benchmarking_fortran import Nemesis_api
import time

folder_name = 'discspec'

### Reference Opacity Data
from helper import lowres_file_paths, cia_file_path

### Reference Planet Input
M_plt = 3.8951064000000004e+27 # kg
R_plt = 74065.70 * 1e3 # m

### Reference Spectral Input
# Wavelength grid in micron
wave_grid = np.array([1.1425, 1.1775, 1.2125, 1.2475, 1.2825, 1.3175, 1.3525, 1.3875,
       1.4225, 1.4575, 1.4925, 1.5275, 1.5625, 1.5975, 1.6325, 3.6   ,
       4.5   ])

# Stellar spectrum
wasp43_spec = np.array([3.341320e+25, 3.215455e+25, 3.101460e+25, 2.987110e+25,
       2.843440e+25, 2.738320e+25, 2.679875e+25, 2.598525e+25,
       2.505735e+25, 2.452230e+25, 2.391140e+25, 2.345905e+25,
       2.283720e+25, 2.203690e+25, 2.136015e+25, 1.234010e+24,
       4.422200e+23])

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

# Phase for disc averaging
orbital_phase = 0
central_long = 360 - orbital_phase

# Gas Volume Mixing Ratio, constant with height
gas_id = np.array([  1, 2,  5,  6, 40, 39])
iso_id = np.array([0, 0, 0, 0, 0, 0])
H2_ratio = 0.864
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

###############################################################################
### Run Fortran forward model
# Create nemesis folder
if not os.path.isdir(folder_name):
    os.mkdir(folder_name)
file_path = os.path.dirname(os.path.realpath(__file__))
os.chdir(file_path+'/'+folder_name) # move to designated process folder
# Initialise python FM_fo
FM_fo = Nemesis_api(name=folder_name, NLAYER=NLAYER, gas_id_list=gas_id,
    iso_id_list=iso_id, wave_grid=wave_grid)
# Run Fortran code
F_start = time.time()
FM_fo.write_files(path_angle=0, H_model=H, P_model=P, T_model=T,
    VMR_model=VMR)
FM_fo.run_forward_model()
wave, yerr, fortran_disc_spec = FM_fo.read_output()
F_end = time.time()
# Read Output files
F_delH, F_totam, F_pres, F_temp, F_scaling = FM_fo.read_drv_file()
H_prf, P_prf, T_prf = FM_fo.read_prf_file()

###############################################################################
### Run Python forward model
# Initialise python forward model
FM_py = ForwardModel()
FM_py.set_planet_model(M_plt=M_plt, R_plt=R_plt, gas_id_list=gas_id,
    iso_id_list=iso_id, NLAYER=NLAYER)
FM_py.set_opacity_data(kta_file_paths=lowres_file_paths,
    cia_file_path=cia_file_path)
python_disc_spec = FM_py.calc_disc_spectrum_uniform(nmu=2,
    P_model=P, T_model=T, VMR_model=VMR, solspec=wasp43_spec)


###############################################################################
### Compare output
### Set up figure
fig, axs = plt.subplots(nrows=2,ncols=2,sharex='all',sharey='col',
    figsize=[8,10],dpi=600)
# add a big axis, hide frame
fig.add_subplot(111,frameon=False)
# hide tick and tick label of the big axis
plt.tick_params(labelcolor='none',which='both',
    top=False,bottom=False,left=False,right=False)
plt.xlabel(r'Wavelength ($\mu$m)')

# Fortran model plot
axs[0,0].plot(wave_grid,fortran_disc_spec,color='k',linewidth=0.5,linestyle='-',
    marker='x',markersize=2,label='Fortran')

# Python model plot
# plot spectrum with profile in .ref, test radtran, layering and hydrostatic
axs[0,0].plot(wave_grid,python_disc_spec,color='b',linewidth=0.5,linestyle='--',
    marker='x',markersize=2,label='Python')

# Plot specs
axs[0,0].legend(loc='upper right')
axs[0,0].grid()
axs[0,0].set_ylabel(r'total radiance(W sr$^{-1}$ $\mu$m$^{-1})$')
axs[0,0].ticklabel_format(axis="y", style="sci", scilimits=(0,0))

# Plot differences
diff = (fortran_disc_spec-python_disc_spec)/fortran_disc_spec
axs[0,1].scatter(wave_grid, diff,marker='.', color='b',label='lay + hydro')

# Plot config
plt.xlabel(r'wavelength($\mu$m)')
plt.ticklabel_format(axis="y", style="sci", scilimits=(0,0))
plt.tight_layout()

plt.legend()
plt.grid()

plt.tight_layout()

plt.savefig('{}.pdf'.format(folder_name))
