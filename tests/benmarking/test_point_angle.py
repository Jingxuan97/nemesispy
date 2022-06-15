import numpy as np
import matplotlib.pyplot as plt
import os
import scipy.interpolate
from nemesispy.radtran.forward_model import ForwardModel
from point_benchmarking_fortran import Nemesis_api
import time

folder_name = 'test_point_angle'

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
NMODEL = 200
NLAYER = NMODEL
# Pressure in pa, note 1 atm = 101325 pa
H = np.linspace(0,1e4,NMODEL)

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

f_PT = scipy.interpolate.interp1d(P,T)

P = np.geomspace(20*1e5,1e-3*1e5,NMODEL)
T = f_PT(P)

# Gas Volume Mixing Ratio, constant with height
gas_id = np.array([  1, 2,  5,  6, 40, 39])
iso_id = np.array([0, 0, 0, 0, 0, 0])
H2_ratio = 0.864
VMR_H2O = 1.0E-4
VMR_CO2 = 1.0E-4
VMR_CO = 1.0E-4
VMR_CH4 = 1.0E-4
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

# Ground temperature in Kelvin and path angle
path_angle_list = [0,30,60,90]

### Set up Python forward model
FM_py = ForwardModel()
FM_py.set_planet_model(M_plt=M_plt, R_plt=R_plt, gas_id_list=gas_id,
    iso_id_list=iso_id, NLAYER=NLAYER)
FM_py.set_opacity_data(
    kta_file_paths=lowres_file_paths,
    cia_file_path=cia_file_path
    )

### Set up Fortran forward model
FM_fo = Nemesis_api(name=folder_name, NLAYER=NLAYER, gas_id_list=gas_id,
    iso_id_list=iso_id, wave_grid=wave_grid)
if not os.path.isdir(folder_name):
    os.mkdir(folder_name)
file_path = os.path.dirname(os.path.realpath(__file__))
os.chdir(file_path+'/'+folder_name) # move to designated process folder

### Set up figure
fig, axs = plt.subplots(nrows=4,ncols=2,sharex='all',sharey='col',
    figsize=[8,10],dpi=600)
# add a big axis, hide frame
fig.add_subplot(111,frameon=False)
# hide tick and tick label of the big axis
plt.tick_params(labelcolor='none',which='both',
    top=False,bottom=False,left=False,right=False)
plt.xlabel(r'Wavelength ($\mu$m)')

for ipath, path_angle in enumerate(path_angle_list):

    spec_py = FM_py.calc_point_spectrum_hydro(
        P_model=P, T_model=T, VMR_model=VMR, path_angle=path_angle,
        solspec=wasp43_spec)

    FM_fo.write_files(path_angle=path_angle, H_model=H, P_model=P, T_model=T,
        VMR_model=VMR)
    FM_fo.run_forward_model()
    wave, yerr, spec_fo = FM_fo.read_output()
    F_delH,F_totam,F_pres,F_temp,scaling = FM_fo.read_drv_file()
    H_prf, P_prf, T_prf = FM_fo.read_prf_file()

    ## Upper panel
    # Fortran plot
    axs[ipath,0].plot(wave_grid,spec_fo,color='k',linewidth=0.5,linestyle='-',
        marker='x',markersize=2,label='Fortran')

    # Python model plot
    axs[ipath,0].plot(wave_grid, spec_py,color='b',linewidth=0.5,linestyle='--',
        marker='x',markersize=2,label='Python')

    # Mark the path angle
    axs[ipath,0].text(1.3,2.5e-3,r"$\theta$={}$^\circ$".format(path_angle))

    ## Upper panel format
    axs[ipath,0].ticklabel_format(axis="y", style="sci", scilimits=(0,0))
    # axs[ipath,0].legend(loc='upper left')
    axs[ipath,0].grid()
    axs[ipath,0].set_ylabel('Flux ratio')
    handles, labels = axs[ipath,0].get_legend_handles_labels()

    ## Lower panel
    diff = (spec_fo-spec_py)/spec_fo
    axs[ipath,1].plot(wave_grid, diff, color='r', linewidth=0.5,linestyle=':',
        marker='x',markersize=2,)
    axs[ipath,1].set_ylabel('Relative residual')
    axs[ipath,1].set_ylim(-1e-2,1e-2)

    ## Lower panel format
    axs[ipath,1].ticklabel_format(axis="y", style="sci", scilimits=(0,0))
    axs[ipath,1].grid()

    # Plot config

plt.tight_layout()

fig.legend(handles, labels, loc='upper left', fontsize='x-small')

plt.savefig('{}.pdf'.format(folder_name))