import matplotlib.pyplot as plt
import numpy as np
import time
# Import constants, opacity data, and ForwardModel class
from nemesispy.common.constants import *
from nemesispy.data.helper import lowres_file_paths, cia_file_path
from nemesispy.radtran.forward_model import ForwardModel
# Import GCM data to use as atmospheric model
from nemesispy.data.gcm.process_gcm import (nlon,nlat,xlon,xlat,npv,pv,\
    tmap,h2omap,comap,co2map,ch4map,hemap,h2map,vmrmap,\
    tmap_mod,h2omap_mod,comap_mod,co2map_mod,ch4map_mod,\
    hemap_mod,h2map_mod,vmrmap_mod,phase_grid,\
    kevin_phase_by_wave,kevin_wave_by_phase,\
    pat_phase_by_wave,pat_wave_by_phase,\
    vmrmap_mod_new,tmap_hot)

### The code below calculate the disc integrated emission spectrum at an
# arbitary orbital phase give a 3D atmospheric model based on a GCM.
# We model the HST/WFC3 and Spitzer/IRAC emission spectrum of WASP-43b published
# in Stevenson et al. 2018.
# The GCM data is provided by Vivien Parmentier and used in Irwin et al. 2020
# and Yang et al. 2023.

print('creating example orbital phase resolved emission spectrum...')
# Wavelengths grid for the emission spectrum
wave_grid = np.array(
    [1.1425, 1.1775, 1.2125, 1.2475, 1.2825, 1.3175, 1.3525, 1.3875,
    1.4225, 1.4575, 1.4925, 1.5275, 1.5625, 1.5975, 1.6325, 3.6   ,
    4.5   ]
    )
nwave = len(wave_grid)
# Orbital phase grid
phase_grid = np.array(
    [ 22.5,  45. ,  67.5,  90. , 112.5, 135. , 157.5, 180. , 202.5,
    225. , 247.5, 270. , 292.5, 315. , 337.5]
    )
nphase = len(phase_grid)
# WASP-43 stellar spectrum to convert flux to Fp/Fs
wasp43_spec = np.array(
    [3.341320e+25, 3.215455e+25, 3.101460e+25, 2.987110e+25,
    2.843440e+25, 2.738320e+25, 2.679875e+25, 2.598525e+25,
    2.505735e+25, 2.452230e+25, 2.391140e+25, 2.345905e+25,
    2.283720e+25, 2.203690e+25, 2.136015e+25, 1.234010e+24,
    4.422200e+23]
    )
# Pick an orbital phase
phasenumber = 3 # secondary eclipse
phase = phase_grid[phasenumber]
# Pick resolution for the disc average
nmu = 5 # Number of mu bins
# Reference planetary parameters
M_plt = 2.034 * M_JUP # kg
R_plt = 1.036 * R_JUP_E # m
# List of gas species to include in the model
gas_id = np.array([1, 2,  5,  6, 40, 39]) # H2O, CO2, CO, CH4, H2, He
iso_id = np.array([0, 0, 0, 0, 0, 0]) # Isotopologue identifier
# Define the atmospheric model
NLAYER = 40 # Number of layers
top_pressure = 100 # Top pressure in Pa
botttom_pressure = 20e5 # Bottom pressure in Pa
P_model = np.geomspace(botttom_pressure,top_pressure,NLAYER) # Pressure grid in Pa

# Set up forward model
FM = ForwardModel()
FM.set_planet_model(
    M_plt=M_plt,R_plt=R_plt,
    gas_id_list=gas_id,iso_id_list=iso_id,
    NLAYER=NLAYER
    )
FM.set_opacity_data(
    kta_file_paths=lowres_file_paths,
    cia_file_path=cia_file_path
    )

# The first time a forward model is run, it takes longer to run because as the
# just-in-time compiler compiles Python code into machine code.
print('start just-in-time compilation...')
start1 = time.time()
one_phase =  FM.calc_disc_spectrum(phase=phase, nmu=nmu, P_model = P_model,
    global_model_P_grid=pv,
    global_T_model=tmap_mod, global_VMR_model=vmrmap_mod,
    mod_lon=xlon,
    mod_lat=xlat,
    solspec=wasp43_spec)
end1 = time.time()
print('compilation + run time = ', end1-start1)

# After the just-in-time compilation, the forward model runs MUCH faster.
start2 = time.time()
one_phase =  FM.calc_disc_spectrum(phase=phase, nmu=nmu, P_model = P_model,
    global_model_P_grid=pv,
    global_T_model=tmap_mod, global_VMR_model=vmrmap_mod,
    mod_lon=xlon,
    mod_lat=xlat,
    solspec=wasp43_spec)
end2 = time.time()
print('run time = ', end2-start2)

# Plot the results
# We compare the emission spectrum calculated using the GCM model from NemesiPy
# with the emission spectrum calculated using the GCM model from NEMESIS,
# and the observed data from Stevenson et al. 2018.
fig, axs = plt.subplots(nrows=2,ncols=1,sharex=True,
    dpi=200)
axs[0].set_title('phase = {}'.format(phase))
axs[0].plot(wave_grid,one_phase,color='b',
    linewidth=0.5,linestyle='--',
    marker='x',markersize=2,label='GCM model (NemesiPy)')
axs[0].plot(wave_grid,pat_phase_by_wave[phasenumber],color='k',
    linewidth=0.5,linestyle='-',
    marker='x',markersize=2,label='GCM model (NEMESIS)')
axs[0].scatter(
    wave_grid,kevin_phase_by_wave[phasenumber,:,0],
    color='r',marker='+',label='Stevenson+ Data')
axs[0].legend()
axs[0].grid()
axs[1].set_title('Relative difference between NemesiPy and NEMESIS')
diff = (one_phase - pat_phase_by_wave[phasenumber,:])/one_phase
axs[1].scatter(wave_grid,diff,marker='.',color='b')
axs[1].grid()
plt.savefig('emission_example.pdf')

# # The routine below calculate the average run time of the forward model
# start2 = time.time()
# niter = 10
# for i in range(niter):
#     one_phase =  FM.calc_disc_spectrum(phase=phase, nmu=nmu, P_model = P_model,
#         global_model_P_grid=pv,
#         global_T_model=tmap_mod, global_VMR_model=vmrmap_mod,
#         mod_lon=xlon,
#         mod_lat=xlat,
#         solspec=wasp43_spec)
# end2 = time.time()
# print('average run time = ',(end2-start2)/niter)
