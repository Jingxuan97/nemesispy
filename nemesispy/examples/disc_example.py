import matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os
import time
from nemesispy.common.constants import G
from nemesispy.radtran.forward_model import ForwardModel

matplotlib.interactive(True)
# Read GCM data
from nemesispy.data.gcm.process_gcm import (nlon,nlat,xlon,xlat,npv,pv,\
    tmap,h2omap,comap,co2map,ch4map,hemap,h2map,vmrmap,\
    tmap_mod,h2omap_mod,comap_mod,co2map_mod,ch4map_mod,\
    hemap_mod,h2map_mod,vmrmap_mod,phase_grid,\
    kevin_phase_by_wave,kevin_wave_by_phase,\
    pat_phase_by_wave,pat_wave_by_phase,\
    vmrmap_mod_new,tmap_hot)

from nemesispy.data.helper import lowres_file_paths, cia_file_path

print('creating example phase curve')
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
NITER = 1

T_star = 4520 # star temperature in K
R_star = 463892759.99999994 # m, 0.6668 * R_SUN
SMA = 2243970000.0 # m, 0.015*AU
M_plt = 3.8951064000000004e+27 # kg
R_plt = 74065.70 * 1e3 # m
T_irr = T_star * (R_star/SMA)**0.5 # 2055 K
T_eq = T_irr/2**0.5 # 1453 K
g = G*M_plt/R_plt**2 # 47.39 ms-2
gas_id = np.array([  1, 2,  5,  6, 40, 39])
iso_id = np.array([0, 0, 0, 0, 0, 0])


### Set up forward model
FM = ForwardModel()
FM.set_planet_model(M_plt=M_plt,R_plt=R_plt,gas_id_list=gas_id,iso_id_list=iso_id,
    NLAYER=NLAYER)
FM.set_opacity_data(kta_file_paths=lowres_file_paths, cia_file_path=cia_file_path)

log_kappa_day = -2.2
log_gamma_day = -1
log_f_day = - 1
T_int_day = 200

log_kappa_night = -4
log_gamma_night = 0
log_f_night = -2
T_int_night = 200

h2o = 1e-4
co2 = 1e-4
co = 1e-4
ch4 = 1e-4
h2_frac = 0.84
he_frac = 1 - h2_frac
vmr_grid = np.ones((NLAYER,6))
vmr_grid[:,0] *= 10**h2o
vmr_grid[:,1] *= 10**co2
vmr_grid[:,2] *= 10**co
vmr_grid[:,3] *= 10**ch4
vmr_grid[:,4] *= he_frac * (1-10**h2o-10**co2-10**co-10**ch4)
vmr_grid[:,5] *= h2_frac * (1-10**h2o-10**co2-10**co-10**ch4)

from nemesispy.models.TP_profiles import TP_Guillot
T_day =  TP_Guillot(P_model,g,T_eq,10**log_kappa_day,10**log_gamma_day,
    10**log_f_day,T_int_day)
T_night =  TP_Guillot(P_model,g,T_eq,10**log_kappa_night,10**log_gamma_night,
    10**log_f_night,T_int_night)
spec1 = FM.calc_disc_spectrum_uniform(nmu, P_model,T_day,vmr_grid)
print(spec1)

phase=180
daymin=-90
daymax=90
spec2 = FM.calc_disc_spectrum_2tp(phase,nmu,daymin,daymax,
    P_model, T_day,T_night,vmr_grid)
print(spec2)

s = time.time()
spec2 = FM.calc_disc_spectrum_2tp(phase,nmu,daymin,daymax,
    P_model, T_day,T_night,vmr_grid)
e = time.time()
print('time 2tp',e-s)

s = time.time()
spec1 = FM.calc_disc_spectrum_uniform(nmu, P_model,T_day,vmr_grid)
e = time.time()
print('time disc',e-s)