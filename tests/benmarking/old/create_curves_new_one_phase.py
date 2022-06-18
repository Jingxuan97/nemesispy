import sys
# from nemesispy.radtran.disc_benchmarking import NLAYER
sys.path.append('/Users/jingxuanyang/Desktop/Workspace/nemesispy2022/')
import matplotlib.pyplot as plt
import numpy as np
from nemesispy.common.constants import R_SUN, R_JUP_E, AMU, AU, M_JUP, R_JUP
from nemesispy.AAwaitlist.utils import calc_mmw
from nemesispy.models.models import Model2
from nemesispy.radtran.read import read_kls
from nemesispy.radtran.calc_radiance import calc_radiance, calc_planck
from nemesispy.radtran.read import read_cia
from nemesispy.common.calc_trig import gauss_lobatto_weights
from nemesispy.radtran.forward_model import ForwardModel
import time
# from nemesispy.radtran.runner import interpolate_to_lat_lon

### Opacity data
lowres_files = ['/Users/jingxuanyang/ktables/h2owasp43.kta',
'/Users/jingxuanyang/ktables/cowasp43.kta',
'/Users/jingxuanyang/ktables/co2wasp43.kta',
'/Users/jingxuanyang/ktables/ch4wasp43.kta']
cia_file_path='/Users/jingxuanyang/Desktop/Workspace/nemesispy2022/nemesispy/data/cia/exocia_hitran12_200-3800K.tab'

### Reference Planet Input
M_plt = 3.8951064000000004e+27 # kg
R_plt = 74065.70 * 1e3  # m
gas_id = np.array([  1, 2,  5,  6, 40, 39])
iso_id = np.array([0, 0, 0, 0, 0, 0])
NLAYER = 80

################################################################################
################################################################################
# Read GCM data
from nemesispy.data.gcm.process_gcm import (nlon,nlat,xlon,xlat,npv,pv,pvmap,\
    tmap,h2omap,comap,co2map,ch4map,hemap,h2map,vmrmap,hvmap,\
    tmap_mod,h2omap_mod,comap_mod,co2map_mod,ch4map_mod,\
    hemap_mod,h2map_mod,vmrmap_mod,hvmap_mod,phase_grid,\
    kevin_phase_by_wave,kevin_wave_by_phase,\
    pat_phase_by_wave,pat_wave_by_phase)



start = time.time()
### Set up forward model
FM = ForwardModel()
FM.set_planet_model(M_plt=M_plt,R_plt=R_plt,gas_id_list=gas_id,iso_id_list=iso_id,
    NLAYER=NLAYER)
FM.set_opacity_data(kta_file_paths=lowres_files, cia_file_path=cia_file_path)

### Code to actually simulate a phase curve
wave_grid = np.array([1.1425, 1.1775, 1.2125, 1.2475, 1.2825, 1.3175, 1.3525, 1.3875,
       1.4225, 1.4575, 1.4925, 1.5275, 1.5625, 1.5975, 1.6325, 3.6   ,
       4.5   ])
phase_grid = np.array([ 22.5,  45. ,  67.5,  90. , 112.5, 135. , 157.5, 180. , 202.5,
       225. , 247.5, 270. , 292.5, 315. , 337.5])
nwave = len(wave_grid)
nphase = len(phase_grid)

my_gcm_phase_by_wave = np.zeros((nphase,nwave))
my_gcm_wave_by_phase = np.zeros((nwave,nphase))

wasp43_spec = np.array([3.341320e+25, 3.215455e+25, 3.101460e+25, 2.987110e+25,
       2.843440e+25, 2.738320e+25, 2.679875e+25, 2.598525e+25,
       2.505735e+25, 2.452230e+25, 2.391140e+25, 2.345905e+25,
       2.283720e+25, 2.203690e+25, 2.136015e+25, 1.234010e+24,
       4.422200e+23])

iphase = 0
phasenumber = iphase
phase = phase_grid[phasenumber]

one_phase =  FM.calc_disc(phase, nmu=5, global_H_model=hvmap,
    global_P_model=pvmap,global_T_model=tmap, global_VMR_model=vmrmap,
    mod_lon=xlon,
    mod_lat=xlat,
    solspec=wasp43_spec)
    # my_gcm_phase_by_wave[iphase,:] = one_phase

# new_one_phase =  FM.calc_disc_spectrum(phase, nmu=5, global_H_model=new_hvmap,
#     global_P_model=new_pvmap,global_T_model=new_tmap, global_VMR_model=new_vmrmap,
#     global_model_longitudes=xlon,
#     global_model_lattitudes=xlat,
#     solspec=wasp43_spec)
#     # my_gcm_phase_by_wave[iphase,:] = one_phase



fig, axs = plt.subplots(nrows=2,ncols=1,sharex=True,
    dpi=800)
axs[0].set_title('phase = {}'.format(phase))
axs[0].plot(wave_grid,one_phase,color='b',label='Python old')
# axs[0].plot(wave_grid,new_one_phase,color='y',label='Python new')
axs[0].scatter(wave_grid,kevin_phase_by_wave[phasenumber,:,0],color='r',marker='+',label='Data')
axs[0].plot(wave_grid,pat_phase_by_wave[phasenumber,:],color ='k',label='Fortran')
axs[0].legend()
axs[0].grid()

diff = (one_phase - pat_phase_by_wave[phasenumber,:])/one_phase
axs[1].scatter(wave_grid,diff,marker='.',color='b')
axs[1].grid()

plt.tight_layout()
plt.savefig('discav_phase{}.pdf'.format(iphase),dpi=400)
plt.show()
