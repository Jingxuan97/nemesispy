import sys

# from nemesispy.radtran.disc_benchmarking import NLAYER
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
from nemesispy.radtran.trig import gauss_lobatto_weights, interpolate_to_lat_lon
from nemesispy.radtran.forward_model import ForwardModel
import time
# from nemesispy.radtran.runner import interpolate_to_lat_lon

### Opacity data
lowres_files = ['/Users/jingxuanyang/ktables/h2owasp43.kta',
'/Users/jingxuanyang/ktables/cowasp43.kta',
'/Users/jingxuanyang/ktables/co2wasp43.kta',
'/Users/jingxuanyang/ktables/ch4wasp43.kta']
cia_file_path \
    ='/Users/jingxuanyang/Desktop/Workspace/nemesispy2022/nemesispy/data/cia/exocia_hitran12_200-3800K.tab'

### Reference Planet Input
M_plt = 3.8951064000000004e+27 # kg
R_plt = 74065.70 * 1e3 # m
gas_id = np.array([  1, 2,  5,  6, 40, 39])
iso_id = np.array([0, 0, 0, 0, 0, 0])
NLAYER = 20

################################################################################
################################################################################
# Read GCM data
f = open('process_vivien.txt')
vivien_gcm = f.read()
f.close()
vivien_gcm = vivien_gcm.split()
vivien_gcm = [float(i) for i in vivien_gcm]

### Parse GCM data
# iread = 0
# nlon = int(vivien_gcm[iread])
# iread += 1
# nlat = int(vivien_gcm[iread])
# iread += 1
# xlon = np.zeros(nlon) # regular lon lat grid
# xlat = np.zeros(nlat)
# for i in range(nlon):
#     xlon[i] = vivien_gcm[iread]
#     iread+=1
# for i in range(nlat):
#     xlat[i] = vivien_gcm[iread]
#     iread+=1
# npv = int(vivien_gcm[iread])
# iread += 1
# pv = np.zeros(npv)
# for i in range(npv):
#     pv[i] = vivien_gcm[iread]
#     iread+=1
# print('iread',iread)


iread = 152
nlon = 64
xlon = np.array([-177.19  , -171.56  , -165.94  , -160.31  , -154.69  , -149.06 ,
       -143.44  , -137.81  , -132.19  , -126.56  , -120.94  , -115.31  ,
       -109.69  , -104.06  ,  -98.438 ,  -92.812 ,  -87.188 ,  -81.562 ,
        -75.938 ,  -70.312 ,  -64.688 ,  -59.062 ,  -53.438 ,  -47.812 ,
        -42.188 ,  -36.562 ,  -30.938 ,  -25.312 ,  -19.688 ,  -14.062 ,
         -8.4375,   -2.8125,    2.8125,    8.4375,   14.062 ,   19.688 ,
         25.312 ,   30.938 ,   36.562 ,   42.188 ,   47.812 ,   53.438 ,
         59.062 ,   64.688 ,   70.312 ,   75.938 ,   81.562 ,   87.188 ,
         92.812 ,   98.438 ,  104.06  ,  109.69  ,  115.31  ,  120.94  ,
        126.56  ,  132.19  ,  137.81  ,  143.44  ,  149.06  ,  154.69  ,
        160.31  ,  165.94  ,  171.56  ,  177.19  ])
nlat = 32
xlat = np.array([-87.188 , -81.562 , -75.938 , -70.312 , -64.688 , -59.062 ,
       -53.438 , -47.812 , -42.188 , -36.562 , -30.938 , -25.312 ,
       -19.688 , -14.062 ,  -8.4375,  -2.8125,   2.8125,   8.4375,
        14.062 ,  19.688 ,  25.312 ,  30.938 ,  36.562 ,  42.188 ,
        47.812 ,  53.438 ,  59.062 ,  64.688 ,  70.312 ,  75.938 ,
        81.562 ,  87.188 ])
npv = 53
pv = np.array([1.7064e+02, 1.2054e+02, 8.5152e+01, 6.0152e+01, 4.2492e+01,
       3.0017e+01, 2.1204e+01, 1.4979e+01, 1.0581e+01, 7.4747e+00,
       5.2802e+00, 3.7300e+00, 2.6349e+00, 1.8613e+00, 1.3148e+00,
       9.2882e-01, 6.5613e-01, 4.6350e-01, 3.2742e-01, 2.3129e-01,
       1.6339e-01, 1.1542e-01, 8.1532e-02, 5.7595e-02, 4.0686e-02,
       2.8741e-02, 2.0303e-02, 1.4342e-02, 1.0131e-02, 7.1569e-03,
       5.0557e-03, 3.5714e-03, 2.5229e-03, 1.7822e-03, 1.2589e-03,
       8.8933e-04, 6.2823e-04, 4.4379e-04, 3.1350e-04, 2.2146e-04,
       1.5644e-04, 1.1051e-04, 7.8066e-05, 5.5146e-05, 3.8956e-05,
       2.7519e-05, 1.9440e-05, 1.3732e-05, 9.7006e-06, 6.8526e-06,
       4.8408e-06, 3.4196e-06, 2.4156e-06])*1e5

pvmap = np.zeros((nlon,nlat,npv))
for ilon in range(nlon):
    for ilat in range(nlat):
        pvmap[ilon,ilat,:] = pv

fake_hv =  np.linspace(0, 1404644.74126812, num=53)
fake_hvmap = np.zeros((nlon,nlat,npv))
for ilon in range(nlon):
    for ilat in range(nlat):
        fake_hvmap[ilon,ilat,:] = fake_hv

tmp = np.zeros((7,npv))
tmap = np.zeros((nlon,nlat,npv))
co2map = np.zeros((nlon,nlat,npv))
h2map = np.zeros((nlon,nlat,npv))
hemap = np.zeros((nlon,nlat,npv))
ch4map = np.zeros((nlon,nlat,npv))
comap = np.zeros((nlon,nlat,npv))
h2omap = np.zeros((nlon,nlat,npv))

for ilon in range(nlon):
    for ilat in range(nlat):
        for ipv in range(npv):
            tmap[ilon,ilat,ipv] = vivien_gcm[iread]
            h2omap[ilon,ilat,ipv] = vivien_gcm[iread+6]
            co2map[ilon,ilat,ipv] = vivien_gcm[iread+1]
            comap[ilon,ilat,ipv] = vivien_gcm[iread+5]
            ch4map[ilon,ilat,ipv] = vivien_gcm[iread+4]
            hemap[ilon,ilat,ipv] = vivien_gcm[iread+3]
            h2map[ilon,ilat,ipv] = vivien_gcm[iread+2]
            iread+=7

vmrmap = np.zeros((nlon,nlat,npv,6))
for ilon in range(nlon):
    for ilat in range(nlat):
        for ipv in range(npv):
            vmrmap[ilon,ilat,ipv,0] = h2omap[ilon,ilat,ipv]
            vmrmap[ilon,ilat,ipv,1] = co2map[ilon,ilat,ipv]
            vmrmap[ilon,ilat,ipv,2] = comap[ilon,ilat,ipv]
            vmrmap[ilon,ilat,ipv,3] = ch4map[ilon,ilat,ipv]
            vmrmap[ilon,ilat,ipv,4] = hemap[ilon,ilat,ipv]
            vmrmap[ilon,ilat,ipv,5] = h2map[ilon,ilat,ipv]

from nemesispy.radtran.hydrostatic import simple_hydro
hvmap  = np.zeros((nlon,nlat,npv))
for ilon in range(nlon):
    for ilat in range(nlat):
        hvmap[ilon,ilat,:] = simple_hydro(fake_hv[:],pvmap[ilon,ilat,:],tmap[ilon,ilat,:],
            vmrmap[ilon,ilat,:,:],R_plt,M_plt)
################################################################################
################################################################################

### Set up forward model
FM = ForwardModel()
FM.set_planet_model(M_plt=M_plt,R_plt=R_plt,gas_id_list=gas_id,iso_id_list=iso_id,
    NLAYER=NLAYER)
FM.set_opacity_data(kta_file_paths=lowres_files, cia_file_path=cia_file_path)

### Code to actually simulate a phase curve
wave_grid = np.array([1.1425, 1.1775, 1.2125, 1.2475, 1.2825, 1.3175, 1.3525, 1.3875,
       1.4225, 1.4575, 1.4925, 1.5275, 1.5625, 1.5975, 1.6325, 3.6   ,
       4.5   ])

nwave = len(wave_grid)

phase_grid = np.array([ 22.5,  45. ,  67.5,  90. , 112.5, 135. , 157.5, 180. , 202.5,
       225. , 247.5, 270. , 292.5, 315. , 337.5])

nphase = len(phase_grid)

phase_by_wave = np.zeros((nphase,nwave))
wave_by_phase = np.zeros((nwave,nphase))

wasp43_spec = np.array([3.341320e+25, 3.215455e+25, 3.101460e+25, 2.987110e+25,
       2.843440e+25, 2.738320e+25, 2.679875e+25, 2.598525e+25,
       2.505735e+25, 2.452230e+25, 2.391140e+25, 2.345905e+25,
       2.283720e+25, 2.203690e+25, 2.136015e+25, 1.234010e+24,
       4.422200e+23])

start = time.time()
for iphase, phase in enumerate(phase_grid):
    one_phase =  FM.calc_disc_spectrum(phase, nmu=5, global_H_model=hvmap, global_P_model=pvmap,
        global_T_model=tmap, global_VMR_model=vmrmap,
        global_model_longitudes=xlon,
        global_model_lattitudes=xlat,
        solspec=wasp43_spec)
    phase_by_wave[iphase,:] = one_phase
end = time.time()
runtime = end - start

for iwave in range(len(wave_grid)):
    for iphase in range(len(phase_grid)):
        wave_by_phase[iwave,iphase] = phase_by_wave[iphase,iwave]

# # plt.plot(wave_grid,one_phase)
# plt.show()
# print('disc averaged spec',one_phase)
# print('runtime',runtime)
"""
one_phase =  FM.calc_disc_spectrum(phase=247.5, nmu=5, global_H_model=hvmap, global_P_model=pvmap,
    global_T_model=tmap, global_VMR_model=vmrmap,
    global_model_longitudes=xlon,
    global_model_lattitudes=xlat,
    solspec=wasp43_spec)

zero_phase_gcm = np.array([1.25672e-04, 1.34992e-04, 1.95970e-04, 2.88426e-04, 3.37525e-04,
        2.81610e-04, 1.09969e-04, 7.62626e-05, 7.37120e-05, 8.76705e-05,
        1.14051e-04, 1.57063e-04, 2.17236e-04, 2.93693e-04, 3.60125e-04,
        1.91405e-03, 2.07116e-03])

plt.scatter(wave_grid,zero_phase_gcm,label='Fortran',color='k',marker='x',s=5)
plt.plot(wave_grid,zero_phase_gcm,color='k',)
plt.scatter(wave_grid,one_phase,label='Python',color='r',marker='o',s=5)
plt.plot(wave_grid,one_phase,color='r',)
plt.legend()
plt.grid()
plt.show()
"""
"""
Mod.M_plt == M_plt
Mod.R_plt == R_plt
Mod.T_star == T_star
Mod.semi_major_axis == semi_major_axis
Mod.NLAYER == NLAYER
Mod.is_planet_model_set == True

Mod.gas_id_list == gas_id_list
Mod.iso_id_list == iso_id_list
Mod.wave_grid == wave_grid
Mod.g_ord == g_ord
Mod.del_g == del_g
Mod.k_table_P_grid == k_table_P_grid
Mod.k_table_T_grid == k_table_T_grid
Mod.k_gas_w_g_p_t == k_gas_w_g_p_t
Mod.cia_nu_grid == cia_nu_grid
Mod.cia_T_grid == cia_T_grid
Mod.k_cia_pair_t_w == k_cia_pair_t_w
"""
