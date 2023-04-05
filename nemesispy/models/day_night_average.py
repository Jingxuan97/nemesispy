import numpy as np
import matplotlib.pyplot as plt
from nemesispy.models.TP_profiles import TP_Guillot
from nemesispy.models.master import plot_TP_equator
from nemesispy.common.constants import G

def grid(P_max=20*1e5,P_min=1e-3*1e5,nP=20,nlon=359,nlat=89):
    lon_grid = np.linspace(-179,179,nlon)
    lat_grid = np.linspace(0,89,nlat)
    p_grid = np.geomspace(P_max,P_min,nP)
    return lon_grid,lat_grid,p_grid



### Reference Planet Input: WASP 43b
T_star = 4520 # star temperature in K
R_star = 463892759.99999994 # m, 0.6668 * R_SUN
SMA = 2243970000.0 # m, 0.015*AU
M_plt = 3.8951064000000004e+27 # kg
R_plt = 74065.70 * 1e3 # m
T_irr = T_star * (R_star/SMA)**0.5
T_eq = T_irr/2**0.5
g = G*M_plt/R_plt**2

P_range = np.geomspace(20*1e5,1e-3*1e5,20)
log_kappa_day = -0.237221506017800676E+01
log_gamma_day = -0.290589914308588992E+00
log_f_day = -0.372957867443207802E+00
T_int_day = 0.375720127259632818E+03

log_kappa_night = -0.231753423743880260E+01
log_gamma_night = -0.137276521979532151E+01
log_f_night = -0.141794421727358344E+01
T_int_night = 0.498458581709161649E+03

T_day = TP_Guillot(P_range,g,T_eq,10**log_kappa_day,10**log_gamma_day,
    10**log_f_day,T_int_day) * 1.05

T_night = TP_Guillot(P_range,g,T_eq,10**log_kappa_night,10**log_gamma_night,
    10**log_f_night,T_int_night)

nP=20
nlon=359
nlat=89
lon_grid,lat_grid,p_grid = grid(nP=nP,nlon=nlon,nlat=nlat)
TP_grid = np.zeros((nlon,nlat,nP))


def cos_smooth(longitude, power=0.25):
    dtr = np.pi/180
    longitude = longitude
    if longitude >= 90 or longitude <=-90:
        output = 0
    else:
        output = np.cos(dtr*longitude) ** power
    return output


# for ilon,lon in enumerate(lon_grid):
#     for ilat,lat in enumerate(lat_grid):
#         TP_grid[ilon,ilat,:] = T_day * cos_smooth(lon,29) \
#             + T_night * (cos_smooth(abs(lon)-180,0)) \
#             + (T_day+T_night)/2 * (1-(cos_smooth(abs(lon)-180,0))- cos_smooth(lon,29))


def T_smooth_cos(T_hot, T_cold, offset, scale, p_grid, lon_grid, lat_grid,
        power=0.25):
    nlon = len(lon_grid)
    nlat = len(lat_grid)
    nP = len(p_grid)
    dtr = np.pi/180
    TP_grid = np.zeros((nlon,nlat,nP))
    for ilon,lon in enumerate(lon_grid):
        for ilat,lat in enumerate(lat_grid):
            a = cos_smooth( (lon-offset)/scale )
            if offset == 0:
                b = cos_smooth( ((abs(lon)-180) ) * scale )
            elif offset > 0:
                if lon <= 0:
                    b = cos_smooth( abs(lon-(-180+offset)) * scale )
                else:
                    b = cos_smooth( abs(lon-(180+offset)) * scale )
            else: # offset < 0
                if lon <= 0:
                    b = cos_smooth( abs(lon -(-180+offset)) * scale )
                else:
                    b = cos_smooth( abs(lon-(180+offset)) * scale )
            c = 1 - a - b

            TP_grid[ilon,ilat,:] = T_hot * a + T_cold * b\
                + (T_night+T_day)/2*c
    return TP_grid

# for i in (np.linspace(-90,90)):
#     TP_grid = T_smooth_cos(T_day,T_night,i,1,p_grid,lon_grid,lat_grid)
#     plot_TP_equator(TP_grid,lon_grid,P_range,lon_grid,lat_grid,P_range,
#         figname='plots/smooth_map{}.png'.format(i))


TP_grid = T_smooth_cos(T_day,T_night,29,0.8,p_grid,lon_grid,lat_grid)
plot_TP_equator(TP_grid,lon_grid,P_range,lon_grid,lat_grid,P_range,
    figname='smooth_map.png')

from nemesispy.data.helper import lowres_file_paths, cia_file_path
from nemesispy.radtran.forward_model import ForwardModel

from nemesispy.common.constants import G
from nemesispy.data.gcm.process_gcm import (nlon,nlat,xlon,xlat,npv,pv,\
    tmap,h2omap,comap,co2map,ch4map,hemap,h2map,vmrmap,\
    tmap_mod,h2omap_mod,comap_mod,co2map_mod,ch4map_mod,\
    hemap_mod,h2map_mod,vmrmap_mod,phase_grid,\
    kevin_phase_by_wave,kevin_wave_by_phase,\
    pat_phase_by_wave,pat_wave_by_phase,\
    vmrmap_mod_new,tmap_hot)
from nemesispy.models.gas_profiles import gen_vmrmap1

wasp43_spec = np.array([3.341320e+25, 3.215455e+25, 3.101460e+25, 2.987110e+25,
       2.843440e+25, 2.738320e+25, 2.679875e+25, 2.598525e+25,
       2.505735e+25, 2.452230e+25, 2.391140e+25, 2.345905e+25,
       2.283720e+25, 2.203690e+25, 2.136015e+25, 1.234010e+24,
       4.422200e+23])
wave_grid = np.array([1.1425, 1.1775, 1.2125, 1.2475, 1.2825, 1.3175, 1.3525, 1.3875,
       1.4225, 1.4575, 1.4925, 1.5275, 1.5625, 1.5975, 1.6325, 3.6   ,
       4.5   ])
phase_grid = np.array([ 22.5,  45. ,  67.5,  90. , 112.5, 135. , 157.5, 180. , 202.5,
       225. , 247.5, 270. , 292.5, 315. , 337.5])


nwave = len(wave_grid)
nphase = len(phase_grid)
gas_id = np.array([  1, 2,  5,  6, 40, 39])
iso_id = np.array([0, 0, 0, 0, 0, 0])
NLAYER = 20
P_range = np.geomspace(20*1e5,1e-3*1e5,NLAYER)


# ### Set up forward model
# FM = ForwardModel()
# FM.set_planet_model(M_plt=M_plt,R_plt=R_plt,gas_id_list=gas_id,iso_id_list=iso_id,
#     NLAYER=NLAYER)
# FM.set_opacity_data(kta_file_paths=lowres_file_paths, cia_file_path=cia_file_path)

# retrieved_TP_phase_by_wave_20par = np.zeros((nphase,nwave))
# retrieved_TP_wave_by_phase_20par = np.zeros((nwave,nphase))
# print('P_range')
# # 20 par fit spec

# # generate uniform abundance map
# vmr_grid = gen_vmrmap1(
#     -0.326856261789093328E+01,
#     -0.567806248364052024E+01,
#     -0.507547232597232689E+01,
#     -0.699057998842726924E+01,
#     nlon=len(lon_grid), nlat=len(lat_grid),
#     npress=len(P_range))

# for iphase, phase in enumerate(phase_grid):
#     one_phase =  FM.calc_disc_spectrum(phase=phase, nmu=5, P_model=P_range,
#         global_model_P_grid=P_range, global_T_model=TP_grid,
#         global_VMR_model=vmr_grid,
#         mod_lon=lon_grid,
#         mod_lat=lat_grid,
#         solspec=wasp43_spec)
#     retrieved_TP_phase_by_wave_20par[iphase,:] = one_phase
# for iwave in range(len(wave_grid)):
#     for iphase in range(len(phase_grid)):
#         retrieved_TP_wave_by_phase_20par[iwave,iphase] \
#             = retrieved_TP_phase_by_wave_20par[iphase,iwave]

# # Plot phase curve at each wavelength
# fig, axs = plt.subplots(nrows=9,ncols=2,sharex=True,sharey=False,
#                         figsize=[8.25,11.75],dpi=600)

# xticks = np.array(
#     [0, 90, 180, 270, 360]
#     )
# # fig.supxlabel('phase')
# # fig.supylabel(r'Wavelength [$\mu$m]')

# ix = 0
# iy = 0
# for iwave,wave in enumerate(wave_grid[::-1]):

#     # axs[ix,iy].plot(phase_grid, retrieved_TP_wave_by_phase_Guillot[16-iwave,:]*1e3,
#     #     marker='s',ms=0.1,mfc='b',color='b',linewidth=0.5,linestyle='-.',
#     #     label='2-stream')

#     axs[ix,iy].plot(phase_grid, retrieved_TP_wave_by_phase_20par[16-iwave,:]*1e3,
#         marker='s',ms=0.1,mfc='r',color='r',linewidth=0.5,linestyle='-.',
#         label='Retrieval')

#     axs[ix,iy].errorbar(phase_grid, pat_wave_by_phase[16-iwave,:]*1e3,
#         yerr = kevin_wave_by_phase[16-iwave,:,1]/2*1e3,
#         marker='s',ms=0.1,mfc='k',color='k',
#         linewidth=0.5,label='shitfted GCM')

#     axs[ix,iy].set_yticklabels([])
#     # wave = np.around(wave,decimals=2)
#     # axs[ix,iy].set_ylabel(wave,rotation=0,fontsize=8)
#     handles, labels = axs[ix,iy].get_legend_handles_labels()

#     wave = np.around(wave,decimals=2)
#     axs[ix,iy].set_ylabel('{} $\mu$m '.format(wave),rotation=90,fontsize=16)

#     axs[ix,iy].grid()
#     ix += 1
#     if ix == 9:
#         ix = 0
#         iy += 1

# axs[8,1].set_visible(False)
# axs[8,0].set_xticks(xticks)
# axs[8,0].tick_params(labelsize=16)

# fig.legend(handles, labels, ncol=1, loc='lower right',
#     fontsize=16)
# fig.tight_layout()

# plt.savefig('compare_phase_curves.png',dpi=400)
# plt.close()