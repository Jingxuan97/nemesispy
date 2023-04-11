from nemesispy.data.gcm.process_gcm import (nlon,nlat,xlon,xlat,npv,pv,\
    tmap,h2omap,comap,co2map,ch4map,hemap,h2map,vmrmap,\
    tmap_mod,h2omap_mod,comap_mod,co2map_mod,ch4map_mod,\
    hemap_mod,h2map_mod,vmrmap_mod,phase_grid,\
    kevin_phase_by_wave,kevin_wave_by_phase,\
    pat_phase_by_wave,pat_wave_by_phase,\
    vmrmap_mod_new,tmap_hot)
import numpy as np
import matplotlib.pyplot as plt
from planet_input import planet
from nemesispy.radtran.forward_model import ForwardModel
from nemesispy.common.interpolate_gcm import interp_gcm_X
from nemesispy.retrieval.plot_slice import plot_TP_equator,\
    plot_TP_equator_weighted, plot_TP_equator_weighted_diff
from nemesispy.models.gas_profiles import gen_vmrmap1

nmu = 5

def T_weighted_average(tmap, output_lon_grid, output_lat_grid, output_p_grid,
    tmap_lon_grid, tmap_lat_grid, tmap_p_grid):

    # Npress = len(pressure_grid)
    # Nlon = len(longitude_grid)
    # Nlat = len(latitude_grid)
    dtr = np.pi/180
    qudrature = np.array(
        [ 2.5,  7.5, 12.5, 17.5, 22.5, 27.5, 32.5, 37.5, 42.5, 47.5, 52.5,
        57.5, 62.5, 67.5, 72.5, 77.5, 82.5, 87.5, ])

    weight = np.cos(qudrature*dtr) * 5 * dtr


    qudrature = np.linspace(0,50,60)
    weight = np.cos(qudrature*dtr) * (qudrature[1]-qudrature[0]) * dtr

    sum_weight = np.sum(weight)
    Tout = np.zeros((len(output_lon_grid),
        len(output_lat_grid),
        len(output_p_grid)))

    TPs = np.zeros((len(output_p_grid), len(output_lon_grid)))
    print(TPs.shape)
    for ilon, lon in enumerate(output_lon_grid):
        for ilat, lat in enumerate(qudrature):
            iT = interp_gcm_X(lon,lat,output_p_grid,
                gcm_p=tmap_p_grid,gcm_lon=tmap_lon_grid,gcm_lat=tmap_lat_grid,
                X=tmap, substellar_point_longitude_shift=0)
            TPs[:,ilon] += iT * weight[ilat]

    TPs = TPs/sum_weight
    for ilon, lon in enumerate(output_lon_grid):
        for ilat, lat in enumerate(output_lat_grid):
            Tout[ilon,ilat,:] = TPs[:,ilon]

    return Tout

tmap_weighted = T_weighted_average(tmap, xlon, xlat, planet['P_range'],
    xlon,xlat,pv)

# generate uniform abundance map
vmr_grid = gen_vmrmap1(
    np.log10(h2omap_mod[0,0,0]),
    np.log10(co2map_mod[0,0,0]),
    np.log10(comap_mod[0,0,0]),
    np.log10(ch4map_mod[0,0,0]),
    nlon=len(xlon), nlat=len(xlat),
    npress=len(planet['P_range']))

### Set up forward model
FM = ForwardModel()
FM.read_input_dict(planet)

TP_phase_by_wave = np.zeros((planet['nphase'],planet['nwave']))
TP_wave_by_phase = np.zeros((planet['nwave'],planet['nphase']))

# 20 par fit spec
for iphase, phase in enumerate(planet['phase_grid']):
    print('phase',phase)
    print('nmu',nmu)
    print("planet['P_range']",planet['P_range'])
    one_phase =  FM.calc_disc_spectrum(phase=phase, nmu=nmu, P_model=planet['P_range'],
        global_model_P_grid=planet['P_range'], global_T_model=tmap_weighted,
        global_VMR_model=vmr_grid,
        mod_lon=xlon,
        mod_lat=xlat,
        solspec=planet['stellar_spec'])
    TP_phase_by_wave[iphase,:] = one_phase
for iwave in range(len(planet['wave_grid'])):
    for iphase in range(len(phase_grid)):
        TP_wave_by_phase[iwave,iphase] \
            = TP_phase_by_wave[iphase,iwave]


# Plot phase curve at each wavelength
fig, axs = plt.subplots(nrows=9,ncols=2,sharex=True,sharey=False,
                        figsize=[8.25,11.75],dpi=600)

xticks = np.array(
    [0, 90, 180, 270, 360]
    )
# fig.supxlabel('phase')
# fig.supylabel(r'Wavelength [$\mu$m]')

ix = 0
iy = 0
for iwave,wave in enumerate(planet['wave_grid'][::-1]):

    # axs[ix,iy].plot(phase_grid, retrieved_TP_wave_by_phase_Guillot[16-iwave,:]*1e3,
    #     marker='s',ms=0.1,mfc='b',color='b',linewidth=0.5,linestyle='-.',
    #     label='2-stream')

    axs[ix,iy].plot(phase_grid, TP_wave_by_phase[16-iwave,:]*1e3,
        marker='s',ms=0.1,mfc='r',color='r',linewidth=0.5,linestyle='-.',
        label='Weighted')

    axs[ix,iy].errorbar(phase_grid, pat_wave_by_phase[16-iwave,:]*1e3,
        yerr = kevin_wave_by_phase[16-iwave,:,1]/2*1e3,
        marker='s',ms=0.1,mfc='k',color='k',
        linewidth=0.5,label='GCM')

    axs[ix,iy].set_yticklabels([])
    # wave = np.around(wave,decimals=2)
    # axs[ix,iy].set_ylabel(wave,rotation=0,fontsize=8)
    handles, labels = axs[ix,iy].get_legend_handles_labels()


    wave = np.around(wave,decimals=2)
    axs[ix,iy].set_ylabel('{} $\mu$m '.format(wave),rotation=90,fontsize=16)

    ix += 1
    if ix == 9:
        ix = 0
        iy += 1

axs[8,1].set_visible(False)
axs[8,0].set_xticks(xticks)
axs[8,0].tick_params(labelsize=16)
fig.legend(handles, labels, ncol=1, loc='lower right',
    fontsize=16)
fig.tight_layout()

plt.savefig('compare_phase_curves_range_50.pdf',dpi=400)
plt.close()