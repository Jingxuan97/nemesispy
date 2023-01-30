# -*- coding: utf-8 -*-
"""Plot 2 stream fit parameters"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# Read GCM data
from nemesispy.data.gcm.wasp43b_vivien.process_wasp43b_gcm_vivien import (
    nlon,nlat,xlon,xlat,npv,pv,\
    tmap,h2omap,comap,co2map,ch4map,hemap,h2map,vmrmap,\
    tmap_mod,h2omap_mod,comap_mod,co2map_mod,ch4map_mod,\
    hemap_mod,h2map_mod,vmrmap_mod,phase_grid,\
    kevin_phase_by_wave,kevin_wave_by_phase,\
    pat_phase_by_wave,pat_wave_by_phase)
from nemesispy.common.function_fit import constant, lorentz, \
    lorentz_plus_C, normal, normal_plus_C, skew_normal, voigt, voigt_plus_C, \
    skew_voigt, fourier1, fourier2, fourier3, fourier4


## Load the best fitting 1D model parameters
# f
param_name = [r'log $\gamma$',r'log $\kappa$',r'log $f$',r'T$_{int}$']
iparam = 2
nparam = len(param_name)
params = np.loadtxt('best_fit_params_2_stream_Guillot.txt',
    unpack=True,delimiter=',')
params = params.T

## Assign the parameters to their (longitude,latitude) grid positions
best_params = np.zeros((nlon,nlat,nparam))
for ilon in range(nlon):
    for ilat in range(nlat):
        best_params[ilon,ilat,:] = params[ilon*nlat+ilat,:]

## Set up figure
fig, axs = plt.subplots(nrows=8,ncols=2,sharex=True,sharey=True,
                        figsize=[8.25,11.75],dpi=600)
plt.xlim(-180,180)
# plt.ylim(0,0.42)
fig.supxlabel('Longitude')
fig.supylabel('{} parameter value'.format(param_name[iparam]))

ix = 0
iy = 0
for ilat in range(int(nlat/2)):
    axs[ix,iy].text(-10,-2,'lat = {:.1f}'.format(xlat[-ilat-1]),
    fontsize='x-small')

    # best fit parameter data
    xdata = xlon
    ydata_south = np.log10(best_params[:,ilat,iparam])
    ydata_north = np.log10(best_params[:,-ilat-1,iparam])
    xdata_total = np.concatenate([xlon,xlon])
    ydata_total = np.concatenate([ydata_south,ydata_north])

    # fit functions to data
    # fit constant
    popt_constant, pcov_constant \
        = curve_fit(constant, xdata_total, ydata_total)
    yconstant = constant(xdata,*popt_constant)
    err_constant = np.average( ((yconstant-ydata_north)**2\
         + (yconstant-ydata_south)**2) / np.std(ydata_total)**2)

    # fit lorentz_plus_C
    popt_lorentz_plus_C, pcov_lorentz_plus_C \
        = curve_fit(lorentz_plus_C, xdata_total, ydata_total,
            maxfev=500000)
    ylorentz_plus_C = lorentz_plus_C(xdata,*popt_lorentz_plus_C)
    err_lorentz_plus_C = np.average( ((ylorentz_plus_C-ydata_north)**2\
         + (ylorentz_plus_C-ydata_south)**2) / np.std(ydata_total)**2)

    # fit normal_plus_C
    popt_normal_plus_C, pcov_normal_plus_C \
        = curve_fit(normal_plus_C, xdata_total, ydata_total,
            maxfev=500000)
    ynormal_plus_C = normal_plus_C(xdata,*popt_normal_plus_C)
    err_normal_plus_C = np.average(((ynormal_plus_C-ydata_north)**2\
         + (ynormal_plus_C-ydata_south)**2) / np.std(ydata_total)**2)

    # fit fourier1
    popt_fourier1, pcov_fourier1 \
        = curve_fit(fourier1, xdata_total, ydata_total)
    yfourier1 = fourier1(xdata,*popt_fourier1)
    err_fourier1 = np.average(((yfourier1-ydata_north)**2\
         + (yfourier1-ydata_south)**2) / np.std(ydata_total)**2)

    # fit fourier2
    popt_fourier2, pcov_fourier2 \
        = curve_fit(fourier2, xdata_total, ydata_total)
    yfourier2 = fourier2(xdata,*popt_fourier2)
    err_fourier2 = np.average(((yfourier2-ydata_north)**2\
         + (yfourier2-ydata_south)**2) / np.std(ydata_total)**2)
    # print(popt_fourier2)

    # fit fourier3
    popt_fourier3, pcov_fourier3 \
        = curve_fit(fourier3, xdata_total, ydata_total)
    yfourier3 = fourier3(xdata,*popt_fourier3)
    err_fourier3 = np.average(((yfourier3-ydata_north)**2\
         + (yfourier3-ydata_south)**2) / np.std(ydata_total)**2)

    # plot data
    l1, = axs[ix,iy].plot(xlon,ydata_north,
        marker='x', ms=3, lw=0.0, markeredgewidth=0.4, color='r')
    l2, = axs[ix,iy].plot(xlon,ydata_south,
        marker='x', ms=3, lw=0.0, markeredgewidth=0.4, color='b')
    if ix == 0 and iy == 0:
        leg1 = axs[ix,iy].legend([l1,l2],['north','south'],
            loc='upper right',fontsize='xx-small')
        axs[ix,iy].add_artist(leg1)

    # plot fits
    l3, = axs[ix,iy].plot(
        xlon,yconstant,color='orange',lw=0.5, linestyle="-")

    l4, = axs[ix,iy].plot(
        xlon,yfourier1, color='teal', lw=0.8, linestyle='--')

    l5, = axs[ix,iy].plot(
        xlon,ynormal_plus_C,color='k', lw=1, linestyle=":")

    l6, = axs[ix,iy].plot(
        xlon,yfourier2,color='olive',lw=0.8, linestyle='--')

    l7, = axs[ix,iy].plot(
        xlon,ylorentz_plus_C,color='g', lw=1, linestyle=':')

    l8, = axs[ix,iy].plot(
        xlon,yfourier3,color='purple',lw=0.8, linestyle='--')

    leg2 = axs[ix,iy].legend(
        [
            l3,
            l4,
            l5,
            l6,
            l7,
            l8,
        ],
        [
            'constant: {:.2f}'.format(err_constant),
            'fourier1: {:.2f}'.format(err_fourier1),
            'normal: {:.2f}'.format(err_normal_plus_C),
            'fourier2: {:.2f}'.format(err_fourier2),
            'lorentz: {:.2f}'.format(err_lorentz_plus_C),
            'fourier3: {:.2f}'.format(err_fourier3),
        ],
        ncol=3, loc='lower left', fontsize='xx-small')
    axs[ix,iy].add_artist(leg2)

    ix += 1
    if ix == 8:
        ix = 0
        iy += 1

fig.tight_layout()
plt.savefig('figures/function_fit_log_f.pdf',dpi=800)