import matplotlib
#matplotlib.use('PDF')
import numpy as np
import matplotlib.pyplot as plt
import pymultinest
import pickle
import os
import sys
# sys.path.append('/Users/jingxuanyang/Desktop/Workspace/nemesispy2022/')
from corner import corner

from models import Model2
# from nemesispy.data.constants import R_SUN, R_JUP_E, AMU, AU, M_JUP, R_JUP, SIGMA_SB
# from nemesispy.radtran.utils import calc_mmw
# from nemesispy.radtran.trig import interpvivien_point

# Read GCM data
from process_gcm import (nlon,nlat,xlon,xlat,npv,pv,\
    tmap,h2omap,comap,co2map,ch4map,hemap,h2map,vmrmap,hvmap,\
    tmap_mod,h2omap_mod,comap_mod,co2map_mod,ch4map_mod,\
    hemap_mod,h2map_mod,vmrmap_mod,hvmap_mod,phase_grid,\
    kevin_phase_by_wave,kevin_wave_by_phase,\
    pat_phase_by_wave,pat_wave_by_phase,\
    vmrmap_mod_new,tmap_hot)

### Reference Planet Input: WASP 43b
T_star = 4520 # star temperature in K
R_star = 463892759.99999994 # m, 0.6668 * R_SUN
SMA = 2243970000.0 # m, 0.015*AU
M_plt = 3.8951064000000004e+27 # kg
R_plt = 74065.70 * 1e3 # m
gas_id = np.array([  1, 2,  5,  6, 40, 39])
iso_id = np.array([0, 0, 0, 0, 0, 0])
# T_irr = 2055

### Model parameters (focus to pressure range where Transmission WF peaks)
# Pressure range to be fitted (focus to pressure range where Transmission WF peaks)
NLAYER = 20
P_range = np.geomspace(20*1e5,1e-3*1e5,20)

# VMR map used by Pat is smoothed (HOW)
# Hence mmw is constant with longitude, lattitude and altitude
mmw = 3.92945509119087e-27 # kg

### GCM data
ilon=0
ilat=0
T_GCM = tmap_mod[ilon,ilat,:]
T_GCM_interped = np.interp(P_range,pv[::-1],T_GCM[::-1])

titles=np.array([r'$\log$ $\kappa$', r'$\log$ $\gamma_1$',
                r'$\log$ $\gamma_2$', r'$\alpha$', r'$\beta$'])
n_params = len(titles)

a = pymultinest.Analyzer(n_params=n_params,outputfiles_basename='chains/{}{}-'.format(ilon,ilat))
values = a.get_equal_weighted_posterior()
# values, [parameter values,..,ln pro], Nsamp x Npars
params = values[:, :n_params] # a collection of parameter values
lnprob = values[:, -1] # loglike corresponding to the parameter values
samples = params
Nsamp = values.shape[0]
Npars = n_params

### Plot 1
# plot corner plot
figure=corner(samples,
              quantiles=[0.16,0.5,0.84],
              labels=titles,
              show_titles='True',
              plot_contours='True',
              truths=None)
figure.savefig("TP_triangle.pdf")
plt.close()


def calc_T(cube):
    kappa = 10**cube[0]
    gamma1 = 10**cube[1]
    gamma2 = 10**cube[2]
    alpha = cube[3]
    beta = cube[4]
    T_int = 200

    Mod = Model2(T_star, R_star, M_plt, R_plt, SMA, P_range, mmw,
                    kappa = kappa,
                    gamma1 = gamma1,
                    gamma2 = gamma2,
                    alpha = alpha,
                    T_irr = beta,
                    T_int = T_int)
    T_model = Mod.temperature()
    # print(T_model)
    return T_model


NN = 1000 # has 1000 live points
draws = np.random.randint(len(samples), size=NN)
xrand = samples[draws, :]
Tarr1 = np.array([])


### Plot 2
# plot a random selection of TP profiles from remaining live points
for i in range(NN):
    T = calc_T(xrand[i,:])
    plt.plot(T,P_range/1e5,lw=1.0, alpha=0.3, color='#BF3EFF')
    Tarr1 = np.concatenate([Tarr1, T])

plt.errorbar(T_GCM,pv/1e5,xerr=10,fmt='o',lw=0.8, ms = 2,color='k',mfc='k',label='Data')
plt.errorbar(T_GCM_interped,P_range/1e5,xerr=10,fmt='o',lw=0.8, ms = 2,color='r',mfc='r',label='Smoothed')
plt.xlabel('Temperature [K]',size='x-large')
plt.ylabel('Pressure [bar]',size='x-large')
plt.minorticks_on()
plt.semilogy()
plt.gca().invert_yaxis()
plt.tick_params(length=10,width=1,labelsize='x-large',which='major')
plt.tight_layout()
plt.savefig('Spread_TP.pdf', format='pdf')
plt.show()
# plt.close()


### Plot 3
# Plot confidence interval
# Plot best fit TP profile with confidence interval
Tarr1=Tarr1.reshape(NN,P_range.shape[0])
Tmedian=np.zeros(P_range.shape[0])
Tlow_1sig=np.zeros(P_range.shape[0])
Thigh_1sig=np.zeros(P_range.shape[0])
Tlow_2sig=np.zeros(P_range.shape[0])
Thigh_2sig=np.zeros(P_range.shape[0])

for i in range(P_range.shape[0]):
    percentiles=np.percentile(Tarr1[:,i],[4.55, 15.9, 50, 84.1, 95.45])
    Tlow_2sig[i]=percentiles[0]
    Tlow_1sig[i]=percentiles[1]
    Tmedian[i]=percentiles[2]
    Thigh_1sig[i]=percentiles[3]
    Thigh_2sig[i]=percentiles[4]

#plt.fill_betweenx(P_range,Tlow_2sig,Thigh_2sig,facecolor='#BF3EFF',edgecolor='None',alpha=0.2)
plt.fill_betweenx(P_range/1e5,Tlow_1sig,Thigh_1sig,facecolor='#BF3EFF',edgecolor='None',alpha=0.5, label="TP1")
plt.plot(Tmedian, P_range/1e5,'#9E00FF',lw=1.5)
plt.tick_params(length=10,width=1,labelsize='x-large',which='major')
plt.xlabel('Temperature [K]',size='x-large')
plt.ylabel('Pressure [bar]',size='x-large')
# plt.xlim(0,3000)
plt.errorbar(T_GCM,pv/1e5,xerr=10,fmt='o',lw=0.8, ms = 2,color='k',mfc='k',label='Data')
plt.errorbar(T_GCM_interped,P_range/1e5,xerr=10,fmt='o',lw=0.8, ms = 2,color='r',mfc='r',label='Smoothed')
plt.legend()
plt.semilogy()
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig('Tmedian.pdf',format='pdf')
plt.show()
# plt.close()
