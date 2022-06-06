import matplotlib
#matplotlib.use('PDF')
import numpy as np
import matplotlib.pyplot as plt
import pymultinest
import pickle
import os
import sys
sys.path.append('/Users/jingxuanyang/Desktop/Workspace/nemesispy2022/')
from corner import corner
from nemesispy.data.constants import R_SUN, R_JUP_E, AMU, AU, M_JUP, R_JUP, SIGMA_SB
from nemesispy.radtran.models import Model2
from nemesispy.radtran.utils import calc_mmw
from nemesispy.radtran.trig import interpvivien_point

# Read GCM data
from nemesispy.data.gcm.process_gcm import (nlon,nlat,xlon,xlat,npv,pv,\
    tmap,h2omap,comap,co2map,ch4map,hemap,h2map,vmrmap,hvmap,\
    tmap_mod,h2omap_mod,comap_mod,co2map_mod,ch4map_mod,\
    hemap_mod,h2map_mod,vmrmap_mod,hvmap_mod,phase_grid,\
    kevin_phase_by_wave,kevin_wave_by_phase,\
    pat_phase_by_wave,pat_wave_by_phase,\
    vmrmap_mod_new,tmap_hot)

### Reference Planet Input: WASP 43b
T_star = 4520 # star temperature in K
R_star = 0.6668 * R_SUN
SMA = 0.015*AU
M_plt = 3.8951064000000004e+27 # kg
R_plt = 74065.70 * 1e3 # m
gas_id = np.array([  1, 2,  5,  6, 40, 39]) # H2O,CO,CO2,CH4,He,H2
iso_id = np.array([0, 0, 0, 0, 0, 0])

### Model parameters (focus to pressure range where Transmission WF peaks)
P_range = np.array(
        [2.00000000e+06, 1.18757213e+06, 7.05163778e+05, 4.18716424e+05,
       2.48627977e+05, 1.47631828e+05, 8.76617219e+04, 5.20523088e+04,
       3.09079355e+04, 1.83527014e+04, 1.08975783e+04, 6.47083012e+03,
       3.84228874e+03, 2.28149751e+03, 1.35472142e+03, 8.04414701e+02,
       4.77650239e+02, 2.83622055e+02, 1.68410824e+02, 1.00000000e+02])

### GCM data
longitude = 180
lattitude = 0
T_GCM, VMR_GCM = interpvivien_point( xlon=longitude,xlat=lattitude,xp=P_range,
                    vp=pv, vt=tmap, vvmr=vmrmap, mod_lon=xlon, mod_lat=xlat)
mmw = calc_mmw(gas_id,VMR_GCM[0,:])

titles=np.array([r'$\log$ $\kappa$', r'$\log$ $\gamma_1$',
                r'$\log$ $\gamma_2$', r'$\alpha$', r'$\beta$'])
n_params = len(titles)

a = pymultinest.Analyzer(n_params=n_params)
values = a.get_equal_weighted_posterior()
# values, [parameter values,..,ln pro], Nsamp x Npars
params = values[:, :n_params] # a collection of parameter values
lnprob = values[:, -1] # loglike corresponding to the parameter values
samples = params
Nsamp = values.shape[0]
Npars = n_params

"""
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
"""

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


P_interp = np.geomspace(2e6,100,20)
T_interp, VMR_interp = interpvivien_point( xlon=longitude,xlat=lattitude,xp=P_interp,
                    vp=pv, vt=tmap, vvmr=vmrmap, mod_lon=xlon, mod_lat=xlat)


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

plt.errorbar(T_GCM,P_range/1e5,xerr=10,fmt='o',lw=0.8, ms = 2,color='k',mfc='k',label='Data')
plt.errorbar(T_interp,P_interp/1e5,xerr=10,fmt='o',lw=0.8, ms = 2,color='r',mfc='r',label='Smoothed')
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
plt.errorbar(T_GCM,P_range/1e5,xerr=10,fmt='o',lw=0.8, ms = 2,color='k',mfc='k',label='Data')
plt.errorbar(T_interp,P_interp/1e5,xerr=10,fmt='o',lw=0.8, ms = 2,color='r',mfc='r',label='Smoothed')
plt.legend()
plt.semilogy()
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig('Tmedian.pdf',format='pdf')
plt.show()
# plt.close()
