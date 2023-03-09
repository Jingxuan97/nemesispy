#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
An interactive plot routine for exploring the effects of chemical abundances,
thermal structure and zenith angle on the emission spectrum of a plane
parallel atmopshere.
"""
print('Loading libraries...')
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

from nemesispy.radtran.forward_model import ForwardModel
from nemesispy.common.constants import G
from nemesispy.models.TP_profiles import TP_Guillot
from nemesispy.data.helper import lowres_file_paths, cia_file_path

from nemesispy.data.gcm.process_gcm import kevin_phase_by_wave

### Wavelength grid
phase_grid = np.array(
    [ 22.5,  45. ,  67.5,  90. , 112.5, 135. , 157.5, 180. , 202.5,
    225. , 247.5, 270. , 292.5, 315. , 337.5])
wave_grid = np.array(
    [1.1425, 1.1775, 1.2125, 1.2475, 1.2825, 1.3175, 1.3525, 1.3875,
    1.4225, 1.4575, 1.4925, 1.5275, 1.5625, 1.5975, 1.6325, 3.6,
    4.5]
    )
wasp43_spec = np.array([3.341320e+25, 3.215455e+25, 3.101460e+25, 2.987110e+25,
       2.843440e+25, 2.738320e+25, 2.679875e+25, 2.598525e+25,
       2.505735e+25, 2.452230e+25, 2.391140e+25, 2.345905e+25,
       2.283720e+25, 2.203690e+25, 2.136015e+25, 1.234010e+24,
       4.422200e+23])


### VMR and pressure models
def gen_vmr(NLAYER,h2o,co2,co,ch4):
    h2 = (1-h2o-co2-co-ch4)*0.85
    he = (1-h2o-co2-co-ch4)*0.15
    VMR_model = np.zeros((NLAYER,6))
    VMR_model[:,0] = h2o
    VMR_model[:,1] = co2
    VMR_model[:,2] = co
    VMR_model[:,3] = ch4
    VMR_model[:,4] = he
    VMR_model[:,5] = h2
    return VMR_model

def gen_pressure(low,high,NLAYER):
    P_model = np.geomspace(low,high,NLAYER)
    return P_model

### Reference Planet Input: WASP 43b
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

### Set up an initial atmospheric model
# Pressure range
NLAYER = 20
P_deep_init = 20e5
P_top_init = 100
P_model = gen_pressure(P_deep_init,P_top_init,NLAYER)
# Gas volume mixing ratios
H2O_init = -8
CO2_init = -8
CO_init = -8
CH4_init = -8
VMR_model_init = gen_vmr(NLAYER,10**H2O_init,10**CO2_init,10**CO_init,
    10**CH4_init)
# Temperature-pressure profile
k_IR0 = 10**-3
gamma0 = 10**-2
f0 = 0.1
T_int0 = 100
T_model = TP_Guillot(P_model,g,T_eq,k_IR0,gamma0,f0,T_int0)

print("Compiling forward model...")
### Set up forward model
FM = ForwardModel()
FM.set_planet_model(M_plt=M_plt,R_plt=R_plt,gas_id_list=gas_id,iso_id_list=iso_id,
    NLAYER=NLAYER)
FM.set_opacity_data(kta_file_paths=lowres_file_paths, cia_file_path=cia_file_path)
# spec = FM.calc_point_spectrum_hydro(
#     P_model,T_model,VMR_model_init,path_angle=0,solspec=[])
spec = FM.calc_disc_spectrum_uniform(
    3,P_model,T_model,VMR_model_init,
    solspec=wasp43_spec)
print("Set up interactive plot")
### Set up figure and axes
fig, ax = plt.subplots(1,2)
plt.subplots_adjust(bottom=0.5)
ax[0].set_title('spectra')
ax[0].set_ylabel('flux ratio')
ax[0].set_xlabel('wavelength (micron)')
ax[1].set_title('TP profile')
ax[1].set_ylabel('pressure (Pa)')
ax[1].set_xlabel('temperature (T)')
line_spec, = ax[0].plot(wave_grid,spec,marker='x',markersize=3,
        label='model')
line_data, = ax[0].plot(wave_grid,kevin_phase_by_wave[8,:,0],color='k',
        marker='s',ms=0.1,mfc='k',linewidth=0.5,
        label='data')
line_tp, = ax[1].plot(T_model,P_model)
ax[0].set_ylim(0,0.005)
ax[1].set_ylim(1e-5*1e5,200*1e5)
ax[1].set_xlim(300,5000)
ax[1].invert_yaxis()
ax[1].set_yscale('log')
ax[0].legend()
plt.subplots_adjust(wspace=0.3)

### Slider Axes
axcolor = 'lightgoldenrodyellow'

axkappa = plt.axes([0.6, 0.35, 0.30, 0.03], facecolor=axcolor)
axgamma = plt.axes([0.6, 0.30, 0.30, 0.03], facecolor=axcolor)
axf = plt.axes([0.6, 0.25, 0.30, 0.03], facecolor=axcolor)
axT_int = plt.axes([0.6, 0.20, 0.30, 0.03], facecolor=axcolor)
axN_phase = plt.axes([0.6, 0.15, 0.30, 0.03], facecolor=axcolor)

axP_top = plt.axes([0.10, 0.35, 0.30, 0.03], facecolor=axcolor)
axP_deep = plt.axes([0.10, 0.30, 0.30, 0.03], facecolor=axcolor)
axNlayer = plt.axes([0.10, 0.25, 0.30, 0.03], facecolor=axcolor)
axH2O = plt.axes([0.10, 0.20, 0.30, 0.03], facecolor=axcolor)
axCO2 = plt.axes([0.10, 0.15, 0.30, 0.03], facecolor=axcolor)
axCO = plt.axes([0.10, 0.10, 0.30, 0.03], facecolor=axcolor)
axCH4 = plt.axes([0.10, 0.05, 0.30, 0.03], facecolor=axcolor)

### Slider values
skappa = Slider(axkappa, 'kappa', -4, 2, valinit=-3 , valstep=0.1)
sgamma = Slider(axgamma, 'gamma', -4, 1, valinit=-2, valstep=0.1)
sf = Slider(axf, 'f', 0.0, 2.0, valinit=f0, valstep=0.01)
sT_int = Slider(axT_int, 'T_int', 0., 1000.0, valinit=T_int0, valstep=1)

sP_top = Slider(axP_top, 'P_top', -5 , -2, valinit=np.log10(P_top_init*1e-5),valstep=0.1)
sP_deep = Slider(axP_deep, 'P_deep', -1 , 2, valinit=np.log10(P_deep_init*1e-5),valstep=0.1)
sNlayer = Slider(axNlayer, 'Nlayer', 5, 200, valinit=NLAYER,valstep=1)
sH2O = Slider(axH2O, 'H2O', -10, -1, valinit=H2O_init,valstep=0.1)
sCO2 = Slider(axCO2, 'CO2', -10, -1, valinit=CO2_init,valstep=0.1)
sCO = Slider(axCO, 'CO', -10, -1, valinit=CO_init,valstep=0.1)
sCH4 = Slider(axCH4, 'CH4', -10, -1, valinit=CH4_init,valstep=0.1)
sN_phase = Slider(axN_phase, 'N_phase', 0, 14, valinit=8,valstep=1)

def update(val):

    kappa = 10**skappa.val
    gamma = 10**sgamma.val
    f = sf.val
    T_int = sT_int.val
    Nphase = int(sN_phase.val)

    P_top = 10**sP_top.val*1e5
    P_deep = 10**sP_deep.val*1e5
    NLAYER = int(sNlayer.val)
    H2O = 10**sH2O.val
    CO2 = 10**sCO2.val
    CO = 10**sCO.val
    CH4 = 10**sCH4.val

    P_model = gen_pressure(P_deep,P_top,NLAYER)
    # print('P_model',P_model)
    VMR_model = gen_vmr(NLAYER,H2O,CO2,CO,CH4)
    # print('VMR_model',VMR_model)
    VMR_model_init = gen_vmr(NLAYER,10**H2O_init,10**CO2_init,10**CO_init,10**CH4_init)
    T_model = TP_Guillot(P_model,g,T_eq,kappa,gamma,f,T_int)
    # print('T_model',T_model)
    spec = FM.calc_disc_spectrum_uniform(
        3,P_model,T_model,VMR_model,
        solspec=wasp43_spec)
    line_spec.set_ydata(spec)
    line_tp.set_data(T_model,P_model)
    line_data.set_ydata(kevin_phase_by_wave[Nphase,:,0])
    # print(spec)
    fig.canvas.draw_idle()

skappa.on_changed(update)
sgamma.on_changed(update)
sf.on_changed(update)
sT_int.on_changed(update)
sP_top.on_changed(update)
sP_deep.on_changed(update)
sNlayer.on_changed(update)
sH2O.on_changed(update)
sCO2.on_changed(update)
sCO.on_changed(update)
sCH4.on_changed(update)
sN_phase.on_changed(update)

if __name__ == "__main__":
    plt.show()