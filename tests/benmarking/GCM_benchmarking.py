"""
Present results of end to end comparisons
"""
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
sys.path.append('/Users/jingxuanyang/Desktop/Workspace/nemesispy2022/')
from nemesispy.radtran.forward_model import ForwardModel
from nemesispy.radtran.benchmarking.GCM_benchmarking_fortran import Nemesis_api
from nemesispy.backup_functions.hydrostatic import adjust_hydrostatH
import time

### Reference Opacity Data
lowres_files = [
'/Users/jingxuanyang/Desktop/Workspace/nemesispy2022/nemesispy/data/ktables/h2o',
'/Users/jingxuanyang/Desktop/Workspace/nemesispy2022/nemesispy/data/ktables/co2',
'/Users/jingxuanyang/Desktop/Workspace/nemesispy2022/nemesispy/data/ktables/co',
'/Users/jingxuanyang/Desktop/Workspace/nemesispy2022/nemesispy/data/ktables/ch4']
cia_file_path='/Users/jingxuanyang/Desktop/Workspace/nemesispy2022/nemesispy/data/cia/exocia_hitran12_200-3800K.tab'
folder_name = 'testing'

### Reference Constants
pi = np.pi
const = {
    'R_SUN': 6.95700e8,      # m solar radius
    'R_JUP': 7.1492e7,       # m nominal equatorial Jupiter radius (1 bar pressure level)
    'AU': 1.49598e11,        # m astronomical unit
    'k_B': 1.38065e-23,      # J K-1 Boltzmann constant
    'R': 8.31446,            # J mol-1 K-1 universal gas constant
    'G': 6.67430e-11,        # m3 kg-1 s-2 universal gravitational constant
    'N_A': 6.02214e23,       # Avagadro's number
    'AMU': 1.66054e-27,      # kg atomic mass unit
    'ATM': 101325,           # Pa atmospheric pressure
}

### Reference Planet Input for WASP0-43b
M_plt = 3.8951064000000004e+27 # kg
R_plt = 74065.70 * 1e3 # m

### Reference Spectral Input
wave_grid = np.array([1.1425, 1.1775, 1.2125, 1.2475, 1.2825, 1.3175, 1.3525, 1.3875,
       1.4225, 1.4575, 1.4925, 1.5275, 1.5625, 1.5975, 1.6325, 3.6   ,
       4.5   ])

stellar_spec = np.array([3.341320e+25, 3.215455e+25, 3.101460e+25, 2.987110e+25,
       2.843440e+25, 2.738320e+25, 2.679875e+25, 2.598525e+25,
       2.505735e+25, 2.452230e+25, 2.391140e+25, 2.345905e+25,
       2.283720e+25, 2.203690e+25, 2.136015e+25, 1.234010e+24,
       4.422200e+23])


### Reference Atmospheric Model Input
# Gas Volume Mixing Ratio, constant with height
gas_id = np.array([  1, 2,  5,  6, 40, 39])
iso_id = np.array([0, 0, 0, 0, 0, 0])

# GCM specs
# Read GCM data
f = open('process_vivien.txt')
vivien_gcm = f.read()
f.close()
vivien_gcm = vivien_gcm.split()
vivien_gcm = [float(i) for i in vivien_gcm]
iread = 152
nlon = 64
nlat = 32
npv = 53
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
xlat = np.array([-87.188 , -81.562 , -75.938 , -70.312 , -64.688 , -59.062 ,
       -53.438 , -47.812 , -42.188 , -36.562 , -30.938 , -25.312 ,
       -19.688 , -14.062 ,  -8.4375,  -2.8125,   2.8125,   8.4375,
        14.062 ,  19.688 ,  25.312 ,  30.938 ,  36.562 ,  42.188 ,
        47.812 ,  53.438 ,  59.062 ,  64.688 ,  70.312 ,  75.938 ,
        81.562 ,  87.188 ])
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

"""loop"""
for ilon in [0,20,40,60]:
    for ilat in [0,10,20,30]:
        for iang in [0,20,40,60,90]:
            # Set up NEMESIS model input
            # ilon = 0
            # ilat = 0
            path_angle = iang
            NLAYER = 53
            NMODEL = len(pv)
            H = np.linspace(0,1404762,NMODEL)
            P = pv
            T = tmap[ilon,ilat,:]
            VMR = vmrmap[ilon,ilat,:,:]


            ### Benchmark Fortran forward model
            folder_name = 'gcm'
            if not os.path.isdir(folder_name):
                os.mkdir(folder_name)
            file_path = os.path.dirname(os.path.realpath(__file__))
            os.chdir(file_path+'/'+folder_name) # move to designated process folder

            API = Nemesis_api(name=folder_name, NLAYER=NLAYER, gas_id_list=gas_id,
                iso_id_list=iso_id, wave_grid=wave_grid)
            API.write_files(path_angle=path_angle, H_model=H, P_model=P, T_model=T,
                VMR_model=VMR)
            API.run_forward_model()
            wave, yerr, fotran_spec = API.read_output()


            ### Benchmark Python forward model
            FM = ForwardModel()
            FM.set_planet_model(M_plt=M_plt, R_plt=R_plt, gas_id_list=gas_id,
                iso_id_list=iso_id, NLAYER=NLAYER)
            FM.set_opacity_data(kta_file_paths=lowres_files, cia_file_path=cia_file_path)

            # Test hydrostatic routine
            new_H = adjust_hydrostatH(H=H,P=P,T=T,ID=gas_id,VMR=VMR,M_plt=M_plt,R_plt=R_plt)
            python_spec = FM.run_point_spectrum(H_model=new_H, P_model=P,
                T_model=T, VMR_model=VMR, path_angle=path_angle, solspec=stellar_spec)

            ### Compare output
            # Fortran model plot
            fig, axs = plt.subplots(nrows=2,ncols=1,sharex=True,dpi=800)

            axs[0].set_title('ilon={} ilat={} angle={}'.format(
                ilon,ilat,path_angle),fontsize=8)

            # plot spectrum from Fortran code
            axs[0].scatter(wave,fotran_spec,marker='x',color='k',linewidth=1,s=10,
                label='Fortran')
            axs[0].plot(wave,fotran_spec,color='k',linewidth=0.5)

            # plot spectrum from Python code
            axs[0].scatter(wave,python_spec,marker='.',color='b',linewidth=1,s=10,
            label='Python')
            axs[0].plot(wave,python_spec,color='b',linewidth=0.5,linestyle='--')

            axs[0].legend(loc='upper left')
            axs[0].grid()
            axs[0].set_ylabel('Flux ratio')

            # difference between fortran and python end to end
            diff = (fotran_spec-python_spec)/fotran_spec
            axs[1].scatter(wave_grid, diff, marker='.',color='r')
            axs[1].set_ylabel('Relative diff')

            # Plot config
            plt.xlabel(r'wavelength($\mu$m)')
            plt.ticklabel_format(axis="y", style="sci", scilimits=(0,0))
            plt.grid()
            plt.tight_layout()
            # plt.savefig('{:.3e}H2O_{:.3e}CO2_{:.3e}CO_{:.3e}CH4_{:.3e}He_{:.3e}H2.pdf'.format(
            #    VMR_H2O,VMR_CO2,VMR_CO, VMR_CH4, VMR_He[0], VMR_H2[0]),dpi=400)
            # plt.savefig('comparison.pdf',dpi=400)
            plt.show()

            print(max(abs(diff)))
