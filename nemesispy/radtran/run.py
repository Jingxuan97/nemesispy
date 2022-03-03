import sys
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

# from nemesispy.radtran.runner import interpolate_to_lat_lon
lowres_files = ['/Users/jingxuanyang/Desktop/Workspace/nemesispy2022/nemesispy/data/ktables/h2o',
         '/Users/jingxuanyang/Desktop/Workspace/nemesispy2022/nemesispy/data/ktables/co2',
         '/Users/jingxuanyang/Desktop/Workspace/nemesispy2022/nemesispy/data/ktables/co',
         '/Users/jingxuanyang/Desktop/Workspace/nemesispy2022/nemesispy/data/ktables/ch4']
kta_file_paths = lowres_files

lowres_files = ['/Users/jingxuanyang/ktables/h2owasp43.kta',
'/Users/jingxuanyang/ktables/cowasp43.kta',
'/Users/jingxuanyang/ktables/co2wasp43.kta',
'/Users/jingxuanyang/ktables/ch4wasp43.kta']

cia_file_path='/Users/jingxuanyang/Desktop/Workspace/nemesispy2022/nemesispy/data/cia/exocia_hitran12_200-3800K.tab'
# Gas identifiers.
ID = np.array([1,2,5,6,40,39])
ISO = np.array([0,0,0,0,0,0])
NMODEL = 20
NVMR = len(ID)

# Volume Mixing Ratio
# VMR_model[i,j] is the Volume Mixing Ratio of jth gas at ith layer.
H2ratio = 1
VMR_H2O = np.ones(NMODEL)*1e-4
VMR_CO2 = np.ones(NMODEL)*1e-4 *0
VMR_CO = np.ones(NMODEL)*1e-4 *0
VMR_CH4 = np.ones(NMODEL)*1e-4 *0
VMR_He = (np.ones(NMODEL)-VMR_H2O-VMR_CO2-VMR_CO-VMR_CH4)*(1-H2ratio)
VMR_H2 = (np.ones(NMODEL)-VMR_H2O-VMR_CO2-VMR_CO-VMR_CH4)*H2ratio
VMR_model = np.zeros((NMODEL,NVMR))
VMR_model[:,0] = VMR_H2O
VMR_model[:,1] = VMR_CO2
VMR_model[:,2] = VMR_CO
VMR_model[:,3] = VMR_CH4
VMR_model[:,4] = VMR_He
VMR_model[:,5] = VMR_H2
mmw = calc_mmw(ID,VMR_model[0,:])


###############################################################################
### MODEL INPUT
# Planet/star parameters
T_star = 4520
M_plt = 3.8951064000000004e+27 # kg
semi_major_axis = 0.015*AU
R_star = 463892759.99999994 #km
R_plt = 74065.70 * 1e3 #km
# Atmosphere layout
NMODEL = 20
NLAYER = 20
P_range = np.logspace(np.log10(20/1.01325),np.log10(1e-3/1.01325),NMODEL)*1e5
P_range = np.logspace(np.log10(20),np.log10(1e-3),NMODEL)*1e5
# Atmospheric model params
kappa = 1e-3
gamma1 = 1e-1
gamma2 = 1e-1
alpha = 0.5
T_irr =  1500
atm = Model2(T_star, R_star, M_plt, R_plt, semi_major_axis, P_range, mmw,
                      kappa, gamma1, gamma2, alpha, T_irr)
###############################################################################
H_model = atm.height()
P_model = atm.pressure()
T_model = atm.temperature()
# Model output: H_model, P_model, T_model, VMR_model,
###############################################################################

# Get raw k table infos from files
gas_id_list, iso_id_list, wave_grid, g_ord, del_g, P_grid, T_grid,\
        k_gas_w_g_p_t = read_kls(kta_file_paths)
CIA_NU_GRID,CIA_TEMPS,K_CIA = read_cia(cia_file_path)
wave_grid = np.array([1.1425, 1.1775, 1.2125, 1.2475, 1.2825, 1.3175, 1.3525,
1.3875, 1.4225, 1.4575, 1.4925, 1.5275, 1.5625, 1.5975, 1.6325, 3.6, 4.5])

# Get raw stellar spectrum
StarSpectrum = np.ones(len(wave_grid)) # *4*(R_star)**2*np.pi # NWAVE

# DO Gauss Labatto quadrature averaging
# angles = np.array([80.4866,61.4500,42.3729,23.1420,0.00000])
H_layer,P_layer,T_layer,VMR_layer,U_layer,Gas_layer,scale,del_S\
    = calc_layer(R_plt, H_model, P_model, T_model, VMR_model, ID, NLAYER,
    path_angle=0, layer_type=1, H_0=0.0, NSIMPS=101)
print('scale',scale)

"""
before jit
1000 nemesis ver:381.06474781036377
1000 chimera ver:341.1965470314026

after jit

1. jit just for interp
1000 nemesis ver : f_combined is fucking things up
1000 chimera ver : 32.83659029006958

2. jit for interp and radiance


FULLY JIT
runtime 1000 = 0.004322124004364014 per
runtime 10000 = 0.002207125186920166
"""

SPECOUT = calc_radiance(wave_grid, U_layer, P_layer, T_layer, VMR_layer, k_gas_w_g_p_t,
        P_grid, T_grid, del_g, ScalingFactor=scale,
        RADIUS=R_plt, solspec=StarSpectrum,
        k_cia=K_CIA,ID=ID,cia_nu_grid=CIA_NU_GRID,cia_T_grid=CIA_TEMPS, DEL_S=del_S)

import time
run_number = 1
start = time.time()
for i in range(run_number):
# Radiative Transfer
    SPECOUT = calc_radiance(wave_grid, U_layer, P_layer, T_layer, VMR_layer, k_gas_w_g_p_t,
            P_grid, T_grid, del_g, ScalingFactor=scale,
            RADIUS=R_plt, solspec=StarSpectrum,
            k_cia=K_CIA,ID=ID,cia_nu_grid=CIA_NU_GRID,cia_T_grid=CIA_TEMPS, DEL_S=del_S)

end = time.time()
print('run time = ', (end-start)/run_number)

# 1e-4 h2o, pure H2, fixed ground radiation at 45 zenith angle
fortran_model = [2.0095757e+22, 1.9855963e+22, 2.4384866e+22, 3.0096336e+22, 3.1821019e+22,
 2.6848518e+22, 1.1246998e+22, 7.8110916e+21, 7.0568513e+21, 7.7212110e+21,
 9.0706302e+21, 1.1298792e+22, 1.4093987e+22, 1.6893329e+22, 1.8325981e+22,
 4.0654737e+21, 2.0768640e+21]

# 1e-4 h2o, pure H2, fixed ground radiation at 0 zenith angle
fortran_model = [2.3386727e+22, 2.3223329e+22, 2.7884951e+22, 3.3525034e+22, 3.5133009e+22,
 3.0026636e+22, 1.3094568e+22, 9.2723774e+21, 8.3360171e+21, 9.0995923e+21,
 1.0637932e+22, 1.3140588e+22, 1.6189699e+22, 1.9094999e+22, 2.0491682e+22,
 4.3127540e+21, 2.1802355e+21]

diff = (SPECOUT-fortran_model)
plt.scatter(wave_grid, diff,label='diff',marker='x',color='r',s=20)
diff = (SPECOUT-fortran_model)/SPECOUT

# start plot
plt.title('debug')
# lt.plot(wave_grid,SPECOUT)

plt.scatter(wave_grid,SPECOUT,marker='o',color='b',linewidth=0.5,s=10, label='python')
plt.scatter(wave_grid,fortran_model,label='fortran',marker='x',color='k',s=20)

# BB = calc_planck(wave_grid,2285.991)*np.pi*4.*np.pi*(74065.70*1e5)**2
# plt.plot(wave_grid,BB,label='black body',marker='*')

plt.xlabel(r'wavelength($\mu$m)')
plt.ylabel(r'total radiance(W sr$^{-1}$ $\mu$m$^{-1})$')
plt.legend()
plt.tight_layout()
plt.plot()
plt.grid()
# plt.savefig('comparison.pdf',dpi=400)
plt.show()
# plt.close()

print('diff')
print(diff)
print(max(diff))
print('spec')
print(SPECOUT)

print("Class methods")
class ForwardModel():

    def __init__(self):
        """
        Attributes to store data that doesn't change during a retrieval
        """

        # planet and planetary system data
        self.M_plt = None
        self.R_plt = None
        self.M_star = None # currently not used
        self.R_star = None
        self.T_star = None
        self.semi_major_axis = None
        self.NLAYER = None
        self.is_planet_model_set = False

        # opacity data
        self.gas_id_list = None
        self.iso_id_list = None
        self.wave_grid = None
        self.g_ord = None
        self.del_g = None
        self.k_table_P_grid = None
        self.k_table_T_grid = None
        self.k_gas_w_g_p_t = None
        self.cia_nu_grid = None
        self.cia_T_grid = None
        self.k_cia_pair_t_w = None
        self.is_opacity_data_set = False


    def set_planet_model(self, M_plt, R_plt, R_star, T_star, semi_major_axis,
        gas_id_list, iso_id_list, NLAYER):
        """
        Set the basic system parameters for running the
        """
        self.M_plt = M_plt
        self.R_plt = R_plt
        self.R_star = R_star
        self.T_star = T_star
        self.semi_major_axis = semi_major_axis
        self.gas_id_list = gas_id_list
        self.iso_id_list = iso_id_list
        self.NLAYER = NLAYER

        self.is_planet_model_set = True

    def set_opacity_data(self, kta_file_paths, cia_file_path):
        """
        Read gas ktables and cia opacity files and store as class attributes.
        """
        gas_id_list, iso_id_list, wave_grid, g_ord, del_g, k_table_P_grid,\
            k_table_T_grid, k_gas_w_g_p_t = read_kls(kta_file_paths)
        """
        Some gases (e.g. H2 and He) have no k table data so gas id lists need
        to be passed somewhere else.
        """
        # self.gas_id_list = gas_id_list
        # self.iso_id_list = iso_id_list
        self.wave_grid = wave_grid
        self.g_ord = g_ord
        self.del_g = del_g
        self.k_table_P_grid = k_table_P_grid
        self.k_table_T_grid = k_table_T_grid
        self.k_gas_w_g_p_t = k_gas_w_g_p_t

        cia_nu_grid, cia_T_grid, k_cia_pair_t_w = read_cia(cia_file_path)
        self.cia_nu_grid = cia_nu_grid
        self.cia_T_grid = cia_T_grid
        self.k_cia_pair_t_w = k_cia_pair_t_w

        self.is_opacity_data_set = True

    def calc_point_spectrum(self, H_model, P_model, T_model, VMR_model, path_angle,
        solspec=None):
        """
        First get layer properties from model inputs
        Then calculate the spectrum at a single point on the disc.
        """

        H_layer,P_layer,T_layer,VMR_layer,U_layer, Gas_layer,scale,del_S\
            = calc_layer(self.R_plt, H_model, P_model, T_model, VMR_model,
            self.gas_id_list, self.NLAYER, path_angle, layer_type=1, H_0=0.0, NSIMPS=101)
        if solspec == None:
            solspec = np.ones(len(self.wave_grid))
        point_spectrum = calc_radiance(self.wave_grid, U_layer, P_layer, T_layer,
            VMR_layer, self.k_gas_w_g_p_t, self.k_table_P_grid,
            self.k_table_T_grid, self.del_g, ScalingFactor=scale,
            RADIUS=self.R_plt, solspec=solspec, k_cia=self.k_cia_pair_t_w,
            ID=self.gas_id_list,cia_nu_grid=self.cia_nu_grid,
            cia_T_grid=self.cia_T_grid, DEL_S=del_S)

        return point_spectrum

    def calc_disc_spectrum(self,phase,nmu,global_H_model,global_P_model,
        global_T_model,global_VMR_model,global_model_longitudes,
        global_model_lattitudes,solspec=None):

        nav, wav = gauss_lobatto_weights(phase, nmu)
        print('calc_disc_spectrum')
        print('nav, wav',nav,wav)
        """Want a monotonic array for interpolation"""
        fov_longitudes = wav[1,:]
        # convert to [-180,180]
        for index, ilon in enumerate (fov_longitudes):
            if ilon>180:
                fov_longitudes[index] = ilon - 360

        fov_lattitudes = wav[0,:]
        fov_emission_angles = wav[3,:]
        fov_weights = wav[-1,:]

        # n_fov_loc = len(fov_longitudes)
        fov_locations = np.zeros((nav,2))
        fov_locations[:,0] = fov_longitudes
        fov_locations[:,1] = fov_lattitudes

        fov_H_model = interpolate_to_lat_lon(fov_locations, global_H_model,
            global_model_longitudes, global_model_lattitudes)

        fov_P_model = interpolate_to_lat_lon(fov_locations, global_P_model,
            global_model_longitudes, global_model_lattitudes)

        fov_T_model = interpolate_to_lat_lon(fov_locations, global_T_model,
            global_model_longitudes, global_model_lattitudes)

        fov_VMR_model = interpolate_to_lat_lon(fov_locations, global_VMR_model,
            global_model_longitudes, global_model_lattitudes)

        disc_spectrum = np.zeros(len(self.wave_grid))

        for ilocation in range(nav):
            # print('ilocation',ilocation)
            H_model = fov_H_model[ilocation]
            # print('fov_H_model',fov_H_model)
            # print('H_model', H_model)
            P_model = fov_P_model[ilocation]
            # print('P_model',P_model)
            T_model = fov_T_model[ilocation]
            VMR_model = fov_VMR_model[ilocation]
            path_angle = fov_emission_angles[ilocation]
            weight = fov_weights[ilocation]
            point_spectrum = self.calc_point_spectrum(
                H_model, P_model, T_model, VMR_model, path_angle,
                solspec=solspec)
            disc_spectrum += point_spectrum * weight
        return disc_spectrum

    def run_point_spectrum(self, H_model, P_model, T_model,\
            VMR_model, path_angle, solspec=None):
        if self.is_planet_model_set == False:
            raise('Planet model has not been set yet')
        elif self.is_opacity_data_set == False:
            raise('Opacity data has not been set yet')
        point_spectrum = self.calc_point_spectrum(H_model, P_model, T_model,\
            VMR_model, path_angle, solspec=solspec)
        return point_spectrum

    """TBD"""
    def run_disc_spectrum(self,phase,nmu,global_H_model,global_P_model,
        global_T_model,global_VMR_model,global_model_longitudes,
        global_model_lattitudes,solspec=None):
        if self.is_planet_model_set == False:
            raise('Planet model has not been set yet')
        elif self.is_opacity_data_set == False:
            raise('Opacity data has not been set yet')
        disc_spectrum = self.calc_disc_spectrum(phase,nmu,global_H_model,
            global_P_model, global_T_model,global_VMR_model,global_model_longitudes,
            global_model_lattitudes,solspec=solspec)
        return disc_spectrum

"""# Basic test for the class
Mod = ForwardModel()
Mod.set_planet_model(M_plt,R_plt,R_star,T_star,semi_major_axis,ID,
    iso_id_list,NLAYER)
Mod.set_opacity_data(kta_file_paths,cia_file_path)
point_spectrum = Mod.run_point_spectrum(H_model, P_model, T_model, VMR_model,
    path_angle=0)
print('point_spectrum',point_spectrum)

VMR1 = np.array([[1.000e-04, 0.000e+00, 0.000e+00, 0.000e+00, 0.000e+00, 9.999e-01],
       [1.000e-04, 0.000e+00, 0.000e+00, 0.000e+00, 0.000e+00, 9.999e-01],
       [1.000e-04, 0.000e+00, 0.000e+00, 0.000e+00, 0.000e+00, 9.999e-01],
       [1.000e-04, 0.000e+00, 0.000e+00, 0.000e+00, 0.000e+00, 9.999e-01],
       [1.000e-04, 0.000e+00, 0.000e+00, 0.000e+00, 0.000e+00, 9.999e-01],
       [1.000e-04, 0.000e+00, 0.000e+00, 0.000e+00, 0.000e+00, 9.999e-01],
       [1.000e-04, 0.000e+00, 0.000e+00, 0.000e+00, 0.000e+00, 9.999e-01],
       [1.000e-04, 0.000e+00, 0.000e+00, 0.000e+00, 0.000e+00, 9.999e-01],
       [1.000e-04, 0.000e+00, 0.000e+00, 0.000e+00, 0.000e+00, 9.999e-01],
       [1.000e-04, 0.000e+00, 0.000e+00, 0.000e+00, 0.000e+00, 9.999e-01],
       [1.000e-04, 0.000e+00, 0.000e+00, 0.000e+00, 0.000e+00, 9.999e-01],
       [1.000e-04, 0.000e+00, 0.000e+00, 0.000e+00, 0.000e+00, 9.999e-01],
       [1.000e-04, 0.000e+00, 0.000e+00, 0.000e+00, 0.000e+00, 9.999e-01],
       [1.000e-04, 0.000e+00, 0.000e+00, 0.000e+00, 0.000e+00, 9.999e-01],
       [1.000e-04, 0.000e+00, 0.000e+00, 0.000e+00, 0.000e+00, 9.999e-01],
       [1.000e-04, 0.000e+00, 0.000e+00, 0.000e+00, 0.000e+00, 9.999e-01],
       [1.000e-04, 0.000e+00, 0.000e+00, 0.000e+00, 0.000e+00, 9.999e-01],
       [1.000e-04, 0.000e+00, 0.000e+00, 0.000e+00, 0.000e+00, 9.999e-01],
       [1.000e-04, 0.000e+00, 0.000e+00, 0.000e+00, 0.000e+00, 9.999e-01],
       [1.000e-04, 0.000e+00, 0.000e+00, 0.000e+00, 0.000e+00, 9.999e-01]])

lon_coord = np.linspace(0,360,num=5)
lat_coord = np.linspace(0,90,num=10)
nlon = len(lon_coord )
nlat = len(lat_coord)

global_VMR_model_shape = (nlon,nlat) + VMR_model.shape
global_VMR_model = np.ones(global_VMR_model_shape)*1e-4

global_H_model_shape = (nlon,nlat) + H_model.shape
global_H_model = np.ones(global_H_model_shape)

global_P_model_shape = (nlon,nlat) + P_model.shape
global_P_model = np.ones(global_P_model_shape)

global_T_model_shape = (nlon,nlat) + T_model.shape
global_T_model = np.ones(global_T_model_shape)

# for ilon in range(nlon):
#     if ilon<=1:
#         global_VMR_model[ilon,:] = VMR1
#     elif ilon == 4:
#         global_VMR_model[ilon,:] = VMR1
#     else:
#         global_VMR_model[ilon,:] = VMR2

for ilon in range(nlon):
    global_VMR_model[ilon,:] = VMR1
    global_H_model[ilon,:] = H_model
    global_P_model[ilon,:] = P_model
    global_T_model[ilon,:] = T_model

phase = 0
nmu = 5
disc_spec = Mod.run_disc_spectrum(phase,nmu,global_H_model,global_P_model,
    global_T_model,global_VMR_model,global_model_longitudes=lon_coord,
    global_model_lattitudes=lat_coord,solspec=None)

# nmu = 5,
# 1e-4 h2o, pure H2, fixed ground radiation at 0 zenith angle
fortran_disc = np.array([1.8998773e+22, 1.8772364e+22, 2.3128123e+22, 2.8684045e+22, 3.0408838e+22,
 2.5618934e+22, 1.0726985e+22, 7.4668730e+21, 6.7616848e+21, 7.3960444e+21,
 8.6816295e+21, 1.0804962e+22, 1.3479171e+22, 1.6188206e+22, 1.7597941e+22,
 3.9828552e+21, 2.0438334e+21])

disc_diff = (disc_spec-fortran_disc)
plt.scatter(wave_grid, disc_diff,label='disc_diff',marker='x',color='r',s=20)
disc_diff = (disc_spec-fortran_disc)/disc_spec

# start plot
plt.title('disc debug')
# lt.plot(wave_grid,SPECOUT)

plt.scatter(wave_grid,disc_spec,marker='o',color='b',linewidth=0.5,s=10, label='python disc')
plt.scatter(wave_grid,fortran_disc,label='fortran disc',marker='x',color='k',s=20)

# BB = calc_planck(wave_grid,2285.991)*np.pi*4.*np.pi*(74065.70*1e5)**2
# plt.plot(wave_grid,BB,label='black body',marker='*')

plt.xlabel(r'wavelength($\mu$m)')
plt.ylabel(r'total radiance(W sr$^{-1}$ $\mu$m$^{-1})$')
plt.legend()
plt.tight_layout()
plt.plot()
plt.grid()
plt.savefig('disc_comparison.pdf',dpi=400)
plt.show()

print('disc diff')
print(disc_diff)
print(max(disc_diff))
print('disc spec')
print(disc_diff)
"""

# Read GCM data
f = open('process_vivien.txt')
vivien_gcm = f.read()
f.close()
vivien_gcm = vivien_gcm.split()
vivien_gcm = [float(i) for i in vivien_gcm]

### Parse GCM data
iread = 0
nlon = int(vivien_gcm[iread])
iread += 1
nlat = int(vivien_gcm[iread])
iread += 1
xlon = np.zeros(nlon) # regular lon lat grid
xlat = np.zeros(nlat)

for i in range(nlon):
    xlon[i] = vivien_gcm[iread]
    iread+=1

for i in range(nlat):
    xlat[i] = vivien_gcm[iread]
    iread+=1

npv = int(vivien_gcm[iread])
iread += 1

pv = np.zeros(npv)
for i in range(npv):
    pv[i] = vivien_gcm[iread]
    iread+=1
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

pvmap = np.zeros((nlon,nlat,npv))
for ilon in range(nlon):
    for ilat in range(nlat):
        pvmap[ilon,ilat,:] = pv

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

phase = 0
nmu = 2
# global_H_model =
Mod = ForwardModel()
Mod.set_planet_model(M_plt,R_plt,R_star,T_star,semi_major_axis,ID,
    iso_id_list,NLAYER)
Mod.set_opacity_data(kta_file_paths,cia_file_path)
Mod.calc_disc_spectrum(phase,nmu, global_H_model, global_P_model=pvmap,
    global_T_model=tmap, global_VMR_model=vmrmap,
    global_model_longitudes=xlon,
    global_model_lattitudes=xlat,
    solspec=None)


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