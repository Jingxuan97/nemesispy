import numpy as np
import os
import pymultinest
from nemesispy.radtran.forward_model import ForwardModel
from nemesispy.models.TP_profiles import TP_Guillot
from nemesispy.common.constants import G
from nemesispy.data.helper import lowres_file_paths, cia_file_path

"""
h2o = 1e-4
co2 = 1e-8
co = 1e-8
ch4 = 1e-8
h2 = (1-h2o-co2-co-ch4)*0.85
he = (1-h2o-co2-co-ch4)*0.15
T_model = np.array([2495.9, 2260. , 2098.5, 2158.4, 2350. , 2468.6, 2486.2, 2429.8,
       2325.2, 2192.7, 2068.7, 1967.3, 1888.5, 1827.3, 1772.7, 1734.9,
       1725. , 1739.8, 1761.1, 1769.4])

Disc = np.array([0.0017210951274934459, 0.0018492499499404029, 0.0019384668555974917,
    0.001998634281408214, 0.00207270366174539, 0.002125586423814779, 0.0020402553756137466,
    0.0020265981584232026, 0.0019649135191983024, 0.0019994768796606274, 0.0020870490593229403,
    0.002216670807213318, 0.0023592438780058074, 0.0024992427225295267, 0.0025568885719237993,
    0.007021424564799133, 0.009822875362071945]
"""

Disc = np.array([0.0017210951274934459, 0.0018492499499404029, 0.0019384668555974917,
    0.001998634281408214, 0.00207270366174539, 0.002125586423814779, 0.0020402553756137466,
    0.0020265981584232026, 0.0019649135191983024, 0.0019994768796606274, 0.0020870490593229403,
    0.002216670807213318, 0.0023592438780058074, 0.0024992427225295267, 0.0025568885719237993,
    0.007021424564799133, 0.009822875362071945]
)

err = np.array([6.60e-05, 6.10e-05, 5.80e-05, 5.60e-05, 5.70e-05, 5.30e-05,
       5.50e-05, 5.10e-05, 5.60e-05, 5.50e-05, 5.50e-05, 5.50e-05,
       5.80e-05, 5.70e-05, 6.30e-05, 7.90e-05, 1.03e-04])

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

### Reference Planet Input: WASP 43b
T_star = 4520 # star temperature in K
R_star = 463892759.99999994 # m, 0.6668 * R_SUN
SMA = 2243970000.0 # m, 0.015*AU
M_plt = 3.8951064000000004e+27 # kg
R_plt = 74065.70 * 1e3 # m
T_irr = T_star * (R_star/SMA)**0.5
T_eq = T_irr/2**0.5
g = G*M_plt/R_plt**2
gas_id = np.array([  1, 2,  5,  6, 40, 39])
iso_id = np.array([0, 0, 0, 0, 0, 0])
nmu = 5
NLAYER = 20
P_model = np.geomspace(20e5,100,NLAYER)

### Set up forward model
FM = ForwardModel()
FM.set_planet_model(M_plt=M_plt,R_plt=R_plt,gas_id_list=gas_id,iso_id_list=iso_id,
    NLAYER=NLAYER)
FM.set_opacity_data(kta_file_paths=lowres_file_paths, cia_file_path=cia_file_path)

### MultiNest functions
def Prior(cube,ndim,nparams):
    # log_kappa
    cube[0] = -4 + (2 - (-4)) * cube[0]
    # log_gamma
    cube[1] = -4 + (1 - (-4)) * cube[1]
    # log_f
    cube[2] = -3 + (1 - (-3)) * cube[2]
    # T_int
    cube[3] = 100 + (1000 - (100)) * cube[3]
    # log VMRs
    cube[4] = -8 + (-2 - (-8)) * cube[4] # log H2O
    cube[5] = -8 + (-2 - (-8)) * cube[5] # log CO2
    cube[6] = -8 + (-2 - (-8)) * cube[6] # log CO
    cube[7] = -8 + (-2 - (-8)) * cube[7] # log CH4

def LogLikelihood(cube,ndim,nparams):


    k_IR = 10**cube[0]
    gamma = 10**cube[1]
    f = 10**cube[2]
    T_int = cube[3]
    T_model = TP_Guillot(P_model,g,T_eq,k_IR,gamma,f,T_int)

    h2o = 10**cube[4]
    co2 = 10**cube[5]
    co = 10**cube[6]
    ch4 = 10**cube[7]
    h2 = (1-h2o-co2-co-ch4)*0.85
    he = (1-h2o-co2-co-ch4)*0.15
    VMR_model = np.zeros((NLAYER,6))
    VMR_model[:,0] = h2o
    VMR_model[:,1] = co2
    VMR_model[:,2] = co
    VMR_model[:,3] = ch4
    VMR_model[:,4] = h2
    VMR_model[:,5] = he

    disc_spectrum = FM.calc_disc_spectrum_uniform(
        nmu,
        P_model,
        T_model,
        VMR_model,
        solspec=wasp43_spec
    )

    chi = np.sum( (disc_spectrum-Disc)**2/err**2)

    like = -0.5*chi
    print('likelihood',like)
    return like

n_params = 8
print('start')
if not os.path.isdir('uniform'):
    os.mkdir('uniform')
pymultinest.run(LogLikelihood,
                Prior,
                n_params,
                n_live_points=400,
                outputfiles_basename='uniform/fix-'
                )