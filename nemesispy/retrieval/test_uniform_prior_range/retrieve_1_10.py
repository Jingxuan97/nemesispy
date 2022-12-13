import numpy as np
import os
import pymultinest
from nemesispy.radtran.forward_model import ForwardModel
from nemesispy.models.TP_profiles import TP_Guillot
from nemesispy.common.constants import G
from nemesispy.data.helper import lowres_file_paths, cia_file_path

runname = 'retrieve_1_10'
upper = -1
lower = -10

### Wavelengths grid and orbital phase grid
Disc = np.array([0.005567804440763946, 0.005647632993896657, 0.0057161568693303196,
    0.005787405965537214, 0.005912133133091917, 0.005936504622456347,
    0.005828391791365244, 0.005804286017157853, 0.005805763268629676,
    0.005722935861283406, 0.005658389310799759, 0.005559032856244599,
    0.005498844502880942, 0.005484607994019324, 0.005442398861867252,
    0.011783632224625347, 0.0145235471350626])

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

# P_model = np.geomspace(20e5,100,NLAYER)
P_model = np.array([2.00000000e+06, 1.18757213e+06, 7.05163778e+05, 4.18716424e+05,
       2.48627977e+05, 1.47631828e+05, 8.76617219e+04, 5.20523088e+04,
       3.09079355e+04, 1.83527014e+04, 1.08975783e+04, 6.47083012e+03,
       3.84228874e+03, 2.28149751e+03, 1.35472142e+03, 8.04414701e+02,
       4.77650239e+02, 2.83622055e+02, 1.68410824e+02, 1.00000000e+02])

# TP_Guillot(P_model,g_plt=25,T_eq=1400,k_IR=1,gamma=0.1,f=1,T_int=100)
T_model = np.array([2992.76753424, 2969.77241316, 2955.86138448, 2947.50736545,
       2942.51305711, 2939.53543107, 2937.76306763, 2936.70914315,
       2936.08280017, 2935.71069668, 2935.48967984, 2935.3584195 ,
       2935.2804706 , 2935.23409378, 2935.1520517 , 2932.71351647,
       2911.08706571, 2839.30212502, 2707.17861832, 2534.66751905])


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
    cube[4] = lower + (upper - (lower)) * cube[4] # log H2O
    cube[5] = lower + (upper - (lower)) * cube[5] # log CO2
    cube[6] = lower + (upper - (lower)) * cube[6] # log CO
    cube[7] = lower + (upper - (lower)) * cube[7] # log CH4

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
if not os.path.isdir(runname):
    os.mkdir(runname)
pymultinest.run(LogLikelihood,
                Prior,
                n_params,
                n_live_points=400,
                outputfiles_basename=runname+'/fix-'
                )