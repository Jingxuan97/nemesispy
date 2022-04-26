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

class ForwardModel():

    def __init__(self):
        """
        Attributes to store data that doesn't change during a retrieval
        """
        # planet and planetary system data
        self.M_plt = None
        self.R_plt = None
        self.M_star = None # currently not used
        self.R_star = None # currently not used
        self.T_star = None # currently not used
        self.semi_major_axis = None # currently not used
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

        # debug data
        self.U_layer = None
        self.del_S = None

    def set_planet_model(self, M_plt, R_plt, gas_id_list, iso_id_list, NLAYER,
        R_star=None, T_star=None, semi_major_axis=None):
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

    def test_point_spectrum(self,U_layer,P_layer,T_layer,VMR_layer,del_S,
        scale,solspec,path_angle=None):
        """
        wrapper for calc_radiance
        """
        print('test scale',scale)
        point_spectrum = calc_radiance(self.wave_grid, U_layer, P_layer, T_layer,
            VMR_layer, self.k_gas_w_g_p_t, self.k_table_P_grid,
            self.k_table_T_grid, self.del_g, ScalingFactor=scale,
            RADIUS=self.R_plt, solspec=solspec, k_cia=self.k_cia_pair_t_w,
            ID=self.gas_id_list,cia_nu_grid=self.cia_nu_grid,
            cia_T_grid=self.cia_T_grid, DEL_S=del_S)
        # debug
        print('P_layer',P_layer)
        print('T_layer',T_layer)
        print('U_layer',U_layer)
        print('del_S',del_S)
        return point_spectrum

    def calc_point_spectrum(self, H_model, P_model, T_model, VMR_model, path_angle,
        solspec=None):
        """
        First get layer properties from model inputs
        Then calculate the spectrum at a single point on the disc.
        """

        H_layer,P_layer,T_layer,VMR_layer,U_layer,Gas_layer,scale,del_S\
            = calc_layer(self.R_plt, H_model, P_model, T_model, VMR_model,
            self.gas_id_list, self.NLAYER, path_angle, layer_type=1, H_0=0.0, NSIMPS=101)
        # debug
        # print('H_layer',H_layer)
        print('own layering')
        print('P_layer',P_layer)
        print('T_layer',T_layer)
        print('U_layer',U_layer)
        print('del_S',del_S)
        # scale = np.around(scale,4)
        # print('scale',scale)
        """
        scale = np.array([5.6395, 5.4208, 5.231 , 5.0667, 4.9247, 4.8022, 4.6967, 4.6045,
        4.5219, 4.4484, 4.3807, 4.3182, 4.26  , 4.2037, 4.151 , 4.1007,
        4.0525, 4.0058, 3.9608, 3.9176])
        U_layer = np.array([5.0862e+30, 3.0826e+30, 1.8692e+30, 1.1350e+30, 6.9017e+29,
        4.2012e+29, 2.5590e+29, 1.5589e+29, 9.4976e+28, 5.7864e+28,
        3.5260e+28, 2.1497e+28, 1.3114e+28, 8.0071e+27, 4.8924e+27,
        2.9919e+27, 1.8308e+27, 1.1212e+27, 6.8713e+26, 4.2136e+26])
        P_layer = np.array([1.64055307e+06, 9.92539170e+05, 6.01171357e+05, 3.64597747e+05,
        2.21395125e+05, 1.34559600e+05, 8.18412158e+04, 4.98022507e+04,
        3.03215063e+04, 1.84715475e+04, 1.12602473e+04, 6.86892308e+03,
        4.19313248e+03, 2.56149600e+03, 1.56577522e+03, 9.57683370e+02,
        5.86023270e+02, 3.58761427e+02, 2.19733395e+02, 1.34630527e+02])
        T_layer = np.array([2286.031, 2253.709, 2185.85 , 2082.549, 1956.123, 1822.772,
        1696.444, 1586.561, 1497.773, 1430.593, 1382.584, 1349.852,
        1328.321, 1314.522, 1305.835, 1300.424, 1297.081, 1295.024,
        1293.761, 1292.988])
        """
        # layer normal
        # del_S = np.array([99680., 98320., 95370., 90900., 85470., 79770., 74380., 69680.,
        # 65880., 62990., 60910., 59480., 58540., 57930., 57540., 57300.,
        # 57150., 57050., 56980., 56930.])
        """
        [562129.70510089,533022.94775755,498862.65361552,460547.31305169,
        420916.88414793,383089.03993097,349322.31574516,320825.40682556,
        297920.3264484,280196.42525044,266834.25748818,256864.84808146,
        249351.05352987,243547.0369775,238868.80239676,234978.22598176,
        231569.71453952,228517.2699914,225708.64932763,223048.12139662]
        """

        if solspec.any() == None:
            solspec = np.ones(len(self.wave_grid))
        point_spectrum = calc_radiance(self.wave_grid, U_layer, P_layer, T_layer,
            VMR_layer, self.k_gas_w_g_p_t, self.k_table_P_grid,
            self.k_table_T_grid, self.del_g, ScalingFactor=scale,
            RADIUS=self.R_plt, solspec=solspec, k_cia=self.k_cia_pair_t_w,
            ID=self.gas_id_list,cia_nu_grid=self.cia_nu_grid,
            cia_T_grid=self.cia_T_grid, DEL_S=del_S)
        self.U_layer = U_layer
        self.del_S = del_S
        return point_spectrum

    def calc_disc_spectrum(self,phase,nmu,global_H_model,global_P_model,
        global_T_model,global_VMR_model,global_model_longitudes,
        global_model_lattitudes,solspec=None):

        nav, wav = gauss_lobatto_weights(phase, nmu)
        # print('calc_disc_spectrum')
        # print('nav, wav',nav,wav)
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