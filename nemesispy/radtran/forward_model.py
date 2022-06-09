#!/usr/local/bin/python3
# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import numpy as np
from nemesispy.common.constants import R_SUN, R_JUP_E, AMU, AU, M_JUP, R_JUP
from nemesispy.radtran.calc_mmw import calc_mmw
from nemesispy.radtran.calc_layer import calc_layer
from nemesispy.radtran.read import read_kls
from nemesispy.radtran.calc_radiance import calc_radiance, calc_planck
from nemesispy.radtran.read import read_cia
from nemesispy.common.calc_trig import gauss_lobatto_weights, interpvivien_point
from nemesispy.radtran.calc_hydrostat import calc_hydrostat

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

        ### debug data
        # # layering debug data
        # self.U_layer = None
        # self.del_S = None
        # self.del_H = None
        # self.P_layer = None
        # self.T_layer = None
        # self.scale = None

        # phase curve debug data
        self.fov_H_model = None
        self.fake_fov_H_model = None
        fov_lattitudes = None
        fov_longitudes = None
        fov_emission_angles = None
        fov_weights = None
        self.total_weight = None

    def set_planet_model(self, M_plt, R_plt, gas_id_list, iso_id_list, NLAYER,
        gas_name_list=None, solspec=None, R_star=None, T_star=None,
        semi_major_axis=None):
        """
        Store the planetary system parameters
        """
        self.M_plt = M_plt
        self.R_plt = R_plt
        self.R_star = R_star
        self.T_star = T_star
        self.semi_major_axis = semi_major_axis
        self.gas_name_list = gas_name_list
        self.gas_id_list = gas_id_list
        self.iso_id_list = iso_id_list
        self.NLAYER = NLAYER

        self.is_planet_model_set = True

    def set_opacity_data(self, kta_file_paths, cia_file_path):
        """
        Read gas ktables and cia opacity files and store as class attributes.
        """
        k_gas_id_list, k_iso_id_list, wave_grid, g_ord, del_g, k_table_P_grid,\
            k_table_T_grid, k_gas_w_g_p_t = read_kls(kta_file_paths)
        """
        Some gases (e.g. H2 and He) have no k table data so gas id lists need
        to be passed somewhere else.
        """
        self.k_gas_id_list = k_gas_id_list
        self.k_iso_id_list = k_iso_id_list
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

    def calc_point_spectrum(self, H_model, P_model, T_model, VMR_model,
        path_angle, solspec=np.array([0])):
        """
        First get layer properties from model inputs
        Then calculate the spectrum at a single point on the disc.
        """

        H_layer,P_layer,T_layer,VMR_layer,U_layer,MMW_layer,Gas_layer,scale,del_S,del_H\
            = calc_layer(self.R_plt, H_model, P_model, T_model, VMR_model,
            self.gas_id_list, self.NLAYER, path_angle, layer_type=1, H_0=0.0,
            NSIMPS=101)

        if not solspec.all():
            solspec = np.ones(len(self.wave_grid))
        point_spectrum = calc_radiance(self.wave_grid, U_layer, P_layer, T_layer,
            VMR_layer, self.k_gas_w_g_p_t, self.k_table_P_grid,
            self.k_table_T_grid, self.del_g, ScalingFactor=scale,
            RADIUS=self.R_plt, solspec=solspec, k_cia=self.k_cia_pair_t_w,
            ID=self.gas_id_list,cia_nu_grid=self.cia_nu_grid,
            cia_T_grid=self.cia_T_grid, DEL_S=del_H)

        return point_spectrum

    def calc_disc_spectrum(self,phase,nmu,P_model,
        global_model_P_grid,global_T_model,global_VMR_model,
        mod_lon,mod_lat,solspec):
        """
        Parameters
        ----------
        phase : real
            Orbital phase, increase from 0 at primary transit to 180 and secondary
            eclipse.

        """
        # initialise output array
        disc_spectrum = np.zeros(len(self.wave_grid))

        # get locations and angles for disc averaging
        nav, wav = gauss_lobatto_weights(phase, nmu)
        wav = np.around(wav,decimals=8)
        fov_lattitudes = wav[0,:]
        fov_longitudes = wav[1,:]
        fov_stellar_zen = wav[2,:]
        fov_emission_angles = wav[3,:]
        fov_stellar_azi = wav[4,:]
        fov_weights = wav[5,:]

        for iav in range(nav):
            xlon = fov_longitudes[iav]
            xlat = fov_lattitudes[iav]
            T_model, VMR_model = interpvivien_point(
                xlon=xlon,xlat=xlat,xp=P_model,
                vp=global_model_P_grid,
                vt=global_T_model,vvmr=global_VMR_model,
                mod_lon=mod_lon,
                mod_lat=mod_lat)
            path_angle = fov_emission_angles[iav]
            weight = fov_weights[iav]
            NPRO = len(P_model)
            mmw = np.zeros(NPRO)
            for ipro in range(NPRO):
                mmw[ipro] = calc_mmw(self.gas_id_list,VMR_model[ipro,:])
            H_model = calc_hydrostat(P=P_model, T=T_model, mmw=mmw,
                M_plt=self.M_plt, R_plt=self.R_plt)
            point_spectrum = self.calc_point_spectrum(
                H_model, P_model, T_model, VMR_model, path_angle,
                solspec=solspec)
            disc_spectrum += point_spectrum * weight
        # print('number of point spectrum = ',nav)
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


    # def test_point_spectrum(self,U_layer,P_layer,T_layer,VMR_layer,del_S,
    #     scale,solspec,path_angle=None):
    #     """
    #     wrapper for calc_radiance
    #     """
    #     # print('test scale',scale)
    #     point_spectrum = calc_radiance(self.wave_grid, U_layer, P_layer, T_layer,
    #         VMR_layer, self.k_gas_w_g_p_t, self.k_table_P_grid,
    #         self.k_table_T_grid, self.del_g, ScalingFactor=scale,
    #         RADIUS=self.R_plt, solspec=solspec, k_cia=self.k_cia_pair_t_w,
    #         ID=self.gas_id_list,cia_nu_grid=self.cia_nu_grid,
    #         cia_T_grid=self.cia_T_grid, DEL_S=del_S)
    #     # debug
    #     # print('P_layer',P_layer)
    #     # print('T_layer',T_layer)
    #     # print('U_layer',U_layer)
    #     # print('del_S',del_S)
    #     return point_spectrum

    # def test_disc_spectrum(self,phase,nmu,P_model,
    #     global_model_P_grid,global_T_model,
    #     global_VMR_model,mod_lon,mod_lat,
    #     solspec=None):

    #     # get locations and angles for disc averaging
    #     nav, wav = gauss_lobatto_weights(phase, nmu)
    #     fov_lattitudes = wav[0,:]
    #     fov_longitudes = wav[1,:]
    #     fov_stellar_zen = wav[2,:]
    #     fov_emission_angles = wav[3,:]
    #     fov_stellar_azi = wav[4,:]
    #     fov_weights = wav[5,:]

    #     """Convert to Vivien's longitude scheme"""
    #     """CONVERSION NOW DONE IN THE TRIG ROUTINE"""
    #     # for index, ilon in enumerate (fov_longitudes):
    #     #     fov_longitudes[index] = np.mod((ilon - 180),360)
    #     """Want a monotonic array for interpolation"""
    #     # convert to [-180,180]
    #     # for index, ilon in enumerate (fov_longitudes):
    #     #     if ilon>180:
    #     #         fov_longitudes[index] = ilon - 360

    #     # fov_locations = np.zeros((nav,2))
    #     # fov_locations[:,0] = fov_longitudes
    #     # fov_locations[:,1] = fov_lattitudes

    #     self.fov_lattitudes = fov_lattitudes
    #     self.fov_longitudes = fov_longitudes
    #     self.fov_emission_angles = fov_emission_angles
    #     self.fov_weights = fov_weights

    #     disc_spectrum = np.zeros(len(self.wave_grid))

    #     total_weight = 0
    #     for iav in range(nav):
    #         xlon = fov_longitudes[iav]
    #         xlat = fov_lattitudes[iav]
    #         """now the interpol"""
    #         T_model, VMR_model = interpvivien_point(
    #             xlon=xlon,xlat=xlat,xp=P_model,
    #             vp=global_model_P_grid,
    #             vt=global_T_model,vvmr=global_VMR_model,
    #             mod_lon=mod_lon,
    #             mod_lat=mod_lat)
    #         path_angle = fov_emission_angles[iav]
    #         weight = fov_weights[iav]
    #         NPRO = len(P_model)

    #         mmw = np.zeros(NPRO)
    #         for ipro in range(NPRO):
    #             mmw[ipro] = calc_mmw(self.gas_id_list,VMR_model[ipro,:])
    #         fake_H_model = np.linspace(0,1e5,NPRO)
    #         H_model = calc_hydrostat(H=fake_H_model, P=P_model, T=T_model,
    #             ID=self.gas_id_list, VMR=VMR_model, M_plt=self.M_plt,
    #             R_plt=self.R_plt, xmolwt=mmw)

    #         # H_model = np.linspace(0,1e5,NPRO)

    #         # print('H_model',H_model)
    #         point_spectrum = self.calc_point_spectrum(
    #             H_model, P_model, T_model, VMR_model, path_angle,
    #             solspec=solspec)
    #         disc_spectrum += point_spectrum * weight
    #         total_weight += weight
    #         # print('total_weight',total_weight)
    #         # print('disc_spectrum',disc_spectrum)
    #         # print('H_model',H_model)
    #         # print('P_model',P_model)
    #         # print('T_model',T_model)
    #         # print('VMR_model',VMR_model[0,:])
    #         # print('point_spectrum',point_spectrum)
    #         # print('xlon',xlon)
    #         # print('xlat',xlat)
    #     self.total_weight = total_weight
    #     return disc_spectrum
