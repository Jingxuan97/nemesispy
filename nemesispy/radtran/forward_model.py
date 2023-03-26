#!/usr/local/bin/python3
# -*- coding: utf-8 -*-
"""
Interface class for running forward models.
"""
import numpy as np
from nemesispy.radtran.calc_mmw import calc_mmw
from nemesispy.radtran.read import read_kls
from nemesispy.radtran.calc_radiance import calc_radiance
from nemesispy.radtran.read import read_cia
from nemesispy.radtran.calc_layer import calc_layer
from nemesispy.common.calc_trig import gauss_lobatto_weights
from nemesispy.common.calc_trig_fast import disc_weights_2tp
from nemesispy.common.interpolate_gcm import interp_gcm
from nemesispy.common.calc_hydrostat import calc_hydrostat

class ForwardModel():

    def __init__(self):
        """
        Attributes to store planet and opacity data.
        These attributes shouldn't change during a retrieval.
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

    def sanity_check(self):
        pass

    def set_planet_model(self, M_plt, R_plt, gas_id_list, iso_id_list, NLAYER,
        gas_name_list=None, solspec=None, R_star=None, T_star=None,
        semi_major_axis=None):
        """
        Store the planetary system parameters
        """
        self.M_plt = M_plt
        self.R_plt = R_plt
        # self.R_star = R_star
        # self.T_star = T_star
        # self.semi_major_axis = semi_major_axis
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
        path_angle, solspec=[]):
        """
        Calculate average layer properties from model inputs,
        then compute the spectrum at a single point on the disc.
        """
        H_layer,P_layer,T_layer,VMR_layer,U_layer,dH,scale \
            = calc_layer(
            self.R_plt, H_model, P_model, T_model, VMR_model,
            self.gas_id_list, self.NLAYER, path_angle, layer_type=1,
            H_0=0.0
            )

        if len(solspec)==0:
            solspec = np.ones(len(self.wave_grid))

        try:
            point_spectrum = calc_radiance(self.wave_grid, U_layer, P_layer, T_layer,
                VMR_layer, self.k_gas_w_g_p_t, self.k_table_P_grid,
                self.k_table_T_grid, self.del_g, ScalingFactor=scale,
                R_plt=self.R_plt, solspec=solspec, k_cia=self.k_cia_pair_t_w,
                ID=self.gas_id_list,cia_nu_grid=self.cia_nu_grid,
                cia_T_grid=self.cia_T_grid, dH=dH)
        except ZeroDivisionError:
            print('P_model',P_model)
            print('H_model',H_model)
            print('T_model',T_model)
            print('VMR_model',VMR_model)
            print('path_angle',path_angle)
            print('H_layer',H_layer)
            print('U_layer',U_layer)
            print('P_layer',P_layer)
            print('T_layer',T_layer)
            print('VMR_layer',VMR_layer)
            print('scale',scale)
            raise(Exception('Division by zero'))
        return point_spectrum

    def calc_point_spectrum_hydro(self, P_model, T_model, VMR_model,
        path_angle, solspec=[]):
        """
        Use the hydrodynamic equation to calculate layer height
        First get layer properties from model inputs
        Then calculate the spectrum at a single point on the disc.
        """
        NPRO = len(P_model)
        mmw = np.zeros(P_model.shape)

        for ipro in range(NPRO):
            mmw[ipro] = calc_mmw(self.gas_id_list,VMR_model[ipro,:])

        H_model = calc_hydrostat(P=P_model, T=T_model, mmw=mmw,
            M_plt=self.M_plt, R_plt=self.R_plt)

        point_spectrum = self.calc_point_spectrum(H_model, P_model,
            T_model, VMR_model, path_angle, solspec)

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
        # s1 = time.time()
        nav, wav = gauss_lobatto_weights(phase, nmu)
        # s2 = time.time()
        # print('gauss_lobatto_weights',s2-s1)
        wav = np.around(wav,decimals=8)
        fov_latitudes = wav[0,:]
        fov_longitudes = wav[1,:]
        fov_stellar_zen = wav[2,:]
        fov_emission_angles = wav[3,:]
        fov_stellar_azi = wav[4,:]
        fov_weights = wav[5,:]

        for iav in range(nav):
            xlon = fov_longitudes[iav]
            xlat = fov_latitudes[iav]
            # print(xlon,xlat)
            T_model, VMR_model = interp_gcm(
                lon=xlon,lat=xlat, p=P_model,
                gcm_lon=mod_lon, gcm_lat=mod_lat,
                gcm_p=global_model_P_grid,
                gcm_t=global_T_model, gcm_vmr=global_VMR_model,
                substellar_point_longitude_shift=180)

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
        return disc_spectrum

    def calc_disc_spectrum_uniform(self, nmu, P_model, T_model, VMR_model,
        H_model=[],solspec=[]):
        """Caculate the disc integrated spectrum of a homogeneous atmosphere
        """
        # initialise output array
        disc_spectrum = np.zeros(len(self.wave_grid))
        nav, wav = gauss_lobatto_weights(0, nmu)
        fov_emission_angles = wav[3,:]
        fov_weights = wav[5,:]

        # Hydrostatic case
        if len(H_model) == 0:
            for iav in range(nav):
                path_angle = fov_emission_angles[iav]
                weight = fov_weights[iav]
                point_spectrum = self.calc_point_spectrum_hydro(
                    P_model, T_model, VMR_model, path_angle,
                    solspec=solspec)
                disc_spectrum += point_spectrum * weight
        else:
            for iav in range(nav):
                path_angle = fov_emission_angles[iav]
                weight = fov_weights[iav]
                point_spectrum = self.calc_point_spectrum(
                    H_model, P_model, T_model, VMR_model, path_angle,
                    solspec=solspec)
                disc_spectrum += point_spectrum * weight
        return disc_spectrum

    def calc_disc_spectrum_2tp(self,phase,nmu,daybound1,daybound2,
        P_model,global_model_P_grid,global_T_model,global_VMR_model,
        mod_lon,mod_lat,solspec):
        """
        Parameters
        ----------
        phase : real
            Orbital phase, increase from 0 at primary transit to 180 and secondary
            eclipse.
        daybound1 : real
            Longitude of the west boundary of dayside. Assume [-180,180] grid.
        daybound2 : real
            Longitude of the east boundary of dayside. Assume [-180,180] grid.
        """
        # initialise output array
        disc_spectrum = np.zeros(len(self.wave_grid))

        # get locations and angles for disc averaging
        day_lat, day_lon, day_zen, day_wt,\
            night_lat, night_lon, night_zen, night_wt \
            = disc_weights_2tp(phase, nmu, daybound1, daybound2)

        for iav in range(len(day_lat)):
            xlon = day_lon[iav]
            xlat = day_lat[iav]
            # print(xlon,xlat)
            T_model, VMR_model = interp_gcm(
                lon=xlon,lat=xlat, p=P_model,
                gcm_lon=mod_lon, gcm_lat=mod_lat,
                gcm_p=global_model_P_grid,
                gcm_t=global_T_model, gcm_vmr=global_VMR_model,
                substellar_point_longitude_shift=180)
            path_angle = day_zen,[iav]
            weight = day_wt[iav]
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

        for iav in range(len(night_lat)):
            xlon = night_lon[iav]
            xlat = night_lat[iav]
            # print(xlon,xlat)
            T_model, VMR_model = interp_gcm(
                lon=xlon,lat=xlat, p=P_model,
                gcm_lon=mod_lon, gcm_lat=mod_lat,
                gcm_p=global_model_P_grid,
                gcm_t=global_T_model, gcm_vmr=global_VMR_model,
                substellar_point_longitude_shift=180)
            path_angle = night_zen,[iav]
            weight = night_wt[iav]
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

        return disc_spectrum
