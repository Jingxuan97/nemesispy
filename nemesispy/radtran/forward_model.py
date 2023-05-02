#!/usr/local/bin/python3
# -*- coding: utf-8 -*-
"""
Interface class for running forward models.
"""
import numpy as np
from nemesispy.radtran.calc_mmw import calc_mmw
from nemesispy.radtran.read import read_kls
from nemesispy.radtran.calc_radiance \
    import calc_radiance,calc_weighting,calc_contribution
from nemesispy.radtran.read import read_cia
from nemesispy.radtran.calc_layer import calc_layer
from nemesispy.common.calc_trig import gauss_lobatto_weights
from nemesispy.common.calc_trig_fast import disc_weights_2tp
from nemesispy.common.calc_trig_fast import disc_weights_3tp
from nemesispy.common.interpolate_gcm import interp_gcm
from nemesispy.common.calc_hydrostat import calc_hydrostat
from nemesispy.common.get_gas_info import get_gas_id, get_gas_name
from nemesispy.common.calc_trig_new import disc_weights_new


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
        self.gas_name_list = None
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
        assert self.M_plt > 0
        assert self.R_plt > 0
        assert self.NLAYER > 0
        assert self.is_planet_model_set
        assert self.is_opacity_data_set

    def set_planet_model(self, M_plt, R_plt, gas_id_list, iso_id_list, NLAYER,
        semi_major_axis=None):
        """
        Store the planetary system parameters
        """
        self.M_plt = M_plt
        self.R_plt = R_plt
        gas_name_list = []
        for index, id in enumerate(gas_id_list):
            gas_name_list.append(get_gas_name(id))
        self.gas_name_list = np.array(gas_name_list)
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

    def read_input_dict(self,input_dict):
        NLAYER = input_dict['NLAYER']
        # P_range = input_dict['P_range']
        # T_star = input_dict['T_star']
        # R_star = input_dict['R_star']
        # SMA = input_dict['SMA']
        M_plt = input_dict['M_plt']
        R_plt = input_dict['R_plt']
        # T_irr = input_dict['T_irr']
        # T_eq = input_dict['T_eq']
        gas_id_list = input_dict['gas_id']
        iso_id_list = input_dict['iso_id']
        # wave_grid = input_dict['wave_grid']
        # phase_grid = input_dict['phase_grid']
        # stellar_spec = input_dict['stellar_spec']
        # nwave = input_dict['nwave']
        # nphase = input_dict['nphase']
        kta_file_paths = input_dict['kta_file_paths']
        cia_file_path = input_dict['cia_file_path']
        self.set_planet_model(
            M_plt, R_plt, gas_id_list, iso_id_list, NLAYER
        )
        self.set_opacity_data(
            kta_file_paths, cia_file_path
        )

    def calc_weighting_function(self, P_model, T_model, VMR_model,
        path_angle=0, solspec=[]):

        NPRO = len(P_model)
        mmw = np.zeros(P_model.shape)
        for ipro in range(NPRO):
            mmw[ipro] = calc_mmw(self.gas_id_list,VMR_model[ipro,:])
        H_model = calc_hydrostat(P=P_model, T=T_model, mmw=mmw,
            M_plt=self.M_plt, R_plt=self.R_plt)
        H_layer,P_layer,T_layer,VMR_layer,U_layer,dH,scale \
            = calc_layer(
            self.R_plt, H_model, P_model, T_model, VMR_model,
            self.gas_id_list, self.NLAYER, path_angle, layer_type=1,
            H_0=0.0
            )

        if len(solspec)==0:
            solspec = np.ones(len(self.wave_grid))

        weighting_function = calc_weighting(self.wave_grid, U_layer, P_layer, T_layer,
            VMR_layer, self.k_gas_w_g_p_t, self.k_table_P_grid,
            self.k_table_T_grid, self.del_g, ScalingFactor=scale,
            k_cia=self.k_cia_pair_t_w,
            ID=self.gas_id_list,cia_nu_grid=self.cia_nu_grid,
            cia_T_grid=self.cia_T_grid, dH=dH)

        return weighting_function

    def calc_contribution_function(self, P_model, T_model, VMR_model,
        path_angle=0, solspec=[]):

        NPRO = len(P_model)
        mmw = np.zeros(P_model.shape)
        for ipro in range(NPRO):
            mmw[ipro] = calc_mmw(self.gas_id_list,VMR_model[ipro,:])
        H_model = calc_hydrostat(P=P_model, T=T_model, mmw=mmw,
            M_plt=self.M_plt, R_plt=self.R_plt)
        H_layer,P_layer,T_layer,VMR_layer,U_layer,dH,scale \
            = calc_layer(
            self.R_plt, H_model, P_model, T_model, VMR_model,
            self.gas_id_list, self.NLAYER, path_angle, layer_type=1,
            H_0=0.0
            )

        if len(solspec)==0:
            solspec = np.ones(len(self.wave_grid))

        contribution_function = calc_contribution(self.wave_grid, U_layer, P_layer, T_layer,
            VMR_layer, self.k_gas_w_g_p_t, self.k_table_P_grid,
            self.k_table_T_grid, self.del_g, ScalingFactor=scale,
            k_cia=self.k_cia_pair_t_w,
            ID=self.gas_id_list,cia_nu_grid=self.cia_nu_grid,
            cia_T_grid=self.cia_T_grid, dH=dH)

        return contribution_function

    def calc_point_spectrum(self, H_model, P_model, T_model, VMR_model,
        path_angle, solspec=[]):
        """
        Calculate average layer properties from model inputs,
        then compute the spectrum of a plane parallel atmosphere.
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
        nmu : int
            Number of zenith angle quadratures.
        """
        # initialise output array
        disc_spectrum = np.zeros(len(self.wave_grid))

        # get locations and angles for disc averaging
        nav, wav = gauss_lobatto_weights(phase, nmu)
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
        # nav, wav = gauss_lobatto_weights(0, nmu)
        # fov_emission_angles = wav[3,:]
        # fov_weights = wav[5,:]

        if nmu == 2:
            mu = [0.447213595499958,1.000000]                   # cos zenith angle
            wtmu = [0.8333333333333333,0.166666666666666666]    # corresponding weights
        if nmu == 3:
            mu = [0.28523151648064509,0.7650553239294646,1.0000]
            wtmu = [0.5548583770354863,0.3784749562978469,0.06666666666666666]
        if nmu == 4:
            mu = [0.2092992179024788,0.5917001814331423,0.8717401485096066,
                1.00000]
            wtmu = [0.4124587946587038,0.3411226924835043,0.2107042271435060,
                    0.035714285714285]
        if nmu == 5:
            mu = [0.165278957666387,0.477924949810444,0.738773865105505,
                0.919533908166459,1.00000000000000]
            wtmu = [0.327539761183898,0.292042683679684,0.224889342063117,
                    0.133305990851069,2.222222222222220E-002]

        # Hydrostatic case
        if len(H_model) == 0:
            for iav,cos in enumerate(mu):
                path_angle = np.arccos(cos)
                weight = wtmu[iav]
                point_spectrum = self.calc_point_spectrum_hydro(
                    P_model, T_model, VMR_model, path_angle,
                    solspec=solspec)
                disc_spectrum += point_spectrum * weight
        else:
            for iav,cos in enumerate(mu):
                path_angle = np.arccos(cos)
                weight = wtmu[iav]
                point_spectrum = self.calc_point_spectrum(
                    H_model, P_model, T_model, VMR_model, path_angle,
                    solspec=solspec)
                disc_spectrum += point_spectrum * weight
        return disc_spectrum

    def calc_disc_spectrum_2tp(self, phase, nmu, daymin, daymax,
        P_model, T_day, T_night, VMR_model, solspec=[]):
        """
        Fast calculation of disc averaged spectrum of an atmosphere made
        up of 2 TP profiles (i.e. a dayside and a nightside).

        Parameters
        ----------
        phase : real
            Orbital phase, increase from 0 at primary transit to 180 and secondary
            eclipse.
        daymin : real
            Longitude of the west boundary of dayside. Assume [-180,180] grid.
        daymax : real
            Longitude of the east boundary of dayside. Assume [-180,180] grid.

        Returns
        -------
        disc_spectrum : ndarray
            Disc averaged spectrum.
        """
        # initialise output array
        disc_spectrum = np.zeros(len(self.wave_grid))

        # get locations and angles for disc averaging
        day_zen, day_wt,\
        night_zen, night_wt, \
        day_lat, day_lon, \
        night_lat, night_lon, \
            = disc_weights_2tp(phase, nmu, daymin, daymax)

        # # print('day_lat',day_lat)
        # print('day_zen',day_zen)
        # # print('night_lat',night_lat)
        # print('night_zen',night_zen)
        for iav in range(len(day_zen)):
            T_model = T_day
            path_angle = day_zen[iav]
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

        for iav in range(len(night_zen)):
            T_model = T_night
            path_angle = night_zen[iav]
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

    def calc_disc_spectrum_3tp(self, phase, nmu, hotmin, hotmax,
        daymin, daymax, P_model, T_hot, T_day, T_night,
        VMR_model, solspec=[]):
        """
        Calculation of disc averaged spectrum of a atmosphere made up of 3
        TP profiles (hot region, dayside, nightside)
        """
        # initialise output array
        disc_spectrum = np.zeros(len(self.wave_grid))

        # get locations and angles for disc averaging
        hot_zen, hot_wt, day_zen, day_wt, night_zen, night_wt,\
        hot_lat, hot_lon, day_lat, day_lon, night_lat, night_lon \
            =  disc_weights_3tp(phase, nmu, hotmin, hotmax, daymin, daymax)

        for iav in range(len(hot_zen)):
            T_model = T_hot
            path_angle = hot_zen[iav]
            weight = hot_wt[iav]
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

        for iav in range(len(day_zen)):
            T_model = T_day
            path_angle = day_zen[iav]
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

        for iav in range(len(night_zen)):
            T_model = T_night
            path_angle = night_zen[iav]
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

    def calc_disc_spectrum_new(self,phase,nmu,P_model,
        global_model_P_grid,global_T_model,global_VMR_model,
        mod_lon,mod_lat,solspec):
        """
        Parameters
        ----------
        phase : real
            Orbital phase, increase from 0 at primary transit to 180 and secondary
            eclipse.
        nmu : int
            Number of zenith angle quadratures.
        """
        # initialise output array
        disc_spectrum = np.zeros(len(self.wave_grid))

        # get locations and angles for disc averaging
        nav, wav = disc_weights_new(phase, nmu)
        wav = np.around(wav,decimals=8)
        fov_latitudes = wav[0,:]
        fov_longitudes = wav[1,:]
        fov_emission_angles = wav[2,:]
        fov_weights = wav[3,:]

        for iav in range(nav):
            xlon = fov_longitudes[iav]
            xlat = fov_latitudes[iav]
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