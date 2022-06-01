class ForwardModel():
    def calc_disc_spectrum(self,phase,nmu,global_H_model,global_P_model,
        global_T_model,global_VMR_model,global_model_longitudes,
        global_model_lattitudes,solspec=None):

        nav, wav = gauss_lobatto_weights(phase, nmu)
        # print('calc_disc_spectrum')
        # print('nav, wav',nav,wav)

        fov_lattitudes = wav[0,:]
        fov_longitudes = wav[1,:]
        fov_stellar_zen = wav[2,:]
        fov_emission_angles = wav[3,:]
        fov_stellar_azi = wav[4,:]
        fov_weights = wav[5,:]

        """Convert to Vivien's longitude scheme"""
        for index, ilon in enumerate (fov_longitudes):
            fov_longitudes[index] = np.mod((ilon - 180),360)
        """Want a monotonic array for interpolation"""
        # convert to [-180,180]
        for index, ilon in enumerate (fov_longitudes):
            if ilon>180:
                fov_longitudes[index] = ilon - 360

        # print(fov_longitudes)


        # n_fov_loc = len(fov_longitudes)
        fov_locations = np.zeros((nav,2))
        fov_locations[:,0] = fov_longitudes
        fov_locations[:,1] = fov_lattitudes

        self.fov_lattitudes = fov_lattitudes
        self.fov_longitudes = fov_longitudes
        self.fov_emission_angles = fov_emission_angles
        self.fov_weights = fov_weights

        # print('fov_lattitudes\n',fov_lattitudes)
        # print('fov_longitudes\n',fov_longitudes)
        # print('fov_emission_angles\n',fov_emission_angles)
        # print('fov_weights\n',fov_weights)

        NLONM,NLATM,NPM = global_H_model.shape

        fov_H_model = interpolate_to_lat_lon(fov_locations, global_H_model,
            global_model_longitudes, global_model_lattitudes)

        fov_P_model = interpolate_to_lat_lon(fov_locations, global_P_model,
            global_model_longitudes, global_model_lattitudes)

        fov_T_model = interpolate_to_lat_lon(fov_locations, global_T_model,
            global_model_longitudes, global_model_lattitudes)

        fov_VMR_model = interpolate_to_lat_lon(fov_locations, global_VMR_model,
            global_model_longitudes, global_model_lattitudes)

        # fake_fov_H_model = interpolate_to_lat_lon(fov_locations, global_H_model,
        #     global_model_longitudes, global_model_lattitudes)
        # self.fake_fov_H_model = fake_fov_H_model
        # """H cannot be interpolated as it does not vary linearly"""
        # # fov_H_model = np.zeros((nav,NPM))
        # for iav in range(nav):
        #     fov_H_model[iav,:] = adjust_hydrostatH(fake_fov_H_model[iav,:],
        #         fov_P_model[iav,:],fov_T_model[iav,:],self.gas_id_list,
        #         fov_VMR_model[iav,:,:],self.M_plt,self.R_plt)

        fake_fov_H_model = interpolate_to_lat_lon(fov_locations, global_H_model,
            global_model_longitudes, global_model_lattitudes)

        self.fov_H_model = fov_H_model
        self.fov_P_model = fov_P_model
        self.fov_T_model = fov_T_model
        self.fov_VMR_model = fov_VMR_model

        disc_spectrum = np.zeros(len(self.wave_grid))

        total_weight = 0
        # print('nav',nav)
        for iav in range(nav):
            # print('iav',iav)
            H_model = fov_H_model[iav,:]
            # print('fov_H_model',fov_H_model)
            # print('H_model', H_model)
            P_model = fov_P_model[iav,:]
            # print('P_model',P_model)
            T_model = fov_T_model[iav,:]
            VMR_model = fov_VMR_model[iav,:,:]
            path_angle = fov_emission_angles[iav]
            weight = fov_weights[iav]
            point_spectrum = self.calc_point_spectrum(
                H_model, P_model, T_model, VMR_model, path_angle,
                solspec=solspec)
            total_weight += weight
            # print('H_model',H_model)
            # print('P_model',P_model)
            print('T_model',T_model)
            # print('VMR_model',VMR_model)
            # print('path_angle',path_angle)
            # print('point_spectrum',point_spectrum)
            # print('weight',weight)
            # print('total_weight',total_weight)
            disc_spectrum += point_spectrum * weight
            # print('disc_spectrum',disc_spectrum)
        # print('total_weight',total_weight)
        return disc_spectrum

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