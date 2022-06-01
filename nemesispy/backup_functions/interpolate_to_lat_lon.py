
def interpolate_to_lat_lon(chosen_location, global_model,
    global_model_longitudes, global_model_lattitudes):
    """
    Given a global model of some physical quantity defined at a range of
    locations specified by their longitudes and lattitudes,
    interpolate the model to the desired chosen_locations using bilinear
    interpolation.

    The model at (global_model_longitudes[i],global_model_lattitudes[j])
    is global_model[i,j,:].

    Parameters
    ----------
    chosen_location(NLOCOUT,2) : ndarray
        A array of [lattitude, longitude] at which the global
        model is to be interpolated.
    global_model(NLONIN, NLATIN, NPRESSIN) : ndarray
        Model defined at the global_model_locations.
        NlONIN x NLATIN x NPRESSIN
        NPRESSIN might be a tuple if the model is a 2D array.
    global_model_longitudes(NLONIN) : ndarray
        Longitude grid specifying where the model is define on the planet.
    global_model_lattitudes(NLATIN) : ndarray
        Longitude grid specifying where the model is define on the planet.

    Returns
    -------
    interp_model(NLOCOUT,NPRESSIN) : ndarray
        Model interpolated to the desired locations.

    """


    NLONIN, NLATIN = global_model.shape[0], global_model.shape[1]
    NLOCOUT = chosen_location.shape[0] # number of locations in the output

    # 1D model
    if len(global_model.shape) == 3:
        # NPRESSIN is the number of points in the MODEL
        NPRESSIN = global_model.shape[2]
        # NMODELDIM is the dimension of the model
        NMODELDIM = 1
        interped_model =  np.zeros((NLOCOUT,NPRESSIN))


    if len(global_model.shape) == 4:
        # NPRESSIN is the number of points in the MODEL
        NPRESSIN = global_model.shape[2]
        # NMODELDIM is the dimension of the model
        NMODELDIM = global_model.shape[3]
        interped_model = np.zeros((NLOCOUT,NPRESSIN,NMODELDIM))

    # print('global_model',global_model)
    # print('NLONIN, NLATIN, NPRESSIN', NLONIN, NLATIN, NPRESSIN)
    # add an extra data point for the periodic longitude
    # global_model_location = np.append(global_model_location,)
    # make sure there is a point at lon = 0

    # print('NLOCOUT',NLOCOUT)

    # # Interp MODEL : NLOCOUT x
    # interp_model_shape = (NLOCOUT,) + NPRESSIN
    # # print('interp_model_shape',interp_model_shape)
    # interped_model =  np.zeros(interp_model_shape) # output model

    lon_grid = global_model_longitudes
    lat_grid = global_model_lattitudes
    # print('lon_grid',lon_grid)
    # print('lat_grid',lat_grid)
    for ilocout, location in enumerate(chosen_location):

        # print('chosen_location')
        # print(chosen_location)
        lon = location[0]
        lat = location[1]

        # print('lon,lat',lon,lat)
        # if lon > np.max(lon_grid):
        #     lon = np.max(lon_grid)
        # if lon <= np.min(lon_grid):
        #     lon = np.min(lon_grid) + 1e-10
        # if lat > np.max(lat_grid):
        #     lat = np.max(lat_grid)
        # if lat <= np.min(lat_grid):
        #     lat = np.min(lat_grid) + 1e-10

        if lon > lon_grid[-1]:
            lon = lon_grid[-1]
        if lon <= lon_grid[0]:
            lon = lon_grid[0] + 1e-10
        if lat > lat_grid[-1]:
            lat = lat_grid[-1]
        if lat <= lat_grid[0]:
            lat = lat_grid[0] + 1e-10

        lon_index_hi = np.where(lon_grid >= lon)[0][0]
        lon_index_low = np.where(lon_grid < lon)[0][-1]
        lat_index_hi = np.where(lat_grid >= lat)[0][0]
        lat_index_low = np.where(lat_grid < lat)[0][-1]

        lon_hi = lon_grid[lon_index_hi]
        lon_low = lon_grid[lon_index_low]
        lat_hi = lat_grid[lat_index_hi]
        lat_low = lat_grid[lat_index_low]

        # problem here
        if len(global_model.shape)==3:
            for ipress in range(NPRESSIN):
                arr = global_model[:,:,ipress]
                Q11 = arr[lon_index_low,lat_index_low]
                Q12 = arr[lon_index_hi,lat_index_low]
                Q22 = arr[lon_index_hi,lat_index_hi]
                Q21 = arr[lon_index_low,lat_index_hi]
                fxy1 = (lat_hi-lat)/(lat_hi-lat_low)*Q11 \
                    + (lat-lat_low)/(lat_hi-lat_low)*Q21
                fxy2 = (lat_hi-lat)/(lat_hi-lat_low)*Q12 \
                    + (lat-lat_low)/(lat_hi-lat_low)*Q22
                fxy = (lon_hi-lon)/(lon_hi-lon_low)*fxy1 \
                    + (lon-lon_low)/(lon_hi-lon_low)*fxy2
                interped_model[ilocout,ipress] = fxy

        if len(global_model.shape)==4:
            for ipress in range(NPRESSIN):
                for imodel in range(NMODELDIM):
                    arr = global_model[:,:,ipress,imodel]
                    Q11 = arr[lon_index_low,lat_index_low]
                    Q12 = arr[lon_index_hi,lat_index_low]
                    Q22 = arr[lon_index_hi,lat_index_hi]
                    Q21 = arr[lon_index_low,lat_index_hi]
                    fxy1 = (lat_hi-lat)/(lat_hi-lat_low)*Q11 \
                        + (lat-lat_low)/(lat_hi-lat_low)*Q21
                    fxy2 = (lat_hi-lat)/(lat_hi-lat_low)*Q12 \
                        + (lat-lat_low)/(lat_hi-lat_low)*Q22
                    fxy = (lon_hi-lon)/(lon_hi-lon_low)*fxy1 \
                        + (lon-lon_low)/(lon_hi-lon_low)*fxy2
                    interped_model[ilocout,ipress,imodel] = fxy

        # fxy1 = (lat_hi-lat)/(lat_hi-lat_low)*Q11 + (lat-lat_low)/(lat_hi-lat_low)*Q21
        # fxy2 = (lat_hi-lat)/(lat_hi-lat_low)*Q12 + (lat-lat_low)/(lat_hi-lat_low)*Q22
        # fxy = (lon_hi-lon)/(lon_hi-lon_low)*fxy1 + (lon-lon_low)/(lon_hi-lon_low)*fxy2
        # print('interped_model',interped_model)
    return interped_model
